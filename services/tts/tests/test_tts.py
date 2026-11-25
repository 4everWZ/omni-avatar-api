"""Unit tests for the standalone voice-cloning TTS worker service."""

import asyncio
import base64
import io
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
import wave

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import numpy as np

from services.tts import __main__ as tts_main
from services.tts import model as tts_model
from services.tts.app import create_app
from services.tts.backends import (
    GenieTTSBackend,
    Qwen3TTSBackend,
    TTSBackend,
    VoiceHandle,
    create_backend,
    decode_wav_to_float32,
    encode_wav_pcm16,
    pcm_float_to_int16_le,
)
from services.tts.cache import VoicePromptCache


def make_pcm_wav(
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    width: int = 2,
    frames: int = 1200,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(sample_rate)
        if width == 1:
            unit = bytes([140]) * channels
        elif width == 2:
            unit = b"\x20\x01" * channels
        elif width == 3:
            unit = b"\x00\x20\x01" * channels
        else:
            unit = b"\x00\x00\x20\x01" * channels
        writer.writeframes(unit * frames)
    return buf.getvalue()


class StubTTSBackend(TTSBackend):
    name = "stub-qwen3"
    sample_rate_hz = 24000
    requires_reference_audio = True

    def __init__(self):
        self.loaded = False
        self.warmed = False
        self.closed = False
        self.prepare_calls: list[tuple[str, str, str]] = []
        self.synthesize_calls: list[tuple[str, str, str]] = []
        self.stream_calls: list[tuple[str, str, str, int]] = []
        self.prepare_gate: asyncio.Event | None = None

    async def load(self) -> None:
        self.loaded = True

    async def warmup(self) -> None:
        self.warmed = True

    async def close(self) -> None:
        self.closed = True

    async def prepare_voice(
        self,
        voice_key: str,
        ref_wav_bytes: bytes,
        ref_text: str = "",
        language: str = "Auto",
    ) -> VoiceHandle:
        if self.prepare_gate is not None:
            await self.prepare_gate.wait()
        self.prepare_calls.append((voice_key, ref_text, language))
        return VoiceHandle(
            voice_key=voice_key,
            ref_text=ref_text,
            language=language,
            prompt_data={"key": voice_key, "bytes": len(ref_wav_bytes)},
            x_vector_only=not bool(ref_text.strip()),
        )

    async def synthesize(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
    ) -> tuple[bytes, int]:
        self.synthesize_calls.append((text, voice.voice_key, language))
        pcm = pcm_float_to_int16_le(np.linspace(-0.25, 0.25, 480, dtype=np.float32))
        return encode_wav_pcm16(pcm, sample_rate=self.sample_rate_hz), self.sample_rate_hz

    async def synthesize_stream(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
        chunk_size: int = 4,
    ):
        self.stream_calls.append((text, voice.voice_key, language, chunk_size))
        for val in (0.1, -0.1, 0.2):
            yield pcm_float_to_int16_le(np.full(240, val, dtype=np.float32))


class ModelPinningTests(unittest.TestCase):
    def test_location_defaults_and_checkpoint_validation(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            model_id, revision, path = tts_model.location()
        self.assertEqual(model_id, "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
        self.assertEqual(revision, "5d83992436eae1d760afd27aff78a71d676296fc")
        self.assertEqual(path.as_posix(), "/models/5d83992436eae1d760afd27aff78a71d676296fc")

        for bad in ("", "abc", "5D83992436eae1d760afd27aff78a71d676296fc", "z" * 40):
            with mock.patch.dict(os.environ, {"TTS_CHECKPOINT": bad}, clear=True):
                with self.assertRaises(ValueError):
                    tts_model.location()

    def test_validate_fetch_and_require_ready_checks_bundled_speech_tokenizer(self):
        sha = "5d83992436eae1d760afd27aff78a71d676296fc"
        model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        with tempfile.TemporaryDirectory() as tmpdir:
            snap = os.path.join(tmpdir, sha)

            def fake_snapshot_download(*, repo_id, revision, local_dir):
                self.assertEqual(repo_id, model_id)
                self.assertEqual(revision, sha)
                for rel in tts_model.REQUIRED_SNAPSHOT_FILES:
                    target = os.path.join(local_dir, rel)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with open(target, "wb") as fh:
                        fh.write(b"{}")

            fake_hf = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
            with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hf}):
                tts_model.fetch(model_id, sha, snap)

            ready_marker = os.path.join(snap, ".snapshot-ready")
            self.assertTrue(os.path.isfile(ready_marker))
            with open(ready_marker, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), f"{model_id}@{sha}\n")

            tts_model.require_ready(model_id, sha, snap, backend="qwen3")

            # Missing bundled speech_tokenizer file must fail validation.
            os.remove(os.path.join(snap, "speech_tokenizer", "model.safetensors"))
            with self.assertRaises(FileNotFoundError) as ctx:
                tts_model.require_ready(model_id, sha, snap, backend="qwen3")
            self.assertIn("speech_tokenizer/model.safetensors", str(ctx.exception))

            # Genie backend does not require Qwen3 snapshot files.
            tts_model.require_ready(model_id, sha, snap, backend="genie")

            # Mismatched .snapshot-ready fails require_ready.
            with open(ready_marker, "w", encoding="utf-8") as fh:
                fh.write("other/model@" + sha + "\n")
            with self.assertRaises(RuntimeError):
                tts_model.require_ready(model_id, sha, snap, backend="qwen3")


class BackendAndAudioHelperTests(unittest.IsolatedAsyncioTestCase):
    def test_pcm_float_and_wav_roundtrip(self):
        raw = pcm_float_to_int16_le([-2.0, -0.5, 0.0, 0.5, 2.0])
        self.assertEqual(len(raw), 10)
        wav = encode_wav_pcm16(raw, sample_rate=24000)
        samples, sr = decode_wav_to_float32(wav)
        self.assertEqual(sr, 24000)
        self.assertEqual(len(samples), 5)
        self.assertLess(samples[0], -0.99)
        self.assertGreater(samples[-1], 0.99)

        # Stereo 24-bit WAV should downmix cleanly to mono float32.
        stereo24 = make_pcm_wav(sample_rate=16000, channels=2, width=3, frames=200)
        mono, mono_sr = decode_wav_to_float32(stereo24)
        self.assertEqual(mono_sr, 16000)
        self.assertEqual(len(mono), 200)

        with self.assertRaises(ValueError):
            decode_wav_to_float32(b"not-a-wav")

    def test_create_backend_factory_and_genie_missing_dependency(self):
        qwen = create_backend("qwen3", model_path="/models/abc")
        self.assertIsInstance(qwen, Qwen3TTSBackend)
        genie = create_backend("genie")
        self.assertIsInstance(genie, GenieTTSBackend)
        with self.assertRaises(ValueError):
            create_backend("unknown")

    async def test_genie_backend_missing_package_raises_clear_runtime_error(self):
        backend = GenieTTSBackend(model_dir="/tmp/genie-char")
        with mock.patch.dict(sys.modules, {"genie_tts": None}):
            with self.assertRaises(RuntimeError) as ctx:
                await backend.load()
        self.assertIn("genie-tts", str(ctx.exception))

    async def test_genie_backend_honors_env_and_interface(self):
        calls = []

        class FakeGenieModule:
            @staticmethod
            def load_character(name, model_dir, language):
                calls.append(("load", name, model_dir, language, os.environ.get("GENIE_DATA_DIR")))

            @staticmethod
            def set_reference_audio(name, audio_path, audio_text, language):
                calls.append(("ref", name, os.path.isfile(audio_path), audio_text, language))

            @staticmethod
            def tts(name, text, save_path):
                calls.append(("tts", name, text, save_path))
                with open(save_path, "wb") as fh:
                    fh.write(make_pcm_wav(sample_rate=32000, channels=1, width=2, frames=320))

            @staticmethod
            def unload_character(name):
                calls.append(("unload", name))

            @staticmethod
            def stop():
                calls.append(("stop",))

        env = {
            "GENIE_DATA_DIR": "/opt/genie-data",
            "GENIE_MODEL_DIR": "/opt/genie-model",
            "GENIE_CHARACTER_NAME": "mika",
            "GENIE_LANGUAGE": "Japanese",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.dict(sys.modules, {"genie_tts": FakeGenieModule}):
                backend = create_backend("genie")
                await backend.load()
                await backend.warmup()
                handle = await backend.prepare_voice("v1", make_pcm_wav(), ref_text="こんにちは")
                wav_out, sr_out = await backend.synthesize("テスト", handle)
                chunks = [c async for c in backend.synthesize_stream("テスト", handle, chunk_size=2)]
                await backend.close()

        self.assertEqual(sr_out, 24000)
        with wave.open(io.BytesIO(wav_out), "rb") as reader:
            self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, 24000))
        self.assertGreater(len(b"".join(chunks)), 0)
        self.assertEqual(calls[0], ("load", "mika", "/opt/genie-model", "Japanese", "/opt/genie-data"))
        self.assertEqual(calls[1][0:2], ("ref", "mika"))
        self.assertTrue(calls[1][2])
        self.assertIn(("unload", "mika"), calls)
        self.assertIn(("stop",), calls)

    async def test_qwen3_backend_serializes_concurrent_cuda_calls(self):
        active_gpu_threads = 0
        max_concurrent_gpu = 0
        lock = threading.Lock()
        init_kwargs = {}

        class FakeInnerModel:
            def create_voice_clone_prompt(self, *, ref_audio, ref_text, x_vector_only_mode):
                return {"sr": ref_audio[1], "ref_text": ref_text, "xvec": x_vector_only_mode}

        class FakeFasterQwen3TTS:
            def __init__(self):
                self.model = FakeInnerModel()
                self.warmups = []

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                init_kwargs["path"] = path
                init_kwargs.update(kwargs)
                return cls()

            def warmup(self, prefill_len=100):
                self.warmups.append(prefill_len)

            def generate_voice_clone(self, *, text, language, voice_clone_prompt):
                return [np.zeros(240, dtype=np.float32) + 0.1], 24000

            def generate_voice_clone_streaming(self, *, text, language, voice_clone_prompt, chunk_size=4):
                nonlocal active_gpu_threads, max_concurrent_gpu
                with lock:
                    active_gpu_threads += 1
                    max_concurrent_gpu = max(max_concurrent_gpu, active_gpu_threads)
                try:
                    for _ in range(3):
                        time.sleep(0.01)
                        yield np.full(120, 0.2, dtype=np.float32), 24000, {"step": 1}
                finally:
                    with lock:
                        active_gpu_threads -= 1

        fake_torch = types.SimpleNamespace(
            bfloat16="bf16",
            cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
        )
        fake_faster = types.SimpleNamespace(FasterQwen3TTS=FakeFasterQwen3TTS)

        with mock.patch.dict(sys.modules, {"torch": fake_torch, "faster_qwen3_tts": fake_faster}):
            backend = Qwen3TTSBackend("/models/sha", device="cuda:1", max_seq_len=1024)
            await backend.load()
            await backend.warmup()
            handle = await backend.prepare_voice("k1", make_pcm_wav(), ref_text="", language="zh")
            self.assertTrue(handle.x_vector_only)
            self.assertEqual(handle.language, "Chinese")

            async def drain_stream():
                out = bytearray()
                async for chunk in backend.synthesize_stream("你好", handle, language="zh", chunk_size=4):
                    out.extend(chunk)
                return bytes(out)

            results = await asyncio.gather(drain_stream(), drain_stream())
            self.assertEqual(len(results), 2)
            self.assertTrue(all(len(r) == 3 * 120 * 2 for r in results))
            self.assertEqual(max_concurrent_gpu, 1)
            self.assertEqual(init_kwargs["attn_implementation"], "sdpa")
            self.assertEqual(init_kwargs["device"], "cuda:1")
            self.assertEqual(init_kwargs["max_seq_len"], 1024)

            # Mid-stream cancellation (barge-in) stops the thread and frees GPU lock.
            gen = backend.synthesize_stream("打断测试", handle, language="zh", chunk_size=4)
            first_chunk = await anext(gen)
            self.assertEqual(len(first_chunk), 120 * 2)
            await gen.aclose()
            self.assertEqual(active_gpu_threads, 0)
            wav_after, sr_after = await backend.synthesize("恢复测试", handle)
            self.assertEqual(sr_after, 24000)
            self.assertGreater(len(wav_after), 44)
            await backend.close()

    def test_cli_entrypoint_fetch_validate_serve(self):
        sha = "5d83992436eae1d760afd27aff78a71d676296fc"
        with tempfile.TemporaryDirectory() as tmpdir:
            env = {
                "TTS_MODEL_ROOT": tmpdir,
                "TTS_CHECKPOINT": sha,
                "TTS_BACKEND": "genie",
                "OMNI_TTS_INTERNAL_TOKEN": "cli-secret",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                tts_main.main(["validate"])
                with mock.patch("aiohttp.web.run_app") as run_app:
                    tts_main.main(["serve"])
                    run_app.assert_called_once()
                    _, kwargs = run_app.call_args
                    self.assertTrue(kwargs["handler_cancellation"])
            with self.assertRaises(SystemExit):
                tts_main.main(["unknown"])


class TtsHttpApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.asr_calls = []
        self.asr_mode = "ok"

        async def handle_transcribe(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer asr-secret")
            reader = await request.multipart()
            fields = {}
            while (part := await reader.next()) is not None:
                if part.name == "file":
                    fields["file"] = (part.filename, await part.read())
                else:
                    fields[part.name] = await part.text()
            self.asr_calls.append(fields)
            if self.asr_mode == "error":
                return web.json_response({"error": "boom"}, status=500)
            if self.asr_mode == "empty":
                return web.json_response({"text": ""})
            return web.json_response({"text": "这是参考音频。"})

        asr_app = web.Application()
        asr_app.router.add_post("/transcribe", handle_transcribe)
        self.asr_server = TestServer(asr_app)
        await self.asr_server.start_server()
        self.addAsyncCleanup(self.asr_server.close)
        self.asr_url = str(self.asr_server.make_url("")).rstrip("/")

        self.backend = StubTTSBackend()
        self.cache = VoicePromptCache(
            max_entries=2,
            asr_url=self.asr_url,
            asr_token="asr-secret",
            asr_timeout=5.0,
        )
        self.app = create_app(
            self.backend,
            self.cache,
            token="tts-secret",
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            checkpoint="5d83992436eae1d760afd27aff78a71d676296fc",
            model_root="/models",
        )
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.auth = {"Authorization": "Bearer tts-secret"}
        self.ref_wav = make_pcm_wav(sample_rate=24000, channels=1, width=2, frames=600)
        self.ref_b64 = base64.b64encode(self.ref_wav).decode("ascii")

    async def test_livez_and_health_unauthenticated(self):
        async with await self.client.get("/livez") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(await resp.json(), {"state": "running"})

        async with await self.client.get("/health") as resp:
            self.assertEqual(resp.status, 200)
            body = await resp.json()
            self.assertEqual(body["status"], "ready")
            self.assertEqual(body["backend"], "stub-qwen3")
            self.assertEqual(body["checkpoint"], "5d83992436eae1d760afd27aff78a71d676296fc")
            self.assertEqual(body["root"], "/models/5d83992436eae1d760afd27aff78a71d676296fc")
            self.assertEqual(body["sample_rate_hz"], 24000)
            self.assertEqual(body["voice_cache_size"], 0)
        self.assertTrue(self.backend.loaded)
        self.assertTrue(self.backend.warmed)

    async def test_auth_required_on_tts_endpoints(self):
        payload = {"text": "你好", "reference_audio_base64": self.ref_b64}
        for path in ("/v1/tts/synthesize", "/v1/tts/stream"):
            async with await self.client.post(path, json=payload) as resp:
                self.assertEqual(resp.status, 401)
            async with await self.client.post(
                path, headers={"Authorization": "Bearer wrong"}, json=payload
            ) as resp:
                self.assertEqual(resp.status, 401)

    async def test_synthesize_caches_asr_and_voice_prompt_across_requests(self):
        payload = {"text": "第一句话。", "reference_audio_base64": self.ref_b64}
        async with await self.client.post("/v1/tts/synthesize", headers=self.auth, json=payload) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.content_type, "audio/wav")
            self.assertEqual(resp.headers["Cache-Control"], "no-store")
            wav_body = await resp.read()
            first_key = resp.headers["X-TTS-Voice-Key"]
            first_ref_text = base64.b64decode(resp.headers["X-TTS-Ref-Text"]).decode("utf-8")
            self.assertEqual(first_ref_text, "这是参考音频。")

        with wave.open(io.BytesIO(wav_body), "rb") as reader:
            self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()), (1, 2, 24000))

        self.assertEqual(len(self.asr_calls), 1)
        self.assertEqual(self.asr_calls[0]["file"][0], "ref.wav")
        self.assertEqual(self.asr_calls[0]["response_format"], "json")
        self.assertEqual(len(self.backend.prepare_calls), 1)
        self.assertEqual(len(self.cache), 1)

        # Second request with the exact same reference audio hits the LRU cache (0 ASR calls).
        payload2 = {"text": "第二句话。", "reference_audio_base64": self.ref_b64}
        async with await self.client.post("/v1/tts/synthesize", headers=self.auth, json=payload2) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers["X-TTS-Voice-Key"], first_key)
            second_ref_text = base64.b64decode(resp.headers["X-TTS-Ref-Text"]).decode("utf-8")
            self.assertEqual(second_ref_text, "这是参考音频。")

        self.assertEqual(len(self.asr_calls), 1)
        self.assertEqual(len(self.backend.prepare_calls), 1)
        self.assertEqual(len(self.backend.synthesize_calls), 2)

    async def test_single_flight_deduplicates_parallel_cache_misses(self):
        self.backend.prepare_gate = asyncio.Event()
        payload = {"text": "并发测试", "reference_audio_base64": self.ref_b64}

        req1 = asyncio.create_task(
            self.client.post("/v1/tts/synthesize", headers=self.auth, json=payload)
        )
        req2 = asyncio.create_task(
            self.client.post("/v1/tts/synthesize", headers=self.auth, json=payload)
        )
        while len(self.asr_calls) < 1:
            await asyncio.sleep(0.005)
        self.backend.prepare_gate.set()
        r1, r2 = await asyncio.gather(req1, req2)
        try:
            self.assertEqual(r1.status, 200)
            self.assertEqual(r2.status, 200)
        finally:
            r1.release()
            r2.release()
        self.assertEqual(len(self.asr_calls), 1)
        self.assertEqual(len(self.backend.prepare_calls), 1)

    async def test_failed_or_empty_asr_falls_back_to_x_vector_only(self):
        for mode in ("error", "empty"):
            self.asr_mode = mode
            wav_bytes = make_pcm_wav(sample_rate=24000, frames=500 if mode == "error" else 520)
            b64 = base64.b64encode(wav_bytes).decode("ascii")
            async with await self.client.post(
                "/v1/tts/synthesize",
                headers=self.auth,
                json={"text": "Fallback test", "reference_audio_base64": b64},
            ) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(base64.b64decode(resp.headers["X-TTS-Ref-Text"]).decode("utf-8"), "")
            handle = await self.cache.get_or_prepare(self.backend, wav_bytes)
            self.assertTrue(handle.x_vector_only)

    async def test_stream_returns_raw_pcm16_chunks_and_headers(self):
        payload = {
            "text": "流式输出测试。",
            "reference_audio_base64": self.ref_b64,
            "ref_text": "显式文本",
            "language": "zh",
            "chunk_size": 6,
        }
        async with await self.client.post("/v1/tts/stream", headers=self.auth, json=payload) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers["Content-Type"], "application/octet-stream")
            self.assertEqual(resp.headers["X-TTS-Sample-Rate"], "24000")
            self.assertEqual(resp.headers["X-TTS-Channels"], "1")
            self.assertEqual(resp.headers["X-TTS-Sample-Width"], "2")
            self.assertEqual(
                base64.b64decode(resp.headers["X-TTS-Ref-Text"]).decode("utf-8"),
                "显式文本",
            )
            raw_pcm = await resp.read()
        self.assertEqual(len(raw_pcm), 3 * 240 * 2)
        # Explicit ref_text skips ASR altogether.
        self.assertEqual(len(self.asr_calls), 0)
        self.assertEqual(self.backend.stream_calls[-1][2:], ("zh", 6))

    async def test_lru_eviction_and_validation_errors(self):
        for frames in (100, 110, 120):
            b64 = base64.b64encode(make_pcm_wav(frames=frames)).decode("ascii")
            async with await self.client.post(
                "/v1/tts/synthesize",
                headers=self.auth,
                json={"text": "evict", "reference_audio_base64": b64, "ref_text": "hi"},
            ) as resp:
                self.assertEqual(resp.status, 200)
        self.assertEqual(len(self.cache), 2)

        # Missing text -> 400
        async with await self.client.post(
            "/v1/tts/synthesize",
            headers=self.auth,
            json={"text": "   ", "reference_audio_base64": self.ref_b64},
        ) as resp:
            self.assertEqual(resp.status, 400)

        # Missing reference audio when no default configured -> 400 missing_reference_audio
        async with await self.client.post(
            "/v1/tts/synthesize",
            headers=self.auth,
            json={"text": "hello"},
        ) as resp:
            self.assertEqual(resp.status, 400)
            err = await resp.json()
            self.assertEqual(err["error"]["code"], "missing_reference_audio")

        # Invalid base64 -> 400
        async with await self.client.post(
            "/v1/tts/synthesize",
            headers=self.auth,
            json={"text": "hello", "reference_audio_base64": "!!!not-base64!!!"},
        ) as resp:
            self.assertEqual(resp.status, 400)

        # Invalid chunk_size -> 400
        async with await self.client.post(
            "/v1/tts/stream",
            headers=self.auth,
            json={"text": "hello", "reference_audio_base64": self.ref_b64, "chunk_size": True},
        ) as resp:
            self.assertEqual(resp.status, 400)

    async def test_default_reference_audio_path_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            default_wav_path = os.path.join(tmpdir, "default.wav")
            with open(default_wav_path, "wb") as fh:
                fh.write(make_pcm_wav(frames=300))

            cache = VoicePromptCache(max_entries=4)
            app = create_app(
                self.backend,
                cache,
                token="tts-secret",
                default_ref_audio_path=default_wav_path,
                default_ref_text="默认音色文本",
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                async with await client.post(
                    "/v1/tts/stream",
                    headers=self.auth,
                    json={"text": "默认参考音频测试"},
                ) as resp:
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(
                        base64.b64decode(resp.headers["X-TTS-Ref-Text"]).decode("utf-8"),
                        "默认音色文本",
                    )
                    self.assertEqual(len(await resp.read()), 3 * 240 * 2)
            finally:
                await client.close()


if __name__ == "__main__":
    unittest.main()
