"""Unit and end-to-end async tests for CPU engineered full-duplex omni-engine."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from aiohttp import ClientSession, WSServerHandshakeError, web

from services.omni.app import create_app
from services.omni.chunker import SentenceChunker
from services.omni.duplex import DuplexConfig
from services.omni.model import (
    DEFAULT_VAD_CHECKPOINT,
    DEFAULT_VAD_MODEL_ID,
    fetch,
    require_ready,
    vad_location,
    validate_sha40,
)
from services.omni.vad import EnergyVad, SileroOnnxVad

VALID_SHA = "e71cae966052b992a7eca6b17738916ce0eca4ec"


def make_sine_pcm16(duration_ms: int, sample_rate: int = 16000, amplitude: int = 4000, freq: float = 440.0) -> bytes:
    total_samples = int(sample_rate * duration_ms / 1000)
    samples = [int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate)) for i in range(total_samples)]
    return struct.pack(f"<{total_samples}h", *samples)


def make_silence_pcm16(duration_ms: int, sample_rate: int = 16000) -> bytes:
    return b"\x00" * (int(sample_rate * duration_ms / 1000) * 2)


def make_wav_bytes(pcm16: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


class ModelSnapshotTest(unittest.TestCase):
    def test_sha_validation_and_fetch_marker(self):
        self.assertEqual(validate_sha40(VALID_SHA), VALID_SHA)
        for bad in ("", "short", VALID_SHA.upper(), "z" * 40):
            with self.assertRaises(ValueError):
                validate_sha40(bad, "VAD_CHECKPOINT")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"VAD_MODEL_ROOT": tmpdir}, clear=True):
                model_id, rev, path = vad_location()
            self.assertEqual(model_id, DEFAULT_VAD_MODEL_ID)
            self.assertEqual(rev, DEFAULT_VAD_CHECKPOINT)
            self.assertEqual(path, Path(tmpdir) / DEFAULT_VAD_CHECKPOINT)

            def fake_download(*, repo_id, revision, local_dir, allow_patterns):
                self.assertEqual(repo_id, DEFAULT_VAD_MODEL_ID)
                self.assertEqual(revision, DEFAULT_VAD_CHECKPOINT)
                self.assertIn("onnx/model.onnx", allow_patterns)
                target = Path(local_dir) / "onnx" / "model.onnx"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fake-onnx")

            fake_hub = SimpleNamespace(snapshot_download=fake_download)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                onnx_path = fetch(model_id, rev, path)
            self.assertTrue(onnx_path.is_file())
            self.assertEqual((path / ".snapshot-ready").read_text(encoding="utf-8"), f"{model_id}@{rev}\n")
            self.assertEqual(require_ready(model_id, rev, path), onnx_path)


class ChunkerTest(unittest.TestCase):
    def test_streaming_clause_splitting(self):
        c = SentenceChunker(min_clause_chars=6, max_clause_chars=24)
        self.assertEqual(c.feed("今天天气"), [])
        self.assertEqual(c.feed("真不错，我们"), ["今天天气真不错，"])
        self.assertEqual(c.feed("去散步吧！还有"), ["我们去散步吧！"])
        self.assertEqual(c.feed(" 3.14 "), [])
        self.assertEqual(c.flush(), ["还有 3.14"])

        # Overflow splitting on unbroken text
        c.reset()
        out = c.feed("abcdefghij klmnopqrstuvwxyz")
        self.assertTrue(len(out) >= 1)
        self.assertTrue(all(len(x) <= 24 for x in out))
        self.assertEqual(c.feed("   "), [])


class VadTest(unittest.TestCase):
    def test_energy_vad_click_rejection_prefix_and_turn(self):
        vad = EnergyVad(
            threshold=0.5,
            prefix_padding_ms=128,
            silence_duration_ms=256,
            barge_in_min_ms=160,  # 5 frames
        )
        # 1 frame (32ms) click noise -> ignored, no speech_started
        click = make_sine_pcm16(32) + make_silence_pcm16(128)
        events = vad.process_pcm16(click)
        self.assertEqual(events, [])

        # Feed 100ms chunks (non-multiple of 32ms!) to verify cross-chunk buffering
        speech = make_sine_pcm16(400)
        silence = make_silence_pcm16(400)
        all_events = []
        stream = speech + silence
        for i in range(0, len(stream), 3200):  # 100ms chunks
            all_events.extend(vad.process_pcm16(stream[i : i + 3200]))

        kinds = [e.kind for e in all_events]
        self.assertIn("speech_started", kinds)
        self.assertIn("speech_active", kinds)
        self.assertIn("speech_stopped", kinds)
        stopped = [e for e in all_events if e.kind == "speech_stopped"][0]
        # Includes prefix (128ms = 4096B) + speech (400ms) + silence tail (>=256ms)
        self.assertGreater(len(stopped.pcm16), len(speech))

    def test_silero_onnx_vad_tensor_contract(self):
        import numpy as np

        calls = []

        class FakeSession:
            def run(self, _outputs, inputs):
                inp = inputs["input"]
                state = inputs["state"]
                sr = inputs["sr"]
                self.assert_shapes(inp, state, sr)
                calls.append((inp.copy(), state.copy()))
                prob = np.array([[0.9 if len(calls) <= 6 else 0.05]], dtype=np.float32)
                new_state = np.full((2, 1, 128), float(len(calls)), dtype=np.float32)
                return prob, new_state

            def assert_shapes(self_inner, inp, state, sr):
                self.assertEqual(inp.shape, (1, 576))
                self.assertEqual(inp.dtype, np.float32)
                self.assertEqual(state.shape, (2, 1, 128))
                self.assertEqual(state.dtype, np.float32)
                self.assertEqual(int(sr), 16000)

        vad = SileroOnnxVad(
            session=FakeSession(),
            threshold=0.5,
            prefix_padding_ms=64,
            silence_duration_ms=96,
            min_speech_frames=2,
        )
        # Send 10 frames (320ms) -> first 6 frames high prob, last 4 low prob
        pcm = make_sine_pcm16(320)
        events = vad.process_pcm16(pcm)
        self.assertEqual(len(calls), 10)
        # Verify rolling 64-sample context propagated from frame 0 to frame 1
        np.testing.assert_allclose(calls[1][0][0, :64], calls[0][0][0, -64:])
        self.assertEqual(calls[1][1][0, 0, 0], 1.0)
        self.assertIn("speech_started", [e.kind for e in events])
        self.assertIn("speech_stopped", [e.kind for e in events])

        vad.reset()
        self.assertEqual(float(vad._state.sum()), 0.0)
        self.assertEqual(float(vad._context.sum()), 0.0)


class OmniDuplexServerTest(unittest.IsolatedAsyncioTestCase):
    async def serve(self, app):
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0]
        return f"http://{host}:{port}"

    async def asyncSetUp(self):
        self.asr_calls = []
        self.asr_transcript = "你好，请看画面"
        self.llm_responses_calls = []
        self.llm_chat_calls = []
        self.tts_stream_calls = []
        self.tts_synth_calls = []
        self.llm_clauses = ["你好！", "今天天气不错。"]
        self.llm_slow_delay = 0.0
        self.llm_cancelled_on_server = asyncio.Event()
        self.tts_cancelled_on_server = asyncio.Event()

        async def asr_transcribe(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer asr-secret")
            reader = await request.multipart()
            audio = b""
            resp_fmt = "json"
            while (part := await reader.next()) is not None:
                if part.name == "file":
                    audio = await part.read()
                elif part.name == "response_format":
                    resp_fmt = await part.text()
                else:
                    await part.release()
            self.asr_calls.append({"bytes": len(audio), "format": resp_fmt})
            return web.json_response({"text": self.asr_transcript})

        async def llm_tokenize(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer gw-secret")
            body = await request.json()
            self.assertEqual(body["model"], "llm-default")
            return web.json_response({"count": 42})

        async def llm_responses(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer gw-secret")
            body = await request.json()
            self.llm_responses_calls.append(body)
            if not body.get("stream"):
                return web.json_response(
                    {
                        "id": "resp_mock",
                        "output": [{"type": "message", "content": [{"type": "output_text", "text": "vision ok"}]}],
                    }
                )
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            try:
                await resp.write(b'event: response.created\ndata: {"type":"response.created"}\n\n')
                for clause in self.llm_clauses:
                    if self.llm_slow_delay:
                        await asyncio.sleep(self.llm_slow_delay)
                    ev = json.dumps({"type": "response.output_text.delta", "delta": clause}, ensure_ascii=False)
                    await resp.write(f"event: response.output_text.delta\ndata: {ev}\n\n".encode("utf-8"))
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return resp
            except asyncio.CancelledError:
                self.llm_cancelled_on_server.set()
                raise

        async def llm_chat(request):
            body = await request.json()
            self.llm_chat_calls.append(body)
            if body.get("stream"):
                resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await resp.prepare(request)
                await resp.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return resp
            return web.json_response(
                {"model": body["model"], "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
            )

        async def tts_stream(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer tts-secret")
            body = await request.json()
            self.tts_stream_calls.append(body)
            resp = web.StreamResponse(headers={"Content-Type": "application/octet-stream"})
            await resp.prepare(request)
            try:
                # 500ms of 24kHz mono PCM16 per clause (24000 bytes), split into 2 chunks
                pcm = make_sine_pcm16(500, sample_rate=24000, amplitude=3000)
                mid = len(pcm) // 2
                await resp.write(pcm[:mid])
                if self.llm_slow_delay:
                    await asyncio.sleep(self.llm_slow_delay)
                await resp.write(pcm[mid:])
                await resp.write_eof()
                return resp
            except asyncio.CancelledError:
                self.tts_cancelled_on_server.set()
                raise

        async def tts_synthesize(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer tts-secret")
            body = await request.json()
            self.tts_synth_calls.append(body)
            wav = make_wav_bytes(make_sine_pcm16(200, sample_rate=24000), sample_rate=24000)
            return web.Response(body=wav, content_type="audio/wav")

        asr_app = web.Application()
        asr_app.add_routes([web.post("/transcribe", asr_transcribe)])
        llm_app = web.Application()
        llm_app.add_routes(
            [
                web.post("/llm/v1/responses", llm_responses),
                web.post("/llm/v1/chat/completions", llm_chat),
                web.post("/llm/v1/chat/prompt_tokens", llm_tokenize),
            ]
        )
        tts_app = web.Application()
        tts_app.add_routes([web.post("/v1/tts/stream", tts_stream), web.post("/v1/tts/synthesize", tts_synthesize)])

        asr_url = await self.serve(asr_app)
        llm_url = await self.serve(llm_app)
        tts_url = await self.serve(tts_app)

        self.config = DuplexConfig(
            model_id="omni-duplex-qwen38",
            checkpoint=VALID_SHA,
            max_model_len=131072,
            asr_url=asr_url,
            asr_token="asr-secret",
            tts_url=tts_url,
            tts_token="tts-secret",
            llm_responses_url=f"{llm_url}/llm/v1/responses",
            llm_chat_url=f"{llm_url}/llm/v1/chat/completions",
            llm_tokenize_url=f"{llm_url}/llm/v1/chat/prompt_tokens",
            llm_api_key="gw-secret",
            llm_model="llm-default",
            barge_in_min_ms=96,  # 3 frames
        )
        app = create_app(
            self.config,
            vad_factory=lambda: EnergyVad(
                threshold=0.5,
                prefix_padding_ms=64,
                silence_duration_ms=160,
                barge_in_min_ms=96,
            ),
            workstation_cap=2,
        )
        self.base_url = await self.serve(app)
        self.client = ClientSession(base_url=self.base_url)
        self.addAsyncCleanup(self.client.close)

    async def _recv_until(self, ws, target_type: str, timeout: float = 4.0) -> list[dict]:
        events = []
        async with asyncio.timeout(timeout):
            while True:
                msg = await ws.receive_json()
                events.append(msg)
                if msg.get("type") == target_type:
                    return events

    async def test_health_models_and_auth_middleware(self):
        async with self.client.get("/livez") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(await resp.json(), {"state": "running"})
        async with self.client.get("/health") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(await resp.json(), {"status": "ready", "engine": "omni-duplex", "vad_ready": True})
        async with self.client.get("/version") as resp:
            self.assertEqual(await resp.json(), {"version": "omni-duplex 1.0"})
        async with self.client.get("/v1/models") as resp:
            data = await resp.json()
            self.assertEqual(data["data"][0]["id"], "omni-duplex-qwen38")
            self.assertEqual(data["data"][0]["root"], f"/models/{VALID_SHA}")
            self.assertEqual(data["data"][0]["max_model_len"], 131072)

        # Authenticated app leaves /livez & /health open, protects /v1/*
        authed_url = await self.serve(create_app(self.config, engine_token="eng-secret", workstation_cap=1))
        async with ClientSession(base_url=authed_url) as client:
            async with client.get("/health") as r:
                self.assertEqual(r.status, 200)
            async with client.get("/v1/models") as r:
                self.assertEqual(r.status, 401)
            async with client.get("/v1/models", headers={"Authorization": "Bearer eng-secret"}) as r:
                self.assertEqual(r.status, 200)

    async def test_autostart_variants_heartbeat_and_capacity(self):
        # duplex=1 required
        async with self.client.get("/v1/realtime") as resp:
            self.assertEqual(resp.status, 400)

        # autostart=1 emits session.created immediately
        async with self.client.ws_connect("/v1/realtime?duplex=1") as ws1:
            created = await ws1.receive_json()
            self.assertEqual(created["type"], "session.created")
            self.assertEqual(created["attachment_generation"], 1)
            self.assertTrue(created["resume_token"].startswith("rt_"))
            self.assertEqual(created["session"]["capabilities"]["implementation_level"], "engineered_duplex")
            self.assertEqual(created["session"]["audio"]["input"]["sample_rate_hz"], 16000)
            self.assertEqual(created["session"]["audio"]["output"]["format"]["rate"], 24000)

            await ws1.send_json({"type": "session.heartbeat"})
            hb = await ws1.receive_json()
            self.assertEqual(hb["type"], "session.heartbeat_ack")
            self.assertEqual(hb["session_id"], created["session"]["id"])

            await ws1.send_json({"type": "session.resume", "session_id": "x", "resume_token": "y"})
            err = await ws1.receive_json()
            self.assertEqual(err["type"], "error")
            self.assertEqual(err["error"]["code"], "session_resume_expired")

            # autostart=0 holds session.created until first session.update
            async with self.client.ws_connect("/v1/realtime?duplex=1&autostart=0") as ws2:
                # 3rd concurrent connection exceeds cap=2 -> 429
                with self.assertRaises(WSServerHandshakeError) as denied:
                    await self.client.ws_connect("/v1/realtime?duplex=1")
                self.assertEqual(denied.exception.status, 429)

                await ws2.send_json({"type": "session.update", "session": {"instructions": "Be brief."}})
                e1 = await ws2.receive_json()
                e2 = await ws2.receive_json()
                self.assertEqual(e1["type"], "session.created")
                self.assertEqual(e2["type"], "session.updated")
                self.assertEqual(e2["session"]["instructions"], "Be brief.")

            await ws1.send_json({"type": "session.close"})
            closed = await ws1.receive_json()
            self.assertEqual(closed["type"], "session.closed")

    async def test_full_voice_turn_with_video_frames_and_ref_audio(self):
        ref_wav_b64 = base64.b64encode(make_wav_bytes(make_sine_pcm16(100, 16000), 16000)).decode("ascii")
        bare_jpeg_b64 = base64.b64encode(b"\xff\xd8\xff\xe0fakejpeg").decode("ascii")

        async with self.client.ws_connect("/v1/realtime?duplex=1&autostart=0") as ws:
            await ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": "Speak concisely.",
                        "ref_audio": f"data:audio/wav;base64,{ref_wav_b64}",
                        "modalities": ["text", "audio"],
                        "extra_body": {"auto_response": True},
                    },
                }
            )
            await self._recv_until(ws, "session.updated")

            # Stream speech + bare base64 video_frames, then silence to close turn
            speech = make_sine_pcm16(240, 16000)
            silence = make_silence_pcm16(240, 16000)
            await ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(speech).decode("ascii"),
                    "video_frames": [bare_jpeg_b64],
                }
            )
            await ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(silence).decode("ascii"),
                }
            )

            events = await self._recv_until(ws, "response.done")
            types = [e["type"] for e in events]
            for required in (
                "input_audio_buffer.speech_started",
                "input_audio_buffer.speech_stopped",
                "conversation.item.input_audio_transcription.completed",
                "response.created",
                "response.speak",
                "response.output_audio.delta",
                "response.output_audio_transcript.delta",
                "response.output_audio.done",
                "response.output_audio_transcript.done",
                "response.done",
                "rate_limits.updated",
            ):
                # rate_limits.updated arrives right after response.done; check all up to response.done here
                if required != "rate_limits.updated":
                    self.assertIn(required, types)
            rl = await ws.receive_json()
            self.assertEqual(rl["type"], "rate_limits.updated")

            # Verify strictly monotonic server_event_seq
            seqs = [e["server_event_seq"] for e in events]
            self.assertEqual(seqs, sorted(seqs))
            self.assertEqual(len(set(seqs)), len(seqs))

            # Verify LLM received disable-thinking, reasoning.effort='none', and prefixed input_image
            llm_req = self.llm_responses_calls[-1]
            self.assertEqual(llm_req["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(llm_req["reasoning"], {"effort": "none"})
            self.assertEqual(llm_req["instructions"], "Speak concisely.")
            user_content = llm_req["input"][-1]["content"]
            self.assertEqual(user_content[0]["text"], self.asr_transcript)
            self.assertEqual(user_content[1]["type"], "input_image")
            self.assertEqual(user_content[1]["image_url"], f"data:image/jpeg;base64,{bare_jpeg_b64}")

            # Verify TTS received stripped ref_audio and produced 24kHz PCM16 audio deltas
            self.assertEqual(self.tts_stream_calls[0]["reference_audio_base64"], ref_wav_b64)
            audio_deltas = [e for e in events if e["type"] == "response.output_audio.delta"]
            self.assertTrue(len(audio_deltas) >= 2)
            self.assertTrue(all(d["sample_rate_hz"] == 24000 and d["format"] == "pcm16" for d in audio_deltas))
            tx_done = [e for e in events if e["type"] == "response.output_audio_transcript.done"][0]
            self.assertEqual(tx_done["transcript"], "你好！今天天气不错。")

    async def test_barge_in_cancels_streams_and_playback_ack_truncates_history(self):
        self.llm_slow_delay = 0.08
        self.llm_clauses = ["First clause!", "Second clause!"]

        async with self.client.ws_connect("/v1/realtime?duplex=1") as ws:
            created = await ws.receive_json()
            self.assertEqual(created["session"]["epoch"], 0)

            await ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Start"}]},
                }
            )
            await self._recv_until(ws, "conversation.item.done")
            await ws.send_json({"type": "response.create"})

            # Wait until all 500ms (24000 bytes @ 24kHz PCM16) of the first clause arrives,
            # while the second clause is still delayed on the mock server.
            started_events = []
            received_audio_bytes = 0
            async with asyncio.timeout(4.0):
                while received_audio_bytes < 24000:
                    msg = await ws.receive_json()
                    started_events.append(msg)
                    if msg.get("type") == "response.output_audio.delta":
                        received_audio_bytes += len(base64.b64decode(msg["delta"]))
            resp_id = next(e["response_id"] for e in started_events if "response_id" in e)
            item_id = next(e["item_id"] for e in started_events if "item_id" in e)

            # Trigger VAD barge-in via incoming user speech PCM during slow second clause
            await ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(make_sine_pcm16(160, 16000)).decode("ascii"),
                }
            )
            cancel_events = await self._recv_until(ws, "response.done")
            cancel_types = [e["type"] for e in cancel_events]
            self.assertIn("overlap.decision", cancel_types)
            self.assertIn("output_audio_buffer.cleared", cancel_types)
            cleared = [e for e in cancel_events if e["type"] == "output_audio_buffer.cleared"][0]
            done = [e for e in cancel_events if e["type"] == "response.done"][0]
            self.assertEqual(cleared["response_id"], resp_id)
            self.assertGreaterEqual(cleared["epoch"], 1)
            self.assertEqual(done["response"]["status"], "cancelled")
            self.assertEqual(done["response"]["status_details"]["reason"], "barge_in")

            # Acknowledge only 250ms of the 500ms first clause -> proportional truncation!
            await ws.send_json(
                {
                    "type": "playback.ack",
                    "response_id": resp_id,
                    "item_id": item_id,
                    "played_ms": 250,
                    "committed_ms": 250,
                }
            )
            acked_events = await self._recv_until(ws, "playback.acknowledged")
            acked = acked_events[-1]
            self.assertTrue(acked["event"]["history_committed"])

            await ws.send_json({"type": "conversation.item.retrieve", "item_id": item_id})
            ret = (await self._recv_until(ws, "conversation.item.retrieved"))[-1]
            kept = ret["item"]["content"][0]["transcript"]
            # "First clause!" (13 chars) truncated at 50% -> ~6-7 chars, definitely not "Second clause!"
            self.assertTrue(0 < len(kept) < len("First clause!"))
            self.assertNotIn("Second", kept)

            # Also verify explicit conversation.item.truncate to 0ms clears transcript
            await ws.send_json(
                {
                    "type": "conversation.item.truncate",
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": 0,
                }
            )
            trunc = (await self._recv_until(ws, "conversation.item.truncated"))[-1]
            self.assertEqual(trunc["audio_end_ms"], 0)
            await ws.send_json({"type": "conversation.item.retrieve", "item_id": item_id})
            ret_zero = (await self._recv_until(ws, "conversation.item.retrieved"))[-1]
            self.assertEqual(ret_zero["item"]["content"][0]["transcript"], "")

            # Also verify response.cancel on an active response bumps epoch and emits client_cancelled
            self.llm_clauses = ["One.", "Two."]
            await ws.send_json({"type": "response.create"})
            await self._recv_until(ws, "response.speak")
            await ws.send_json({"type": "response.cancel"})
            rc_events = await self._recv_until(ws, "response.done")
            rc_done = rc_events[-1]
            self.assertEqual(rc_done["response"]["status"], "cancelled")
            self.assertEqual(rc_done["response"]["status_details"]["reason"], "client_cancelled")
            self.assertGreaterEqual(rc_done["epoch"], 2)

    async def test_manual_commit_listen_clear_and_initial_user_text(self):
        async with self.client.ws_connect("/v1/realtime?duplex=1&autostart=0") as ws:
            await ws.send_json(
                {
                    "type": "session.update",
                    "session": {
                        "turn_detection": None,
                        "extra_body": {"auto_response": False, "duplex_initial_user_text": "Hello!"},
                    },
                }
            )
            events = await self._recv_until(ws, "response.done")
            types = [e["type"] for e in events]
            self.assertIn("session.created", types)
            self.assertIn("session.updated", types)
            self.assertEqual(self.llm_responses_calls[-1]["input"][-1]["content"][0]["text"], "Hello!")
            await self._recv_until(ws, "rate_limits.updated")

            # is_speech=False commit returns response.listen without calling ASR
            asr_before = len(self.asr_calls)
            await ws.send_json({"type": "input_audio_buffer.commit", "is_speech": False})
            listen_ev = (await self._recv_until(ws, "response.listen"))[-1]
            self.assertEqual(listen_ev["response"]["status"], "listening")
            self.assertEqual(len(self.asr_calls), asr_before)

            # input.cancel clears uncommitted buffer and bumps epoch
            epoch_before = listen_ev["epoch"]
            await ws.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(make_sine_pcm16(100, 16000)).decode("ascii"),
                }
            )
            await ws.send_json({"type": "input.cancel"})
            cleared = (await self._recv_until(ws, "input_audio_buffer.cleared"))[-1]
            self.assertEqual(cleared["epoch"], epoch_before + 1)

            # output_audio_buffer.clear bumps epoch and emits output_audio_buffer.cleared
            await ws.send_json({"type": "output_audio_buffer.clear"})
            out_cleared = (await self._recv_until(ws, "output_audio_buffer.cleared"))[-1]
            self.assertEqual(out_cleared["epoch"], epoch_before + 2)

    async def test_chat_completions_prompt_tokens_and_input_audio(self):
        async with self.client.post(
            "/v1/chat/prompt_tokens",
            json={"model": "omni-default", "messages": [{"role": "user", "content": "hi"}]},
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(await resp.json(), {"count": 42})

        # 1. Direct ASR transcription (gateway/input_transcription.py contract) -> 0 LLM calls
        wav_b64 = base64.b64encode(make_wav_bytes(make_sine_pcm16(100, 16000), 16000)).decode("ascii")
        async with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "omni-default",
                "modalities": ["text"],
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请仔细听这段音频片段，并将其内容逐字记录。"},
                            {"type": "input_audio", "input_audio": {"format": "wav", "data": wav_b64}},
                        ],
                    }
                ],
            },
        ) as resp:
            self.assertEqual(resp.status, 200)
            body = await resp.json()
            self.assertEqual(body["choices"][0]["message"]["content"], self.asr_transcript)
            self.assertEqual(len(self.llm_chat_calls), 0)

        # 2. Regular text chat + audio modality -> calls llm-engine and omni-tts
        async with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "omni-default",
                "modalities": ["text", "audio"],
                "messages": [{"role": "user", "content": "hello"}],
            },
        ) as resp:
            self.assertEqual(resp.status, 200)
            body = await resp.json()
            msg = body["choices"][0]["message"]
            self.assertEqual(msg["content"], "ok")
            self.assertEqual(msg["audio"]["format"], "wav")
            self.assertTrue(base64.b64decode(msg["audio"]["data"]).startswith(b"RIFF"))

        # 3. Streaming text chat
        async with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "omni-default",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        ) as resp:
            self.assertEqual(resp.status, 200)
            raw = await resp.read()
            self.assertIn(b'"content":"hi"', raw)
            self.assertIn(b"data: [DONE]", raw)

        # 4. Vision chat routes via /v1/responses cleanly
        async with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "omni-default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                        ],
                    }
                ],
            },
        ) as resp:
            self.assertEqual(resp.status, 200)
            body = await resp.json()
            self.assertEqual(body["choices"][0]["message"]["content"], "vision ok")
            self.assertEqual(self.llm_responses_calls[-1]["input"][0]["content"][1]["type"], "input_image")


if __name__ == "__main__":
    unittest.main()
