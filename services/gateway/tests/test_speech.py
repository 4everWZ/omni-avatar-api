import base64
import io
import json
import unittest
import wave

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from services.gateway.app import Config, create_app


class SpeechTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setparams((1, 2, 24000, 0, "NONE", "PCM"))
            w.writeframes(b"\x04\x00" * 1200)
        self.wav = wav.getvalue()
        self.audio = base64.b64encode(self.wav).decode()
        self.input = {"model": "omni-default", "input": "收到！", "reference_audio": {"format": "wav", "data": self.audio}}
        self.calls = []
        self.heard = "收到。"

        async def engine(request):
            payload = await request.json()
            self.calls.append(payload)
            if payload["modalities"] == ["text"]:
                self.assertNotIn("收到", json.dumps(payload, ensure_ascii=False))
                msg = {"content": json.dumps({"transcript": self.heard})}
            else:
                self.assertEqual(payload["ref_audio"], "data:audio/wav;base64," + self.audio)
                system, user = payload["messages"]
                self.assertEqual(system["role"], "system")
                self.assertEqual([part["type"] for part in system["content"]], ["text", "input_audio", "text"])
                self.assertEqual(system["content"][1]["input_audio"], self.input["reference_audio"])
                self.assertIn(self.input["input"], user["content"])
                self.assertEqual(payload["sampling_params_list"][1]["min_tokens"], 1)
                msg = {"content": "收到！", "audio": {"data": self.audio}}
            return web.json_response({"choices": [{"finish_reason": "stop", "message": msg}]})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", engine)
        upstream = TestServer(app)
        await upstream.start_server()
        self.addAsyncCleanup(upstream.close)
        config = Config(engine_url=str(upstream.make_url("")).rstrip("/"), model="actual", checkpoint="fixture",
                        tokens=("ordinary-test-key",), admin_token="unused-fixture-key", queue_limit=0)
        self.gateway = create_app(config)
        self.client = TestClient(TestServer(self.gateway))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.auth = {"Authorization": "Bearer ordinary-test-key"}

    async def test_verified_audio_requires_separate_audio_only_readback(self):
        response = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input | {"voice": "reference"})
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), self.wav)
        self.assertEqual(base64.b64decode(response.headers["X-Omni-Transcript"]).decode(), self.heard)
        self.assertEqual(len(self.calls), 2)
        self.heard = "收到就收到"
        mismatch = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(mismatch.status, 422)
        self.assertNotIn("X-Omni-Transcript", mismatch.headers)

    async def test_auth_reference_boundary_and_shared_admission(self):
        response = await self.client.post("/v1/audio/speech", json=self.input)
        self.assertEqual(response.status, 401)
        for patch in ({"reference_audio": "https://example.com/sample.wav"}, {"model": "another"}, {"voice": "another"},
                      {"sampling_params_list": []}, {"reference_audio": {"format": "wav", "data": "aGVsbG8="}}):
            response = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input | patch)
            self.assertEqual(response.status, 400)
        async with self.gateway["gateway"].admission.hold("chat"):
            response = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
            self.assertEqual(response.status, 429)
        self.assertEqual(self.calls, [])


class WorkerSpeechTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setparams((1, 2, 24000, 0, "NONE", "PCM"))
            w.writeframes(b"\x04\x00" * 1200)
        self.wav = wav.getvalue()
        bad_wav = io.BytesIO()
        with wave.open(bad_wav, "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", "PCM"))
            w.writeframes(b"\x04\x00" * 800)
        self.bad_wav = bad_wav.getvalue()
        self.audio = base64.b64encode(self.wav).decode()
        self.input = {"model": "omni-default", "input": "收到！", "reference_audio": {"format": "wav", "data": self.audio}}
        self.tts_calls = []
        self.asr_calls = []
        self.omni_calls = []
        self.tts_status = 200
        self.tts_body = self.wav
        self.tts_delay = 0.0
        self.asr_status = 200
        self.asr_text = "收到。"
        self.asr_delay = 0.0

        async def omni_chat(request):
            self.omni_calls.append(await request.json())
            return web.json_response({"choices": []})

        async def tts_synthesize(request):
            import asyncio
            if self.tts_delay:
                await asyncio.sleep(self.tts_delay)
            self.tts_calls.append({
                "auth": request.headers.get("Authorization"),
                "json": await request.json(),
            })
            if self.tts_status != 200:
                return web.Response(status=self.tts_status, text="tts-err")
            return web.Response(body=self.tts_body, content_type="audio/wav")

        async def asr_transcribe(request):
            import asyncio
            if self.asr_delay:
                await asyncio.sleep(self.asr_delay)
            reader = await request.multipart()
            parts = {}
            while (field := await reader.next()) is not None:
                if field.name == "file":
                    parts["filename"] = field.filename
                    parts["content_type"] = field.headers.get("Content-Type")
                    parts["bytes"] = await field.read()
                else:
                    parts[field.name] = await field.text()
            self.asr_calls.append({
                "auth": request.headers.get("Authorization"),
                "parts": parts,
            })
            if self.asr_status != 200:
                return web.Response(status=self.asr_status, text="asr-err")
            return web.json_response({"text": self.asr_text})

        omni_app = web.Application()
        omni_app.router.add_post("/v1/chat/completions", omni_chat)
        self.omni_srv = TestServer(omni_app)
        await self.omni_srv.start_server()
        self.addAsyncCleanup(self.omni_srv.close)

        tts_app = web.Application()
        tts_app.router.add_post("/v1/tts/synthesize", tts_synthesize)
        self.tts_srv = TestServer(tts_app)
        await self.tts_srv.start_server()
        self.addAsyncCleanup(self.tts_srv.close)

        asr_app = web.Application()
        asr_app.router.add_post("/transcribe", asr_transcribe)
        self.asr_srv = TestServer(asr_app)
        await self.asr_srv.start_server()
        self.addAsyncCleanup(self.asr_srv.close)

        self.config = Config(
            engine_url=str(self.omni_srv.make_url("")).rstrip("/"),
            model="actual",
            checkpoint="fixture",
            tokens=("ordinary-test-key",),
            admin_token="unused-fixture-key",
            queue_limit=0,
            tts_worker_url=str(self.tts_srv.make_url("")).rstrip("/"),
            tts_internal_token="tts-secret",
            tts_timeout=0.3,
            asr_worker_url=str(self.asr_srv.make_url("")).rstrip("/"),
            asr_internal_token="asr-secret",
            asr_timeout=0.3,
        )
        self.gateway = create_app(self.config)
        self.client = TestClient(TestServer(self.gateway))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.auth = {"Authorization": "Bearer ordinary-test-key"}

    async def test_tts_and_asr_blind_readback_success_and_mismatch(self):
        response = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.content_type, "audio/wav")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(await response.read(), self.wav)
        self.assertEqual(base64.b64decode(response.headers["X-Omni-Transcript"]).decode(), "收到。")
        self.assertEqual(self.omni_calls, [])
        self.assertEqual(self.tts_calls, [{
            "auth": "Bearer tts-secret",
            "json": {"text": "收到！", "reference_audio_base64": self.audio},
        }])
        self.assertEqual(self.asr_calls, [{
            "auth": "Bearer asr-secret",
            "parts": {
                "filename": "speech.wav",
                "content_type": "audio/wav",
                "bytes": self.wav,
                "response_format": "json",
            },
        }])

        self.asr_text = "收到就收到"
        mismatch = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(mismatch.status, 422)
        self.assertEqual((await mismatch.json())["error"]["code"], "speech_unverified")
        self.assertNotIn("X-Omni-Transcript", mismatch.headers)

        self.tts_body = self.bad_wav
        bad_format = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(bad_format.status, 422)
        self.assertEqual(len(self.asr_calls), 2)

    async def test_worker_speech_error_and_timeout_mapping(self):
        async with self.gateway["gateway"].admission.hold("chat"):
            busy = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
            self.assertEqual(busy.status, 429)

        self.tts_status = 429
        tts_busy = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(tts_busy.status, 429)

        self.tts_status = 500
        tts_err = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(tts_err.status, 502)

        self.tts_status = 200
        self.asr_status = 503
        asr_err = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(asr_err.status, 502)

        self.asr_status = 200
        self.tts_delay = 0.5
        tts_to = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(tts_to.status, 504)
        self.tts_delay = 0.0

        self.asr_delay = 0.5
        asr_to = await self.client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(asr_to.status, 504)
        self.asr_delay = 0.0

        from dataclasses import replace
        no_asr = create_app(replace(self.config, asr_worker_url=""))
        no_asr_client = TestClient(TestServer(no_asr))
        await no_asr_client.start_server()
        self.addAsyncCleanup(no_asr_client.close)
        missing_asr = await no_asr_client.post("/v1/audio/speech", headers=self.auth, json=self.input)
        self.assertEqual(missing_asr.status, 502)
