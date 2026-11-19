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
