"""Unit tests for the standalone CPU faster-whisper ASR worker."""

import unittest

from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

from services.asr.app import create_app


class FakeWhisperBackend:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_bytes: bytes, *, language: str | None = None, filename: str = "audio.mp3") -> dict:
        self.calls.append((audio_bytes, language, filename))
        return {
            "text": "第一句 测试音频",
            "language": language or "zh",
            "duration": 5.2,
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": "第一句"},
                {"id": 1, "start": 2.6, "end": 5.2, "text": "测试音频"},
            ],
        }


class AsrWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = FakeWhisperBackend()
        app = create_app(self.backend, "asr-secret-token", max_concurrency=1, max_bytes=128)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.auth = {"Authorization": "Bearer asr-secret-token"}

    async def test_livez_allows_unauthenticated_probe(self):
        async with await self.client.get("/livez") as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(await resp.json(), {"state": "running"})

    async def test_transcribe_requires_worker_bearer_token(self):
        form = FormData()
        form.add_field("file", b"abc", filename="clip.mp3")
        async with await self.client.post("/transcribe", data=form) as resp:
            self.assertEqual(resp.status, 401)

    async def test_transcribe_supports_verbose_json_json_and_text(self):
        for fmt in ("verbose_json", "json", "text"):
            form = FormData()
            form.add_field("file", b"fake-mp3-data", filename="sample.mp3")
            form.add_field("language", "zh")
            form.add_field("response_format", fmt)
            async with await self.client.post("/transcribe", headers=self.auth, data=form) as resp:
                self.assertEqual(resp.status, 200)
                if fmt == "verbose_json":
                    data = await resp.json()
                    self.assertEqual(data["text"], "第一句 测试音频")
                    self.assertEqual(data["language"], "zh")
                    self.assertEqual(len(data["segments"]), 2)
                elif fmt == "json":
                    self.assertEqual(await resp.json(), {"text": "第一句 测试音频"})
                else:
                    self.assertEqual(await resp.text(), "第一句 测试音频")
        self.assertEqual(len(self.backend.calls), 3)
        self.assertEqual(self.backend.calls[0][1:], ("zh", "sample.mp3"))

    async def test_transcribe_rejects_missing_empty_or_oversized_audio(self):
        form = FormData()
        form.add_field("language", "zh")
        async with await self.client.post("/transcribe", headers=self.auth, data=form) as resp:
            self.assertEqual(resp.status, 400)

        empty_form = FormData()
        empty_form.add_field("file", b"", filename="empty.mp3")
        async with await self.client.post("/transcribe", headers=self.auth, data=empty_form) as resp:
            self.assertEqual(resp.status, 400)

        big_form = FormData()
        big_form.add_field("file", b"x" * 200, filename="big.mp3")
        async with await self.client.post("/transcribe", headers=self.auth, data=big_form) as resp:
            self.assertEqual(resp.status, 413)
