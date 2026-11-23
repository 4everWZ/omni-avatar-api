"""Tests for gateway /v1/audio/transcriptions proxy route."""

import unittest

from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from services.gateway.app import create_app
from services.gateway.config import Config
from services.gateway.transcription import AudioTranscriptionProxy


class GatewayTranscriptionTests(unittest.IsolatedAsyncioTestCase):
    async def serve_worker(self, handler):
        worker = web.Application()
        worker.add_routes([web.post("/transcribe", handler)])
        runner = web.AppRunner(worker, access_log=None)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0]
        return f"http://{host}:{port}"

    async def test_unconfigured_asr_returns_503_after_auth(self):
        cfg = Config("http://engine", "model", "pin", ("client",), "admin")
        app = create_app(cfg)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)

        form = FormData()
        form.add_field("file", b"audio", filename="a.mp3")
        async with await client.post("/v1/audio/transcriptions", data=form) as resp:
            self.assertEqual(resp.status, 401)

        form2 = FormData()
        form2.add_field("file", b"audio", filename="a.mp3")
        async with await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer client"},
            data=form2,
        ) as resp:
            self.assertEqual(resp.status, 503)
            data = await resp.json()
            self.assertEqual(data["error"]["code"], "asr_unavailable")

    async def test_configured_asr_proxies_multipart_to_worker(self):
        received = {}

        async def handle_worker(request):
            received["auth"] = request.headers.get("Authorization")
            reader = await request.multipart()
            while True:
                part = await reader.next()
                if part is None:
                    break
                if part.name == "file":
                    received["filename"] = part.filename
                    received["bytes"] = await part.read()
                else:
                    received[part.name] = await part.text()
            return web.json_response(
                {
                    "task": "transcribe",
                    "language": received.get("language") or "zh",
                    "duration": 3.4,
                    "text": "字幕转写成功",
                    "segments": [{"id": 0, "start": 0.0, "end": 3.4, "text": "字幕转写成功"}],
                }
            )

        worker_url = await self.serve_worker(handle_worker)
        cfg = Config("http://engine", "model", "pin", ("client",), "admin")
        app = create_app(
            cfg,
            asr_factory=lambda: AudioTranscriptionProxy(worker_url, "asr-worker-secret", timeout=5),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)

        form = FormData()
        form.add_field("file", b"mp3-bytes", filename="video.mp3", content_type="audio/mpeg")
        form.add_field("language", "zh")
        form.add_field("response_format", "verbose_json")
        async with await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer client"},
            data=form,
        ) as resp:
            self.assertEqual(resp.status, 200)
            payload = await resp.json()
            self.assertEqual(payload["text"], "字幕转写成功")
            self.assertEqual(payload["segments"][0]["end"], 3.4)
        self.assertEqual(received["auth"], "Bearer asr-worker-secret")
        self.assertEqual(received["filename"], "video.mp3")
        self.assertEqual(received["bytes"], b"mp3-bytes")
        self.assertEqual(received["language"], "zh")

    async def test_upstream_error_maps_to_502_and_timeout_to_504(self):
        import asyncio
        import socket

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            closed_port = sock.getsockname()[1]

        cfg = Config("http://engine", "model", "pin", ("client",), "admin")
        app = create_app(
            cfg,
            asr_factory=lambda: AudioTranscriptionProxy(
                f"http://127.0.0.1:{closed_port}", "asr-worker-secret", timeout=5
            ),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)

        form = FormData()
        form.add_field("file", b"mp3-bytes", filename="video.mp3")
        async with await client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer client"},
            data=form,
        ) as resp:
            self.assertEqual(resp.status, 502)
            data = await resp.json()
            self.assertEqual(data["error"]["code"], "engine_unavailable")

        async def slow_worker(request):
            await asyncio.sleep(0.5)
            return web.json_response({"text": "late"})

        slow_url = await self.serve_worker(slow_worker)
        slow_app = create_app(
            cfg,
            asr_factory=lambda: AudioTranscriptionProxy(slow_url, "asr-worker-secret", timeout=0.05),
        )
        slow_client = TestClient(TestServer(slow_app))
        await slow_client.start_server()
        self.addAsyncCleanup(slow_client.close)

        slow_form = FormData()
        slow_form.add_field("file", b"mp3-bytes", filename="video.mp3")
        async with await slow_client.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer client"},
            data=slow_form,
        ) as resp:
            self.assertEqual(resp.status, 504)
            data = await resp.json()
            self.assertEqual(data["error"]["code"], "engine_timeout")
