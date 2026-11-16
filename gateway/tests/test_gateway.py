import asyncio
import unittest

from aiohttp import ClientSession, ClientPayloadError, WSServerHandshakeError, web

from gateway import Config, create_app

SERVICE_AUTH = {"Authorization": "Bearer fixture-client"}
ADMIN_AUTH = {"Authorization": "Bearer fixture-admin"}
MEDIA_AUTH = {"Authorization": "Bearer fixture-media"}


class GatewayTest(unittest.IsolatedAsyncioTestCase):
    async def serve(self, app):
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        host, port = runner.addresses[0]
        return f"http://{host}:{port}"

    async def asyncSetUp(self):
        self.model = "fixtures/actual-omni"
        self.model_root = "model-fixture"
        self.expose_version = True
        self.chat_started = asyncio.Event()
        self.chat_ended = asyncio.Event()
        self.chat_release = asyncio.Event()
        self.realtime_ended = asyncio.Event()
        self.realtime_query = None

        async def models(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer fixture-engine")
            return web.json_response({"data": [{"id": self.model, "root": self.model_root}]})

        async def version(request):
            if self.expose_version:
                return web.json_response({"version": "fixture-engine-version"})
            return web.Response(status=404)

        async def health(request):
            return web.Response()

        async def chat(request):
            body = await request.json()
            self.chat_payload = body
            self.assertEqual(request.headers.get("Authorization"), "Bearer fixture-engine")
            self.assertEqual(body["model"], self.model)
            if not body.get("stream"):
                return web.json_response({"model": body["model"], "choices": [{"message": {"content": "ok"}}]})
            try:
                response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
                await response.prepare(request)
                await response.write(b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n')
                self.chat_started.set()
                await self.chat_release.wait()
                await response.write(b"data: [DONE]\n\n")
                return response
            finally:
                self.chat_ended.set()

        async def prompt_tokens(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer fixture-engine")
            body = await request.json()
            self.assertEqual(body["model"], self.model)
            self.assertEqual(body["modalities"], ["text"])
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
            return web.json_response({"count": 40900})

        async def realtime(request):
            self.assertEqual(request.headers.get("Authorization"), "Bearer fixture-engine")
            self.realtime_query = dict(request.query)
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            try:
                await socket.send_json({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": 1000}}})
                async for frame in socket:
                    await socket.send_str(frame.data)
            finally:
                self.realtime_ended.set()
            return socket

        engine = web.Application()
        engine.add_routes([
            web.get("/health", health), web.get("/version", version), web.get("/v1/models", models),
            web.post("/v1/chat/completions", chat), web.get("/v1/realtime", realtime),
            web.post("/v1/chat/prompt_tokens", prompt_tokens),
        ])
        engine_url = await self.serve(engine)
        self.url = await self.serve(create_app(Config(
            engine_url=engine_url, model=self.model, checkpoint="fixture-pinned-revision",
            tokens=("fixture-client", "fixture-second-client", "fixture-media"), admin_token="fixture-admin",
            program_token="fixture-media",
            engine_token="fixture-engine", engine_version="fixture-declared-version", model_root=self.model_root,
            queue_timeout=0.1, request_timeout=0.8, realtime_idle_timeout=0.4,
        )))
        self.client = ClientSession(base_url=self.url)
        self.addAsyncCleanup(self.client.close)

    async def chat(self, stream=False, *, headers=SERVICE_AUTH, **options):
        return await self.client.post("/v1/chat/completions", headers=headers, json={
            "model": "omni-default", "stream": stream, "messages": [{"role": "user", "content": "test"}],
            **options,
        })

    async def test_chat_defaults_to_text_and_preserves_explicit_modalities(self):
        for options, expected in (
            ({"response_format": {"type": "json_object"}}, ["text"]),
            ({"modalities": None}, ["text"]),
            ({"modalities": ["text", "audio"]}, ["text", "audio"]),
        ):
            async with await self.chat(**options) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(self.chat_payload["modalities"], expected)

    async def test_prompt_tokens_use_authenticated_engine_route_without_starting_chat(self):
        body = {"model": "omni-default", "messages": [{"role": "user", "content": "原文"}],
                "chat_template_kwargs": {"enable_thinking": True}}
        async with self.client.post("/v1/chat/prompt_tokens", json=body) as response:
            self.assertEqual(response.status, 401)
        async with self.client.post("/v1/chat/prompt_tokens", headers=SERVICE_AUTH, json=body) as response:
            self.assertEqual(await response.json(), {"count": 40900})
        self.assertFalse(self.chat_started.is_set())

    async def test_auth_model_discovery_and_health_are_engine_bound(self):
        async with await self.chat(headers={}) as response:
            self.assertEqual(response.status, 401)
        async with self.client.put("/api/mode", headers=SERVICE_AUTH, json={"mode": "live"}) as response:
            self.assertEqual(response.status, 403)
        async with self.client.get("/v1/models", headers=SERVICE_AUTH) as response:
            self.assertEqual((await response.json())["data"][0]["id"], "omni-default")
        async with await self.chat(headers={"Authorization": "Bearer fixture-second-client"}) as response:
            self.assertEqual((await response.json())["choices"][0]["message"]["content"], "ok")
        async with self.client.get("/healthz", headers=SERVICE_AUTH) as response:
            result = await response.json()
            self.assertEqual(result["model"], self.model)
            self.assertEqual(result["engine_version"], "fixture-engine-version")
            self.assertEqual(result["engine_version_source"], "engine")
            self.assertEqual(result["checkpoint_source"], "deployment")
            self.assertTrue(result["checkpoint_root_verified"])
            self.assertTrue(result["protocols"]["chat_completions"])
            self.assertFalse(result["protocols"]["realtime_duplex"])
        self.expose_version = False
        async with self.client.get("/healthz", headers=SERVICE_AUTH) as response:
            result = await response.json()
            self.assertEqual(result["engine_version"], "fixture-declared-version")
            self.assertEqual(result["engine_version_source"], "deployment")
        self.model_root = "unexpected"
        async with self.client.get("/healthz", headers=SERVICE_AUTH) as response:
            self.assertEqual(response.status, 503)
            self.assertIsNone((await response.json())["model"])

    async def test_live_mode_reserves_websocket_while_chat_is_busy(self):
        async with await self.chat(stream=True) as response:
            self.assertIn(b"first", await response.content.readline())
            with self.assertRaises(WSServerHandshakeError) as denied:
                await self.client.ws_connect("/v1/realtime?duplex=1", headers=SERVICE_AUTH)
            self.assertEqual(denied.exception.status, 429)
            async with self.client.put("/api/mode", headers=ADMIN_AUTH, json={"mode": "live"}) as mode:
                self.assertEqual((await mode.json())["mode"], "live")
            async with self.client.ws_connect(
                "/v1/realtime?duplex=1&model=omni-default&autostart=0", headers=SERVICE_AUTH,
            ) as socket:
                self.assertEqual((await socket.receive_json())["type"], "session.created")
                self.assertEqual(self.realtime_query, {"model": self.model, "duplex": "1", "autostart": "0"})
                await socket.send_json({"type": "playback.ack", "played_ms": 20})
                self.assertEqual(await socket.receive_json(), {"type": "playback.ack", "played_ms": 20})
                async with await self.chat() as ordinary:
                    self.assertEqual(ordinary.status, 429)
                async with self.client.get("/healthz", headers=SERVICE_AUTH) as health:
                    result = await health.json()
                    self.assertEqual(result["active"], {"chat": 1, "realtime": 1})
                    self.assertEqual(result["realtime_capabilities"], {"chunk_period_ms": 1000})
                self.chat_release.set()
                self.assertTrue((await response.read()).endswith(b"data: [DONE]\n\n"))
        await asyncio.wait_for(self.realtime_ended.wait(), 2)
        async with await self.chat() as response:
            self.assertEqual(response.status, 200)

    async def test_stream_disconnect_cancels_upstream_and_releases_lane(self):
        response = await self.chat(stream=True)
        self.assertIn(b"first", await response.content.readline())
        response.close()
        await asyncio.wait_for(self.chat_ended.wait(), 2)
        async with await self.chat() as next_response:
            self.assertEqual(next_response.status, 200)

    async def test_scoped_media_socket_controls_live_mode_until_disconnect(self):
        url = "/v1/realtime?duplex=1&program=1"
        with self.assertRaises(WSServerHandshakeError) as denied:
            await self.client.ws_connect(url, headers=SERVICE_AUTH)
        self.assertEqual(denied.exception.status, 403)
        async with await self.chat(stream=True) as response:
            self.assertIn(b"first", await response.content.readline())
            async with self.client.ws_connect(url, headers=MEDIA_AUTH) as socket:
                self.assertEqual((await socket.receive_json())["type"], "session.created")
                self.assertEqual(self.realtime_query, {"duplex": "1"})
                async with self.client.get("/api/mode", headers=SERVICE_AUTH) as status:
                    self.assertEqual((await status.json())["mode"], "live")
            async with asyncio.timeout(2):
                while True:
                    async with self.client.get("/api/mode", headers=SERVICE_AUTH) as status:
                        if (await status.json())["mode"] == "batch":
                            break
                    await asyncio.sleep(0.01)
            self.chat_release.set()
            await response.read()

    async def test_sse_timeout_never_fabricates_completion(self):
        async with await self.chat(stream=True) as response:
            self.assertIn(b"first", await response.content.readline())
            with self.assertRaises(ClientPayloadError):
                await response.read()
        await asyncio.wait_for(self.chat_ended.wait(), 2)
        async with await self.chat() as next_response:
            self.assertEqual(next_response.status, 200)

    async def test_realtime_idle_timeout_closes_both_connections(self):
        async with self.client.ws_connect("/v1/realtime?duplex=1", headers=SERVICE_AUTH) as socket:
            await socket.receive_json()
            close = await asyncio.wait_for(socket.receive(), 2)
            self.assertEqual(close.data, 1001)
        await asyncio.wait_for(self.realtime_ended.wait(), 2)
        async with await self.chat() as response:
            self.assertEqual(response.status, 200)
