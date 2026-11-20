import asyncio
import base64
import hashlib
import hmac
import io
import json
import os
import unittest
import wave
from unittest.mock import patch

from aiohttp import ClientSession, ClientPayloadError, WSServerHandshakeError, web

from services.gateway.app import Config, create_app

SERVICE_AUTH = {"Authorization": "Bearer fixture-client"}
ADMIN_AUTH = {"Authorization": "Bearer fixture-admin"}
MEDIA_AUTH = {"Authorization": "Bearer fixture-media"}


class ConfigTest(unittest.TestCase):
    def test_chat_deadline_default_and_explicit_override(self):
        fixture = {
            "OMNI_GATEWAY_TOKEN": "fixture-client",
            "OMNI_GATEWAY_ADMIN_TOKEN": "fixture-admin",
            "OMNI_CHECKPOINT": "fixture-checkpoint",
            "OMNI_MODEL_ID": "fixtures/actual-omni",
        }
        with patch.dict(os.environ, fixture, clear=True):
            self.assertEqual(Config.from_env().request_timeout, 1200)
            os.environ["OMNI_REQUEST_TIMEOUT"] = "425"
            config = Config.from_env()
            self.assertEqual(config.request_timeout, 425)
            self.assertEqual(config.realtime_idle_timeout, 120)


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
                return web.json_response({"model": body["model"], "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}]})
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
        self.engine_url = engine_url
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

    async def test_retired_media_routes_are_unavailable(self):
        async with self.client.post('/v1/avatar/sessions', headers=SERVICE_AUTH, json={}) as response:
            self.assertEqual(response.status, 404)
        async with self.client.post('/v1/browser-tickets', headers=SERVICE_AUTH,
                                    json={'path': '/v1/avatar/sessions/test/events'}) as response:
            self.assertEqual(response.status, 400)

    async def test_browser_ticket_is_single_use_and_path_scoped(self):
        async with self.client.post('/v1/browser-tickets', headers=SERVICE_AUTH,
                                    json={'path': '/api/mode'}) as response:
            self.assertEqual(response.status, 400)
        async with self.client.post('/v1/browser-tickets', headers=SERVICE_AUTH,
                                    json={'path': '/v1/realtime'}) as response:
            ticket = (await response.json())['ticket']
        async with self.client.ws_connect(f'/v1/realtime?duplex=1&ticket={ticket}') as socket:
            self.assertEqual((await socket.receive_json())['type'], 'session.created')
        with self.assertRaises(WSServerHandshakeError) as rejected:
            await self.client.ws_connect(f'/v1/realtime?duplex=1&ticket={ticket}')
        self.assertEqual(rejected.exception.status, 401)

    async def test_explicit_private_audio_transcription(self):
        from services.gateway.input_transcription import PREFIX

        app = create_app(Config(engine_url=self.engine_url, model=self.model, checkpoint="fixture",
                                engine_token="fixture-engine", tokens=("fixture-client",), admin_token="fixture-admin",
                                batch_cap=2, shared_capacity=True))
        origin = await self.serve(app)
        mic = {"type":"input_audio_buffer.append", "sample_rate_hz":16000, "format":"pcm16"}
        async with ClientSession(base_url=origin) as client:
            async with client.post("/v1/browser-tickets", headers=SERVICE_AUTH, json={"path":"/v1/realtime"}) as issued:
                ticket = (await issued.json())["ticket"]
            async with client.ws_connect("/v1/realtime?duplex=1&input_transcription=explicit&ticket="+ticket) as ws:
                caps = (await ws.receive_json())["session"]["capabilities"]
                self.assertEqual(caps["gateway_input_transcription"]["mode"], "explicit_once")
                self.assertEqual(caps["gateway_input_transcription"]["transcript_receipt"], "hmac-sha256-v1")
                self.assertEqual(self.realtime_query, {"duplex":"1"})
                async def echo(event):
                    await ws.send_json(event)
                    self.assertEqual(await ws.receive_json(), event)
                await echo(mic | {"audio": "AAAA"})
                await ws.send_json({"event_id":"x"*192, "type":PREFIX+"start"})
                started = await ws.receive_json()
                self.assertEqual(started["type"], PREFIX+"started")
                capture = started["capture_id"]
                await echo({"type":"response.output_audio_transcript.done","transcript":"assistant answer"})
                await ws.send_json({"type":PREFIX+"finish", "capture_id":"wrong"})
                await echo(mic | {"audio":"AQIDBA=="})
                self.assertFalse(hasattr(self, "chat_payload"))
                await ws.send_json({"type":PREFIX+"finish", "capture_id":capture})
                result = await ws.receive_json()
                fields = ["omni.mic.v1", hashlib.sha256(ticket.encode()).hexdigest(), capture, 2, "ok"]
                signed = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()
                self.assertEqual(result.pop("receipt"), hmac.digest(b"fixture-client", signed, "sha256").hex())
                fields[0] = "omni.asr.v1"
                signed = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()
                self.assertEqual(result.pop("transcript_receipt"), hmac.digest(b"fixture-client", signed, "sha256").hex())
                self.assertEqual(result, {"type":PREFIX+"completed", "capture_id":capture,
                                 "samples":2, "source":"explicit_user_audio", "transcript":"ok"})
                message = self.chat_payload["messages"][-1]
                self.assertEqual(message["role"], "user")
                prompt, part = message["content"]
                self.assertEqual(prompt, {"type":"text", "text":"请仔细听这段音频片段，并将其内容逐字记录。"})
                with wave.open(io.BytesIO(base64.b64decode(part["input_audio"]["data"]))) as wav:
                    self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.readframes(100)), (16000, 1, bytes([1,2,3,4])))
                self.assertNotIn("assistant answer", str(self.chat_payload))
                await ws.send_json({"type":PREFIX+"start"})
                second = (await ws.receive_json())["capture_id"]
                self.assertNotEqual(second, capture)
                await ws.send_json({"type":PREFIX+"finish", "capture_id":second})
                self.assertEqual((await ws.receive_json())["reason"], "empty_audio")
        for query, auth in [("", SERVICE_AUTH), ("&input_transcription=explicit", SERVICE_AUTH),
                            ("&program=1&input_transcription=explicit", MEDIA_AUTH)]:
            async with self.client.ws_connect("/v1/realtime?duplex=1"+query, headers=auth) as ws:
                self.assertNotIn("gateway_input_transcription", (await ws.receive_json())["session"]["capabilities"])

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

    async def test_realtime_window_options_pass_through_at_client_creation(self):
        for extra in ({'sliding_window_mode': 'basic', 'basic_window_high_tokens': 8000,
                       'basic_window_low_tokens': 6000},
                      {'sliding_window_mode': 'context', 'context_max_units': 24,
                       'context_previous_max_tokens': 500}):
            async with self.client.ws_connect(
                '/v1/realtime?duplex=1&autostart=0', headers=SERVICE_AUTH,
            ) as socket:
                await socket.receive_json()
                self.assertEqual(self.realtime_query, {'duplex': '1', 'autostart': '0'})
                await socket.send_json({'type': 'session.update', 'session': {
                    'model': 'omni-default', 'extra_body': extra}})
                forwarded = await socket.receive_json()
                self.assertEqual(forwarded['session'], {'model': self.model, 'extra_body': extra})

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
