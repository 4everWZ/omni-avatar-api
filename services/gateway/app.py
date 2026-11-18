"""One local admission authority for an upstream vLLM-Omni deployment."""

import asyncio
import json
import os
import secrets
import hashlib
from pathlib import Path
from time import monotonic
from contextlib import AsyncExitStack

from aiohttp import ClientError, ClientSession, ClientTimeout, ClientWSTimeout, DummyCookieJar, UnixConnector, WSMsgType, web
from .admission import Admission, Busy, BudgetExceeded
from .token_budget import reservation, ReservationRejected
from .config import Config, MODEL_ALIAS
from .avatar_proxy import AvatarProxy
from .responses_proxy import ResponsesProxy
from .image_jobs import DedicatedImageJobs, ImageJobs
from .thinking import apply_thinking_default

MAX_BODY = 32 * 1024 * 1024
MAX_FRAME = 8 * 1024 * 1024


def failure(status, code, message, *, headers=None):
    return web.json_response({"error": {"code": code, "message": message}}, status=status, headers=headers)


class Gateway:
    def __init__(self, config):
        self.config = config
        self.admission = Admission(config)
        self.capabilities = None
        self.chat_observed = False
        self.avatar = AvatarProxy(config.avatar_url, config.avatar_token)
        self.browser_tickets = {}
        self.responses = ResponsesProxy(self) if not config.realtime_enabled else None

    @web.middleware
    async def authenticate(self, request, handler):
        path = request.path.removeprefix(self.config.public_prefix) if self.config.public_prefix else request.path
        if path in ("/livez", "/", "/app.js", "/style.css", "/mic-worklet.js"):
            return await handler(request)
        if request.method == 'GET' and 'ticket' in request.query:
            entry = self.browser_tickets.pop(request.query['ticket'], None)
            if entry and entry['expires'] > monotonic() and entry['path'] == request.path:
                request['owner'] = entry['owner']
                request['service_token'] = entry['token']
                return await handler(request)
            return failure(401, 'invalid_ticket', 'Browser ticket expired or already used.')
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        token = token.encode()
        admin = secrets.compare_digest(token, self.config.admin_token.encode())
        if scheme.lower() != "bearer" or not (admin or any(
            secrets.compare_digest(token, value.encode()) for value in self.config.tokens
        )):
            return failure(401, "unauthorized", "A valid service bearer token is required.",
                           headers={"WWW-Authenticate": "Bearer"})
        if path == "/api/mode" and request.method == "PUT" and not admin:
            return failure(403, "forbidden", "An admin bearer token is required.")
        request['owner'] = hashlib.sha256(token).hexdigest()
        request['service_token'] = token.decode()
        return await handler(request)

    async def browser_ticket(self, request):
        try:
            body = await request.json()
            path = body['path']
            if path != '/v1/realtime' and not (
                    isinstance(path, str) and path.startswith('/v1/avatar/sessions/')
                    and path.endswith('/events') and len(path.split('/')) == 6):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            return failure(400, 'invalid_ticket_path', 'Select a Realtime or Avatar event socket.')
        self.browser_tickets = {key: value for key, value in self.browser_tickets.items()
                                if value['expires'] > monotonic()}
        if len(self.browser_tickets) >= 256:
            return failure(429, 'ticket_capacity', 'Too many pending browser connections.')
        ticket = secrets.token_urlsafe(32)
        self.browser_tickets[ticket] = {'owner': request['owner'], 'token': request['service_token'],
                                       'path': path, 'expires': monotonic() + 30}
        return web.json_response({'ticket': ticket, 'expires_in': 30}, headers={'Cache-Control': 'no-store'})

    async def session_context(self, app):
        headers = {"Authorization": f"Bearer {self.config.engine_token}"} if self.config.engine_token else {}
        async with ClientSession(
            base_url=self.config.engine_url, headers=headers, cookie_jar=DummyCookieJar(),
            timeout=ClientTimeout(total=None, connect=self.config.health_timeout), trust_env=False,
        ) as session:
            self.engine = session
            async with ClientSession(timeout=ClientTimeout(total=None, connect=3),
                                     connector=UnixConnector(path=self.config.avatar_socket) if self.config.avatar_socket else None,
                                     cookie_jar=DummyCookieJar(), trust_env=False) as avatar_http:
                app['avatar_http'] = avatar_http
                try:
                    yield
                finally:
                    if self.responses:
                        await self.responses.close()

    async def responses_request(self, request):
        try:
            return await self.responses.handle(request)
        except Busy:
            return failure(429, 'gateway_busy', 'The inference lane is at capacity.', headers={'Retry-After': '1'})
        except ReservationRejected as error:
            return error.response()

    async def mode(self, request):
        if request.method == "PUT":
            try:
                value = await request.json()
            except (ValueError, UnicodeError):
                value = None
            if not isinstance(value, dict) or value.get("mode") not in ("live", "batch"):
                return failure(400, "invalid_mode", "Select live or batch mode.")
            self.admission.set_mode(value["mode"])
        return web.json_response(self.admission.view())

    async def list_engine_models(self):
        async with self.engine.get("/v1/models", allow_redirects=False) as response:
            response.raise_for_status()
            data = (await response.json())["data"]
        model = next((item for item in data if item["id"] == self.config.model), None)
        if self.config.model_root and model and model.get("root") != self.config.model_root:
            return None
        return model

    async def ready_model(self):
        try:
            async with asyncio.timeout(self.config.health_timeout):
                async with self.engine.get("/health", allow_redirects=False) as response:
                    if response.status == 200:
                        return await self.list_engine_models()
        except (TimeoutError, ClientError, ValueError, TypeError, KeyError):
            pass

    async def engine_version(self):
        try:
            async with asyncio.timeout(self.config.health_timeout):
                async with self.engine.get("/version", allow_redirects=False) as response:
                    if response.status == 200 and (actual := (await response.json()).get("version")):
                        return actual, "engine"
        except (TimeoutError, ClientError, ValueError, AttributeError):
            pass
        return (self.config.engine_version, "deployment") if self.config.engine_version else (None, None)

    async def health(self, request):
        model, (version, version_source) = await asyncio.gather(self.ready_model(), self.engine_version())
        state = 'ready' if model else 'unavailable'
        if self.admission.paused == 'image_gpu_switch':
            state = 'busy'
        elif self.admission.paused:
            state = 'unavailable'
        if not model:
            self.capabilities, self.chat_observed = None, False
        return web.json_response({
            "state": state, "alias": self.config.model_alias,
            "model": model["id"] if model else None, "configured_model": self.config.model,
            "model_root": model.get("root") if model else None,
            "checkpoint": self.config.checkpoint, "checkpoint_source": "deployment",
            "checkpoint_root_verified": bool(model and self.config.model_root),
            "engine_version": version, "engine_version_source": version_source,
            "protocols": {"chat_completions": self.chat_observed, "realtime_duplex": self.capabilities is not None},
            "realtime_capabilities": self.capabilities, **self.admission.view(),
        }, status=200 if state in ('ready', 'busy') else 503)

    async def aliased_models(self, request):
        try:
            async with asyncio.timeout(self.config.health_timeout):
                model = await self.list_engine_models()
        except (TimeoutError, ClientError, ValueError, TypeError, KeyError):
            model = None
        if model is None:
            return failure(503, "model_unavailable", "The configured engine model is unavailable.")
        return web.json_response({"object": "list", "data": [model | {"id": self.config.model_alias, "root": self.config.model_alias}]})

    async def chat_payload(self, request):
        try:
            async with asyncio.timeout(self.config.request_timeout):
                payload = await request.json()
        except (ValueError, UnicodeError):
            return None, failure(400, "invalid_json", "Expected a JSON Chat Completions request.")
        except TimeoutError:
            return None, failure(408, "request_timeout", "The request body did not arrive in time.")
        if not isinstance(payload, dict) or payload.get("model") != self.config.model_alias:
            return None, failure(400, "model_not_found", f"Use the {self.config.model_alias} alias.")
        payload["model"] = self.config.model
        apply_thinking_default(payload)
        if payload.get("modalities") is None:
            payload["modalities"] = ["text"]
        return payload, None

    async def prompt_tokens(self, request):
        payload, error = await self.chat_payload(request)
        if error is not None:
            return error
        try:
            async with asyncio.timeout(self.config.health_timeout):
                if self.config.prompt_tokens_path == "/tokenize":
                    payload = {key: value for key, value in payload.items()
                               if key in {"model", "messages", "tools", "chat_template_kwargs"}}
                    payload["add_generation_prompt"] = True
                async with self.engine.post(self.config.prompt_tokens_path, json=payload, allow_redirects=False) as upstream:
                    if self.config.prompt_tokens_path == "/tokenize" and upstream.status == 200:
                        return web.json_response({"count": (await upstream.json())["count"]})
                    return web.Response(status=upstream.status, body=await upstream.read(), content_type="application/json")
        except TimeoutError:
            return failure(504, "tokenizer_timeout", "The engine did not return a prompt token count in time.")
        except ClientError:
            return failure(502, "engine_unavailable", "The engine could not count prompt tokens.")

    async def chat(self, request):
        payload, error = await self.chat_payload(request)
        if error is not None:
            return error
        outgoing = None
        try:
            async with asyncio.timeout(self.config.request_timeout):
                budget = await reservation(self, 'chat', payload)
            async with self.admission.hold("chat", **budget), asyncio.timeout(self.config.request_timeout):
                async with self.engine.post("/v1/chat/completions", json=payload, allow_redirects=False) as upstream:
                    content_type = upstream.headers.get("Content-Type", "application/json")
                    accepted = 200 <= upstream.status < 300
                    if upstream.content_type != "text/event-stream":
                        body = await upstream.read()
                        self.chat_observed |= accepted
                        return web.Response(status=upstream.status, body=body,
                                            headers={"Content-Type": content_type})
                    outgoing = web.StreamResponse(status=upstream.status, headers={
                        "Content-Type": content_type, "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                    })
                    await outgoing.prepare(request)
                    async for chunk in upstream.content.iter_any():
                        await outgoing.write(chunk)
                    await outgoing.write_eof()
                    self.chat_observed |= accepted
                    return outgoing
        except Busy:
            return failure(429, "gateway_busy", "The inference lane is at capacity.", headers={"Retry-After": "1"})
        except BudgetExceeded as error:
            return failure(400, 'request_budget_exceeded', str(error))
        except ReservationRejected as error:
            return error.response()
        except (TimeoutError, ClientError) as error:
            if outgoing is not None and outgoing.prepared:
                if request.transport:
                    request.transport.close()
                return outgoing
            if isinstance(error, TimeoutError):
                return failure(504, "engine_timeout", "Inference exceeded the gateway deadline.")
            return failure(502, "engine_unavailable", "The inference engine could not serve the request.")

    async def realtime(self, request):
        if request.query.get("duplex") != "1" or request.query.get("model", MODEL_ALIAS) != MODEL_ALIAS:
            return failure(400, "invalid_realtime_request", f"Use duplex=1 with the {MODEL_ALIAS} alias.")
        program = request.query.get("program") == "1"
        if program:
            token = request['service_token']
            if not self.config.program_token or not secrets.compare_digest(
                token.encode(), self.config.program_token.encode()
            ):
                return failure(403, "program_forbidden", "The media service credential is required.")
        downstream = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_FRAME, compress=False)
        if not downstream.can_prepare(request).ok:
            return failure(400, "upgrade_required", "A WebSocket upgrade is required.")
        query = {name: request.query[name] for name in ("autostart", "resume") if name in request.query}
        query["duplex"] = "1"
        if "model" in request.query:
            query["model"] = self.config.model
        try:
            avatar_live = request.query.get('avatar') == '1' and bool(self.config.avatar_url)
            async with self.admission.program(program or avatar_live), self.admission.hold("realtime"), AsyncExitStack() as stack:
                async with asyncio.timeout(self.config.health_timeout):
                    upstream = await stack.enter_async_context(self.engine.ws_connect(
                        "/v1/realtime", params=query, heartbeat=20, max_msg_size=MAX_FRAME,
                        timeout=ClientWSTimeout(ws_close=3),
                    ))
                await downstream.prepare(request)
                tap = None
                if avatar_live and self.config.avatar_token:
                    source_id = secrets.token_urlsafe(24)
                    tap = await stack.enter_async_context(self.avatar.tap(
                        request.app['avatar_http'], source_id, request['owner']))
                    await downstream.send_json({'type': 'gateway.session.created',
                                                'source_session_id': source_id, 'avatar_available': not tap.failed})
                elif request.query.get('avatar') == '1':
                    await downstream.send_json({'type': 'gateway.session.created',
                                                'source_session_id': None, 'avatar_available': False})
                await self.relay(upstream, downstream, tap)
                return downstream
        except Busy:
            return failure(429, "gateway_busy", "The realtime lane is at capacity.", headers={"Retry-After": "1"})
        except (TimeoutError, ClientError, OSError):
            if downstream.prepared:
                await downstream.close(code=1011, message=b"Realtime upstream disconnected")
                return downstream
            return failure(502, "engine_unavailable", "The realtime engine could not open a session.")

    def alias_realtime_update(self, raw):
        if '"session.update"' in raw and MODEL_ALIAS in raw:
            try:
                event = json.loads(raw)
                if event.get("type") == "session.update" and event["session"].get("model") == MODEL_ALIAS:
                    event["session"]["model"] = self.config.model
                    return json.dumps(event, ensure_ascii=False)
            except (ValueError, AttributeError, KeyError):
                pass
        return raw

    async def relay(self, upstream, downstream, tap=None):
        loop = asyncio.get_running_loop()
        last_activity = loop.time()

        async def pump(source, target):
            nonlocal last_activity
            async for frame in source:
                if frame.type == WSMsgType.TEXT:
                    last_activity = loop.time()
                    if source is upstream and '"session.created"' in frame.data:
                        try:
                            event = json.loads(frame.data)
                            if event.get("type") == "session.created":
                                capabilities = event["session"].get("capabilities")
                                if isinstance(capabilities, dict):
                                    self.capabilities = capabilities
                        except (ValueError, AttributeError, KeyError):
                            pass
                    text = self.alias_realtime_update(frame.data) if source is downstream else frame.data
                    if tap:
                        tap.publish(text, 'downstream' if source is downstream else 'upstream')
                    await target.send_str(text)
                elif frame.type == WSMsgType.BINARY:
                    last_activity = loop.time()
                    await target.send_bytes(frame.data)
                elif frame.type == WSMsgType.ERROR:
                    return 1011, b"Realtime transport failed"
            code = source.close_code
            return code if code in {1000, 1001, 1002, 1003, 1007, 1008, 1009, 1011, 1012, 1013} else 1001, b""

        async def idle():
            while (remaining := self.config.realtime_idle_timeout - (loop.time() - last_activity)) > 0:
                await asyncio.sleep(remaining)
            return 1001, b"Realtime session idle timeout"

        tasks = [asyncio.create_task(coro) for coro in (pump(downstream, upstream), pump(upstream, downstream), idle())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            code, reason = next(task for task in tasks if task in done).result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(downstream.close(code=code, message=reason), upstream.close(code=code, message=reason))


def create_app(config=None, text_config=None, *, image_factory=None):
    if config is None:
        config = Config.from_env()
        if os.environ.get("LLM_ENGINE_URL"):
            text_config = config.text_backend()
        if os.environ.get('OMNI_IMAGE_WORKER_URL'):
            placement = os.environ.get('OMNI_IMAGE_PLACEMENT', 'shared')
            if placement not in ('shared', 'dedicated'):
                raise ValueError('Unknown image worker placement')
            if placement == 'shared' and text_config is None:
                raise ValueError('Shared image placement requires a text engine')
            image_factory = DedicatedImageJobs.from_env if placement == 'dedicated' else ImageJobs.from_env
    gateway = Gateway(config)
    app = web.Application(client_max_size=MAX_BODY, middlewares=[gateway.authenticate])
    app['gateway'] = gateway
    app.cleanup_ctx.append(gateway.session_context)

    async def livez(request):
        return web.json_response({"state": "running"})

    app.add_routes([
        web.get("/livez", livez), web.get("/healthz", gateway.health),
        web.get("/api/mode", gateway.mode), web.put("/api/mode", gateway.mode),
        web.get("/v1/models", gateway.aliased_models), web.post("/v1/chat/completions", gateway.chat),
        web.post("/v1/chat/prompt_tokens", gateway.prompt_tokens),
    ])
    if text_config is not None:
        text_app = create_app(text_config)
        app.add_subapp('/llm', text_app)
    if image_factory:
        images = image_factory(text_app['gateway'] if text_config is not None else None)
        app.cleanup_ctx.append(images.lifecycle)
        app.add_routes([
            web.post('/v1/images/jobs', images.submit),
            web.get('/v1/images/jobs/{id}', images.detail),
            web.get('/v1/images/jobs/{id}/result', images.detail),
            web.post('/v1/images/jobs/{id}/cancel', images.detail),
        ])
    if not config.realtime_enabled:
        app.add_routes([
            web.post('/v1/responses', gateway.responses_request),
            web.get('/v1/responses/{response_id}', gateway.responses_request),
            web.post('/v1/responses/{response_id}/cancel', gateway.responses_request),
        ])
        return app
    app.add_routes([
        web.get("/v1/realtime", gateway.realtime),
        web.post('/v1/browser-tickets', gateway.browser_ticket),
        web.route('*', '/v1/avatar/{tail:.*}', gateway.avatar.handle),
    ])
    static_root = Path(__file__).with_name('web')
    async def static(request):
        name = request.path.lstrip('/') or 'index.html'
        return web.FileResponse(static_root / name, headers={
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Security-Policy': "default-src 'self'; connect-src 'self'; media-src 'self' blob:; "
                                       "script-src 'self'; style-src 'self'; img-src 'self' data:",
            'Referrer-Policy': 'no-referrer',
        })
    for path in ('/', '/app.js', '/style.css', '/mic-worklet.js'):
        app.router.add_get(path, static)
    return app


def main():
    web.run_app(create_app(), host="0.0.0.0", port=8000, handler_cancellation=True, access_log=None)


if __name__ == "__main__":
    main()
