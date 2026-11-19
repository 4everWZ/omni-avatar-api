"""Responses transport with caller-bound IDs and shared backend admission."""
import asyncio
from contextlib import AsyncExitStack
import hashlib
import hmac
import json
import re
import secrets

from aiohttp import ClientError, web
from .admission import BudgetExceeded
from .token_budget import ReservationRejected, measure, reservation
from .thinking import apply_thinking_default


def inline_image(url):
    return isinstance(url, str) and url.startswith((
        'data:image/png;base64,', 'data:image/jpeg;base64,', 'data:image/webp;base64,'))


def error(status, code, message):
    return web.json_response({'error': {'code': code, 'message': message}}, status=status)


class ResponsesProxy:
    def __init__(self, gateway):
        self.gateway = gateway
        self.monitors = set()

    def signature(self, owner, nonce):
        data = ('responses-owner-v1:' + owner + ':' + nonce).encode()
        return hmac.new(self.gateway.config.admin_token.encode(), data, hashlib.sha256).hexdigest()[:32]

    def owns(self, owner, response_id):
        match = re.fullmatch(r'resp_([0-9a-f]{32})_([0-9a-f]{32})', response_id or '')
        return bool(match and secrets.compare_digest(match[2], self.signature(owner, match[1])))

    async def close(self):
        for task in tuple(self.monitors):
            task.cancel()
        await asyncio.gather(*tuple(self.monitors), return_exceptions=True)

    def track_background(self, response_id, lease):
        task = asyncio.create_task(self.monitor(response_id, lease))
        self.monitors.add(task)
        task.add_done_callback(self.monitors.discard)

    async def monitor(self, response_id, lease):
        terminal = False
        try:
            async with asyncio.timeout(self.gateway.config.request_timeout):
                while True:
                    async with self.gateway.engine.get('/v1/responses/' + response_id, allow_redirects=False) as response:
                        if response.status != 200:
                            return
                        data = await response.json()
                    if data.get('status') in {'completed', 'cancelled', 'failed', 'incomplete'}:
                        terminal = True
                        return
                    await asyncio.sleep(.25)
        except (ClientError, TimeoutError, ValueError):
            pass
        finally:
            if not terminal:
                try:
                    async with asyncio.timeout(3):
                        async with self.gateway.engine.post('/v1/responses/' + response_id + '/cancel',
                                                            allow_redirects=False) as response:
                            await response.read()
                except (ClientError, TimeoutError):
                    pass
            await lease.aclose()

    async def creation(self, request):
        owner = request['owner']
        try:
            async with asyncio.timeout(self.gateway.config.request_timeout):
                payload = await request.json()
        except (ValueError, UnicodeError, TimeoutError):
            return None, error(400, 'invalid_json', 'Expected a Responses request.')
        if not isinstance(payload, dict) or payload.get('model') != self.gateway.config.model_alias:
            return None, error(400, 'model_not_found', 'Use the llm-default alias.')
        previous = payload.get('previous_response_id')
        if previous is not None and (not isinstance(previous, str) or not self.owns(owner, previous)):
            return None, error(404, 'response_not_found', 'Response not found.')
        tools = payload.get('tools') or []
        if (not isinstance(tools, list) or any(not isinstance(t, dict) or t.get('type') != 'function' for t in tools)):
            return None, error(400, 'unsupported_tool', 'Only function-call output is supported; tools are executed by the caller.')
        items = payload.get('input', [])
        if payload.get('conversation') is not None or (isinstance(items, list) and any(
                isinstance(item, dict) and item.get('type') == 'item_reference' for item in items)):
            return None, error(400, 'unsupported_reference', 'Use inline input or an owned previous_response_id.')
        if isinstance(items, list) and any(
                isinstance(part, dict) and part.get('type') == 'input_image'
                and (part.get('file_id') is not None or not inline_image(part.get('image_url')))
                for item in items if isinstance(item, dict) and isinstance(item.get('content'), list)
                for part in item['content']):
            return None, error(400, 'invalid_image_source', 'Use inline PNG, JPEG, or WebP image data.')
        nonce = secrets.token_hex(16)
        payload['request_id'] = 'resp_' + nonce + '_' + self.signature(owner, nonce)
        payload['model'] = self.gateway.config.model
        apply_thinking_default(payload)
        return payload, None

    async def prompt_tokens(self, request):
        payload, rejection = await self.creation(request)
        if rejection is not None:
            return rejection
        try:
            async with asyncio.timeout(self.gateway.config.request_timeout):
                counted = await measure(self.gateway, 'responses', payload)
            count = counted['input_tokens']
            if type(count) is not int or count < 1:
                raise ValueError('Invalid engine input token count')
            return web.json_response({'count': count})
        except ReservationRejected as exc:
            return exc.response()
        except (TimeoutError, ClientError, KeyError, TypeError, ValueError):
            return error(504, 'tokenizer_timeout', 'Responses token counting did not finish within its deadline.')

    async def handle(self, request):
        owner = request['owner']
        payload = None
        path = '/v1/responses'
        creating = request.method == 'POST' and 'response_id' not in request.match_info
        if creating:
            payload, rejection = await self.creation(request)
            if rejection is not None:
                return rejection
        else:
            response_id = request.match_info['response_id']
            if not self.owns(owner, response_id):
                return error(404, 'response_not_found', 'Response not found.')
            path += '/' + response_id
            if request.method == 'POST':
                path += '/cancel'

        outgoing = None
        lease = AsyncExitStack()
        try:
            if creating:
                async with asyncio.timeout(self.gateway.config.request_timeout):
                    budget = await reservation(self.gateway, 'responses', payload)
                await lease.enter_async_context(self.gateway.admission.hold('chat', **budget))
            async with asyncio.timeout(self.gateway.config.request_timeout):
                async with self.gateway.engine.request(request.method, path,
                        json=payload if creating else None,
                        params={k: v for k, v in request.query.items() if k in {'stream', 'starting_after'}},
                        allow_redirects=False) as upstream:
                    content_type = upstream.headers.get('Content-Type', 'application/json')
                    if upstream.content_type == 'text/event-stream':
                        if creating and upstream.status == 200 and payload.get('background'):
                            self.track_background(payload['request_id'], lease)
                            lease = None
                        outgoing = web.StreamResponse(status=upstream.status, headers={
                            'Content-Type': content_type, 'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
                        await outgoing.prepare(request)
                        async for chunk in upstream.content.iter_any():
                            await outgoing.write(chunk)
                        await outgoing.write_eof()
                        return outgoing
                    body = await upstream.read()
                    if creating and upstream.status == 200 and payload.get('background'):
                        result = json.loads(body)
                        if result.get('id') != payload['request_id']:
                            return error(502, 'response_identity_mismatch', 'Backend returned an unexpected response identity.')
                        self.track_background(payload['request_id'], lease)
                        lease = None
                    return web.Response(status=upstream.status, body=body, headers={'Content-Type': content_type})
        except ReservationRejected as exc:
            return exc.response()
        except BudgetExceeded as exc:
            return error(400, 'request_budget_exceeded', str(exc))
        except (TimeoutError, ClientError, ValueError):
            if outgoing is not None and outgoing.prepared:
                if request.transport:
                    request.transport.close()
                return outgoing
            return error(504, 'response_upstream_unavailable', 'Responses inference did not finish within its deadline.')
        finally:
            if lease is not None:
                await lease.aclose()
