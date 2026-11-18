"""Private single-task image engine. GPU ownership is asserted by the coordinator."""
import asyncio
import logging
from pathlib import Path
import re
import secrets
from threading import Event

from aiohttp import web
from .model import Cancelled


def create_app(backend, token, switch_root):
    if not token:
        raise ValueError('Image worker token required')
    marker = Path(switch_root) / 'image-active'
    current = None
    job = {}

    @web.middleware
    async def authenticate(request, handler):
        if request.path != '/livez' and not secrets.compare_digest(
                request.headers.get('Authorization', '').encode(), f'Bearer {token}'.encode()):
            raise web.HTTPUnauthorized()
        return await handler(request)

    async def run(call, value, *, next_state, result_key):
        try:
            output = await asyncio.to_thread(call, value, job['cancel'])
            if not job['cancel'].is_set():
                job.update(state=next_state)
                job[result_key] = output
            else:
                job.update(state='cancelled')
        except Cancelled:
            job.update(state='cancelled')
        except Exception:
            logging.exception('Image worker failed')
            job.update(state='failed', error='image_inference_failed')

    async def livez(request):
        return web.json_response({'state': 'running'})

    async def state(request):
        return web.json_response({'busy': bool(current and not current.done()),
                                  'id': job.get('id'), 'free_bytes': backend.free_memory()})

    async def prepare(request):
        nonlocal current
        identifier = request.match_info['id']
        payload = await request.json()
        if not re.fullmatch('[a-f0-9]{32}', identifier):
            raise web.HTTPBadRequest()
        if job.get('id') == identifier:
            return web.json_response({'state': job['state']}, status=202)
        if current and not current.done():
            raise web.HTTPConflict(reason='Image worker busy')
        job.clear()
        job.update(id=identifier, state='preparing', cancel=Event())
        current = asyncio.create_task(run(backend.prepare, payload, next_state='prepared',
                                          result_key='prepared'))
        return web.json_response({'state': 'preparing'}, status=202)

    async def render(request):
        nonlocal current
        identifier = request.match_info['id']
        try:
            allowed = marker.read_text(encoding='ascii') == identifier
        except FileNotFoundError:
            allowed = False
        if not allowed:
            raise web.HTTPConflict(reason='GPU lease is unavailable')
        if job.get('id') != identifier:
            raise web.HTTPNotFound()
        if job['state'] == 'prepared':
            job['state'] = 'running'
            current = asyncio.create_task(run(backend.generate, job.pop('prepared'),
                                              next_state='completed', result_key='output'))
        elif job['state'] == 'preparing':
            raise web.HTTPConflict(reason='Image inputs still preparing')
        return web.json_response({'state': job['state']}, status=202)

    async def detail(request):
        if request.match_info['id'] != job.get('id'):
            raise web.HTTPNotFound()
        if request.method == 'POST':
            job['cancel'].set()
            if job['state'] == 'prepared':
                job.pop('prepared', None)
                job['state'] = 'cancelled'
        return web.json_response({k: v for k, v in job.items() if k not in ('cancel', 'prepared')})

    async def close(app):
        if current and not current.done():
            job['cancel'].set()
            await current

    app = web.Application(client_max_size=6 * 1024 * 1024, middlewares=[authenticate])
    app.on_cleanup.append(close)
    app.add_routes([web.get('/livez', livez), web.get('/state', state),
                    web.put('/prepare/{id}', prepare),
                    web.put('/render/{id}', render), web.get('/render/{id}', detail),
                    web.post('/render/{id}/cancel', detail)])
    return app
