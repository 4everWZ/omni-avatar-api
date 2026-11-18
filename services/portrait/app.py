"""Private reference-image preparation and timestamped neural rendering API."""
import asyncio
import base64
import binascii
import io
import json
import logging
import os
import re
import secrets
import struct
from contextlib import asynccontextmanager
from pathlib import Path

from aiohttp import WSMsgType, web
from PIL import Image, ImageOps, UnidentifiedImageError

ID = r'[a-f0-9]{32}'
MAX_IMAGE = 4 * 1024 * 1024


class PortraitService:
    def __init__(self, root, token, backends, gpu_gate=None):
        if not token:
            raise ValueError('Internal token required')
        self.root, self.token = Path(root), token
        self.root.mkdir(parents=True, exist_ok=True)
        self.slots = list(backends)
        self.slot_returned = asyncio.Event()
        self.gpu_gate = gpu_gate
        self.tasks, self.active = {}, {}
        self.sockets = set()
        # Interrupted jobs cannot be advertised as ready after restart.
        for path in self.root.glob('*/jobs/*.json'):
            job = json.loads(path.read_text())
            if job['state'] in ('queued', 'preparing'):
                job.update(state='failed', error='worker_restarted')
                self.save(path, job)

    @staticmethod
    def save(path, body):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(body), encoding='utf-8')
        temporary.replace(path)

    @web.middleware
    async def auth(self, request, handler):
        if request.path == '/livez':
            return await handler(request)
        if not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + self.token):
            raise web.HTTPUnauthorized()
        owner = request.headers.get('X-Omni-Owner', '')
        if not re.fullmatch('[a-f0-9]{64}', owner):
            raise web.HTTPForbidden()
        request['owner'] = owner
        try:
            return await handler(request)
        except (ValueError, TypeError, KeyError, binascii.Error):
            return web.json_response({'error': {'code': 'invalid_portrait_request'}}, status=400)

    def directory(self, owner, kind):
        path = self.root / owner / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def job(self, owner, identifier):
        if not re.fullmatch(ID, identifier):
            raise web.HTTPNotFound()
        path = self.directory(owner, 'jobs') / (identifier + '.json')
        if not path.exists():
            raise web.HTTPNotFound()
        return path, json.loads(path.read_text())

    async def asset(self, request):
        owner = request['owner']
        assets = self.directory(owner, 'assets')
        if request.method == 'DELETE':
            identifier = request.match_info['id']
            if not re.fullmatch(ID, identifier) or not (path := assets / (identifier + '.png')).exists():
                raise web.HTTPNotFound()
            for meta in self.directory(owner, 'jobs').glob('*.json'):
                if json.loads(meta.read_text())['reference_asset_id'] == identifier:
                    raise web.HTTPConflict(reason='Delete dependent generation first')
            path.unlink()
            return web.Response(status=204)
        if len(list(assets.glob('*.png'))) >= 16 or sum(1 for _ in self.root.glob('*/assets/*.png')) >= 64:
            raise web.HTTPTooManyRequests(reason='Reference image capacity reached')
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('Expected object')
        if body.get('mime_type') not in ('image/png', 'image/jpeg'):
            raise ValueError('Expected PNG or JPEG')
        encoded = body['image_base64']
        if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE + 2) // 3 * 4:
            raise ValueError('Image too large')
        data = base64.b64decode(encoded, validate=True)
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ('PNG', 'JPEG') or image.width * image.height > 8_000_000:
                    raise ValueError('Unsupported reference image')
                image.load()
                image = ImageOps.exif_transpose(image).convert('RGB')
                image = ImageOps.pad(image, (640, 720), color='#ece7df')
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ValueError('Invalid image') from error
        identifier = secrets.token_hex(16)
        image.save(assets / (identifier + '.png'))
        return web.json_response({'id': identifier, 'width': 640, 'height': 720}, status=201)

    async def generate(self, request):
        owner = request['owner']
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('Expected object')
        identifier = body['reference_asset_id']
        if body.get('driver', 'ditto') != 'ditto' or not re.fullmatch(ID, identifier):
            raise ValueError('Unknown driver or reference')
        image = self.directory(owner, 'assets') / (identifier + '.png')
        if not image.exists():
            raise web.HTTPNotFound()
        jobs = self.directory(owner, 'jobs')
        if len(list(jobs.glob('*.json'))) >= 16 or sum(1 for _ in self.root.glob('*/jobs/*.json')) >= 64:
            raise web.HTTPTooManyRequests(reason='Generation capacity reached')
        if sum(not task.done() for task in self.tasks.values()) >= 4:
            raise web.HTTPTooManyRequests(reason='Preparation queue full')
        job_id = secrets.token_hex(16)
        path = jobs / (job_id + '.json')
        job = {'id': job_id, 'state': 'queued', 'driver': 'ditto',
               'reference_asset_id': identifier, 'avatar_id': None, 'error': None}
        self.save(path, job)
        self.tasks[job_id] = asyncio.create_task(self.prepare(path, job, image))
        self.tasks[job_id].portrait_job = job
        return web.json_response(job, status=202)

    async def prepare(self, path, job, image):
        prepared = path.with_suffix('.pkl')
        try:
            async with self.acquire(wait=True) as backend:
                if job['state'] == 'cancelled':
                    return
                job['state'] = 'preparing'
                self.save(path, job)
                # Cancelling an asyncio task cannot cancel CUDA/to_thread. The
                # lease stays held until preparation actually leaves the GPU.
                await asyncio.to_thread(backend.prepare, image, prepared)
                if job['state'] != 'cancelled':
                    job.update(state='ready', avatar_id='portrait_' + job['id'])
                    self.save(path, job)
        except Exception:
            logging.exception('Portrait preparation failed')
            if job['state'] != 'cancelled':
                job.update(state='failed', error='reference_preparation_failed')
                self.save(path, job)
        finally:
            if job['state'] != 'ready':
                prepared.unlink(missing_ok=True)
            self.tasks.pop(job['id'], None)

    async def detail(self, request):
        path, job = self.job(request['owner'], request.match_info['id'])
        if request.method == 'DELETE':
            if job['id'] in self.active:
                raise web.HTTPConflict(reason='Avatar has an active session')
            # Keep cancellation visible while work drains; next DELETE removes it.
            task = self.tasks.get(job['id'])
            if task:
                # Share the running task's state object, without cancelling its GPU work.
                task_job = getattr(task, 'portrait_job', None)
                if task_job is not None:
                    task_job['state'] = 'cancelled'
                job['state'] = 'cancelled'
                self.save(path, job)
            else:
                path.with_suffix('.pkl').unlink(missing_ok=True)
                path.unlink()
                return web.Response(status=204)
        return web.json_response(job)

    def avatar(self, request):
        identifier = request.match_info['id']
        if not identifier.startswith('portrait_'):
            raise web.HTTPNotFound()
        path, job = self.job(request['owner'], identifier.removeprefix('portrait_'))
        if job['state'] != 'ready' or not path.with_suffix('.pkl').exists():
            raise web.HTTPNotFound()
        return path, job

    async def describe(self, request):
        _, job = self.avatar(request)
        if request.path.endswith('/preview'):
            return web.FileResponse(self.directory(request['owner'], 'assets') / (job['reference_asset_id'] + '.png'))
        return web.json_response({'id': job['avatar_id'], 'driver': 'ditto', 'state': 'ready',
                                  'video': '640x720', 'fps': 25, 'input_lead_ms': 1200})

    @asynccontextmanager
    async def acquire(self, *, wait=False):
        while True:
            guard = reserve_gpu(self.gpu_gate) if self.slots and self.gpu_gate else None
            if self.slots and (not self.gpu_gate or guard is not None):
                backend = self.slots.pop()
                break
            if not wait:
                raise web.HTTPTooManyRequests(reason='Neural renderer busy', headers={'Retry-After': '1'})
            try:
                await asyncio.wait_for(self.slot_returned.wait(), .25)
            except TimeoutError:
                pass
        try:
            yield backend
        finally:
            if guard is not None:
                guard.close()
            self.slots.append(backend)
            self.slot_returned.set()
            self.slot_returned.clear()

    async def render(self, request):
        path, job = self.avatar(request)
        async with self.acquire() as backend:
            identifier = job['id']
            self.active[identifier] = self.active.get(identifier, 0) + 1
            socket = web.WebSocketResponse(heartbeat=20, max_msg_size=1024 * 1024)
            loop = asyncio.get_running_loop()
            frames = asyncio.Queue(maxsize=64)
            overflow = asyncio.Event()
            def offer(value):
                if frames.full():
                    overflow.set()
                else:
                    frames.put_nowait(value)
            def emit(timestamp, generation, jpeg):
                loop.call_soon_threadsafe(offer, struct.pack('!dI', timestamp, generation) + jpeg)
            run = None
            try:
                run = await asyncio.to_thread(backend.start, path.with_suffix('.pkl'), emit)
                await socket.prepare(request)
                self.sockets.add(socket)
                await socket.send_json({'type': 'ready', 'input_lead_ms': 1200})
                async def send():
                    while not socket.closed:
                        if run.failure or overflow.is_set():
                            await socket.close(code=1011, message=b'Portrait rendering failed')
                            return
                        try:
                            value = await asyncio.wait_for(frames.get(), .5)
                            await socket.send_bytes(value)
                        except TimeoutError:
                            continue
                sender = asyncio.create_task(send())
                try:
                    async for message in socket:
                        if message.type == WSMsgType.BINARY:
                            if len(message.data) < 14 or (len(message.data) - 12) % 2:
                                raise ValueError('Malformed PCM packet')
                            start, generation = struct.unpack('!QI', message.data[:12])
                            run.append(message.data[12:], start, generation)
                        elif message.type == WSMsgType.TEXT:
                            event = json.loads(message.data)
                            if event['type'] != 'clear' or type(event['generation']) is not int:
                                raise ValueError('Invalid render control')
                            run.clear(event['generation'])
                finally:
                    sender.cancel()
                    await asyncio.gather(sender, return_exceptions=True)
            finally:
                self.sockets.discard(socket)
                if run:
                    try:
                        await asyncio.shield(asyncio.to_thread(run.shutdown))
                    except Exception:
                        # Do not release a potentially still-running GPU lease.
                        os._exit(1)
                self.active[identifier] -= 1
                if not self.active[identifier]:
                    del self.active[identifier]
            return socket

    async def cleanup(self, app):
        # Handler cancellation is disabled so a render lease drains before exit.
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks.values()), return_exceptions=True)

    async def shutdown(self, app):
        await asyncio.gather(*(socket.close(code=1001) for socket in tuple(self.sockets)),
                             return_exceptions=True)


def reserve_gpu(root):
    import fcntl
    guard = (Path(root) / 'portrait.lock').open('a')
    try:
        fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if not (Path(root) / 'image-active').exists():
            return guard
    except BlockingIOError:
        pass
    guard.close()
    return None


def create_app(root, token, backend, *, additional_backends=(), gpu_gate=None):
    service = PortraitService(root, token, (backend, *additional_backends), gpu_gate)
    app = web.Application(client_max_size=6 * 1024 * 1024, middlewares=[service.auth])
    app['service'] = service
    app.on_cleanup.append(service.cleanup)
    app.on_shutdown.append(service.shutdown)
    async def livez(request):
        return web.json_response({'state': 'running'})
    app.add_routes([
        web.get('/livez', livez),
        web.post('/v1/avatar/assets', service.asset),
        web.delete('/v1/avatar/assets/{id}', service.asset),
        web.post('/v1/avatar/generations', service.generate),
        web.get('/v1/avatar/generations/{id}', service.detail),
        web.delete('/v1/avatar/generations/{id}', service.detail),
        web.get('/v1/avatar/avatars/{id}', service.describe),
        web.get('/v1/avatar/avatars/{id}/preview', service.describe),
        web.get('/internal/render/{id}', service.render),
    ])
    return app


def main():
    from .backend import DittoBackend
    capacity = int(os.environ.get('OMNI_PORTRAIT_CAP', '1'))
    if capacity not in (1, 2):
        raise ValueError('OMNI_PORTRAIT_CAP must be 1 or 2')
    gpu_gate = os.environ.get('OMNI_GPU_SWITCH_ROOT')
    guard = reserve_gpu(gpu_gate) if gpu_gate else None
    if gpu_gate and guard is None:
        raise RuntimeError('Waiting for image GPU release before portrait startup')
    try:
        backends = [DittoBackend(os.environ.get('PORTRAIT_MODELS', '/portrait'), slot)
                    for slot in range(capacity)]
    finally:
        if guard:
            guard.close()
    web.run_app(create_app(os.environ.get('PORTRAIT_DATA', '/data'),
                          os.environ['OMNI_AVATAR_INTERNAL_TOKEN'], backends[0],
                          additional_backends=backends[1:], gpu_gate=gpu_gate),
                path=os.environ.get('PORTRAIT_SOCKET', '/run/omni-portrait/control.sock'),
                access_log=None, handler_cancellation=False)


if __name__ == "__main__":
    main()
