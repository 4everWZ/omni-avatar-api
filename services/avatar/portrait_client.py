"""Private neural-worker transport implementing the renderer lifecycle."""
import asyncio
import io
import logging
import struct
from collections import deque
from time import monotonic

from aiohttp import ClientError, WSMsgType, web
from PIL import Image

from .sessions import PLAYBACK_HISTORY_SECONDS


class PortraitClient:
    def __init__(self, http, token):
        self.http, self.token = http, token

    def headers(self, owner):
        return {'Authorization': 'Bearer ' + self.token, 'X-Omni-Owner': owner}

    async def forward(self, request):
        try:
            async with self.http.request(request.method, request.path,
                    headers=self.headers(request['owner']) | {'Content-Type': request.content_type},
                    data=await request.read(), allow_redirects=False) as response:
                return web.Response(status=response.status, body=await response.read(),
                                    headers={'Content-Type': response.headers.get('Content-Type', 'application/json'),
                                             'Cache-Control': 'no-store'})
        except (ClientError, TimeoutError):
            return web.json_response({'error': {'code': 'portrait_unavailable'}}, status=503)

    async def renderer(self, owner, avatar_id):
        headers = self.headers(owner)
        url = '/v1/avatar/avatars/' + avatar_id
        try:
            async with self.http.get(url + '/preview', headers=headers, allow_redirects=False) as response:
                if response.status == 404:
                    raise web.HTTPNotFound()
                if response.status != 200:
                    raise web.HTTPServiceUnavailable(reason='Portrait preview unavailable')
                with Image.open(io.BytesIO(await response.read())) as image:
                    preview = image.convert('RGB')
            socket = await self.http.ws_connect('/internal/render/' + avatar_id,
                                                headers=headers, heartbeat=20, max_msg_size=1024 * 1024)
            try:
                hello = await socket.receive_json(timeout=30)
                if hello.get('type') != 'ready':
                    raise ValueError('Portrait handshake failed')
            except BaseException:
                await socket.close()
                raise
            return PortraitRenderer(socket, preview)
        except (ClientError, TimeoutError) as error:
            if getattr(error, 'status', None) == 429:
                raise web.HTTPTooManyRequests(reason='Neural renderer busy', headers={'Retry-After': '1'}) from error
            raise web.HTTPServiceUnavailable(reason='Portrait worker unavailable') from error


class PortraitRenderer:
    capabilities = {'video': '640x720', 'fps': 25, 'alpha': False, 'driver': 'ditto', 'online': True, 'input_lead_ms': 1200}

    def __init__(self, socket, preview):
        self.socket, self.preview = socket, preview
        # WHEP supplies its timeline origin before ICE connects. Keep enough
        # compressed history for the first media read after negotiation.
        self.frames = deque(maxlen=25 * PLAYBACK_HISTORY_SECONDS)
        self.audio = asyncio.Queue(maxsize=128)
        self.generation = 0
        self.failed = False
        self.closed = False
        self.latest = preview
        self.latest_time = 0
        self.generated_frames = 0
        self.last_generated = 0
        self.created = monotonic()
        self.reset_deadline = self.created + 2
        self.received = self.created
        self.sender = asyncio.create_task(self.send())
        self.reader = asyncio.create_task(self.receive())

    async def send(self):
        try:
            while True:
                value = await self.audio.get()
                if isinstance(value, bytes):
                    await self.socket.send_bytes(value)
                else:
                    await self.socket.send_json(value)
        except (ClientError, OSError, RuntimeError):
            self.failed = True

    async def receive(self):
        try:
            async for frame in self.socket:
                if frame.type == WSMsgType.BINARY:
                    timestamp, generation = struct.unpack('!dI', frame.data[:12])
                    if generation == self.generation:
                        self.frames.append((timestamp, frame.data[12:]))
                        self.generated_frames += 1
                        self.last_generated = timestamp
                        self.received = monotonic()
        except (ClientError, ValueError, OSError, struct.error):
            logging.exception('Portrait frame transport failed')
        finally:
            if not self.closed:
                self.failed = True

    def enqueue(self, value):
        if self.failed or self.closed:
            raise ValueError('Portrait renderer unavailable')
        try:
            self.audio.put_nowait(value)
        except asyncio.QueueFull as error:
            self.failed = True
            raise ValueError('Portrait input overflow') from error

    def push_audio(self, pcm, sample_rate, start_sample, generation):
        self.enqueue(struct.pack('!QI', start_sample, generation) + pcm)

    def render(self, timestamp, rms):
        if self.failed or monotonic() - self.received > 2:
            raise RuntimeError('Portrait output stalled')
        # Peers negotiate independently and can request an earlier session pose.
        selected = next((frame for frame in reversed(self.frames) if frame[0] <= timestamp), None)
        if selected:
            selected_time, encoded = selected
            if self.latest is self.preview or selected_time != self.latest_time:
                self.latest_time = selected_time
                with Image.open(io.BytesIO(encoded)) as image:
                    self.latest = image.convert('RGB')
        elif timestamp < self.latest_time:
            return self.preview
        if monotonic() > self.reset_deadline and timestamp - self.latest_time > .75:
            raise RuntimeError('Portrait output behind playback')
        return self.latest

    def status(self):
        return {'generated_frames': self.generated_frames,
                'last_generated_ms': round(self.last_generated * 1000),
                'displayed_frame_ms': round(self.latest_time * 1000)}

    def clear(self, generation):
        self.generation = generation
        self.frames.clear()
        self.latest = self.preview
        # Allow reset/setup to recover while old audio/video is fenced out.
        self.latest_time = monotonic() - self.created
        self.reset_deadline = monotonic() + 1.5
        while not self.audio.empty():
            self.audio.get_nowait()
        self.enqueue({'type': 'clear', 'generation': generation})

    def close(self):
        if self.closed:
            return
        self.closed = True
        for task in (self.sender, self.reader):
            task.cancel()
        async def release():
            await asyncio.gather(self.sender, self.reader, return_exceptions=True)
            await self.socket.close()
        return asyncio.create_task(release())
