"""Private Avatar transport and a nonblocking tap of the existing duplex lane."""
import asyncio
import json
import secrets

from aiohttp import ClientError, WSMsgType, web


class SourceTap:
    def __init__(self, client, url, headers):
        self.client, self.url, self.headers = client, url, headers
        self.queue = asyncio.Queue(maxsize=32)
        self.socket = self.task = None
        self.failed = False

    async def __aenter__(self):
        try:
            async with asyncio.timeout(3):
                self.socket = await self.client.ws_connect(self.url, headers=self.headers, heartbeat=20)
                await self.socket.receive_json()
            self.task = asyncio.create_task(self.send())
        except (ClientError, TimeoutError, ValueError):
            self.failed = True
        return self

    def publish(self, raw, direction):
        if self.failed:
            return
        try:
            event = json.loads(raw)
            if not isinstance(event, dict) or event.get('type') not in (
                    'response.audio.delta', 'response.output_audio.delta', 'response.audio_transcript.delta',
                    'response.output_audio_transcript.delta', 'response.text.delta', 'response.output_text.delta',
                    'response.cancel', 'output_audio_buffer.clear', 'output_audio_buffer.cleared', 'barge_in'):
                return
            self.queue.put_nowait({'event': event, 'direction': direction})
        except ValueError:
            return
        except asyncio.QueueFull:
            self.failed = True
            if self.task:
                self.task.cancel()

    async def send(self):
        try:
            while True:
                value = await self.queue.get()
                async with asyncio.timeout(3):
                    await self.socket.send_json(value)
        except (ClientError, TimeoutError, OSError):
            self.failed = True
        finally:
            if self.socket:
                await self.socket.close()

    async def __aexit__(self, *args):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if self.socket:
            await self.socket.close()


class AvatarProxy:
    def __init__(self, base_url, token):
        self.base_url, self.token = base_url.rstrip('/'), token

    def headers(self, owner):
        return {'Authorization': f'Bearer {self.token}', 'X-Omni-Owner': owner}

    def tap(self, client, source_id, owner):
        return SourceTap(client, f'{self.base_url}/internal/sources/{source_id}', self.headers(owner))

    async def handle(self, request):
        if not self.base_url or not self.token:
            return web.json_response({'error': {'code': 'avatar_unavailable'}}, status=503)
        client = request.app['avatar_http']
        headers = self.headers(request['owner'])
        if 'Content-Type' in request.headers:
            headers['Content-Type'] = request.headers['Content-Type']
        url = self.base_url + request.path
        downstream = web.WebSocketResponse(heartbeat=20, max_msg_size=16384)
        try:
            if downstream.can_prepare(request).ok:
                async with client.ws_connect(url, headers=headers, heartbeat=20) as upstream:
                    await downstream.prepare(request)

                    async def pump(source, target):
                        async for frame in source:
                            if frame.type == WSMsgType.TEXT:
                                await target.send_str(frame.data)

                    tasks = [asyncio.create_task(pump(a, b)) for a, b in
                             ((upstream, downstream), (downstream, upstream))]
                    try:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
                        await downstream.close()
                    return downstream
            async with asyncio.timeout(20):
                async with client.request(request.method, url, headers=headers,
                                          data=await request.read(), allow_redirects=False) as upstream:
                    out = {key: upstream.headers[key] for key in ('Content-Type', 'Location', 'Cache-Control',
                                                                  'X-Avatar-Timeline-Origin-Ms')
                           if key in upstream.headers}
                    return web.Response(status=upstream.status, body=await upstream.read(), headers=out)
        except ClientError as error:
            status = getattr(error, 'status', 502)
            if downstream.prepared:
                await downstream.close(code=1011)
                return downstream
            return web.json_response({'error': {'code': 'avatar_unavailable'}}, status=status)
        except TimeoutError:
            return web.json_response({'error': {'code': 'avatar_timeout'}}, status=504)
