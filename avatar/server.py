"""Private Avatar service. All public requests enter through the gateway."""
import asyncio
import json
import os
import re
import secrets
from pathlib import Path
from time import monotonic

from aiohttp import WSMsgType, web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

from media import TimelineTrack
from renderers import DRIVERS
from sessions import Session

ROOT = Path(__file__).parent


class AvatarService:
    def __init__(self, token, registry=None, capacity=4, idle_timeout=180):
        if not token:
            raise ValueError('OMNI_AVATAR_INTERNAL_TOKEN is required')
        self.token = token
        if registry is None:
            registry_path = Path(os.environ.get('OMNI_AVATAR_REGISTRY', ROOT / 'avatars.json'))
            registry = json.loads(registry_path.read_text(encoding='utf-8'))
            registry = {name: item | {'root': registry_path.parent / item['root']}
                        for name, item in registry.items()}
        self.registry = registry
        self.capacity, self.idle_timeout = capacity, idle_timeout
        self.sessions, self.sources = {}, {}

    @web.middleware
    async def authenticate(self, request, handler):
        if request.path == '/livez':
            return await handler(request)
        if not secrets.compare_digest(request.headers.get('Authorization', ''), f'Bearer {self.token}'):
            raise web.HTTPUnauthorized()
        owner = request.headers.get('X-Omni-Owner', '')
        if not re.fullmatch('[a-f0-9]{64}', owner):
            raise web.HTTPForbidden(reason='Missing caller identity')
        request['owner'] = owner
        try:
            return await handler(request)
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response({'error': {'code': 'invalid_avatar_request', 'message': str(error)}}, status=400)

    def owned(self, request):
        item = self.sessions.get(request.match_info['id'])
        if not item or not secrets.compare_digest(item.owner, request['owner']):
            raise web.HTTPNotFound()
        item.touched = monotonic()
        return item

    async def create(self, request):
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError('Expected an object')
        avatar_id = body.get('avatar_id')
        if avatar_id not in self.registry:
            raise ValueError('Unknown avatar_id')
        source = body.get('source')
        if not isinstance(source, dict) or source.get('type') not in ('realtime', 'audio'):
            raise ValueError('source.type must be realtime or audio')
        source_id = source.get('session_id') if source['type'] == 'realtime' else None
        if source['type'] == 'realtime' and self.sources.get(source_id, {}).get('owner') != request['owner']:
            raise web.HTTPNotFound(reason='Realtime source not available for this caller')
        if sum(s.state != 'closed' for s in self.sessions.values()) >= self.capacity:
            raise web.HTTPTooManyRequests(reason='Avatar capacity reached')
        registered = self.registry[avatar_id]
        renderer = DRIVERS[registered['driver']](Path(registered['root']))
        source_state = self.sources.get(source_id, {})
        session = Session(request['owner'], avatar_id, source_id, renderer,
                          epoch=source_state.get('epoch', -1), response_id=source_state.get('response_id'))
        self.sessions[session.id] = session
        return web.json_response(session.view(), status=201)

    async def detail(self, request):
        session = self.owned(request)
        if request.method == 'DELETE':
            await session.close()
            return web.Response(status=204)
        return web.json_response(session.view())

    async def audio(self, request):
        session = self.owned(request)
        if session.state != 'active':
            raise web.HTTPConflict(reason='Session is not active')
        if session.source:
            raise web.HTTPConflict(reason='Realtime sessions consume only their bound source')
        event = await request.json()
        if not isinstance(event, dict):
            raise ValueError('Expected an audio event')
        if event.get('type') == 'clear':
            session.clear()
        else:
            if 'generation' not in event:
                raise ValueError('generation is required')
            session.append(event)
        return web.json_response({'generation': session.generation})

    async def events(self, request):
        session = self.owned(request)
        if session.state == 'closed':
            raise web.HTTPGone()
        socket = web.WebSocketResponse(heartbeat=20, max_msg_size=16384)
        await socket.prepare(request)
        queue = asyncio.Queue(maxsize=64)
        session.events.add(queue)
        await socket.send_json({'type': 'avatar.snapshot', **session.view()})

        async def send():
            while True:
                event = await queue.get()
                await socket.send_json(event)
                if event['type'] in ('avatar.closed', 'avatar.resync_required'):
                    await socket.close()
                    break

        task = asyncio.create_task(send())
        try:
            async for frame in socket:
                session.touched = monotonic()
                if frame.type == WSMsgType.TEXT:
                    try:
                        ack = json.loads(frame.data)
                        if (ack.get('type') != 'playback.ack' or type(ack.get('played_ms')) not in (int, float)
                                or not 0 <= ack['played_ms'] <= session.now_sample() * 1000 / 24000
                                or ack.get('generation') != session.generation):
                            raise ValueError('Invalid playback receipt')
                        session.last_ack = {'played_ms': ack['played_ms'], 'generation': session.generation,
                                            'source': 'consumer_report', 'received_at_ms': session.now_sample() // 24}
                    except (ValueError, TypeError, AttributeError):
                        await socket.send_json({'type': 'avatar.error', 'message': 'Invalid playback receipt'})
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            session.events.discard(queue)
        return socket

    async def source(self, request):
        source_id = request.match_info['id']
        if source_id in self.sources:
            raise web.HTTPConflict()
        socket = web.WebSocketResponse(heartbeat=20, max_msg_size=1024 * 1024)
        source_state = {'owner': request['owner'], 'socket': socket, 'epoch': 0, 'response_id': None}
        self.sources[source_id] = source_state
        await socket.prepare(request)
        await socket.send_json({'type': 'source.ready'})
        try:
            async for frame in socket:
                if frame.type != WSMsgType.TEXT:
                    continue
                envelope = json.loads(frame.data)
                event = envelope.get('event', {})
                kind = event.get('type')
                if kind not in (
                        'response.audio.delta', 'response.output_audio.delta', 'response.audio_transcript.delta',
                        'response.output_audio_transcript.delta', 'response.text.delta', 'response.output_text.delta',
                        'response.cancel', 'output_audio_buffer.clear', 'output_audio_buffer.cleared', 'barge_in',
                        'session.created', 'session.updated', 'session.resumed', 'response.created', 'response.speak'):
                    continue
                epoch = event.get('epoch', (event.get('metadata') or {}).get('epoch',
                                  (event.get('session') or {}).get('epoch')))
                if type(epoch) is int:
                    source_state['epoch'] = max(source_state['epoch'], epoch)
                if envelope.get('direction') == 'downstream' and kind in ('response.cancel', 'output_audio_buffer.clear', 'barge_in'):
                    source_state['epoch'] += 1
                if kind in ('response.created', 'response.speak'):
                    source_state['response_id'] = event.get('response_id') or event.get('response', {}).get('id')
                for session in tuple(self.sessions.values()):
                    if session.source == source_id and session.state == 'active':
                        try:
                            session.ingest(envelope['event'], envelope.get('direction', 'upstream'))
                        except (ValueError, TypeError, AttributeError):
                            session.state, session.failure = 'failed', 'source_audio_invalid'
                            session.emit('avatar.failed', reason=session.failure, voice_continues=True)
        finally:
            self.sources.pop(source_id, None)
            await asyncio.gather(*(s.close() for s in self.sessions.values() if s.source == source_id))
        return socket

    async def whep(self, request):
        session = self.owned(request)
        if request.method == 'DELETE':
            peer = session.peers.pop(request.match_info['peer'], None)
            if peer:
                await peer.close()
            return web.Response(status=204)
        if session.state != 'active':
            raise web.HTTPGone()
        if request.content_type != 'application/sdp':
            raise web.HTTPUnsupportedMediaType()
        if len(session.peers) >= 2:
            raise web.HTTPTooManyRequests(reason='At most two media subscribers per session')
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peer_id = secrets.token_urlsafe(16)
        session.peers[peer_id] = pc

        @pc.on('connectionstatechange')
        async def state_changed():
            if pc.connectionState in ('failed', 'closed'):
                session.peers.pop(peer_id, None)
                if pc.connectionState != 'closed':
                    await pc.close()

        try:
            async with asyncio.timeout(15):
                offer = await request.text()
                await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type='offer'))
                if {t.kind for t in pc.getTransceivers()} != {'audio', 'video'}:
                    raise ValueError('Offer must request both audio and video')
                origin = monotonic()
                pc.addTrack(TimelineTrack(session, 'audio', origin))
                pc.addTrack(TimelineTrack(session, 'video', origin))
                await pc.setLocalDescription(await pc.createAnswer())
                sdp = pc.localDescription.sdp
                # Host networking exposes routable host ICE candidates, no Docker bridge addresses.
                return web.Response(status=201, text=sdp, content_type='application/sdp', headers={
                    'Location': f'/v1/avatar/sessions/{session.id}/whep/{peer_id}',
                    'Cache-Control': 'no-store',
                    'X-Avatar-Timeline-Origin-Ms': str(round((origin - session.origin) * 1000)),
                })
        except BaseException:
            session.peers.pop(peer_id, None)
            await pc.close()
            raise

    async def lifecycle(self, app):
        async def reap():
            while True:
                await asyncio.sleep(15)
                now = monotonic()
                for sid, session in tuple(self.sessions.items()):
                    if now - session.touched > self.idle_timeout:
                        await session.close()
                        self.sessions.pop(sid, None)
        task = asyncio.create_task(reap())
        yield
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(*(item['socket'].close() for item in tuple(self.sources.values())))
        await asyncio.gather(*(s.close() for s in tuple(self.sessions.values())))


def create_app(token=None, **options):
    service = AvatarService(token or os.environ.get('OMNI_AVATAR_INTERNAL_TOKEN'), **options)
    app = web.Application(client_max_size=1024 * 1024, middlewares=[service.authenticate])
    app['service'] = service
    app.cleanup_ctx.append(service.lifecycle)
    async def livez(request):
        return web.json_response({'state': 'running'})
    app.add_routes([
        web.get('/livez', livez),
        web.get('/internal/sources/{id}', service.source),
        web.post('/v1/avatar/sessions', service.create),
        web.get('/v1/avatar/sessions/{id}', service.detail),
        web.delete('/v1/avatar/sessions/{id}', service.detail),
        web.post('/v1/avatar/sessions/{id}/audio', service.audio),
        web.get('/v1/avatar/sessions/{id}/events', service.events),
        web.post('/v1/avatar/sessions/{id}/whep', service.whep),
        web.delete('/v1/avatar/sessions/{id}/whep/{peer}', service.whep),
    ])
    return app


if __name__ == '__main__':
    if socket_path := os.environ.get('OMNI_AVATAR_SOCKET'):
        web.run_app(create_app(), path=socket_path, handler_cancellation=True, access_log=None)
    else:
        web.run_app(create_app(), host=os.environ.get('OMNI_AVATAR_BIND', '127.0.0.1'),
                    port=int(os.environ.get('OMNI_AVATAR_PORT', '8090')), handler_cancellation=True, access_log=None)
