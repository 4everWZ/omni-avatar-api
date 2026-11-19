import asyncio
import base64
import hashlib
import math
import struct
import sys
import unittest
from time import perf_counter
from unittest.mock import patch

from aiohttp import ClientSession, web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from PIL import ImageChops, ImageStat

from services.avatar.app import create_app
from services.avatar.sessions import Session

AUTH = {'Authorization': 'Bearer internal', 'X-Omni-Owner': hashlib.sha256(b'one').hexdigest()}
OTHER = AUTH | {'X-Omni-Owner': hashlib.sha256(b'two').hexdigest()}
PCM = base64.b64encode(struct.pack('<24000h', *([6000] * 24000))).decode()


class AvatarFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.app = create_app('internal')
        runner = web.AppRunner(self.app, access_log=None, handler_cancellation=True)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        self.client = ClientSession(base_url=f'http://127.0.0.1:{runner.addresses[0][1]}')
        self.addAsyncCleanup(self.client.close)

    async def create(self, auth=AUTH, source=None):
        async with self.client.post('/v1/avatar/sessions', headers=auth, json={
                'avatar_id': 'ava', 'source': source or {'type': 'audio'}}) as response:
            self.assertEqual(response.status, 201, await response.text())
            return await response.json()


class AvatarTest(AvatarFixture):
    async def test_render_failure_releases_async_worker(self):
        released = asyncio.Event()
        class Driver:
            capabilities = {}
            def close(self):
                async def release():
                    await asyncio.sleep(0)
                    released.set()
                return asyncio.create_task(release())
        session = Session('owner', 'test-avatar', None, Driver())
        await session.fail('renderer_failed')
        self.assertEqual(session.state, 'failed')
        self.assertTrue(released.is_set())
        self.assertTrue(session.recent_events[-1]['voice_continues'])

    async def test_owner_isolation_and_idempotent_close(self):
        item = await self.create()
        path = '/v1/avatar/sessions/' + item['id']
        for suffix, method in (('', 'get'), ('', 'delete'), ('/audio', 'post'), ('/events', 'get'), ('/whep', 'post')):
            async with getattr(self.client, method)(path + suffix, headers=OTHER) as response:
                self.assertEqual(response.status, 404)
        for _ in range(2):
            async with self.client.delete(path, headers=AUTH) as response:
                self.assertEqual(response.status, 204)
        async with self.client.get(path, headers=AUTH) as response:
            self.assertEqual((await response.json())['state'], 'closed')

    async def test_failed_source_handshake_does_not_leave_a_reserved_source(self):
        async with self.client.get('/internal/sources/failed-upgrade', headers=AUTH) as response:
            self.assertEqual(response.status, 400)
        self.assertNotIn('failed-upgrade', self.app['service'].sources)
        source = await self.client.ws_connect('/internal/sources/failed-upgrade', headers=AUTH)
        self.assertEqual((await source.receive_json())['type'], 'source.ready')
        await source.close()

    async def test_invalid_inputs_and_stale_generation(self):
        item = await self.create()
        for payload in ({'audio': PCM}, {'audio': '!bad', 'generation': 0},
                        {'audio': PCM, 'generation': 0, 'sample_rate_hz': 16000}):
            async with self.client.post(item['audio_url'], headers=AUTH, json=payload) as response:
                self.assertEqual(response.status, 400)
        async with self.client.post(item['audio_url'], headers=AUTH, json={'audio': PCM, 'generation': 0}) as response:
            self.assertEqual(response.status, 200)
        async with self.client.post(item['audio_url'], headers=AUTH, json={'type': 'clear'}) as response:
            self.assertEqual((await response.json())['generation'], 1)
        self.assertFalse(self.app['service'].sessions[item['id']].segments)
        async with self.client.post(item['audio_url'], headers=AUTH, json={'audio': PCM, 'generation': 0}) as response:
            self.assertEqual(response.status, 400)

    async def test_source_ownership_clear_fence_and_disconnect(self):
        source = await self.client.ws_connect('/internal/sources/source-one', headers=AUTH)
        await source.receive_json()
        for auth in (OTHER,):
            async with self.client.post('/v1/avatar/sessions', headers=auth, json={
                    'avatar_id': 'ava', 'source': {'type': 'realtime', 'session_id': 'source-one'}}) as response:
                self.assertEqual(response.status, 404)
        item = await self.create(source={'type': 'realtime', 'session_id': 'source-one'})
        events = await self.client.ws_connect(item['events_url'], headers=AUTH)
        await events.receive_json()
        audio = {'type': 'response.output_audio.delta', 'response_id': 'old', 'delta': PCM,
                 'metadata': {'epoch': 0}, 'sample_rate_hz': 24000}
        await source.send_json({'event': audio})
        self.assertEqual((await events.receive_json(timeout=2))['type'], 'avatar.audio.queued')
        await source.send_json({'direction': 'downstream', 'event': {'type': 'response.cancel'}})
        self.assertEqual((await events.receive_json(timeout=2))['type'], 'avatar.cleared')
        await source.send_json({'event': audio})
        await source.send_json({'event': audio | {'response_id': 'new', 'metadata': {'epoch': 1}}})
        while (event := await events.receive_json(timeout=2))['type'] != 'avatar.audio.queued':
            pass
        self.assertEqual(event['response_id'], 'new')
        session = self.app['service'].sessions[item['id']]
        self.assertEqual(len(session.segments), 1)
        async with self.client.post(item['audio_url'], headers=AUTH, json={'audio': PCM, 'generation': session.generation}) as response:
            self.assertEqual(response.status, 409)
        await source.close()
        while (event := await events.receive_json(timeout=2))['type'] != 'avatar.closed':
            pass
        self.assertEqual(session.state, 'closed')
        await events.close()

    async def test_bounded_capacity_and_buffer(self):
        item = await self.create()
        for _ in range(3):
            await self.create()
        async with self.client.post('/v1/avatar/sessions', headers=AUTH, json={
                'avatar_id': 'ava', 'source': {'type': 'audio'}}) as response:
            self.assertEqual(response.status, 429)
        session = self.app['service'].sessions[item['id']]
        for _ in range(29):
            session.append({'audio': PCM, 'generation': 0})
        with self.assertRaises(ValueError):
            session.append({'audio': PCM, 'generation': 0})

    async def test_cancel_before_first_audio_rejects_old_epoch(self):
        source = await self.client.ws_connect('/internal/sources/source-early-cancel', headers=AUTH)
        await source.receive_json()
        item = await self.create(source={'type': 'realtime', 'session_id': 'source-early-cancel'})
        events = await self.client.ws_connect(item['events_url'], headers=AUTH)
        await events.receive_json()
        await source.send_json({'direction': 'downstream', 'event': {'type': 'response.cancel'}})
        self.assertEqual((await events.receive_json(timeout=2))['type'], 'avatar.cleared')
        for response, epoch in (('stale', 0), ('fresh', 1)):
            await source.send_json({'event': {'type': 'response.output_audio.delta', 'response_id': response,
                                             'delta': PCM, 'metadata': {'epoch': epoch}}})
        self.assertEqual((await events.receive_json(timeout=2))['response_id'], 'fresh')
        self.assertEqual(len(self.app['service'].sessions[item['id']].segments), 1)
        await events.close()
        await source.close()

    async def test_receipts_cannot_claim_future_or_old_generation(self):
        item = await self.create()
        events = await self.client.ws_connect(item['events_url'], headers=AUTH)
        await events.receive_json()
        await events.send_json({'type': 'playback.ack', 'generation': 0, 'played_ms': 999999})
        self.assertEqual((await events.receive_json())['type'], 'avatar.error')
        await events.send_json({'type': 'playback.ack', 'generation': 0, 'played_ms': 0})
        async with self.client.get('/v1/avatar/sessions/' + item['id'], headers=AUTH) as response:
            detail = await response.json()
        # Synchronize with the websocket handler via a rejected message.
        await events.send_json({'type': 'invalid'})
        await events.receive_json()
        self.assertEqual(self.app['service'].sessions[item['id']].last_ack['source'], 'consumer_report')
        await events.close()


class WhepTest(AvatarFixture):
    @classmethod
    def setUpClass(cls):
        if sys.platform == 'win32':
            # Use selector UDP sockets: Proactor shutdown can leave aioice.close() pending.
            cls.addClassCleanup(asyncio.set_event_loop_policy, asyncio.get_event_loop_policy())
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    async def test_whep_real_audio_video_and_peer_delete(self):
        self.enterContext(patch('aioice.ice.get_host_addresses', return_value=['127.0.0.1']))
        item = await self.create()
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        self.addAsyncCleanup(peer.close)
        tracks = {}
        @peer.on('track')
        def track_started(track):
            tracks[track.kind] = track
        peer.addTransceiver('audio', direction='recvonly')
        peer.addTransceiver('video', direction='recvonly')
        await peer.setLocalDescription(await peer.createOffer())
        async with self.client.post(item['media_url'], headers=AUTH | {'Content-Type': 'application/sdp'},
                                    data=peer.localDescription.sdp) as response:
            self.assertEqual(response.status, 201, await response.text())
            location = response.headers['Location']
            origin_ms = int(response.headers['X-Avatar-Timeline-Origin-Ms'])
            self.assertEqual(origin_ms % 40, 0)
            answer = RTCSessionDescription(type='answer', sdp=await response.text())
        # Queue a silence / tone transition before the receiver accepts the
        # answer. RTP cannot flow for the next second.
        tone = struct.pack('<9600h', *(round(10000 * math.sin(2 * math.pi * 440 * i / 24000))
                                      for i in range(9600)))
        waveform = base64.b64encode(bytes(2 * 7200) + tone).decode()
        async with self.client.post(item['audio_url'], headers=AUTH, json={'audio': waveform, 'generation': 0}) as r:
            self.assertEqual(r.status, 200)
        await asyncio.sleep(1)
        await peer.setRemoteDescription(answer)

        renderer = self.app['service'].sessions[item['id']].renderer
        opened = renderer.render(1, 1).resize((80, 90))
        closed = renderer.render(1, 0).resize(opened.size)

        async def transition(kind):
            first = None
            while True:
                frame = await tracks[kind].recv()
                wall = perf_counter()
                if first is None:
                    first = wall
                if kind == 'audio':
                    pcm = bytes(frame.planes[0])[:frame.samples * len(frame.layout.channels) * 2]
                    active = any(abs(sample) > 1200 for sample, in struct.iter_unpack('<h', pcm))
                else:
                    self.assertEqual((frame.width, frame.height), (640, 720))
                    img = frame.to_image().resize(opened.size)
                    def distance(pose):
                        return sum(ImageStat.Stat(ImageChops.difference(img, pose)).mean)
                    active = distance(opened) < distance(closed)
                if active:
                    return frame.time, wall, wall - first

        async with asyncio.timeout(15):
            audio, video = await asyncio.gather(transition('audio'), transition('video'))
        self.assertLess(abs(audio[0] - video[0]), .10, (audio, video))
        self.assertLess(abs(audio[1] - video[1]), .16, (audio, video))
        for pts, _, elapsed in (audio, video):
            self.assertGreater(pts, .20)
            self.assertGreater(elapsed, .20, 'Negotiation must not compress the received leading silence')
        played = origin_ms + round(audio[0] * 1000)
        async with self.client.ws_connect(item['events_url'], headers=AUTH) as events:
            await events.receive_json()
            await events.send_json({'type': 'playback.ack', 'generation': 0, 'played_ms': played})
            await events.send_json({'type': 'invalid'})
            self.assertEqual((await events.receive_json())['type'], 'avatar.error')
            self.assertEqual(self.app['service'].sessions[item['id']].last_ack['played_ms'], played)
        async with self.client.delete(location, headers=AUTH) as response:
            self.assertEqual(response.status, 204)
        self.assertFalse(self.app['service'].sessions[item['id']].peers)


if __name__ == '__main__':
    unittest.main()
