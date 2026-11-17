"""Exercise the real proxy and tap, with only the GPU engine replaced."""
import asyncio
import base64
import unittest
from unittest.mock import patch

from aiohttp import ClientSession, web
from yarl import URL

from services.gateway.app import Config, create_app as gateway_app
from services.avatar.app import create_app as avatar_app

AUTH = {'Authorization': 'Bearer client-one'}
OTHER = {'Authorization': 'Bearer client-two'}


class BridgeTest(unittest.IsolatedAsyncioTestCase):
    async def serve(self, app):
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        return f'http://127.0.0.1:{runner.addresses[0][1]}'

    async def asyncSetUp(self):
        async def realtime(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            await socket.send_json({'type': 'session.created', 'session': {'capabilities': {}}})
            async for message in socket:
                if message.type.name == 'TEXT':
                    await socket.send_json({'type': 'response.output_audio.delta', 'response_id': 'reply',
                                            'delta': base64.b64encode(b'\x00\x10' * 24000).decode(),
                                            'sample_rate_hz': 24000, 'metadata': {'epoch': 0}})
            return socket
        async def chat(request):
            return web.json_response({'choices': [{'message': {'content': 'concurrent'}}]})
        upstream = web.Application()
        upstream.add_routes([web.get('/v1/realtime', realtime), web.post('/v1/chat/completions', chat)])
        self.engine_url = await self.serve(upstream)
        self.avatar = avatar_app('internal')
        self.avatar_url = await self.serve(self.avatar)

    async def gateway(self, url=None):
        app = gateway_app(Config(engine_url=self.engine_url, model='model', checkpoint='test',
                                 tokens=('client-one', 'client-two'), admin_token='admin',
                                 avatar_url=url or self.avatar_url, avatar_token='internal'))
        client = ClientSession(base_url=await self.serve(app))
        self.addAsyncCleanup(client.close)
        return client

    async def test_tap_owner_media_events_and_live_chat(self):
        client = await self.gateway()
        voice = await client.ws_connect('/v1/realtime?duplex=1&avatar=1', headers=AUTH)
        source = await voice.receive_json()
        self.assertTrue(source['avatar_available'])
        self.assertEqual((await voice.receive_json())['type'], 'session.created')
        body = {'avatar_id': 'ava', 'source': {'type': 'realtime', 'session_id': source['source_session_id']}}
        async with client.post('/v1/avatar/sessions', headers=OTHER, json=body) as response:
            self.assertEqual(response.status, 404)
        async with client.post('/v1/avatar/sessions', headers=AUTH, json=body) as response:
            self.assertEqual(response.status, 201)
            avatar = await response.json()
        events = await client.ws_connect(avatar['events_url'], headers=AUTH)
        await events.receive_json()
        await voice.send_json({'type': 'fixture.generate'})
        self.assertEqual((await voice.receive_json())['type'], 'response.output_audio.delta')
        self.assertEqual((await events.receive_json(timeout=2))['type'], 'avatar.audio.queued')
        async with client.post('/v1/chat/completions', headers=OTHER, json={
                'model': 'omni-default', 'messages': [{'role': 'user', 'content': 'test'}]}) as response:
            self.assertEqual(response.status, 200)
        await voice.close()
        self.assertEqual((await events.receive_json(timeout=3))['type'], 'avatar.closed')
        await events.close()

    async def test_unavailable_renderer_keeps_original_voice_working(self):
        client = await self.gateway('http://127.0.0.1:1')
        voice = await client.ws_connect('/v1/realtime?duplex=1&avatar=1', headers=AUTH)
        self.assertFalse((await voice.receive_json())['avatar_available'])
        await voice.receive_json()
        await voice.send_json({'type': 'fixture.generate'})
        self.assertEqual((await voice.receive_json(timeout=2))['type'], 'response.output_audio.delta')
        await voice.close()

    async def test_source_tap_processes_heartbeat_pongs(self):
        original = ClientSession.ws_connect
        def faster_internal_heartbeat(client, url, **kwargs):
            if '/internal/sources/' in str(url):
                kwargs['heartbeat'] = .1
            return original(client, url, **kwargs)
        with patch.object(ClientSession, 'ws_connect', faster_internal_heartbeat):
            client = await self.gateway()
            voice = await client.ws_connect('/v1/realtime?duplex=1&avatar=1', headers=AUTH)
            source = await voice.receive_json()
            await voice.receive_json()
            async with client.post('/v1/avatar/sessions', headers=AUTH, json={
                    'avatar_id': 'ava', 'source': {'type': 'realtime', 'session_id': source['source_session_id']}}) as response:
                avatar = await response.json()
            events = await client.ws_connect(avatar['events_url'], headers=AUTH)
            await events.receive_json()
            await asyncio.sleep(.6)
            await voice.send_json({'type': 'fixture.generate'})
            self.assertEqual((await voice.receive_json(timeout=2))['type'], 'response.output_audio.delta')
            self.assertEqual((await events.receive_json(timeout=2))['type'], 'avatar.audio.queued')
            await voice.close()
            await events.close()

    async def test_plain_realtime_keeps_original_event_stream_with_avatar_enabled(self):
        client = await self.gateway()
        voice = await client.ws_connect('/v1/realtime?duplex=1', headers=AUTH)
        self.assertEqual((await voice.receive_json(timeout=2))['type'], 'session.created')
        self.assertFalse(self.avatar['service'].sources)
        await voice.close()

    async def test_proxy_cannot_normalize_a_public_path_into_private_source_control(self):
        client = await self.gateway()
        for dots in ('%2e%2e', '%252e%252e'):
            path = URL(f'/v1/avatar/{dots}/{dots}/internal/sources/injected', encoded=True)
            async with client.get(path, headers=AUTH) as response:
                self.assertEqual(response.status, 404)
            self.assertFalse(self.avatar['service'].sources)
