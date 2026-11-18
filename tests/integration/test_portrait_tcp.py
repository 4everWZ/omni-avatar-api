import asyncio
import base64
from io import BytesIO
import os
import tempfile
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from services.avatar.app import create_app as avatar_app
from services.portrait.app import create_app as portrait_app


class PortraitTransportTest(unittest.IsolatedAsyncioTestCase):
    async def test_remote_preparation_render_and_disconnect(self):
        image = BytesIO()
        Image.new('RGB', (32, 32), 'blue').save(image, 'JPEG')
        received = asyncio.Event()
        marks = {'face_box': [100, 100, 300, 420]}

        class Backend:
            failure = None

            def prepare(self, reference, output, *, crop_scale, crop_vy_ratio, calibration):
                if (crop_scale, crop_vy_ratio, calibration) != (2.2, 0, marks):
                    raise ValueError("Expected caller crop and calibration options")
                output.write_bytes(b'features')

            def start(self, path, emit):
                self.emit = emit
                return self

            def append(self, pcm, start, generation):
                self.emit(start / 24000, generation, image.getvalue())
                received.set()

            def shutdown(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            async with TestServer(portrait_app(folder, 'private', Backend())) as server:
                with patch.dict(os.environ, {'OMNI_PORTRAIT_URL': str(server.make_url('/')),
                                             'OMNI_PORTRAIT_SOCKET': ''}):
                    async with TestClient(TestServer(avatar_app('private'))) as client:
                        headers = {'Authorization': 'Bearer private', 'X-Omni-Owner': 'a' * 64}
                        asset = await client.post('/v1/avatar/assets', headers=headers, json={
                            'mime_type': 'image/jpeg', 'image_base64': base64.b64encode(image.getvalue()).decode()})
                        self.assertEqual(asset.status, 201)
                        job = await client.post('/v1/avatar/generations', headers=headers, json={
                            'reference_asset_id': (await asset.json())['id'],
                            'infra': {'version': 1, 'portrait': {'crop_scale': 2.2, 'crop_vy_ratio': 0,
                                                                'calibration': marks}}})
                        path = '/v1/avatar/generations/' + (await job.json())['id']
                        async with asyncio.timeout(5):
                            while (record := await (await client.get(path, headers=headers)).json())['state'] != 'ready':
                                await asyncio.sleep(.01)
                        params = {'avatar_id': record['avatar_id'], 'source': {'type': 'audio'}}
                        offline = await client.post('/v1/avatar/sessions', headers=headers, json=params | {
                            'infra': {'version': 1, 'portrait': {'online': False}}})
                        self.assertEqual(offline.status, 400)
                        params['infra'] = {'version': 1, 'portrait': {'online': True}}
                        forbidden = await client.post('/v1/avatar/sessions', json=params,
                                                      headers=headers | {'X-Omni-Owner': 'b' * 64})
                        self.assertEqual(forbidden.status, 404)
                        response = await client.post('/v1/avatar/sessions', json=params, headers=headers)
                        self.assertEqual(response.status, 201)
                        session = await response.json()
                        self.assertTrue(session['format']['online'])
                        audio = await client.post(session['audio_url'], headers=headers,
                                                  json={'generation': 0, 'audio': 'AAA='})
                        self.assertEqual(audio.status, 200)
                        await asyncio.wait_for(received.wait(), 2)
                        path = '/v1/avatar/sessions/' + session['id']
                        async with asyncio.timeout(2):
                            while not (await (await client.get(path, headers=headers)).json())['renderer']['generated_frames']:
                                await asyncio.sleep(.01)
                        self.assertEqual((await client.delete(path, headers=headers)).status, 204)
