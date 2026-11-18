import asyncio
import base64
import io
import tempfile
import threading
import unittest
from pathlib import Path

from aiohttp import ClientSession, web
from PIL import Image
from services.portrait.app import create_app
from services.portrait.registration import PreparationError

AUTH = {'Authorization': 'Bearer private', 'X-Omni-Owner': 'a' * 64}
OTHER = AUTH | {'X-Omni-Owner': 'b' * 64}


class Backend:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
    def prepare(self, image, output):
        self.entered.set()
        self.release.wait(5)
        Path(output).write_bytes(b'prepared')


class PreparationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = Backend()
        self.app = create_app(self.directory.name, 'private', self.backend)
        runner = web.AppRunner(self.app)
        await runner.setup()
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        self.addAsyncCleanup(runner.cleanup)
        self.http = ClientSession('http://127.0.0.1:' + str(runner.addresses[0][1]))
        self.addAsyncCleanup(self.http.close)
        self.addCleanup(self.backend.release.set)

    async def image(self):
        output = io.BytesIO()
        Image.new('RGB', (400, 400), 'white').save(output, 'PNG')
        async with self.http.post('/v1/avatar/assets', headers=AUTH,
                json={'mime_type': 'image/png', 'image_base64': base64.b64encode(output.getvalue()).decode()}) as response:
            self.assertEqual(response.status, 201)
            return (await response.json())['id']

    async def generate(self, reference, headers=AUTH):
        return await self.http.post('/v1/avatar/generations', headers=headers,
                                    json={'reference_asset_id': reference})

    async def test_cancel_holds_lease_until_gpu_work_returns(self):
        reference = await self.image()
        response = await self.generate(reference)
        job = await response.json()
        response.release()
        await asyncio.to_thread(self.backend.entered.wait, 2)
        path = '/v1/avatar/generations/' + job['id']
        async with self.http.delete(path, headers=AUTH) as response:
            self.assertEqual((await response.json())['state'], 'cancelled')
        self.assertFalse(self.app['service'].slots)
        self.backend.release.set()
        async with asyncio.timeout(2):
            while self.app['service'].tasks:
                await asyncio.sleep(.01)
        async with self.http.get(path, headers=AUTH) as response:
            self.assertEqual((await response.json())['state'], 'cancelled')
        self.assertFalse(list(Path(self.directory.name).glob('*/jobs/*.pkl')))

    async def test_reference_and_generated_avatar_ownership(self):
        reference = await self.image()
        async with await self.generate(reference, OTHER) as response:
            self.assertEqual(response.status, 404)
        self.backend.release.set()
        async with await self.generate(reference) as response:
            job = await response.json()
        path = '/v1/avatar/generations/' + job['id']
        async with asyncio.timeout(2):
            while self.app['service'].tasks:
                await asyncio.sleep(.01)
        async with self.http.get(path, headers=OTHER) as response:
            self.assertEqual(response.status, 404)
        async with self.http.get(path, headers=AUTH) as response:
            result = await response.json()
            self.assertEqual(result['state'], 'ready')
        avatar = '/v1/avatar/avatars/' + result['avatar_id']
        for suffix in ('', '/preview'):
            async with self.http.get(avatar + suffix, headers=OTHER) as response:
                self.assertEqual(response.status, 404)
        async with self.http.delete(path, headers=AUTH) as response:
            self.assertEqual(response.status, 204)
        async with self.http.delete('/v1/avatar/assets/' + reference, headers=AUTH) as response:
            self.assertEqual(response.status, 204)

    async def test_two_preparations_get_independent_slots(self):
        other = Backend()
        self.addCleanup(other.release.set)
        self.app['service'].slots.append(other)
        reference = await self.image()
        try:
            for _ in range(2):
                async with await self.generate(reference) as response:
                    self.assertEqual(response.status, 202)
            self.assertTrue(await asyncio.to_thread(self.backend.entered.wait, 2))
            self.assertTrue(await asyncio.to_thread(other.entered.wait, 2))
            self.assertFalse(self.app['service'].slots)
        finally:
            self.backend.release.set()
            other.release.set()

    async def test_image_validation_and_no_url_fetch(self):
        for body in ({'mime_type': 'image/png', 'image_base64': '!'},
                     {'mime_type': 'image/svg+xml', 'image_base64': 'YQ=='},
                     {'url': 'http://localhost/private'}, []):
            async with self.http.post('/v1/avatar/assets', headers=AUTH, json=body) as response:
                self.assertEqual(response.status, 400)

    async def test_calibration_is_per_generation_and_failure_is_readable_by_owner(self):
        reference = await self.image()
        calibration = {'face_box': [20, 30, 150, 200]}
        received = []

        def fail(image, output, **options):
            received.append(options)
            raise PreparationError('eye_mesh')

        self.backend.prepare = fail
        body = {'reference_asset_id': reference, 'infra': {'version': 1, 'portrait': {
            'crop_scale': 2.1, 'calibration': calibration}}}
        path = '/v1/avatar/generations'
        async with self.http.post(path, headers=OTHER, json=body) as response:
            self.assertEqual(response.status, 404)
        async with self.http.post(path, headers=AUTH, json=body) as response:
            self.assertEqual(response.status, 202)
            job_path = path + '/' + (await response.json())['id']
        async with asyncio.timeout(2):
            while self.app['service'].tasks:
                await asyncio.sleep(.01)
        async with self.http.get(job_path, headers=AUTH) as response:
            job = await response.json()
        self.assertEqual(received, [body['infra']['portrait']])
        self.assertEqual(job['crop']['calibration'], calibration)
        self.assertEqual((job['state'], job['error'], job['avatar_id']),
                         ('failed', 'reference_preparation_failed', None))
        self.assertEqual(job['diagnostic'], {'stage': 'eye_mesh', 'reason': 'stage_failed'})
        async with self.http.get(job_path, headers=OTHER) as response:
            self.assertEqual(response.status, 404)
        body['infra']['portrait']['calibration'] = {'face_box': [6, 2, 1, 4]}
        async with self.http.post(path, headers=AUTH, json=body) as response:
            self.assertEqual(response.status, 400)
        self.assertEqual(len(received), 1)
