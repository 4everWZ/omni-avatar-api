import asyncio
import base64
import tempfile
import unittest

from aiohttp.test_utils import TestClient, TestServer

from services.gateway.app import create_app as gateway_app
from services.gateway.config import Config
from services.gateway.image_jobs import DedicatedImageJobs
from services.images.app import create_app as worker_app


class DedicatedImagesTest(unittest.IsolatedAsyncioTestCase):
    async def test_job_without_a_text_engine_or_shared_volume(self):
        class Backend:
            def prepare(self, payload, cancel):
                return payload['prompt'].encode()

            def generate(self, prepared, cancel):
                return {'png': base64.b64encode(prepared).decode(), 'timing': {},
                        'worker_peak_reserved_bytes': 17}

        with tempfile.TemporaryDirectory() as root:
            async with TestServer(worker_app(Backend(), 'private', None)) as worker:
                def factory(text):
                    self.assertIsNone(text)
                    return DedicatedImageJobs(root, str(worker.make_url('')).rstrip('/'), 'private', 3)

                config = Config('http://unused', 'model', 'revision', ('one', 'two'), 'admin')
                async with TestClient(TestServer(gateway_app(config, image_factory=factory))) as client:
                    auth = {'Authorization': 'Bearer one'}
                    reply = await client.post('/v1/images/jobs', headers=auth, json={'prompt': 'test'})
                    self.assertEqual(reply.status, 202)
                    path = reply.headers['Location']
                    async with asyncio.timeout(4):
                        while True:
                            job = await (await client.get(path, headers=auth)).json()
                            if job['state'] not in ('running', 'queued'):
                                break
                            await asyncio.sleep(.02)
                    self.assertEqual(job['state'], 'completed', job)
                    self.assertNotIn('gpu_release', job)
                    result = job['result_url']
                    self.assertEqual((await client.get(result, headers={'Authorization': 'Bearer two'})).status, 404)
                    self.assertEqual(await (await client.get(result, headers=auth)).read(), b'test')
