import asyncio
import base64
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from services.gateway.admission import Busy
from services.gateway.app import create_app
from services.gateway.config import Config
from services.gateway.image_jobs import ImageJobs


class FixtureJobs(ImageJobs):
    engine_phase = 'awake'
    run_phase = 'idle'
    portrait_active = False
    broken_wake = False

    def __init__(self, text, root):
        super().__init__(text, root, root, 'http://worker', 'worker', 'engine', 1)
        self.calls = []

    def claim_portrait_window(self, *, probe=False):
        if self.portrait_active:
            return False
        if not probe:
            self.atomic(self.marker, b'blocked')
        return True

    async def llm(self, action):
        self.calls.append(action)
        if action == 'wake' and self.broken_wake:
            raise OSError('fixture')
        self.engine_phase = {'wake': 'awake', 'release_kv': 'kv_released',
                             'offload': 'weights_offloaded'}[action]
        return self.engine_phase

    async def worker(self, method, path, **options):
        if path == '/state':
            return {'busy': self.run_phase == 'running', 'id': getattr(self, 'worker_id', None),
                    'free_bytes': 4_000_000 if self.engine_phase == 'weights_offloaded' else 0}
        if method == 'PUT' and path.startswith('/prepare/'):
            self.worker_id = path.rsplit('/', 1)[1]
            self.run_phase = 'prepared'
        elif method == 'PUT':
            self.worker_id = path.rsplit('/', 1)[1]
            self.run_phase = 'running'
        if path.endswith('/cancel'):
            self.run_phase = 'cancelled'
        return {'state': self.run_phase, 'output': {'png': base64.b64encode(b'test pixels').decode(),
                'timing': {'inference': .02}, 'worker_peak_reserved_bytes': 42}}


class ImageJobTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        root_config = Config('http://engine', 'model', 'pin', ('client', 'other'), 'admin')
        llm = Config('http://text', 'text', 'pin', ('client', 'other'), 'admin',
                     realtime_enabled=False, batch_cap=4, token_budget=1000, queue_timeout=.025)

        def image_factory(text):
            self.jobs = FixtureJobs(text, root)
            return self.jobs

        app = create_app(root_config, llm, image_factory=image_factory)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.auth = {'Authorization': 'Bearer client'}
        await self.until(lambda: self.jobs.text.admission.paused is None)

    async def until(self, predicate):
        async with asyncio.timeout(3):
            while not predicate():
                await asyncio.sleep(.005)

    async def submit(self):
        async with await self.client.post('/v1/images/jobs', headers=self.auth,
                                         json={'prompt': 'anime portrait', 'seed': 7}) as reply:
            self.assertEqual(reply.status, 202)
            return reply.headers['Location']

    async def test_portrait_priority_drains_llm_and_restores_after_completion(self):
        self.jobs.portrait_active = True
        self.jobs.calls.clear()
        admission = self.jobs.text.admission
        async with admission.hold('chat', tokens=1):
            path = await self.submit()
            await asyncio.sleep(.05)
            self.assertFalse(self.jobs.calls)
            self.assertIsNone(admission.paused)
            self.jobs.portrait_active = False
            await self.until(lambda: admission.paused)
            self.assertFalse(self.jobs.calls)
            with self.assertRaises(Busy):
                async with admission.hold('chat', tokens=1):
                    self.fail('Generation entered during a drain')
        await self.until(lambda: self.jobs.run_phase == 'running')
        self.assertEqual(self.jobs.calls, ['release_kv', 'wake', 'offload'])
        self.jobs.run_phase = 'completed'
        await self.until(lambda: self.jobs.active is None)
        self.assertEqual(self.jobs.engine_phase, 'awake')
        self.assertIsNone(admission.paused)
        self.assertFalse(self.jobs.marker.exists())
        async with await self.client.get(path, headers=self.auth) as reply:
            data = await reply.json()
            self.assertEqual(data['state'], 'completed')
            self.assertEqual(data['gpu_release'], 'weights_and_kv')
        async with await self.client.get(data['result_url'], headers=self.auth) as reply:
            self.assertEqual(await reply.read(), b'test pixels')
        async with await self.client.get(data['result_url'], headers={'Authorization': 'Bearer other'}) as reply:
            self.assertEqual(reply.status, 404)

    async def test_cancel_is_terminal_only_after_worker_drains_and_llm_wakes(self):
        path = await self.submit()
        await self.until(lambda: self.jobs.run_phase == 'running')
        async with await self.client.post(path + '/cancel', headers=self.auth) as reply:
            self.assertEqual((await reply.json())['state'], 'cancelling')
        await self.until(lambda: self.jobs.active is None)
        async with await self.client.get(path, headers=self.auth) as reply:
            self.assertEqual((await reply.json())['state'], 'cancelled')
        self.assertEqual(self.jobs.engine_phase, 'awake')
        self.assertFalse(list(Path(self.jobs.root).glob('*.png')))

    async def test_failed_restore_holds_admission_until_confirmed_recovery(self):
        path = await self.submit()
        await self.until(lambda: self.jobs.run_phase == 'running')
        self.jobs.broken_wake = True
        self.jobs.run_phase = 'completed'
        await self.until(lambda: self.jobs.active is None)
        async with await self.client.get(path, headers=self.auth) as reply:
            data = await reply.json()
            self.assertEqual((data['state'], data['error']), ('failed', 'llm_restore_failed'))
        self.assertEqual(self.jobs.text.admission.paused, 'image_gpu_recovery_failed')
        self.assertTrue(self.jobs.marker.exists())
        self.jobs.broken_wake = False
        await self.until(lambda: self.jobs.text.admission.paused is None)
        self.assertFalse(self.jobs.marker.exists())
