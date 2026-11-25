import asyncio
import importlib
import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def _import_gpu_control():
    try:
        import fastapi  # noqa: F401
        return importlib.import_module('services.llm.patches.llm_gpu_control')
    except ModuleNotFoundError:
        stub = ModuleType('fastapi')

        class HTTPException(Exception):
            def __init__(self, status_code, detail=None):
                super().__init__(detail)
                self.status_code = status_code
                self.detail = detail

        class Request:
            pass

        stub.HTTPException = HTTPException
        stub.Request = Request
        with patch.dict(sys.modules, {'fastapi': stub}):
            return importlib.import_module('services.llm.patches.llm_gpu_control')


llm_gpu_control = _import_gpu_control()


class FakeApp:
    def __init__(self, engine):
        self.state = SimpleNamespace(engine_client=engine)
        self.routes = {}

    def add_api_route(self, path, endpoint, methods=None):
        self.routes[(path, tuple(methods or ()))] = endpoint


class FakeEngine:
    def __init__(self, *, sleeping=False, wake_ok=True, with_is_sleeping=True):
        self.sleeping = sleeping
        self.wake_ok = wake_ok
        self.calls = []
        self.fail_on = None
        if with_is_sleeping:
            self.is_sleeping = self._is_sleeping

    async def _is_sleeping(self):
        self.calls.append(('is_sleeping',))
        if self.fail_on == 'is_sleeping':
            raise RuntimeError('is_sleeping failed')
        return self.sleeping

    async def sleep(self, level, mode):
        self.calls.append(('sleep', level, mode))
        if self.fail_on == f'sleep:{level}':
            raise RuntimeError(f'sleep {level} failed')
        self.sleeping = True

    async def release_kv_cache_memory(self):
        self.calls.append(('release_kv_cache_memory',))
        if self.fail_on == 'release_kv_cache_memory':
            raise RuntimeError('release kv failed')

    async def wake_up(self, tags):
        self.calls.append(('wake_up', tags))
        if self.fail_on == 'wake_up':
            raise RuntimeError('wake_up failed')
        if self.wake_ok:
            self.sleeping = False
        return self.wake_ok


class GpuControlTests(unittest.IsolatedAsyncioTestCase):
    def attach(self, engine, token='gpu-secret'):
        app = FakeApp(engine)
        with patch.dict(os.environ, {'LLM_GPU_CONTROL_TOKEN': token} if token else {}, clear=True):
            llm_gpu_control.attach_gpu_control(app)
        return app

    def request(self, app, token='gpu-secret'):
        headers = {'X-GPU-Control': token} if token is not None else {}
        return SimpleNamespace(headers=headers, app=app)

    async def test_requires_token_configuration_and_valid_header(self):
        unconfigured = self.attach(FakeEngine(), token='')
        self.assertEqual(unconfigured.routes, {})

        engine = FakeEngine()
        app = self.attach(engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        for bad_token in (None, '', 'wrong-token'):
            with self.subTest(bad_token=bad_token), self.assertRaises(llm_gpu_control.HTTPException) as ctx:
                await handler(self.request(app, bad_token), 'state')
            self.assertEqual(ctx.exception.status_code, 403)

    async def test_awake_release_kv_direct_offload_and_wake_lifecycle(self):
        engine = FakeEngine()
        app = self.attach(engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        self.assertEqual(await handler(req, 'state'), {'phase': 'awake', 'is_sleeping': False})
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})
        self.assertEqual(await handler(req, 'release_kv'), {'phase': 'kv_released'})
        self.assertEqual(await handler(req, 'offload'), {'phase': 'weights_offloaded'})
        self.assertEqual(await handler(req, 'state'), {'phase': 'weights_offloaded', 'is_sleeping': True})
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})
        self.assertEqual(engine.calls, [
            ('is_sleeping',),
            ('sleep', 0, 'wait'),
            ('release_kv_cache_memory',),
            ('sleep', 1, 'wait'),
            ('is_sleeping',),
            ('wake_up', None),
        ])

    async def test_release_kv_wakes_only_kv_cache_tags(self):
        engine = FakeEngine(with_is_sleeping=False)
        app = self.attach(engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        self.assertEqual(await handler(req, 'state'), {'phase': 'awake'})
        self.assertEqual(await handler(req, 'release_kv'), {'phase': 'kv_released'})
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})
        self.assertEqual(engine.calls, [
            ('sleep', 0, 'wait'),
            ('release_kv_cache_memory',),
            ('wake_up', ['kv_cache']),
        ])

    async def test_unknown_phase_recovers_via_wake_when_sleeping_or_awake_or_without_probe(self):
        # Case 1: failure leaves engine sleeping; wake calls wake_up(None) and recovers.
        sleeping_engine = FakeEngine()
        sleeping_engine.fail_on = 'release_kv_cache_memory'
        app = self.attach(sleeping_engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        with self.assertLogs(level='ERROR'), self.assertRaises(RuntimeError):
            await handler(req, 'release_kv')
        self.assertEqual(await handler(req, 'state'), {'phase': 'unknown', 'is_sleeping': True})
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})
        self.assertIn(('wake_up', None), sleeping_engine.calls)

        # Case 2: failure occurred before sleep took effect; wake sees is_sleeping() == False and skips wake_up.
        awake_engine = FakeEngine()
        awake_engine.fail_on = 'sleep:1'
        app = self.attach(awake_engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        with self.assertLogs(level='ERROR'), self.assertRaises(RuntimeError):
            await handler(req, 'offload')
        self.assertEqual(await handler(req, 'state'), {'phase': 'unknown', 'is_sleeping': False})
        awake_engine.calls.clear()
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})
        self.assertEqual(awake_engine.calls, [('is_sleeping',)])

        # Case 3: engine without is_sleeping attribute calls wake_up(None) unconditionally from unknown.
        legacy_engine = FakeEngine(with_is_sleeping=False)
        legacy_engine.fail_on = 'sleep:0'
        app = self.attach(legacy_engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        with self.assertLogs(level='ERROR'), self.assertRaises(RuntimeError):
            await handler(req, 'release_kv')
        self.assertEqual(await handler(req, 'state'), {'phase': 'unknown'})
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})
        self.assertIn(('wake_up', None), legacy_engine.calls)

    async def test_failed_wake_retains_unknown_until_retry_succeeds(self):
        engine = FakeEngine(wake_ok=False)
        engine.fail_on = 'sleep:1'
        app = self.attach(engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        with self.assertLogs(level='ERROR'), self.assertRaises(RuntimeError):
            await handler(req, 'offload')
        engine.sleeping = True
        with self.assertLogs(level='ERROR'), self.assertRaises(RuntimeError):
            await handler(req, 'wake')
        self.assertEqual(await handler(req, 'state'), {'phase': 'unknown', 'is_sleeping': True})
        engine.wake_ok = True
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})

    async def test_cancelled_error_sets_unknown_and_allows_recovery(self):
        class CancellingEngine(FakeEngine):
            async def sleep(self, level, mode):
                raise asyncio.CancelledError()

        engine = CancellingEngine()
        app = self.attach(engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        with self.assertLogs(level='ERROR'), self.assertRaises(asyncio.CancelledError):
            await handler(req, 'offload')
        self.assertEqual(await handler(req, 'state'), {'phase': 'unknown', 'is_sleeping': False})
        self.assertEqual(await handler(req, 'wake'), {'phase': 'awake'})

    async def test_invalid_transitions_return_409(self):
        engine = FakeEngine()
        app = self.attach(engine)
        handler = app.routes[('/internal/gpu-control/{action}', ('POST',))]
        req = self.request(app)

        with self.assertRaises(llm_gpu_control.HTTPException) as ctx:
            await handler(req, 'bogus')
        self.assertEqual(ctx.exception.status_code, 409)

        await handler(req, 'release_kv')
        with self.assertRaises(llm_gpu_control.HTTPException) as ctx:
            await handler(req, 'release_kv')
        self.assertEqual(ctx.exception.status_code, 409)

        await handler(req, 'offload')
        for bad_action in ('release_kv', 'offload'):
            with self.subTest(bad_action=bad_action), self.assertRaises(llm_gpu_control.HTTPException) as ctx:
                await handler(req, bad_action)
            self.assertEqual(ctx.exception.status_code, 409)
