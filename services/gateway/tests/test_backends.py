import asyncio
from dataclasses import replace
import unittest
import os
from unittest.mock import patch

from aiohttp import ClientSession, web
from services.gateway.app import Config, create_app
from services.gateway.config import parse_model_aliases


class BackendPolicyTest(unittest.TestCase):
    def setUp(self):
        self.parent = Config(engine_url='http://omni', model='omni', checkpoint='omni',
                             tokens=('client',), admin_token='admin', queue_limit=9,
                             queue_timeout=2, request_timeout=7)
        self.environment = {'LLM_ENGINE_URL': 'http://llm', 'LLM_MODEL_ID': 'text', 'LLM_CHECKPOINT': 'text'}

    def test_text_policy_is_independent(self):
        with patch.dict(os.environ, self.environment | {'LLM_QUEUE_LIMIT': '32', 'LLM_QUEUE_TIMEOUT': '120',
                                                       'LLM_REQUEST_TIMEOUT': '900'}, clear=True):
            child = self.parent.text_backend()
        self.assertEqual((child.queue_limit, child.queue_timeout, child.request_timeout), (32, 120, 900))
        self.assertEqual((self.parent.queue_limit, self.parent.queue_timeout, self.parent.request_timeout), (9, 2, 7))
        self.assertEqual(child.batch_cap, 4)
        self.assertEqual(child.token_budget, 262144)
        self.assertEqual(child.engine_version, 'vLLM 0.34.0')
        self.assertIsNone(child.default_thinking_token_budget)
        self.assertFalse(child.realtime_enabled)

    def test_workstation_cap_thinking_budget_engine_version_and_worker_configs(self):
        base_env = {
            'OMNI_GATEWAY_TOKEN': 'client',
            'OMNI_GATEWAY_ADMIN_TOKEN': 'admin',
            'OMNI_CHECKPOINT': 'a' * 40,
            'OMNI_MODEL_ID': 'omni-fixture',
            'OMNI_SHARED_CAPACITY': '1',
        }
        for cap in ('1', '4', '16'):
            with self.subTest(cap=cap), patch.dict(os.environ, base_env | {'OMNI_WORKSTATION_CAP': cap}, clear=True):
                cfg = Config.from_env()
                self.assertEqual(cfg.batch_cap, int(cap))
        for bad_cap in ('0', '-1', '17', '1.5'):
            with self.subTest(bad_cap=bad_cap), patch.dict(os.environ, base_env | {'OMNI_WORKSTATION_CAP': bad_cap}, clear=True):
                with self.assertRaises(ValueError):
                    Config.from_env()

        with patch.dict(os.environ, base_env | self.environment | {
            'LLM_ENGINE_VERSION': 'vLLM 0.32.0',
            'LLM_DEFAULT_THINKING_TOKEN_BUDGET': '16384',
            'OMNI_TTS_WORKER_URL': 'http://omni-tts:8000/',
            'OMNI_TTS_INTERNAL_TOKEN': 'tts-secret',
            'OMNI_TTS_TIMEOUT': '45',
            'OMNI_ASR_WORKER_URL': 'http://omni-asr:8000/',
            'OMNI_ASR_INTERNAL_TOKEN': 'asr-secret',
            'OMNI_ASR_TIMEOUT': '90',
        }, clear=True):
            root = Config.from_env()
            child = root.text_backend()
        self.assertIsNone(root.default_thinking_token_budget)
        self.assertEqual((root.tts_worker_url, root.tts_internal_token, root.tts_timeout),
                         ('http://omni-tts:8000', 'tts-secret', 45.0))
        self.assertEqual((root.asr_worker_url, root.asr_internal_token, root.asr_timeout),
                         ('http://omni-asr:8000', 'asr-secret', 90.0))
        self.assertEqual(child.engine_version, 'vLLM 0.32.0')
        self.assertEqual(child.default_thinking_token_budget, 16384)

        for raw, expected in [('', None), ('0', 0), ('8192', 8192)]:
            with self.subTest(budget=raw), patch.dict(os.environ, self.environment | {'LLM_DEFAULT_THINKING_TOKEN_BUDGET': raw}, clear=True):
                self.assertEqual(self.parent.text_backend().default_thinking_token_budget, expected)

        for bad in ('-1', '1.5', '1.0', 'True', 'False', 'true', 'false', 'nan', 'inf', 'abc'):
            with self.subTest(bad_budget=bad), patch.dict(os.environ, self.environment | {'LLM_DEFAULT_THINKING_TOKEN_BUDGET': bad}, clear=True):
                with self.assertRaises(ValueError):
                    self.parent.text_backend()

        for bad_raw in (True, False, 1.5):
            with self.subTest(bad_raw=bad_raw), patch.object(os, 'environ', self.environment | {'LLM_DEFAULT_THINKING_TOKEN_BUDGET': bad_raw}):
                with self.assertRaises(ValueError):
                    self.parent.text_backend()

        for key, bad_val in [
            ('OMNI_TTS_WORKER_URL', 'http://omni-tts:8000/bad'),
            ('OMNI_ASR_WORKER_URL', 'not-an-origin'),
            ('OMNI_TTS_TIMEOUT', '0'),
            ('OMNI_ASR_TIMEOUT', '-5'),
        ]:
            with self.subTest(key=key, bad_val=bad_val), patch.dict(os.environ, base_env | {key: bad_val}, clear=True):
                with self.assertRaises(ValueError):
                    Config.from_env()

    def test_unset_or_empty_values_inherit(self):
        for extra in ({}, {'LLM_QUEUE_LIMIT': '', 'LLM_QUEUE_TIMEOUT': '', 'LLM_REQUEST_TIMEOUT': ''}):
            with patch.dict(os.environ, self.environment | extra, clear=True):
                child = self.parent.text_backend()
            self.assertEqual((child.queue_limit, child.queue_timeout, child.request_timeout), (9, 2, 7))

    def test_invalid_settings_fail_before_serving(self):
        for key, values in {'LLM_MAX_NUM_SEQS': ['0', '-1', '1.5'],
                            'LLM_TOKEN_BUDGET': ['0', '-1', 'nan'],
                            'LLM_QUEUE_TIMEOUT': ['0', '-1', 'inf', 'nan'],
                            'LLM_REQUEST_TIMEOUT': ['0', 'inf'], 'LLM_QUEUE_LIMIT': ['0', '1.5']}.items():
            for value in values:
                with self.subTest(key=key, value=value), patch.dict(os.environ, self.environment | {key: value}, clear=True):
                    with self.assertRaises(ValueError):
                        self.parent.text_backend()

    def test_extra_model_aliases_parsing_and_validation(self):
        for raw in ('', '   '):
            with self.subTest(raw=raw), patch.dict(os.environ, self.environment | {'LLM_EXTRA_MODEL_ALIASES': raw}, clear=True):
                child = self.parent.text_backend()
            self.assertEqual(child.model_aliases, ())
            self.assertEqual(child.accepted_aliases(), ('llm-default',))
            self.assertEqual(child.resolve_upstream_model('llm-default'), 'text')
            self.assertIsNone(child.resolve_upstream_model('llm-zh-chat'))
            self.assertIsNone(child.resolve_upstream_model(None))

        with patch.dict(os.environ, self.environment | {
            'LLM_EXTRA_MODEL_ALIASES': ' llm-zh-chat = llm-zh-chat , alt-chat = custom-lora ',
        }, clear=True):
            child = self.parent.text_backend()
        self.assertEqual(child.model_aliases, (('llm-zh-chat', 'llm-zh-chat'), ('alt-chat', 'custom-lora')))
        self.assertEqual(child.accepted_aliases(), ('llm-default', 'llm-zh-chat', 'alt-chat'))
        self.assertEqual(child.resolve_upstream_model('llm-default'), 'text')
        self.assertEqual(child.resolve_upstream_model('llm-zh-chat'), 'llm-zh-chat')
        self.assertEqual(child.resolve_upstream_model('alt-chat'), 'custom-lora')
        self.assertIsNone(child.resolve_upstream_model('unknown'))

        for bad in (
            'llm-zh-chat',
            '=llm-zh-chat',
            'llm-zh-chat=',
            'a=b=c',
            'llm-default=override',
            'llm-zh-chat=a,llm-zh-chat=b',
            'llm-zh-chat=a,',
            ',llm-zh-chat=a',
            ',',
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_model_aliases(bad, 'llm-default')
                with patch.dict(os.environ, self.environment | {'LLM_EXTRA_MODEL_ALIASES': bad}, clear=True):
                    with self.assertRaises(ValueError):
                        self.parent.text_backend()


class BackendIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def serve(self, app):
        runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        return 'http://127.0.0.1:' + str(runner.addresses[0][1])

    async def asyncSetUp(self):
        self.ended = asyncio.Event()
        self.release = asyncio.Event()
        self.responses = {}
        self.counted = []
        self.chat_payloads = []
        self.response_payloads = []

        async def engine(model):
            async def models(request):
                return web.json_response({'data': [{'id': model, 'root': '/models/' + model}]})

            async def chat(request):
                payload = await request.json()
                self.chat_payloads.append((model, payload))
                self.assertEqual(payload['model'], model)
                if payload.get('stream'):
                    response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                    await response.prepare(request)
                    try:
                        await response.write(b'data: {"choices":[]}\n\n')
                        await self.release.wait()
                        await response.write(b'data: [DONE]\n\n')
                    finally:
                        self.ended.set()
                    return response
                return web.json_response({'model': model, 'choices': [{'message': {'content': model}}]})

            async def tokenize(request):
                payload = await request.json()
                self.assertEqual(payload['model'], model)
                self.assertTrue(payload['add_generation_prompt'])
                self.assertNotIn('modalities', payload)
                return web.json_response({'count': 17, 'tokens': [1, 2]})

            async def responses(request):
                if request.method == 'POST' and 'response_id' not in request.match_info:
                    payload = await request.json()
                    self.response_payloads.append((model, payload))
                    self.assertEqual(payload['model'], model)
                    response_id = payload['request_id']
                    data = {'id': response_id, 'object': 'response', 'model': model,
                            'status': 'in_progress' if payload.get('background') else 'completed', 'output': []}
                    self.responses[response_id] = data
                    if payload.get('stream'):
                        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                        await response.prepare(request)
                        try:
                            import json
                            await response.write(('event: response.created\ndata: ' + json.dumps({
                                'type': 'response.created', 'response': data}) + '\n\n').encode())
                            await response.write(b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hello"}\n\n')
                            await self.release.wait()
                        finally:
                            self.ended.set()
                        return response
                    return web.json_response(data)
                data = self.responses.get(request.match_info['response_id'])
                if data is None:
                    return web.Response(status=404)
                if request.method == 'POST':
                    data['status'] = 'cancelled'
                return web.json_response(data)

            async def count(request):
                payload = await request.json()
                self.counted.append(payload)
                if payload.get('input') == 'rejected':
                    return web.json_response({'error': {'message': 'Invalid multimodal input'}}, status=422)
                return web.json_response({'input_tokens': 100, 'reserved_tokens': 1001 if payload.get('input') == 'oversized' else 250,
                                          'sequences': 1})

            app = web.Application()
            app.add_routes([web.get('/health', lambda r: web.Response()),
                            web.get('/v1/models', models), web.post('/v1/chat/completions', chat),
                            web.post('/tokenize', tokenize), web.post('/v1/responses', responses),
                            web.post('/internal/admission/responses', count),
                            web.post('/internal/admission/chat', count),
                            web.get('/v1/responses/{response_id}', responses),
                            web.post('/v1/responses/{response_id}/cancel', responses)])
            return await self.serve(app)

        config = Config(engine_url=await engine('omni'), model='omni', checkpoint='omni',
                        tokens=('client', 'foreign'), admin_token='admin', model_root='/models/omni', queue_timeout=.1)
        text = replace(config, engine_url=await engine('text'), model='text', checkpoint='text',
                       model_root='/models/text', model_alias='llm-default', public_prefix='/llm',
                       realtime_enabled=False, prompt_tokens_path='/tokenize')
        self.parent_config, self.text_config = config, text
        self.url = await self.serve(create_app(config, text))
        self.http = ClientSession(base_url=self.url, headers={'Authorization': 'Bearer client'})
        self.addAsyncCleanup(self.http.close)

    async def test_responses_image_and_count_use_same_renderer(self):
        image = 'data:image/png;base64,AAAA'
        input_items = [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'Read image'},
                                                       {'type': 'input_image', 'image_url': image}]}]
        payload = {'model': 'llm-default', 'input': input_items, 'max_output_tokens': 150}
        async with self.http.post('/llm/v1/responses/prompt_tokens', json=payload) as response:
            self.assertEqual((response.status, await response.json()), (200, {'count': 100}))
        self.assertEqual(self.counted[-1]['input'], input_items)
        self.assertEqual(self.counted[-1]['model'], 'text')
        self.assertEqual(self.responses, {})
        async with self.http.post('/llm/v1/responses', json=payload) as response:
            self.assertEqual(response.status, 200)
        input_items[0]['content'][1]['image_url'] = 'https://example.com/image.png'
        for path in ('responses/prompt_tokens', 'responses'):
            async with self.http.post('/llm/v1/' + path, json=payload) as response:
                self.assertEqual(response.status, 400)
                self.assertEqual((await response.json())['error']['code'], 'invalid_image_source')
        url = await self.serve(create_app(self.parent_config, replace(self.text_config, token_budget=1024)))
        async with ClientSession(base_url=url, headers={'Authorization': 'Bearer client'}) as client:
            for path in ('responses/prompt_tokens', 'responses'):
                async with client.post('/llm/v1/' + path, json=payload | {'input': 'rejected'}) as response:
                    self.assertEqual(response.status, 422)
                    self.assertEqual((await response.json())['error']['message'], 'Invalid multimodal input')

    async def test_weighted_background_leases_span_four_slots_and_cancellation(self):
        url = await self.serve(create_app(self.parent_config, replace(
            self.text_config, batch_cap=4, token_budget=1000)))
        async with ClientSession(base_url=url, headers={'Authorization': 'Bearer client'}) as http:
            identifiers = []
            for _ in range(4):
                async with http.post('/llm/v1/responses', json={
                        'model': 'llm-default', 'input': 'test', 'background': True}) as response:
                    self.assertEqual(response.status, 200)
                    identifiers.append((await response.json())['id'])
            async with http.get('/llm/api/mode') as response:
                state = await response.json()
                self.assertEqual((state['active']['chat'], state['reserved_tokens']), (4, 1000))
            async with http.post('/llm/v1/responses', json={'model': 'llm-default', 'input': 'test'}) as response:
                self.assertEqual(response.status, 429)
            async with http.post('/llm/v1/responses/' + identifiers.pop() + '/cancel') as response:
                self.assertEqual(response.status, 200)
            async with asyncio.timeout(2):
                while True:
                    async with http.get('/llm/api/mode') as response:
                        if (await response.json())['reserved_tokens'] == 750:
                            break
                    await asyncio.sleep(.01)
            before = len(self.counted)
            async with http.post('/llm/v1/responses', headers={'Authorization': 'Bearer foreign'}, json={
                    'model': 'llm-default', 'input': 'next', 'previous_response_id': identifiers[0]}) as response:
                self.assertEqual(response.status, 404)
            self.assertEqual(len(self.counted), before)
            async with http.post('/llm/v1/responses', json={
                    'model': 'llm-default', 'input': 'oversized'}) as response:
                self.assertEqual(response.status, 400)
            for identifier in identifiers:
                async with http.post('/llm/v1/responses/' + identifier + '/cancel') as response:
                    self.assertEqual(response.status, 200)
            self.assertTrue(all(payload['model'] == 'text' for payload in self.counted))

    async def test_explicit_routes_models_auth_and_tokenizer(self):
        for prefix, alias, model in [('', 'omni-default', 'omni'), ('/llm', 'llm-default', 'text')]:
            async with self.http.get(prefix + '/v1/models') as response:
                self.assertEqual((await response.json())['data'][0]['id'], alias)
            async with self.http.post(prefix + '/v1/chat/completions', json={
                    'model': alias, 'messages': [{'role': 'user', 'content': 'test'}]}) as response:
                self.assertEqual((await response.json())['model'], model)
        async with self.http.post('/llm/v1/chat/prompt_tokens', json={
                'model': 'llm-default', 'messages': [], 'max_tokens': 10}) as response:
            self.assertEqual(await response.json(), {'count': 17})
        async with self.http.put('/llm/api/mode', json={'mode': 'live'}) as response:
            self.assertEqual(response.status, 403)
        async with self.http.get('/llm/v1/realtime?duplex=1') as response:
            self.assertEqual(response.status, 404)
        async with self.http.post('/llm/v1/chat/completions', json={'model': 'omni-default'}) as response:
            self.assertEqual(response.status, 400)
        async with self.http.get('/llm/healthz', headers={'Authorization': 'Bearer wrong'}) as response:
            self.assertEqual(response.status, 401)

    async def test_llm_saturation_does_not_block_omni_and_cancel_releases(self):
        stream = await self.http.post('/llm/v1/chat/completions', json={'model': 'llm-default', 'stream': True})
        await stream.content.readline()
        async with self.http.post('/v1/chat/completions', json={'model': 'omni-default'}) as response:
            self.assertEqual(response.status, 200)
        async with self.http.post('/llm/v1/chat/completions', json={'model': 'llm-default'}) as response:
            self.assertEqual(response.status, 429)
        stream.close()
        await asyncio.wait_for(self.ended.wait(), 2)
        async with self.http.post('/llm/v1/chat/completions', json={'model': 'llm-default'}) as response:
            self.assertEqual(response.status, 200)

    async def test_responses_ownership_continuation_and_background_admission(self):
        async with self.http.post('/llm/v1/responses', json={
                'model': 'llm-default', 'input': 'hello', 'background': True,
                'request_id': 'caller-cannot-select-response-id'}) as response:
            self.assertEqual(response.status, 200)
            response_id = (await response.json())['id']
        self.assertRegex(response_id, r'^resp_[a-f0-9]{32}_[a-f0-9]{32}$')
        foreign = {'Authorization': 'Bearer foreign'}
        for method, path, options in [
                ('GET', '/llm/v1/responses/' + response_id, {}),
                ('POST', '/llm/v1/responses/' + response_id + '/cancel', {}),
                ('POST', '/llm/v1/responses', {'json': {'model': 'llm-default', 'input': 'next',
                                                        'previous_response_id': response_id}})]:
            async with self.http.request(method, path, headers=foreign, **options) as response:
                self.assertEqual(response.status, 404)
        async with self.http.post('/llm/v1/responses', json={'model': 'llm-default', 'input': 'busy'}) as response:
            self.assertEqual(response.status, 429)
        async with self.http.post('/llm/v1/responses/' + response_id + '/cancel') as response:
            self.assertEqual((await response.json())['status'], 'cancelled')
        async with asyncio.timeout(2):
            while True:
                async with self.http.get('/llm/api/mode') as response:
                    if (await response.json())['active']['chat'] == 0:
                        break
                await asyncio.sleep(.02)
        async with self.http.post('/llm/v1/responses', json={'model': 'llm-default', 'input': 'next',
                                                          'previous_response_id': response_id}) as response:
            self.assertEqual(response.status, 200)
        async with self.http.post('/llm/v1/responses', json={'model': 'llm-default', 'input': 'test',
                                                          'tools': [{'type': 'web_search'}]}) as response:
            self.assertEqual(response.status, 400)

    async def test_responses_sse_is_incremental_and_disconnect_cancels(self):
        response = await self.http.post('/llm/v1/responses', json={
            'model': 'llm-default', 'input': 'hello', 'stream': True})
        async with asyncio.timeout(1):
            data = bytearray()
            while b'"delta":"hello"' not in data:
                data.extend(await response.content.readany())
        self.assertIn(b'event: response.output_text.delta', data)
        response.close()
        await asyncio.wait_for(self.ended.wait(), 2)

    async def test_background_stream_disconnect_keeps_quota_until_cancelled(self):
        import json
        stream = await self.http.post('/llm/v1/responses', json={
            'model': 'llm-default', 'input': 'hello', 'stream': True, 'background': True})
        response_id = None
        async with asyncio.timeout(1):
            while response_id is None:
                line = await stream.content.readline()
                if line.startswith(b'data: '):
                    response_id = json.loads(line[6:])['response']['id']
        stream.close()
        await asyncio.wait_for(self.ended.wait(), 2)
        async with self.http.post('/llm/v1/responses', json={'model': 'llm-default', 'input': 'busy'}) as response:
            self.assertEqual(response.status, 429)
        async with self.http.post('/llm/v1/responses/' + response_id + '/cancel') as response:
            self.assertEqual((await response.json())['status'], 'cancelled')

    async def test_default_thinking_budget_applies_only_to_text_backend(self):
        url = await self.serve(create_app(
            self.parent_config,
            replace(self.text_config, default_thinking_token_budget=16384),
        ))
        async with ClientSession(base_url=url, headers={'Authorization': 'Bearer client'}) as http:
            async with http.post('/v1/chat/completions', json={
                'model': 'omni-default', 'messages': [{'role': 'user', 'content': 'hi'}],
            }) as response:
                self.assertEqual(response.status, 200)
            self.assertNotIn('thinking_token_budget', self.chat_payloads[-1][1])

            async with http.post('/llm/v1/chat/completions', json={
                'model': 'llm-default', 'messages': [{'role': 'user', 'content': 'hi'}],
            }) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(self.chat_payloads[-1][1]['thinking_token_budget'], 16384)

            async with http.post('/llm/v1/responses', json={
                'model': 'llm-default', 'input': 'hi',
            }) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(self.response_payloads[-1][1]['thinking_token_budget'], 16384)

            async with http.post('/llm/v1/responses', json={
                'model': 'llm-default', 'input': 'hi', 'thinking_token_budget': 0,
            }) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(self.response_payloads[-1][1]['thinking_token_budget'], 0)

            async with http.post('/llm/v1/responses', json={
                'model': 'llm-default', 'input': 'hi', 'reasoning': {'effort': 'none'},
            }) as response:
                self.assertEqual(response.status, 200)
            self.assertNotIn('thinking_token_budget', self.response_payloads[-1][1])


class ExtraModelAliasRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def serve(self, app):
        runner = web.AppRunner(app, handler_cancellation=True, access_log=None)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, '127.0.0.1', 0).start()
        return 'http://127.0.0.1:' + str(runner.addresses[0][1])

    async def asyncSetUp(self):
        self.exposed_models = [
            {'id': 'text', 'root': '/models/text'},
            {'id': 'llm-zh-chat', 'root': '/models/lora/rev'},
        ]
        self.chat_models = []
        self.tokenize_models = []
        self.response_models = []
        self.admission_calls = []
        self.responses = {}
        self.release = asyncio.Event()
        self.stream_ended = asyncio.Event()

        async def omni_models(request):
            return web.json_response({'data': [{'id': 'omni', 'root': '/models/omni'}]})

        async def llm_models(request):
            return web.json_response({'data': list(self.exposed_models)})

        async def llm_chat(request):
            payload = await request.json()
            self.chat_models.append(payload['model'])
            if payload.get('stream'):
                response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
                await response.prepare(request)
                try:
                    await response.write(b'data: {"choices":[]}\n\n')
                    await self.release.wait()
                    await response.write(b'data: [DONE]\n\n')
                finally:
                    self.stream_ended.set()
                return response
            return web.json_response({'model': payload['model'], 'choices': [{'message': {'content': payload['model']}}]})

        async def llm_tokenize(request):
            payload = await request.json()
            self.tokenize_models.append(payload['model'])
            return web.json_response({'count': 23, 'tokens': [1, 2]})

        async def llm_responses(request):
            if request.method == 'POST' and 'response_id' not in request.match_info:
                payload = await request.json()
                self.response_models.append(payload['model'])
                response_id = payload['request_id']
                data = {
                    'id': response_id, 'object': 'response', 'model': payload['model'],
                    'status': 'in_progress' if payload.get('background') else 'completed', 'output': [],
                }
                self.responses[response_id] = data
                return web.json_response(data)
            data = self.responses.get(request.match_info['response_id'])
            if data is None:
                return web.Response(status=404)
            if request.method == 'POST':
                data['status'] = 'cancelled'
            return web.json_response(data)

        async def llm_admission(request):
            payload = await request.json()
            self.admission_calls.append((request.path, payload['model']))
            reserved = 300 if payload.get('input') == 'large' or payload.get('messages') == [{'role': 'user', 'content': 'large'}] else 250
            return web.json_response({'input_tokens': 100, 'reserved_tokens': reserved, 'sequences': 1})

        omni_app = web.Application()
        omni_app.add_routes([web.get('/health', lambda r: web.Response()), web.get('/v1/models', omni_models)])
        llm_app = web.Application()
        llm_app.add_routes([
            web.get('/health', lambda r: web.Response()),
            web.get('/v1/models', llm_models),
            web.post('/v1/chat/completions', llm_chat),
            web.post('/tokenize', llm_tokenize),
            web.post('/v1/responses', llm_responses),
            web.get('/v1/responses/{response_id}', llm_responses),
            web.post('/v1/responses/{response_id}/cancel', llm_responses),
            web.post('/internal/admission/chat', llm_admission),
            web.post('/internal/admission/responses', llm_admission),
        ])

        omni_url = await self.serve(omni_app)
        llm_url = await self.serve(llm_app)
        parent = Config(
            engine_url=omni_url, model='omni', checkpoint='omni',
            tokens=('client',), admin_token='admin', model_root='/models/omni', queue_timeout=0.05,
        )
        text = replace(
            parent, engine_url=llm_url, model='text', checkpoint='text',
            model_root='/models/text', model_alias='llm-default', public_prefix='/llm',
            realtime_enabled=False, prompt_tokens_path='/tokenize',
            batch_cap=2, token_budget=500, queue_timeout=0.05,
            model_aliases=(('llm-zh-chat', 'llm-zh-chat'),),
        )
        self.url = await self.serve(create_app(parent, text))
        self.http = ClientSession(base_url=self.url, headers={'Authorization': 'Bearer client'})
        self.addAsyncCleanup(self.http.close)

    async def test_models_health_routing_shared_budget_and_unknown_alias(self):
        async with self.http.get('/llm/v1/models') as response:
            self.assertEqual(response.status, 200)
            models = (await response.json())['data']
        self.assertEqual([(item['id'], item['root']) for item in models], [
            ('llm-default', 'llm-default'),
            ('llm-zh-chat', 'llm-zh-chat'),
        ])

        async with self.http.get('/llm/healthz') as response:
            self.assertEqual(response.status, 200)
            health = await response.json()
        self.assertEqual(health['alias'], 'llm-default')
        self.assertEqual(health['extra_aliases'], ['llm-zh-chat'])

        # Missing LoRA module in upstream omits extra alias without failing base readiness
        saved_models = list(self.exposed_models)
        self.exposed_models = [{'id': 'text', 'root': '/models/text'}]
        async with self.http.get('/llm/v1/models') as response:
            self.assertEqual(response.status, 200)
            self.assertEqual([item['id'] for item in (await response.json())['data']], ['llm-default'])
        async with self.http.get('/llm/healthz') as response:
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())['extra_aliases'], [])
        self.exposed_models = saved_models

        # Routing across all 4 endpoints for both llm-default and llm-zh-chat
        for alias, expected in (('llm-default', 'text'), ('llm-zh-chat', 'llm-zh-chat')):
            async with self.http.post('/llm/v1/chat/completions', json={
                'model': alias, 'messages': [{'role': 'user', 'content': 'hi'}],
            }) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())['model'], expected)
            self.assertEqual(self.chat_models[-1], expected)

            async with self.http.post('/llm/v1/chat/prompt_tokens', json={
                'model': alias, 'messages': [{'role': 'user', 'content': 'hi'}],
            }) as response:
                self.assertEqual((response.status, await response.json()), (200, {'count': 23}))
            self.assertEqual(self.tokenize_models[-1], expected)

            async with self.http.post('/llm/v1/responses', json={
                'model': alias, 'input': 'hi',
            }) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual((await response.json())['model'], expected)
            self.assertEqual(self.response_models[-1], expected)

            async with self.http.post('/llm/v1/responses/prompt_tokens', json={
                'model': alias, 'input': 'hi',
            }) as response:
                self.assertEqual((response.status, await response.json()), (200, {'count': 100}))
            self.assertEqual(self.admission_calls[-1], ('/internal/admission/responses', expected))

        # Unknown alias returns 400 model_not_found listing accepted aliases on all 4 endpoints
        for path, body in (
            ('/llm/v1/chat/completions', {'model': 'bad-alias', 'messages': [{'role': 'user', 'content': 'hi'}]}),
            ('/llm/v1/chat/prompt_tokens', {'model': 'bad-alias', 'messages': [{'role': 'user', 'content': 'hi'}]}),
            ('/llm/v1/responses', {'model': 'bad-alias', 'input': 'hi'}),
            ('/llm/v1/responses/prompt_tokens', {'model': 'bad-alias', 'input': 'hi'}),
        ):
            async with self.http.post(path, json=body) as response:
                self.assertEqual(response.status, 400)
                err = (await response.json())['error']
                self.assertEqual(err['code'], 'model_not_found')
                self.assertIn('llm-default', err['message'])
                self.assertIn('llm-zh-chat', err['message'])

        # Shared admission slots and token budget across llm-default and llm-zh-chat
        stream = await self.http.post('/llm/v1/chat/completions', json={
            'model': 'llm-default', 'stream': True, 'messages': [{'role': 'user', 'content': 'large'}],
        })
        await stream.content.readline()
        # 300/500 tokens reserved by llm-default; 250-token llm-zh-chat request exceeds remaining 200 budget
        async with self.http.post('/llm/v1/responses', json={
            'model': 'llm-zh-chat', 'input': 'hi',
        }) as response:
            self.assertEqual(response.status, 429)
        stream.close()
        await asyncio.wait_for(self.stream_ended.wait(), 2)

        # Hold slot 1 via llm-default chat stream and slot 2 via llm-zh-chat background response
        self.stream_ended.clear()
        stream = await self.http.post('/llm/v1/chat/completions', json={
            'model': 'llm-default', 'stream': True, 'messages': [{'role': 'user', 'content': 'hi'}],
        })
        await stream.content.readline()
        async with self.http.post('/llm/v1/responses', json={
            'model': 'llm-zh-chat', 'input': 'hi', 'background': True,
        }) as response:
            self.assertEqual(response.status, 200)
            resp_id = (await response.json())['id']
        async with self.http.get('/llm/api/mode') as response:
            state = await response.json()
            self.assertEqual((state['active']['chat'], state['reserved_tokens']), (2, 500))
        async with self.http.post('/llm/v1/chat/completions', json={
            'model': 'llm-zh-chat', 'messages': [{'role': 'user', 'content': 'hi'}],
        }) as response:
            self.assertEqual(response.status, 429)

        stream.close()
        await asyncio.wait_for(self.stream_ended.wait(), 2)
        async with self.http.post('/llm/v1/responses/' + resp_id + '/cancel') as response:
            self.assertEqual(response.status, 200)
        async with asyncio.timeout(2):
            while True:
                async with self.http.get('/llm/api/mode') as response:
                    if (await response.json())['reserved_tokens'] == 0:
                        break
                await asyncio.sleep(0.01)
