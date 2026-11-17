import asyncio
from dataclasses import replace
import unittest

from aiohttp import ClientSession, web
from gateway import Config, create_app


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

        async def engine(model):
            async def models(request):
                return web.json_response({'data': [{'id': model, 'root': '/models/' + model}]})

            async def chat(request):
                payload = await request.json()
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

            app = web.Application()
            app.add_routes([web.get('/health', lambda r: web.Response()),
                            web.get('/v1/models', models), web.post('/v1/chat/completions', chat),
                            web.post('/tokenize', tokenize), web.post('/v1/responses', responses),
                            web.get('/v1/responses/{response_id}', responses),
                            web.post('/v1/responses/{response_id}/cancel', responses)])
            return await self.serve(app)

        config = Config(engine_url=await engine('omni'), model='omni', checkpoint='omni',
                        tokens=('client', 'foreign'), admin_token='admin', model_root='/models/omni', queue_timeout=.1)
        text = replace(config, engine_url=await engine('text'), model='text', checkpoint='text',
                       model_root='/models/text', model_alias='llm-default', public_prefix='/llm',
                       realtime_enabled=False, prompt_tokens_path='/tokenize')
        self.url = await self.serve(create_app(config, text))
        self.http = ClientSession(base_url=self.url, headers={'Authorization': 'Bearer client'})
        self.addAsyncCleanup(self.http.close)

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
