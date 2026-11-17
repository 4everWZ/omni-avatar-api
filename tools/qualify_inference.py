"""Responses + multi-turn Omni + Ava functional qualification, without an Agent."""
import argparse
import asyncio
import base64
import io
import json
from pathlib import Path
from time import monotonic
from PIL import Image

from qualify_workstation import Qualification
from long_context import inventory_document


def output_text(response):
    return ''.join(part.get('text', '') for item in response.get('output', [])
                   if item.get('type') == 'message' for part in item.get('content', [])
                   if part.get('type') == 'output_text')


class InferenceQualification(Qualification):
    def __init__(self, args):
        super().__init__(args)
        self.report['scope'] = 'decoupled_responses_omni_avatar_functional'

    async def responses(self, input, **options):
        body = {'model': 'llm-default', 'input': input, 'max_output_tokens': 256,
                'chat_template_kwargs': {'enable_thinking': False}, **options}
        return await self.json('POST', '/llm/v1/responses', json=body)

    async def ordinary_request(self):
        reply = await self.json('POST', '/llm/v1/responses', other=True, json={
            'model': 'llm-default', 'input': '只回答数字：2+3等于几？', 'store': False,
            'max_output_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}})
        assert output_text(reply).strip() == '5'

    async def vision_check(self):
        encoded = io.BytesIO()
        Image.new('RGB', (192, 192), 'red').save(encoded, 'PNG')
        reply = await self.json('POST', '/v1/chat/completions', json={
            'model': 'omni-default', 'max_tokens': 64,
            'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': '这张纯色图片是什么颜色？只回答颜色。'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                                                  base64.b64encode(encoded.getvalue()).decode()}}]}]})
        answer = reply['choices'][0]['message']['content'].lower()
        assert '红' in answer or 'red' in answer
        self.check('single_gpu_omni_vision_color_correct')

    async def omni_context_checks(self):
        def candidate(records):
            messages = [{'role': 'system', 'content': '准确使用本次对话历史回答问题。'},
                        {'role': 'user', 'content': '本项目的代号是蓝鲸5738，请记住。'},
                        {'role': 'assistant', 'content': '收到，本项目代号蓝鲸5738。'}]
            for turn in range(8):
                rows = '\n'.join(f'任务{turn:02d}-{i:03d}：工位{(i*7+turn)%91}，零件数量{11+i*3}，'
                                 f'状态已登记；批次T{turn:02d}{i:04d}。' for i in range(records))
                messages.extend([{'role': 'user', 'content': f'第{turn+1}轮的任务记录如下：\n' + rows},
                                 {'role': 'assistant', 'content': f'第{turn+1}轮任务已登记。'}])
            messages.append({'role': 'user', 'content': '最初约定的项目代号是什么？只回答代号。'})
            return {'model': 'omni-default', 'messages': messages, 'max_tokens': 128,
                    'chat_template_kwargs': {'enable_thinking': False}}
        low, high, body, count = 1, 64, None, 0
        while low <= high:
            middle = (low + high) // 2
            current = candidate(middle)
            measured = (await self.json('POST', '/v1/chat/prompt_tokens', json=current))['count']
            if measured <= 7936:
                body, count, low = current, measured, middle + 1
            else:
                high = middle - 1
        assert body and count > 7400
        reply = await self.json('POST', '/v1/chat/completions', json=body)
        assert '5738' in reply['choices'][0]['message']['content']
        self.check('omni_near_full_multi_turn_recall', input_tokens=count, messages=len(body['messages']))
        overflow = candidate(high + 8)
        count = (await self.json('POST', '/v1/chat/prompt_tokens', json=overflow))['count']
        assert count + 128 > 8192
        response = await self.request('POST', '/v1/chat/completions', json=overflow, expected=400)
        response.release()
        self.check('omni_over_window_rejected', input_tokens=count, status=400)

    async def chat_checks(self):
        for prefix, window in [('', 8192), ('/llm', 262144)]:
            health = await self.json('GET', prefix + '/healthz')
            models = await self.json('GET', prefix + '/v1/models')
            assert health['checkpoint_root_verified']
            assert models['data'][0]['max_model_len'] == window
            self.check('backend_identity', backend=prefix or 'omni', checkpoint=health['checkpoint'], window=window)
        messages = [{'role': 'user', 'content': '记住本次项目代号是蓝鲸5738，只回答收到。'},
                    {'role': 'assistant', 'content': '收到。'},
                    {'role': 'user', 'content': '项目部署在两张3090上，只回答收到。'},
                    {'role': 'assistant', 'content': '收到。'},
                    {'role': 'user', 'content': '本次项目代号是什么？只回答代号。'}]
        reply = await self.json('POST', '/v1/chat/completions', json={
            'model': 'omni-default', 'messages': messages, 'max_tokens': 128,
            'chat_template_kwargs': {'enable_thinking': False}})
        assert '5738' in reply['choices'][0]['message']['content']
        self.check('omni_multi_turn_history')
        await self.vision_check()
        await self.omni_context_checks()

        first = await self.responses('记住项目代号是蓝鲸5738。只回答收到。')
        for method, path, options in [
            ('GET', '/llm/v1/responses/' + first['id'], {}),
            ('POST', '/llm/v1/responses/' + first['id'] + '/cancel', {}),
            ('POST', '/llm/v1/responses', {'json': {'model': 'llm-default', 'input': '继续',
                                                  'previous_response_id': first['id']}})]:
            response = await self.request(method, path, other=True, expected=404, **options)
            response.release()
        second = await self.responses('项目代号是什么？只回答代号。', previous_response_id=first['id'])
        assert '5738' in output_text(second)
        self.check('responses_owned_continuation')

        schema = {'type': 'object', 'properties': {'sum': {'type': 'integer', 'const': 5}},
                  'required': ['sum'], 'additionalProperties': False}
        structured = await self.responses('Return JSON with integer sum equal to 2+3.',
            text={'format': {'type': 'json_schema', 'name': 'sum', 'strict': True, 'schema': schema}})
        assert json.loads(output_text(structured)) == {'sum': 5}
        self.check('responses_json_schema')

        tools = [{'type': 'function', 'name': 'lookup_status', 'description': 'Return a ticket status.',
                  'parameters': {'type': 'object', 'properties': {'ticket': {'type': 'string'}},
                                 'required': ['ticket'], 'additionalProperties': False}, 'strict': True}]
        call = await self.responses('Use lookup_status to look up ticket T-57.', tools=tools,
                                    tool_choice={'type': 'function', 'name': 'lookup_status'})
        function = next(item for item in call['output'] if item['type'] == 'function_call')
        assert function['name'] == 'lookup_status' and json.loads(function['arguments'])['ticket'] == 'T-57'
        follow = await self.responses([{'type': 'function_call_output', 'call_id': function['call_id'],
                                       'output': '{"status":"ok"}'}], previous_response_id=call['id'],
                                     tools=tools, instructions='Summarize the returned status without another tool call.')
        assert 'ok' in output_text(follow).lower()
        self.check('responses_function_call_roundtrip', execution='literal_test_fixture_only')

        start = monotonic()
        deltas, text, first_delta, completed = 0, '', None, False
        async with await self.request('POST', '/llm/v1/responses', json={
            'model': 'llm-default', 'input': 'Write the integers from 1 to 80, one per line. Do not omit any.',
            'stream': True, 'store': False, 'max_output_tokens': 512,
            'chat_template_kwargs': {'enable_thinking': False}}) as stream:
            async for line in stream.content:
                if line.startswith(b'data: '):
                    event = json.loads(line[6:])
                    if event['type'] == 'response.output_text.delta':
                        deltas += 1
                        text += event['delta']
                        first_delta = first_delta or monotonic()
                    elif event['type'] == 'response.completed':
                        completed = True
        assert completed and deltas > 2 and '80' in text
        self.check('responses_incremental_stream', delta_events=deltas,
                   first_delta_seconds=first_delta - start, total_seconds=monotonic() - start)

        history = [{'role': 'user', 'content': '记住：项目代号是蓝鲸5738。后续库存资料不改变代号。'}]
        for i in range(24):
            history.extend([{'role': 'assistant', 'content': '收到，继续登记。'},
                            {'role': 'user', 'content': f'批次{i}资料：\n' + inventory_document(10)}])
        seed = await self.responses(history, max_output_tokens=32, instructions='Only acknowledge receipt.')
        later = []
        for i in range(24, 48):
            later.extend([{'role': 'user', 'content': f'批次{i}资料：\n' + inventory_document(10)},
                          {'role': 'assistant', 'content': '已登记。'}])
        later.append({'role': 'user', 'content': '最初设定的项目代号是什么？只回答代号。'})
        remembered = await self.responses(later, previous_response_id=seed['id'])
        assert '5738' in output_text(remembered)
        self.check('responses_long_multi_turn_history', input_tokens=remembered.get('usage', {}).get('input_tokens'),
                   output=output_text(remembered))

    async def admission_checks(self):
        job = await self.responses('Return the required JSON array of ones.',
            background=True, max_output_tokens=8192, text={'format': {
                'type': 'json_schema', 'name': 'long_array', 'strict': True,
                'schema': {'type': 'array', 'items': {'type': 'integer', 'const': 1},
                           'minItems': 2048, 'maxItems': 2048}}})
        assert job['status'] in {'queued', 'in_progress'}
        self.background_id = job['id']
        try:
            await self.request('POST', '/llm/v1/responses', expected=429,
                               json={'model': 'llm-default', 'input': 'Should queue out.'})
            reply = await self.json('POST', '/v1/chat/completions', json={
                'model': 'omni-default', 'messages': [{'role': 'user', 'content': '只回答数字：2+3等于几？'}],
                'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}})
            assert reply['choices'][0]['message']['content'].strip() == '5'
            cancelled = await self.json('POST', '/llm/v1/responses/' + job['id'] + '/cancel')
            assert cancelled['status'] == 'cancelled'
            async with asyncio.timeout(10):
                while (await self.json('GET', '/llm/api/mode'))['active']['chat']:
                    await asyncio.sleep(.1)
            await self.ordinary_request()
            self.check('background_cancel_and_independent_admission')
        finally:
            async with self.http.post(self.args.gateway.rstrip('/') + '/llm/v1/responses/' + job['id'] + '/cancel',
                    headers={'Authorization': 'Bearer ' + self.token}) as response:
                await response.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--wav', default=str(Path(__file__).with_name('fixtures') / 'question.wav'))
    parser.add_argument('--report', default='tmp/inference-functional.json')
    args = parser.parse_args()
    args.avatar_only = False
    asyncio.run(InferenceQualification(args).run())
