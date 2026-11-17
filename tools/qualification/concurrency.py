"""Qualify real four-way Responses decoding and shared-token admission."""
import argparse
import asyncio
import json
from pathlib import Path
import re
from time import monotonic

from aiohttp import ClientSession, ClientTimeout
from .workstation import read_env


async def stream_group(http, gateway, token, inputs=None):
    """Require all four output intervals to overlap, not just HTTP submission."""
    origin = monotonic()
    inputs = inputs or ['Write the integers from 1 to 200, one integer per line.'] * 4
    async def one(index, text):
        first, last, terminal, deltas = None, None, None, 0
        output = []
        usage = None
        async with http.post(gateway + '/llm/v1/responses',
                headers={'Authorization': 'Bearer ' + token}, json={
                    'model': 'llm-default', 'input': text, 'stream': True, 'store': False,
                    'max_output_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}}) as response:
            assert response.status == 200, ('stream', index, response.status)
            async for line in response.content:
                if not line.startswith(b'data: '):
                    continue
                event = json.loads(line[6:])
                if event['type'] == 'response.output_text.delta':
                    now = monotonic() - origin
                    first = now if first is None else first
                    last = now
                    deltas += 1
                    output.append(event['delta'])
                elif event['type'] == 'response.completed':
                    terminal = monotonic() - origin
                    usage = event['response'].get('usage')
        assert first is not None and last > first and terminal and deltas > 2
        return {'index': index, 'first_delta_seconds': first, 'last_delta_seconds': last,
                'terminal_seconds': terminal, 'delta_events': deltas, 'usage': usage,
                'output_excerpt': ''.join(output)[:96]}
    results = await asyncio.gather(*(one(i, text) for i, text in enumerate(inputs)))
    overlap = min(item['last_delta_seconds'] for item in results) - max(item['first_delta_seconds'] for item in results)
    assert overlap > 0, results
    return {'requests': results, 'four_output_intervals_overlap_seconds': overlap}


class ConcurrencyQualification:
    def __init__(self, args):
        self.args = args
        self.env = read_env(args.env)
        self.token = self.env['OMNI_GATEWAY_TOKEN']
        self.report = {'state': 'running', 'checks': [], 'samples': []}
        self.stop = asyncio.Event()
        self.background = None

    def save(self):
        path = Path(self.args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report, indent=2, ensure_ascii=False))

    async def json(self, method, path, expected=200, **options):
        async with self.http.request(method, self.args.gateway + path, **options) as response:
            assert response.status == expected, (path, response.status)
            return await response.json()

    async def sample(self):
        while not self.stop.is_set():
            state = await self.json('GET', '/llm/api/mode')
            assert state['reserved_tokens'] <= state['token_budget']
            sample = {'time': monotonic(), 'admission': state}
            if self.args.engine_metrics:
                async with self.http.get(self.args.engine_metrics) as response:
                    assert response.status == 200
                    text = await response.text()
                values = re.findall(r'^vllm:num_requests_running(?:\{[^\n]*\})?\s+([0-9.e+]+)', text, re.M)
                sample['engine_running'] = sum(float(value) for value in values)
            self.report['samples'].append(sample)
            try:
                await asyncio.wait_for(self.stop.wait(), .1)
            except TimeoutError:
                pass

    async def run(self):
        self.save()
        headers = {'Authorization': 'Bearer ' + self.token}
        async with ClientSession(headers=headers, timeout=ClientTimeout(total=1200)) as self.http:
            sampler = asyncio.create_task(self.sample())
            try:
                model = await self.json('GET', '/llm/v1/models')
                assert model['data'][0]['max_model_len'] == 262144
                self.report['checks'].append({'single_request_window': 262144})
                self.report['checks'].append({'four_short_requests': await stream_group(
                    self.http, self.args.gateway, self.token)})
                print('four_short_requests_simultaneously_decoding', flush=True)

                # One sizeable inline history plus three short generations.
                history = ('库存记录：编号A1729，数量42，状态已登记。\n' * 1500)
                self.report['checks'].append({'long_short_mix': await stream_group(
                    self.http, self.args.gateway, self.token,
                    [history + '\nWrite the integers from 1 to 200, one per line.'] +
                    ['Write the integers from 1 to 200, one per line.'] * 3)})
                print('long_short_mix_simultaneously_decoding', flush=True)

                seed = await self.json('POST', '/llm/v1/responses', json={
                    'model': 'llm-default', 'input': 'Remember the project code ALPHA1729. Say received.',
                    'max_output_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}})
                continued = await self.json('POST', '/llm/v1/responses', json={
                    'model': 'llm-default', 'input': 'What is the project code? Only return the code.',
                    'previous_response_id': seed['id'], 'max_output_tokens': 32,
                    'chat_template_kwargs': {'enable_thinking': False}})
                output = ''.join(part.get('text', '') for item in continued['output']
                                 for part in item.get('content', []))
                assert 'ALPHA1729' in output
                self.report['checks'].append({'owned_continuation': True, 'usage': continued['usage']})

                # Reserve most of the budget with a low-input, long-running background job.
                job = await self.json('POST', '/llm/v1/responses', json={
                    'model': 'llm-default', 'input': 'Write all integers from 1 to 100000, one per line.',
                    'background': True, 'max_output_tokens': 200000,
                    'chat_template_kwargs': {'enable_thinking': False}})
                self.background = job['id']
                state = await self.json('GET', '/llm/api/mode')
                assert 200000 <= state['reserved_tokens'] < 201000 and state['active']['chat'] == 1
                blocked = await self.json('POST', '/llm/v1/responses', expected=429, json={
                    'model': 'llm-default', 'input': 'Write integers from 1 to 100000.',
                    'max_output_tokens': 100000, 'chat_template_kwargs': {'enable_thinking': False}})
                assert blocked['error']['code'] == 'gateway_busy'
                # A short request can still share the remaining token budget.
                short = await self.json('POST', '/llm/v1/chat/completions', json={
                    'model': 'llm-default', 'messages': [{'role': 'user', 'content': 'Only answer 5.'}],
                    'max_tokens': 32, 'chat_template_kwargs': {'enable_thinking': False}})
                assert '5' in short['choices'][0]['message']['content']
                await self.json('POST', '/llm/v1/responses/' + self.background + '/cancel')
                self.background = None
                async with asyncio.timeout(10):
                    while (await self.json('GET', '/llm/api/mode'))['reserved_tokens']:
                        await asyncio.sleep(.1)
                self.report['checks'].append({'token_pressure_returns_429_with_free_sequence_slots': True,
                    'short_chat_shares_remaining_budget': True, 'cancel_releases_tokens': True})
                print('weighted_budget_and_cancel_release', flush=True)
                if self.args.engine_metrics:
                    peak = max(sample.get('engine_running', 0) for sample in self.report['samples'])
                    assert peak == 4, peak
                    self.report['engine_peak_running_sequences'] = peak
                self.report['state'] = 'passed'
            except BaseException as error:
                self.report.update(state='failed', failure=type(error).__name__ + ': ' + str(error))
                raise
            finally:
                if self.background:
                    await self.json('POST', '/llm/v1/responses/' + self.background + '/cancel')
                self.stop.set()
                await sampler
                self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--engine-metrics', help='Private engine /metrics URL; never expose publicly')
    parser.add_argument('--report', default='tmp/llm-concurrency.json')
    asyncio.run(ConcurrencyQualification(parser.parse_args()).run())
