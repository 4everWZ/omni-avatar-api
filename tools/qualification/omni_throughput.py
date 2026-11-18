"""Measure Omni Chat token throughput, idle and alongside native voice/portrait."""
import argparse
import asyncio
import json
from pathlib import Path
from time import monotonic

from .concurrency import SHORT_PROMPT, stream_group
from .portrait import PortraitQualification


async def stream_chat(http, gateway, token, concurrency=1, max_tokens=256, thinking=True):
    origin = monotonic()
    async def one(index):
        first = last = usage = None
        deltas, output = 0, []
        channels = {'content': 0, 'reasoning': 0}
        first_content = first_reasoning = None
        async with http.post(gateway.rstrip('/') + '/v1/chat/completions',
                headers={'Authorization': 'Bearer ' + token}, json={
                    'model': 'omni-default', 'messages': [{'role': 'user', 'content': SHORT_PROMPT}],
                    'max_tokens': max_tokens, 'temperature': 0, 'stream': True,
                    'stream_options': {'include_usage': True},
                    'chat_template_kwargs': {'enable_thinking': thinking}}) as response:
            assert response.status == 200, ('omni_chat', response.status)
            async for line in response.content:
                if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                    continue
                event = json.loads(line[6:])
                assert 'error' not in event, 'Omni stream returned an error'
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    content = delta.get('content')
                    reasoning = delta.get('reasoning') or delta.get('reasoning_content')
                    if content or reasoning:
                        now = monotonic() - origin
                        first = now if first is None else first
                        last = now
                        deltas += 1
                        if content:
                            output.append(content)
                            channels['content'] += 1
                            first_content = now if first_content is None else first_content
                        if reasoning:
                            channels['reasoning'] += 1
                            first_reasoning = now if first_reasoning is None else first_reasoning
        terminal = monotonic() - origin
        assert first is not None and last > first and usage, 'Missing timed content or final token usage'
        tokens = usage['completion_tokens']
        assert tokens > 1
        return {'index': index, 'usage': usage, 'first_delta_seconds': first,
                'last_delta_seconds': last, 'terminal_seconds': terminal, 'delta_events': deltas,
                'channel_events': channels, 'first_content_seconds': first_content,
                'first_reasoning_seconds': first_reasoning,
                'end_to_end_output_tps': tokens / terminal,
                'output_interval_tps': (tokens - 1) / (last - first),
                'output_excerpt': ''.join(output)[:96]}
    rows = await asyncio.gather(*(one(i) for i in range(concurrency)))
    overlap = min(r['last_delta_seconds'] for r in rows) - max(r['first_delta_seconds'] for r in rows)
    # Record actual overlap across deployment profiles.
    return {'requests': rows, 'concurrency': concurrency, 'outputs_overlap': overlap > 0,
            'output_intervals_overlap_seconds': max(0, overlap),
            'aggregate_end_to_end_output_tps': sum(r['usage']['completion_tokens'] for r in rows)
                / max(r['terminal_seconds'] for r in rows)}


class OmniThroughputQualification(PortraitQualification):
    async def chat_checks(self):
        health = await self.json('GET', '/healthz')
        models = await self.json('GET', '/v1/models')
        assert health['checkpoint_root_verified'] and models['data'][0]['max_model_len'] == 8192
        self.report.update(scope='omni_text_chat_tps_with_native_voice_portrait_and_four_llm',
            model=health['configured_model'], checkpoint=health['checkpoint'],
            chat_window=8192, sampling_temperature=0, maximum_output_tokens=256,
            thinking_enabled=self.args.thinking == 'enabled', thinking_token_budget=None,
            realtime_window=self.args.realtime_extra_body,
            token_metric='Chat usage.completion_tokens; not native speech token rate')
        self.thinking = self.args.thinking == 'enabled'
        await stream_chat(self.http, self.args.gateway, self.other, max_tokens=32, thinking=self.thinking)
        self.check('omni_idle_single_chat_throughput', **await stream_chat(
            self.http, self.args.gateway, self.other, thinking=self.thinking))
        self.check('omni_idle_two_chat_throughput', **await stream_chat(
            self.http, self.args.gateway, self.other, concurrency=2, thinking=self.thinking))

    async def ordinary_request(self):
        front, back = await asyncio.gather(
            stream_chat(self.http, self.args.gateway, self.other, thinking=self.thinking),
            stream_group(self.http, self.args.gateway, self.other, request_options=[
                {'chat_template_kwargs': {'enable_thinking': self.thinking}} for _ in range(4)]))
        self.check('omni_single_chat_while_native_portrait_and_four_llm_active', **front)
        self.check('four_simultaneous_responses_while_neural_avatar_active', **back)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--seconds', type=int, default=90)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument('--reference')
    reference.add_argument('--avatar-id')
    parser.add_argument('--sliding-window-mode', choices=('basic', 'context'), default='basic')
    parser.add_argument('--thinking', choices=('enabled', 'disabled'), default='enabled')
    parser.add_argument('--wav', default=str(Path(__file__).resolve().parents[2] / 'evaluations/fixtures/question.wav'))
    parser.add_argument('--report', default='tmp/omni-throughput.json')
    args = parser.parse_args()
    args.avatar_only, args.concurrent_llm = False, 4
    args.realtime_extra_body = {'sliding_window_mode': args.sliding_window_mode}
    if args.sliding_window_mode == 'basic':
        args.realtime_extra_body.update(basic_window_high_tokens=8000, basic_window_low_tokens=6000)
    else:
        args.realtime_extra_body.update(context_max_units=24, context_previous_max_tokens=500)
    asyncio.run(OmniThroughputQualification(args).run())
