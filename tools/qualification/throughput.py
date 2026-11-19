"""Single/four-request TPS, both idle and during native speech and CPU-rendered media."""
import argparse
import asyncio
from pathlib import Path

from .concurrency import SHORT_PROMPT, stream_group, stream_requests
from .inference import InferenceQualification


class ThroughputQualification(InferenceQualification):
    async def chat_checks(self):
        await self.ordinary_request()
        await self.vision_check()
        health = await self.json('GET', '/llm/healthz')
        models = await self.json('GET', '/llm/v1/models')
        assert health['checkpoint_root_verified']
        assert models['data'][0]['max_model_len'] == 262144
        self.report.update(scope='single_and_four_tps_with_native_cpu_avatar',
            model=health['configured_model'], checkpoint=health['checkpoint'],
            sampling_temperature=0, output_tokens_per_request=256,
            thinking_enabled=self.args.thinking == 'enabled', thinking_token_budget=None,
            regression_guards={'idle_single_e2e_tps': self.args.minimum_idle_tps,
                               'active_four_each_e2e_tps': self.args.minimum_active_tps})
        # Warm the exact short-prompt kernel shapes, without prefix caching.
        self.options = {'chat_template_kwargs': {'enable_thinking': self.args.thinking == 'enabled'}}
        await self.responses(SHORT_PROMPT, max_output_tokens=32, store=False, **self.options)
        single = await stream_requests(self.http, self.args.gateway, self.other, [SHORT_PROMPT], [self.options])
        assert single['requests'][0]['end_to_end_output_tps'] >= self.args.minimum_idle_tps, single
        self.check('idle_single_request_throughput', **single)
        self.check('idle_four_request_throughput', **await stream_group(
            self.http, self.args.gateway, self.other, request_options=[self.options] * 4))

    async def admission_checks(self):
        # Admission and cancellation have their own functional qualification.
        pass

    async def exercise(self):
        await super().exercise()
        assert any(check['name'] == 'four_simultaneous_responses_while_cpu_avatar_active'
                   for check in self.report['checks']), 'Four-way generation did not finish during media playback'

    async def ordinary_request(self):
        if self.avatar is None:
            return await super().ordinary_request()
        self.check('active_single_request_throughput', **await stream_requests(
            self.http, self.args.gateway, self.other, [SHORT_PROMPT], [self.options]))
        group = await stream_group(self.http, self.args.gateway, self.other, request_options=[self.options] * 4)
        assert min(item['end_to_end_output_tps'] for item in group['requests']) >= self.args.minimum_active_tps, group
        self.check('four_simultaneous_responses_while_cpu_avatar_active', **group)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--seconds', type=int, default=90)
    parser.add_argument('--avatar-id', default='ava')
    parser.add_argument('--wav', default=str(Path(__file__).resolve().parents[2] / 'evaluations/fixtures/question.wav'))
    parser.add_argument('--report', default='tmp/throughput-workstation.json')
    parser.add_argument('--thinking', choices=('enabled', 'disabled'), default='enabled')
    parser.add_argument('--minimum-idle-tps', type=float, default=45,
                        help='Short-prompt regression guard, not a long-context SLA')
    parser.add_argument('--minimum-active-tps', type=float, default=15,
                        help='Per-request guard while four outputs and CPU media are active')
    args = parser.parse_args()
    args.avatar_only = False
    asyncio.run(ThroughputQualification(args).run())
