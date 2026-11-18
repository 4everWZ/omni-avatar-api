"""Measure two real Omni and Ditto streams and Qwen single/four-way throughput."""
import argparse
import asyncio
import base64
from copy import copy
from pathlib import Path
from time import monotonic

from aiohttp import ClientSession, ClientTimeout, WSServerHandshakeError

from .concurrency import SHORT_PROMPT, stream_group, stream_requests
from .omni_throughput import stream_chat
from .portrait import PortraitQualification
from .workstation import audio_chunks


class CapacityQualification(PortraitQualification):
    async def ordinary_request(self):
        pass

    async def text_load(self, label):
        options = {'chat_template_kwargs': {'enable_thinking': True}}
        for run in range(self.args.repeats):
            single = await stream_requests(self.http, self.args.gateway, self.other, [SHORT_PROMPT], [options])
            group = await stream_group(self.http, self.args.gateway, self.other, request_options=[options] * 4)
            self.check(label, run=run, single=single, four=group)

    async def mixed_omni(self):
        audio_before = self.native_audio

        async def speak():
            for chunk in [*audio_chunks(self.args.wav), *([bytes(32000)] * 8)]:
                await self.voice.send_json({'type': 'input_audio_buffer.append', 'format': 'pcm16',
                    'sample_rate_hz': 16000, 'audio': base64.b64encode(chunk).decode(),
                    'is_speech': any(chunk)})
                await asyncio.sleep(1)

        talk = asyncio.create_task(speak())
        try:
            async with asyncio.timeout(30):
                while self.native_audio == audio_before:
                    if talk.done():
                        talk.result()
                    await asyncio.sleep(.05)
            started = monotonic()
            chunk_count = self.native_audio
            result = await stream_chat(self.http, self.args.gateway, self.other)
            assert self.native_audio > chunk_count, 'Native audio did not progress during Chat'
            self.check('realtime_and_chat_overlap', audio_chunks=self.native_audio - chunk_count,
                       seconds=monotonic() - started, chat=result)
        finally:
            talk.cancel()
            await asyncio.gather(talk, return_exceptions=True)

    async def run(self):
        self.report.update(scope='dual_omni_dual_ditto_single_and_four_qwen',
                           mps_client_percent=self.args.mps_percent, state='running')
        self.save()
        async with ClientSession(timeout=ClientTimeout(total=1200), trust_env=False) as self.http:
            second = None
            try:
                if not (self.args.text_only or self.args.avatar_only):
                    chat = await stream_chat(self.http, self.args.gateway, self.other, concurrency=2)
                    self.check('two_chat_outputs', **chat)
                    assert chat['outputs_overlap'], 'Omni Chat output intervals did not overlap'
                self.report['llm_checkpoint'] = (await self.json('GET', '/llm/healthz'))['checkpoint']
                await self.responses(SHORT_PROMPT, max_output_tokens=32, store=False,
                                     chat_template_kwargs={'enable_thinking': True})
                await self.text_load('idle_qwen')
                if self.args.text_only:
                    self.report.update(scope='idle_qwen_single_and_four', state='passed')
                    return
                await self.open_avatar()
                if not self.args.avatar_only:
                    await self.mixed_omni()
                second_args = copy(self.args)
                second_args.avatar_id = self.avatar_id
                second_args.report = str(Path(self.args.report).with_suffix('.second.json'))
                second = CapacityQualification(second_args)
                second.http = self.http
                await second.open_avatar()
                if not self.args.avatar_only:
                    try:
                        async with await self.socket('/v1/realtime?duplex=1&model=omni-default'):
                            raise AssertionError('Third Omni session entered')
                    except WSServerHandshakeError as error:
                        assert error.status == 429
                    assert (await self.json('GET', '/healthz'))['active']['realtime'] == 2
                for current in (self, second):
                    detail = await current.json('GET', '/v1/avatar/sessions/' + current.avatar['id'])
                    assert detail['state'] == 'active', detail
                async with await self.request('POST', '/v1/avatar/sessions', expected=429,
                        json={'avatar_id': self.avatar_id, 'source': {'type': 'audio'}}):
                    pass
                async with asyncio.TaskGroup() as group:
                    group.create_task(self.exercise())
                    group.create_task(second.exercise())
                    group.create_task(self.text_load('dual_ditto_qwen'))
                self.check('second_neural_session', checks=second.report['checks'])
                await second.cleanup()
                second = None
                # The first session continues while the second capacity is reused.
                if not self.args.avatar_only:
                    async with await self.socket('/v1/realtime?duplex=1&model=omni-default'):
                        self.check('second_realtime_slot_reusable')
                self.report['state'] = 'passed'
            except BaseException:
                self.report['state'] = 'failed'
                raise
            finally:
                if second:
                    self.report['second_renderer_samples'] = second.render_samples[-120:]
                    await second.cleanup()
                await self.cleanup()
                self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--seconds', type=int, default=90)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--mps-percent', type=int, required=True)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument('--text-only', action='store_true')
    workload.add_argument('--portraits-only', dest='avatar_only', action='store_true')
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument('--reference')
    reference.add_argument('--avatar-id')
    parser.add_argument('--wav', default='evaluations/fixtures/question.wav')
    parser.add_argument('--report', default='tmp/workstation-capacity.json')
    args = parser.parse_args()
    if not args.text_only and not (args.reference or args.avatar_id):
        parser.error('Choose --reference or --avatar-id for mixed-load qualification')
    asyncio.run(CapacityQualification(args).run())
