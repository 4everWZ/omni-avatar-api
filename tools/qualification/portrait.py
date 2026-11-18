"""Qualify reference preparation and real Ditto/WHEP with both inference APIs."""
import argparse
import asyncio
import base64
from pathlib import Path
from time import monotonic

from .inference import InferenceQualification


class PortraitQualification(InferenceQualification):
    def __init__(self, args):
        super().__init__(args)
        self.report['scope'] = 'reference_ditto_native_omni_responses'
        self.render_samples = []

    async def ordinary_request(self):
        if getattr(self.args, 'concurrent_llm', 1) == 4 and self.avatar is not None:
            from .concurrency import stream_group
            result = await stream_group(self.http, self.args.gateway.rstrip('/'), self.other)
            self.check('four_simultaneous_responses_while_neural_avatar_active', **result)
        else:
            await super().ordinary_request()

    async def chat_checks(self):
        # These backends have a separate qualification; here exercise co-residency.
        await self.ordinary_request()
        await self.vision_check()

    async def admission_checks(self):
        pass

    async def open_avatar(self):
        if getattr(self.args, 'avatar_id', None):
            self.avatar_id = self.args.avatar_id
            self.report['avatar_id'] = self.avatar_id
            return await self.open_prepared_avatar()
        asset = await self.json('POST', '/v1/avatar/assets', expected=201, json={
            'mime_type': 'image/png', 'image_base64': base64.b64encode(Path(self.args.reference).read_bytes()).decode()})
        self.reference_id = asset['id']
        job = await self.json('POST', '/v1/avatar/generations', expected=202,
                              json={'reference_asset_id': asset['id'], 'driver': 'ditto'})
        self.job_id = job['id']
        self.check('reference_uploaded_and_preparation_queued', job_id=job['id'])
        await self.request('GET', '/v1/avatar/generations/' + job['id'], other=True, expected=404)
        started = monotonic()
        async with asyncio.timeout(120):
            while job['state'] in ('queued', 'preparing'):
                await asyncio.sleep(.25)
                job = await self.json('GET', '/v1/avatar/generations/' + job['id'])
        assert job['state'] == 'ready', job
        self.avatar_id = job['avatar_id']
        self.report['avatar_id'] = self.avatar_id
        for suffix in ('', '/preview'):
            await self.request('GET', '/v1/avatar/avatars/' + self.avatar_id + suffix, other=True, expected=404)
        self.check('prepared_avatar_owned', seconds=monotonic() - started, avatar_id=self.avatar_id)
        await self.open_prepared_avatar()

    async def open_prepared_avatar(self):
        await super().open_avatar()
        assert self.avatar['format']['driver'] == 'ditto'
        self.tasks.append(asyncio.create_task(self.inspect_renderer()))

    async def inspect_renderer(self):
        while True:
            session = await self.json('GET', '/v1/avatar/sessions/' + self.avatar['id'])
            assert session['state'] == 'active', session
            sample = session['renderer'] | {'timeline_ms': session['timeline_ms']}
            sample['video_lag_ms'] = sample['timeline_ms'] - sample['displayed_frame_ms']
            self.render_samples.append(sample)
            self.report['renderer_samples'] = self.render_samples[-120:]
            await asyncio.sleep(1)

    async def consume(self, track):
        if track.kind != 'video':
            return await super().consume(track)
        while True:
            frame = await track.recv()
            self.frames['video'] += 1
            if self.frames['video'] in (50, 250, 500):
                frame.to_image().save(str(Path(self.args.report).with_suffix('')) + f'-frame-{self.frames["video"]}.png')

    async def exercise(self):
        await super().exercise()
        if getattr(self.args, 'concurrent_llm', 1) == 4:
            assert any(item['name'] == 'four_simultaneous_responses_while_neural_avatar_active'
                       for item in self.report['checks']), 'Four-way generation did not finish during the media check'
        final = await self.json('GET', '/v1/avatar/sessions/' + self.avatar['id'])
        last = final['renderer']
        assert last['generated_frames'] >= self.args.seconds * 25, last
        assert last['last_generated_ms'] - last['displayed_frame_ms'] < 1500, last
        self.check('real_neural_frame_throughput', **last)
        lag = [item['video_lag_ms'] for item in self.render_samples[3:]]
        assert lag and max(lag) < 160, max(lag) if lag else None
        self.check('video_matches_audio_timeline', max_sampled_lag_ms=max(lag),
                   audio_input_lead_ms=self.avatar['format']['input_lead_ms'])
        if self.voice is None:
            return
        previous = self.generation
        await self.voice.send_json({'type': 'response.cancel'})
        async with asyncio.timeout(5):
            while self.generation <= previous:
                await asyncio.sleep(.05)
        await asyncio.sleep(2)
        recovered = await self.json('GET', '/v1/avatar/sessions/' + self.avatar['id'])
        assert recovered['state'] == 'active', recovered
        assert recovered['renderer']['last_generated_ms'] > recovered['timeline_ms'] - 160
        self.check('native_cancel_fences_old_frames_and_rendering_recovers', generation=self.generation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--concurrent-llm', type=int, choices=(1, 4), default=1)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument('--reference')
    reference.add_argument('--avatar-id')
    parser.add_argument('--wav', default=str(Path(__file__).resolve().parents[2] / 'evaluations/fixtures/question.wav'))
    parser.add_argument('--report', default='tmp/portrait-functional.json')
    args = parser.parse_args()
    args.avatar_only = False
    asyncio.run(PortraitQualification(args).run())
