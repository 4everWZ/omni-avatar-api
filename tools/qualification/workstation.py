"""Qualify Omni Chat and native duplex using public inference APIs."""
import argparse
import asyncio
import base64
from contextlib import AsyncExitStack, suppress
import io
import json
import math
import os
from pathlib import Path
import shutil
import socket
import struct
import subprocess
from datetime import datetime, timezone
from time import monotonic

import av
from aiohttp import ClientError, ClientSession, ClientTimeout
from PIL import Image
from evaluations.fixtures.long_context import DOCUMENTS, NEEDLE


def read_env(path):
    result = dict(os.environ)
    if path:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                key, _, value = line.partition('=')
                result[key] = value.strip().strip("'\"") if not value.strip().startswith('{') else value.strip()
    return result


def audio_chunks(path):
    resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
    pcm = bytearray()
    with av.open(str(path)) as source:
        for frame in source.decode(audio=0):
            for out in resampler.resample(frame):
                pcm.extend(bytes(out.planes[0])[:out.samples * 2])
        for out in resampler.resample(None):
            pcm.extend(bytes(out.planes[0])[:out.samples * 2])
    return [bytes(pcm[i:i+32000]).ljust(32000, b'\0') for i in range(0, len(pcm), 32000)]


class Qualification:
    def __init__(self, args):
        self.args = args
        env = read_env(args.env)
        self.token = env['OMNI_GATEWAY_TOKEN']
        others = json.loads(env.get('OMNI_GATEWAY_TOKENS', '{}') or '{}')
        self.other = next((v for k, v in others.items() if k != 'media'), None)
        if not self.other:
            raise ValueError('Configure a second independent service token in OMNI_GATEWAY_TOKENS')
        self.report = {'started_utc': datetime.now(timezone.utc).isoformat(),
                       'metrics_collector_host': socket.gethostname(),
                       'scope': 'omni_chat_and_duplex',
                       'requested_seconds': args.seconds, 'state': 'running', 'checks': [], 'samples': []}
        self.voice = None
        self.audible_chunks = 0
        self.native_audio = 0
        self.transcript = ''
        self.tasks = []

    def save(self):
        path = Path(self.args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report, ensure_ascii=False, indent=2), encoding='utf-8')

    def check(self, name, **evidence):
        self.report['checks'].append({'name': name, **evidence})
        self.save()
        print(name, flush=True)

    async def request(self, method, path, *, other=False, expected=200, **options):
        headers = {'Authorization': 'Bearer ' + (self.other if other else self.token)}
        headers.update(options.pop('headers', {}))
        response = await self.http.request(method, self.args.gateway.rstrip('/') + path, headers=headers, **options)
        if response.status != expected:
            # Do not include upstream error bodies: they may contain user data or deployment secrets.
            response.release()
            raise AssertionError(f'{method} {path}: expected {expected}, got {response.status}')
        return response

    async def json(self, method, path, **options):
        async with await self.request(method, path, **options) as response:
            return await response.json()

    async def chat_checks(self):
        health = await self.json('GET', '/healthz')
        models = await self.json('GET', '/v1/models')
        window = models['data'][0]['max_model_len']
        assert window >= 40960, f'Window was reduced to {window}'
        self.check('engine_identity', window=window, checkpoint=health['checkpoint'],
                   root_verified=health['checkpoint_root_verified'])
        assert health['checkpoint_root_verified']
        payload = {'model': 'omni-default', 'messages': [{'role': 'user', 'content': '只回答数字：2+3等于几？'}],
                   'max_tokens': 256, 'chat_template_kwargs': {'enable_thinking': False}}
        start = monotonic()
        ordinary = await self.json('POST', '/v1/chat/completions', json=payload)
        assert ordinary['choices'][0]['message']['content'].strip() == '5'
        self.check('ordinary_chat', seconds=monotonic() - start)
        schema = {'type': 'object', 'properties': {'sum': {'type': 'integer', 'const': 5}},
                  'required': ['sum'], 'additionalProperties': False}
        structured = await self.json('POST', '/v1/chat/completions', json=payload | {
            'messages': [{'role': 'user', 'content': 'Return a JSON object with one integer property sum, the sum of 2 and 3.'}],
            'response_format': {'type': 'json_schema', 'json_schema': {'name': 'sum', 'strict': True, 'schema': schema}}})
        assert structured['choices'][0]['finish_reason'] == 'stop'
        assert json.loads(structured['choices'][0]['message']['content']) == {'sum': 5}
        self.check('structured_chat')
        content = ''
        async with await self.request('POST', '/v1/chat/completions', json=payload | {'stream': True}, other=True) as response:
            async for line in response.content:
                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                    chunk = json.loads(line[6:])
                    for choice in chunk.get('choices', []):
                        content += choice.get('delta', {}).get('content') or ''
        assert content.strip() == '5'
        self.check('second_caller_streaming_chat')
        image = Image.new('RGB', (192, 192), 'red')
        encoded = io.BytesIO()
        image.save(encoded, format='PNG')
        vision = await self.json('POST', '/v1/chat/completions', json=payload | {'messages': [{
            'role': 'user', 'content': [{'type': 'text', 'text': '这张纯色图片是什么颜色？只回答颜色。'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(encoded.getvalue()).decode()}}]}]})
        answer = vision['choices'][0]['message']['content'].lower()
        assert '红' in answer or 'red' in answer
        self.check('vision_color_correct')
        for name, document in DOCUMENTS.items():
            def candidate(records):
                return payload | {'max_tokens': 1024, 'messages': [{'role': 'user', 'content': document(records)}]}
            empty = (await self.json('POST', '/v1/chat/prompt_tokens', json=candidate(0)))['count']
            hundred = (await self.json('POST', '/v1/chat/prompt_tokens', json=candidate(100)))['count']
            # Estimate an upper bound before binary search, avoiding million-token
            # count requests for a model whose entire window is only 40960.
            low, high = 0, math.ceil(200 * window / (hundred - empty))
            long_payload, count = None, 0
            while low <= high:
                middle = (low + high) // 2
                body = candidate(middle)
                measured = await self.json('POST', '/v1/chat/prompt_tokens', json=body)
                if measured['count'] <= window - 1024:
                    long_payload, count = body, measured['count']
                    low = middle + 1
                else:
                    high = middle - 1
            assert long_payload and count > window - 2048
            start = monotonic()
            reply = await self.json('POST', '/v1/chat/completions', json=long_payload)
            answer = reply['choices'][0]['message']['content'] or ''
            self.report.setdefault('long_context_attempts', []).append({'case': name, 'prompt_tokens': count,
                'finish_reason': reply['choices'][0]['finish_reason'],
                'completion_tokens': reply.get('usage', {}).get('completion_tokens'),
                'visible_body_characters': len(answer), 'correct': NEEDLE in answer})
            self.save()
            assert NEEDLE in answer, f'Long-context retrieval failed for {name} at {count} prompt tokens'
            self.check('long_context_correct', case=name, prompt_tokens=count, seconds=monotonic() - start)

    async def socket(self, path):
        return await self.http.ws_connect(self.args.gateway.rstrip('/') + path,
                                          headers={'Authorization': 'Bearer ' + self.token}, heartbeat=20)

    async def admission_checks(self):
        payload = {'model': 'omni-default', 'max_tokens': 4096, 'stream': True,
                   'chat_template_kwargs': {'enable_thinking': False}, 'messages': [{
                       'role': 'user', 'content': '请从1开始逐个写出整数，一直写到6000。每个数字单独一行，不要省略，不要解释，直接开始。'}]}
        health = await self.json('GET', '/healthz')
        capacity = health['batch_cap'] if health['mode'] == 'batch' else health['live_chat_cap']
        async with AsyncExitStack() as streams:
            async def start_stream():
                response = await self.request('POST', '/v1/chat/completions', json=payload)
                streams.callback(response.close)
                async with asyncio.timeout(30):
                    while not (await response.content.readline()).startswith(b'data: '):
                        if response.content.at_eof():
                            raise AssertionError('Streaming request ended before queue verification')

            async with asyncio.TaskGroup() as group:
                for _ in range(capacity):
                    group.create_task(start_stream())
            health = await self.json('GET', '/healthz')
            assert health['active']['chat'] == capacity, 'Long streams did not hold all Chat slots'
            async with asyncio.timeout(30):
                async with await self.request('POST', '/v1/chat/completions', other=True, expected=429, json={
                    'model': 'omni-default', 'max_tokens': 128,
                    'messages': [{'role': 'user', 'content': '只回答数字：2+3等于几？'}]}):
                    pass
            self.check('queue_deadline_returns_429')
        async with asyncio.timeout(30):
            while (await self.json('GET', '/healthz'))['active']['chat']:
                await asyncio.sleep(.2)
        reply = await self.json('POST', '/v1/chat/completions', json={
            'model': 'omni-default', 'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False},
            'messages': [{'role': 'user', 'content': '只回答数字：2+3等于几？'}]})
        assert '5' in reply['choices'][0]['message']['content']
        self.check('stream_disconnect_releases_lane_and_engine_responds')

    async def open_voice(self):
        self.voice = await self.socket('/v1/realtime?duplex=1&autostart=0&model=omni-default')
        await self.voice.send_json({'type': 'session.update', 'session': {
            'modalities': ['audio', 'text'], 'input_audio_format': 'pcm16', 'output_audio_format': 'pcm16',
            'extra_body': {'auto_response': True, 'force_listen_count': 0,
                           **getattr(self.args, 'realtime_extra_body', {})}}})
        self.tasks.append(asyncio.create_task(self.receive_voice()))

    async def receive_voice(self):
        async for frame in self.voice:
            if frame.type.name != 'TEXT':
                continue
            event = json.loads(frame.data)
            if event['type'] in ('response.audio.delta', 'response.output_audio.delta'):
                pcm = base64.b64decode(event['delta'], validate=True)
                if any(abs(sample) > 500 for (sample,) in struct.iter_unpack('<h', pcm)):
                    self.audible_chunks += 1
                self.native_audio += 1
            elif event['type'] == 'error':
                raise AssertionError('Realtime returned an error event')
            elif event['type'] in ('response.audio_transcript.delta', 'response.output_audio_transcript.delta'):
                self.transcript = (self.transcript + event.get('delta', ''))[-8192:]
        raise AssertionError('Realtime disconnected during qualification')

    async def metrics(self):
        item = {'elapsed_seconds': round(monotonic() - self.started, 2),
                'audible_chunks': self.audible_chunks, 'native_audio_chunks': self.native_audio}
        if shutil.which('nvidia-smi'):
            result = await asyncio.to_thread(subprocess.run,
                ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                item['gpu'] = [list(map(int, line.split(','))) for line in result.stdout.strip().splitlines()]
        meminfo = Path('/proc/meminfo')
        if meminfo.exists():
            values = dict(line.split(':', 1) for line in meminfo.read_text().splitlines())
            item['host_kib'] = {k: int(values[k].strip().split()[0]) for k in ('MemAvailable', 'SwapFree')}
        self.report['samples'].append(item)
        self.report['speech_question_transcript'] = self.transcript[:1024]
        self.save()

    async def exercise(self):
        chunks = audio_chunks(self.args.wav)
        self.started = monotonic()
        self.report['load_started_utc'] = datetime.now(timezone.utc).isoformat()
        async def produce():
            iteration = 0
            while True:
                tick = monotonic()
                phase = iteration % (len(chunks) + 15)
                pcm = chunks[phase] if phase < len(chunks) else bytes(32000)
                await self.voice.send_json({'type': 'input_audio_buffer.append', 'format': 'pcm16',
                    'sample_rate_hz': 16000, 'audio': base64.b64encode(pcm).decode(), 'is_speech': phase < len(chunks)})
                iteration += 1
                await asyncio.sleep(max(0, tick + 1 - monotonic()))

        async def ordinary_calls():
            while True:
                await self.ordinary_request()
                self.report['ordinary_calls'] = self.report.get('ordinary_calls', 0) + 1
                await asyncio.sleep(30)

        self.tasks.append(asyncio.create_task(produce()))
        self.tasks.append(asyncio.create_task(ordinary_calls()))
        next_sample = self.started
        while monotonic() - self.started < self.args.seconds:
            for task in tuple(self.tasks):
                if task.done():
                    task.result()
                    raise AssertionError('A continuous worker stopped unexpectedly')
            if monotonic() - self.started > 90 and self.native_audio == 0:
                raise AssertionError('Native duplex produced no audio in the first ninety seconds')
            if monotonic() >= next_sample:
                await self.metrics()
                next_sample = monotonic() + 30
            await asyncio.sleep(.2)
        assert self.audible_chunks > 0, 'Native duplex produced no audible PCM'
        assert any(value in self.transcript.lower() for value in ('five', '5', '五')), 'Unexpected speech reply'
        await self.metrics()
        self.check('continuous_native_audio', elapsed_seconds=monotonic() - self.started,
                   native_audio_chunks=self.native_audio, audible_chunks=self.audible_chunks)

    async def ordinary_request(self):
        reply = await self.json('POST', '/v1/chat/completions', other=True, json={
            'model': 'omni-default', 'messages': [{'role': 'user', 'content': '只回答数字：2+3等于几？'}],
            'max_tokens': 128, 'chat_template_kwargs': {'enable_thinking': False}})
        assert '5' in reply['choices'][0]['message']['content']

    async def cleanup(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.voice:
            if not self.voice.closed:
                with suppress(ClientError):
                    await self.voice.send_json({'type': 'session.close'})
            await self.voice.close()

    async def run(self):
        self.save()
        async with ClientSession(timeout=ClientTimeout(total=1200), trust_env=False) as self.http:
            try:
                await self.chat_checks()
                await self.admission_checks()
                await self.open_voice()
                await self.exercise()
                self.report['state'] = 'passed'
            except BaseException as error:
                self.report['state'] = 'failed'
                self.report['failure'] = type(error).__name__ + ': ' + str(error)
                raise
            finally:
                try:
                    await self.cleanup()
                except BaseException as error:
                    self.report['state'] = 'failed'
                    self.report['cleanup_failure'] = type(error).__name__ + ': ' + str(error)
                    raise
                finally:
                    self.report['finished_utc'] = datetime.now(timezone.utc).isoformat()
                    self.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--seconds', type=int, default=7200)
    parser.add_argument('--wav', default=str(Path(__file__).resolve().parents[2] / 'evaluations/fixtures/question.wav'))
    parser.add_argument('--report', default='tmp/workstation-qualification.json')
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error('--seconds must be positive')
    asyncio.run(Qualification(args).run())
