"""Qualify original-precision image jobs, LLM continuity, and workstation peak memory."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import shutil
from time import monotonic

from aiohttp import ClientSession, ClientTimeout

from .inference import output_text
from .workstation import read_env


async def sample_memory(report):
    tool = shutil.which('nvidia-smi')
    if not tool:
        return
    process = await asyncio.create_subprocess_exec(
        tool, '--query-gpu=index,memory.used', '--format=csv,noheader,nounits', '--loop-ms=100',
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        while line := await process.stdout.readline():
            gpu, mib = map(int, line.split(b','))
            key = str(gpu)
            report['peak_gpu_used_mib'][key] = max(report['peak_gpu_used_mib'].get(key, 0), mib)
            meminfo = Path('/proc/meminfo')
            if meminfo.is_file():
                values = {name: int(value.split()[0]) for name, value in
                          (line.split(':', 1) for line in meminfo.read_text().splitlines())}
                used = values['MemTotal'] - values['MemAvailable']
                report['peak_host_used_kib'] = max(report['peak_host_used_kib'], used)
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


async def qualify(args):
    token = read_env(args.env)['OMNI_GATEWAY_TOKEN']
    prefix = '/v1/images/jobs'
    args.directory.mkdir(parents=True, exist_ok=True)
    report = {'state': 'running', 'cases': []}
    async with ClientSession(args.gateway, headers={'Authorization': 'Bearer ' + token},
                             timeout=ClientTimeout(total=1200)) as http:
        async def call(method, path, *, expected=200, **options):
            async with http.request(method, path, **options) as response:
                assert response.status == expected, (method, path, response.status)
                return await response.json()

        async def llm(prompt, **options):
            return await call('POST', '/llm/v1/responses', json={
                'model': 'llm-default', 'input': prompt, 'max_output_tokens': 64,
                'chat_template_kwargs': {'enable_thinking': False}, **options})

        async def wait(path, states):
            async with asyncio.timeout(1200):
                while (job := await call('GET', path))['state'] not in states:
                    await asyncio.sleep(.2)
                return job

        first = await llm('请记住本次代号是蓝鲸5738。只回答收到。')
        cases = [
            ('reference_person', args.reference,
             'Keep the same person and identifying facial features. A candid outdoor portrait, green garden, daylight.'),
            ('anime_character', args.anime_reference,
             'Keep the recognizable character, colors, hair and accessories. Anime illustration, blue sky, lively pose.'),
            ('text_only', None,
             'A watercolor illustration of a tiny observatory on a snowy hill with a clear blue sky.'),
        ]
        try:
            for label, reference, prompt in cases:
                sample = {'name': label, 'peak_gpu_used_mib': {}, 'peak_host_used_kib': 0}
                report['cases'].append(sample)
                metrics = asyncio.create_task(sample_memory(sample))
                request = {'prompt': prompt, 'seed': 42}
                if reference:
                    request['reference_image_base64'] = base64.b64encode(reference.read_bytes()).decode()
                started = monotonic()
                job = await call('POST', prefix, expected=202, json=request)
                feature = f"{prefix}/{job['id']}"
                try:
                    job = await wait(feature, {'completed', 'failed', 'cancelled'})
                    sample.update(job=job, observed_seconds=monotonic() - started)
                    assert job['state'] == 'completed', job
                    async with http.get(job['result_url']) as png:
                        assert png.status == 200
                        data = await png.read()
                    assert data.startswith(b'\x89PNG\r\n\x1a\n')
                    (args.directory / (label + '.png')).write_bytes(data)
                finally:
                    if job['state'] not in {'completed', 'failed', 'cancelled'}:
                        await call('POST', feature + '/cancel')
                    metrics.cancel()
                    evidence = await asyncio.gather(metrics, return_exceptions=True)
                    if evidence and isinstance(evidence[0], Exception):
                        raise evidence[0]
            second = await llm('本次代号是什么？只回答代号。', previous_response_id=first['id'])
            assert '5738' in output_text(second)
            report.update(responses_continuation='passed', state='passed',
                          visual_review='Compare output identity, character design, style and defects against references.')
        except BaseException:
            report['state'] = 'failed'
            raise
        finally:
            (args.directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                       encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gateway', default='http://127.0.0.1:8000')
    parser.add_argument('--env', default='.env')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--anime-reference', type=Path, required=True)
    parser.add_argument('--directory', type=Path, default=Path('tmp/image-qualification'))
    asyncio.run(qualify(parser.parse_args()))
