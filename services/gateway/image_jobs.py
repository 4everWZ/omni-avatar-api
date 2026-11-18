"""Caller-owned image jobs and the workstation GPU switch."""
import asyncio
import base64
import binascii
from collections import deque
import json
import logging
import os
from pathlib import Path
import secrets
from time import monotonic, time

from aiohttp import ClientError, ClientSession, ClientTimeout, web

TERMINAL = {'completed', 'failed', 'cancelled'}
IO_ERRORS = (ClientError, TimeoutError, OSError, ValueError, KeyError, TypeError)
MAX_QUEUE = 8
MAX_RECORDS = 256
RETENTION = 24 * 3600
QUEUE_TIMEOUT = 1200


class Cancelled(Exception):
    pass


class ImageQueue:
    def __init__(self, root, worker_url, worker_token, timeout):
        self.root = Path(root)
        self.worker_url, self.worker_token = worker_url, worker_token
        self.timeout = timeout
        self.jobs, self.cancelled = {}, {}
        self.queue = deque()
        self.changed = asyncio.Event()
        self.active = None
        self.stopping = False

    @staticmethod
    def atomic(path, data):
        temporary = path.with_suffix('.new')
        temporary.write_bytes(data)
        temporary.replace(path)

    def save(self, job, **fields):
        job.update(fields)
        self.atomic(self.root / (job['id'] + '.json'), json.dumps(job).encode())

    def prune(self):
        for identifier, job in tuple(self.jobs.items()):
            if job['state'] in TERMINAL and time() - job['finished_at'] >= RETENTION:
                (self.root / (identifier + '.json')).unlink(missing_ok=True)
                (self.root / (identifier + '.png')).unlink(missing_ok=True)
                del self.jobs[identifier]

    @staticmethod
    def view(job):
        result = {key: value for key, value in job.items() if key != 'owner'}
        if job['state'] == 'completed':
            result['result_url'] = f"/v1/images/jobs/{job['id']}/result"
        return result

    async def submit(self, request):
        try:
            body = await request.json()
            prompt = body['prompt']
            seed = body.get('seed', secrets.randbits(32))
            reference = body.get('reference_image_base64')
            if (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4096
                    or type(seed) is not int or not 0 <= seed < 2 ** 32):
                raise ValueError()
            if reference is not None and (not isinstance(reference, str) or not reference
                    or len(reference) > 5592408 or len(base64.b64decode(reference, validate=True)) > 4194304):
                raise ValueError()
        except (TypeError, KeyError, ValueError, binascii.Error):
            return web.json_response({'error': {'code': 'invalid_image_request'}}, status=400)
        self.prune()
        if len(self.queue) >= MAX_QUEUE or len(self.jobs) >= MAX_RECORDS:
            return web.json_response({'error': {'code': 'image_queue_full'}}, status=429,
                                     headers={'Retry-After': '1'})
        identifier = secrets.token_hex(16)
        job = {'id': identifier, 'owner': request['owner'], 'state': 'queued',
               'seed': seed, 'created_at': time()}
        self.save(job)
        self.jobs[identifier] = job
        self.cancelled[identifier] = asyncio.Event()
        self.queue.append((job, {'prompt': prompt, 'seed': seed, 'reference_image_base64': reference}))
        self.changed.set()
        return web.json_response(self.view(job), status=202, headers={'Location': f'/v1/images/jobs/{identifier}'})

    async def detail(self, request):
        self.prune()
        job = self.jobs.get(request.match_info['id'])
        if job is None or job['owner'] != request['owner']:
            raise web.HTTPNotFound()
        identifier = job['id']
        if request.method == 'POST' and job['state'] not in TERMINAL:
            self.cancelled[identifier].set()
            if any(item[0] is job for item in self.queue):
                self.queue = deque(item for item in self.queue if item[0] is not job)
                self.save(job, state='cancelled', finished_at=time())
                del self.cancelled[identifier]
            else:
                self.save(job, state='cancelling')
        if request.path.endswith('/result'):
            if job['state'] != 'completed':
                raise web.HTTPConflict()
            return web.FileResponse(self.root / (identifier + '.png'), headers={'Cache-Control': 'private, no-store'})
        return web.json_response(self.view(job), headers={'Cache-Control': 'no-store'})

    async def worker(self, method, path, **options):
        async with self.http.request(method, self.worker_url + path,
                headers={'Authorization': 'Bearer ' + self.worker_token},
                allow_redirects=False, **options) as response:
            response.raise_for_status()
            return await response.json()

    async def await_worker(self, path, cancel, *, success):
        async with asyncio.timeout(self.timeout):
            while True:
                if cancel.is_set():
                    await self.worker('POST', path + '/cancel')
                job = await self.worker('GET', path)
                if job['state'] in TERMINAL or job['state'] == success:
                    if cancel.is_set() or job['state'] == 'cancelled':
                        raise Cancelled()
                    return job
                await asyncio.sleep(.25)

    async def recover(self):
        pass

    async def after_job(self):
        pass

    def initializing(self):
        pass

    async def process(self):
        while not self.stopping:
            while self.queue and time() - self.queue[0][0]['created_at'] >= QUEUE_TIMEOUT:
                job, _ = self.queue.popleft()
                self.save(job, state='failed', error='image_queue_timeout', finished_at=time())
                self.cancelled.pop(job['id'], None)
            await self.recover()
            if self.stopping:
                break
            if not self.queue:
                self.changed.clear()
                await self.changed.wait()
                continue
            self.active, payload = self.queue.popleft()
            await self.run_job(self.active, payload)
            self.active = None
            await self.after_job()

    async def lifecycle(self, app):
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.root.glob('*.json'):
            job = json.loads(path.read_text())
            if job['state'] not in TERMINAL:
                self.save(job, state='failed', error='image_coordinator_restarted', finished_at=time())
                path.with_suffix('.png').unlink(missing_ok=True)
            self.jobs[job['id']] = job
        self.prune()
        self.initializing()
        async with ClientSession(timeout=ClientTimeout(total=15, connect=3), trust_env=False) as session:
            self.http = session
            task = asyncio.create_task(self.process())
            try:
                yield
            finally:
                self.stopping = True
                self.changed.set()
                if self.active:
                    if cancel := self.cancelled.get(self.active['id']):
                        cancel.set()
                    try:
                        await asyncio.wait_for(asyncio.shield(task), 10)
                    except TimeoutError:
                        pass
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


class ImageJobs(ImageQueue):
    def __init__(self, text, root, switch_root, worker_url, worker_token, llm_token, minimum_mib,
                 kv_bytes=0):
        super().__init__(root, worker_url, worker_token, text.config.request_timeout)
        self.text = text
        self.switch_root = Path(switch_root)
        self.marker = self.switch_root / 'image-active'
        self.llm_token = llm_token
        self.minimum_free = minimum_mib * 1024 ** 2
        self.kv_bytes = kv_bytes

    @classmethod
    def from_env(cls, text):
        minimum_mib = int(os.environ.get('OMNI_IMAGE_MIN_FREE_MIB', '13312'))
        kv_bytes = int(os.environ.get('LLM_KV_CACHE_BYTES', '9663676416'))
        if minimum_mib <= 0 or kv_bytes <= 0:
            raise ValueError('Image GPU memory threshold and LLM KV allocation must be positive')
        return cls(text, os.environ['OMNI_IMAGE_DATA'], os.environ['OMNI_GPU_SWITCH_ROOT'],
                   os.environ['OMNI_IMAGE_WORKER_URL'], os.environ['OMNI_IMAGE_INTERNAL_TOKEN'],
                   os.environ['LLM_GPU_CONTROL_TOKEN'], minimum_mib, kv_bytes)

    async def llm(self, action):
        async with self.http.post(self.text.config.engine_url + '/internal/gpu-control/' + action,
                headers={'X-GPU-Control': self.llm_token}, allow_redirects=False,
                timeout=ClientTimeout(total=self.text.config.request_timeout, connect=3)) as response:
            response.raise_for_status()
            return (await response.json())['phase']

    def claim_portrait_window(self, *, probe=False):
        import fcntl
        with (self.switch_root / 'portrait.lock').open('a') as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if self.marker.exists():
                raise OSError('Previous GPU lease must be recovered')
            if not probe:
                self.atomic(self.marker, b'blocked')
            return True

    async def restore(self):
        if self.marker.exists():
            # Invalidate delayed dispatches before querying the worker.
            self.atomic(self.marker, b'blocked')
            async with asyncio.timeout(self.text.config.request_timeout):
                while (state := await self.worker('GET', '/state'))['busy']:
                    await self.worker('POST', f"/render/{state['id']}/cancel")
                    await asyncio.sleep(.25)
        if await self.llm('wake') != 'awake':
            raise ValueError('LLM did not restore its GPU allocation')
        self.marker.unlink(missing_ok=True)
        self.text.admission.pause(None)

    async def wait_until_drained(self, cancel):
        drain = asyncio.create_task(self.text.admission.drain())
        stop = asyncio.create_task(cancel.wait())
        try:
            async with asyncio.timeout(self.text.config.request_timeout):
                await asyncio.wait((drain, stop), return_when=asyncio.FIRST_COMPLETED)
            if cancel.is_set():
                raise Cancelled()
        finally:
            drain.cancel()
            stop.cancel()
            await asyncio.gather(drain, stop, return_exceptions=True)

    async def run_job(self, job, payload):
        identifier = job['id']
        cancel = self.cancelled[identifier]
        timing = {}
        owned = False
        result = {}
        preparing = prepared = False
        path = '/render/' + identifier
        try:
            while not cancel.is_set():
                if time() - job['created_at'] >= QUEUE_TIMEOUT:
                    result.update(state='failed', error='image_queue_timeout')
                    break
                if self.claim_portrait_window(probe=True):
                    if not prepared:
                        started = monotonic()
                        preparing = True
                        await self.worker('PUT', '/prepare/' + identifier, json=payload)
                        response = await self.await_worker(path, cancel, success='prepared')
                        timing['prepare'] = monotonic() - started
                        if response['state'] != 'prepared':
                            result.update(state='failed', error=response.get('error', 'image_prepare_failed'))
                            return
                        prepared = True
                    if not self.claim_portrait_window(probe=True):
                        continue
                    self.text.admission.pause('image_gpu_switch')
                    started = monotonic()
                    await self.wait_until_drained(cancel)
                    timing['drain_llm'] = monotonic() - started
                    if self.claim_portrait_window():
                        owned = True
                        break
                    self.text.admission.pause(None)
                try:
                    await asyncio.wait_for(cancel.wait(), .25)
                except TimeoutError:
                    pass
            if cancel.is_set():
                raise Cancelled()
            if not owned:
                return
            timing['queue'] = max(0, time() - job['created_at'] - timing['drain_llm'] - timing['prepare'])
            self.save(job, state='running')
            started = monotonic()
            free_before = (await self.worker('GET', '/state'))['free_bytes']
            check_kv_first = not self.kv_bytes or free_before + self.kv_bytes >= self.minimum_free
            if check_kv_first:
                await self.llm('release_kv')
            if (not check_kv_first or (not cancel.is_set() and
                    (await self.worker('GET', '/state'))['free_bytes'] < self.minimum_free)):
                if check_kv_first:
                    await self.llm('wake')
                await self.llm('offload')
                strategy = 'weights_and_kv'
            else:
                strategy = 'kv_only'
            timing['release_llm'] = monotonic() - started
            result['gpu_release'] = strategy
            if cancel.is_set():
                raise Cancelled()
            if (await self.worker('GET', '/state'))['free_bytes'] < self.minimum_free:
                result.update(state='failed', error='insufficient_gpu_memory')
                return
            self.atomic(self.marker, identifier.encode('ascii'))
            await self.worker('PUT', path)
            rendered = await self.await_worker(path, cancel, success='completed')
            if rendered['state'] == 'completed':
                output = rendered['output']
                self.atomic(self.root / (identifier + '.png'), base64.b64decode(output['png'], validate=True))
                timing.update(output['timing'])
                result['worker_peak_reserved_bytes'] = output['worker_peak_reserved_bytes']
            result.update(state=rendered['state'])
            if 'error' in rendered:
                result['error'] = rendered['error']
        except Cancelled:
            result.update(state='cancelled')
        except IO_ERRORS:
            logging.exception('Image job %s failed', identifier)
            result.update(state='failed', error='image_worker_or_switch_failed')
        finally:
            if preparing and (not result or result.get('state') != 'completed'):
                try:
                    await self.worker('POST', path + '/cancel')
                except IO_ERRORS:
                    logging.warning('Could not signal cancellation to image worker for %s', identifier)
            if owned:
                started = monotonic()
                try:
                    await self.restore()
                except IO_ERRORS:
                    logging.exception('GPU recovery failed after image job %s', identifier)
                    self.text.admission.pause('image_gpu_recovery_failed')
                    result.update(state='failed', error='llm_restore_failed')
                timing['restore_llm'] = monotonic() - started
            elif self.text.admission.paused == 'image_gpu_switch':
                self.text.admission.pause(None)
            if result:
                if cancel.is_set() and not self.text.admission.paused:
                    result.update(state='cancelled')
                    result.pop('error', None)
                if result.get('state') != 'completed':
                    (self.root / (identifier + '.png')).unlink(missing_ok=True)
                finished = time()
                timing['total'] = finished - job['created_at']
                self.save(job, **result, timing=timing, finished_at=finished)
            self.cancelled.pop(identifier, None)

    def initializing(self):
        self.text.admission.pause('image_gpu_initializing')

    async def recover(self):
        while self.text.admission.paused and not self.stopping:
            try:
                await self.restore()
            except IO_ERRORS:
                self.text.admission.pause('image_gpu_recovery_failed')
                await asyncio.sleep(2)

    async def after_job(self):
        # Resume the current head before another image can close admission.
        head = self.text.admission.waiters[0] if self.text.admission.waiters else None
        while head is not None and head in self.text.admission.waiters:
            await self.text.admission.changed.wait()


class DedicatedImageJobs(ImageQueue):
    @classmethod
    def from_env(cls, text=None):
        return cls(os.environ['OMNI_IMAGE_DATA'], os.environ['OMNI_IMAGE_WORKER_URL'],
                   os.environ['OMNI_IMAGE_INTERNAL_TOKEN'], float(os.environ.get('OMNI_REQUEST_TIMEOUT', 1200)))

    async def run_job(self, job, payload):
        identifier = job['id']
        path = '/render/' + identifier
        cancel = self.cancelled[identifier]
        result = {'state': 'failed', 'error': 'image_worker_unavailable'}
        try:
            self.save(job, state='running')
            await self.worker('PUT', '/prepare/' + identifier, json=payload)
            prepared = await self.await_worker(path, cancel, success='prepared')
            if prepared['state'] != 'prepared':
                result['error'] = prepared.get('error', 'image_prepare_failed')
                return
            await self.worker('PUT', path)
            rendered = await self.await_worker(path, cancel, success='completed')
            if rendered['state'] == 'completed':
                output = rendered['output']
                self.atomic(self.root / (identifier + '.png'), base64.b64decode(output['png'], validate=True))
                result = {'state': 'completed', 'timing': output['timing'],
                          'worker_peak_reserved_bytes': output['worker_peak_reserved_bytes']}
            else:
                result['error'] = rendered.get('error', 'image_inference_failed')
        except Cancelled:
            result = {'state': 'cancelled'}
        except IO_ERRORS:
            logging.exception('Dedicated image worker failed for %s', identifier)
            try:
                await self.worker('POST', path + '/cancel')
            except IO_ERRORS:
                pass
        finally:
            self.save(job, **result, finished_at=time())
            self.cancelled.pop(identifier, None)
