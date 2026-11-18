"""Pinned Ditto adapter. Model execution stays outside the media event loop."""
import importlib
import io
import json
import logging
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import resample_poly

CODE_REVISION = 'c3e47eee2e626500017a0556b470d6d4182f85e8'
MODEL_REVISION = 'e4a2f60328ee7c32af585ac4b3cce299e4c8e254'
setup_writer = threading.local()


class DittoBackend:
    """One SDK, one GPU lease. Prepared data is created locally, never uploaded."""
    def __init__(self, root, slot=0):
        import torch
        self.torch = torch
        self.slot = slot
        root = Path(root)
        sys.path.insert(0, str(root / 'code' / CODE_REVISION))
        self.pipeline = importlib.import_module('stream_pipeline_online')
        self.pipeline.VideoWriterByImageIO = lambda _: setup_writer.value
        # Progress bars are unsuitable for a continuously running API worker.
        self.pipeline.tqdm = lambda **_: type('Progress', (), {'update': lambda *a: None,
                                                              'close': lambda *a: None})()
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        models = root / 'models' / MODEL_REVISION
        cfg_path = models / 'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl'
        with cfg_path.open('rb') as source:
            cfg = pickle.load(source)
        for item in cfg['base_cfg'].values():
            if isinstance(item, dict) and (str(item.get('model_path', '')).endswith('.onnx')
                                          or item.get('force_ori_type')):
                item['device'] = 'cpu'
        self.sdk = self.pipeline.StreamSDK(str(cfg_path), str(models / 'ditto_pytorch'),
                                           base_cfg=cfg['base_cfg'])
        motion = self.sdk.audio2motion.lmdm.model
        motion.model_predictions = CudaPrediction(motion, torch)
        # One graph keeps warp/decode intermediates on CUDA and removes the
        # many small Python/kernel launches that limit the eager 3090 path.
        self.frame = CudaFrame(self.sdk.warp_f3d.warp_net, self.sdk.decode_f3d.decoder, torch)
        self.sdk.warp_f3d = self.frame
        self.sdk.decode_f3d = lambda image: image
        # Compile/capture before listening, including on restart with existing
        # prepared avatars. No caller's first session pays a compilation timeout.
        self.frame.prepare({'f_s_lst': [np.zeros((1, 32, 16, 64, 64), dtype=np.float32)],
                            'x_s_info_lst': [{'kp': np.zeros((1, 63), dtype=np.float32)}]})
        # The SDK hands CPU arrays across these stage queues. Give each GPU
        # stage its own stream so synchronous copies cannot drain other slots.
        self.sdk.audio2motion_worker = self.cuda_worker(self.sdk.audio2motion_worker)
        self.sdk.warp_f3d_worker = self.cuda_worker(self.sdk.warp_f3d_worker)
        self.registrar = self.sdk.avatar_registrar
        self.warmed = False

    def cuda_worker(self, worker):
        stream = self.torch.cuda.Stream()

        def run():
            with self.torch.cuda.stream(stream):
                worker()

        return run

    def setup(self, prepared, writer):
        self.frame.prepare(prepared)
        self.sdk.avatar_registrar = lambda *args, **kwargs: prepared
        setup_writer.value = writer
        try:
            self.sdk.setup('', f'/tmp/portrait-slot-{self.slot}.mp4', online_mode=True, overlap_v2=75,
                           sampling_timesteps=10, max_size=720)
        finally:
            del setup_writer.value
        # Bound queued decoded-frame memory instead of the SDK's hundred items.
        self.sdk.decode_f3d_queue.maxsize = 2

    def prepare(self, image, output, *, crop_scale=2.3, crop_vx_ratio=0, crop_vy_ratio=-.125, crop_flag_do_rot=True):
        data = self.registrar(str(image), max_dim=720, crop_scale=crop_scale,
                              crop_vx_ratio=crop_vx_ratio, crop_vy_ratio=crop_vy_ratio, crop_flag_do_rot=crop_flag_do_rot)
        if not self.warmed:
            self.setup(data, NullWriter())
            try:
                for _ in range(9):
                    self.sdk.run_chunk(np.zeros(6480, dtype=np.float32), (3, 5, 2))
                self.sdk.close()
                self.warmed = True
            except BaseException:
                self.abort()
                raise
        with Path(output).open('wb') as target:
            pickle.dump(data, target, protocol=5)

    def abort(self):
        if hasattr(self.sdk, 'stop_event'):
            self.sdk.stop_event.set()
            deadline = time.monotonic() + 5
            queues = [getattr(self.sdk, name + '_queue') for name in
                      ('audio2motion', 'motion_stitch', 'warp_f3d', 'decode_f3d', 'putback', 'writer')]
            while any(t.is_alive() for t in self.sdk.thread_list) and time.monotonic() < deadline:
                # SDK producers use blocking put(). Drain while stopping so a
                # full tensor queue cannot strand a producer after its consumer exits.
                for pending in queues:
                    try:
                        while True:
                            pending.get_nowait()
                    except queue.Empty:
                        pass
                for thread in self.sdk.thread_list:
                    thread.join(timeout=.02)
            if any(t.is_alive() for t in self.sdk.thread_list):
                # A stuck CUDA call must not allow a second SDK user.
                os._exit(1)
            self.sdk.writer.close()

    def start(self, path, emit):
        with Path(path).open('rb') as source:
            prepared = pickle.load(source)
        return PortraitRun(self, prepared, emit)


class NullWriter:
    def __call__(self, rgb, fmt='rgb'):
        pass
    def close(self):
        pass


class CudaPrediction:
    """Graph one deterministic diffusion step; keep the sampler/RNG unchanged."""
    def __init__(self, model, torch):
        self.torch = torch
        reference = model.model_predictions
        def mixed_predictions(*values):
            with torch.autocast('cuda', dtype=torch.float16):
                return reference(*values)
        original = torch.compile(mixed_predictions, fullgraph=True)
        self.inputs = [torch.zeros((1, model.seq_frames, model.motion_feat_dim), device=model.device),
                       torch.zeros((1, model.motion_feat_dim), device=model.device),
                       torch.zeros((1, model.seq_frames, model.audio_feat_dim), device=model.device),
                       torch.full((1,), 500, dtype=torch.long, device=model.device)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad(), torch.device(model.device):
            for _ in range(3):
                eager = original(*self.inputs)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.no_grad(), torch.device(model.device):
            # Keep FP16 precision drift separate from graph/compiled parity.
            # A private generator leaves the live diffusion RNG untouched.
            from torch.utils._pytree import tree_flatten
            generator = torch.Generator(device=model.device).manual_seed(1729)
            for magnitude in (0., 1., 2.):
                trial = [torch.randn(value.shape, dtype=value.dtype, device=value.device,
                                     generator=generator) * magnitude
                         if value.is_floating_point() else value for value in self.inputs]
                expected = tree_flatten(reference(*trial))[0]
                actual = tree_flatten(original(*trial))[0]
                for fp32, fp16 in zip(expected, actual):
                    torch.testing.assert_close(fp16, fp32, rtol=1e-2, atol=5e-3, check_dtype=False)
                logging.warning('Portrait motion FP16/FP32 difference: max=%g',
                    max((a-b).abs().max().item() for a,b in zip(expected, actual)))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad(), torch.device(model.device):
            self.output = original(*self.inputs)
        self.graph.replay()
        torch.testing.assert_close(self.output, eager, rtol=1e-3, atol=1e-4)

    def __call__(self, *inputs):
        for target, value in zip(self.inputs, inputs):
            target.copy_(value)
        self.graph.replay()
        return self.output


class CudaFrame:
    def __init__(self, warp, decoder, torch):
        self.warp, self.decoder, self.torch = warp.model, decoder.model, torch
        self.device = warp.device
        self.graph = None
        self.source = None
        # Pinned models construct the same coordinate grid via CPU on every
        # forward. Cache this immutable constant before capture.
        from core.models.modules import dense_motion, util
        coordinate_grid = util.make_coordinate_grid
        grids = {}
        def cached_grid(spatial_size, ref):
            key = (tuple(spatial_size), ref.dtype, ref.device)
            if key not in grids:
                grids[key] = coordinate_grid(spatial_size, ref)
            return grids[key]
        util.make_coordinate_grid = dense_motion.make_coordinate_grid = cached_grid
        def layout(tensor):
            if tensor.ndim == 4:
                return tensor.contiguous(memory_format=torch.channels_last)
            if tensor.ndim == 5:
                return tensor.contiguous(memory_format=torch.channels_last_3d)
            return tensor
        self.warp._apply(layout)
        self.decoder._apply(layout)
        # The pinned dense-motion head uses view() to flatten depth.
        self.warp.dense_motion_network.hourglass.register_forward_hook(
            lambda module, inputs, output: output.contiguous())
        self.eager_warp, self.eager_decoder = self.warp, self.decoder
        self.trt_frame = None
        frame_backend = os.environ.get('OMNI_PORTRAIT_FRAME_BACKEND', 'cuda-graph')
        if frame_backend == 'tensorrt':
            from .tensorrt_frame import TensorRTFrame
            self.trt_frame = TensorRTFrame(self.warp, self.decoder, torch,
                os.environ.get('OMNI_PORTRAIT_TRT_CACHE', '/cache/tensorrt'))
        elif frame_backend == 'cuda-graph':
            self.warp = torch.compile(self.warp, fullgraph=True)
            self.decoder = torch.compile(self.decoder, fullgraph=True)
        else:
            raise ValueError('Unknown portrait frame backend: ' + frame_backend)

    def forward(self):
        torch = self.torch
        if self.trt_frame is not None:
            return self.trt_frame(self.feature, self.kp_source, self.kp_driving)
        with torch.device(self.device), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            feature = self.warp(self.feature, self.kp_source, self.kp_driving).float()
            return self.decoder(feature.contiguous(memory_format=torch.channels_last)).float()

    def prepare(self, prepared):
        torch = self.torch
        source = prepared['f_s_lst'][0]
        self.source = source
        if self.graph is not None:
            self.feature.copy_(torch.from_numpy(source))
            self.kp_source.copy_(torch.from_numpy(prepared['x_s_info_lst'][0]['kp']).reshape(1, 21, 3))
            self.kp_driving.copy_(self.kp_source)
            return

        layout = torch.contiguous_format if self.trt_frame is not None else torch.channels_last_3d
        self.feature = torch.from_numpy(source).to(self.device).contiguous(memory_format=layout)
        self.kp_source = torch.from_numpy(prepared['x_s_info_lst'][0]['kp']).reshape(1, 21, 3).to(self.device)
        self.kp_driving = self.kp_source.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                eager = self.forward().clone()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.output = self.forward()
        graph.replay()
        torch.testing.assert_close(self.output, eager, rtol=1e-3, atol=1e-4)
        self.graph = graph
        with torch.device(self.device), torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            reference = self.eager_warp(self.feature, self.kp_source, self.kp_driving).float()
            reference = self.eager_decoder(reference.contiguous(memory_format=torch.channels_last)).float()
        # Compilation can fuse FP16 operations and change rounding. Characterize
        # that difference separately; graph replay itself is checked above
        # against the compiled forward. Bitwise eager parity is not a contract.
        difference = (self.output - reference).abs()
        logging.warning('Portrait %s/eager FP16 difference: max=%g mean=%g',
                        'TensorRT' if self.trt_frame is not None else 'compiled',
                        difference.max().item(), difference.mean().item())
        del reference, difference
        if self.trt_frame is not None:
            # TensorRT retains its own engine weights. Do not keep separate
            # PyTorch weights or run the eager check when creating a session.
            self.warp = self.decoder = self.eager_warp = self.eager_decoder = None
        # Eager comparison/autotuning scratch is disposable. Graph-owned pools
        # and model tensors remain live; return unused global cache to the LLM.
        torch.cuda.empty_cache()

    def __call__(self, feature, source, driving):
        if feature is not self.source:
            raise ValueError('Unexpected source features')
        self.kp_source.copy_(self.torch.from_numpy(source))
        self.kp_driving.copy_(self.torch.from_numpy(driving))
        self.graph.replay()
        out = self.output.cpu().numpy()
        return np.transpose(out[0], (1, 2, 0)).clip(0, 1) * 255


class PortraitRun:
    """Bounded 24kHz PCM timeline; worker sends timestamped generated JPEGs."""
    def __init__(self, backend, prepared, emit):
        self.backend, self.prepared, self.emit = backend, prepared, emit
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.generation = 0
        self.fences = [(0, 0)]
        self.segments = []
        self.origin = 0
        clock = os.environ.get('OMNI_GPU_RENDER_CLOCK')
        self.clock = Path(clock.replace('{slot}', str(self.backend.slot))) if clock else None
        self.frame_index = 0
        self.base_time = 0
        self.failure = None
        self.warming = True
        self.warm_frames = 0
        self.warm_ready = threading.Event()
        self.backend.setup(prepared, self)
        # cuDNN/CUDA initialization also occurs in the SDK's new worker threads.
        # Warm these exact threads, then keep them for the full session, including
        # clears. Readiness is based on completed frames, not a fixed sleep.
        for _ in range(9):
            self.backend.sdk.run_chunk(np.zeros(6480, dtype=np.float32), (3, 5, 2))
        deadline = time.monotonic() + 25
        while not self.warm_ready.wait(.1):
            error = self.backend.sdk.worker_exception
            if error or time.monotonic() > deadline:
                self.backend.abort()
                raise RuntimeError('Portrait session warmup failed') from error
        self.frame_index = -5  # The first online output belongs to the preceding .2s.
        self.warming = False
        # Fill the initial .6s of video before starting the public clock. This
        # is silence: clients cannot submit PCM before readiness, and their
        # first audio has a 1.2s lead beyond the prefetched feature windows.
        self.buffer_ready = threading.Event()
        for _ in range(4):
            self.backend.sdk.run_chunk(np.zeros(6480, dtype=np.float32), (3, 5, 2))
        if not self.buffer_ready.wait(10):
            self.backend.abort()
            raise RuntimeError('Portrait initial buffer failed')
        self.origin = time.monotonic()
        self.thread = threading.Thread(target=self.work, name='portrait-timeline', daemon=True)
        self.thread.start()

    def __call__(self, rgb, fmt='rgb'):
        if self.warming:
            self.warm_frames += 1
            if self.warm_frames >= 40:
                self.warm_ready.set()
            return
        timestamp = self.base_time + self.frame_index / 25
        self.frame_index += 1
        with self.lock:
            if timestamp < 0 or self.stop.is_set():
                return
            generation = next((gen for start, gen in reversed(self.fences) if timestamp >= start), 0)
        output = io.BytesIO()
        Image.fromarray(rgb).save(output, 'JPEG', quality=85)
        self.emit(timestamp, generation, output.getvalue())
        if self.clock and self.origin:
            temporary = self.clock.with_suffix('.new')
            temporary.write_text(json.dumps({'updated': time.monotonic(), 'deadline': self.origin + timestamp}))
            temporary.replace(self.clock)
        if self.frame_index >= 15 and hasattr(self, 'buffer_ready'):
            self.buffer_ready.set()

    def close(self):
        # The SDK writer close callback is separate from stopping this run.
        pass

    def append(self, pcm, start, generation):
        with self.lock:
            if generation != self.generation:
                raise ValueError('Stale generation')
            now = max(0, int((time.monotonic() - self.origin) * 24000))
            if start < now - 24000 or start + len(pcm) // 2 > now + 30 * 24000:
                raise ValueError('Audio outside the playback window')
            if len(self.segments) >= 1600:
                raise ValueError('Too many pending audio chunks')
            self.segments.append((start, np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768))

    def clear(self, generation):
        with self.lock:
            if generation <= self.generation:
                raise ValueError('Generation must increase')
            self.generation = generation
            self.segments.clear()
            # Frames already in the SDK queues keep the previous generation.
            # Cover queued future chunks and Hubert's right context.
            fence = time.monotonic() - self.origin + 1.1
            self.fences.append((fence, generation))
            self.fences = self.fences[-128:]

    def samples(self, start, count):
        out = np.zeros(count, dtype=np.float32)
        with self.lock:
            for position, audio in self.segments:
                left, right = max(start, position), min(start + count, position + len(audio))
                if right > left:
                    out[left-start:right-start] = audio[left-position:right-position]
            self.segments = [(p, a) for p, a in self.segments if p + len(a) > start - 24000]
        return out

    def work(self):
        chunk = 4
        try:
            while not self.stop.is_set():
                # Generate .8s ahead. Session input lead is 1.2 seconds,
                # leaving both Hubert lookahead and GPU execution time.
                due = self.origin + self.base_time + chunk * .2 - .8
                if self.stop.wait(max(0, due - time.monotonic())):
                    break
                if self.backend.sdk.worker_exception:
                    raise self.backend.sdk.worker_exception
                start = round((self.base_time + chunk * .2) * 24000) - 2880
                audio = resample_poly(self.samples(start, 9720), 2, 3)
                self.backend.sdk.run_chunk(audio, (3, 5, 2))
                chunk += 1
        except BaseException as error:
            logging.exception('Portrait timeline failed')
            self.failure = error
        finally:
            try:
                self.backend.abort()
            except BaseException as error:
                self.failure = error

    def shutdown(self):
        self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError('Portrait timeline failed to stop')
        if self.clock:
            self.clock.unlink(missing_ok=True)
