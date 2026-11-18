"""Original-precision Klein, with CPU text encoding and task-scoped GPU weights."""
import base64
from io import BytesIO
from time import monotonic

MODEL = 'black-forest-labs/FLUX.2-klein-4B'
REVISION = 'e7b7dc27f91deacad38e78976d1f2b499d76a294'


class Cancelled(Exception):
    pass


class Klein:
    def __init__(self, path):
        self.path = path
        self.pipe = None
        self.cpu_weights = None

    @staticmethod
    def free_memory():
        import pynvml as nvml
        nvml.nvmlInit()
        try:
            return nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(0)).free
        finally:
            nvml.nvmlShutdown()

    def prepare(self, payload, cancelled):
        import torch
        from PIL import Image, ImageOps
        from diffusers import Flux2KleinPipeline

        class MixedDevicePipeline(Flux2KleinPipeline):
            @property
            def _execution_device(self):
                return self.transformer.device

        started = monotonic()
        check_cancelled(cancelled)
        reference = None
        if data := payload.get('reference_image_base64'):
            with Image.open(BytesIO(base64.b64decode(data, validate=True))) as source:
                if source.format not in ('PNG', 'JPEG') or source.width * source.height > 8_000_000:
                    raise ValueError('Expected a PNG or JPEG of at most eight megapixels')
                reference = ImageOps.exif_transpose(source).convert('RGB')
                reference.thumbnail((1024, 1024))
        if self.pipe is None:
            torch.set_num_threads(min(torch.get_num_threads(), 6))
            self.pipe = MixedDevicePipeline.from_pretrained(self.path, torch_dtype=torch.bfloat16,
                                                            local_files_only=True)
            self.pipe.set_progress_bar_config(disable=True)
            self.cpu_weights = [(module, module.state_dict(), nonpersistent_buffers(module))
                                for module in (self.pipe.transformer, self.pipe.vae)]
        loaded = monotonic()
        check_cancelled(cancelled)
        with torch.inference_mode():
            embeddings, _ = self.pipe.encode_prompt(prompt=payload['prompt'], device=torch.device('cpu'))
        check_cancelled(cancelled)
        return (reference, embeddings, payload['seed'],
                {'load_cpu': loaded - started, 'encode_cpu': monotonic() - loaded})

    def generate(self, prepared, cancelled):
        import torch
        reference, embeddings, seed, timing = prepared
        pipe = self.pipe

        def step(pipe, index, timestep, state):
            check_cancelled(cancelled)
            return state

        check_cancelled(cancelled)
        started = monotonic()
        with torch.inference_mode():
            torch.cuda.reset_peak_memory_stats()
            try:
                pipe.transformer.to('cuda')
                pipe.vae.to('cuda')
                check_cancelled(cancelled)
                uploaded = monotonic()
                image = pipe(prompt_embeds=embeddings.to('cuda'), image=reference,
                             width=1024, height=1024, guidance_scale=1.0, num_inference_steps=4,
                             generator=torch.Generator(device='cuda').manual_seed(seed),
                             callback_on_step_end=step).images[0]
                torch.cuda.synchronize()
                generated = monotonic()
                peak = torch.cuda.max_memory_reserved()
                check_cancelled(cancelled)
            finally:
                try:
                    for module, weights, buffers in self.cpu_weights:
                        module.load_state_dict(weights, assign=True)
                        for child, name, buffer in buffers:
                            child.register_buffer(name, buffer, persistent=False)
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except BaseException:
                    # Terminate the CUDA context before a replacement worker can report idle.
                    import os
                    os._exit(1)
        output = BytesIO()
        image.save(output, format='PNG')
        return {'png': base64.b64encode(output.getvalue()).decode('ascii'),
                'timing': timing | {'upload_gpu': uploaded - started, 'inference': generated - uploaded,
                                    'cleanup_and_png': monotonic() - generated},
                'worker_peak_reserved_bytes': peak}


def check_cancelled(cancelled):
    if cancelled.is_set():
        raise Cancelled()


def nonpersistent_buffers(module):
    return [(child, name, buffer) for child in module.modules()
            for name, buffer in child.named_buffers(recurse=False)
            if name in child._non_persistent_buffers_set]
