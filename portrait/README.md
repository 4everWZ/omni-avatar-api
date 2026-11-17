# Reference-driven talking portrait worker

The optional `infra-portrait.compose.yml` overlay runs this worker on physical GPU 0 alongside the long-context LLM. It uses pinned [Ditto code](https://github.com/antgroup/ditto-talkinghead/tree/c3e47eee2e626500017a0556b470d6d4182f85e8) and [PyTorch models](https://huggingface.co/digital-avatar/ditto-talkinghead/tree/e4a2f60328ee7c32af585ac4b3cce299e4c8e254). Model assets stay outside Git. The worker has no network interface; control and frame transport use a private Unix socket. Public API and ownership are defined in [docs/api.md](../docs/api.md#参考图与会议式头像).

Build the existing inference image first, then this image. Additional dependencies live in `/opt/portrait-env`, separate from the inference environment. `portrait-assets` fetches pinned code and the selected PyTorch/Hubert assets; serving mounts them read-only and offline. Uploaded references and locally generated feature caches live in the `portrait-data` volume. Uploaded data is decoded only as PNG/JPEG; users never supply model files or serialized features. Internal feature caches are trusted server-created pickle files.

There is one neural GPU lease across active rendering and reference preparation. Up to four preparation jobs can wait. Each caller can retain 16 images and 16 jobs, with a total limit of 64 each; clients delete unused jobs and references. Preparation cancellation marks the job cancelled immediately but holds the GPU lease until the running inference actually exits. Completed avatars persist across restarts; interrupted preparations become failed. Live sessions end on restart.

A captured CUDA graph combines warp and decode without CPU intermediate copies; channels-last convolution layout preserves a contiguous boundary where the pinned model flattens depth. Graph replay is checked against the compiled forward before use. Compilation may fuse FP16 operations and change rounding; the worker records the difference from uncompiled FP16 output rather than claiming bitwise equality. Diffusion weights, random sampling and ten sampling steps are retained.

Compilation caches persist separately from user assets. Compilation and graph capture finish before the worker listens, including when reusing avatars after a restart.

The SDK's exact worker threads warm up before a render session is ready. The driver generates 25 fps, normalized 640×720 output, using 75-frame overlap and 10 diffusion steps. Hubert and face auxiliaries use CPU; core motion/warp/decode use GPU and the provider's mixed precision. PCM is mono 24 kHz, resampled with a continuous timeline into the provider's 16 kHz streaming windows. A 1.2-second Avatar input lead covers feature lookahead and rendering; native Omni output remains separately available.

On clear, pending PCM is removed and old generated frames are fenced by timestamp/generation. SDK threads stay alive, avoiding a new CUDA cold start on every interruption. The SDK motion state can retain a natural motion transition; old speech audio is never replayed. Slow transport, renderer failure, output more than 750 ms behind playback, or a two-second output stall fails the Avatar session; the source Omni socket continues. WHEP's transport fps alone does not prove neural inference throughput: the session's `renderer.generated_frames` counts real generated frames.

Run preparation/ownership/cancellation regressions without model weights:

```sh
uv run --no-project --with aiohttp --with pillow python -B -m unittest discover -s tests -v
```

For real dual-GPU co-residency, use [tools/qualify_portrait.py](../tools/qualify_portrait.py) with the Avatar dependencies and a clear single-face PNG reference. It creates a retained prepared avatar, binds native Omni speech, consumes actual WHEP media and calls the background Responses backend. Test captures are diagnostic images; frame delivery is not proof of physical speaker playback or subjective lip-sync quality.
