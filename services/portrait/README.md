# Reference-driven talking portrait worker

The optional `deploy/compose/portrait.yml` overlay runs this worker on physical GPU 0 alongside the long-context LLM. It uses pinned [Ditto code](https://github.com/antgroup/ditto-talkinghead/tree/c3e47eee2e626500017a0556b470d6d4182f85e8) and [PyTorch models](https://huggingface.co/digital-avatar/ditto-talkinghead/tree/e4a2f60328ee7c32af585ac4b3cce299e4c8e254). Model assets stay outside Git. Control and frame transport use a private Unix socket in a network-isolated container. Public API and ownership are defined in [docs/api.md](../../docs/api.md#参考图与会议式头像).

Build the Portrait image directly from its pinned upstream runtime. Additional dependencies live in `/opt/portrait-env`, separate from the inference environment. `portrait-assets` fetches pinned code and the selected PyTorch/Hubert assets; serving mounts them read-only and offline. Uploaded references and locally generated feature caches live in the `portrait-data` volume. The upload contract accepts PNG/JPEG; model files come from pinned downloads and serialized features are created by the server. Internal feature caches are trusted server-created pickle files.

The workstation enables one neural GPU slot by default. `OMNI_PORTRAIT_CAP=2` enables a second slot for qualification. Rendering and reference preparation share the enabled slots, each with independent render state and a frame clock. Up to four preparation jobs can be outstanding. Each caller can retain 16 images and 16 jobs, with a total limit of 64 each; clients delete unused jobs and references. Preparation cancellation marks the job cancelled immediately but holds its slot until the running inference exits. Completed avatars persist across restarts; interrupted preparations become failed. Live sessions end on restart.

A captured CUDA graph combines GPU-resident warp and decode; channels-last convolution layout preserves a contiguous boundary where the pinned model flattens depth. Graph replay is checked against the compiled forward before use. Compilation may fuse FP16 operations and change rounding; the worker records numerical differences against uncompiled FP16 output. Diffusion weights, random sampling and ten sampling steps are retained.

Compilation caches persist separately from user assets. Compilation and graph capture finish before the worker listens, including when reusing avatars after a restart.

Motion prediction uses FP16 autocast and compiled CUDA graphs, retaining the checkpoint, diffusion sampler and ten steps. Startup compares three fixed-seed input scales with the FP32 reference (`rtol=0.01`, `atol=0.005`); the private generator keeps validation sampling independent of live RNG state. Graph replay still must match the compiled forward at the original tighter tolerance. The isolated motion-step measurement is about 2.06 ms; full-session throughput is measured by generated frame counts and timeline checks. The portrait overlay shares GPU 0 through CUDA MPS, with normal renderer priority and a 20% active-thread limit on the LLM; see [deployment](../../docs/deployment/workstation.md#四路并发与计算预算).

The SDK's exact worker threads warm up before a render session is ready. The driver generates 25 fps, normalized 640×720 output, using 75-frame overlap and 10 diffusion steps. Hubert and face auxiliaries use CPU; core motion/warp/decode use GPU and the provider's mixed precision. PCM is mono 24 kHz, resampled with a continuous timeline into the provider's 16 kHz streaming windows. A 1.2-second Avatar input lead covers feature lookahead and rendering; native Omni output remains separately available.

On clear, pending PCM is removed and old generated frames are fenced by timestamp/generation. SDK threads stay alive, avoiding a new CUDA cold start on every interruption. The SDK motion state can retain a natural transition while audio playback follows the current generation. Slow transport, renderer failure, output more than 750 ms behind playback, or a two-second output stall fails the Avatar session; the source Omni socket continues. The session's `renderer.generated_frames` measures neural throughput; WHEP frame delivery is counted separately.

From the repository root, run lightweight preparation/ownership/cancellation regressions:

```sh
uv run --no-project --with aiohttp --with pillow python -B -m unittest discover -s services/portrait/tests -v
```

For real dual-GPU co-residency, use [tools/qualification/portrait.py](../../tools/qualification/portrait.py) with the Avatar dependencies and a clear single-face PNG reference. It creates a retained prepared avatar, binds native Omni speech, consumes actual WHEP media and calls the background Responses backend. Test captures support media diagnostics. Physical speaker playback and subjective lip-sync quality are covered by manual acceptance.

The render timeline additionally publishes an atomic, expiring deadline clock in the private MPS control volume (`OMNI_GPU_RENDER_CLOCK`). It contains only monotonic frame timing. The text engine cooperates when remaining headroom is below 240 ms, yielding at most 18 ms per dispatch and resuming early once frames catch up. Closing the render run removes its clock; an abandoned clock expires after 500 ms. The renderer uses 640×720 output, ten-step sampling and a 1.2-second audio input lead. [Current mixed-load evidence](../../evaluations/results/omni-throughput-tensorrt-workstation.json).

## Private TCP worker

Use `python tools/compose.py --target portrait -- build omni-portrait`, run the `portrait-assets` profile to fetch models, then `--target portrait -- up -d --wait`. Set `PORTRAIT_GPU_DEVICE` on this node. It owns its data and compiler cache volumes. The default published address is loopback:8091. For a private network, set `PORTRAIT_PUBLISH_BIND` and point Avatar's `OMNI_PORTRAIT_URL` at the reachable origin over a protected link. Both ends use the same internal bearer. Omit the workstation `--portrait` overlay on the Avatar node when using a remote worker. The colocated profile continues to use its Unix socket and MPS.

## TensorRT frame backend

`OMNI_PORTRAIT_FRAME_BACKEND` selects `cuda-graph` (portable deployment default) or `tensorrt`. The TensorRT backend combines warp and decoder in one target-built engine and captures its enqueue in the existing CUDA graph. Convolutions use FP16; keypoints and coordinate calculations use FP32. Motion prediction, ten diffusion steps, 640×720 delivery and the 1.2-second input lead follow the shared renderer configuration.

Select the TensorRT runtime stage before building the worker:

```sh
export OMNI_PORTRAIT_IMAGE=omni-api-portrait:ditto-trt
export OMNI_PORTRAIT_DOCKERFILE=Dockerfile
export OMNI_PORTRAIT_FRAME_BACKEND=tensorrt
python tools/compose.py --portrait -- build omni-portrait
```

The shared Dockerfile installs the optional TensorRT layer before copying worker source. Older installations selecting `Dockerfile.tensorrt` can switch to `Dockerfile` to use this build layout. The TensorRT requirements pin TensorRT CUDA 13 `11.4.0.106` and ONNX Script `0.7.2` inside the portrait environment. TensorRT 11.2 introduced native rank-5 GridSample, which the warp model uses. [Operator support](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.2.1.html).

For a first engine build, drain GPU 0 work and reserve the card for compilation:

```sh
python tools/compose.py --portrait -- stop llm-engine omni-portrait
python tools/compose.py --portrait -- run --rm --no-deps omni-portrait python -m portrait.tensorrt_frame
python tools/compose.py --portrait -- up -d --wait llm-engine omni-portrait
```

Engines persist under `/cache/tensorrt`. Cache identity includes the pinned code/model, GPU model and capability, TensorRT/PyTorch versions and precision policy; startup validates the tensor contract and deserialization. Build and serve with the same image on the target GPU. NVIDIA's hardware compatibility mode expands supported GPU targets while runtime compatibility follows its own contract; this backend uses target-specific builds. [Engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html).

Use [portrait_frame](../../tools/qualification/portrait_frame.py) for isolated GPU latency and pixel drift on prepared features, then the existing throughput tools for real speech, 25 fps, four LLM generations and cancellation. Current measurements are maintained in the [workstation guide](../../docs/deployment/workstation.md#tensorrt-fp16-对照).
