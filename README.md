# Omni Avatar API

[English](README.md) | [简体中文](README.zh-CN.md)

Shared, self-hosted inference and talking-avatar services for real-time assistants and applications that need a long-context LLM. One authenticated gateway exposes multimodal Omni, a separate Responses API backend, and pluggable avatar rendering.

Applications own their users, conversation memory, agent orchestration and tool execution. This repository provides the inference and media services those applications call.

## Features

- **Real-time multimodal assistant:** MiniCPM-o Chat Completions and native duplex Realtime with an 8K recent-context window.
- **Long-context inference:** a separate Qwen3.5 backend with a 262K per-request window, up to four generation sequences under a shared token budget, Responses, incremental SSE, structured output and function-call output. Chat Completions remains available for compatible clients.
- **Pluggable avatars:** built-in Ava 2D, or a reference-image-driven Ditto talking portrait; audio and video are delivered over WHEP with interruption and generation handling.
- **Shared infrastructure:** per-application bearer credentials, independent backend admission, bounded queues, cancellation and caller-owned response continuation.
- **Pinned deployments:** model revisions and inference images are fixed; model downloads are separate from offline serving.

## Architecture

| Component | Responsibility |
| --- | --- |
| Gateway | Public authentication, model aliases, streaming proxies and independent backend admission. |
| Omni engine | Low-latency voice/vision interaction and recent conversation context. |
| LLM engine | Long-context text inference for external applications and orchestrators. |
| Avatar service | Owned sessions, audio timeline, renderer lifecycle and WHEP media. |
| Portrait worker | Optional GPU-backed reference preparation and neural frame generation. |

The validated workstation layout uses **two RTX 3090s**: GPU 1 runs Omni; GPU 0 runs the text LLM and the optional portrait worker. Ava 2D rendering uses the host CPU. Historical AGX Orin profiles remain available for reproduction. See [architecture](docs/architecture.md) and [deployment profiles](docs/deployment/workstation.md).

## Quick start

Run the deployment on a Linux NVIDIA workstation with Docker Engine, a recent Docker Compose plugin, NVIDIA Container Toolkit and Python 3.12+. Python is used by the deployment entrypoint; application dependencies run inside containers.

From the repository root:

```sh
cp .env.example .env
```

Set **different random values** for `OMNI_GATEWAY_TOKEN`, `OMNI_GATEWAY_ADMIN_TOKEN` and `OMNI_AVATAR_INTERNAL_TOKEN`. Set `OMNI_MODEL_DATA` to storage with enough space for the pinned model snapshots. The [.env example](.env.example) documents the available settings.

Create the shared application network once, then build, download and start:

```sh
docker network create tuber-omni
python tools/compose.py -- build omni-engine omni-gateway omni-avatar
python tools/compose.py -- --profile assets run --rm model-fetch
python tools/compose.py -- --profile assets run --rm llm-model-fetch
python tools/compose.py -- up -d --wait
```

The default stack includes Omni, the text LLM, the gateway and Ava 2D. The API listens on **`http://127.0.0.1:8000`**. The browser demo is served at `/`. WHEP media requires direct LAN/VPN connectivity; remote microphone and screen capture need a trusted browser context.

To add neural portraits:

```sh
python tools/compose.py --portrait -- build omni-portrait
python tools/compose.py --portrait -- --profile assets run --rm portrait-assets
python tools/compose.py --portrait -- up -d --wait
```

The entrypoint anchors `.env`, build contexts and bind mounts to the repository root. Add `--vpn` to publish on `OMNI_GATEWAY_VPN_BIND`, or `--media` for optional MediaMTX transport. Alternative hardware/evaluation stacks use `--target`. See the [workstation guide](docs/deployment/workstation.md) before changing profiles or migrating an existing installation.

## API

All inference and avatar endpoints require an application bearer token. Administrative credentials are separate.

| Base/path | Model | Use |
| --- | --- | --- |
| `/v1/chat/completions` | `omni-default` | Multimodal Chat, including streamed text. |
| `/v1/realtime?duplex=1&model=omni-default` | `omni-default` | Native duplex WebSocket. |
| `/llm/v1/responses` | `llm-default` | Long-context inference and incremental Responses SSE. |
| `/llm/v1/chat/completions` | `llm-default` | Chat-compatible clients. |
| `/v1/avatar/*` | — | Reference uploads, preparation jobs, sessions, audio and WHEP. |
| `/healthz`, `/llm/healthz` | — | Authenticated backend identity, readiness and admission. |

For example, with the OpenAI Python SDK and an application token in the client environment:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/llm/v1",
    api_key=os.environ["OMNI_GATEWAY_TOKEN"],
)
with client.responses.stream(
    model="llm-default",
    input="Say hello in one sentence.",
    max_output_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

The service implements the pinned engine's HTTP/SSE Responses subset. Function tools are returned for the caller to execute; response continuation is process-local. See the [API contract](docs/api.md) and [gateway reference](services/gateway/README.md) for supported operations, ownership and cancellation.

## Repository layout

```text
services/                 Independently built service boundaries
  gateway/                Authentication and inference/media proxies
  inference/              Launchers, model configs, pinned patches and images
  avatar/                 Sessions, renderers and WebRTC media
  portrait/               Reference preparation and neural rendering
deploy/                   Compose layers and optional media configuration
tools/                    Deployment, qualification and diagnostic entrypoints
tests/                    Cross-service and deployment-tooling tests
docs/                     Architecture, API, development and deployment guides
evaluations/              Shared fixtures and retained real-device evidence
```

`data/`, `runtime/`, `tmp/` and `.env` are local, ignored artifacts. Service dependencies are isolated by image; services communicate through HTTP/WebSocket/Unix-socket contracts rather than importing each other's implementation.

## Development and validation

See [development](docs/development.md) for dependency-isolated tests and local service entrypoints, and [tools](tools/README.md) for real-device qualification commands.

Recorded checks cover 8K Omni recall, Responses continuation, about 22K tokens of LLM history, and four simultaneous shorter generations with a 25 fps neural portrait.

Current LLM throughput uses W4A16 weights, CUDA MPS (LLM active-thread limit 20%) and decode CUDA graphs, with no fixed batch sleep. Short requests have 29 input tokens and 256 output tokens, temperature 0 and thinking disabled:

| Concurrent requests | Portrait activity | Per-request output TPS | Aggregate output TPS |
| --- | --- | --- | --- |
| 1 | Resident, idle | 47.18 | 47.18 |
| 4 | Resident, idle | 37.12–37.27 | 148.47 |
| 1 | Active native speech and 25 fps neural portrait | 22.92–23.90 | 22.92–23.90 |
| 4 | Active native speech and 25 fps neural portrait | 18.68–19.42 | 74.73–77.34 |

TPS includes prefill and waiting, measured from the group's request start to completion. Idle single-request decode throughput alone is 47.75 TPS. The portrait produced 2,269 real neural frames in the 90-second check, with a maximum sampled timeline lag of 79 ms. See the [workstation guide](docs/deployment/workstation.md#四路并发与计算预算) for calculations, raw records and MPS restart instructions. These short-prompt measurements do not establish full 262K quality or sustained-load throughput; physical speaker/microphone playback remains a manual check.

## Documentation

- [API contract](docs/api.md)
- [Architecture and module boundaries](docs/architecture.md)
- [Workstation deployment and migration](docs/deployment/workstation.md)
- [AGX Orin reproduction](docs/deployment/orin.md) · [JetPack 6 / CUDA 12.9](docs/deployment/jetpack6-cu129.md)
- [Development](docs/development.md) · [Deployment and qualification tools](tools/README.md)
