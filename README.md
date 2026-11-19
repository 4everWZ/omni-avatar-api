# Omni Avatar API

[English](README.md) | [简体中文](README.zh-CN.md)

Shared, self-hosted inference and talking-avatar services for real-time assistants and applications that need a long-context LLM. One authenticated gateway exposes multimodal Omni, a separate Responses API backend, and pluggable avatar rendering.

Applications own their users, conversation memory, agent orchestration and tool execution. This repository provides the inference and media services those applications call.

## Features

- **Real-time multimodal assistant:** MiniCPM-o Chat Completions and native duplex Realtime with an 8K recent-context window.
- **Long-context inference:** a separate Qwen3.5 backend with a 262K per-request window, up to four generation sequences under a shared token budget, Responses, incremental SSE, structured output and function-call output. Chat Completions remains available for compatible clients.
- **CPU avatars** support built-in Ava 2D sprites and registered Inochi2D rigs. WHEP delivers synchronized audio and video with interruption support.
- **Shared infrastructure:** per-application bearer credentials, independent backend admission, bounded queues, cancellation and caller-owned response continuation.
- **Pinned deployments:** model revisions and inference images are fixed; model downloads are separate from offline serving.

## Architecture

| Component | Responsibility |
| --- | --- |
| Gateway | Public authentication, model aliases, streaming proxies and independent backend admission. |
| Omni engine | Low-latency voice/vision interaction and recent conversation context. |
| LLM engine | Long-context text inference for external applications and orchestrators. |
| Avatar service | Owned sessions, audio timeline, renderer lifecycle and WHEP media. |

The validated workstation layout uses **two RTX 3090s**: Omni runs on one selected device. The text LLM uses the other, which can also serve queued Klein image jobs. Avatar rendering uses the host CPU. AGX Orin profiles provide a separate hardware evaluation path. See [architecture](docs/architecture.md) and [deployment profiles](docs/deployment/workstation.md).

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
python tools/compose.py -- build omni-engine llm-engine omni-gateway omni-avatar
python tools/compose.py -- --profile assets run --rm model-fetch
python tools/compose.py -- --profile assets run --rm llm-model-fetch
python tools/compose.py -- up -d --wait
```

For independent nodes, the Compose entrypoint also supports `--target gateway|omni|llm|avatar|images` (select one). Configure private endpoints and device placement in the [workstation guide](docs/deployment/workstation.md).

The default stack includes Omni, the text LLM, the gateway and Ava 2D. The API listens on **`http://127.0.0.1:8000`**. The browser demo is served at `/`. WHEP media requires direct LAN/VPN connectivity; remote microphone and screen capture need a trusted browser context.

The entrypoint anchors `.env`, build contexts and bind mounts to the repository root. Add `--vpn` to publish on `OMNI_GATEWAY_VPN_BIND`, or `--media` for optional MediaMTX transport. Alternative hardware/evaluation stacks use `--target`. See the [workstation guide](docs/deployment/workstation.md) for profile configuration and service updates.

## API

All inference and avatar endpoints require an application bearer token. Administrative credentials are separate.

| Base/path | Model | Use |
| --- | --- | --- |
| `/v1/chat/completions` | `omni-default` | Multimodal Chat, including streamed text. |
| `/v1/realtime?duplex=1&model=omni-default` | `omni-default` | Native duplex WebSocket. |
| `/llm/v1/responses` | `llm-default` | Long-context inference and incremental Responses SSE. |
| `/llm/v1/chat/completions` | `llm-default` | Chat-compatible clients. |
| `/v1/avatar/*` | — | Registered rigs, sessions, audio and WHEP. |
| `/v1/images/jobs` | Klein 4B | Optional queued image generation with reference images. |
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
    extra_body={"thinking_token_budget": 128},
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
  llm/                    Long-context text engine and vLLM patches
  inference/              Launchers, model configs, pinned patches and images
  avatar/                 Sessions, renderers and WebRTC media
deploy/                   Compose layers and optional media configuration
tools/                    Deployment, qualification and diagnostic entrypoints
tests/                    Cross-service and deployment-tooling tests
docs/                     Architecture, API, development and deployment guides
evaluations/              Shared fixtures and retained real-device evidence
```

`data/`, `runtime/`, `tmp/` and `.env` are local, ignored artifacts. Service dependencies are isolated by image; services communicate through HTTP/WebSocket/Unix-socket contracts.

## Development and validation

See [development](docs/development.md) for dependency-isolated tests and local service entrypoints, and [tools](tools/README.md) for real-device qualification commands.

Recorded checks cover 8K Omni recall, Responses continuation, about 22K tokens of LLM history, and four simultaneous shorter generations in the historical GPU-sharing configuration.

Omni Chat and LLM Chat/Responses default to CoT, with upstream thinking-budget defaults. Native duplex retains the official model policy. Realtime defaults to `basic` (8000/6000); applications can override window policy at session creation and text reasoning budgets per request. See the [API](docs/api.md#cot-默认与调用方预算).

The 256-token fixture measures total generation throughput and can spend its budget on reasoning. Separate checks cover completed CoT answers, budgets, JSON Schema and tool continuation. See the [workstation guide](docs/deployment/workstation.md#四路并发与计算预算) for formulas, raw records and validation coverage. The [acceptance plan](docs/plans/avatar-workstation.md) tracks full 262K quality, sustained load and physical speaker/microphone playback.

## Documentation

- [API contract](docs/api.md)
- [Architecture and module boundaries](docs/architecture.md)
- [Workstation deployment and operations](docs/deployment/workstation.md)
- [AGX Orin evaluation](docs/deployment/orin.md) · [JetPack 6 / CUDA 12.9](docs/deployment/jetpack6-cu129.md)
- [Development](docs/development.md) · [Deployment and qualification tools](tools/README.md)
