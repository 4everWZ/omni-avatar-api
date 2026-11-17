# Omni inference gateway

Applications use a configured gateway URL and the `omni-default` model alias. On the shared Compose network, the URL is `http://omni-gateway:8000/v1`. The gateway translates the requested alias to the engine's configured model ID. Upstream response bodies and Realtime events keep their wire format. The workstation overlay registers a second backend at `/llm`: each backend has its own upstream session, health checks and admission counters. All callers of each engine share that engine’s admission.

| Endpoint | Behavior |
| --- | --- |
| `GET /livez` | Local process liveness. This endpoint is public and does not contact the engine. |
| `GET /healthz` | Bounded upstream `/health` and model discovery checks, plus engine identity, observed wire capabilities and admission. |
| `POST /v1/chat/prompt_tokens` | Authenticated text-only count using the same upstream Chat renderer and model alias. Returns `count` without entering GPU admission. |
| `GET /v1/models` | Advertises `omni-default` when the configured engine model matches. |
| `POST /v1/chat/completions` | Forwards Chat Completions JSON, including an incremental SSE response when the engine streams. Omitted or null output modalities request text. Explicit modalities pass through. |
| `GET /v1/realtime?duplex=1&model=omni-default` | WebSocket relay for actual upstream duplex events. Forwards `autostart` and `resume` parameters. A media service session with `program=1` requests live admission for the lifetime of its socket. |
| `GET /api/mode` | Current admission mode, active lanes and queue occupancy. |
| `PUT /api/mode` | Admin-only mode update. Send `{"mode":"live"}` or `{"mode":"batch"}`. |

Except for liveness, requests require `Authorization: Bearer <token>`, including the WebSocket upgrade. The gateway attaches its own optional engine bearer and keeps it out of application responses. Use a different bearer for mode changes.

Batch mode shares a capacity of one by default. Live mode permits one ordinary Chat Completions request and reserves one separate Realtime WebSocket. Set the `media` entry in `OMNI_GATEWAY_TOKENS` to the media bridge's dedicated bearer for automatic mode changes. A manual live setting remains active when the program disconnects. Already admitted work can drain during a mode transition. Queued requests wait for capacity within the configured deadline, then receive `429`. Closing a client connection releases its admission after closing the upstream request or socket. The engine owns actual cancellation and any detached-session grace period.

| Environment | Default or meaning |
| --- | --- |
| `OMNI_ENGINE_URL` | `http://omni-engine:8000` |
| `OMNI_MODEL_ID` | Required exact ID in upstream `/v1/models`. |
| `OMNI_CHECKPOINT` | Required revision recorded by the deployment. |
| `OMNI_EXPECTED_MODEL_ROOT` | Optional exact upstream `root` value. When set, the model must match both ID and root. |
| `OMNI_ENGINE_VERSION` | Version from the deployed image if duplex mode has no `/version` endpoint. |
| `OMNI_GATEWAY_TOKEN` | Required application bearer. |
| `OMNI_GATEWAY_ADMIN_TOKEN` | Required separate administrator bearer. |
| `OMNI_GATEWAY_TOKENS` | Optional JSON object mapping additional service names to their independent tokens. |
| `OMNI_ENGINE_TOKEN` | Optional bearer for upstream only. |
| `OMNI_BATCH_CAP` | `1` |
| `OMNI_QUEUE_LIMIT` | `16` queued requests |
| `OMNI_QUEUE_TIMEOUT` | `5` seconds |
| `OMNI_REQUEST_TIMEOUT` | `1200` seconds to receive the Chat request body. After admission, a fresh total deadline of the same duration covers the upstream request and complete SSE response. Set an explicit value to override both. |
| `OMNI_HEALTH_TIMEOUT` | `3` seconds for upstream probes, connects and WebSocket handshake. |
| `OMNI_REALTIME_IDLE_TIMEOUT` | `120` seconds without application frames in either direction. |

Health checks confirm the engine serves the configured model ID and report the engine's registered root. `checkpoint_root_verified` indicates that root matched the configured expected value. The checkpoint revision remains deployment metadata. Capabilities update after an actual successful Chat Completions response and upstream Realtime `session.created`. The pinned engine version is shown with deployment provenance if this engine profile does not expose a version endpoint. [vLLM-Omni duplex API](https://docs.vllm.ai/projects/vllm-omni/en/v0.32.0/serving/realtime_duplex_api/)

Run the loopback tests from the repository root. They use an HTTP and WebSocket fixture without loading model weights.

The optional Avatar overlay sets `OMNI_AVATAR_URL` and a separate `OMNI_AVATAR_INTERNAL_TOKEN`. Public `/v1/avatar/*` requests retain the caller's identity through a private authenticated proxy. Realtime callers opt in with `avatar=1`: the gateway taps duplex output without opening another engine session and reserves live admission for that socket. A slow or failed tap closes rendering without blocking the original speech socket. Connections without the opt-in retain their original event stream. The browser page at `/` uses short-lived, single-use `/v1/browser-tickets` for authenticated WebSocket upgrades. See the [Avatar contract](../../docs/api.md#avatar-会话与音画) for ownership, WHEP and playback semantics.

```sh
uv run --no-project --with-requirements services/gateway/requirements.txt python -B -m unittest discover -s services/gateway/tests -v
```

## Long-context Responses backend

With `deploy/compose/inference.yml`, use `/llm/v1` and `llm-default`. The primary endpoint is `POST /llm/v1/responses`; Chat Completions remains compatible. `/llm/healthz`, `/llm/v1/models` and `/llm/api/mode` describe only this backend. LLM capacity does not consume Omni admission. Text-backend configuration uses `LLM_ENGINE_URL`, `LLM_MODEL_ID`, `LLM_CHECKPOINT`, optional `LLM_ENGINE_TOKEN`, and LLM-only admission/deadline overrides; see the Compose overlay for pinned values.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/llm/v1", api_key="<application bearer>")
with client.responses.stream(model="llm-default", input="你好", max_output_tokens=512) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

HTTP Responses supports incremental SSE (`stream: true`), JSON Schema, function-call output, owned `previous_response_id`, retrieval via `GET /llm/v1/responses/{id}` and cancellation via `POST /llm/v1/responses/{id}/cancel`. Only function tools are accepted; the service executes no tools. Hosted tools, conversation IDs, item references, cloud WebSocket/compact/delete endpoints are outside the pinned engine's supported subset. `/llm/v1/chat/prompt_tokens` counts Chat templates, not the entire Responses continuation lineage.

The gateway replaces caller-supplied request IDs with authenticated owner-bound response IDs. Foreign retrieval, cancellation and continuation return 404. Stored responses live in engine RAM and disappear on restart; callers retain durable business history. Foreground stream disconnection cancels upstream work. Background work, including `background: true` plus streaming, keeps its admission lease after disconnection until completion, cancellation or deadline. Keep the admin secret stable across gateway restarts to retain response ownership verification.

The deployed OpenAI Python SDK 3.35.1 also passed typed `responses.stream()`, `get_final_response()`, owned continuation and retrieval against the real backend. [SDK evidence](../../evaluations/results/responses-sdk-workstation.json).

## External orchestration clients

Multiple worker conversations can share this provider through `/llm/v1` and `llm-default`. Keep each worker's input history or `previous_response_id` chain separate. Long-running controllers can use `store: false` and persist their own history; stored continuation is process-local and ephemeral. Responses/SSE, function-call output, caller-supplied tool results, JSON Schema, retrieval and cancellation remain inference operations; the caller owns worker scheduling, handoffs, memory, tools and workspace/Git actions.

The text engine and gateway default to four generation sequences (`LLM_MAX_NUM_SEQS=4`), preserving the 262144-token per-request window. Active requests also share `LLM_TOKEN_BUDGET=262144`: reserve fully rendered input plus maximum output, including tools, instructions and stored continuation history. Chat choices consume separate sequence slots and conservative token reservations. A private engine endpoint uses the same renderer and output-limit calculation as generation. It is not published by the gateway.

Set explicit `max_output_tokens` (Chat: `max_completion_tokens` / `max_tokens`) for shorter requests. Omitting it preserves the engine's default output limit and can reserve the entire remaining window, serializing requests. FIFO queueing prevents a stream of short requests from starving a long request. Capacity waits time out with `429` and `Retry-After`; an individually impossible reservation returns `400`. `/llm/api/mode` exposes reserved/available tokens and active sequences. Foreground, background, disconnect and cancellation use the same resource lease. Four short simultaneous generations are the qualification target; four full-window requests are outside this memory budget.

Optional `LLM_QUEUE_LIMIT`, `LLM_QUEUE_TIMEOUT` and `LLM_REQUEST_TIMEOUT` configure only this backend; unset/empty values inherit the parent gateway's values (normally 16 / 5 s / 1200 s). Omni admission remains independent. Raising sequence or token limits requires renewed GPU and portrait co-residency qualification.

The neural-portrait overlay sets `LLM_GPU_STEP_PAUSE_MS=250`: synchronous GPU batches yield compute after completion, while retaining up to four sequences per batch. This is engine scheduling, not delayed SSE delivery. The measured portrait-priority profile produces about 3.4 tokens/s per request for four simultaneous 256-token outputs. Deploying without the portrait overlay leaves this pause disabled by default. Token reservations protect memory; batch yielding separately protects the renderer's compute budget.

Antigravity documents an [external OpenAI-compatible provider interface](https://antigravity.google/docs/sdk/local-models/#external-openai-compatible-servers); [Agent Orchestrator](https://github.com/OrchestratorInc/agent-orchestrator) manages agent clients as a separate layer. This service is validated as a provider, with two independently continued caller histories and concurrently submitted typed SSE requests. Specific orchestration clients have not been integrated here. [Provider fixture evidence](../../evaluations/results/orchestrator-provider-workstation.json).
