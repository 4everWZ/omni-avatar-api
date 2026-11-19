# Omni inference gateway

Applications use a configured gateway URL and the `omni-default` model alias. On the shared Compose network, the URL is `http://omni-gateway:8000/v1`. The gateway translates the requested alias to the engine's configured model ID. Upstream response bodies and Realtime events keep their wire format. The workstation overlay registers a second backend at `/llm`: each backend has its own upstream session, health checks and admission counters. All callers of each engine share that engine’s admission.

| Endpoint | Behavior |
| --- | --- |
| `GET /livez` | Public, local process liveness. |
| `GET /healthz` | Bounded upstream `/health` and model discovery checks, plus engine identity, observed wire capabilities and admission. |
| `POST /v1/chat/prompt_tokens` | Authenticated text-only count using the same upstream Chat renderer and model alias. Returns `count` through a CPU template/tokenizer operation. |
| `GET /v1/models` | Advertises `omni-default` when the configured engine model matches. |
| `POST /v1/audio/speech` | Reads literal input with an inline reference WAV, then transcribes the resulting audio on a second admitted Omni call. Returns WAV with the verified transcript. [Speech contract](../../docs/api.md#参考音色朗读) |
| `POST /v1/chat/completions` | Forwards Chat Completions JSON, including an incremental SSE response when the engine streams. Omitted or null output modalities request text. Explicit modalities pass through. |
| `GET /v1/realtime?duplex=1&model=omni-default` | WebSocket relay for actual upstream duplex events. Forwards `autostart` and `resume` parameters. A media service session with `program=1` requests live admission for the lifetime of its socket. |
| `GET /api/mode` | Current admission mode, active lanes and queue occupancy. |
| `PUT /api/mode` | Admin-only mode update. Send `{"mode":"live"}` or `{"mode":"batch"}`. |

Except for liveness, requests require `Authorization: Bearer <token>`, including the WebSocket upgrade. The gateway attaches its own optional engine bearer and keeps it out of application responses. Use a different bearer for mode changes.

The split workstation uses two FIFO slots shared by Chat Completions and Realtime in both modes. On legacy hardware profiles, batch mode defaults to one shared slot and live mode reserves one Realtime socket alongside one Chat request. Set the `media` entry in `OMNI_GATEWAY_TOKENS` to the media bridge's dedicated bearer for automatic mode changes. A manual live setting remains active when the program disconnects. Queued requests wait for capacity within the configured deadline, then receive `429`. Closing a client connection releases its admission after closing the upstream request or socket. The engine owns cancellation and any detached-session grace period.

| Environment | Default or meaning |
| --- | --- |
| `OMNI_ENGINE_URL` | `http://omni-engine:8000` |
| `OMNI_MODEL_ID` | Required exact ID in upstream `/v1/models`. |
| `OMNI_CHECKPOINT` | Required revision recorded by the deployment. |
| `OMNI_EXPECTED_MODEL_ROOT` | Optional exact upstream `root` value. When set, the model must match both ID and root. |
| `OMNI_ENGINE_VERSION` | Deployment image version for duplex profiles that report version through image provenance. |
| `OMNI_GATEWAY_TOKEN` | Required application bearer. |
| `OMNI_GATEWAY_ADMIN_TOKEN` | Required separate administrator bearer. |
| `OMNI_GATEWAY_TOKENS` | Optional JSON object mapping additional service names to their independent tokens. |
| `OMNI_ENGINE_TOKEN` | Optional bearer for upstream only. |
| `OMNI_BATCH_CAP` | `1` |
| `OMNI_WORKSTATION_CAP` | `2` shared Chat/Realtime slots in the split workstation. |
| `OMNI_QUEUE_LIMIT` | `16` queued requests |
| `OMNI_QUEUE_TIMEOUT` | `5` seconds |
| `OMNI_REQUEST_TIMEOUT` | `1200` seconds to receive the Chat request body. After admission, a fresh total deadline of the same duration covers the upstream request and complete SSE response. Set an explicit value to override both. |
| `OMNI_HEALTH_TIMEOUT` | `3` seconds for upstream probes, connects and WebSocket handshake. |
| `OMNI_REALTIME_IDLE_TIMEOUT` | `120` seconds of application-frame inactivity across both directions. |

Health checks confirm the engine serves the configured model ID and report the engine's registered root. `checkpoint_root_verified` indicates that root matched the configured expected value. The checkpoint revision remains deployment metadata. Capabilities update after an actual successful Chat Completions response and upstream Realtime `session.created`. Duplex profiles report the pinned engine version with deployment provenance. [vLLM-Omni duplex API](https://docs.vllm.ai/projects/vllm-omni/en/v0.32.0/serving/realtime_duplex_api/)

Run the loopback tests from the repository root. They use lightweight HTTP and WebSocket fixtures.

The optional Avatar overlay sets `OMNI_AVATAR_URL` and a separate `OMNI_AVATAR_INTERNAL_TOKEN`. Public `/v1/avatar/*` requests retain the caller's identity through a private authenticated proxy. Realtime callers opt in with `avatar=1`: the gateway taps the existing duplex session and reserves live admission for that socket. The isolated tap closes rendering on slowness or failure while the original speech socket continues. Plain Realtime connections retain their original event stream. The browser page at `/` uses short-lived, single-use `/v1/browser-tickets` for authenticated WebSocket upgrades. See the [Avatar contract](../../docs/api.md#avatar-会话与音画) for ownership, WHEP and playback semantics.

```sh
uv run --no-project --with-requirements services/gateway/requirements.txt python -B -m unittest discover -s services/gateway/tests -v
```

## Thinking defaults

Omni Chat and LLM Chat/Responses default thinking on before template counting and inference. Explicit template options or `reasoning.effort: "none"` remain caller-owned. Thinking-token budget and effort follow the pinned upstream defaults. The pinned adapters forward an optional `thinking_token_budget` (nonnegative, or `-1` for unlimited reasoning within the total output cap) to the sampler for Chat and Responses; reasoning and answer tokens share the total output cap. `include_reasoning: false` hides returned reasoning; generated reasoning continues to count toward usage. These text reasoning controls apply to Chat/Responses. Native Realtime uses the model's official duplex policy. See [request fields and budget boundaries](../../docs/api.md#cot-默认与调用方预算).

## Realtime window configuration

Keep the pinned MiniCPM-o serving default: `basic`, high/low watermarks 8000/6000. The gateway forwards the caller's `session.update.session.extra_body` unchanged, including `sliding_window_mode`, basic watermarks and context retention limits. Connect with `duplex=1&autostart=0` before the initial update to configure a newly created session; window policy is immutable afterwards. Applications configure each new session through that initial update. `context` retains recent input units plus earlier model text within the model window; applications own durable memory. See the [wire example and mode boundaries](../../docs/api.md#omni-实时历史窗口).

## Long-context Responses backend

With `deploy/compose/inference.yml`, use `/llm/v1` and `llm-default`. The primary endpoint is `POST /llm/v1/responses`; Chat Completions remains compatible. `/llm/healthz`, `/llm/v1/models` and `/llm/api/mode` describe only this backend. LLM and Omni have separate admission capacity. Text-backend configuration uses `LLM_ENGINE_URL`, `LLM_MODEL_ID`, `LLM_CHECKPOINT`, optional `LLM_ENGINE_TOKEN`, and LLM-only admission/deadline overrides; see the Compose overlay for pinned values.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/llm/v1", api_key="<application bearer>")
with client.responses.stream(model="llm-default", input="你好", max_output_tokens=512) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="", flush=True)
```

HTTP Responses supports incremental SSE (`stream: true`), JSON Schema, function-call output, owned `previous_response_id`, retrieval via `GET /llm/v1/responses/{id}` and cancellation via `POST /llm/v1/responses/{id}/cancel`. The tool contract accepts function definitions and returns calls for application execution. These operations define the pinned engine's supported HTTP/SSE subset. `/llm/v1/chat/prompt_tokens` counts Chat templates; Responses admission separately renders the complete continuation lineage.

The gateway replaces caller-supplied request IDs with authenticated owner-bound response IDs. Foreign retrieval, cancellation and continuation return 404. Stored responses live in engine RAM and disappear on restart; callers retain durable business history. Foreground stream disconnection cancels upstream work. Background work, including `background: true` plus streaming, keeps its admission lease after disconnection until completion, cancellation or deadline. Keep the admin secret stable across gateway restarts to retain response ownership verification.

The deployed OpenAI Python SDK 3.35.1 also passed typed `responses.stream()`, `get_final_response()`, owned continuation and retrieval against the real backend. [SDK evidence](../../evaluations/results/responses-sdk-workstation.json).

## External orchestration clients

Multiple worker conversations can share this provider through `/llm/v1` and `llm-default`. Keep each worker's input history or `previous_response_id` chain separate. Long-running controllers can use `store: false` and persist their own history; stored continuation is process-local and ephemeral. Responses/SSE, function-call output, caller-supplied tool results, JSON Schema, retrieval and cancellation remain inference operations; the caller owns worker scheduling, handoffs, memory, tools and workspace/Git actions.

The text engine and gateway default to four generation sequences (`LLM_MAX_NUM_SEQS=4`), preserving the 262144-token per-request window. Active requests also share `LLM_TOKEN_BUDGET=262144`: reserve fully rendered input plus maximum output, including tools, instructions and stored continuation history. Chat choices consume separate sequence slots and conservative token reservations. A private engine endpoint uses the same renderer and output-limit calculation as generation. Access stays on the engine's private network.

Set explicit `max_output_tokens` (Chat: `max_completion_tokens` / `max_tokens`) for shorter requests. Omitting it preserves the engine's default output limit and can reserve the entire remaining window, serializing requests. FIFO queueing prevents a stream of short requests from starving a long request. Capacity waits time out with `429` and `Retry-After`; an individually impossible reservation returns `400`. `/llm/api/mode` exposes reserved/available tokens and active sequences. Foreground, background, disconnect and cancellation use the same resource lease. Qualification covers four short simultaneous generations under the shared budget. Multiple full-window requests require a separate capacity evaluation.

Optional `LLM_QUEUE_LIMIT`, `LLM_QUEUE_TIMEOUT` and `LLM_REQUEST_TIMEOUT` configure only this backend; unset/empty values inherit the parent gateway's values (normally 16 / 5 s / 1200 s). Omni admission remains independent. Raising sequence or token limits requires renewed GPU and portrait co-residency qualification.

The neural-portrait overlay shares the selected LLM device through CUDA MPS. The LLM has a configurable active-thread limit (default 20%) and below-normal client priority, with decode CUDA graphs for batches 1/2/4 and synchronous scheduling. The LLM also reads an expiring private render clock: below 240 ms of frame headroom, it yields up to 18 ms per dispatch and resumes early when frames catch up. Missing, invalid or stale feedback selects normal dispatch; SSE is forwarded immediately. W4A16 weights retain BF16 KV and FP32 SSM. Token reservations protect memory separately from compute sharing; see [current TPS and restart operations](../../docs/deployment/workstation.md#四路并发与计算预算).

Antigravity documents an [external OpenAI-compatible provider interface](https://antigravity.google/docs/sdk/local-models/#external-openai-compatible-servers); [Agent Orchestrator](https://github.com/OrchestratorInc/agent-orchestrator) manages agent clients as a separate layer. This service is validated as a provider, with two independently continued caller histories and concurrently submitted typed SSE requests. Application projects own integration with their selected orchestration client. [Provider fixture evidence](../../evaluations/results/orchestrator-provider-workstation.json).
