# Omni inference gateway

Applications use a configured gateway URL and the `omni-default` model alias. On the shared Compose network, the URL is `http://omni-gateway:8000/v1`. The gateway translates the requested alias to the engine's configured model ID. Upstream response bodies and Realtime events keep their wire format. Deploy one gateway process for each upstream engine so its admission counts cover all callers of that engine.

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

Run the loopback tests from this directory. They use an HTTP and WebSocket fixture without loading model weights.

```sh
uv run --no-project --with-requirements requirements.txt python -B -m unittest discover -s tests -v
```
