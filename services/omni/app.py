"""HTTP and WebSocket application for the CPU full-duplex omni-engine."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
import secrets
from typing import Any, Callable

from aiohttp import ClientSession, ClientTimeout, DummyCookieJar, web

from .duplex import (
    DuplexConfig,
    DuplexSession,
    _extract_responses_text,
    _iter_sse_events,
    strip_audio_data_url,
    synthesize_tts_wav,
    transcribe_audio_wav,
)
from .model import omni_identity
from .vad import VadDetector

MAX_BODY = 32 * 1024 * 1024
MAX_WS_FRAME = 8 * 1024 * 1024
ENGINE_VERSION = "omni-duplex 1.0"
UNAUTHENTICATED_PATHS = frozenset({"/livez", "/health"})


def failure(status: int, code: str, message: str, *, error_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"type": error_type, "code": code, "message": message}},
        status=status,
    )


def config_from_env() -> tuple[DuplexConfig, str, int]:
    """Load DuplexConfig, optional OMNI_ENGINE_TOKEN, and OMNI_WORKSTATION_CAP from env."""
    model_id, checkpoint, max_model_len = omni_identity()
    cap = int(os.environ.get("OMNI_WORKSTATION_CAP", "4"))
    if cap < 1:
        raise ValueError("OMNI_WORKSTATION_CAP must be >= 1")
    barge_in_min_ms = int(os.environ.get("OMNI_BARGE_IN_MIN_MS", "160"))
    cfg = DuplexConfig(
        model_id=model_id,
        checkpoint=checkpoint,
        max_model_len=max_model_len,
        asr_url=os.environ.get("OMNI_ASR_WORKER_URL", "").strip().rstrip("/"),
        asr_token=os.environ.get("OMNI_ASR_INTERNAL_TOKEN", ""),
        asr_timeout=float(os.environ.get("OMNI_ASR_TIMEOUT", "60")),
        tts_url=os.environ.get("OMNI_TTS_WORKER_URL", "").strip().rstrip("/"),
        tts_token=os.environ.get("OMNI_TTS_INTERNAL_TOKEN", ""),
        tts_timeout=float(os.environ.get("OMNI_TTS_TIMEOUT", "60")),
        llm_responses_url=os.environ.get("OMNI_LLM_RESPONSES_URL", "").strip(),
        llm_chat_url=os.environ.get("OMNI_LLM_CHAT_URL", "").strip(),
        llm_tokenize_url=os.environ.get("OMNI_LLM_TOKENIZE_URL", "").strip(),
        llm_api_key=os.environ.get("OMNI_LLM_API_KEY", ""),
        llm_model=os.environ.get("OMNI_LLM_MODEL", "llm-default").strip() or "llm-default",
        llm_timeout=float(os.environ.get("OMNI_LLM_TIMEOUT", "120")),
        barge_in_min_ms=barge_in_min_ms,
    )
    engine_token = os.environ.get("OMNI_ENGINE_TOKEN", "")
    return cfg, engine_token, cap


def _estimate_prompt_tokens(messages: Any) -> int:
    total_chars = 0
    image_count = 0
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("text", "input_text", "output_text"):
                        total_chars += len(str(part.get("text", "")))
                    elif part.get("type") in ("image_url", "input_image"):
                        image_count += 1
    return max(1, (total_chars // 3) + (image_count * 256) + 8)


def _has_image_parts(messages: list[Any]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "input_image"):
                    return True
    return False


def _is_direct_asr_request(messages: list[Any], modalities: list[str], audio_transcripts: list[str]) -> bool:
    if os.environ.get("OMNI_DIRECT_AUDIO_TRANSCRIPTION", "1") == "0":
        return False
    if "audio" in modalities or len(audio_transcripts) != 1:
        return False
    non_system = [m for m in messages if isinstance(m, dict) and m.get("role") != "system"]
    if len(non_system) != 1 or non_system[0].get("role") != "user":
        return False
    content = non_system[0].get("content")
    if not isinstance(content, list):
        return False
    texts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
    combined = " ".join(texts).strip()
    lower = combined.lower()
    if not combined:
        return True
    keywords = ("逐字", "转写", "记录", "transcrib", "verbatim")
    return len(combined) <= 200 and any(kw in lower for kw in keywords)


def _chat_to_responses_payload(
    payload: dict[str, Any],
    messages: list[dict[str, Any]],
    *,
    llm_model: str,
    stream: bool,
) -> dict[str, Any]:
    instructions_parts: list[str] = []
    input_items: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str):
                instructions_parts.append(content)
            elif isinstance(content, list):
                instructions_parts.extend(
                    str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            continue
        text_type = "output_text" if role == "assistant" else "input_text"
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"type": text_type, "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("text", "input_text", "output_text"):
                    parts.append({"type": text_type, "text": str(part.get("text", ""))})
                elif ptype == "image_url":
                    img = part.get("image_url")
                    url = img.get("url") if isinstance(img, dict) else img
                    if isinstance(url, str) and url:
                        item_img: dict[str, Any] = {"type": "input_image", "image_url": url}
                        if isinstance(img, dict) and img.get("detail"):
                            item_img["detail"] = img["detail"]
                        parts.append(item_img)
                elif ptype == "input_image" and part.get("image_url"):
                    parts.append({"type": "input_image", "image_url": part["image_url"]})
        if parts:
            input_items.append({"role": role, "content": parts})

    out: dict[str, Any] = {
        "model": llm_model,
        "input": input_items,
        "store": False,
        "stream": stream,
    }
    if instructions_parts:
        out["instructions"] = "\n\n".join(p for p in instructions_parts if p)
    if "chat_template_kwargs" in payload:
        out["chat_template_kwargs"] = payload["chat_template_kwargs"]
    if "temperature" in payload:
        out["temperature"] = payload["temperature"]
    max_tok = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if max_tok is not None:
        out["max_output_tokens"] = max_tok
    if isinstance(payload.get("response_format"), dict):
        rf = payload["response_format"]
        if rf.get("type") == "json_schema" and isinstance(rf.get("json_schema"), dict):
            js = rf["json_schema"]
            out["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": js.get("name", "response"),
                    "schema": js.get("schema", {}),
                    "strict": bool(js.get("strict", True)),
                }
            }
        elif rf.get("type") == "json_object":
            out["text"] = {"format": {"type": "json_object"}}
    return out


def create_app(
    config: DuplexConfig | None = None,
    *,
    vad_factory: Callable[[], VadDetector] | None = None,
    engine_token: str = "",
    workstation_cap: int | None = None,
    vad_ready: bool = True,
) -> web.Application:
    """Create the aiohttp Application for CPU omni-engine."""
    if config is None:
        config, env_token, env_cap = config_from_env()
        if not engine_token:
            engine_token = env_token
        if workstation_cap is None:
            workstation_cap = env_cap
    cap = int(workstation_cap if workstation_cap is not None else 4)
    if cap < 1:
        raise ValueError("workstation_cap must be >= 1")
    if vad_factory is None:
        from .vad import EnergyVad

        vad_factory = lambda: EnergyVad(barge_in_min_ms=config.barge_in_min_ms)

    realtime_semaphore = asyncio.Semaphore(cap)
    state: dict[str, Any] = {"http": None}

    @web.middleware
    async def authenticate(request: web.Request, handler):
        if engine_token and request.path not in UNAUTHENTICATED_PATHS:
            expected = f"Bearer {engine_token}".encode()
            provided = request.headers.get("Authorization", "").encode()
            if not secrets.compare_digest(provided, expected):
                return failure(401, "unauthorized", "A valid engine bearer token is required.")
        return await handler(request)

    async def lifecycle(app: web.Application):
        async with ClientSession(
            cookie_jar=DummyCookieJar(),
            timeout=ClientTimeout(total=None, connect=10),
            trust_env=False,
        ) as session:
            state["http"] = session
            yield

    async def livez(_request: web.Request) -> web.Response:
        return web.json_response({"state": "running"})

    async def health(_request: web.Request) -> web.Response:
        # Never block /health on omni-gateway to avoid circular Compose healthcheck deadlocks.
        if not vad_ready:
            return web.json_response(
                {"status": "unavailable", "engine": "omni-duplex", "vad_ready": False},
                status=503,
            )
        return web.json_response({"status": "ready", "engine": "omni-duplex", "vad_ready": True})

    async def version(_request: web.Request) -> web.Response:
        return web.json_response({"version": ENGINE_VERSION})

    async def models(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": config.model_id,
                        "object": "model",
                        "root": f"/models/{config.checkpoint}",
                        "max_model_len": config.max_model_len,
                    }
                ],
            }
        )

    async def prompt_tokens(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except ValueError:
            return failure(400, "invalid_json", "Expected a JSON payload.")
        if not isinstance(payload, dict):
            return failure(400, "invalid_request", "Expected a JSON object.")

        messages = payload.get("messages") or []
        headers = {"Authorization": f"Bearer {config.llm_api_key}"} if config.llm_api_key else {}
        http: ClientSession = state["http"]

        if isinstance(messages, list) and _has_image_parts(messages) and config.llm_responses_url:
            resp_payload = _chat_to_responses_payload(
                payload, messages, llm_model=config.llm_model, stream=False
            )
            try:
                async with asyncio.timeout(10):
                    async with http.post(
                        config.llm_responses_url.rstrip("/") + "/prompt_tokens",
                        json=resp_payload,
                        headers=headers,
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data.get("count"), int):
                                return web.json_response({"count": data["count"]})
            except Exception:
                pass
        elif config.llm_tokenize_url:
            fwd = {
                k: v
                for k, v in payload.items()
                if k in {"messages", "tools", "chat_template_kwargs"}
            }
            fwd["model"] = config.llm_model
            try:
                async with asyncio.timeout(10):
                    async with http.post(config.llm_tokenize_url, json=fwd, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data.get("count"), int):
                                return web.json_response({"count": data["count"]})
            except Exception:
                pass

        return web.json_response({"count": _estimate_prompt_tokens(messages)})

    async def _resolve_audio_inputs(messages: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
        http: ClientSession = state["http"]
        resolved: list[dict[str, Any]] = []
        transcripts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                resolved.append(dict(msg))
                continue
            new_parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "input_audio":
                    if isinstance(part, dict):
                        new_parts.append(part)
                    continue
                audio_info = part.get("input_audio")
                if not isinstance(audio_info, dict) or not isinstance(audio_info.get("data"), str):
                    raise ValueError("input_audio part must contain base64 'data'")
                fmt = str(audio_info.get("format") or "wav").strip().lower() or "wav"
                raw_bytes = base64.b64decode(audio_info["data"], validate=True)
                transcript = await transcribe_audio_wav(
                    http,
                    asr_url=config.asr_url,
                    asr_token=config.asr_token,
                    wav_bytes=raw_bytes,
                    timeout=config.asr_timeout,
                    filename=f"input.{fmt}",
                )
                transcripts.append(transcript)
                new_parts.append({"type": "text", "text": f"[Audio transcript]: {transcript}"})
            resolved.append(dict(msg) | {"content": new_parts})
        return resolved, transcripts

    async def _chat_sse_single(request: web.Request, content: str) -> web.StreamResponse:
        chunk_id = f"chatcmpl-{secrets.token_hex(8)}"
        out = web.StreamResponse(
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        await out.prepare(request)
        first = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": config.model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
        }
        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "model": config.model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await out.write(f"data: {json.dumps(first, ensure_ascii=False)}\n\n".encode("utf-8"))
        await out.write(f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode("utf-8"))
        await out.write(b"data: [DONE]\n\n")
        await out.write_eof()
        return out

    async def chat_completions(request: web.Request) -> web.StreamResponse:
        try:
            payload = await request.json()
        except ValueError:
            return failure(400, "invalid_json", "Expected a JSON Chat Completions payload.")
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return failure(400, "invalid_request", "Chat Completions requires a 'messages' list.")

        modalities = [str(m) for m in (payload.get("modalities") or ["text"])]
        wants_audio = "audio" in modalities
        stream = bool(payload.get("stream", False))

        try:
            resolved_messages, audio_transcripts = await _resolve_audio_inputs(payload["messages"])
        except (ValueError, binascii.Error) as exc:
            return failure(400, "invalid_audio", str(exc))
        except Exception as exc:
            logging.exception("Chat input_audio ASR failed")
            return failure(502, "asr_unavailable", str(exc), error_type="server_error")

        if _is_direct_asr_request(payload["messages"], modalities, audio_transcripts):
            transcript = audio_transcripts[0]
            structured = payload.get("structured_outputs")
            if isinstance(structured, dict) and isinstance(structured.get("json"), dict):
                props = structured["json"].get("properties") or {}
                if "transcript" in props:
                    transcript = json.dumps({"transcript": transcript}, ensure_ascii=False)
            if stream:
                return await _chat_sse_single(request, transcript)
            return web.json_response(
                {
                    "id": f"chatcmpl-{secrets.token_hex(8)}",
                    "object": "chat.completion",
                    "model": config.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": transcript},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )

        http: ClientSession = state["http"]
        llm_headers = {"Authorization": f"Bearer {config.llm_api_key}"} if config.llm_api_key else {}
        ref_audio_b64: str | None = None
        if wants_audio and isinstance(payload.get("ref_audio"), str):
            try:
                ref_audio_b64 = strip_audio_data_url(payload["ref_audio"])
            except (ValueError, binascii.Error) as exc:
                return failure(400, "invalid_ref_audio", str(exc))

        use_responses = bool(config.llm_responses_url) and (
            _has_image_parts(resolved_messages) or not config.llm_chat_url
        )

        if use_responses:
            if not config.llm_responses_url:
                return failure(503, "llm_unavailable", "OMNI_LLM_RESPONSES_URL is not configured.")
            use_stream = stream and not wants_audio
            resp_payload = _chat_to_responses_payload(
                payload, resolved_messages, llm_model=config.llm_model, stream=use_stream
            )
            try:
                async with asyncio.timeout(config.llm_timeout):
                    async with http.post(
                        config.llm_responses_url, json=resp_payload, headers=llm_headers
                    ) as upstream:
                        if upstream.status >= 400:
                            body = await upstream.read()
                            return web.Response(
                                status=upstream.status,
                                body=body,
                                content_type=upstream.headers.get("Content-Type", "application/json"),
                            )
                        if use_stream:
                            chunk_id = f"chatcmpl-{secrets.token_hex(8)}"
                            out = web.StreamResponse(
                                headers={
                                    "Content-Type": "text/event-stream",
                                    "Cache-Control": "no-cache",
                                    "X-Accel-Buffering": "no",
                                }
                            )
                            await out.prepare(request)
                            async for ev_name, ev_obj in _iter_sse_events(upstream):
                                ev_type = ev_obj.get("type") or ev_name
                                if ev_type == "response.output_text.delta":
                                    delta = str(ev_obj.get("delta", "") or "")
                                    if delta:
                                        chunk = {
                                            "id": chunk_id,
                                            "object": "chat.completion.chunk",
                                            "model": config.model_id,
                                            "choices": [
                                                {
                                                    "index": 0,
                                                    "delta": {"content": delta},
                                                    "finish_reason": None,
                                                }
                                            ],
                                        }
                                        await out.write(
                                            f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")
                                        )
                            end_chunk = {
                                "id": chunk_id,
                                "object": "chat.completion.chunk",
                                "model": config.model_id,
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                            }
                            await out.write(f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                            await out.write(b"data: [DONE]\n\n")
                            await out.write_eof()
                            return out
                        data = await upstream.json()
                text_out = _extract_responses_text(data)
            except TimeoutError:
                return failure(504, "llm_timeout", "LLM upstream timed out.", error_type="server_error")
            except Exception as exc:
                return failure(502, "llm_unavailable", str(exc), error_type="server_error")

            message: dict[str, Any] = {"role": "assistant", "content": text_out}
            if wants_audio:
                try:
                    wav_bytes = await synthesize_tts_wav(
                        http,
                        tts_url=config.tts_url,
                        tts_token=config.tts_token,
                        text=text_out,
                        ref_audio_b64=ref_audio_b64,
                        timeout=config.tts_timeout,
                    )
                    message["audio"] = {
                        "data": base64.b64encode(wav_bytes).decode("ascii"),
                        "format": "wav",
                        "transcript": text_out,
                    }
                except Exception as exc:
                    return failure(502, "tts_unavailable", str(exc), error_type="server_error")
            return web.json_response(
                {
                    "id": f"chatcmpl-{secrets.token_hex(8)}",
                    "object": "chat.completion",
                    "model": config.model_id,
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                }
            )

        if not config.llm_chat_url:
            return failure(503, "llm_unavailable", "OMNI_LLM_CHAT_URL is not configured.")

        fwd_payload = {
            k: v
            for k, v in payload.items()
            if k not in {"modalities", "audio", "ref_audio", "sampling_params_list"}
        }
        fwd_payload["model"] = config.llm_model
        fwd_payload["messages"] = resolved_messages
        if wants_audio:
            fwd_payload["stream"] = False

        try:
            async with asyncio.timeout(config.llm_timeout):
                async with http.post(config.llm_chat_url, json=fwd_payload, headers=llm_headers) as upstream:
                    if stream and not wants_audio and upstream.content_type == "text/event-stream":
                        out = web.StreamResponse(
                            status=upstream.status,
                            headers={
                                "Content-Type": upstream.headers.get("Content-Type", "text/event-stream"),
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                            },
                        )
                        await out.prepare(request)
                        async for chunk in upstream.content.iter_any():
                            await out.write(chunk)
                        await out.write_eof()
                        return out

                    raw_body = await upstream.read()
                    if upstream.status >= 400 or not wants_audio:
                        return web.Response(
                            status=upstream.status,
                            body=raw_body,
                            content_type=upstream.headers.get("Content-Type", "application/json"),
                        )
                    data = json.loads(raw_body)
        except TimeoutError:
            return failure(504, "llm_timeout", "LLM upstream timed out.", error_type="server_error")
        except Exception as exc:
            return failure(502, "llm_unavailable", str(exc), error_type="server_error")

        choices = data.get("choices") or []
        if choices:
            msg = choices[0].setdefault("message", {})
            text_out = str(msg.get("content") or "")
            if text_out:
                try:
                    wav_bytes = await synthesize_tts_wav(
                        http,
                        tts_url=config.tts_url,
                        tts_token=config.tts_token,
                        text=text_out,
                        ref_audio_b64=ref_audio_b64,
                        timeout=config.tts_timeout,
                    )
                    msg["audio"] = {
                        "data": base64.b64encode(wav_bytes).decode("ascii"),
                        "format": "wav",
                        "transcript": text_out,
                    }
                except Exception as exc:
                    return failure(502, "tts_unavailable", str(exc), error_type="server_error")
        return web.json_response(data)

    async def realtime(request: web.Request) -> web.StreamResponse:
        if request.query.get("duplex") != "1":
            return failure(400, "invalid_realtime_request", "duplex=1 query parameter is required.")
        autostart_raw = request.query.get("autostart", "1")
        if autostart_raw not in ("0", "1"):
            return failure(400, "invalid_realtime_request", "autostart must be '0' or '1'.")
        if realtime_semaphore.locked():
            return failure(
                429,
                "resource_exhausted",
                f"duplex_session_capacity_exhausted: limit={cap}",
                error_type="rate_limit_error",
            )

        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=MAX_WS_FRAME, compress=False)
        if not ws.can_prepare(request).ok:
            return failure(400, "upgrade_required", "A WebSocket upgrade is required.")

        model_override = request.query.get("model") or config.model_id
        async with realtime_semaphore:
            await ws.prepare(request)
            session = DuplexSession(
                config,
                state["http"],
                vad_factory(),
                model=model_override,
                autostart=(autostart_raw == "1"),
            )
            await session.run(ws, request)
        return ws

    app = web.Application(client_max_size=MAX_BODY, middlewares=[authenticate])
    app.cleanup_ctx.append(lifecycle)
    app.add_routes(
        [
            web.get("/livez", livez),
            web.get("/health", health),
            web.get("/version", version),
            web.get("/v1/models", models),
            web.post("/v1/chat/prompt_tokens", prompt_tokens),
            web.post("/v1/chat/completions", chat_completions),
            web.get("/v1/realtime", realtime),
        ]
    )
    return app
