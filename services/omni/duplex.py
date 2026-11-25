"""Engineered CPU full-duplex Realtime WebSocket orchestrator (/v1/realtime?duplex=1)."""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass, field
import io
import json
import logging
import secrets
from typing import Any, AsyncIterator
import wave

import aiohttp
from aiohttp import WSMsgType, web

from .chunker import SentenceChunker
from .vad import SAMPLE_RATE_HZ as INPUT_SAMPLE_RATE_HZ, VadDetector, VadEvent

OUTPUT_SAMPLE_RATE_HZ = 24000
MAX_UNCOMMITTED_BYTES = INPUT_SAMPLE_RATE_HZ * 2 * 60  # 60s safety cap
DEFAULT_INSTRUCTIONS = (
    "You are a concise, helpful voice assistant. Respond naturally for spoken conversation."
)


@dataclass(frozen=True)
class DuplexConfig:
    """Upstream worker URLs, credentials, and session limits for omni-engine."""

    model_id: str = "omni-duplex-qwen38"
    checkpoint: str = ""
    max_model_len: int = 131072
    asr_url: str = ""
    asr_token: str = ""
    asr_timeout: float = 60.0
    tts_url: str = ""
    tts_token: str = ""
    tts_timeout: float = 60.0
    llm_responses_url: str = ""
    llm_chat_url: str = ""
    llm_tokenize_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "llm-default"
    llm_timeout: float = 120.0
    default_instructions: str = DEFAULT_INSTRUCTIONS
    barge_in_min_ms: int = 160


def pcm16_to_wav_bytes(pcm16: bytes, sample_rate: int = INPUT_SAMPLE_RATE_HZ) -> bytes:
    """Wrap mono 16-bit signed LE PCM bytes into a standard RIFF WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16)
    return buf.getvalue()


def strip_audio_data_url(value: str | None) -> str | None:
    """Strip an optional data:audio/...;base64, prefix and validate base64 payload."""
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("data:"):
        header, sep, body = raw.partition(",")
        if not sep or ";base64" not in header.lower():
            raise ValueError("Invalid ref_audio data URL")
        raw = body.strip()
    base64.b64decode(raw, validate=True)
    return raw


def normalize_image_data_url(frame: str) -> str:
    """Prepend a data:image/*;base64, URL prefix to bare base64 video frames."""
    raw = frame.strip()
    if raw.startswith("data:image/"):
        return raw
    if raw.startswith("iVBORw0KGgo"):
        mime = "png"
    elif raw.startswith("UklGR"):
        mime = "webp"
    else:
        mime = "jpeg"
    return f"data:image/{mime};base64,{raw}"


async def transcribe_audio_wav(
    http: aiohttp.ClientSession,
    *,
    asr_url: str,
    asr_token: str,
    wav_bytes: bytes,
    timeout: float = 60.0,
    filename: str = "utterance.wav",
) -> str:
    """Send WAV audio to omni-asr /transcribe and return the stripped transcript text."""
    if not asr_url:
        raise RuntimeError("OMNI_ASR_WORKER_URL is not configured")
    form = aiohttp.FormData()
    form.add_field("file", wav_bytes, filename=filename, content_type="audio/wav")
    form.add_field("response_format", "json")
    headers = {"Authorization": f"Bearer {asr_token}"} if asr_token else {}
    async with asyncio.timeout(timeout):
        async with http.post(f"{asr_url.rstrip('/')}/transcribe", data=form, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"omni-asr returned {resp.status}: {body[:200]}")
            data = await resp.json()
    return str(data.get("text", "") or "").strip()


async def synthesize_tts_wav(
    http: aiohttp.ClientSession,
    *,
    tts_url: str,
    tts_token: str,
    text: str,
    ref_audio_b64: str | None = None,
    timeout: float = 60.0,
) -> bytes:
    """Call omni-tts /v1/tts/synthesize and return 24 kHz mono PCM16 WAV bytes."""
    if not tts_url:
        raise RuntimeError("OMNI_TTS_WORKER_URL is not configured")
    payload: dict[str, Any] = {"text": text}
    if ref_audio_b64:
        payload["reference_audio_base64"] = ref_audio_b64
    headers = {"Authorization": f"Bearer {tts_token}"} if tts_token else {}
    async with asyncio.timeout(timeout):
        async with http.post(f"{tts_url.rstrip('/')}/v1/tts/synthesize", json=payload, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"omni-tts synthesize returned {resp.status}: {body[:200]}")
            return await resp.read()


async def stream_tts_pcm(
    http: aiohttp.ClientSession,
    *,
    tts_url: str,
    tts_token: str,
    text: str,
    ref_audio_b64: str | None = None,
    timeout: float = 60.0,
) -> AsyncIterator[bytes]:
    """Stream 24 kHz mono 16-bit LE PCM chunks from omni-tts /v1/tts/stream."""
    if not tts_url:
        raise RuntimeError("OMNI_TTS_WORKER_URL is not configured")
    payload: dict[str, Any] = {"text": text}
    if ref_audio_b64:
        payload["reference_audio_base64"] = ref_audio_b64
    headers = {"Authorization": f"Bearer {tts_token}"} if tts_token else {}
    pcm_tail = bytearray()
    async with asyncio.timeout(timeout):
        async with http.post(f"{tts_url.rstrip('/')}/v1/tts/stream", json=payload, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"omni-tts stream returned {resp.status}: {body[:200]}")
            async for chunk in resp.content.iter_chunked(8192):
                if not chunk:
                    continue
                pcm_tail.extend(chunk)
                even_len = (len(pcm_tail) // 2) * 2
                if even_len > 0:
                    out = bytes(pcm_tail[:even_len])
                    del pcm_tail[:even_len]
                    yield out


@dataclass
class _ResponseRecord:
    response_id: str
    item_id: str
    segments: list[tuple[str, float, float]] = field(default_factory=list)
    generated_ms: float = 0.0
    full_text: str = ""
    interrupted: bool = False
    has_audio: bool = False


def _truncate_text_by_ms(record: _ResponseRecord, cut_ms: float) -> str:
    """Truncate generated assistant text proportionally against its audio timeline."""
    target_ms = max(0.0, float(cut_ms))
    if target_ms <= 0.0:
        return ""
    if record.segments:
        kept: list[str] = []
        for seg_text, start_ms, end_ms in record.segments:
            if target_ms <= start_ms:
                break
            if target_ms >= end_ms:
                kept.append(seg_text)
                continue
            span = end_ms - start_ms
            ratio = (target_ms - start_ms) / span if span > 0 else 0.0
            keep_chars = max(0, min(len(seg_text), int(round(len(seg_text) * ratio))))
            if keep_chars > 0:
                kept.append(seg_text[:keep_chars])
            break
        return "".join(kept)
    if record.generated_ms <= 0.0 or not record.full_text:
        return ""
    ratio = min(1.0, target_ms / record.generated_ms)
    return record.full_text[: int(round(len(record.full_text) * ratio))]


async def _iter_sse_events(resp: aiohttp.ClientResponse) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
    """Parse SSE frames from an aiohttp streaming response."""
    event_name: str | None = None
    data_lines: list[str] = []
    buf = bytearray()

    def flush_event() -> tuple[str | None, dict[str, Any]] | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = None
            return None
        joined = "\n".join(data_lines).strip()
        ev = event_name
        event_name = None
        data_lines = []
        if not joined or joined == "[DONE]":
            return None
        try:
            parsed = json.loads(joined)
        except ValueError:
            return None
        if isinstance(parsed, dict):
            return ev, parsed
        return None

    async for chunk in resp.content.iter_any():
        if not chunk:
            continue
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                break
            line_bytes = bytes(buf[:nl])
            del buf[: nl + 1]
            line = line_bytes.rstrip(b"\r").decode("utf-8", errors="replace")
            if line == "":
                item = flush_event()
                if item is not None:
                    yield item
            elif line.startswith(":"):
                continue
            elif line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
    if buf:
        tail = bytes(buf).rstrip(b"\r\n").decode("utf-8", errors="replace")
        if tail.startswith("data:"):
            data_lines.append(tail[5:].lstrip())
    item = flush_event()
    if item is not None:
        yield item


class DuplexSession:
    """Manages one full-duplex WebSocket session over CPU VAD + ASR + LLM + TTS."""

    def __init__(
        self,
        config: DuplexConfig,
        http: aiohttp.ClientSession,
        vad: VadDetector,
        *,
        model: str | None = None,
        autostart: bool = True,
    ) -> None:
        self.config = config
        self.http = http
        self.vad = vad
        self.model = model or config.model_id
        self.autostart = autostart
        self.started = False

        self.session_id = f"duplex-{secrets.token_hex(8)}"
        self.resume_token = f"rt_{secrets.token_hex(12)}"
        self.state = "open"
        self.turn_state = "listening"
        self.epoch = 0
        self.turn_id = 0
        self._seq = 0

        self.instructions = config.default_instructions
        self.voice: str | None = None
        self.ref_audio_b64: str | None = None
        self.modalities: list[str] = ["text", "audio"]
        self.output_modalities: list[str] = ["text", "audio"]
        self.overlap_policy = "barge_in_on_speech"
        self.playback_commit_policy = "ack_only"
        self.turn_detection: dict[str, Any] | None = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 256,
            "silence_duration_ms": 600,
        }
        self.temperature: float | None = None
        self.max_response_output_tokens: int | None = None
        self.extra_body: dict[str, Any] = {"auto_response": True}
        self.auto_response = True

        self.playback_generated_ms = 0
        self.playback_sent_ms = 0
        self.playback_played_ms = 0
        self.playback_committed_ms = 0

        self.active_response_id: str | None = None
        self.active_item_id: str | None = None
        self.active_response_turn_id: int | None = None

        self._ws: web.WebSocketResponse | None = None
        self._send_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._active_gen_task: asyncio.Task[None] | None = None
        self._cancel_reason = "client_cancelled"
        self._emit_cleared_on_cancel = True
        self._bg_tasks: set[asyncio.Task[None]] = set()
        self._closed = False

        self._uncommitted_pcm = bytearray()
        self._total_appended_ms = 0
        self._pending_speech_start_ms = 0
        self._pending_item_id: str | None = None
        self._latest_video_frames: list[str] = []

        # Ordered conversation items for stable Qwen3.8-27B prefix caching + truncation/CRUD.
        self._item_order: list[str] = []
        self._items: dict[str, dict[str, Any]] = {}
        self._response_records: dict[str, _ResponseRecord] = {}
        self._item_to_response: dict[str, str] = {}
        self._last_audio_response_id: str | None = None

    def _session_payload(self) -> dict[str, Any]:
        return {
            "object": "realtime.session",
            "type": "realtime",
            "id": self.session_id,
            "model": self.model,
            "state": self.state,
            "turn_state": self.turn_state,
            "epoch": self.epoch,
            "turn_id": self.turn_id,
            "active_request_id": self.active_response_id,
            "active_response_id": self.active_response_id,
            "active_response_turn_id": self.active_response_turn_id,
            "modalities": list(self.modalities),
            "output_modalities": list(self.output_modalities),
            "instructions": self.instructions,
            "voice": self.voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": INPUT_SAMPLE_RATE_HZ}, "sample_rate_hz": INPUT_SAMPLE_RATE_HZ},
                "output": {"format": {"type": "audio/pcm", "rate": OUTPUT_SAMPLE_RATE_HZ}},
            },
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_response_output_tokens,
            "overlap_policy": self.overlap_policy,
            "playback_commit_policy": self.playback_commit_policy,
            "turn_detection": self.turn_detection,
            "extra_body": dict(self.extra_body),
            "playback": self._playback_dict(),
            "capabilities": {
                "implementation_level": "engineered_duplex",
                "supports_input_append": True,
                "supports_barge_in": True,
                "supports_audio_truncate": True,
                "supports_session_resume": False,
                "supports_session_lease": True,
                "supports_multi_session": True,
                "chunk_period_ms": 1000,
                "input_modes": ["append_audio_chunk"],
            },
        }

    def _playback_dict(self) -> dict[str, int]:
        return {
            "generated_ms": int(self.playback_generated_ms),
            "sent_ms": int(self.playback_sent_ms),
            "played_ms": int(self.playback_played_ms),
            "committed_ms": int(self.playback_committed_ms),
        }

    async def _emit(self, event_type: str, **fields: Any) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            return
        async with self._send_lock:
            if ws.closed:
                return
            self._seq += 1
            payload = dict(fields)
            payload["type"] = event_type
            payload["server_event_seq"] = self._seq
            payload.setdefault("epoch", self.epoch)
            meta = dict(payload.get("metadata") or {})
            meta.setdefault("epoch", self.epoch)
            payload["metadata"] = meta
            await ws.send_str(json.dumps(payload, ensure_ascii=False))

    async def _emit_error(
        self,
        code: str,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        event_id: str | None = None,
    ) -> None:
        err: dict[str, Any] = {"type": error_type, "code": code, "message": message}
        if event_id is not None:
            err["event_id"] = event_id
        await self._emit("error", error=err)

    def _previous_item_id(self, item_id: str | None = None) -> str | None:
        if not self._item_order:
            return None
        if item_id is None:
            return self._item_order[-1]
        try:
            idx = self._item_order.index(item_id)
        except ValueError:
            return self._item_order[-1]
        return self._item_order[idx - 1] if idx > 0 else None

    def _add_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_id = item["id"]
        if item_id not in self._items:
            self._item_order.append(item_id)
        self._items[item_id] = item
        return item

    def _spawn(self, coro: Any) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def _emit_session_created(self) -> None:
        self.started = True
        await self._emit(
            "session.created",
            attachment_generation=1,
            resume_token=self.resume_token,
            session=self._session_payload(),
        )

    async def run(self, ws: web.WebSocketResponse, _request: web.Request | None = None) -> None:
        """Drive the bidirectional WebSocket loop until disconnect or session.close."""
        self._ws = ws
        try:
            if self.autostart:
                await self._emit_session_created()
            async for frame in ws:
                if frame.type == WSMsgType.TEXT:
                    try:
                        event = json.loads(frame.data)
                    except ValueError:
                        await self._emit_error("invalid_json", "Expected a JSON event object.")
                        continue
                    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                        await self._emit_error("bad_event", "Event must be a JSON object with a string 'type'.")
                        continue
                    should_close = await self._handle_event(event)
                    if should_close:
                        break
                elif frame.type == WSMsgType.ERROR:
                    break
        finally:
            self._closed = True
            self.state = "closed"
            await self._cancel_all_tasks()
            if not ws.closed:
                await ws.close()

    async def _cancel_all_tasks(self) -> None:
        tasks = list(self._bg_tasks)
        if self._active_gen_task and not self._active_gen_task.done():
            tasks.append(self._active_gen_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _interrupt_active(self, *, reason: str, emit_cleared: bool = True) -> None:
        """Advance epoch, abort any active LLM/TTS response, and emit clear events."""
        self.epoch += 1
        self._cancel_reason = reason
        self._emit_cleared_on_cancel = emit_cleared
        resp_id = self.active_response_id
        task = self._active_gen_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        elif emit_cleared:
            await self._emit(
                "output_audio_buffer.cleared",
                response_id=resp_id or self._last_audio_response_id,
                epoch=self.epoch,
            )

    async def _handle_event(self, event: dict[str, Any]) -> bool:
        kind = event["type"]
        event_id = event.get("event_id")

        if kind == "session.heartbeat":
            await self._emit("session.heartbeat_ack", session_id=self.session_id)
            return False

        if kind == "session.event_ack":
            return False

        if kind == "session.resume":
            await self._emit_error(
                "session_resume_expired",
                "Session resume is not supported; open a new duplex session.",
                event_id=event_id,
            )
            return False

        if kind == "session.close":
            self.state = "closed"
            await self._interrupt_active(reason="session_closed", emit_cleared=False)
            await self._emit(
                "session.closed",
                session_id=self.session_id,
                epoch=self.epoch,
                reason="client_close",
                event={"type": "session.closed", "reason": "client_close"},
            )
            return True

        if kind == "session.update":
            await self._handle_session_update(event)
            return False

        if not self.started:
            await self._emit_session_created()

        if kind == "input_audio_buffer.append":
            await self._handle_audio_append(event)
        elif kind in ("input_audio_buffer.commit", "input.commit"):
            await self._handle_audio_commit(event)
        elif kind in ("input_audio_buffer.clear", "input.cancel"):
            await self._handle_audio_clear(cancel_epoch=(kind == "input.cancel"))
        elif kind == "input.text.append":
            text = str(event.get("text", "") or "").strip()
            if text:
                await self._commit_user_text(text)
                self._spawn(self._launch_response())
        elif kind == "conversation.item.create":
            await self._handle_item_create(event)
        elif kind == "conversation.item.retrieve":
            await self._handle_item_retrieve(event)
        elif kind == "conversation.item.delete":
            await self._handle_item_delete(event)
        elif kind == "conversation.item.truncate":
            await self._handle_item_truncate(event)
        elif kind == "response.create":
            await self._handle_response_create(event)
        elif kind == "response.cancel":
            if self._active_gen_task is not None and not self._active_gen_task.done():
                await self._interrupt_active(reason="client_cancelled", emit_cleared=False)
            else:
                self.epoch += 1
        elif kind == "output_audio_buffer.clear":
            await self._interrupt_active(reason="output_buffer_cleared", emit_cleared=True)
        elif kind == "barge_in":
            await self._emit(
                "overlap.decision",
                session_id=self.session_id,
                epoch=self.epoch,
                policy=self.overlap_policy,
                action="barge_in",
                reason="client_barge_in",
            )
            await self._interrupt_active(reason="barge_in", emit_cleared=True)
        elif kind == "turn.signal":
            sig = event.get("event")
            if sig == "barge_in":
                await self._emit(
                    "turn.event",
                    session_id=self.session_id,
                    epoch=self.epoch,
                    event="barge_in",
                )
                await self._interrupt_active(reason="barge_in", emit_cleared=True)
            else:
                await self._emit("turn.event", session_id=self.session_id, epoch=self.epoch, event=sig)
        elif kind in ("playback.ack", "audio.playback_ack"):
            await self._handle_playback_ack(event)
        else:
            await self._emit_error("unknown_event", f"Unsupported event type: {kind}", event_id=event_id)
        return False

    async def _handle_session_update(self, event: dict[str, Any]) -> None:
        event_id = event.get("event_id")
        session_cfg = event.get("session")
        if not isinstance(session_cfg, dict):
            await self._emit_error("bad_event", "session.update requires a 'session' object.", event_id=event_id)
            return

        if isinstance(session_cfg.get("model"), str) and session_cfg["model"].strip():
            self.model = session_cfg["model"].strip()
        if "instructions" in session_cfg and isinstance(session_cfg["instructions"], str):
            self.instructions = session_cfg["instructions"]
        if "voice" in session_cfg:
            self.voice = session_cfg["voice"]
        if "ref_audio" in session_cfg:
            try:
                self.ref_audio_b64 = strip_audio_data_url(session_cfg["ref_audio"])
            except (ValueError, TypeError, binascii.Error) as exc:
                await self._emit_error("invalid_ref_audio", str(exc), event_id=event_id)
                return

        if "output_modalities" in session_cfg and isinstance(session_cfg["output_modalities"], list):
            mods = [str(m) for m in session_cfg["output_modalities"]]
            self.output_modalities = mods
            if "modalities" not in session_cfg:
                self.modalities = mods
        if "modalities" in session_cfg and isinstance(session_cfg["modalities"], list):
            mods = [str(m) for m in session_cfg["modalities"]]
            self.modalities = mods
            if "output_modalities" not in session_cfg:
                self.output_modalities = mods

        if "overlap_policy" in session_cfg:
            policy = session_cfg["overlap_policy"]
            if policy not in ("barge_in_on_speech", "listen_only"):
                await self._emit_error("bad_event", f"Unsupported overlap_policy: {policy}", event_id=event_id)
                return
            self.overlap_policy = policy

        if "playback_commit_policy" in session_cfg:
            pcp = session_cfg["playback_commit_policy"]
            if pcp not in ("ack_only", "commit_all_on_done"):
                await self._emit_error("bad_event", f"Unsupported playback_commit_policy: {pcp}", event_id=event_id)
                return
            self.playback_commit_policy = pcp

        if "temperature" in session_cfg and session_cfg["temperature"] is not None:
            self.temperature = float(session_cfg["temperature"])
        if "max_response_output_tokens" in session_cfg:
            val = session_cfg["max_response_output_tokens"]
            self.max_response_output_tokens = None if val in (None, "inf") else int(val)
        elif "max_tokens" in session_cfg:
            val = session_cfg["max_tokens"]
            self.max_response_output_tokens = None if val in (None, "inf") else int(val)

        if "turn_detection" in session_cfg:
            td = session_cfg["turn_detection"]
            if td is None:
                self.turn_detection = None
            elif isinstance(td, dict):
                self.turn_detection = dict(td)
                configure = getattr(self.vad, "configure", None)
                if callable(configure):
                    try:
                        configure(
                            threshold=td.get("threshold"),
                            prefix_padding_ms=td.get("prefix_padding_ms"),
                            silence_duration_ms=td.get("silence_duration_ms"),
                            barge_in_min_ms=td.get("barge_in_min_ms"),
                            min_speech_frames=td.get("min_speech_frames"),
                        )
                    except ValueError as exc:
                        await self._emit_error("invalid_turn_detection", str(exc), event_id=event_id)
                        return
            else:
                await self._emit_error("bad_event", "turn_detection must be an object or null.", event_id=event_id)
                return

        initial_user_text = ""
        if isinstance(session_cfg.get("extra_body"), dict):
            self.extra_body = dict(self.extra_body) | dict(session_cfg["extra_body"])
            if "auto_response" in self.extra_body:
                self.auto_response = bool(self.extra_body["auto_response"])
            raw_initial = self.extra_body.get("duplex_initial_user_text")
            if isinstance(raw_initial, str) and raw_initial.strip():
                initial_user_text = raw_initial.strip()

        if not self.started:
            await self._emit_session_created()
        await self._emit("session.updated", session=self._session_payload())

        if initial_user_text:
            await self._commit_user_text(initial_user_text)
            self._spawn(self._launch_response())

    async def _handle_audio_append(self, event: dict[str, Any]) -> None:
        event_id = event.get("event_id")
        encoded = event.get("audio")
        if not isinstance(encoded, str):
            await self._emit_error("invalid_audio", "input_audio_buffer.append requires base64 'audio'.", event_id=event_id)
            return
        try:
            pcm16 = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            await self._emit_error("invalid_audio", "Invalid base64 audio payload.", event_id=event_id)
            return
        if len(pcm16) % 2 != 0:
            await self._emit_error("invalid_audio", "PCM16 payload must have even byte length.", event_id=event_id)
            return

        video_frames = event.get("video_frames")
        if isinstance(video_frames, list):
            cleaned = [f.strip() for f in video_frames if isinstance(f, str) and f.strip()]
            if cleaned:
                self._latest_video_frames = cleaned[-1:]

        if pcm16:
            self._uncommitted_pcm.extend(pcm16)
            if len(self._uncommitted_pcm) > MAX_UNCOMMITTED_BYTES:
                excess = len(self._uncommitted_pcm) - MAX_UNCOMMITTED_BYTES
                del self._uncommitted_pcm[:excess]

        vad_events: list[VadEvent] = self.vad.process_pcm16(pcm16)
        chunk_ms = int((len(pcm16) / 2) * 1000 / INPUT_SAMPLE_RATE_HZ)
        self._total_appended_ms += chunk_ms

        for vad_ev in vad_events:
            if vad_ev.kind == "speech_started":
                self._pending_item_id = f"item_{secrets.token_hex(8)}"
                self._pending_speech_start_ms = max(0, self._total_appended_ms - chunk_ms)
                await self._emit(
                    "input_audio_buffer.speech_started",
                    audio_start_ms=self._pending_speech_start_ms,
                    item_id=self._pending_item_id,
                )
                await self._maybe_trigger_barge_in()
            elif vad_ev.kind == "speech_active":
                await self._maybe_trigger_barge_in()
            elif vad_ev.kind == "speech_stopped":
                item_id = self._pending_item_id or f"item_{secrets.token_hex(8)}"
                self._pending_item_id = None
                self._uncommitted_pcm.clear()
                await self._emit(
                    "input_audio_buffer.speech_stopped",
                    audio_end_ms=self._total_appended_ms,
                    item_id=item_id,
                )
                if self.turn_detection is None and not self.auto_response:
                    continue
                frames = list(self._latest_video_frames)
                self._latest_video_frames.clear()
                self._spawn(
                    self._process_utterance_turn(
                        pcm16=vad_ev.pcm16,
                        item_id=item_id,
                        video_frames=frames,
                        trigger_response=self.auto_response,
                    )
                )

    async def _maybe_trigger_barge_in(self) -> None:
        if self.overlap_policy == "listen_only":
            return
        if self._active_gen_task is None or self._active_gen_task.done():
            return
        await self._emit(
            "overlap.decision",
            session_id=self.session_id,
            epoch=self.epoch,
            policy=self.overlap_policy,
            action="barge_in",
            reason="server_vad_utterance_active",
        )
        await self._interrupt_active(reason="barge_in", emit_cleared=True)

    async def _handle_audio_commit(self, event: dict[str, Any]) -> None:
        if event.get("is_speech") is False:
            self._uncommitted_pcm.clear()
            self.vad.reset()
            await self._emit(
                "response.listen",
                session_id=self.session_id,
                epoch=self.epoch,
                response={
                    "object": "realtime.response",
                    "status": "listening",
                    "metadata": {
                        "type": "response.listen",
                        "session_id": self.session_id,
                        "epoch": self.epoch,
                        "reason": "model_listen",
                    },
                },
            )
            return

        flush_fn = getattr(self.vad, "flush_utterance", None)
        flushed = flush_fn() if callable(flush_fn) else None
        if flushed and flushed.pcm16:
            pcm = flushed.pcm16
            self._uncommitted_pcm.clear()
        else:
            pcm = bytes(self._uncommitted_pcm)
            self._uncommitted_pcm.clear()
        self.vad.reset()

        if not pcm:
            await self._emit_error(
                "input_audio_buffer_empty",
                "input_audio_buffer.commit requires a non-empty input audio buffer.",
                event_id=event.get("event_id"),
            )
            return

        item_id = self._pending_item_id or f"item_{secrets.token_hex(8)}"
        self._pending_item_id = None
        frames = list(self._latest_video_frames)
        self._latest_video_frames.clear()
        trigger_response = event.get("response_create") is not False
        self._spawn(
            self._process_utterance_turn(
                pcm16=pcm,
                item_id=item_id,
                video_frames=frames,
                trigger_response=trigger_response,
            )
        )

    async def _handle_audio_clear(self, *, cancel_epoch: bool) -> None:
        self._uncommitted_pcm.clear()
        self._pending_item_id = None
        self.vad.reset()
        if cancel_epoch:
            self.epoch += 1
        await self._emit("input_audio_buffer.cleared")

    async def _handle_item_create(self, event: dict[str, Any]) -> None:
        raw_item = event.get("item")
        if not isinstance(raw_item, dict):
            await self._emit_error("bad_event", "conversation.item.create requires an 'item' object.", event_id=event.get("event_id"))
            return
        item_id = raw_item.get("id") or f"item_{secrets.token_hex(8)}"
        item = dict(raw_item)
        item["id"] = item_id
        item.setdefault("object", "realtime.item")
        item.setdefault("status", "completed")
        prev_id = self._previous_item_id()
        self._add_item(item)
        await self._emit("conversation.item.added", previous_item_id=prev_id, item=item)
        await self._emit("conversation.item.created", previous_item_id=prev_id, item=item)
        await self._emit("conversation.item.done", previous_item_id=prev_id, item=item)

    async def _handle_item_retrieve(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        item = self._items.get(item_id) if isinstance(item_id, str) else None
        if item is None:
            await self._emit_error("item_not_found", f"Item {item_id!r} was not found.", event_id=event.get("event_id"))
            return
        await self._emit("conversation.item.retrieved", item=item)

    async def _handle_item_delete(self, event: dict[str, Any]) -> None:
        item_id = event.get("item_id")
        if not isinstance(item_id, str) or item_id not in self._items:
            await self._emit_error("item_not_found", f"Item {item_id!r} was not found.", event_id=event.get("event_id"))
            return
        del self._items[item_id]
        self._item_order = [iid for iid in self._item_order if iid != item_id]
        await self._emit(
            "conversation.item.deleted",
            item_id=item_id,
            event={"type": "conversation.item.deleted", "item_id": item_id},
        )

    async def _handle_item_truncate(self, event: dict[str, Any]) -> None:
        event_id = event.get("event_id")
        item_id = event.get("item_id")
        audio_end_ms = event.get("audio_end_ms")
        content_index = int(event.get("content_index", 0) or 0)
        if not isinstance(item_id, str) or item_id not in self._items or not isinstance(audio_end_ms, (int, float)):
            await self._emit_error("bad_event", "Invalid conversation.item.truncate payload.", event_id=event_id)
            return
        resp_id = self._item_to_response.get(item_id)
        record = self._response_records.get(resp_id) if resp_id else None
        if record is None or not record.has_audio:
            await self._emit_error("bad_event", "Target item has no generated audio to truncate.", event_id=event_id)
            return
        truncated = _truncate_text_by_ms(record, float(audio_end_ms))
        self._apply_item_transcript(item_id, truncated)
        await self._emit(
            "conversation.item.truncated",
            item_id=item_id,
            content_index=content_index,
            audio_end_ms=int(audio_end_ms),
            event={
                "type": "conversation.item.truncated",
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": int(audio_end_ms),
            },
        )

    def _apply_item_transcript(self, item_id: str, text: str) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        content = item.get("content")
        if isinstance(content, list) and content:
            part = dict(content[0])
            if part.get("type") in ("output_audio", "audio"):
                part["transcript"] = text
            else:
                part["text"] = text
            item["content"] = [part, *content[1:]]
        else:
            item["content"] = [{"type": "output_audio", "transcript": text}]

    async def _handle_playback_ack(self, event: dict[str, Any]) -> None:
        event_id = event.get("event_id")
        played_ms = max(0, int(event.get("played_ms", 0) or 0))
        committed_ms = max(0, int(event.get("committed_ms", played_ms) or 0))
        truncate_req = bool(event.get("truncate", False))

        req_item_id = event.get("item_id")
        req_resp_id = event.get("response_id")
        target_resp_id: str | None = None
        if isinstance(req_resp_id, str) and req_resp_id in self._response_records:
            target_resp_id = req_resp_id
        elif isinstance(req_item_id, str) and req_item_id in self._item_to_response:
            target_resp_id = self._item_to_response[req_item_id]
        elif self._last_audio_response_id and self._last_audio_response_id in self._response_records:
            target_resp_id = self._last_audio_response_id

        if isinstance(req_item_id, str) and req_item_id not in self._items and target_resp_id is None:
            await self._emit_error("playback_item_not_found", f"Unknown playback item {req_item_id!r}.", event_id=event_id)
            return
        if truncate_req and target_resp_id is None:
            await self._emit_error("bad_event", "Cannot truncate playback without an identifiable response.", event_id=event_id)
            return

        self.playback_played_ms = max(self.playback_played_ms, played_ms)
        self.playback_committed_ms = max(self.playback_committed_ms, committed_ms)

        history_committed = False
        target_item_id = req_item_id
        if target_resp_id is not None:
            record = self._response_records[target_resp_id]
            target_item_id = record.item_id
            should_truncate = truncate_req or (
                self.playback_commit_policy == "ack_only" and record.interrupted
            )
            if should_truncate and target_item_id in self._items:
                truncated = _truncate_text_by_ms(record, played_ms)
                self._apply_item_transcript(target_item_id, truncated)
                history_committed = True
            elif not record.interrupted and record.generated_ms > 0 and committed_ms >= record.generated_ms:
                history_committed = True

        ack_payload = {
            "type": "playback.acknowledged",
            "session_id": self.session_id,
            "epoch": self.epoch,
            "item_id": target_item_id,
            "response_id": target_resp_id,
            "played_ms": played_ms,
            "committed_ms": committed_ms,
            "truncate": truncate_req,
            "playback": self._playback_dict(),
            "history_committed": history_committed,
        }
        await self._emit("playback.acknowledged", event=ack_payload)

    async def _handle_response_create(self, event: dict[str, Any]) -> None:
        resp_cfg = event.get("response") if isinstance(event.get("response"), dict) else {}
        instructions = resp_cfg.get("instructions") if isinstance(resp_cfg.get("instructions"), str) else None
        modalities = (
            [str(m) for m in resp_cfg["modalities"]]
            if isinstance(resp_cfg.get("modalities"), list)
            else None
        )
        temperature = float(resp_cfg["temperature"]) if resp_cfg.get("temperature") is not None else None
        raw_max = resp_cfg.get("max_output_tokens", resp_cfg.get("max_tokens"))
        max_tokens = None if raw_max in (None, "inf") else int(raw_max)
        frames = list(self._latest_video_frames)
        self._latest_video_frames.clear()
        self._spawn(
            self._launch_response(
                instructions_override=instructions,
                modalities_override=modalities,
                temperature_override=temperature,
                max_tokens_override=max_tokens,
                video_frames=frames,
            )
        )

    async def _commit_user_text(self, text: str, *, item_id: str | None = None, is_audio: bool = False) -> str:
        uid = item_id or f"item_{secrets.token_hex(8)}"
        prev_id = self._previous_item_id()
        content_part = (
            {"type": "input_audio", "transcript": text}
            if is_audio
            else {"type": "input_text", "text": text}
        )
        item = {
            "id": uid,
            "object": "realtime.item",
            "type": "message",
            "role": "user",
            "status": "completed",
            "content": [content_part],
        }
        self._add_item(item)
        self.turn_id += 1
        await self._emit("conversation.item.added", previous_item_id=prev_id, item=item)
        await self._emit("conversation.item.created", previous_item_id=prev_id, item=item)
        if is_audio:
            await self._emit(
                "input_audio_buffer.committed",
                previous_item_id=prev_id,
                item_id=uid,
                event={
                    "type": "input.committed",
                    "session_id": self.session_id,
                    "turn_id": self.turn_id,
                    "epoch": self.epoch,
                    "history_len": len(self._item_order),
                    "realtime_item_id": uid,
                },
            )
            await self._emit(
                "conversation.item.input_audio_transcription.completed",
                item_id=uid,
                content_index=0,
                transcript=text,
            )
        await self._emit("conversation.item.done", previous_item_id=prev_id, item=item)
        return uid

    async def _process_utterance_turn(
        self,
        *,
        pcm16: bytes,
        item_id: str,
        video_frames: list[str],
        trigger_response: bool,
    ) -> None:
        if not pcm16 or self._closed:
            return
        try:
            wav_bytes = pcm16_to_wav_bytes(pcm16, INPUT_SAMPLE_RATE_HZ)
            transcript = await transcribe_audio_wav(
                self.http,
                asr_url=self.config.asr_url,
                asr_token=self.config.asr_token,
                wav_bytes=wav_bytes,
                timeout=self.config.asr_timeout,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logging.exception("Duplex ASR transcription failed")
            await self._emit_error("asr_failed", str(exc), error_type="server_error")
            return

        if not transcript:
            # Skip empty/noise turns cleanly without polluting history or calling LLM.
            return

        await self._commit_user_text(transcript, item_id=item_id, is_audio=True)
        if trigger_response and not self._closed:
            await self._launch_response(video_frames=video_frames)

    def _build_responses_payload(
        self,
        *,
        instructions: str,
        temperature: float | None,
        max_tokens: int | None,
        video_frames: list[str],
    ) -> dict[str, Any]:
        """Build /v1/responses input preserving a stable committed prefix for Prefix Caching."""
        input_items: list[dict[str, Any]] = []
        last_user_idx = -1
        for iid in self._item_order:
            item = self._items[iid]
            if item.get("type") == "function_call_output":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.get("call_id", ""),
                        "output": item.get("output", ""),
                    }
                )
                continue
            role = item.get("role", "user")
            parts: list[dict[str, Any]] = []
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("input_text", "text", "output_text"):
                    text_val = str(part.get("text", "") or "")
                    if text_val:
                        parts.append(
                            {
                                "type": "output_text" if role == "assistant" else "input_text",
                                "text": text_val,
                            }
                        )
                elif ptype in ("input_audio", "output_audio", "audio"):
                    tx = str(part.get("transcript", "") or "")
                    if tx:
                        parts.append(
                            {
                                "type": "output_text" if role == "assistant" else "input_text",
                                "text": tx,
                            }
                        )
                elif ptype == "input_image" and part.get("image_url"):
                    parts.append({"type": "input_image", "image_url": part["image_url"]})
            if not parts:
                continue
            if role == "user":
                last_user_idx = len(input_items)
            input_items.append({"role": role, "content": parts})

        if video_frames:
            image_parts = [{"type": "input_image", "image_url": normalize_image_data_url(f)} for f in video_frames]
            if last_user_idx >= 0:
                input_items[last_user_idx]["content"] = list(input_items[last_user_idx]["content"]) + image_parts
            else:
                input_items.append({"role": "user", "content": image_parts})

        payload: dict[str, Any] = {
            "model": self.config.llm_model,
            "instructions": instructions,
            "input": input_items,
            "store": False,
            "stream": True,
            "reasoning": {"effort": "none"},
            "chat_template_kwargs": {"enable_thinking": False},
            "max_output_tokens": max_tokens or self.max_response_output_tokens or 1024,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        return payload

    async def _launch_response(
        self,
        *,
        instructions_override: str | None = None,
        modalities_override: list[str] | None = None,
        temperature_override: float | None = None,
        max_tokens_override: int | None = None,
        video_frames: list[str] | None = None,
    ) -> None:
        async with self._turn_lock:
            if self._closed:
                return
            response_epoch = self.epoch
            task = asyncio.create_task(
                self._generate_response(
                    response_epoch=response_epoch,
                    instructions=instructions_override or self.instructions,
                    output_modalities=modalities_override or list(self.output_modalities),
                    temperature=temperature_override if temperature_override is not None else self.temperature,
                    max_tokens=max_tokens_override,
                    video_frames=list(video_frames or ()),
                )
            )
            self._active_gen_task = task
            try:
                await task
            finally:
                if self._active_gen_task is task:
                    self._active_gen_task = None

    async def _generate_response(
        self,
        *,
        response_epoch: int,
        instructions: str,
        output_modalities: list[str],
        temperature: float | None,
        max_tokens: int | None,
        video_frames: list[str],
    ) -> None:
        response_id = f"resp_{secrets.token_hex(8)}"
        item_id = f"item_{response_id}"
        wants_audio = "audio" in output_modalities
        turn_id = self.turn_id
        prev_item_id = self._previous_item_id()
        record = _ResponseRecord(response_id=response_id, item_id=item_id, has_audio=wants_audio)
        self._response_records[response_id] = record
        self._item_to_response[item_id] = response_id
        if wants_audio:
            self._last_audio_response_id = response_id

        self.active_response_id = response_id
        self.active_item_id = item_id
        self.active_response_turn_id = turn_id
        self.turn_state = "thinking"

        assistant_item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        self._add_item(assistant_item)

        await self._emit(
            "response.created",
            response_id=response_id,
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "status_details": None,
                "output": [],
                "modalities": list(output_modalities),
                "metadata": {
                    "duplex_event": {
                        "type": "response.created",
                        "session_id": self.session_id,
                        "response_id": response_id,
                        "epoch": response_epoch,
                        "turn_id": turn_id,
                    }
                },
            },
        )
        await self._emit("conversation.item.added", previous_item_id=prev_item_id, item=assistant_item)
        await self._emit("conversation.item.created", previous_item_id=prev_item_id, item=assistant_item)
        await self._emit(
            "response.output_item.added",
            response_id=response_id,
            output_index=0,
            item=assistant_item,
        )
        initial_part = {"type": "audio", "transcript": ""} if wants_audio else {"type": "text", "text": ""}
        await self._emit(
            "response.content_part.added",
            response_id=response_id,
            item_id=item_id,
            output_index=0,
            content_index=0,
            part=initial_part,
        )

        speak_emitted = False
        generated_text_parts: list[str] = []
        resp_generated_ms = 0.0
        chunker = SentenceChunker()

        async def emit_clause(clause: str, *, is_final: bool = False) -> None:
            nonlocal speak_emitted, resp_generated_ms
            if response_epoch != self.epoch:
                raise asyncio.CancelledError
            if not wants_audio:
                generated_text_parts.append(clause)
                await self._emit(
                    "response.output_text.delta",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    delta=clause,
                )
                return

            if not speak_emitted:
                speak_emitted = True
                self.turn_state = "speaking"
                await self._emit(
                    "response.speak",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    metadata={"session_id": self.session_id, "epoch": response_epoch, "model_speak": True},
                )

            clause_start_ms = resp_generated_ms
            clause_emitted_text = False
            try:
                async for pcm_chunk in stream_tts_pcm(
                    self.http,
                    tts_url=self.config.tts_url,
                    tts_token=self.config.tts_token,
                    text=clause,
                    ref_audio_b64=self.ref_audio_b64,
                    timeout=self.config.tts_timeout,
                ):
                    if response_epoch != self.epoch:
                        raise asyncio.CancelledError
                    dur_ms = len(pcm_chunk) * 1000.0 / (OUTPUT_SAMPLE_RATE_HZ * 2)
                    resp_generated_ms += dur_ms
                    self.playback_generated_ms += int(round(dur_ms))
                    self.playback_sent_ms += int(round(dur_ms))
                    transcript_delta = "" if clause_emitted_text else clause
                    if not clause_emitted_text:
                        clause_emitted_text = True
                        generated_text_parts.append(clause)

                    await self._emit(
                        "response.output_audio.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=base64.b64encode(pcm_chunk).decode("ascii"),
                        format="pcm16",
                        sample_rate_hz=OUTPUT_SAMPLE_RATE_HZ,
                        metadata={
                            "session_id": self.session_id,
                            "epoch": response_epoch,
                            "model_speak": True,
                            "end_of_turn": is_final,
                            "audio_duration_ms": int(round(dur_ms)),
                            "playback": self._playback_dict(),
                        },
                    )
                    await self._emit(
                        "response.output_audio_transcript.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=transcript_delta,
                    )

                if not clause_emitted_text:
                    generated_text_parts.append(clause)
                    await self._emit(
                        "response.output_audio_transcript.delta",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        delta=clause,
                    )
            finally:
                if clause_emitted_text or resp_generated_ms > clause_start_ms:
                    record.segments.append((clause, clause_start_ms, resp_generated_ms))
                    record.generated_ms = resp_generated_ms
                    record.full_text = "".join(generated_text_parts)

        payload = self._build_responses_payload(
            instructions=instructions,
            temperature=temperature,
            max_tokens=max_tokens,
            video_frames=video_frames,
        )
        headers = {"Authorization": f"Bearer {self.config.llm_api_key}"} if self.config.llm_api_key else {}

        status = "completed"
        status_details: dict[str, Any] | None = None
        try:
            async with asyncio.timeout(self.config.llm_timeout):
                async with self.http.post(
                    self.config.llm_responses_url, json=payload, headers=headers
                ) as llm_resp:
                    if llm_resp.status >= 400:
                        body = await llm_resp.text()
                        raise RuntimeError(f"llm-engine returned {llm_resp.status}: {body[:200]}")
                    seen_delta = False
                    async for ev_name, ev_obj in _iter_sse_events(llm_resp):
                        if response_epoch != self.epoch:
                            raise asyncio.CancelledError
                        ev_type = ev_obj.get("type") or ev_name
                        if ev_type == "response.output_text.delta":
                            delta = str(ev_obj.get("delta", "") or "")
                            if delta:
                                seen_delta = True
                                if not wants_audio:
                                    await emit_clause(delta)
                                else:
                                    for clause in chunker.feed(delta):
                                        await emit_clause(clause)
                        elif ev_type == "response.completed" and not seen_delta:
                            fallback = _extract_responses_text(ev_obj.get("response") or ev_obj)
                            if fallback:
                                if not wants_audio:
                                    await emit_clause(fallback, is_final=True)
                                else:
                                    for clause in chunker.feed(fallback):
                                        await emit_clause(clause)
                        elif ev_type in ("response.failed", "error"):
                            msg = str((ev_obj.get("error") or {}).get("message") or "LLM response failed")
                            raise RuntimeError(msg)

                    if wants_audio:
                        tail_clauses = chunker.flush()
                        for idx, clause in enumerate(tail_clauses):
                            await emit_clause(clause, is_final=(idx == len(tail_clauses) - 1))
        except asyncio.CancelledError:
            status = "cancelled"
            record.interrupted = True
            status_details = {"type": "cancelled", "reason": self._cancel_reason}
            if self._emit_cleared_on_cancel:
                await self._emit(
                    "output_audio_buffer.cleared",
                    response_id=response_id,
                    epoch=self.epoch,
                )
        except Exception as exc:
            logging.exception("Duplex response turn failed")
            status = "failed"
            status_details = {"type": "failed", "reason": "upstream_error", "message": str(exc)}
            await self._emit_error("upstream_error", str(exc), error_type="server_error")
        finally:
            full_text = "".join(generated_text_parts)
            record.full_text = full_text
            record.generated_ms = resp_generated_ms
            if self.playback_commit_policy == "commit_all_on_done" and status == "completed":
                self.playback_played_ms = max(self.playback_played_ms, self.playback_generated_ms)
                self.playback_committed_ms = max(self.playback_committed_ms, self.playback_generated_ms)

            done_part = (
                {"type": "audio", "transcript": full_text}
                if wants_audio
                else {"type": "text", "text": full_text}
            )
            item_content = (
                [{"type": "output_audio", "transcript": full_text}]
                if wants_audio
                else [{"type": "output_text", "text": full_text}]
            )
            assistant_item["status"] = status
            assistant_item["content"] = item_content

            if wants_audio:
                await self._emit(
                    "response.output_audio.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                )
                await self._emit(
                    "response.output_audio_transcript.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    transcript=full_text,
                )
            else:
                await self._emit(
                    "response.output_text.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    text=full_text,
                )

            await self._emit(
                "response.content_part.done",
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                part=done_part,
            )
            await self._emit(
                "response.output_item.done",
                response_id=response_id,
                output_index=0,
                item=assistant_item,
            )
            await self._emit(
                "conversation.item.done",
                previous_item_id=prev_item_id,
                item=assistant_item,
            )
            await self._emit(
                "response.done",
                response_id=response_id,
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": status,
                    "status_details": status_details,
                    "output": [assistant_item],
                    "metadata": {
                        "type": "response.done",
                        "session_id": self.session_id,
                        "response_id": response_id,
                        "epoch": self.epoch,
                        "committed": status == "completed" and self.playback_commit_policy == "commit_all_on_done",
                        "playback": self._playback_dict(),
                    },
                },
            )
            await self._emit("rate_limits.updated", rate_limits=[])
            self.active_response_id = None
            self.active_item_id = None
            self.active_response_turn_id = None
            self.turn_state = "listening"


def _extract_responses_text(response_obj: dict[str, Any]) -> str:
    """Extract concatenated output_text from a Responses API response object."""
    texts: list[str] = []
    for item in response_obj.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                val = part.get("text")
                if isinstance(val, str):
                    texts.append(val)
    return "".join(texts)
