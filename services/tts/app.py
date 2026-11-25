"""Authenticated HTTP API for zero-shot voice-cloning and streaming TTS."""

import asyncio
import base64
import binascii
from contextlib import aclosing
import json
import logging
import os
from pathlib import Path
import secrets
import wave

from aiohttp import web

from .backends import TTSBackend, VoiceHandle, decode_wav_to_float32
from .cache import VoicePromptCache
from .model import TTS_CHECKPOINT, TTS_MODEL_ID, TTS_MODEL_ROOT

logger = logging.getLogger(__name__)

MAX_REQUEST_BYTES = 16 * 1024 * 1024


def failure(status: int, code: str, message: str) -> web.Response:
    """Return a structured JSON error payload matching gateway/worker conventions."""
    return web.json_response(
        {"error": {"code": code, "message": message}},
        status=status,
    )


def _strip_audio_data_url(raw_b64: str) -> str:
    if raw_b64.startswith("data:"):
        header, sep, body = raw_b64.partition(",")
        if not sep or ";base64" not in header.lower():
            raise ValueError("Unsupported data URL encoding")
        return body
    return raw_b64


def _decode_reference_wav(raw_b64: str) -> bytes:
    cleaned = _strip_audio_data_url(raw_b64.strip())
    wav_bytes = base64.b64decode(cleaned, validate=True)
    decode_wav_to_float32(wav_bytes)
    return wav_bytes


def create_app(
    backend: TTSBackend,
    cache: VoicePromptCache,
    *,
    token: str = "",
    model_id: str = TTS_MODEL_ID,
    checkpoint: str = TTS_CHECKPOINT,
    model_root: str | Path = TTS_MODEL_ROOT,
    default_ref_audio_path: str | Path | None = None,
    default_ref_text: str | None = None,
    load_on_startup: bool = True,
) -> web.Application:
    """Build the aiohttp application serving /livez, /health, and /v1/tts/*."""
    env_ref_path = os.environ.get("TTS_DEFAULT_REF_AUDIO_PATH", "")
    resolved_default_path = (
        str(default_ref_audio_path)
        if default_ref_audio_path is not None
        else env_ref_path
    ).strip()
    resolved_default_text = (
        default_ref_text
        if default_ref_text is not None
        else os.environ.get("TTS_DEFAULT_REF_TEXT", "")
    ).strip()
    default_chunk_size = int(os.environ.get("TTS_STREAM_CHUNK_SIZE", "4"))
    if not 1 <= default_chunk_size <= 64:
        raise ValueError("TTS_STREAM_CHUNK_SIZE must be between 1 and 64")
    root_display = f"{str(model_root).rstrip('/\\\\')}/{checkpoint}"

    @web.middleware
    async def authenticate(request: web.Request, handler):
        if token and request.path not in ("/livez", "/health"):
            provided = request.headers.get("Authorization", "").encode()
            expected = f"Bearer {token}".encode()
            if not secrets.compare_digest(provided, expected):
                return failure(
                    401,
                    "unauthorized",
                    "A valid worker bearer token is required.",
                )
        return await handler(request)

    async def on_startup(_app: web.Application) -> None:
        if load_on_startup:
            await backend.load()
            await backend.warmup()

    async def on_cleanup(_app: web.Application) -> None:
        await cache.close()
        await backend.close()

    async def _parse_synthesis_request(
        request: web.Request,
        *,
        allow_chunk_size: bool = False,
    ) -> tuple[str, str, VoiceHandle, int] | web.Response:
        try:
            body = await request.json()
        except (json.JSONDecodeError, ValueError):
            return failure(400, "invalid_json", "Request body must be valid JSON.")
        if not isinstance(body, dict):
            return failure(400, "invalid_request", "JSON body must be an object.")

        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            return failure(400, "invalid_text", "Provide a non-empty 'text' string.")
        clean_text = text.strip()

        raw_ref_text = body.get("ref_text")
        if raw_ref_text is None:
            ref_text = ""
        elif isinstance(raw_ref_text, str):
            ref_text = raw_ref_text
        else:
            return failure(400, "invalid_request", "'ref_text' must be a string.")

        raw_language = body.get("language")
        if raw_language is None or raw_language == "":
            language = "Auto"
        elif isinstance(raw_language, str):
            language = raw_language.strip() or "Auto"
        else:
            return failure(400, "invalid_request", "'language' must be a string.")

        chunk_size = default_chunk_size
        if allow_chunk_size and "chunk_size" in body and body["chunk_size"] is not None:
            raw_chunk = body["chunk_size"]
            if isinstance(raw_chunk, bool) or not isinstance(raw_chunk, int) or not (1 <= raw_chunk <= 64):
                return failure(
                    400,
                    "invalid_chunk_size",
                    "'chunk_size' must be an integer between 1 and 64.",
                )
            chunk_size = raw_chunk

        raw_ref_b64 = body.get("reference_audio_base64")
        if raw_ref_b64 is not None and not isinstance(raw_ref_b64, str):
            return failure(
                400,
                "invalid_request",
                "'reference_audio_base64' must be a base64-encoded WAV string.",
            )

        ref_wav_bytes: bytes | None = None
        effective_ref_text = ref_text
        if isinstance(raw_ref_b64, str) and raw_ref_b64.strip():
            try:
                ref_wav_bytes = _decode_reference_wav(raw_ref_b64)
            except (ValueError, binascii.Error, wave.Error, EOFError) as exc:
                return failure(
                    400,
                    "invalid_reference_audio",
                    f"Invalid reference WAV audio: {exc}",
                )
        elif resolved_default_path:
            default_path = Path(resolved_default_path)
            if not default_path.is_file():
                return failure(
                    500,
                    "default_reference_unavailable",
                    "Configured TTS_DEFAULT_REF_AUDIO_PATH file does not exist.",
                )
            try:
                ref_wav_bytes = default_path.read_bytes()
                decode_wav_to_float32(ref_wav_bytes)
            except (OSError, ValueError, wave.Error, EOFError) as exc:
                return failure(
                    500,
                    "default_reference_unavailable",
                    f"Configured default reference WAV is unreadable: {exc}",
                )
            if not effective_ref_text.strip():
                effective_ref_text = resolved_default_text

        if ref_wav_bytes is not None:
            voice = await cache.get_or_prepare(
                backend,
                ref_wav_bytes,
                ref_text=effective_ref_text,
                language=language,
            )
        elif not backend.requires_reference_audio:
            voice = await backend.prepare_voice(
                "default",
                b"",
                ref_text=effective_ref_text or resolved_default_text,
                language=language,
            )
        else:
            return failure(
                400,
                "missing_reference_audio",
                "Provide 'reference_audio_base64' or configure TTS_DEFAULT_REF_AUDIO_PATH.",
            )

        return clean_text, language, voice, chunk_size

    async def livez(_request: web.Request) -> web.Response:
        return web.json_response({"state": "running"})

    async def health(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ready",
                "backend": backend.name,
                "model": model_id,
                "checkpoint": checkpoint,
                "root": root_display,
                "sample_rate_hz": backend.sample_rate_hz,
                "voice_cache_size": len(cache),
            }
        )

    async def synthesize(request: web.Request) -> web.Response:
        parsed = await _parse_synthesis_request(request, allow_chunk_size=False)
        if isinstance(parsed, web.Response):
            return parsed
        clean_text, language, voice, _ = parsed

        try:
            wav_bytes, sample_rate = await backend.synthesize(
                clean_text,
                voice,
                language=language,
            )
        except Exception:
            logger.exception("TTS synthesis failed")
            return failure(500, "tts_synthesis_failed", "Speech synthesis failed.")

        return web.Response(
            body=wav_bytes,
            content_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-TTS-Voice-Key": voice.voice_key,
                "X-TTS-Ref-Text": base64.b64encode(
                    voice.ref_text.encode("utf-8")
                ).decode("ascii"),
                "X-TTS-Sample-Rate": str(sample_rate),
            },
        )

    async def stream(request: web.Request) -> web.StreamResponse:
        parsed = await _parse_synthesis_request(request, allow_chunk_size=True)
        if isinstance(parsed, web.Response):
            return parsed
        clean_text, language, voice, chunk_size = parsed

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "application/octet-stream",
                "Cache-Control": "no-store",
                "X-TTS-Sample-Rate": str(backend.sample_rate_hz),
                "X-TTS-Channels": "1",
                "X-TTS-Sample-Width": "2",
                "X-TTS-Voice-Key": voice.voice_key,
                "X-TTS-Ref-Text": base64.b64encode(
                    voice.ref_text.encode("utf-8")
                ).decode("ascii"),
            },
        )
        await response.prepare(request)
        try:
            async with aclosing(
                backend.synthesize_stream(
                    clean_text,
                    voice,
                    language=language,
                    chunk_size=chunk_size,
                )
            ) as chunks:
                async for chunk in chunks:
                    if chunk:
                        await response.write(chunk)
            await response.write_eof()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            logger.debug("TTS stream client disconnected")
            return response
        except asyncio.CancelledError:
            logger.debug("TTS stream cancelled")
            raise
        return response

    app = web.Application(client_max_size=MAX_REQUEST_BYTES, middlewares=[authenticate])
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.add_routes(
        [
            web.get("/livez", livez),
            web.get("/health", health),
            web.post("/v1/tts/synthesize", synthesize),
            web.post("/v1/tts/stream", stream),
        ]
    )
    return app
