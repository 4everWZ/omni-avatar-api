"""Authenticated CPU ASR worker HTTP API."""

import asyncio
import logging
import secrets

from aiohttp import web

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_FORMATS = {"verbose_json", "json", "text"}


def failure(status: int, code: str, message: str):
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


def create_app(
    backend,
    token: str,
    *,
    max_concurrency: int = 1,
    max_bytes: int = MAX_AUDIO_BYTES,
):
    if not token:
        raise ValueError("ASR worker token required")
    if max_concurrency < 1:
        raise ValueError("ASR max_concurrency must be >= 1")
    semaphore = asyncio.Semaphore(max_concurrency)

    @web.middleware
    async def authenticate(request, handler):
        if request.path != "/livez" and not secrets.compare_digest(
            request.headers.get("Authorization", "").encode(),
            f"Bearer {token}".encode(),
        ):
            return failure(401, "unauthorized", "A valid worker bearer token is required.")
        return await handler(request)

    async def livez(request):
        return web.json_response({"state": "running"})

    async def transcribe(request):
        if not request.content_type.startswith("multipart/"):
            return failure(400, "invalid_request", "Expected multipart/form-data upload.")
        audio_bytes = None
        filename = "audio.mp3"
        language = None
        response_format = "verbose_json"
        try:
            reader = await request.multipart()
            while True:
                field = await reader.next()
                if field is None:
                    break
                name = field.name or ""
                if name == "file":
                    filename = field.filename or "audio.mp3"
                    chunks = bytearray()
                    while True:
                        chunk = await field.read_chunk(65536)
                        if not chunk:
                            break
                        chunks.extend(chunk)
                        if len(chunks) > max_bytes:
                            return failure(
                                413,
                                "payload_too_large",
                                f"Audio file exceeds {max_bytes} byte limit.",
                            )
                    audio_bytes = bytes(chunks)
                elif name == "language":
                    value = (await field.text()).strip()
                    language = value or None
                elif name == "response_format":
                    value = (await field.text()).strip()
                    if value:
                        response_format = value
                else:
                    await field.release()
        except Exception:
            logging.exception("Invalid ASR multipart upload")
            return failure(400, "invalid_multipart", "Malformed multipart audio payload.")

        if response_format not in ALLOWED_FORMATS:
            return failure(
                400,
                "invalid_response_format",
                "response_format must be verbose_json, json, or text.",
            )
        if not audio_bytes:
            return failure(400, "missing_audio", "Provide a non-empty multipart audio file.")

        try:
            async with semaphore:
                result = await asyncio.to_thread(
                    backend.transcribe,
                    audio_bytes,
                    language=language,
                    filename=filename,
                )
        except ValueError as error:
            return failure(400, "invalid_audio", str(error))
        except Exception:
            logging.exception("ASR inference failed")
            return failure(500, "asr_inference_failed", "Speech transcription failed.")

        if response_format == "text":
            return web.Response(
                text=str(result.get("text", "")),
                content_type="text/plain",
                headers={"Cache-Control": "no-store"},
            )
        if response_format == "json":
            return web.json_response(
                {"text": str(result.get("text", ""))},
                headers={"Cache-Control": "no-store"},
            )
        return web.json_response(
            {
                "task": "transcribe",
                "language": result.get("language", ""),
                "duration": result.get("duration", 0.0),
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
            },
            headers={"Cache-Control": "no-store"},
        )

    app = web.Application(client_max_size=max_bytes + 1024 * 1024, middlewares=[authenticate])
    app.add_routes([web.get("/livez", livez), web.post("/transcribe", transcribe)])
    return app
