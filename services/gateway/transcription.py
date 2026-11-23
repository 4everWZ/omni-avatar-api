"""Gateway proxy for OpenAI-compatible /v1/audio/transcriptions routed to the ASR worker."""

import asyncio
import os

from aiohttp import ClientError, ClientSession, ClientTimeout, DummyCookieJar, FormData, web

from .config import positive, validate_origin

MAX_AUDIO_BYTES = 25 * 1024 * 1024
ALLOWED_FORMATS = {"verbose_json", "json", "text"}


def _failure(status: int, code: str, message: str):
    return web.json_response({"error": {"code": code, "message": message}}, status=status)


class AudioTranscriptionProxy:
    """Forwards authenticated multipart transcription uploads to the private ASR worker."""

    def __init__(
        self,
        worker_url: str,
        worker_token: str,
        *,
        timeout: float = 180.0,
        max_bytes: int = MAX_AUDIO_BYTES,
    ):
        validate_origin(worker_url.rstrip("/"), "OMNI_ASR_WORKER_URL")
        if not worker_token:
            raise ValueError("Missing OMNI_ASR_INTERNAL_TOKEN")
        self.worker_url = worker_url.rstrip("/")
        self.worker_token = worker_token
        self.timeout = float(timeout)
        self.max_bytes = max_bytes
        self.session: ClientSession | None = None

    @classmethod
    def from_env(cls):
        worker_url = os.environ.get("OMNI_ASR_WORKER_URL", "").strip().rstrip("/")
        if not worker_url:
            raise ValueError("Missing OMNI_ASR_WORKER_URL")
        token = os.environ.get("OMNI_ASR_INTERNAL_TOKEN", "").strip()
        timeout = positive("OMNI_ASR_TIMEOUT", 180)
        return cls(worker_url, token, timeout=timeout)

    async def lifecycle(self, app):
        async with ClientSession(
            base_url=self.worker_url,
            headers={"Authorization": f"Bearer {self.worker_token}"},
            cookie_jar=DummyCookieJar(),
            timeout=ClientTimeout(total=self.timeout, connect=5),
            trust_env=False,
        ) as session:
            self.session = session
            try:
                yield
            finally:
                self.session = None

    async def transcribe(self, request):
        if self.session is None:
            return _failure(503, "asr_unavailable", "ASR session is not initialized.")
        if not request.content_type.startswith("multipart/"):
            return _failure(400, "invalid_transcription_request", "Expected multipart/form-data upload.")

        audio_bytes = None
        filename = "audio.mp3"
        content_type = "application/octet-stream"
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
                    content_type = field.headers.get("Content-Type") or "application/octet-stream"
                    chunks = bytearray()
                    while True:
                        chunk = await field.read_chunk(65536)
                        if not chunk:
                            break
                        chunks.extend(chunk)
                        if len(chunks) > self.max_bytes:
                            return _failure(
                                413,
                                "payload_too_large",
                                f"Audio file exceeds {self.max_bytes} byte limit.",
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
            return _failure(400, "invalid_transcription_request", "Malformed multipart audio request.")

        if response_format not in ALLOWED_FORMATS:
            return _failure(
                400,
                "invalid_response_format",
                "response_format must be verbose_json, json, or text.",
            )
        if not audio_bytes:
            return _failure(400, "missing_audio", "Provide a non-empty multipart audio file.")

        form = FormData()
        form.add_field("file", audio_bytes, filename=filename, content_type=content_type)
        if language:
            form.add_field("language", language)
        form.add_field("response_format", response_format)

        try:
            async with asyncio.timeout(self.timeout):
                async with self.session.post(
                    "/transcribe",
                    data=form,
                    allow_redirects=False,
                ) as upstream:
                    body = await upstream.read()
                    upstream_ct = upstream.headers.get("Content-Type", "application/json")
                    return web.Response(
                        status=upstream.status,
                        body=body,
                        content_type=upstream_ct.split(";", 1)[0].strip() or "application/json",
                        headers={"Cache-Control": "no-store"},
                    )
        except TimeoutError:
            return _failure(504, "engine_timeout", "ASR transcription exceeded the deadline.")
        except (ClientError, OSError):
            return _failure(502, "engine_unavailable", "The ASR worker could not serve the request.")
