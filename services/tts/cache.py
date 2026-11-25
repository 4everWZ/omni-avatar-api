"""Async-safe LRU cache for speaker voice prompts with optional ASR auto-transcription."""

import asyncio
from collections import OrderedDict
import hashlib
import logging
import os

import aiohttp

from .backends import TTSBackend, VoiceHandle

logger = logging.getLogger(__name__)


class _InFlightLock:
    """Reference-counted single-flight lock per voice cache key."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.waiters = 0


class VoicePromptCache:
    """Bounded async LRU cache mapping reference audio (+ optional ref_text) to VoiceHandle."""

    def __init__(
        self,
        *,
        max_entries: int | None = None,
        asr_url: str | None = None,
        asr_token: str | None = None,
        asr_timeout: float | None = None,
        http_client: aiohttp.ClientSession | None = None,
    ) -> None:
        resolved_max = (
            max_entries
            if max_entries is not None
            else int(os.environ.get("TTS_VOICE_CACHE_MAX", "32"))
        )
        if resolved_max < 1:
            raise ValueError("TTS_VOICE_CACHE_MAX must be >= 1")
        self.max_entries = resolved_max

        raw_asr_url = asr_url if asr_url is not None else os.environ.get("OMNI_ASR_WORKER_URL", "")
        self.asr_url = raw_asr_url.strip().rstrip("/")
        self.asr_token = (
            asr_token if asr_token is not None else os.environ.get("OMNI_ASR_INTERNAL_TOKEN", "")
        )
        resolved_timeout = (
            asr_timeout
            if asr_timeout is not None
            else float(os.environ.get("OMNI_ASR_TIMEOUT", "60"))
        )
        if resolved_timeout <= 0:
            raise ValueError("OMNI_ASR_TIMEOUT must be positive")
        self.asr_timeout = resolved_timeout

        self._client = http_client
        self._owns_client = http_client is None
        self._meta_lock = asyncio.Lock()
        self._entries: OrderedDict[str, VoiceHandle] = OrderedDict()
        self._inflight: dict[str, _InFlightLock] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def make_key(ref_wav_bytes: bytes, *, ref_text: str = "") -> str:
        """Compute deterministic ASCII cache key from reference WAV bytes and explicit ref_text."""
        audio_sha = hashlib.sha256(ref_wav_bytes).hexdigest()
        clean_text = ref_text.strip()
        if clean_text:
            text_sha = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]
            return f"{audio_sha}:{text_sha}"
        return audio_sha

    async def _get_client(self) -> aiohttp.ClientSession:
        if self._client is None or self._client.closed:
            self._client = aiohttp.ClientSession()
            self._owns_client = True
        return self._client

    async def close(self) -> None:
        """Close the internally managed HTTP client session, if any."""
        if self._owns_client and self._client is not None and not self._client.closed:
            await self._client.close()
        if self._owns_client:
            self._client = None

    async def _transcribe_ref(self, ref_wav_bytes: bytes) -> str:
        """Best-effort reference transcript via omni-asr; returns '' on any failure."""
        if not self.asr_url or not ref_wav_bytes:
            return ""
        form = aiohttp.FormData()
        form.add_field(
            "file",
            ref_wav_bytes,
            filename="ref.wav",
            content_type="audio/wav",
        )
        form.add_field("response_format", "json")
        headers = (
            {"Authorization": f"Bearer {self.asr_token}"} if self.asr_token else {}
        )
        try:
            client = await self._get_client()
            async with asyncio.timeout(self.asr_timeout):
                async with client.post(
                    f"{self.asr_url}/transcribe",
                    data=form,
                    headers=headers,
                    allow_redirects=False,
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "ASR reference transcription returned status %s; falling back to x-vector mode",
                            resp.status,
                        )
                        return ""
                    payload = await resp.json()
                    if not isinstance(payload, dict):
                        return ""
                    text = payload.get("text", "")
                    return text.strip() if isinstance(text, str) else ""
        except (
            aiohttp.ClientError,
            TimeoutError,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            logger.warning(
                "ASR reference transcription failed (%s); falling back to x-vector mode",
                exc,
            )
            return ""

    async def get_or_prepare(
        self,
        backend: TTSBackend,
        ref_wav_bytes: bytes,
        *,
        ref_text: str = "",
        language: str = "Auto",
    ) -> VoiceHandle:
        """Return a cached VoiceHandle or prepare it once under a per-key single-flight lock."""
        voice_key = self.make_key(ref_wav_bytes, ref_text=ref_text)
        async with self._meta_lock:
            cached = self._entries.get(voice_key)
            if cached is not None:
                self._entries.move_to_end(voice_key)
                return cached
            flight = self._inflight.get(voice_key)
            if flight is None:
                flight = _InFlightLock()
                self._inflight[voice_key] = flight
            flight.waiters += 1

        try:
            async with flight.lock:
                async with self._meta_lock:
                    cached = self._entries.get(voice_key)
                    if cached is not None:
                        self._entries.move_to_end(voice_key)
                        return cached

                resolved_text = ref_text.strip()
                if not resolved_text and self.asr_url:
                    resolved_text = await self._transcribe_ref(ref_wav_bytes)

                handle = await backend.prepare_voice(
                    voice_key,
                    ref_wav_bytes,
                    ref_text=resolved_text,
                    language=language,
                )

                async with self._meta_lock:
                    self._entries[voice_key] = handle
                    self._entries.move_to_end(voice_key)
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                return handle
        finally:
            async with self._meta_lock:
                flight.waiters -= 1
                if flight.waiters == 0 and self._inflight.get(voice_key) is flight:
                    self._inflight.pop(voice_key, None)
