"""Pluggable TTS backends (Qwen3-TTS via faster-qwen3-tts and CPU ONNX Genie-TTS)."""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import io
import os
from pathlib import Path
import struct
import tempfile
import threading
from typing import Any
import wave

SAMPLE_RATE_HZ = 24000

_LANGUAGE_ALIASES = {
    "": "Auto",
    "auto": "Auto",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
    "zh-tw": "Chinese",
    "zh-hant": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "english": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}


@dataclass
class VoiceHandle:
    """Cached speaker representation prepared by a TTS backend."""

    voice_key: str
    ref_text: str = ""
    language: str = "Auto"
    prompt_data: object = None
    x_vector_only: bool = True


def normalize_language(language: str | None) -> str:
    """Normalize ISO or full language names to Qwen3-TTS canonical identifiers."""
    if not language:
        return "Auto"
    stripped = language.strip()
    return _LANGUAGE_ALIASES.get(stripped.lower(), stripped or "Auto")


def _try_import_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def pcm_float_to_int16_le(samples: Any) -> bytes:
    """Convert normalized float audio samples [-1.0, 1.0] into signed 16-bit LE PCM bytes."""
    if samples is None:
        return b""
    np = _try_import_numpy()
    if np is not None:
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()
        arr = np.asarray(samples)
        if arr.size == 0:
            return b""
        if arr.dtype == np.int16:
            return arr.reshape(-1).astype("<i2", copy=False).tobytes()
        flat = arr.astype(np.float32, copy=False).reshape(-1)
        clipped = np.clip(flat, -1.0, 1.0)
        return np.rint(clipped * 32767.0).astype("<i2").tobytes()

    values = list(samples)
    if not values:
        return b""
    ints = [int(round(max(-1.0, min(1.0, float(v))) * 32767.0)) for v in values]
    return struct.pack(f"<{len(ints)}h", *ints)


def encode_wav_pcm16(
    pcm16_bytes: bytes,
    sample_rate: int = SAMPLE_RATE_HZ,
    channels: int = 1,
) -> bytes:
    """Wrap raw signed 16-bit little-endian PCM bytes in a RIFF/WAVE container."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels < 1:
        raise ValueError("channels must be >= 1")
    if len(pcm16_bytes) % (2 * channels) != 0:
        raise ValueError("PCM16 byte length must align with channel frame size")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm16_bytes)
    return buf.getvalue()


def _decode_pcm_bytes(raw: bytes, width: int, channels: int, np: Any):
    if width == 1:
        if np is not None:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            arr = [(b - 128) / 128.0 for b in raw]
    elif width == 2:
        if np is not None:
            arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        else:
            count = len(raw) // 2
            arr = [v / 32768.0 for v in struct.unpack(f"<{count}h", raw)]
    elif width == 3:
        triples = [
            int.from_bytes(raw[i : i + 3], "little", signed=True)
            for i in range(0, len(raw), 3)
        ]
        if np is not None:
            arr = np.asarray(triples, dtype=np.float32) / 8388608.0
        else:
            arr = [v / 8388608.0 for v in triples]
    elif width == 4:
        if np is not None:
            arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            count = len(raw) // 4
            arr = [v / 2147483648.0 for v in struct.unpack(f"<{count}i", raw)]
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")

    if channels == 2:
        if np is not None:
            arr = arr.reshape(-1, 2).mean(axis=1).astype(np.float32, copy=False)
        else:
            arr = [(arr[i] + arr[i + 1]) * 0.5 for i in range(0, len(arr), 2)]
    return arr


def decode_wav_to_float32(wav_bytes: bytes) -> tuple[Any, int]:
    """Decode PCM WAV bytes into mono float32 samples in [-1.0, 1.0] and sample rate."""
    if not wav_bytes:
        raise ValueError("WAV payload must be non-empty")
    np = _try_import_numpy()
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            channels, width, rate, frames, *_ = reader.getparams()
            if not (1 <= channels <= 2 and width in (1, 2, 3, 4) and 8000 <= rate <= 192000 and frames > 0):
                raise ValueError("Invalid PCM WAV parameters")
            raw = reader.readframes(frames)
            if len(raw) != frames * channels * width:
                raise ValueError("Incomplete PCM WAV frames")
            mono = _decode_pcm_bytes(raw, width, channels, np)
            if np is not None and not np.any(mono):
                raise ValueError("WAV payload contains only silence")
            if np is None and not any(mono):
                raise ValueError("WAV payload contains only silence")
            return mono, int(rate)
    except wave.Error as err:
        try:
            import soundfile as sf

            data, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
            if data.size == 0 or not (8000 <= int(rate) <= 192000) or data.shape[1] > 2:
                raise ValueError("Invalid WAV audio")
            mono = data.mean(axis=1).astype("float32", copy=False)
            return mono if np is not None else mono.tolist(), int(rate)
        except ImportError:
            raise ValueError(f"Invalid or unsupported WAV input: {err}") from err


def _resample_linear(samples: Any, src_rate: int, dst_rate: int) -> Any:
    if src_rate == dst_rate:
        return samples
    np = _try_import_numpy()
    if np is not None:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return arr
        dst_len = max(1, int(round(arr.size * dst_rate / src_rate)))
        src_x = np.linspace(0.0, 1.0, num=arr.size, endpoint=True)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=True)
        return np.interp(dst_x, src_x, arr).astype(np.float32)

    vals = list(samples)
    if not vals:
        return []
    dst_len = max(1, int(round(len(vals) * dst_rate / src_rate)))
    if len(vals) == 1:
        return [vals[0]] * dst_len
    scale = (len(vals) - 1) / max(1, dst_len - 1)
    out = []
    for i in range(dst_len):
        pos = i * scale
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        frac = pos - lo
        out.append(vals[lo] * (1.0 - frac) + vals[hi] * frac)
    return out


class TTSBackend(ABC):
    """Abstract interface for voice-cloning and streaming TTS engines."""

    name: str = "base"
    sample_rate_hz: int = SAMPLE_RATE_HZ
    requires_reference_audio: bool = True

    @abstractmethod
    async def load(self) -> None:
        """Load model weights into memory/device."""

    @abstractmethod
    async def warmup(self) -> None:
        """Warm up inference graphs/kernels prior to serving."""

    @abstractmethod
    async def close(self) -> None:
        """Release model resources cleanly on shutdown."""

    @abstractmethod
    async def prepare_voice(
        self,
        voice_key: str,
        ref_wav_bytes: bytes,
        ref_text: str = "",
        language: str = "Auto",
    ) -> VoiceHandle:
        """Extract and return a reusable speaker prompt handle."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
    ) -> tuple[bytes, int]:
        """Synthesize full 24 kHz mono 16-bit PCM WAV bytes and sample rate."""

    @abstractmethod
    def synthesize_stream(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
        chunk_size: int = 4,
    ) -> AsyncIterator[bytes]:
        """Yield raw 24 kHz mono signed 16-bit LE PCM audio chunks."""


class Qwen3TTSBackend(TTSBackend):
    """GPU Qwen3-TTS backend powered by faster-qwen3-tts with CUDA graph safety."""

    name = "qwen3"
    sample_rate_hz = SAMPLE_RATE_HZ
    requires_reference_audio = True

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        max_seq_len: int | None = None,
    ):
        self.model_path = Path(model_path)
        self.device = device or os.environ.get("TTS_DEVICE", "cuda:0")
        self.max_seq_len = (
            max_seq_len
            if max_seq_len is not None
            else int(os.environ.get("TTS_MAX_SEQ_LEN", "2048"))
        )
        self._gpu_lock = asyncio.Lock()
        self._model = None
        self._warmed = False

    def _ensure_loaded_sync(self):
        if self._model is not None:
            return self._model
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        self._model = FasterQwen3TTS.from_pretrained(
            str(self.model_path),
            device=self.device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            max_seq_len=self.max_seq_len,
        )
        return self._model

    async def load(self) -> None:
        async with self._gpu_lock:
            if self._model is None:
                await asyncio.to_thread(self._ensure_loaded_sync)

    async def warmup(self) -> None:
        async with self._gpu_lock:
            model = await asyncio.to_thread(self._ensure_loaded_sync)
            if not self._warmed:
                await asyncio.to_thread(model.warmup, prefill_len=100)
                self._warmed = True

    async def close(self) -> None:
        async with self._gpu_lock:
            self._model = None
            self._warmed = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    async def prepare_voice(
        self,
        voice_key: str,
        ref_wav_bytes: bytes,
        ref_text: str = "",
        language: str = "Auto",
    ) -> VoiceHandle:
        samples, sr = decode_wav_to_float32(ref_wav_bytes)
        clean_text = ref_text.strip()
        x_vector_only = not bool(clean_text)
        norm_lang = normalize_language(language)

        def _create_prompt():
            model = self._ensure_loaded_sync()
            inner = getattr(model, "model", None)
            creator = (
                getattr(inner, "create_voice_clone_prompt", None)
                if inner is not None
                else None
            ) or getattr(model, "create_voice_clone_prompt")
            return creator(
                ref_audio=(samples, sr),
                ref_text=clean_text,
                x_vector_only_mode=x_vector_only,
            )

        async with self._gpu_lock:
            prompt_data = await asyncio.to_thread(_create_prompt)
        return VoiceHandle(
            voice_key=voice_key,
            ref_text=clean_text,
            language=norm_lang,
            prompt_data=prompt_data,
            x_vector_only=x_vector_only,
        )

    async def synthesize(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
    ) -> tuple[bytes, int]:
        if voice.prompt_data is None:
            raise ValueError("VoiceHandle is missing prepared prompt_data")
        eff_lang = normalize_language(
            language if language and language.strip().lower() != "auto" else voice.language
        )

        def _run_batch():
            model = self._ensure_loaded_sync()
            return model.generate_voice_clone(
                text=text,
                language=eff_lang,
                voice_clone_prompt=voice.prompt_data,
            )

        async with self._gpu_lock:
            wavs, sr = await asyncio.to_thread(_run_batch)
        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(audio, "ndim") and audio.ndim > 1:
            audio = audio[0]
        sr_int = int(sr)
        if sr_int != self.sample_rate_hz:
            audio = _resample_linear(audio, sr_int, self.sample_rate_hz)
            sr_int = self.sample_rate_hz
        pcm16 = pcm_float_to_int16_le(audio)
        return encode_wav_pcm16(pcm16, sample_rate=sr_int), sr_int

    async def synthesize_stream(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
        chunk_size: int = 4,
    ) -> AsyncIterator[bytes]:
        if voice.prompt_data is None:
            raise ValueError("VoiceHandle is missing prepared prompt_data")
        eff_lang = normalize_language(
            language if language and language.strip().lower() != "auto" else voice.language
        )
        # Hold _gpu_lock across the entire streaming thread so concurrent requests
        # or mid-stream cancellations never overlap CUDA graph state.
        async with self._gpu_lock:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
            stop_event = threading.Event()

            def _stream_worker():
                try:
                    model = self._ensure_loaded_sync()
                    for item in model.generate_voice_clone_streaming(
                        text=text,
                        language=eff_lang,
                        voice_clone_prompt=voice.prompt_data,
                        chunk_size=chunk_size,
                    ):
                        if stop_event.is_set():
                            break
                        if isinstance(item, tuple):
                            audio_chunk = item[0]
                            sr = int(item[1])
                        else:
                            audio_chunk = item
                            sr = self.sample_rate_hz
                        if sr != self.sample_rate_hz:
                            audio_chunk = _resample_linear(
                                audio_chunk, sr, self.sample_rate_hz
                            )
                        pcm_bytes = pcm_float_to_int16_le(audio_chunk)
                        if pcm_bytes and not stop_event.is_set():
                            loop.call_soon_threadsafe(queue.put_nowait, pcm_bytes)
                except BaseException as exc:  # pragma: no cover - propagated to caller
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            worker_task = asyncio.create_task(asyncio.to_thread(_stream_worker))
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                stop_event.set()
                cancelled = False
                while not worker_task.done():
                    try:
                        await asyncio.shield(worker_task)
                    except asyncio.CancelledError:
                        cancelled = True
                if cancelled:
                    raise asyncio.CancelledError


class GenieTTSBackend(TTSBackend):
    """Pluggable CPU ONNX GPT-SoVITS v2 backend adapter (0 MiB VRAM)."""

    name = "genie"
    sample_rate_hz = SAMPLE_RATE_HZ
    requires_reference_audio = False

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        model_dir: str | None = None,
        character_name: str | None = None,
        language: str | None = None,
    ):
        self.data_dir = data_dir if data_dir is not None else os.environ.get("GENIE_DATA_DIR", "")
        self.model_dir = model_dir if model_dir is not None else os.environ.get("GENIE_MODEL_DIR", "")
        self.character_name = (
            character_name
            if character_name is not None
            else os.environ.get("GENIE_CHARACTER_NAME", "default")
        )
        self.language = (
            language
            if language is not None
            else os.environ.get("GENIE_LANGUAGE", "Chinese")
        )
        self._lock = asyncio.Lock()
        self._genie = None
        self._loaded = False
        self._active_voice_key: str | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    @staticmethod
    def _import_genie():
        try:
            import genie_tts

            return genie_tts
        except ImportError as exc:
            raise RuntimeError(
                "GenieTTSBackend requires the optional 'genie-tts' package, "
                "which is not installed in this environment."
            ) from exc

    def _ensure_tmpdir(self) -> Path:
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="omni-tts-genie-")
        return Path(self._tmpdir.name)

    def _ensure_loaded_sync(self):
        if self.data_dir:
            os.environ["GENIE_DATA_DIR"] = self.data_dir
        genie = self._import_genie()
        if not self._loaded:
            if not self.model_dir:
                raise ValueError("GENIE_MODEL_DIR must be configured for GenieTTSBackend")
            genie.load_character(self.character_name, self.model_dir, self.language)
            self._loaded = True
        self._genie = genie
        return genie

    async def load(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._ensure_loaded_sync)

    async def warmup(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._ensure_loaded_sync)

    async def close(self) -> None:
        async with self._lock:
            genie = self._genie
            if genie is not None and self._loaded:
                def _unload():
                    unload = getattr(genie, "unload_character", None)
                    if callable(unload):
                        unload(self.character_name)
                    stop = getattr(genie, "stop", None)
                    if callable(stop):
                        stop()

                await asyncio.to_thread(_unload)
            self._loaded = False
            self._genie = None
            self._active_voice_key = None
            if self._tmpdir is not None:
                self._tmpdir.cleanup()
                self._tmpdir = None

    async def prepare_voice(
        self,
        voice_key: str,
        ref_wav_bytes: bytes,
        ref_text: str = "",
        language: str = "Auto",
    ) -> VoiceHandle:
        clean_text = ref_text.strip()
        eff_lang = self.language if not language or language.strip().lower() == "auto" else normalize_language(language)
        if not ref_wav_bytes:
            return VoiceHandle(
                voice_key=voice_key,
                ref_text=clean_text,
                language=eff_lang,
                prompt_data=None,
                x_vector_only=True,
            )

        decode_wav_to_float32(ref_wav_bytes)

        def _bind_ref() -> str:
            genie = self._ensure_loaded_sync()
            ref_path = self._ensure_tmpdir() / f"ref_{voice_key[:24]}.wav"
            ref_path.write_bytes(ref_wav_bytes)
            genie.set_reference_audio(
                self.character_name,
                str(ref_path),
                clean_text,
                eff_lang,
            )
            self._active_voice_key = voice_key
            return str(ref_path)

        async with self._lock:
            ref_path = await asyncio.to_thread(_bind_ref)
        return VoiceHandle(
            voice_key=voice_key,
            ref_text=clean_text,
            language=eff_lang,
            prompt_data=ref_path,
            x_vector_only=not bool(clean_text),
        )

    async def synthesize(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
    ) -> tuple[bytes, int]:
        eff_lang = (
            voice.language
            if not language or language.strip().lower() == "auto"
            else normalize_language(language)
        )

        def _run_genie() -> bytes:
            genie = self._ensure_loaded_sync()
            if voice.prompt_data and self._active_voice_key != voice.voice_key:
                genie.set_reference_audio(
                    self.character_name,
                    str(voice.prompt_data),
                    voice.ref_text,
                    eff_lang,
                )
                self._active_voice_key = voice.voice_key
            out_path = self._ensure_tmpdir() / f"out_{threading.get_ident()}.wav"
            genie.tts(self.character_name, text, save_path=str(out_path))
            return out_path.read_bytes()

        async with self._lock:
            raw_wav = await asyncio.to_thread(_run_genie)
        samples, sr = decode_wav_to_float32(raw_wav)
        if sr != self.sample_rate_hz:
            samples = _resample_linear(samples, sr, self.sample_rate_hz)
        pcm16 = pcm_float_to_int16_le(samples)
        return encode_wav_pcm16(pcm16, sample_rate=self.sample_rate_hz), self.sample_rate_hz

    async def synthesize_stream(
        self,
        text: str,
        voice: VoiceHandle,
        language: str = "Auto",
        chunk_size: int = 4,
    ) -> AsyncIterator[bytes]:
        wav_bytes, sr = await self.synthesize(text, voice, language=language)
        with wave.open(io.BytesIO(wav_bytes), "rb") as reader:
            pcm = reader.readframes(reader.getnframes())
        # Chunk at ~80ms per codec step (1920 samples = 3840 bytes at 24 kHz PCM16).
        frame_bytes = max(3840, max(1, chunk_size) * 1920 * 2)
        for offset in range(0, len(pcm), frame_bytes):
            chunk = pcm[offset : offset + frame_bytes]
            if chunk:
                yield chunk


def create_backend(
    name: str | None = None,
    *,
    model_path: str | Path | None = None,
    device: str | None = None,
    max_seq_len: int | None = None,
    data_dir: str | None = None,
    model_dir: str | None = None,
    character_name: str | None = None,
    language: str | None = None,
) -> TTSBackend:
    """Instantiate the requested TTSBackend ('qwen3' or 'genie')."""
    backend_name = (name if name is not None else os.environ.get("TTS_BACKEND", "qwen3")).strip().lower()
    if backend_name == "qwen3":
        if model_path is None:
            raise ValueError("model_path is required for Qwen3TTSBackend")
        return Qwen3TTSBackend(model_path, device=device, max_seq_len=max_seq_len)
    if backend_name == "genie":
        return GenieTTSBackend(
            data_dir=data_dir,
            model_dir=model_dir,
            character_name=character_name,
            language=language,
        )
    raise ValueError(f"Unsupported TTS backend: {backend_name!r}")
