"""Thread-safe lazy loader and transcriber backed by faster-whisper."""

from pathlib import Path
import shutil
import subprocess
import tempfile
from threading import Lock


class WhisperBackend:
    """Lazily initializes WhisperModel and transcribes bounded audio buffers."""

    def __init__(
        self,
        model_size: str = "small",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        model_dir: str | Path | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model_dir = str(model_dir) if model_dir else None
        self._lock = Lock()
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                kwargs = {"device": self.device, "compute_type": self.compute_type}
                if self.model_dir:
                    kwargs["download_root"] = self.model_dir
                self._model = WhisperModel(self.model_size, **kwargs)
            return self._model

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: str | None = None,
        filename: str = "audio.mp3",
    ) -> dict:
        if not audio_bytes:
            raise ValueError("Audio payload must be non-empty")
        suffix = Path(filename or "audio.mp3").suffix or ".mp3"
        if not suffix.startswith(".") or len(suffix) > 10:
            suffix = ".mp3"
        model = self._get_model()
        with tempfile.TemporaryDirectory(prefix="omni-asr-") as tmpdir:
            audio_path = Path(tmpdir) / f"input{suffix}"
            audio_path.write_bytes(audio_bytes)
            source: object = str(audio_path)
            if shutil.which("ffmpeg"):
                try:
                    import numpy as np

                    proc = subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-i",
                            str(audio_path),
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-f",
                            "f32le",
                            "pipe:1",
                        ],
                        capture_output=True,
                        check=True,
                        timeout=60,
                    )
                except subprocess.CalledProcessError as err:
                    detail = err.stderr.decode("utf-8", errors="ignore").strip()
                    raise ValueError(f"Unsupported or unreadable audio input: {detail or err}") from err
                if not proc.stdout:
                    raise ValueError("Decoded audio stream is empty")
                source = np.frombuffer(proc.stdout, dtype=np.float32)
            segments_iter, info = model.transcribe(
                source,
                language=language or None,
                vad_filter=True,
            )
            segments = []
            texts = []
            for idx, seg in enumerate(segments_iter):
                text = str(getattr(seg, "text", "") or "").strip()
                if not text:
                    continue
                segments.append(
                    {
                        "id": idx,
                        "start": round(float(seg.start), 3),
                        "end": round(float(seg.end), 3),
                        "text": text,
                    }
                )
                texts.append(text)
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            detected_language = str(getattr(info, "language", None) or language or "")
            return {
                "text": " ".join(texts).strip(),
                "language": detected_language,
                "duration": round(duration, 3),
                "segments": segments,
            }
