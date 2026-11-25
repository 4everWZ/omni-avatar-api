"""CPU Voice Activity Detection: Silero ONNX (zero PyTorch) and pure-stdlib EnergyVad."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import struct
from typing import Any, Literal, Protocol

SAMPLE_RATE_HZ = 16000
FRAME_SAMPLES = 512
FRAME_BYTES = FRAME_SAMPLES * 2  # 1024 bytes = 32 ms @ 16 kHz mono PCM16
FRAME_MS = 32


@dataclass(frozen=True)
class VadEvent:
    """Discrete VAD state transition or active-speech frame notification."""

    kind: Literal["speech_started", "speech_active", "speech_stopped"]
    pcm16: bytes = b""
    prob: float = 0.0


class VadDetector(Protocol):
    """Streaming 16 kHz mono PCM16 Voice Activity Detector interface."""

    def process_pcm16(self, pcm16: bytes) -> list[VadEvent]: ...

    def reset(self) -> None: ...


class _BaseFrameVad:
    """Shared 512-sample framing and speech-turn hysteresis state machine."""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        prefix_padding_ms: int = 256,
        silence_duration_ms: int = 600,
        barge_in_min_ms: int = 160,
        min_speech_frames: int | None = None,
        max_utterance_ms: int = 30000,
    ) -> None:
        self._pcm_buf = bytearray()
        self._ring: deque[bytes] = deque(maxlen=0)
        self._pending_speech: list[bytes] = []
        self._utterance = bytearray()
        self._in_speech = False
        self._silence_frames = 0
        self.configure(
            threshold=threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            barge_in_min_ms=barge_in_min_ms,
            min_speech_frames=min_speech_frames,
            max_utterance_ms=max_utterance_ms,
        )

    def configure(
        self,
        *,
        threshold: float | None = None,
        prefix_padding_ms: int | None = None,
        silence_duration_ms: int | None = None,
        barge_in_min_ms: int | None = None,
        min_speech_frames: int | None = None,
        max_utterance_ms: int | None = None,
    ) -> None:
        """Validate and update VAD thresholds and turn-timing limits."""
        if threshold is not None:
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError("VAD threshold must be in [0.0, 1.0]")
            self.threshold = float(threshold)
        if prefix_padding_ms is not None:
            if int(prefix_padding_ms) < 0:
                raise ValueError("prefix_padding_ms must be >= 0")
            self.prefix_padding_ms = int(prefix_padding_ms)
            self.prefix_frames = max(0, math.ceil(self.prefix_padding_ms / FRAME_MS))
            self._ring = deque(self._ring, maxlen=self.prefix_frames)
        if silence_duration_ms is not None:
            if int(silence_duration_ms) <= 0:
                raise ValueError("silence_duration_ms must be > 0")
            self.silence_duration_ms = int(silence_duration_ms)
            self.silence_frames_limit = max(1, math.ceil(self.silence_duration_ms / FRAME_MS))
        if min_speech_frames is not None:
            if int(min_speech_frames) < 1:
                raise ValueError("min_speech_frames must be >= 1")
            self.min_speech_frames = int(min_speech_frames)
        elif barge_in_min_ms is not None:
            if int(barge_in_min_ms) < 0:
                raise ValueError("barge_in_min_ms must be >= 0")
            self.min_speech_frames = max(1, math.ceil(int(barge_in_min_ms) / FRAME_MS))
        if max_utterance_ms is not None:
            if int(max_utterance_ms) <= 0:
                raise ValueError("max_utterance_ms must be > 0")
            self.max_utterance_frames = max(1, math.ceil(int(max_utterance_ms) / FRAME_MS))

    def _score_frame(self, frame_bytes: bytes) -> float:
        raise NotImplementedError

    def _reset_backend(self) -> None:
        pass

    def reset(self) -> None:
        """Reset buffered PCM, hysteresis turn state, and any backend RNN state."""
        self._pcm_buf.clear()
        self._ring.clear()
        self._pending_speech.clear()
        self._utterance.clear()
        self._in_speech = False
        self._silence_frames = 0
        self._reset_backend()

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def flush_utterance(self) -> VadEvent | None:
        """Force-close any active or pending speech turn and return its buffered PCM16."""
        if self._in_speech and self._utterance:
            pcm = bytes(self._utterance)
            self._utterance.clear()
            self._in_speech = False
            self._silence_frames = 0
            return VadEvent(kind="speech_stopped", pcm16=pcm, prob=0.0)
        if self._pending_speech:
            pcm = b"".join(self._ring) + b"".join(self._pending_speech)
            self._ring.clear()
            self._pending_speech.clear()
            return VadEvent(kind="speech_stopped", pcm16=pcm, prob=0.0)
        return None

    def process_pcm16(self, pcm16: bytes) -> list[VadEvent]:
        """Consume arbitrary-length 16 kHz mono PCM16 bytes and emit VAD events."""
        if not pcm16:
            return []
        self._pcm_buf.extend(pcm16)
        events: list[VadEvent] = []
        while len(self._pcm_buf) >= FRAME_BYTES:
            frame = bytes(self._pcm_buf[:FRAME_BYTES])
            del self._pcm_buf[:FRAME_BYTES]
            prob = self._score_frame(frame)
            is_speech = prob >= self.threshold

            if not self._in_speech:
                if is_speech:
                    self._pending_speech.append(frame)
                    if len(self._pending_speech) >= self.min_speech_frames:
                        self._in_speech = True
                        self._utterance.clear()
                        for prefix in self._ring:
                            self._utterance.extend(prefix)
                        for pending in self._pending_speech:
                            self._utterance.extend(pending)
                        self._ring.clear()
                        self._pending_speech.clear()
                        self._silence_frames = 0
                        events.append(VadEvent(kind="speech_started", pcm16=b"", prob=prob))
                else:
                    if self._pending_speech:
                        for transient in self._pending_speech:
                            self._ring.append(transient)
                        self._pending_speech.clear()
                    self._ring.append(frame)
                continue

            # Active speech turn: accumulate audio until trailing silence limit.
            self._utterance.extend(frame)
            if is_speech:
                self._silence_frames = 0
                events.append(VadEvent(kind="speech_active", pcm16=frame, prob=prob))
            else:
                self._silence_frames += 1

            utterance_frames = len(self._utterance) // FRAME_BYTES
            if self._silence_frames >= self.silence_frames_limit or utterance_frames >= self.max_utterance_frames:
                pcm = bytes(self._utterance)
                self._utterance.clear()
                self._in_speech = False
                self._silence_frames = 0
                events.append(VadEvent(kind="speech_stopped", pcm16=pcm, prob=prob))
        return events


def load_silero_session(model_path: str | Path) -> Any:
    """Create a single-threaded CPU onnxruntime InferenceSession for Silero VAD."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


class SileroOnnxVad(_BaseFrameVad):
    """CPU-only Silero VAD v5/v6 ONNX wrapper maintaining (state, context) per session."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        session: Any = None,
        threshold: float = 0.5,
        prefix_padding_ms: int = 256,
        silence_duration_ms: int = 600,
        barge_in_min_ms: int = 160,
        min_speech_frames: int | None = None,
        max_utterance_ms: int = 30000,
    ) -> None:
        import numpy as np

        self._np = np
        if session is not None:
            self._session = session
        elif model_path is not None:
            self._session = load_silero_session(model_path)
        else:
            raise ValueError("Provide model_path or an initialized onnxruntime InferenceSession")

        self._sr = np.array(SAMPLE_RATE_HZ, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        super().__init__(
            threshold=threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            barge_in_min_ms=barge_in_min_ms,
            min_speech_frames=min_speech_frames,
            max_utterance_ms=max_utterance_ms,
        )

    def _reset_backend(self) -> None:
        np = self._np
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def _score_frame(self, frame_bytes: bytes) -> float:
        np = self._np
        chunk_512 = (np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0).reshape(
            1, FRAME_SAMPLES
        )
        inp = np.concatenate([self._context, chunk_512], axis=1)  # (1, 576)
        outputs = self._session.run(
            None,
            {
                "input": inp,
                "state": self._state,
                "sr": self._sr,
            },
        )
        out_prob, new_state = outputs[0], outputs[1]
        self._state = np.asarray(new_state, dtype=np.float32)
        self._context = inp[:, -64:]
        return float(np.asarray(out_prob).reshape(-1)[0])


class EnergyVad(_BaseFrameVad):
    """Pure-stdlib RMS energy VAD sharing the exact state machine for deterministic tests."""

    def __init__(
        self,
        *,
        rms_threshold: float = 0.02,
        threshold: float = 0.5,
        prefix_padding_ms: int = 256,
        silence_duration_ms: int = 600,
        barge_in_min_ms: int = 160,
        min_speech_frames: int | None = None,
        max_utterance_ms: int = 30000,
    ) -> None:
        if rms_threshold <= 0:
            raise ValueError("rms_threshold must be > 0")
        self.rms_threshold = float(rms_threshold)
        super().__init__(
            threshold=threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
            barge_in_min_ms=barge_in_min_ms,
            min_speech_frames=min_speech_frames,
            max_utterance_ms=max_utterance_ms,
        )

    def _score_frame(self, frame_bytes: bytes) -> float:
        samples = struct.unpack(f"<{FRAME_SAMPLES}h", frame_bytes)
        mean_sq = sum(s * s for s in samples) / FRAME_SAMPLES
        rms = math.sqrt(mean_sq) / 32768.0
        # Map RMS onto [0, 1] such that rms == rms_threshold yields self.threshold.
        return min(1.0, (rms / self.rms_threshold) * self.threshold)
