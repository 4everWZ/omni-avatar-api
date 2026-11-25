"""Pinned Silero ONNX VAD snapshot management and engine model identity validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

DEFAULT_VAD_MODEL_ID = "onnx-community/silero-vad"
DEFAULT_VAD_CHECKPOINT = "e71cae966052b992a7eca6b17738916ce0eca4ec"
DEFAULT_OMNI_MODEL_ID = "omni-duplex-qwen38"
DEFAULT_MAX_MODEL_LEN = 131072
READY_MARKER = ".snapshot-ready"
ONNX_REL_PATH = Path("onnx") / "model.onnx"
ALLOW_PATTERNS = ["onnx/model.onnx", "onnx/model_int8.onnx", "README.md"]


def validate_sha40(value: str, name: str = "checkpoint") -> str:
    """Validate that a Hugging Face revision or checkpoint is a 40-char lowercase hex SHA."""
    if not isinstance(value, str) or len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a 40-character lowercase hexadecimal Hugging Face SHA")
    return value


def vad_location(environ: Mapping[str, str] | None = None) -> tuple[str, str, Path]:
    """Return (model_id, revision_sha, snapshot_dir) for the pinned Silero VAD ONNX model."""
    env = os.environ if environ is None else environ
    model_id = env.get("VAD_MODEL_ID", DEFAULT_VAD_MODEL_ID).strip()
    if not model_id:
        raise ValueError("VAD_MODEL_ID must be non-empty")
    revision = validate_sha40(env.get("VAD_CHECKPOINT", DEFAULT_VAD_CHECKPOINT).strip(), "VAD_CHECKPOINT")
    root = Path(env.get("VAD_MODEL_ROOT", "/models").strip() or "/models")
    return model_id, revision, root / revision


def omni_identity(environ: Mapping[str, str] | None = None) -> tuple[str, str, int]:
    """Return validated (model_id, checkpoint_sha, max_model_len) exposed via /v1/models."""
    env = os.environ if environ is None else environ
    model_id = env.get("OMNI_MODEL_ID", DEFAULT_OMNI_MODEL_ID).strip()
    if not model_id:
        raise ValueError("OMNI_MODEL_ID must be non-empty")
    vad_rev = env.get("VAD_CHECKPOINT", DEFAULT_VAD_CHECKPOINT).strip()
    checkpoint = validate_sha40(env.get("OMNI_CHECKPOINT", vad_rev).strip(), "OMNI_CHECKPOINT")
    raw_len = env.get("OMNI_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)).strip()
    try:
        max_model_len = int(raw_len)
    except ValueError as exc:
        raise ValueError("OMNI_MAX_MODEL_LEN must be an integer") from exc
    if max_model_len < 1024:
        raise ValueError("OMNI_MAX_MODEL_LEN must be at least 1024")
    return model_id, checkpoint, max_model_len


def validate_assets(path: Path) -> Path:
    """Verify the pinned Silero VAD ONNX file is present and non-empty."""
    onnx_path = path / ONNX_REL_PATH
    if not onnx_path.is_file() or onnx_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Incomplete Silero VAD snapshot: missing {onnx_path}")
    return onnx_path


def require_ready(model_id: str, revision: str, path: Path) -> Path:
    """Ensure the snapshot marker matches model_id@revision and ONNX assets exist."""
    validate_sha40(revision, "VAD_CHECKPOINT")
    marker = path / READY_MARKER
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != f"{model_id}@{revision}":
        raise RuntimeError(
            f"Pinned Silero VAD snapshot {model_id}@{revision} is not ready at {path}. "
            "Run the fetch command first."
        )
    return validate_assets(path)


def fetch(model_id: str, revision: str, path: Path) -> Path:
    """Download the pinned Silero VAD ONNX snapshot and write an atomic .snapshot-ready marker."""
    validate_sha40(revision, "VAD_CHECKPOINT")
    path.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(path),
        allow_patterns=ALLOW_PATTERNS,
    )
    onnx_path = validate_assets(path)
    tmp_marker = path / f"{READY_MARKER}.{os.getpid()}"
    tmp_marker.write_text(f"{model_id}@{revision}\n", encoding="utf-8")
    tmp_marker.replace(path / READY_MARKER)
    return onnx_path
