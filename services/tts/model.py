"""Model snapshot pinning, download, and offline verification for TTS weights."""

import os
from pathlib import Path

TTS_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TTS_CHECKPOINT = "5d83992436eae1d760afd27aff78a71d676296fc"
TTS_MODEL_ROOT = "/models"

REQUIRED_SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/configuration.json",
    "speech_tokenizer/preprocessor_config.json",
    "speech_tokenizer/model.safetensors",
)


def validate_checkpoint(revision: str) -> str:
    """Enforce a strict 40-character lowercase hexadecimal Hugging Face commit SHA."""
    if not isinstance(revision, str) or len(revision) != 40 or any(
        ch not in "0123456789abcdef" for ch in revision
    ):
        raise ValueError("TTS_CHECKPOINT must be an exact Hugging Face commit")
    return revision


def resolve_backend_name(backend: str | None = None) -> str:
    """Return normalized TTS backend identifier ('qwen3' or 'genie')."""
    value = backend if backend is not None else os.environ.get("TTS_BACKEND", "qwen3")
    return value.strip().lower() or "qwen3"


def location() -> tuple[str, str, Path]:
    """Return configured (model_id, revision, snapshot_path) from environment."""
    model_id = os.environ.get("TTS_MODEL_ID", TTS_MODEL_ID).strip()
    if not model_id:
        raise ValueError("TTS_MODEL_ID must be non-empty")
    revision = validate_checkpoint(os.environ.get("TTS_CHECKPOINT", TTS_CHECKPOINT))
    model_root = os.environ.get("TTS_MODEL_ROOT") or TTS_MODEL_ROOT
    return model_id, revision, Path(model_root) / revision


def validate(path: str | Path, *, backend: str | None = None) -> Path:
    """Validate that required snapshot files exist on disk for the active backend."""
    snapshot_path = Path(path)
    if resolve_backend_name(backend) == "qwen3":
        missing = [
            rel for rel in REQUIRED_SNAPSHOT_FILES if not (snapshot_path / rel).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Incomplete TTS snapshot: " + ", ".join(missing)
            )
    return snapshot_path


def fetch(model: str, revision: str, path: str | Path) -> Path:
    """Download a pinned Hugging Face TTS snapshot, validate it, and mark it ready."""
    validate_checkpoint(revision)
    snapshot_path = Path(path)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model, revision=revision, local_dir=snapshot_path)
    validate(snapshot_path, backend="qwen3")
    temporary = snapshot_path / f".snapshot-ready.{os.getpid()}"
    temporary.write_text(f"{model}@{revision}\n", encoding="utf-8")
    temporary.replace(snapshot_path / ".snapshot-ready")
    return snapshot_path


def require_ready(
    model: str,
    revision: str,
    path: str | Path,
    *,
    backend: str | None = None,
) -> Path:
    """Verify atomic readiness marker and required snapshot files for offline serving."""
    validate_checkpoint(revision)
    snapshot_path = Path(path)
    backend_name = resolve_backend_name(backend)
    if backend_name == "qwen3":
        marker = snapshot_path / ".snapshot-ready"
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != f"{model}@{revision}"
        ):
            raise RuntimeError(
                "Pinned TTS checkpoint is not ready. Run the tts-model-fetch service first."
            )
        validate(snapshot_path, backend=backend_name)
    return snapshot_path
