"""Prepare the fixed model on disk and launch the private Omni engine."""

from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
import tempfile


def model_location():
    model_id = os.environ["OMNI_MODEL_ID"]
    revision = os.environ["OMNI_CHECKPOINT"]
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise ValueError("OMNI_CHECKPOINT must be an exact Hugging Face commit")
    return model_id, revision, Path("/models") / revision


def validate_assets(path):
    weights = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    required = set(weights["weight_map"].values()) | {
        "config.json", "assets/HT_ref_audio.wav", "assets/token2wav/flow.yaml",
        "assets/token2wav/flow.pt", "assets/token2wav/hift.pt",
        "assets/token2wav/campplus.onnx", "assets/token2wav/speech_tokenizer_v2_25hz.onnx",
    }
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise FileNotFoundError(f"Incomplete snapshot: {', '.join(missing)}")


def fetch(model_id, revision, path):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, revision=revision, local_dir=path)
    validate_assets(path)
    temporary = path / f".snapshot-ready.{os.getpid()}"
    temporary.write_text(f"{model_id}@{revision}\n", encoding="utf-8")
    temporary.replace(path / ".snapshot-ready")


def checkpoint_chat_window(path):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("quantization_config", {}).get("quant_method") != "awq":
        raise ValueError("The MiniCPM-o profile requires the AWQ checkpoint")
    window = config.get("max_position_embeddings")
    if type(window) is not int or window < 4096:
        raise ValueError("The AWQ checkpoint did not report a valid Chat context")
    return window


def serve(model_id, revision, path):
    import yaml

    marker = path / ".snapshot-ready"
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != f"{model_id}@{revision}":
        raise RuntimeError("Pinned checkpoint is not ready. Run the Compose model-fetch service first.")
    validate_assets(path)
    spec = find_spec("vllm_omni")
    upstream = Path(next(iter(spec.submodule_search_locations))) / "deploy" / "minicpmo_4_5.yaml"
    if not upstream.is_file():
        raise RuntimeError("Pinned vLLM-Omni package does not include the MiniCPM-o deployment config")
    overlay_path = Path(__file__).with_name("minicpmo-orin.yaml")
    overlay = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    overlay["base_config"] = str(upstream)
    thinker = next(stage for stage in overlay["stages"] if stage["stage_id"] == 0)
    thinker["max_model_len"] = checkpoint_chat_window(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".yaml", delete=False) as deploy:
        yaml.safe_dump(overlay, deploy)
    cmd = [
        "vllm", "serve", str(path), "--omni", "--deploy-config", deploy.name,
        "--served-model-name", model_id, "--trust-remote-code", "--host", "0.0.0.0", "--port", "8000",
    ]
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    actions = {"fetch": fetch, "serve": serve}
    if len(sys.argv) != 2 or sys.argv[1] not in actions:
        raise SystemExit("usage: model.py fetch|serve")
    actions[sys.argv[1]](*model_location())
