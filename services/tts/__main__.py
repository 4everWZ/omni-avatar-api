"""CLI entrypoint for the standalone zero-shot voice-cloning TTS worker."""

import os
import sys

from aiohttp import web

from .app import create_app
from .backends import create_backend
from .cache import VoicePromptCache
from .model import TTS_MODEL_ROOT, fetch, location, require_ready, resolve_backend_name


def main(argv: list[str] | None = None) -> None:
    """Execute fetch, validate, or serve action for the TTS worker."""
    args = list(sys.argv[1:] if argv is None else argv)
    action = args[0] if args else "serve"
    if len(args) > 1 or action not in ("fetch", "validate", "serve"):
        raise SystemExit("usage: python -m tts [fetch|validate|serve]")

    model_id, revision, snapshot_path = location()
    if action == "fetch":
        fetch(model_id, revision, snapshot_path)
        return

    backend_name = resolve_backend_name()
    require_ready(model_id, revision, snapshot_path, backend=backend_name)
    if action == "validate":
        return

    backend = create_backend(backend_name, model_path=snapshot_path)
    cache = VoicePromptCache()
    token = os.environ.get("OMNI_TTS_INTERNAL_TOKEN", "")
    model_root = os.environ.get("TTS_MODEL_ROOT") or TTS_MODEL_ROOT
    app = create_app(
        backend,
        cache,
        token=token,
        model_id=model_id,
        checkpoint=revision,
        model_root=model_root,
    )
    host = os.environ.get("TTS_HOST", "0.0.0.0")
    port = int(os.environ.get("TTS_PORT", "8000"))
    web.run_app(
        app,
        host=host,
        port=port,
        handler_cancellation=True,
        access_log=None,
    )


if __name__ == "__main__":
    main()
