"""CLI entrypoint for the standalone CPU ASR worker."""

import os

from aiohttp import web

from .app import create_app
from .model import WhisperBackend


def main():
    token = os.environ["OMNI_ASR_INTERNAL_TOKEN"]
    model_size = os.environ.get("OMNI_ASR_MODEL", "small")
    device = os.environ.get("OMNI_ASR_DEVICE", "cpu")
    compute_type = os.environ.get("OMNI_ASR_COMPUTE_TYPE", "int8")
    model_dir = os.environ.get("OMNI_ASR_MODEL_DIR", "/models/asr")
    max_concurrency = int(os.environ.get("OMNI_ASR_MAX_CONCURRENCY", "1"))
    backend = WhisperBackend(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        model_dir=model_dir,
    )
    app = create_app(backend, token, max_concurrency=max_concurrency)
    web.run_app(app, host="0.0.0.0", port=8000, handler_cancellation=True, access_log=None)


if __name__ == "__main__":
    main()
