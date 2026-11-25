"""CLI entrypoint for the CPU engineered full-duplex omni-engine."""

from __future__ import annotations

import sys

from aiohttp import web

from .app import config_from_env, create_app
from .model import fetch as fetch_vad, require_ready, vad_location
from .vad import SileroOnnxVad, load_silero_session


def serve() -> None:
    """Validate pinned Silero ONNX assets, load shared ORT session, and serve HTTP/WS."""
    vad_model, vad_rev, vad_path = vad_location()
    onnx_path = require_ready(vad_model, vad_rev, vad_path)
    shared_session = load_silero_session(onnx_path)
    cfg, token, cap = config_from_env()
    app = create_app(
        cfg,
        vad_factory=lambda: SileroOnnxVad(session=shared_session, barge_in_min_ms=cfg.barge_in_min_ms),
        engine_token=token,
        workstation_cap=cap,
        vad_ready=True,
    )
    web.run_app(app, host="0.0.0.0", port=8000, handler_cancellation=True, access_log=None)


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    cmd = args[0] if args else "serve"
    if len(args) > 1 or cmd not in {"fetch", "validate", "serve"}:
        raise SystemExit("usage: python -m omni [fetch|validate|serve]")
    if cmd == "fetch":
        fetch_vad(*vad_location())
    elif cmd == "validate":
        require_ready(*vad_location())
    else:
        serve()


if __name__ == "__main__":
    main()
