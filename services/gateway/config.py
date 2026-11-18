"""Validated gateway configuration and per-backend policy overrides."""
import json
import math
import os
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

MODEL_ALIAS = "omni-default"


def positive(name, default, convert=float):
    value = convert(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


@dataclass(frozen=True)
class Config:
    engine_url: str
    model: str
    checkpoint: str
    tokens: tuple[str, ...] = field(repr=False)
    admin_token: str = field(repr=False)
    engine_token: str = field(default="", repr=False)
    program_token: str = field(default="", repr=False)
    engine_version: str = ""
    model_root: str = ""
    batch_cap: int = 1
    queue_limit: int = 16
    queue_timeout: float = 5
    request_timeout: float = 1200
    health_timeout: float = 3
    realtime_idle_timeout: float = 120
    avatar_url: str = ""
    avatar_token: str = field(default="", repr=False)
    avatar_socket: str = ""
    model_alias: str = MODEL_ALIAS
    public_prefix: str = ""
    realtime_enabled: bool = True
    prompt_tokens_path: str = "/v1/chat/prompt_tokens"
    token_budget: int = 0
    shared_capacity: bool = False

    @classmethod
    def from_env(cls):
        def required(name):
            if value := os.environ.get(name, "").strip():
                return value
            raise ValueError(f"Missing {name}")

        engine_url = os.environ.get("OMNI_ENGINE_URL", "http://omni-engine:8000").rstrip("/")
        parsed = urlsplit(engine_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("OMNI_ENGINE_URL must be an HTTP(S) origin")
        primary = required("OMNI_GATEWAY_TOKEN")
        admin = required("OMNI_GATEWAY_ADMIN_TOKEN")
        additional = json.loads(os.environ.get("OMNI_GATEWAY_TOKENS", "{}") or "{}")
        if not isinstance(additional, dict) or not all(isinstance(v, str) and v.strip() for v in additional.values()):
            raise ValueError("OMNI_GATEWAY_TOKENS must map service names to nonempty tokens")
        tokens = (primary, *additional.values())
        if len(set((*tokens, admin))) != len(tokens) + 1:
            raise ValueError("Gateway service and admin tokens must be distinct")
        shared_capacity = os.environ.get("OMNI_SHARED_CAPACITY") == "1"
        if not shared_capacity and positive("OMNI_LIVE_CAP", 1, int) != 1:
            raise ValueError("OMNI_LIVE_CAP is fixed at one pending mixed-load verification")
        capacity = (positive("OMNI_WORKSTATION_CAP", 2, int) if shared_capacity
                    else positive("OMNI_BATCH_CAP", 1, int))
        if shared_capacity and capacity not in (1, 2):
            raise ValueError("OMNI_WORKSTATION_CAP must be 1 or 2")
        checkpoint = required("OMNI_CHECKPOINT")
        model_root = f"/models/{checkpoint}" if os.environ.get("OMNI_REQUIRE_CHECKPOINT_ROOT") == "1" else ""
        return cls(
            engine_url=engine_url, model=required("OMNI_MODEL_ID"), checkpoint=checkpoint,
            tokens=tokens, admin_token=admin, engine_token=os.environ.get("OMNI_ENGINE_TOKEN", ""),
            program_token=additional.get("media", ""),
            engine_version=os.environ.get("OMNI_ENGINE_VERSION", ""),
            model_root=os.environ.get("OMNI_EXPECTED_MODEL_ROOT", model_root),
            batch_cap=capacity, shared_capacity=shared_capacity,
            queue_limit=positive("OMNI_QUEUE_LIMIT", 16, int),
            queue_timeout=positive("OMNI_QUEUE_TIMEOUT", 5),
            request_timeout=positive("OMNI_REQUEST_TIMEOUT", cls.request_timeout),
            health_timeout=positive("OMNI_HEALTH_TIMEOUT", 3),
            realtime_idle_timeout=positive("OMNI_REALTIME_IDLE_TIMEOUT", 120),
            avatar_url=os.environ.get("OMNI_AVATAR_URL", ""),
            avatar_token=os.environ.get("OMNI_AVATAR_INTERNAL_TOKEN", ""),
            avatar_socket=os.environ.get("OMNI_AVATAR_SOCKET", ""),
        )

    def text_backend(self):
        def override(name, default, convert=float):
            return positive(name, default, convert) if os.environ.get(name) else default
        origin = os.environ["LLM_ENGINE_URL"].rstrip("/")
        parsed = urlsplit(origin)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("LLM_ENGINE_URL must be an HTTP(S) origin")
        model, revision = os.environ["LLM_MODEL_ID"].strip(), os.environ["LLM_CHECKPOINT"].strip()
        if not model or not revision:
            raise ValueError("LLM model and checkpoint are required")
        return replace(self, engine_url=origin, model=model, checkpoint=revision,
                       model_root=f"/models/{revision}", engine_token=os.environ.get("LLM_ENGINE_TOKEN", ""),
                       engine_version="vLLM 0.32.0", model_alias="llm-default", public_prefix="/llm",
                       realtime_enabled=False, prompt_tokens_path="/tokenize", program_token="",
                       queue_limit=override("LLM_QUEUE_LIMIT", self.queue_limit, int),
                       queue_timeout=override("LLM_QUEUE_TIMEOUT", self.queue_timeout),
                       request_timeout=override("LLM_REQUEST_TIMEOUT", self.request_timeout),
                       batch_cap=override("LLM_MAX_NUM_SEQS", 4, int),
                       token_budget=override("LLM_TOKEN_BUDGET", 262144, int),
                       avatar_url="", avatar_token="", avatar_socket="")
