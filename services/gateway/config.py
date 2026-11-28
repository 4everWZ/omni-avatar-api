"""Validated gateway configuration and per-backend policy overrides."""
import json
import math
import os
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit

MODEL_ALIAS = "omni-default"


def validate_origin(value, name):
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username
            or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError(f'{name} must be an HTTP(S) origin')


def positive(name, default, convert=float):
    value = convert(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def optional_non_negative_int(name):
    raw = os.environ.get(name)
    if raw is None:
        return None
    if isinstance(raw, (bool, float)):
        raise ValueError(f"{name} must be a non-negative integer")
    text = raw.strip() if isinstance(raw, str) else str(raw)
    if not text:
        return None
    if not text.isascii():
        raise ValueError(f"{name} must be a non-negative integer")
    value = int(text)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def optional_origin(name):
    value = os.environ.get(name, "").strip().rstrip("/")
    if value:
        validate_origin(value, name)
    return value


def optional_positive(name, default):
    return positive(name, default) if os.environ.get(name, "").strip() else default


def parse_model_aliases(raw: str, primary_alias: str) -> tuple[tuple[str, str], ...]:
    """Parse comma-separated alias=upstream model mappings and reject invalid entries."""
    if not raw or not raw.strip():
        return ()
    aliases = []
    seen = set()
    for item in raw.split(","):
        entry = item.strip()
        if not entry:
            raise ValueError("LLM_EXTRA_MODEL_ALIASES entries must be non-empty alias=upstream pairs")
        parts = entry.split("=")
        if len(parts) != 2:
            raise ValueError(f"Malformed model alias entry {entry!r}; expected alias=upstream")
        alias, upstream = parts[0].strip(), parts[1].strip()
        if not alias or not upstream:
            raise ValueError(f"Model alias entry {entry!r} must have non-empty alias and upstream model")
        if alias == primary_alias:
            raise ValueError(f"Extra model alias {alias!r} cannot collide with primary alias")
        if alias in seen:
            raise ValueError(f"Duplicate model alias {alias!r}")
        seen.add(alias)
        aliases.append((alias, upstream))
    return tuple(aliases)


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
    model_alias: str = MODEL_ALIAS
    public_prefix: str = ""
    realtime_enabled: bool = True
    prompt_tokens_path: str = "/v1/chat/prompt_tokens"
    token_budget: int = 0
    shared_capacity: bool = False
    default_thinking_token_budget: int | None = None
    tts_worker_url: str = ""
    tts_internal_token: str = field(default="", repr=False)
    tts_timeout: float = 60.0
    asr_worker_url: str = ""
    asr_internal_token: str = field(default="", repr=False)
    asr_timeout: float = 180.0
    model_aliases: tuple[tuple[str, str], ...] = ()

    def resolve_upstream_model(self, requested: str | None) -> str | None:
        """Resolve a client model alias to its upstream model identifier."""
        if not isinstance(requested, str) or not requested:
            return None
        if requested == self.model_alias:
            return self.model
        for alias, upstream in self.model_aliases:
            if requested == alias:
                return upstream
        return None

    def accepted_aliases(self) -> tuple[str, ...]:
        """Return all configured public model aliases starting with the primary alias."""
        return (self.model_alias, *(alias for alias, _ in self.model_aliases))

    @classmethod
    def from_env(cls):
        def required(name):
            if value := os.environ.get(name, "").strip():
                return value
            raise ValueError(f"Missing {name}")

        omni_enabled = os.environ.get('OMNI_GATEWAY_OMNI_ENABLED', '1')
        if omni_enabled not in ('0', '1'):
            raise ValueError('OMNI_GATEWAY_OMNI_ENABLED must be 0 or 1')
        engine_url = ''
        if omni_enabled == '1':
            engine_url = os.environ.get('OMNI_ENGINE_URL', 'http://omni-engine:8000').rstrip('/')
            validate_origin(engine_url, 'OMNI_ENGINE_URL')
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
        if shared_capacity and not (1 <= capacity <= 16):
            raise ValueError("OMNI_WORKSTATION_CAP must be between 1 and 16")
        checkpoint = required("OMNI_CHECKPOINT") if engine_url else ""
        model_root = f"/models/{checkpoint}" if os.environ.get("OMNI_REQUIRE_CHECKPOINT_ROOT") == "1" else ""
        return cls(
            engine_url=engine_url, model=required("OMNI_MODEL_ID") if engine_url else "", checkpoint=checkpoint,
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
            tts_worker_url=optional_origin("OMNI_TTS_WORKER_URL"),
            tts_internal_token=os.environ.get("OMNI_TTS_INTERNAL_TOKEN", "").strip(),
            tts_timeout=optional_positive("OMNI_TTS_TIMEOUT", cls.tts_timeout),
            asr_worker_url=optional_origin("OMNI_ASR_WORKER_URL"),
            asr_internal_token=os.environ.get("OMNI_ASR_INTERNAL_TOKEN", "").strip(),
            asr_timeout=optional_positive("OMNI_ASR_TIMEOUT", cls.asr_timeout),
        )

    def text_backend(self):
        def override(name, default, convert=float):
            return positive(name, default, convert) if os.environ.get(name) else default
        origin = os.environ["LLM_ENGINE_URL"].rstrip("/")
        validate_origin(origin, 'LLM_ENGINE_URL')
        model, revision = os.environ["LLM_MODEL_ID"].strip(), os.environ["LLM_CHECKPOINT"].strip()
        if not model or not revision:
            raise ValueError("LLM model and checkpoint are required")
        return replace(self, engine_url=origin, model=model, checkpoint=revision,
                       model_root=f"/models/{revision}", engine_token=os.environ.get("LLM_ENGINE_TOKEN", ""),
                       engine_version=os.environ.get("LLM_ENGINE_VERSION", "").strip() or "vLLM 0.34.0",
                       model_alias="llm-default", public_prefix="/llm",
                       realtime_enabled=False, prompt_tokens_path="/tokenize", program_token="",
                       queue_limit=override("LLM_QUEUE_LIMIT", self.queue_limit, int),
                       queue_timeout=override("LLM_QUEUE_TIMEOUT", self.queue_timeout),
                       request_timeout=override("LLM_REQUEST_TIMEOUT", self.request_timeout),
                       batch_cap=override("LLM_MAX_NUM_SEQS", 4, int),
                       token_budget=override("LLM_TOKEN_BUDGET", 262144, int),
                       default_thinking_token_budget=optional_non_negative_int("LLM_DEFAULT_THINKING_TOKEN_BUDGET"),
                       model_aliases=parse_model_aliases(os.environ.get("LLM_EXTRA_MODEL_ALIASES", ""), "llm-default"))
