# Development

Run commands from the repository root unless an explicit path is shown. Python 3.12 is the service baseline. Use `uv --no-project --with-requirements` for isolated local test dependencies; container dependencies remain local to each image.

## Local service entrypoints

```sh
uv run --no-project --with-requirements services/gateway/requirements.txt python -m services.gateway
uv run --no-project --with-requirements services/avatar/requirements.txt python -m services.avatar
```

Supply the service's documented environment before starting it. Gateway requires engine identity and application/admin credentials. Avatar requires its private bearer. Portrait is normally run in its GPU image because it needs the pinned inference environment and model assets. Local Python entrypoints do not automatically load the Docker `.env` file.

All HTTP factories are importable from `services.<service>.app`. Production code uses relative imports inside its package. Keep external service dependencies behind their transport client; do not add sibling-service imports to application code.

## Contract tests

These checks load no model weights:

```sh
uv run --no-project --with-requirements services/gateway/requirements.txt python -B -m unittest discover -s services/gateway/tests -v
uv run --no-project --with-requirements services/avatar/requirements.txt python -B -m unittest discover -s services/avatar/tests -v
uv run --no-project --with-requirements services/avatar/requirements.txt python -B -m unittest discover -s tests/integration -v
uv run --no-project --with aiohttp --with pillow python -B -m unittest discover -s services/portrait/tests -v
python -B -m unittest services.inference.tests.test_model services.inference.tests.test_chat_validation services.inference.tests.test_llm_admission services.inference.tests.test_patch_minicpmo_awq_stages tests.tooling.test_compose -v
```

INT4 layout/reference tests additionally need PyTorch. Use the pinned inference environment rather than installing GPU libraries into the host's default Python:

```sh
docker run --rm --entrypoint python3 -v "$PWD:/work:ro" -w /work \
  omni-api-engine:0.32.0-workstation \
  -B -m unittest services.inference.tests.test_minicpmo_int4_window -v
```

## Compose and image checks

```sh
python tools/compose.py --dry-run --portrait --vpn -- config --quiet
python tools/compose.py --portrait --vpn -- config --quiet
python tools/compose.py -- build omni-engine omni-gateway omni-avatar
python tools/compose.py --portrait -- build omni-portrait
```

`--dry-run` prints paths and options without reading Docker configuration or displaying credential values. Real configuration checks need an environment file; use `--env-file <path>` for fixtures or alternate installations. The wrapper resolves an explicitly supplied relative environment path from the caller's working directory, then invokes Docker from the repository root.

Gateway, Avatar and Portrait images contain their own package at `/srv/<service>` and start it with `python -m <service>`. Build contexts are service directories; they do not include the repository's `.env`, model data or evaluation output.

## Real-device qualification

```sh
uv run --no-project --with-requirements services/avatar/requirements.txt \
  python -m tools.qualification.inference --env .env --seconds 1200 --report tmp/inference-check.json
uv run --no-project --with-requirements services/avatar/requirements.txt \
  python -m tools.qualification.portrait --env .env --reference /path/to/portrait.png \
  --seconds 120 --report tmp/portrait-check.json
```

See [tools](../tools/README.md) for qualification scope and diagnostics. Run only checks relevant to the change. Source/import/deployment refactors need contract, path and image checks; they do not require repeating long GPU benchmarks when model behavior is unchanged.

## Contributions

Keep service changes, their tests and maintained API/deployment documentation in the same logical change. Update English and Chinese root READMEs together when setup or user-facing capabilities change. Preserve model pins, ownership and cancellation semantics unless a deliberate contract change is agreed.

Use `tmp/` for disposable probes. Put reusable tools under `tools/`, shared inputs under `evaluations/fixtures/`, and retained qualification evidence under `evaluations/results/`. Commit neither credentials nor downloaded model assets.
