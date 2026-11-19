# Development

Run commands from the repository root unless an explicit path is shown. Python 3.12 is the service baseline. Use `uv --no-project --with-requirements` for isolated local test dependencies; container dependencies remain local to each image.

## Local service entrypoints

```sh
uv run --no-project --with-requirements services/gateway/requirements.txt python -m services.gateway
uv run --no-project --with-requirements services/avatar/requirements.txt python -m services.avatar
```

Supply the service's documented environment before starting it. Gateway requires engine identity and application/admin credentials. Avatar requires its private bearer. For local Python entrypoints, export the required values into the process environment.

All HTTP factories are importable from `services.<service>.app`. Production code uses relative imports inside its package. Keep external service dependencies behind their transport client.

## Contract tests

Run lightweight contract checks with fixture engines:

```sh
uv run --no-project --with-requirements services/gateway/requirements.txt python -B -m unittest discover -s services/gateway/tests -v
uv run --no-project --with-requirements services/avatar/requirements.txt python -B -m unittest discover -s services/avatar/tests -v
uv run --no-project --with-requirements services/avatar/requirements.txt python -B -m unittest discover -s tests/integration -v
python -B -m unittest services.inference.tests.test_model services.inference.tests.test_chat_validation services.llm.tests.test_llm_admission services.llm.tests.test_reasoning_budget services.inference.tests.test_reasoning_budget services.inference.tests.test_patch_minicpmo_awq_stages tests.tooling.test_compose -v
```

INT4 layout/reference tests additionally need PyTorch. Run them in the pinned inference environment:

```sh
docker run --rm --entrypoint python3 -v "$PWD:/work:ro" -w /work \
  omni-api-engine:0.32.0-workstation \
  -B -m unittest services.inference.tests.test_minicpmo_int4_window -v
```

## Compose and image checks

```sh
python tools/compose.py --dry-run --vpn -- config --quiet
python tools/compose.py --vpn -- config --quiet
python tools/compose.py -- build omni-engine llm-engine omni-gateway omni-avatar
```

`--dry-run` prints the wrapper's selected paths and options. Real configuration checks need an environment file; use `--env-file <path>` for fixtures or alternate installations. The wrapper resolves an explicitly supplied relative environment path from the caller's working directory, then invokes Docker from the repository root.

Gateway and Avatar images contain their own package at `/srv/<service>` and start it with `python -m <service>`. Build contexts contain the individual service source and dependencies.

## Real-device qualification

```sh
uv run --no-project --with-requirements services/avatar/requirements.txt \
  python -m tools.qualification.inference --env .env --seconds 1200 --report tmp/inference-check.json
```

For single/four-request TPS with an active CPU avatar, run `python -m tools.qualification.throughput --env .env --avatar-id ava --seconds 90 --report tmp/throughput-check.json` in the same isolated environment. Foreground Omni text TPS and full mixed-load checks use `python -m tools.qualification.omni_throughput --avatar-id <id> --seconds 90 --sliding-window-mode basic --report tmp/omni-throughput.json`. See [tools](../tools/README.md) for qualification scope and diagnostics. Run only checks relevant to the change. Source/import/deployment refactors use contract, path and image checks; inference behavior and performance changes also use the corresponding real-device qualification.

## Contributions

Keep service changes, their tests and maintained API/deployment documentation in the same logical change. Update English and Chinese root READMEs together when setup or user-facing capabilities change. Preserve model pins, ownership and cancellation semantics unless a deliberate contract change is agreed.

Use `tmp/` for disposable probes. Put reusable tools under `tools/`, shared inputs under `evaluations/fixtures/`, and retained qualification evidence under `evaluations/results/`. Keep credentials and downloaded model assets in local ignored storage.
