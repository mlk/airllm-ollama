# AGENTS.md

Guidance for AI agents (and humans) working in this repository.

## Project

`airllm-ollama` is an Ollama-compatible REST API wrapper backed by
[AirLLM](https://github.com/lyzhang1999/AirLLM) layer-wise inference. Clients point
at this service exactly as they would at Ollama (`/api/tags`, `/api/generate`,
`/api/chat`).

The GitHub issue tracker is the source of truth for scope. Issues are grouped into
milestones (epics):

| Issue | Story | Milestone |
| --- | --- | --- |
| #1 | 1.1 Project scaffold, dependency management, CI pipeline | Epic 1: Foundation, Quality Gates, and CI Setup |
| #2 | 2.1 AirLLM single-model singleton wrapper | Epic 2: Core AirLLM Backend Integration |
| #3 | 3.1 `/api/tags` + health endpoints | Epic 3: Ollama API Endpoints Implementation |
| #4 | 3.2 `/api/generate` endpoint | Epic 3 |
| #5 | 3.3 `/api/chat` endpoint | Epic 3 |
| #6 | 4.1 End-to-end integration tests & health verification | Epic 4: Integration and Verification |

## Workflow rules (mandatory)

1. **One ticket = one pull request.** Never commit directly to `main`.
2. Branch naming: `feature/<issue-number>-<short-slug>` (e.g. `feature/2-airllm-engine`).
3. Open the PR with `gh pr create`, referencing the issue (`Closes #<n>`).
4. CI must be green before the PR is considered done.
5. Do not reformat or restructure unrelated code — keep diffs focused on the ticket.

## Tooling

Dependency management is [uv](https://docs.astral.sh/uv/) (not pip/poetry).
`uv` is installed at `~/.local/bin/uv` — add it to `PATH` if missing:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

| Task | Command |
| --- | --- |
| Install deps (dev) | `uv sync --dev` |
| Install deps + AirLLM runtime | `uv sync --dev --extra airllm` |
| Add/remove a dependency | `uv add <pkg>` / `uv remove <pkg>` |
| Regenerate the lock file | `uv lock` |
| Verify the lock file is current | `uv lock --check` |
| Format | `uv run ruff format .` |
| Lint | `uv run ruff check .` |
| Type check | `uv run mypy .` |
| Tests (with coverage gate) | `uv run pytest` |

Never edit `uv.lock` by hand. Always commit `pyproject.toml` **and** `uv.lock`
together.

## Quality gates (all enforced in CI)

* **ruff** — line length 100, `src` layout, rule set `E,W,F,I,N,UP,B,C4,SIM,ANN,D,RUF,PT`
  with pydocstyle `google` convention. Every module, class and function needs a
  docstring; every function needs full type annotations.
* **mypy** — `strict = true`, `python_version = "3.11"`. No `Any` leaks, no missing
  annotations, no untyped decorators.
* **pytest** — coverage is configured once in `pyproject.toml`: `--cov=airllm_ollama
  --cov-branch` plus `fail_under = 80` under `[tool.coverage.report]`. Run plain
  `uv run pytest` everywhere (CI included) — never re-declare coverage flags on
  the command line, they would override the single source of truth. The 80 %
  coverage gate is enforced locally *and* in CI; keep new code covered by tests.
* `filterwarnings = ["error"]` — any warning (including deprecations from
  dependencies) fails the suite. Fix the cause instead of suppressing it.

## Environment quirks

* **Use `httpx2`, not `httpx`, for `TestClient`.** Starlette ≥ 1.9 raises
  `StarletteDeprecationWarning` when `httpx` is importable, and
  `filterwarnings = ["error"]` turns that into a collection error. `httpx2` is a
  dev dependency; do not add plain `httpx`.
* Local system Python is 3.9 — always use `uv run` (it uses the 3.11 interpreter
  pinned in `.python-version`).
* `airllm` is an **optional extra** because it drags in `torch>=2.4`, `transformers`
  and `accelerate` (multi-GB). CI installs only the core + dev dependency set, so
  code must import `airllm` lazily (inside functions) or behind a
  `TYPE_CHECKING` guard, and tests must mock the engine.
* `src/airllm_ollama/config.py` uses `get_settings()` with an `lru_cache`. Tests that
  change environment variables must call `get_settings.cache_clear()` afterwards.

## Architecture

```
src/airllm_ollama/
├── __init__.py        # package exports
├── __main__.py        # `python -m airllm_ollama` entry point (uvicorn runner)
├── api/routes.py      # HTTP routes (api_router)
├── config.py          # Settings (env prefix AIRLLM_OLLAMA_) + get_settings()
├── main.py            # create_app() factory + module-level `app`
├── schemas/           # pydantic request/response models
└── version.py         # __version__ (must match pyproject.toml version)
```

Planned extensions (keep the seams open):

* `engine.py` (or `engine/`) — `AirLLMEngine` singleton wrapper around AirLLM:
  lazy load of a single model, thread-safe `generate(prompt, **kwargs)`, clear
  exception on a model-name mismatch, AirLLM imported lazily.
* `schemas/` — Ollama-compatible payloads: `GenerateRequest`/`GenerateResponse`
  (`model`, `prompt`, `response`, `done`, `eval_count`, …), `ChatRequest`/`ChatResponse`
  (`messages`, `message`, …), `TagsResponse` (`models` with `name`, `size`, `digest`).
* Endpoints belong in `api/routes.py` (or a new `api/ollama.py` router) and must
  return Ollama's exact JSON shapes so standard Ollama clients work unchanged.

## Testing conventions

* Tests live in `tests/`, mirroring the module path (`tests/test_api_health.py`,
  `tests/test_config.py`, `tests/test_main.py`).
* Use `pytest-mock`'s `mocker` fixture for patching third-party calls (e.g. AirLLM)
  so the suite stays fast and offline.
* Shared fixtures live in `tests/conftest.py` (`app`, `client`, `settings`).
* Cover the happy path **and** the error paths (validation errors, model mismatch,
  upstream failures) — the acceptance criteria ask for ≥ 80 % coverage.
