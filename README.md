# AirLLM Ollama API Wrapper (`airllm-ollama-proxy`)

## Executive Summary

The **AirLLM Ollama API Wrapper** is a lightweight, high-precision proxy service that exposes a standard [Ollama REST API](https://github.com/ollama/ollama) interface powered by **[AirLLM](https://github.com/lyzhang1999/AirLLM)** execution under the hood. 

In microservice architectures or multi-agent AI pipelines, downstream applications often need to dynamically choose between **speed** (low latency, lower accuracy) and **capability/accuracy** (high parameter count, exact reasoning) on a per-request basis. By providing a transparent API surface compatible with standard Ollama client SDKs, this wrapper allows client applications to switch backends effortlessly—simply by toggling the target URL between a standard Ollama instance and this AirLLM wrapper—without changing any business logic or serialization code.

---

## Key Features

* **Transparent Ollama API Compatibility:** Implements core Ollama endpoints (`/api/generate`, `/api/chat`, `/api/tags`) to ensure full drop-in compatibility with existing Ollama software, client libraries, and SDKs.
* **Layered AirLLM Execution:** Leverages AirLLM’s layer-wise inference memory management to run large-parameter models locally on constrained hardware.
* **Single-Model Specialization:** Optimized for single-model deployment, reducing memory overhead and startup complexity.
* **Network & Timeout Optimized:** Built for same-network internal microservice communication where clients manage execution timeouts appropriate for AirLLM inference speeds.
* **Production-Grade Codebase:** Enforces high code quality standards out of the box, including full static type checking (`mypy`), strict linting (`ruff`), and $\ge 80\%$ unit/integration test coverage.

---

## Architectural Context

```
                   ┌──────────────────────────────────────┐
                   │          Client Application           │
                   └──────────────────┬───────────────────┘
                                      │
                   ┌──────────────────┴───────────────────┐
                   │        Inference Router/Proxy       │
                   └──────────┬───────────────────┬───────┘
                              │                   │
         [High Speed / Draft] │                   │ [High Accuracy / Complex]
                              ▼                   ▼
                      ┌──────────────┐     ┌──────────────┐
                      │ Standard     │     │ AirLLM       │
                      │ Ollama API   │     │ Wrapper      │
                      └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │ AirLLM       │
                                           │ Model Engine │
                                           └──────────────┘
```

When speed is prioritized, requests are routed to standard Ollama nodes. When deep accuracy or large-context reasoning is required, requests are routed seamlessly to this AirLLM wrapper.

---

## Getting Started

### Prerequisites

* [uv](https://docs.astral.sh/uv/) (dependency manager; installs the Python interpreter for you)
* Python 3.11+ (managed automatically via `uv python install`)

### Installation

```bash
# Install the pinned dependencies from uv.lock
uv sync --dev

# Optional: install the AirLLM runtime (pulls in torch/transformers, ~2 GB)
uv sync --dev --extra airllm
```

### Running the application locally

```bash
# Start the API server on http://127.0.0.1:11434
uv run uvicorn airllm_ollama.main:app --reload

# ...or via the module entry point (uses AIRLLM_OLLAMA_HOST/PORT)
uv run python -m airllm_ollama

# Verify the service is alive
curl http://127.0.0.1:11434/health
```

Configuration is read from environment variables prefixed with `AIRLLM_OLLAMA_`
(see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `AIRLLM_OLLAMA_HOST` | `127.0.0.1` | Server bind address |
| `AIRLLM_OLLAMA_PORT` | `11434` | Server port (Ollama-compatible) |
| `AIRLLM_OLLAMA_LOG_LEVEL` | `INFO` | Uvicorn log level |
| `AIRLLM_OLLAMA_MODEL_NAME` | `airllm-model` | Name of the single served model |
| `AIRLLM_OLLAMA_MODEL_PATH` | _(empty)_ | Filesystem path of the model artefacts |

### Running the tests

```bash
# Full suite with coverage (enforces >= 80% coverage)
uv run pytest

# Single file / single test
uv run pytest tests/test_api_health.py
uv run pytest tests/test_api_health.py::test_health_returns_ok

# HTML coverage report
uv run coverage html && open htmlcov/index.html
```

### Quality gates

The same commands run in CI on every push and pull request:

```bash
uv lock --check              # lock file is up to date
uv run ruff format --check . # formatting
uv run ruff check .          # linting
uv run mypy .                # static type checking (strict)
uv run pytest                # tests + coverage gate
```

### Project layout

```
src/airllm_ollama/
├── __init__.py        # package exports
├── __main__.py        # `python -m airllm_ollama` entry point
├── api/routes.py      # HTTP routes
├── config.py          # environment-driven settings
├── main.py            # FastAPI application factory
├── schemas/           # pydantic request/response models
└── version.py         # package version
tests/                 # pytest suite
.github/workflows/ci.yml
```

