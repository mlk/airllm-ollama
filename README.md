# AirLLM Ollama API Wrapper (`airllm-ollama`)

## Executive Summary

The **AirLLM Ollama API Wrapper** is a lightweight, high-precision service that exposes a standard [Ollama REST API](https://github.com/ollama/ollama) interface powered by **[AirLLM](https://github.com/lyzhang1999/AirLLM)** execution under the hood. 

In microservice architectures or multi-agent AI pipelines, downstream applications often need to dynamically choose between **speed** (low latency, lower accuracy) and **capability/accuracy** (high parameter count, exact reasoning) on a per-request basis. By providing a transparent API surface compatible with standard Ollama client SDKs, this wrapper allows client applications to switch backends effortlessly—simply by toggling the target URL between a standard Ollama instance and this AirLLM wrapper—without changing any business logic or serialization code. The model name in the request has to be the one the target backend serves; `GET /api/tags` reports it.

---

## Key Features

* **Transparent Ollama API Compatibility:** Implements core Ollama endpoints (`/api/generate`, `/api/chat`, `/api/tags`) to ensure full drop-in compatibility with existing Ollama software, client libraries, and SDKs.
* **Layered AirLLM Execution:** Leverages AirLLM’s layer-wise inference memory management to run large-parameter models locally on constrained hardware.
* **Single-Model Specialization:** Optimized for single-model deployment, reducing memory overhead and startup complexity.
* **Network & Timeout Optimized:** Built for same-network internal microservice communication where clients manage execution timeouts appropriate for AirLLM inference speeds.
* **Production-Grade Codebase:** Enforces high code quality standards out of the box, including full static type checking (`mypy`), strict linting (`ruff`), and $\ge 80\%$ unit/integration test coverage.

---

## Architectural Context

```mermaid
flowchart TD
    subgraph app["Client Application<br/>(the routing lives here)"]
        router{"Chooses the backend<br/>per request"}
    end

    subgraph repo["airllm-ollama (this repository)"]
        api["Ollama-compatible API<br/>/api/tags · /api/generate · /api/chat"]
        engine["AirLLM Engine<br/>layer-wise inference"]
        api --> engine
    end

    subgraph external["External service"]
        ollama["Standard Ollama API"]
    end

    router -->|"High Speed / Draft"| ollama
    router -->|"High Accuracy / Complex"| api
```

**The routing is not part of this service.** The client application decides per
request whether a prompt needs the speed of a standard Ollama node or the
accuracy of a large model served by AirLLM. This repository provides the
"High Accuracy / Complex" branch of that decision — an Ollama-compatible API in
front of an AirLLM engine. Switching backends is a matter of the base URL and
of the model name the backend serves: this service serves exactly one model
and answers a request for another one with a `404`.

---

## Getting Started

### Prerequisites

* [uv](https://docs.astral.sh/uv/) (dependency manager; installs the Python interpreter for you)
* Python 3.11+ (managed automatically via `uv python install`)

### Installation

```bash
# Install the pinned dependencies from uv.lock
uv sync --dev

# Install the AirLLM runtime (pulls in torch/transformers, ~130 MB for torch
# alone on macOS arm64; the model weights are a separate download).
# On macOS the extra also installs Apple's MLX framework, which AirLLM
# imports but does not depend on.
uv sync --dev --extra airllm
```

### Choosing a model (smallest download first)

The server serves **one** model. The default is deliberately the smallest
checkpoint that exercises the full request path:

| Model | Weights | Notes |
| --- | --- | --- |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | **4.1 MB** | Random weights — smoke tests only. **Default.** |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | 269 MB | Smallest *trained* Llama-architecture instruct model. |
| `Qwen/Qwen2.5-0.5B-Instruct` | 988 MB | Better quality, still small. |

**Where to change it**

* Per install / per run — environment variable or `.env`:
  `AIRLLM_OLLAMA_MODEL_NAME=HuggingFaceTB/SmolLM2-135M-Instruct`
* Code default — `DEFAULT_MODEL_NAME` in `src/airllm_ollama/config.py`
* Local checkpoint (no download at all) — `AIRLLM_OLLAMA_MODEL_PATH=/path/to/checkpoint`

### Downloading only what you need

AirLLM's own bootstrap downloads *every* file of the repository except the
weights — for `SmolLM2-135M-Instruct` that is **1.9 GB of ONNX exports** on top
of the 269 MB of weights. Pre-fetch a filtered snapshot instead and point the
server at the local directory; AirLLM then splits the local checkpoint and
downloads nothing:

```bash
# Optional: put the Hugging Face cache on an external drive
export HF_HOME="$HOME/hf-cache"

# Fetch only the config, tokenizer and weights (~6.5 MB for the tiny model)
uv run --extra airllm python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    allow_patterns=[
        "*.json", "*.safetensors", "tokenizer.model", "merges.txt", "vocab.json",
    ],
    local_dir="models/tiny-random-llama",
))
PY

# Serve the local checkpoint (no further downloads)
export AIRLLM_OLLAMA_MODEL_PATH="$PWD/models/tiny-random-llama"
```

Swap the repo id for `HuggingFaceTB/SmolLM2-135M-Instruct` (≈ 272 MB with this
filter) when you want real completions. AirLLM additionally writes one shard
per layer next to the checkpoint; set
`AIRLLM_OLLAMA_LAYER_SHARDS_SAVING_PATH` to relocate them.

### Running the application locally

```bash
# Start the API server on http://127.0.0.1:8000
# The extra is needed for inference; without it the API still answers, but
# every generation fails with a 503.
uv run --extra airllm uvicorn airllm_ollama.main:app --reload

# ...or via the module entry point (uses AIRLLM_OLLAMA_HOST/PORT)
uv run --extra airllm python -m airllm_ollama

# Verify the service is alive
curl http://127.0.0.1:8000/health
```

The server listens on port **8000** (uvicorn's default). Ollama's own port
11434 is available with `AIRLLM_OLLAMA_PORT=11434` — but a local Ollama
already occupies it.

Configuration is read from environment variables prefixed with `AIRLLM_OLLAMA_`
(see `.env.example`):

| Variable | Default | Description |
| --- | --- | --- |
| `AIRLLM_OLLAMA_APP_NAME` | `airllm-ollama` | Name of the service (title of the OpenAPI schema) |
| `AIRLLM_OLLAMA_HOST` | `127.0.0.1` | Server bind address |
| `AIRLLM_OLLAMA_PORT` | `8000` | Server port |
| `AIRLLM_OLLAMA_LOG_LEVEL` | `INFO` | Uvicorn log level |
| `AIRLLM_OLLAMA_MODEL_NAME` | `hf-internal-testing/tiny-random-LlamaForCausalLM` | Name of the single served model |
| `AIRLLM_OLLAMA_MODEL_PATH` | _(empty)_ | Local checkpoint path; empty means "fetch `<model name>` from Hugging Face" |
| `AIRLLM_OLLAMA_DEVICE` | `cuda:0` | Device passed to AirLLM (ignored by the MLX backend) |
| `AIRLLM_OLLAMA_MAX_SEQ_LEN` | `512` | Maximum context length |
| `AIRLLM_OLLAMA_LAYER_SHARDS_SAVING_PATH` | _(empty)_ | Where AirLLM stores the per-layer shards |
| `AIRLLM_OLLAMA_KEEP_ALIVE` | _(empty)_ | Default lifetime of the loaded model (e.g. `5m`); empty keeps it loaded |
| `AIRLLM_OLLAMA_STREAM_FORMAT` | `ndjson` | Framing of the streaming responses: `ndjson` (like Ollama) or `sse` |

A `.env` file overrides every one of these values. If yours still sets a
variable that has a new default, remove the line to pick the default up again.

### API

The `/api/*` routes mirror Ollama's REST interface, so standard Ollama clients
work against this service unchanged. The `/v1/*` routes mirror the OpenAI API,
so that the official OpenAI client can be pointed at this service.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Plain-text announcement: `airllm-ollama is running` (or `kinda running` without the AirLLM runtime) |
| `GET` | `/health` | Liveness probe; `model_loaded` tells whether the weights are in memory |
| `GET` | `/api/version` | Server version, as in Ollama |
| `GET` | `/api/tags` | The single served model with its metadata, as in Ollama |
| `POST` | `/api/show` | The details of the served model, as in Ollama |
| `GET` | `/api/ps` | The models that are currently loaded, as in Ollama |
| `POST` | `/api/generate` | Completion, as in Ollama (streaming and non-streaming) |
| `POST` | `/api/chat` | Chat completion, as in Ollama (streaming and non-streaming) |
| `POST` | `/api/pull`, `/api/push`, `/api/create`, `/api/copy`, `DELETE /api/delete` | Model management, as in Ollama — always `501`, this service serves a single model |
| `GET` | `/v1/models` | The served model, in the OpenAI format |
| `GET` | `/v1/models/{model_id}` | A single model, in the OpenAI format |
| `POST` | `/v1/chat/completions` | Chat completion, in the OpenAI format (streaming and non-streaming) |
| `POST` | `/v1/completions` | Text completion, in the OpenAI format (streaming and non-streaming) |
| `GET` | `/docs` | Interactive OpenAPI documentation |

```bash
curl http://127.0.0.1:8000/api/tags
```

```json
{
  "models": [
    {
      "name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
      "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
      "modified_at": "2026-10-23T11:29:33.028910Z",
      "size": 4194304,
      "digest": "sha256:35bac10d539b472e8d971660006ffc27ba2736cdf74da9c0a3722603596df9ec",
      "details": {
        "parent_model": "",
        "format": "safetensors",
        "family": "llama",
        "families": ["llama"],
        "parameter_size": "135M",
        "quantization_level": "",
        "context_length": 512
      },
      "capabilities": ["completion"]
    }
  ]
}
```

`/api/tags` derives its metadata from the **local** checkpoint only — size,
`sha256` digest, architecture family and parameter count (read from the
safetensors header). Nothing is ever downloaded to answer it; a model that has
not been fetched yet is listed with empty metadata.

`POST /api/show` answers the details of that model, in the shape Ollama uses:

```bash
curl http://127.0.0.1:8000/api/show -d '{"model": "hf-internal-testing/tiny-random-LlamaForCausalLM"}'
```

```json
{
  "modelfile": "# Modelfile generated by \"airllm-ollama show\"\n…\nFROM /path/to/checkpoint\nTEMPLATE \"\"\"…\"\"\"\nPARAMETER num_ctx 512\n",
  "parameters": "num_ctx                         512\n",
  "template": "…",
  "details": {"parent_model": "", "format": "safetensors", "family": "llama", "families": ["llama"], "parameter_size": "135M", "quantization_level": "", "context_length": 512},
  "model_info": {"model_type": "llama", "general.architecture": "llama", "general.parameter_count": 135000000},
  "capabilities": ["completion"]
}
```

A checkpoint does not hold a `Modelfile`, so the service describes the model it
serves — the way `ollama show` does it.

`GET /api/ps` answers the models that are currently loaded. The model is
loaded lazily, so the list is empty until the first request:

```bash
curl http://127.0.0.1:8000/api/ps
```

```json
{
  "models": [
    {
      "name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
      "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
      "size": 4194304,
      "digest": "sha256:35bac10d539b472e8d971660006ffc27ba2736cdf74da9c0a3722603596df9ec",
      "details": {"…": "…"},
      "expires_at": "2026-10-25T08:00:00Z",
      "size_vram": 0
    }
  ]
}
```

`expires_at` is the moment the model will be unloaded (`AIRLLM_OLLAMA_KEEP_ALIVE`).
`size_vram` is `0`: AirLLM streams the model layer by layer, so the weights are
never in the memory of the device at once.

### Generating

```bash
curl http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "prompt": "Why is the sky blue?",
        "options": {"temperature": 0.7, "num_predict": 128}
      }'
```

```json
{
  "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
  "created_at": "2026-10-23T11:45:10.123456Z",
  "response": "The sky is blue because...",
  "done": true,
  "done_reason": "stop",
  "total_duration": 812345678,
  "load_duration": 402345678,
  "prompt_eval_count": 6,
  "prompt_eval_duration": 1200000,
  "eval_count": 24,
  "eval_duration": 408945678
}
```

With `stream: true` the answer is streamed: one JSON object per line, each
one a partial answer, the last one carrying the timings and `done: true` — the
format Ollama uses (`application/x-ndjson`). The official Ollama clients parse
the response line by line, so this is the default; Server-Sent Events can be
selected with `AIRLLM_OLLAMA_STREAM_FORMAT=sse`.

```bash
curl -N http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model": "...", "prompt": "Why is the sky blue?", "stream": true}'
```

```
{"model":"...","created_at":"...","response":"The ","done":false}
{"model":"...","created_at":"...","response":"sky is ","done":false}
{"model":"...","created_at":"...","response":"blue.","done":false}
{"model":"...","created_at":"...","response":"","done":true,"done_reason":"stop", ...}
```

| Ollama request field | Behaviour |
| --- | --- |
| `model` | Required; an unknown model is answered with `404` and `{"error": "..."}` |
| `prompt` | The text to complete |
| `system` | Rendered with the chat template of the model, like `/api/chat` does; models without a template get the system message in front of the prompt |
| `options.temperature` / `top_p` / `top_k` | Applied (and sampling is enabled) |
| `options.num_predict` | Mapped to the number of generated tokens |
| `options.num_ctx` | Mapped to the maximum sequence length |
| `options.repeat_penalty` | Mapped to the repetition penalty |
| `options.stop` | The response is cut at the first stop sequence |
| `options.seed`, `mirostat*`, `presence_penalty`, `frequency_penalty` | Accepted, but not applied (no llama.cpp sampler) |
| `keep_alive` | Unloads the model after the given idle time (`"30s"`, `"5m"`, `"1h"`, `"0"`). Without it the model stays in memory, because reloading an AirLLM checkpoint is expensive. A service-wide default can be set with `AIRLLM_OLLAMA_KEEP_ALIVE` |
| `raw` | Sends the prompt unchanged, without the chat template (like Ollama). The system message is dropped |
| `format`, `template`, `context`, `suffix` | Accepted, but not applied |

The prompt is rendered with the **chat template of the model**
(`tokenizer.apply_chat_template`), like Ollama does it — the template is
applied to every prompt, not only to the ones with a system message. A model
without a template gets the prompt with the system message in front of it
(`"Be terse.\n\nWhy is the sky blue?"`).

Errors follow Ollama's format: `404` for an unknown model, `503` when the
model cannot be loaded and `400` for an invalid request body.

### Model management

Ollama manages a registry of models: it pulls them, pushes them, creates them
from a `Modelfile`, copies them and deletes them. This service serves a
single, pre-configured model — it has no registry.

The endpoints exist anyway, so that a client gets a clear answer instead of a
"not found":

```bash
curl http://127.0.0.1:8000/api/pull -d '{"model": "llama3.2"}'
```

```json
{
  "error": "This service serves a single, pre-configured model (AIRLLM_OLLAMA_MODEL_NAME); it cannot pull models."
}
```

The status code is `501 Not Implemented`. To serve another model, set
`AIRLLM_OLLAMA_MODEL_NAME` (and `AIRLLM_OLLAMA_MODEL_PATH` for a local
checkpoint).

### Chatting

```bash
curl http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "messages": [
          {"role": "system", "content": "You are terse."},
          {"role": "user", "content": "Why is the sky blue?"}
        ]
      }'
```

```json
{
  "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
  "created_at": "2026-10-23T12:00:19.123456Z",
  "message": {"role": "assistant", "content": "Rayleigh scattering."},
  "done": true,
  "done_reason": "stop",
  "total_duration": 812345678,
  "load_duration": 402345678,
  "prompt_eval_count": 12,
  "prompt_eval_duration": 1200000,
  "eval_count": 6,
  "eval_duration": 408945678
}
```

The history is rendered with the **chat template of the model**
(`tokenizer.apply_chat_template`). The template is the single source of truth
for the format: it renders the special tokens of the model itself (Llama 3
starts its chat with `<|begin_of_text|>`), so the rendered text is tokenized
without adding them a second time. Base models have no template — for them the
history is flattened into a plain transcript:

```
System: You are terse.

User: Why is the sky blue?

Assistant:
```

A template without support for the `system` role (Gemma, older Mistral
versions) refuses the history. It is then asked a second time, with the system
message merged into the first user turn (`"You are terse.\n\nWhy is the sky
blue?"`) — the model keeps its own format instead of getting a transcript.

`prompt_eval_count` counts the tokens of the rendered prompt. Since the prompt
is not tokenized twice, that count is one **lower** than it was before the
double BOS token was removed — the number is the correct one now.

`options`, `keep_alive` and `stream` behave exactly like in `/api/generate`.

### The OpenAI API

The `/v1/*` routes mirror the OpenAI API, so that the official client can be
pointed at this service:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
print(client.models.list())
```

`GET /v1/models` answers with the single model the service serves:

```bash
curl http://127.0.0.1:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "hf-internal-testing/tiny-random-LlamaForCausalLM",
      "object": "model",
      "created": 1792896781,
      "owned_by": "airllm"
    }
  ]
}
```

`GET /v1/models/{model_id}` answers with that model alone. A model id may
contain a slash (a Hugging Face repository id) and a tag (`name:latest`
resolves to the served model, like in Ollama). Another model is answered with
the OpenAI error format:

```json
{
  "error": {
    "message": "The model 'other-model' does not exist",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

`POST /v1/chat/completions` answers a chat. The messages are rendered with
the **chat template of the model**, the same way `/api/chat` does it:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "messages": [{"role": "user", "content": "Why is the sky blue?"}]
      }'
```

```json
{
  "id": "chatcmpl-airllm-d49967a00103",
  "object": "chat.completion",
  "created": 1792898625,
  "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Rayleigh scattering."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 15, "completion_tokens": 4, "total_tokens": 19}
}
```

With `"stream": true` the answer is a stream of Server-Sent Events, the way
the OpenAI API streams it — one `data:` event per chunk, the last one being
`data: [DONE]`.

Besides `model` and `messages`, the endpoint accepts `temperature`, `top_p`,
`max_tokens` (or its newer name `max_completion_tokens`), and `stop` (a string
or a list of strings). Unknown parameters are ignored, so a client that sends
the full option set keeps working.

`POST /v1/completions` is the legacy endpoint: it takes a prompt instead of a
chat history, and answers with a text completion.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "prompt": "Why is the sky blue?"
      }'
```

```json
{
  "id": "cmpl-airllm-6d9907bba9cd",
  "object": "text_completion",
  "created": 1792907101,
  "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
  "choices": [
    {"text": "Rayleigh scattering.", "index": 0, "logprobs": null, "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
}
```

The legacy API accepts a list of prompts as well — the answer then has one
choice per prompt, in the same order. With `"stream": true` the answer is a
stream of Server-Sent Events, the way the OpenAI API streams it.

### Verifying a running instance

```bash
# Check the endpoints of a running service
uv run python -m airllm_ollama --check
uv run python -m airllm_ollama --check --url http://127.0.0.1:8000 --timeout 2
```

```
OK   /              airllm-ollama is running
OK   /health        ok
OK   /api/version   version 0.1.0
OK   /api/tags      1 model(s)
4/4 checks passed
```

The check verifies the status *and* the payload of every endpoint, and exits
with `1` when something is wrong — handy as a container health probe.

### Running the tests

```bash
# Full suite with coverage (enforces >= 80% coverage)
uv run pytest -q

# Single file / single test
uv run pytest -q tests/test_api_health.py
uv run pytest -q tests/test_api_health.py::test_health_returns_ok

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
uv run pytest -q             # tests + coverage gate
```

### Project layout

```
src/airllm_ollama/
├── __init__.py        # package exports
├── __main__.py        # `python -m airllm_ollama` entry point
├── api/routes.py      # HTTP routes (/, /health, /api/*)
├── chat.py            # rendering of chat histories into a prompt
├── config.py          # environment-driven settings
├── engine.py          # AirLLM singleton wrapper (lazy load, thread-safe)
├── exceptions.py      # error hierarchy
├── main.py            # FastAPI application factory
├── model_metadata.py  # checkpoint metadata (size, digest, family, params)
├── options.py         # translation of Ollama's options
├── schemas/           # pydantic request/response models
├── streaming.py       # streamer turning the generated tokens into text
└── version.py         # package version
tests/                 # pytest suite (unit, integration and end-to-end)
.github/workflows/ci.yml
```

