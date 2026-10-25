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
4. CI must be green before the PR is considered done (`gh pr checks <pr>
   --watch`).
5. Do not reformat or restructure unrelated code — keep diffs focused on the ticket.
6. **Answer the automated review.** Every PR gets a `review` job (JetBrains
   Junie, see `.junie/guidelines.md`). After opening the PR, wait for it to
   finish (`gh pr checks <pr> --watch`), read the comment
   (`gh pr view <pr> --json comments`) and deal with **every** suggestion:

   * implement the suggestion (commit on the same branch, push again),
   * ignore it — then say why in a reply to the comment,
   * or, when it is valid but out of scope for this ticket, file a new issue
     (`gh issue create`) or extend an existing one and reference that issue in
     the reply.

   A PR is not done before the reviewer has been answered. Do not leave a
   suggestion unaddressed and unmentioned. The job posts **one** comment per
   pull request and updates it on the next run.

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
| Tests (with coverage gate) | `uv run pytest -q` |

Never edit `uv.lock` by hand. Always commit `pyproject.toml` **and** `uv.lock`
together.

## Quality gates (all enforced in CI)

* **ruff** — line length 100, `src` layout, rule set `E,W,F,I,N,UP,B,C4,SIM,ANN,D,RUF,PT`
  with pydocstyle `google` convention. Every module, class and function needs a
  docstring; every function needs full type annotations.
* **mypy** — `strict = true`, `python_version = "3.11"`. No `Any` leaks, no missing
  annotations, no untyped decorators.
* **pytest** — coverage is configured once in `pyproject.toml`: `--cov=airllm_ollama
  --cov-branch` plus `fail_under = 80` under `[tool.coverage.report]`. Run
  `uv run pytest -q` everywhere (CI included) — never re-declare coverage flags on
  the command line, they would override the single source of truth. The `-q` is
  part of the command: the entry point of the package parses `sys.argv` when it
  is called without arguments, so a test that does that would parse the
  arguments of the *pytest* process and fail as soon as pytest is started with
  an argument. The 80 % coverage gate is enforced locally *and* in CI; keep new
  code covered by tests.
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
* The `review` job (JetBrains Junie) uses the **free** model of the account — the
  workflow pins no `model`. When the job fails with *"Insufficient account
  balance"*, the account is out of credits; the quality gates still decide, the
  review is a second pair of eyes.

## Architecture

```
src/airllm_ollama/
├── __init__.py        # package exports
├── __main__.py        # `python -m airllm_ollama` entry point (uvicorn runner)
├── api/routes.py      # HTTP routes (api_router)
├── config.py          # Settings (env prefix AIRLLM_OLLAMA_) + get_settings()
├── engine.py          # AirLLMEngine: single-model, thread-safe, lazily loaded
├── exceptions.py      # AirLLMError / ModelMismatchError / ModelInitializationError
├── health_check.py    # verification of a running instance
├── main.py            # create_app() factory + module-level `app`
├── model_metadata.py  # checkpoint metadata read from the local checkpoint
├── options.py         # translation of Ollama's options + stop sequence filter
├── schemas/           # pydantic request/response models
├── streaming.py       # streamer turning the generated tokens into text
└── version.py         # __version__ (must match pyproject.toml version)
```

### Endpoints (issue #3, done)

| Method | Path | Payload |
| --- | --- | --- |
| `GET` | `/` | Plain text — `airllm-ollama is running` / `airllm-ollama is kinda running` |
| `GET` | `/health` | `HealthResponse` — status, version, model, `model_loaded` |
| `GET` | `/api/version` | `VersionResponse` — `{"version": "..."}` |
| `GET` | `/api/tags` | `TagsResponse` — Ollama's `{"models": [...]}` shape |
| `POST` | `/api/generate` | `GenerateResponse` — Ollama's non-streaming generate shape |
| `POST` | `/api/chat` | `ChatResponse` — Ollama's non-streaming chat shape |

* The response models mirror Ollama 0.40.x (`/api/tags` was verified against a
  real Ollama instance): `name`, `model`, `modified_at`, `size`, `digest`,
  `details` (`parent_model`, `format`, `family`, `families`, `parameter_size`,
  `quantization_level`, `context_length`) and `capabilities`.
* `model_metadata.collect_metadata()` derives the metadata from the **local**
  checkpoint: file sizes, `sha256` digest of the weights, `model_type` from
  `config.json` and the parameter count from the safetensors header. It is
  cached and never downloads anything — a model that was not fetched yet is
  listed with empty metadata.
* The metadata cache is keyed by the model path and holds the four most recent
  entries; it is never invalidated automatically. Call
  `clear_metadata_cache()` after the checkpoint on disk changed.
* `capabilities` is currently hardcoded to `["completion"]` — extend it when
  the chat/embedding endpoints land (issues #4/#5).

### Engine (issue #2, done)

* `AirLLMEngine` owns exactly one AirLLM model, loads it lazily on first use and
  serialises every model interaction with an `RLock` (AirLLM streams layer
  shards through shared hooks — concurrent generations corrupt that state).
* `get_engine()` / `reset_engine()` provide the process-wide singleton.
* `generate(prompt, **kwargs)` returns text; the reserved `model` keyword is
  validated against the configured model and raises `ModelMismatchError`
  (Ollama's `:latest` tag is normalised away before comparing).
* `airllm` is imported lazily inside `default_model_factory`; the factory is
  injectable so tests never import the runtime.
* On macOS AirLLM uses Apple's **MLX backend**, which needs `mlx` — a package
  AirLLM imports but does not depend on. The `airllm` extra installs it
  (`mlx>=0.21; sys_platform == 'darwin'`).
* The MLX backend works on MLX arrays instead of torch tensors and needs a
  `max_new_tokens`; `engine.is_mlx_backend()` detects it (by its name) and the
  engine converts the token ids (`mx.array(...)`) and sets the limit. Without
  a `num_predict` in the request, the limit is `AIRLLM_OLLAMA_MAX_SEQ_LEN`
  (512) — the transformers backend has no such default.
* Model references follow Ollama's `name:tag` convention. `model_names_match()`
  accepts any tag when the served model has none, but demands an exact tag
  when the served model declares one (`llama3.2:1b` ≠ `llama3.2:70b`).
* The engine uses two locks: `_generation_lock` covers the *whole* inference
  call (AirLLM's streaming hooks mutate shared model state), `_load_lock` only
  guards model creation so `load()` never blocks behind a running inference.
* `engine.is_available` reports whether the optional AirLLM runtime is
  installed. It uses `importlib.util.find_spec` — the package is never
  imported, because importing `airllm` pulls in torch and takes seconds.
  `GET /` uses it to answer `airllm-ollama is running` (runtime present) or
  `airllm-ollama is kinda running` (runtime missing, inference would fail).

## API layer notes (for issues #3–#5)

* Map `ModelMismatchError` to **HTTP 404** (Ollama's behaviour for an unknown
  model), not 500. `ModelInitializationError` maps to 503. A malformed request
  body is answered with **400** (not FastAPI's default 422) so the error shape
  stays Ollama compatible. All three are implemented in `api/errors.py` and
  registered in `create_app()`.
* Ollama's error body is `{"error": "..."}` — use it for every error.
* `get_engine()` caches the engine for the process lifetime; settings are read
  once at startup. Call `reset_engine()` when the settings change (tests,
  hot reloads) — the singleton is *not* invalidated automatically.

### Generation (issue #4, done)

* `POST /api/generate` mirrors Ollama's non-streaming response
  (`response`, `done`, `done_reason`, `total_duration`, `load_duration`,
  `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration`;
  the field names were taken from the Ollama binary). `stream` is accepted but
  always answered non-streaming.
* `options.py` translates Ollama's options into transformers keywords
  (`num_predict` → `max_new_tokens`, `num_ctx` → `max_length`,
  `repeat_penalty` → `repetition_penalty`, sampling parameters imply
  `do_sample=True`). llama.cpp-only options (seed, Mirostat, presence and
  frequency penalties) are accepted and ignored — Ollama clients always send
  the full option set, so rejecting them would break the clients.
* `options.stop` is applied by cutting the response at the first stop sequence.
  This happens *after* the generation, not token by token — Ollama stops the
  model early. Fine for the non-streaming shortcut, revisit with streaming.
* The prompt is rendered with the chat template of the model (issue #31): the
  engine renders the system message and the prompt as a chat
  (`chat.render_completion()`), the same way `/api/chat` does it. The template
  is applied to **every** prompt, not only to the ones with a system message —
  Ollama does it that way. Models without a template get the system message
  joined in front of the prompt (`"Be terse.\n\nWhy is the sky blue?"`), the
  way Ollama does it.
  The rendering happens under the generation lock, after the model was
  validated and loaded — a wrong model never loads the checkpoint.
* A request with `raw: true` is sent unchanged, without the template, and the
  system message is dropped.
* A raw prompt is tokenized with the special tokens, so a client that rendered
  the template itself (and sends the BOS token with it) gets a second one.
  Tracked in #44 — do not work around it with a heuristic.
* `keep_alive` unloads the model after the given idle time. Without it the model
  stays in memory, because reloading an AirLLM checkpoint is expensive. The
  default can be set with `AIRLLM_OLLAMA_KEEP_ALIVE` (e.g. `5m`).
* `done_reason` is `stop`, or `length` when the generation reached the
  requested `num_predict` limit.
* `engine.generate_result()` returns the text plus token counts and timings;
  `engine.generate()` is the thin wrapper that only returns the text.
* The engine counts the generated tokens by subtracting the prompt length from
  the returned sequence; backends that decode themselves are re-tokenized.

### Streaming (issue #13, done)

* `stream: true` on `/api/generate` and `/api/chat` returns a stream of JSON
  objects, one per line (`application/x-ndjson`) — the format Ollama uses and
  the one the official clients parse. `AIRLLM_OLLAMA_STREAM_FORMAT=sse` switches
  to Server-Sent Events.
* `engine.generate_stream()` / `engine.generate_chat_stream()` yield
  `StreamChunk`s. The generation runs in a worker thread, because the model
  only reports the produced tokens from the thread it runs in; the engine waits
  for that thread before it releases the generation lock, so the model is never
  used twice at the same time.
* `streaming.TextStreamer` turns the tokens of the generation loop into text.
  It handles both transformers variants (new tokens or the whole sequence).
* The stop sequences are applied *while* streaming: `options.StopSequenceFilter`
  holds back the tail that could still become a stop sequence, so a stop
  sequence spanning two chunks is detected as well. The generation itself is
  not stopped early yet (issue #15).
* `engine.prepare()` validates the model and loads it *before* the response
  headers are sent, so that a broken checkpoint becomes a 503 instead of a
  truncated stream.

### Chat (issue #5, done)

* `POST /api/chat` takes Ollama's `messages` array and answers with a
  `message` object (`role` + `content`) next to the same timings as
  `/api/generate`.
* `chat.render_chat()` renders the history with the chat template of the model
  (`tokenizer.apply_chat_template`); models without a template get the plain
  transcript of `chat.render_transcript()` (`User: ...\n\nAssistant:`).
* `chat.apply_chat_template()` is the primitive behind both renderings: it
  returns the rendered prompt or `None` when the model has no template or the
  template refuses the messages. `render_chat()` falls back to the transcript,
  `render_completion()` to the plain prompt.
* A template that refuses the messages is asked a second time, with the system
  message merged into the first user turn (`"Be terse.\n\nHello"`). Templates
  without support for the system role (Gemma, older Mistral versions) raise,
  and the model would lose its own format without the retry. The history after
  the first user turn is kept.
* The rendering lives in `chat.py` and not in the route, so it can be unit
  tested without a model.
* `chat.render_chat()` and `chat.render_completion()` return a
  `chat.RenderedPrompt`: the text **and** the way it has to be tokenized. A
  chat template renders the special tokens of the model itself (Llama 3 starts
  with `<|begin_of_text|>`), so a templated prompt is tokenized with
  `add_special_tokens=False` — the template stays the single source of truth
  and the model does not see a second BOS token. Every other prompt (a raw
  prompt, the transcript fallback) is tokenized the way the tokenizer does it
  by default, so a base model still gets its BOS token. This is the same
  behaviour as `tokenizer.apply_chat_template(..., tokenize=True)`.
* A template that does not render `{{ bos_token }}` therefore leaves the model
  without a BOS token — the same as in transformers. Do not work around that
  with a heuristic; the template is the single source of truth.
* `engine.generate_chat()` renders and generates in one go, under the same
  locks as `generate()`.
* The `model` argument of `generate()`, `generate_result()` and
  `generate_chat()` is an explicit keyword-only parameter: it is validated and
  never forwarded to AirLLM. The `raw` argument of `generate()`,
  `generate_result()` and `generate_stream()` is keyword-only as well: it asks
  for the prompt without the chat template and is never forwarded to AirLLM.
  `ChatMessage.role` is free-form (Ollama is permissive too); tool calls are
  not supported yet, so a `tool` role is rendered like any other role.

## Data usage policy (important)

The maintainer is on a **metered connection** — never download large artifacts.

* The default model is the 4.1 MB `hf-internal-testing/tiny-random-LlamaForCausalLM`
  (`DEFAULT_MODEL_NAME` in `src/airllm_ollama/config.py`).
* The service listens on port **8000** (uvicorn's default). Ollama's port
  11434 is available via `AIRLLM_OLLAMA_PORT`, but a local Ollama occupies it.
* Do **not** run `uv sync --extra airllm` or fetch model weights unless the
  maintainer explicitly asks: torch alone is ~130 MB, and the AirLLM bootstrap
  downloads every non-weight file of a repo (1.9 GB of ONNX exports for
  `SmolLM2-135M-Instruct`).
* Tests must keep mocking the engine — the suite may never touch the network.
* When a real model is needed, pre-fetch a filtered snapshot
  (`snapshot_download(..., allow_patterns=[...])`) and point
  `AIRLLM_OLLAMA_MODEL_PATH` at the local directory. See the README section
  "Downloading only what you need".

## Testing conventions

* Tests live in `tests/`, mirroring the module path (`tests/test_api_health.py`,
  `tests/test_config.py`, `tests/test_main.py`).
* `tests/test_e2e.py` is the integration suite: it drives the whole application
  (routing → validation → engine → tokenizer → decoding) and replaces only the
  AirLLM runtime with the doubles from `tests/fakes.py`. `tests/fakes.py` holds
  the shared doubles — import them from there instead of copying them.
* `python -m airllm_ollama --check` verifies a running instance; the logic
  lives in `health_check.py` and only uses the standard library.
* Use `pytest-mock`'s `mocker` fixture for patching third-party calls (e.g. AirLLM)
  so the suite stays fast and offline.
* Shared fixtures live in `tests/conftest.py` (`app`, `client`, `settings`).
* A test that calls the entry point of the package has to pass the arguments
  explicitly — `main([])`, never `main()`. Without arguments the entry point
  parses `sys.argv`, which inside the suite are the arguments of the *pytest*
  process.
* A new setting has to be documented in `.env.example` **and** in the settings
  table of the README — `tests/test_settings_docs.py` fails otherwise. The
  table of the README shows the default in backticks, an empty one as
  `_(empty)_`. A commented out line of the example file that shows a value
  other than the default ends with `# alternative`, so that the guard can tell
  it apart from a default.
* Cover the happy path **and** the error paths (validation errors, model mismatch,
  upstream failures) — the acceptance criteria ask for ≥ 80 % coverage.
