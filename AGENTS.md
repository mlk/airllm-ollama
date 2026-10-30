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
* The entry point (`__main__.py`) takes the model options as **command line
  arguments** as well: `--model`, `--model-path`, `--device`, `--host`,
  `--port`. They win over the environment variables and over the `.env` file.
  They reach the settings through `config.apply_overrides()`, which writes them
  into the environment (the settings are read from there) and clears the cache —
  so it has to run before the settings are read for the first time.
* The version of the package is derived from the **git tags** by `hatch-vcs`:
  `pyproject.toml` has no `version`, and `src/airllm_ollama/_version.py` is
  written when the package is built (it is gitignored). A release is a tag —
  `git tag 0.2.0 && git push origin 0.2.0`; the README describes the rest.
  `uv lock` records the version as *dynamic*, so the lock file does not have to
  be updated for a new release.
* The `review` job (JetBrains Junie) uses the **free** model of the account — the
  workflow pins no `model`. When the job fails with *"Insufficient account
  balance"*, the account is out of credits; the quality gates still decide, the
  review is a second pair of eyes.

## Architecture

```
src/airllm_ollama/
├── __init__.py        # package exports
├── __main__.py        # `python -m airllm_ollama` entry point (uvicorn runner)
├── api/disconnect.py   # streaming of the events + abort of a generation
├── api/routes.py      # HTTP routes (api_router)
├── config.py          # Settings (env prefix AIRLLM_OLLAMA_) + get_settings()
├── decision.py        # the decision models (the questions and their answers)
├── download.py        # the `download-model` command (a checkpoint from the Hub)
├── engine.py          # AirLLMEngine: single-model, thread-safe, lazily loaded
├── exceptions.py      # AirLLMError / ModelMismatchError / ModelInitializationError / GenerationError
├── health_check.py    # verification of a running instance
├── main.py            # create_app() factory + module-level `app`
├── model_metadata.py  # checkpoint metadata read from the local checkpoint
├── options.py         # translation of Ollama's options + stop sequence filter
├── schemas/           # pydantic request/response models
├── stopping.py        # the criterion that ends a generation (stop sequences, abort)
├── streaming.py       # streamer turning the generated tokens into text
└── version.py         # __version__ (derived from the git tags by hatch-vcs)
```

### Endpoints (issue #3, done)

| Method | Path | Payload |
| --- | --- | --- |
| `GET` | `/` | Plain text — `airllm-ollama is running` / `airllm-ollama is kinda running` |
| `GET` | `/health` | `HealthResponse` — status, version, model, `model_loaded` |
| `GET` | `/api/version` | `VersionResponse` — `{"version": "..."}` |
| `GET` | `/api/tags` | `TagsResponse` — Ollama's `{"models": [...]}` shape |
| `POST` | `/api/show` | `ShowResponse` — the details of the model, as in Ollama |
| `GET` | `/api/ps` | `PsResponse` — the models that are loaded, as in Ollama |
| `POST` | `/api/generate` | `GenerateResponse` — Ollama's non-streaming generate shape |
| `POST` | `/api/chat` | `ChatResponse` — Ollama's non-streaming chat shape |
| `POST` | `/api/pull`, `/api/push`, `/api/create`, `/api/copy` | `501` — Ollama's model management, not implemented |
| `DELETE` | `/api/delete` | `501` — Ollama's model management, not implemented |
| `POST` | `/api/blobs/{digest}` | `501` — Ollama's blobs (the files of a model), not implemented |
| `HEAD` | `/api/blobs/{digest}` | `501` — the same, without a body |
| `POST` | `/api/embed`, `/api/embeddings` | `501` — Ollama's embeddings, not implemented |

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
* `POST /api/show` answers the information about the model: the `details` and
  the `capabilities` are the ones `/api/tags` reports, the `model_info` is the
  configuration of the checkpoint and the `template` its chat template (from
  `chat_template.jinja` or the `chat_template` of `tokenizer_config.json`). A
  checkpoint has no `Modelfile`, so the service synthesises one that describes
  the model it serves.
* `GET /api/ps` answers the models that are loaded — the model is loaded
  lazily, so the list is empty until the first request. The `expires_at` is the
  moment the model will be unloaded (`engine.unload_at`), the `size_vram` is `0`
  because AirLLM streams the model layer by layer.
* The `capabilities` of the served model are `["completion"]`, and they live in
  one place (`schemas/tags.py`, `CAPABILITIES`) — `/api/tags` and `/api/show`
  report the same list. The service generates text; the chat endpoint is a
  completion, Ollama has **no** `chat` capability. The other capabilities of
  Ollama (`tools`, `vision`, `embedding`, `decision`, …) come with the features
  that are not implemented yet — `tools` with the tool calls (#16),
  `embedding` with the embeddings (#64).
* The capabilities of the **served** model are configured (issue #81):
  `AIRLLM_OLLAMA_CAPABILITIES` (a comma separated list) or the argument
  `--capabilities`. The default is `completion`; a decision model answers
  `decision`. The service cannot tell the two apart — a decision model is an
  ordinary causal language model — so the operator says what the model is. The
  setting is read by the routes (`settings.capabilities`), the constant of the
  schemas is its default.
* The model management endpoints of Ollama (`/api/pull`, `/api/push`,
  `/api/create`, `/api/copy`, `/api/delete`) are exposed, but they always
  answer `501` with the Ollama error format: this service serves a single,
  pre-configured model and has no registry. A client gets a clear answer
  instead of a "not found".
* The blob endpoints of Ollama (`POST /api/blobs/{digest}` and `HEAD
  /api/blobs/{digest}`) answer `501` as well: the service stores no files, so
  a model cannot be created from them (issue #63). A `HEAD` answer has no
  body, so the status code is the whole answer there.
* The embedding endpoints of Ollama (`/api/embed` and the legacy
  `/api/embeddings`) answer `501` as well: the service generates text, it does
  not produce embeddings (issue #64). The `embedding` capability is therefore
  not reported either. The OpenAI route `POST /v1/embeddings` answers `501` in
  the OpenAI format.

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
* The MLX backend always loads the output head of the checkpoint, so it cannot
  load a model that **ties its embeddings** (every SmolLM2 model does).
  `model_metadata.has_own_output_head()` reads the header of the checkpoint
  and `default_model_factory` refuses such a checkpoint with a clear message
  (issue #77). The torch backends tie the embeddings themselves.
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
* `GenerationError` (issue #78) maps to **503** as well: a checkpoint that is
  broken in a way that only shows up while the model generates cannot be used
  either. The message names the checkpoint, so that the operator knows the
  files to look at.
* The `/v1/*` routes answer in the **OpenAI** format
  (`{"error": {"message": ..., "type": ...}}`) — `api/errors.py` picks the
  format by the path of the request (`error_body()`).
* A **stream** cannot change its status code once the headers are sent, so a
  failed generation ends the stream with an error event
  (`{"error": "..."}` for Ollama, `data: {"error": {...}}` for OpenAI) — the
  way Ollama reports it. The engine raises the error in the thread that reads
  the stream, so that the failure of the worker thread is not swallowed.
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
* `options.stop` ends the generation: the model stops at the next token after a
  stop sequence (see "Stopping a generation" below). The response is cut at the
  first stop sequence as well — that cut is the safety net for the backends
  that cannot stop early.
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
  sequence spanning two chunks is detected as well.
* A stop sequence does not end the stream: the client still gets the last event,
  with the statistics of the generation — the same numbers the non-streaming
  answer reports. Only the text after the stop sequence is dropped. The filter
  releases the tail it held back when the generation is over.
* `engine.prepare()` validates the model and loads it *before* the response
  headers are sent, so that a broken checkpoint becomes a 503 instead of a
  truncated stream.

### Stopping a generation (issues #15 and #24, done)

* `stopping.StopGeneration` is the criterion that ends a generation early. It is
  handed to the model as `stopping_criteria`, so **transformers** checks it
  between two generated tokens: the model stops at the next token.
* The criterion does two things: it ends the generation when a **stop sequence**
  appears in the generated tokens and when the **client is gone** (the `abort`
  event is set). Only the *generated* tokens are decoded — a stop sequence in
  the prompt does not end the generation.
* The tail of the answer is decoded (the longest stop sequence plus a margin),
  not the whole sequence: the criterion runs after every token.
* The `stop` sequences of the request are passed to the engine
  (`generate`, `generate_result`, `generate_chat`, `generate_stream`,
  `generate_chat_stream`); they are never forwarded to AirLLM. The post-hoc cut
  of `options.apply_stop_sequences()` stays as a safety net.
* A client that disconnects stops the generation: `api/disconnect.py` reads the
  events in a thread (`anyio.to_thread.run_sync(..., abandon_on_cancel=True)`)
  and sets the `abort` event when the request is cancelled or the stream is
  closed. The thread is abandoned, so the event loop stays free.
* The engine waits for the worker thread before it releases the generation lock
  (the model must not be used while it is still generating), so the lock is
  released as soon as the generation actually stopped — at the next token.
* **The MLX backend knows neither `stopping_criteria` nor `streamer`**: on
  macOS the stop sequences only cut the answer and a disconnect does not stop
  the generation. The torch backends (Linux, Windows) do both.

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

### OpenAI compatibility (issue #20, done)

* The `/v1/*` routes mirror the OpenAI API, so that the official client can be
  pointed at this service with `base_url="http://host:port/v1"`. They live in
  `api/openai.py` (router `openai_router`) and `schemas/openai.py`.
* `GET /v1/models` answers with the single served model, `GET
  /v1/models/{model_id}` with that model alone. The id of the model is the
  configured `model_name`; the `created` timestamp is the modification time of
  the checkpoint.
* The model id may contain a slash (a Hugging Face repository id), so the route
  uses the `:path` converter. A tag (`name:latest`) resolves to the served
  model, like in Ollama — `engine.model_names_match()` does the check.
* `POST /v1/chat/completions` (issue #22) answers a chat: the messages are
  rendered with the chat template of the model, the same way `/api/chat` does
  it. `temperature`, `max_tokens` and `stop` are translated with
  `options.map_options()`; unknown parameters are ignored, so a client that
  sends the full option set keeps working.
* `POST /v1/completions` (issue #23) is the legacy endpoint: it takes a prompt
  instead of a chat history and answers with a text completion. The prompt may
  be a list — the answer then has one choice per prompt, in the same order.
  The prompts are completed one after the other, the engine serialises the
  generations anyway.
* A streamed answer is a stream of Server-Sent Events (`text/event-stream`),
  one `data:` event per chunk, the last one `data: [DONE]`. The stream always
  ends with the last chunk (with the `finish_reason`), also when a stop
  sequence was found, and it releases the tail that the stop filter held back
  before it.
* The generation of the non-streaming answer runs in a thread
  (`starlette.concurrency.run_in_threadpool`): a generation takes minutes, and
  it must not block the event loop of the service.
* `POST /v1/systemone` (issue #68) answers the questions of a decision model:
  a *choice* between labelled options, a *noul* (yes or no) or a *score* on a
  scale. The model is asked **once** per question — a forward pass without the
  autoregressive loop (`engine.decide()`), and the probabilities of the
  candidates are read from the logits of the last position. The prompt is the
  one the model was trained with (`AIRLLM_OLLAMA_DECISION_TEMPLATE`). The
  path needs the torch backend: the MLX backend has no `forward()`.
* The errors of these routes are answered in the **OpenAI** format
  (`{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`), not
  in the Ollama one: `api.openai.OpenAIError` carries the payload and
  `api.openai.openai_error_handler` renders it. The exception handler is
  registered in `create_app()`. The validation errors follow the format of the
  route: `api.errors.validation_error_handler` answers the `/v1/*` routes in
  the OpenAI format and the `/api/*` routes in the Ollama one.

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
  (`uv run download-model --model <repo> --path <dir>`, the same as
  `snapshot_download(..., allow_patterns=[...])`) and point
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
