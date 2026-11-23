# Review Guidelines

These are the guidelines of the automated review of `airllm-ollama`, an
Ollama-compatible REST API backed by AirLLM layer-wise inference. Before
reviewing, read `AGENTS.md` to understand the project's architecture,
conventions, and quality gates.

The review is posted by the `Ollama Review` workflow
(`.github/workflows/ollama-review.yml`), which runs the
[`mlk/ollama-review-action`](https://github.com/mlk/ollama-review-action). The
action posts every finding as an **inline comment on the line it belongs to**
and the assessment of the pull request as a **summary comment** on the pull
request. The next run edits that summary in place and replaces the inline
comments, so a pull request never fills up with stale reviews.

Name the code element in the text of a finding (`engine._encode()`,
`tests/test_chat.py`), so that the reader can find it.

## Review Process

1. **Read the full diff** — examine every file changed, not just the summary.
2. **Read `AGENTS.md`** — understand the project's architecture, tooling, quality
   gates, and environment quirks before forming opinions.
3. **Check against project conventions** — verify the pull request follows the
   rules in `AGENTS.md` (ruff, mypy strict, pytest coverage ≥80%, no httpx,
   lazy airllm imports, etc.).
4. **Trace the code flow** — for new features, follow the data flow from route →
   engine → model. For fixes, verify the root cause is addressed.
5. **Evaluate test coverage** — check that new code has tests, edge cases are
   covered, and fakes/mocks are appropriate.

## What a Finding Is

Every finding is anchored to a line of the diff and answers three questions:
what is wrong, why it matters, what to do about it.

* **Title** — a short imperative title that names the code element
  (`engine.load()` does not release the lock on failure).
* **Body** — the concern with code-level detail and a concrete fix, with the
  replacement code when there is one. No filler, no praise.
* **Severity** — the most severe correct one:
  * `blocker` — will break production or lose data,
  * `critical` — security hole, data loss, crash on a realistic path,
  * `major` — a real bug or a serious design problem,
  * `minor` — a limited-impact bug or a maintainability trap,
  * `nit` — a small polish item that is still worth saying.

A finding has to point at something in the diff. A concern about code that the
diff does not show is a question, not a finding: raise it in the summary as
"Other observations" and say what would settle it.

**The diff does not show every change of a pull request.** A file that was
deleted, or that was moved to another path, does not appear in it — only the
files that are added or modified do. Never conclude from the diff alone that
something was not removed, renamed or moved: the pull request itself is the
only proof.

## The Summary

The summary is the verdict of the review. It has two parts:

* **A verdict** — *Ready to merge* when there is nothing to fix, *Fix before
  merging* when there is, with the findings listed explicitly. One sentence
  that says why, referring to the main contribution of the pull request.
* **The strengths** — three to ten numbered points with **bold titles** and a
  detailed explanation of *why* each one is good, referring to specific code
  patterns, design decisions, or test strategies.

The strengths are not decoration: they tell the author what to keep. Examples
of what to praise:

* Clean architecture decisions (separation of concerns, DRY refactors).
* Proper error handling and graceful degradation.
* Thread safety and concurrency patterns.
* Test quality (realistic fakes, comprehensive coverage, proper skip
  conditions).
* Schema fidelity to the Ollama API.
* Documentation quality (docstrings, README updates, AGENTS.md notes).
* Smart use of caching, lazy imports, or optional dependencies.

## What Makes a Good Review

* **Be specific** — reference file paths, function names, and line-level
  patterns.
* **Be proportional** — a small fix pull request gets a short review; a large
  feature pull request gets thorough analysis.
* **Distinguish priority** — clearly separate "fix before merge" from "future
  enhancement".
* **Acknowledge shortcuts** — the project uses intentional shortcuts (e.g.,
  non-streaming responses, post-hoc stop sequences). Note them without demanding
  they be fixed now.
* **Check test quality** — verify tests use appropriate fakes, cover edge
  cases, and don't just test the happy path.
* **Reference AGENTS.md** — when suggesting something, check if AGENTS.md
  already addresses it. If it does, note that the pull request follows
  convention. If not, suggest adding it.

## Things to Skip

* Don't demand changes that the project's issue tracker already accounts for as
  future work.
* Don't flag style issues that ruff/mypy would catch in CI.
* Don't suggest adding tests for trivial getters or simple pass-through
  functions.
* Don't require the features that are deliberately not implemented yet: the
  tool calls of the chat and the embeddings. The streaming of the routes, the
  chat template and the stop sequences **are** implemented — they are part of
  the contract of the service, so a regression in them is a finding.