# Code Review Guidelines

You are reviewing a pull request for `airllm-ollama`, an Ollama-compatible REST API backed by AirLLM layer-wise inference. Before reviewing, read `AGENTS.md` to understand the project's architecture, conventions, and quality gates.

## Review Process

1. **Read the full diff** — examine every file changed, not just the summary.
2. **Read `AGENTS.md`** — understand the project's architecture, tooling, quality gates, and environment quirks before forming opinions.
3. **Check against project conventions** — verify the PR follows the rules in `AGENTS.md` (ruff, mypy strict, pytest coverage ≥80%, no httpx, lazy airllm imports, etc.).
4. **Trace the code flow** — for new features, follow the data flow from route → engine → model. For fixes, verify the root cause is addressed.
5. **Evaluate test coverage** — check that new code has tests, edge cases are covered, and fakes/mocks are appropriate.

## Review Format

Structure your review as a PR comment with these sections:

### Review

Start with one sentence summarizing what the PR does and your overall impression.

### Strengths

List 3–10 numbered strengths with **bold titles** and detailed explanations. For each strength, explain _why_ it's good, referencing specific code patterns, design decisions, or test strategies. Be specific — name functions, classes, or patterns.

Examples of what to praise:
- Clean architecture decisions (separation of concerns, DRY refactors)
- Proper error handling and graceful degradation
- Thread safety and concurrency patterns
- Test quality (realistic fakes, comprehensive coverage, proper skip conditions)
- Schema fidelity to Ollama API
- Documentation quality (docstrings, README updates, AGENTS.md notes)
- Smart use of caching, lazy imports, or optional dependencies

### Suggestions (not blockers)

List specific, actionable suggestions. For each:
- Give a **bold title** that names the code element (function, file, pattern).
- Explain the concern with code-level detail.
- Suggest a concrete fix or alternative.
- Clearly state why it's not a blocker (e.g., "the current approach is pragmatic," "this is a future concern," "the test covers the expected path").

Focus on:
- Future-proofing (heuristics that might break, hardcoded values)
- Edge cases (None handling, error paths, boundary conditions)
- Documentation gaps (behavioral differences, conventions not noted)
- Performance considerations (lock scopes, import patterns, cache behavior)
- API consistency (matching Ollama behavior, schema fidelity)

### Verdict

End with a clear verdict:
- **Ready to merge** if no blockers.
- **Fix before merging** if there are issues that should be addressed. List them explicitly.

Include one sentence summarizing why, referencing the PR's main contribution.

## What Makes a Good Review

- **Be specific** — reference file paths, function names, and line-level patterns.
- **Be proportional** — a small fix PR gets a short review; a large feature PR gets thorough analysis.
- **Distinguish priority** — clearly separate "fix before merge" from "future enhancement."
- **Acknowledge shortcuts** — the project uses intentional shortcuts (e.g., non-streaming responses, post-hoc stop sequences). Note them without demanding they be fixed now.
- **Check test quality** — verify tests use appropriate fakes, cover edge cases, and don't just test the happy path.
- **Reference AGENTS.md** — when suggesting something, check if AGENTS.md already addresses it. If it does, note that the PR follows convention. If not, suggest adding it.

## Things to Skip

- Don't demand changes the project's issue tracker already accounts for as future work.
- Don't flag style issues that ruff/mypy would catch in CI.
- Don't suggest adding tests for trivial getters or simple pass-through functions.
- Don't require streaming support, tool calls, or chat template perfection — these are tracked as separate issues.
