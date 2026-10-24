"""Health verification for a running airllm-ollama instance.

The checks only use the standard library, so they can run against a service
without installing anything.

Only read-only endpoints are checked: a health probe must not trigger an
inference. New read-only endpoints belong into :data:`CHECK_PATHS`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

CHECK_PATHS = ("/", "/health", "/api/version", "/api/tags")
EXPECTED_KEYS = {"/health": "status", "/api/version": "version", "/api/tags": "models"}
ROOT_MARKER = "airllm-ollama is"
DEFAULT_TIMEOUT = 5.0


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check."""

    path: str
    ok: bool
    detail: str


def check_service(base_url: str, *, timeout: float = DEFAULT_TIMEOUT) -> list[CheckResult]:
    """Query the endpoints of a running service and report their state.

    Args:
        base_url: Base URL of the service, e.g. ``http://127.0.0.1:8000``.
        timeout: Timeout of a single request in seconds.

    Returns:
        One result per checked endpoint.
    """
    root = base_url.rstrip("/")
    return [_check(f"{root}{path}", path, timeout) for path in CHECK_PATHS]


def _check(url: str, path: str, timeout: float) -> CheckResult:
    """Perform a single check and describe the outcome."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            ok, detail = _validate(path, body, response.status)
            return CheckResult(path=path, ok=ok, detail=detail)
    except urllib.error.HTTPError as exc:
        return CheckResult(path=path, ok=False, detail=f"HTTP {exc.code}")
    except OSError as exc:
        return CheckResult(path=path, ok=False, detail=str(exc))


def _validate(path: str, body: str, status: int) -> tuple[bool, str]:
    """Check that the response of an endpoint looks the way it should."""
    if status != 200:
        return False, f"HTTP {status}"
    if path == "/":
        if ROOT_MARKER in body:
            return True, body.strip()
        return False, "unexpected response"
    key = EXPECTED_KEYS.get(path)
    if key is None:
        return True, body.strip()
    try:
        parsed = json.loads(body)
    except ValueError:
        return False, "invalid JSON"
    if not isinstance(parsed, dict) or key not in parsed:
        return False, f"missing '{key}'"
    return True, _summarise(body)


def _summarise(body: str) -> str:
    """Return a short, single line description of a response body."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return body.strip().splitlines()[0] if body.strip() else "(empty body)"
    if isinstance(parsed, dict):
        if "status" in parsed:
            return str(parsed["status"])
        if "version" in parsed:
            return f"version {parsed['version']}"
        if "models" in parsed:
            models = parsed["models"]
            count = len(models) if isinstance(models, list) else 0
            return f"{count} model(s)"
    return body.strip().splitlines()[0] if body.strip() else "(empty body)"


def format_results(results: list[CheckResult]) -> str:
    """Render the results of a check as a human readable report."""
    width = max((len(result.path) for result in results), default=0)
    lines = [
        f"{'OK  ' if result.ok else 'FAIL'} {result.path:<{width}}  {result.detail}"
        for result in results
    ]
    failed = sum(1 for result in results if not result.ok)
    lines.append(f"{len(results) - failed}/{len(results)} checks passed")
    return "\n".join(lines)
