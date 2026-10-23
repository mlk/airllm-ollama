"""Tests for the health verification of a running instance."""

from __future__ import annotations

import io
import urllib.error
from typing import Any

import pytest
from pytest_mock import MockerFixture

from airllm_ollama.__main__ import check, main
from airllm_ollama.health_check import (
    _summarise,
    _validate,
    check_service,
    format_results,
)


class _Response(io.BytesIO):
    """Minimal stand-in for the object returned by ``urlopen``."""

    def __init__(self, body: str, status: int = 200) -> None:
        """Create a response with the given body and status."""
        super().__init__(body.encode("utf-8"))
        self.status = status

    def __enter__(self) -> _Response:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Leave the context manager."""
        self.close()


def _urlopen_mock(mocker: MockerFixture, responses: dict[str, str]) -> Any:
    """Return a mocked ``urlopen`` answering with the given bodies.

    An unregistered URL raises a ``KeyError``, so a test cannot silently
    receive the response of another endpoint.
    """

    def urlopen(url: str, timeout: float = 5.0) -> _Response:
        """Answer with the body registered for the URL."""
        if url not in responses:
            raise KeyError(f"no response registered for {url}")
        return _Response(responses[url])

    return mocker.patch("urllib.request.urlopen", side_effect=urlopen)


ALL_PATHS = ("/", "/health", "/api/version", "/api/tags")


def _bodies(root: str, body: str, **overrides: str) -> dict[str, str]:
    """Return a body for every checked path; the overrides win per path."""
    bodies = dict.fromkeys(ALL_PATHS, body)
    bodies.update(overrides)
    return {f"{root}{path}": value for path, value in bodies.items()}


def test_check_service_reports_every_endpoint(mocker: MockerFixture) -> None:
    _urlopen_mock(
        mocker,
        {
            "http://localhost:11434/": "airllm-ollama is running",
            "http://localhost:11434/health": '{"status": "ok"}',
            "http://localhost:11434/api/version": '{"version": "0.1.0"}',
            "http://localhost:11434/api/tags": '{"models": [{"name": "test-model"}]}',
        },
    )

    results = check_service("http://localhost:11434/")

    assert [result.path for result in results] == ["/", "/health", "/api/version", "/api/tags"]
    assert all(result.ok for result in results)
    assert results[1].detail == "ok"
    assert results[2].detail == "version 0.1.0"
    assert results[3].detail == "1 model(s)"


def test_check_service_reports_a_failing_endpoint(mocker: MockerFixture) -> None:
    _urlopen_mock(mocker, _bodies("http://localhost:11434", "{}"))

    results = check_service("http://localhost:11434")

    assert all(not result.ok for result in results)


def test_the_mock_rejects_unregistered_urls(mocker: MockerFixture) -> None:
    _urlopen_mock(mocker, {})

    with pytest.raises(KeyError):
        check_service("http://localhost:11434")


def test_check_service_reports_an_http_error(mocker: MockerFixture) -> None:
    def urlopen(url: str, timeout: float = 5.0) -> _Response:
        """Answer with an HTTP error."""
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)  # type: ignore[arg-type]

    mocker.patch("urllib.request.urlopen", side_effect=urlopen)

    results = check_service("http://localhost:11434")

    assert all(not result.ok for result in results)
    assert results[0].detail == "HTTP 503"


def test_check_service_reports_a_connection_error(mocker: MockerFixture) -> None:
    def urlopen(url: str, timeout: float = 5.0) -> _Response:
        """Fail like an unreachable service does."""
        raise OSError("connection refused")

    mocker.patch("urllib.request.urlopen", side_effect=urlopen)

    results = check_service("http://localhost:11434")

    assert all(not result.ok for result in results)
    assert "connection refused" in results[0].detail


def test_check_service_reports_an_invalid_body(mocker: MockerFixture) -> None:
    _urlopen_mock(mocker, _bodies("http://localhost:11434", "{}", **{"/health": "not json"}))

    results = check_service("http://localhost:11434")

    assert results[1].ok is False
    assert results[1].detail == "invalid JSON"


def test_check_service_reports_a_missing_key(mocker: MockerFixture) -> None:
    _urlopen_mock(mocker, _bodies("http://localhost:11434", "{}", **{"/api/tags": '{"other": []}'}))

    results = check_service("http://localhost:11434")

    assert results[3].ok is False
    assert results[3].detail == "missing 'models'"


def test_check_service_reports_an_unexpected_root_response(mocker: MockerFixture) -> None:
    _urlopen_mock(mocker, _bodies("http://localhost:11434", "{}", **{"/": "hello"}))

    results = check_service("http://localhost:11434")

    assert results[0].ok is False
    assert results[0].detail == "unexpected response"


def test_the_summary_of_an_unknown_body_is_the_first_line(mocker: MockerFixture) -> None:
    _urlopen_mock(
        mocker,
        _bodies("http://localhost:11434", "{}", **{"/health": "first line\nsecond line"}),
    )

    results = check_service("http://localhost:11434")

    assert results[1].ok is False
    assert results[1].detail == "invalid JSON"


def test_the_summary_of_a_body_that_is_not_json() -> None:
    assert _summarise("first line\nsecond line") == "first line"


def test_the_summary_of_an_empty_body() -> None:
    assert _summarise("") == "(empty body)"


def test_the_summary_of_a_json_object_without_known_keys() -> None:
    assert _summarise('{"other": 1}') == '{"other": 1}'


def test_the_summary_of_a_json_array(mocker: MockerFixture) -> None:
    _urlopen_mock(mocker, _bodies("http://localhost:11434", "{}", **{"/health": "[1, 2]"}))

    results = check_service("http://localhost:11434")

    assert results[1].ok is False
    assert results[1].detail == "missing 'status'"


def test_a_response_with_an_error_status_fails_the_check(mocker: MockerFixture) -> None:
    def urlopen(url: str, timeout: float = 5.0) -> _Response:
        """Answer with an error status."""
        return _Response("{}", status=500)

    mocker.patch("urllib.request.urlopen", side_effect=urlopen)

    results = check_service("http://localhost:11434")

    assert all(not result.ok for result in results)
    assert results[0].detail == "HTTP 500"


def test_an_unknown_path_is_only_checked_for_the_status() -> None:
    ok, detail = _validate("/unknown", "whatever", 200)

    assert ok is True
    assert detail == "whatever"


def test_format_results_summarises_the_outcome() -> None:
    report = format_results(
        [
            _result("/health", True, "ok"),
            _result("/api/tags", False, "HTTP 500"),
        ]
    )

    assert "OK   /health" in report
    assert "FAIL /api/tags" in report
    assert "1/2 checks passed" in report


def _result(path: str, ok: bool, detail: str) -> Any:
    """Build a check result."""
    from airllm_ollama.health_check import CheckResult

    return CheckResult(path=path, ok=ok, detail=detail)


def test_check_returns_zero_when_everything_passes(mocker: MockerFixture) -> None:
    _urlopen_mock(
        mocker,
        {
            "http://localhost:11434/": "airllm-ollama is running",
            "http://localhost:11434/health": '{"status": "ok"}',
            "http://localhost:11434/api/version": '{"version": "0.1.0"}',
            "http://localhost:11434/api/tags": '{"models": []}',
        },
    )

    assert check("http://localhost:11434") == 0


def test_check_returns_one_when_a_check_fails(mocker: MockerFixture) -> None:
    def urlopen(url: str, timeout: float = 5.0) -> _Response:
        """Fail like an unreachable service does."""
        raise OSError("connection refused")

    mocker.patch("urllib.request.urlopen", side_effect=urlopen)

    assert check("http://localhost:11434") == 1


def test_main_serves_by_default(mocker: MockerFixture) -> None:
    serve = mocker.patch("airllm_ollama.__main__.serve")

    assert main([]) == 0
    serve.assert_called_once_with()


def test_main_checks_an_instance(mocker: MockerFixture) -> None:
    mocker.patch("airllm_ollama.__main__.check", return_value=0)

    assert main(["--check", "--url", "http://example:1234"]) == 0


def test_main_forwards_the_timeout(mocker: MockerFixture) -> None:
    check = mocker.patch("airllm_ollama.__main__.check", return_value=0)

    main(["--check", "--url", "http://example:1234", "--timeout", "1.5"])

    check.assert_called_once_with("http://example:1234", 1.5)


def test_check_passes_the_timeout_to_the_service(mocker: MockerFixture) -> None:
    service = mocker.patch("airllm_ollama.__main__.check_service", return_value=[])

    check("http://example:1234", 2.5)

    service.assert_called_once_with("http://example:1234", timeout=2.5)
