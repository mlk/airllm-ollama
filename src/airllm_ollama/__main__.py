"""Console entry point: ``python -m airllm_ollama``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from airllm_ollama.config import get_settings
from airllm_ollama.health_check import check_service, format_results


def serve() -> None:
    """Run the ASGI server with the configured host/port."""
    settings = get_settings()
    uvicorn.run(
        "airllm_ollama.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


def check(url: str | None) -> int:
    """Verify a running instance and report the result.

    Args:
        url: Base URL of the instance; defaults to the configured host/port.

    Returns:
        ``0`` when every check passed, ``1`` otherwise.
    """
    settings = get_settings()
    base_url = url or f"http://{settings.host}:{settings.port}"
    results = check_service(base_url)
    print(format_results(results))
    return 0 if all(result.ok for result in results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command line and either serve or check the service.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The exit code of the requested action.
    """
    parser = argparse.ArgumentParser(
        prog="airllm-ollama",
        description="Ollama-compatible API wrapper backed by AirLLM.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify a running instance and exit instead of serving",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="base URL of the instance to check (default: the configured host/port)",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check(args.url)
    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
