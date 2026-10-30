"""Console entry point: ``python -m airllm_ollama``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from airllm_ollama.config import apply_overrides, get_settings
from airllm_ollama.health_check import DEFAULT_TIMEOUT, check_service, format_results


def serve() -> None:
    """Run the ASGI server with the configured host/port."""
    settings = get_settings()
    uvicorn.run(
        "airllm_ollama.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


def check(url: str | None, timeout: float = DEFAULT_TIMEOUT) -> int:
    """Verify a running instance and report the result.

    Args:
        url: Base URL of the instance; defaults to the configured host/port.
        timeout: Timeout of a single request in seconds.

    Returns:
        ``0`` when every check passed, ``1`` otherwise.
    """
    settings = get_settings()
    base_url = url or f"http://{settings.host}:{settings.port}"
    results = check_service(base_url, timeout=timeout)
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
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="timeout of a single request in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="name of the model the service serves (default: the configured model)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="path of the local checkpoint (default: the configured one)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device handed to AirLLM, e.g. cuda:0 (default: the configured one)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="host the service listens on (default: the configured one)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="port the service listens on (default: the configured one)",
    )
    args = parser.parse_args(argv)

    if args.check:
        return check(args.url, args.timeout)
    # The arguments win over the environment variables and over the `.env` file.
    apply_overrides(
        {
            "model_name": args.model,
            "model_path": args.model_path,
            "device": args.device,
            "host": args.host,
            "port": args.port,
        }
    )
    serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
