"""Console entry point: ``python -m airllm_ollama``."""

import uvicorn

from airllm_ollama.config import get_settings


def main() -> None:
    """Run the ASGI server with the configured host/port."""
    settings = get_settings()
    uvicorn.run(
        "airllm_ollama.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
