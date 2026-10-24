"""Guard against a drift between ``.env.example`` and the settings defaults."""

from __future__ import annotations

import pathlib

from airllm_ollama.config import Settings

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"


def _active_values() -> dict[str, str]:
    """Return the settings that the example file sets explicitly."""
    values: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.removeprefix("AIRLLM_OLLAMA_").lower()] = value
    return values


def test_every_value_of_the_example_matches_the_default() -> None:
    """The example file must not drift away from the defaults."""
    defaults = {name: field.default for name, field in Settings.model_fields.items()}

    for name, value in _active_values().items():
        assert name in defaults, f"'{name}' is not a setting of the service"
        assert str(defaults[name]) == value, (
            f".env.example sets {name} to '{value}', the default is '{defaults[name]}'"
        )


def test_the_example_mentions_the_settings_it_documents() -> None:
    """The example file has to exist and to use the environment prefix."""
    content = ENV_EXAMPLE.read_text(encoding="utf-8")

    assert content.startswith("# Copy to .env")
    assert "AIRLLM_OLLAMA_" in content
