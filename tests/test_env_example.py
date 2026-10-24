"""Guard against a drift between ``.env.example`` and the settings defaults."""

from __future__ import annotations

import pathlib

from airllm_ollama.config import Settings

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"


# The version comes from the package metadata and would have to be updated on
# every release, so it is not part of the example file.
UNDOCUMENTED = {"version"}


def _example_values() -> dict[str, tuple[str, bool]]:
    """Return the settings of the example file with their value and state.

    The state tells whether the setting is active or commented out.
    """
    values: dict[str, tuple[str, bool]] = {}
    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        commented = line.startswith("#")
        key, _, value = line.lstrip("# ").partition("=")
        name = key.removeprefix("AIRLLM_OLLAMA_").lower()
        values[name] = (value, not commented)
    return values


def _active_values() -> dict[str, str]:
    """Return the settings that the example file sets explicitly."""
    return {name: value for name, (value, active) in _example_values().items() if active}


def test_every_value_of_the_example_matches_the_default() -> None:
    """The example file must not drift away from the defaults."""
    defaults = {name: field.default for name, field in Settings.model_fields.items()}

    for name, value in _active_values().items():
        assert name in defaults, f"'{name}' is not a setting of the service"
        assert str(defaults[name]) == value, (
            f".env.example sets {name} to '{value}', the default is '{defaults[name]}'"
        )


def test_every_key_of_the_example_names_a_setting() -> None:
    """A renamed setting must not survive in the example file."""
    defaults = {name: field.default for name, field in Settings.model_fields.items()}

    for name in _example_values():
        assert name in defaults, f"'{name}' is not a setting of the service"


def test_every_setting_is_documented_in_the_example() -> None:
    """A new setting has to be documented in the example file."""
    documented = set(_example_values())
    defaults = {name: field.default for name, field in Settings.model_fields.items()}

    assert set(defaults) - documented - UNDOCUMENTED == set()
