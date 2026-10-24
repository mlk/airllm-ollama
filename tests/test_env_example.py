"""Guard against a drift between ``.env.example`` and the settings defaults."""

from __future__ import annotations

import pathlib
import re

from airllm_ollama.config import Settings

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"

# The version comes from the package metadata and would have to be updated on
# every release, so it is not part of the example file.
UNDOCUMENTED = {"version"}

SETTING_LINE = re.compile(r"^#?\s*(?P<key>AIRLLM_OLLAMA_[A-Z_]+)=(?P<value>.*)$")


def _defaults() -> dict[str, object]:
    """Return the default of every setting."""
    return {name: field.default for name, field in Settings.model_fields.items()}


def _example_lines() -> list[tuple[str, str, bool]]:
    """Return the settings of the example file with their value and their state.

    The state tells whether the setting is active or commented out.
    """
    lines: list[tuple[str, str, bool]] = []
    for raw_line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = SETTING_LINE.match(line)
        if match is None:
            continue
        name = match.group("key").removeprefix("AIRLLM_OLLAMA_").lower()
        lines.append((name, match.group("value").strip(), not line.startswith("#")))
    return lines


def _active_values() -> dict[str, str]:
    """Return the settings the example file sets explicitly.

    An active line wins over a commented out alternative of the same setting.
    """
    return {name: value for name, value, active in _example_lines() if active}


def _documented_keys() -> set[str]:
    """Return every setting the example file mentions."""
    return {name for name, _, _ in _example_lines()}


def test_every_value_of_the_example_matches_the_default() -> None:
    """The example file must not drift away from the defaults."""
    defaults = _defaults()

    for name, value in _active_values().items():
        assert name in defaults, f"'{name}' is not a setting of the service"
        assert str(defaults[name]) == value, (
            f".env.example sets {name} to '{value}', the default is '{defaults[name]}'"
        )


def test_every_key_of_the_example_names_a_setting() -> None:
    """A renamed setting must not survive in the example file."""
    defaults = _defaults()

    for name in _documented_keys():
        assert name in defaults, f"'{name}' is not a setting of the service"


def test_every_setting_is_documented_in_the_example() -> None:
    """A new setting has to be documented in the example file."""
    undocumented = set(_defaults()) - _documented_keys() - UNDOCUMENTED

    assert undocumented == set()


def test_an_active_line_wins_over_a_commented_alternative() -> None:
    """A commented alternative must not hide the value of the setting."""
    assert _active_values()["model_name"] == Settings.model_fields["model_name"].default
