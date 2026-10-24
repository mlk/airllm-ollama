"""Guard against a drift between ``.env.example`` and the settings defaults."""

from __future__ import annotations

import io
import os
import pathlib
import re
from unittest import mock

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from airllm_ollama.config import Settings

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"

# pydantic-settings reads the environment regardless of the case, so the guard
# has to do the same.
PREFIX = "airllm_ollama_"

# A commented out line that shows a value other than the default has to be
# marked as an alternative, so that the guard can tell the two apart.
ALTERNATIVE = "# alternative"

# The version comes from the package metadata and would have to be updated on
# every release, so it is not part of the example file.
UNDOCUMENTED = {"version"}

# Assignments, active or commented out. Spaces around the = are allowed, like
# dotenv allows them.
ACTIVE_SETTING = re.compile(
    r"^(?:export\s+)?(?P<key>airllm_ollama_[A-Za-z0-9_]+)\s*=", re.IGNORECASE
)
COMMENTED_SETTING = re.compile(
    r"^#\s*(?:export\s+)?(?P<key>airllm_ollama_[A-Za-z0-9_]+)\s*=", re.IGNORECASE
)


def _defaults() -> dict[str, object]:
    """Return the default of every setting."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
    }


def _parse(name: str, value: str) -> object:
    """Build the setting from a value of the example file.

    The value goes through the environment of the settings, so the validators
    and the decoding of the settings apply. The environment of the machine is
    cleared, so only the value of the example file is read.
    """
    variable = f"{PREFIX}{name.upper()}"
    with mock.patch.dict(os.environ, {variable: value}, clear=True):
        return getattr(Settings(_env_file=None), name)


def _name(key: str) -> str:
    """Return the name of the setting an environment variable belongs to."""
    return key.lower().removeprefix(PREFIX)


def _active_values(text: str) -> dict[str, str]:
    """Return the settings the given text sets explicitly."""
    values = dotenv_values(stream=io.StringIO(text), interpolate=False)
    return {
        _name(key): value or "" for key, value in values.items() if key.lower().startswith(PREFIX)
    }


def _commented_entries(text: str) -> list[tuple[str, str, bool]]:
    """Return the settings the given text mentions as a commented out line.

    A setting can be mentioned more than once, with several alternatives, so
    every entry is returned. The flag tells whether the line is marked as an
    alternative.
    """
    entries: list[tuple[str, str, bool]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if COMMENTED_SETTING.match(stripped) is None:
            continue
        parsed = dotenv_values(stream=io.StringIO(stripped.lstrip("#")), interpolate=False)
        for key, value in parsed.items():
            if key.lower().startswith(PREFIX):
                entries.append((_name(key), value or "", stripped.endswith(ALTERNATIVE)))
    return entries


def _commented_values(text: str) -> dict[str, str]:
    """Return the commented values, the last one of a setting wins."""
    return {name: value for name, value, _ in _commented_entries(text)}


def _documented_keys(text: str) -> set[str]:
    """Return every setting the given text mentions, active or commented out."""
    return set(_active_values(text)) | set(_commented_values(text))


def _mismatches(text: str) -> list[str]:
    """Return the active values that differ from the default of the setting."""
    mismatches: list[str] = []
    for name, value in _active_values(text).items():
        field = Settings.model_fields.get(name)
        if field is None:
            mismatches.append(f"'{name}' is not a setting of the service")
            continue
        try:
            parsed = _parse(name, value)
        except ValidationError:
            # A value that does not fit its setting is reported separately.
            continue
        if parsed != field.get_default(call_default_factory=True):
            mismatches.append(f"{name} is set to '{value}'")
    return mismatches


def _invalid_values(text: str) -> list[str]:
    """Return the values of the example file that do not fit their setting."""
    invalid: list[str] = []
    entries = list(_active_values(text).items()) + [
        (name, value) for name, value, _ in _commented_entries(text)
    ]
    for name, value in entries:
        if name not in Settings.model_fields:
            continue
        try:
            _parse(name, value)
        except ValidationError:
            invalid.append(f"{name} is set to '{value}'")
    return invalid


def _stale_commented_values(text: str) -> list[str]:
    """Return the commented values that show a default that is not the default."""
    stale: list[str] = []
    for name, value, alternative in _commented_entries(text):
        if alternative:
            continue
        field = Settings.model_fields.get(name)
        if field is None:
            continue
        try:
            parsed = _parse(name, value)
        except ValidationError:
            continue
        if parsed != field.get_default(call_default_factory=True):
            stale.append(f"{name} is documented as '{value}'")
    return stale


def test_the_allow_list_of_undocumented_settings_is_valid() -> None:
    """The allow list may only name settings that exist."""
    unknown = sorted(UNDOCUMENTED - set(_defaults()))

    assert unknown == [], f"undocumented settings that do not exist: {unknown}"


def test_every_value_of_the_example_matches_the_default() -> None:
    """The example file must not drift away from the defaults."""
    mismatches = _mismatches(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert mismatches == [], f".env.example does not match the defaults: {mismatches}"


def test_every_documented_value_matches_the_default() -> None:
    """A commented out line that shows the default has to show the real one."""
    stale = _stale_commented_values(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert stale == [], (
        f".env.example documents outdated defaults: {stale}. A commented out line that "
        f"shows a value other than the default has to end with '{ALTERNATIVE}'."
    )


def test_every_value_of_the_example_fits_its_setting() -> None:
    """A typo in a value has to be caught as well."""
    invalid = _invalid_values(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert invalid == [], f".env.example has values that do not fit: {invalid}"


def test_every_key_of_the_example_names_a_setting() -> None:
    """A renamed setting must not survive in the example file."""
    defaults = _defaults()
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    unknown = sorted(name for name in _documented_keys(text) if name not in defaults)

    assert unknown == [], f"unknown settings in .env.example: {unknown}"


def _undocumented(text: str) -> list[str]:
    """Return the settings that the example file does not mention."""
    return sorted(set(_defaults()) - _documented_keys(text) - UNDOCUMENTED)


def test_every_setting_is_documented_in_the_example() -> None:
    """A new setting has to be documented in the example file."""
    undocumented = _undocumented(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert undocumented == [], f"settings missing in .env.example: {undocumented}"


def _duplicates(text: str) -> list[str]:
    """Return the settings that are set more than once."""
    keys = [
        match.group("key").lower()
        for match in (ACTIVE_SETTING.match(line.strip()) for line in text.splitlines())
        if match is not None
    ]
    return sorted({key for key in keys if keys.count(key) > 1})


def test_no_setting_is_set_twice() -> None:
    """A setting that is set twice in the example file is a mistake."""
    duplicates = _duplicates(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert duplicates == [], f"settings set more than once in .env.example: {duplicates}"


def test_a_commented_alternative_does_not_hide_the_active_value() -> None:
    """The parser reads the active line, not the commented out alternative."""
    text = "\n".join(
        [
            "AIRLLM_OLLAMA_MODEL_NAME=the-default",
            "# AIRLLM_OLLAMA_MODEL_NAME=the-alternative",
        ]
    )

    assert _active_values(text) == {"model_name": "the-default"}


@pytest.mark.parametrize(
    ("text", "mismatches", "invalid"),
    [
        ("AIRLLM_OLLAMA_PORT=11434\n", ["port is set to '11434'"], []),
        ("AIRLLM_OLLAMA_PORT=80OO\n", [], ["port is set to '80OO'"]),
        ("AIRLLM_OLLAMA_PORTS=8000\n", ["'ports' is not a setting of the service"], []),
        ("# AIRLLM_OLLAMA_PORT=11434\n", [], []),
    ],
)
def test_the_checks_catch_a_drift(text: str, mismatches: list[str], invalid: list[str]) -> None:
    """The guard has to fail when the example file drifts away."""
    assert _mismatches(text) == mismatches
    assert _invalid_values(text) == invalid


def test_the_checks_report_a_missing_setting() -> None:
    """A setting that is not documented at all has to be reported."""
    text = "AIRLLM_OLLAMA_HOST=127.0.0.1\n"

    assert "port" in _undocumented(text)


def test_the_checks_catch_a_duplicate_setting() -> None:
    """A setting that is set twice has to be reported, whatever its case is."""
    text = "AIRLLM_OLLAMA_PORT=8000\nairllm_ollama_port=9000\n"

    assert _duplicates(text) == ["airllm_ollama_port"]


def test_an_alternative_is_exempt_from_the_default_check() -> None:
    """A line that is marked as an alternative may show another value."""
    text = "# AIRLLM_OLLAMA_KEEP_ALIVE=5m  # alternative\n"

    assert _stale_commented_values(text) == []


def test_the_checks_report_a_stale_documented_default() -> None:
    """A commented out default that is not the default has to be reported."""
    text = "# AIRLLM_OLLAMA_STREAM_FORMAT=sse\n"

    assert _stale_commented_values(text) == ["stream_format is documented as 'sse'"]
