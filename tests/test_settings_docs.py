"""Guard against a drift between the documentation and the settings defaults.

The defaults of the settings are documented twice: in the example file
``.env.example`` and in the settings table of the README. Both have to follow
the code.
"""

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
README = pathlib.Path(__file__).parent.parent / "README.md"

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

# A row of the settings table of the README: the variable, its default and a
# description. The default is written in backticks.
README_ROW = re.compile(r"^\|\s*`(?P<key>AIRLLM_OLLAMA_[A-Za-z0-9_]+)`\s*\|\s*(?P<value>[^|]*)\|")

# The README writes an empty default like this.
EMPTY = "_(empty)_"


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


def _readme_values(text: str) -> dict[str, str]:
    """Return the default the settings table of the README documents.

    Args:
        text: The content of the README.

    Returns:
        The documented default of every setting of the table.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = README_ROW.match(line.strip())
        if match is None:
            continue
        values[_name(match.group("key"))] = _readme_value(match.group("value"))
    return values


def _readme_value(cell: str) -> str:
    """Return the value a cell of the settings table documents.

    Args:
        cell: The cell of the table.

    Returns:
        The documented value, without the backticks.
    """
    text = cell.strip()
    if text == EMPTY:
        return ""
    if text.startswith("`") and text.endswith("`"):
        return text[1:-1]
    return text


def _readme_mismatches(text: str) -> list[str]:
    """Return the documented defaults that differ from the default of the setting.

    The documented value is read through the settings, so that it is compared
    with the default as a typed value — a boolean or a list is not compared as
    its string form.

    Args:
        text: The content of the README.

    Returns:
        A message for every setting whose documented default is not the real one.
    """
    mismatches: list[str] = []
    defaults = _defaults()
    for name, value in _readme_values(text).items():
        if name not in defaults:
            # An unknown setting is reported by the key check.
            continue
        try:
            parsed = _parse(name, value)
        except ValidationError:
            mismatches.append(f"{name} is documented as '{value}', which does not fit the setting")
            continue
        if parsed != defaults[name]:
            mismatches.append(f"{name} is documented as '{value}'")
    return mismatches


def _readme_keys(text: str) -> set[str]:
    """Return the settings the settings table of the README documents.

    Args:
        text: The content of the README.

    Returns:
        The name of every documented setting.
    """
    return set(_readme_values(text))


def _readme_duplicates(text: str) -> list[str]:
    """Return the variables that the settings table documents more than once.

    Args:
        text: The content of the README.

    Returns:
        The name of every duplicated variable.
    """
    keys = [
        match.group("key").lower()
        for match in (README_ROW.match(line.strip()) for line in text.splitlines())
        if match is not None
    ]
    return sorted({key for key in keys if keys.count(key) > 1})


def _mismatches(text: str) -> list[str]:
    """Return the active values that differ from the default of the setting."""
    mismatches: list[str] = []
    defaults = _defaults()
    for name, value in _active_values(text).items():
        if name not in defaults:
            # An unknown setting is reported by the key check.
            continue
        try:
            parsed = _parse(name, value)
        except ValidationError:
            # A value that does not fit its setting is reported separately.
            continue
        if parsed != defaults[name]:
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
    ("text", "mismatches", "invalid", "stale"),
    [
        ("AIRLLM_OLLAMA_PORT=11434\n", ["port is set to '11434'"], [], []),
        ("AIRLLM_OLLAMA_PORT=80OO\n", [], ["port is set to '80OO'"], []),
        ("AIRLLM_OLLAMA_PORTS=8000\n", [], [], []),  # an unknown key
        ("# AIRLLM_OLLAMA_PORT=11434\n", [], [], ["port is documented as '11434'"]),
    ],
)
def test_the_checks_catch_a_drift(
    text: str, mismatches: list[str], invalid: list[str], stale: list[str]
) -> None:
    """The guard has to fail when the example file drifts away."""
    assert _mismatches(text) == mismatches
    assert _invalid_values(text) == invalid
    assert _stale_commented_values(text) == stale


def test_an_unknown_setting_is_reported() -> None:
    """A key that is not a setting has to be reported."""
    text = "AIRLLM_OLLAMA_PORTS=8000\n"
    unknown = sorted(name for name in _documented_keys(text) if name not in _defaults())

    assert unknown == ["ports"]


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


def test_every_default_of_the_readme_matches_the_default() -> None:
    """The settings table of the README must not drift away from the defaults."""
    mismatches = _readme_mismatches(README.read_text(encoding="utf-8"))

    assert mismatches == [], f"the README documents outdated defaults: {mismatches}"


def test_every_key_of_the_readme_names_a_setting() -> None:
    """A renamed setting must not survive in the settings table of the README."""
    text = README.read_text(encoding="utf-8")
    unknown = sorted(name for name in _readme_keys(text) if name not in _defaults())

    assert unknown == [], f"unknown settings in the README: {unknown}"


def test_every_setting_is_documented_in_the_readme() -> None:
    """A new setting has to be documented in the settings table of the README."""
    text = README.read_text(encoding="utf-8")
    undocumented = sorted(set(_defaults()) - _readme_keys(text) - UNDOCUMENTED)

    assert undocumented == [], f"settings missing in the README: {undocumented}"


def test_no_setting_is_documented_twice_in_the_readme() -> None:
    """A setting that the table documents twice is a mistake."""
    duplicates = _readme_duplicates(README.read_text(encoding="utf-8"))

    assert duplicates == [], f"settings documented more than once in the README: {duplicates}"


@pytest.mark.parametrize(
    ("row", "mismatches"),
    [
        ("| `AIRLLM_OLLAMA_PORT` | `11434` | Server port |", ["port is documented as '11434'"]),
        ("| `AIRLLM_OLLAMA_PORT` | `8000` | Server port |", []),
        (
            "| `AIRLLM_OLLAMA_MAX_SEQ_LEN` | `1024` | Context |",
            ["max_seq_len is documented as '1024'"],
        ),
        ("| `AIRLLM_OLLAMA_MODEL_PATH` | _(empty)_ | Path |", []),
        ("| `AIRLLM_OLLAMA_KEEP_ALIVE` | `5m` | Lifetime |", ["keep_alive is documented as '5m'"]),
        (
            "| `AIRLLM_OLLAMA_PORT` | `80OO` | Server port |",
            ["port is documented as '80OO', which does not fit the setting"],
        ),
    ],
)
def test_the_readme_checks_catch_a_drift(row: str, mismatches: list[str]) -> None:
    """The guard has to fail when the settings table drifts away."""
    assert _readme_mismatches(row) == mismatches


def test_the_readme_checks_report_an_empty_default() -> None:
    """An empty default is written as `_(empty)_` in the README."""
    text = "| `AIRLLM_OLLAMA_MODEL_PATH` | _(empty)_ | Path |"

    assert _readme_values(text) == {"model_path": ""}


def test_the_readme_checks_report_a_duplicate_setting() -> None:
    """A variable that the table documents twice has to be reported."""
    text = (
        "| `AIRLLM_OLLAMA_PORT` | `8000` | Server port |\n"
        "| `AIRLLM_OLLAMA_PORT` | `9000` | Server port |\n"
    )

    assert _readme_duplicates(text) == ["airllm_ollama_port"]


def test_the_readme_checks_ignore_other_tables() -> None:
    """Only the settings table is read, not the tables of the API."""
    text = "| Method | Path | Description |\n| --- | --- | --- |\n| `GET` | `/` | The root |"

    assert _readme_values(text) == {}
