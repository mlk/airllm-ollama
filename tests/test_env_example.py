"""Guard against a drift between ``.env.example`` and the settings defaults."""

from __future__ import annotations

import io
import pathlib
import re

from dotenv import dotenv_values
from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from airllm_ollama.config import Settings

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"

PREFIX = "AIRLLM_OLLAMA_"

# The version comes from the package metadata and would have to be updated on
# every release, so it is not part of the example file.
UNDOCUMENTED = {"version"}

# Assignments, active or commented out. Spaces around the = are allowed, like
# dotenv allows them.
ACTIVE_SETTING = re.compile(r"^(?:export\s+)?(?P<key>AIRLLM_OLLAMA_[A-Z0-9_]+)\s*=")
COMMENTED_SETTING = re.compile(r"^#\s*(?:export\s+)?(?P<key>AIRLLM_OLLAMA_[A-Z0-9_]+)\s*=")


def _defaults() -> dict[str, object]:
    """Return the default of every setting."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
    }


def _parse(field: FieldInfo, value: str) -> object:
    """Parse a value of the example file with the type of the setting."""
    annotation = field.annotation
    assert annotation is not None
    return TypeAdapter(annotation).validate_python(value)


def _active_values(text: str) -> dict[str, str]:
    """Return the settings the given text sets explicitly."""
    values = dotenv_values(stream=io.StringIO(text))
    return {
        key.removeprefix(PREFIX).lower(): value or ""
        for key, value in values.items()
        if key.startswith(PREFIX)
    }


def _commented_values(text: str) -> dict[str, str]:
    """Return the settings the given text mentions as a commented out line."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if COMMENTED_SETTING.match(stripped) is None:
            continue
        for key, value in dotenv_values(stream=io.StringIO(stripped.lstrip("#"))).items():
            if key.startswith(PREFIX):
                values[key.removeprefix(PREFIX).lower()] = value or ""
    return values


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
            parsed = _parse(field, value)
        except ValidationError:
            # A value that does not fit its setting is reported separately.
            continue
        if parsed != field.get_default(call_default_factory=True):
            mismatches.append(f"{name} is set to '{value}'")
    return mismatches


def _invalid_values(text: str) -> list[str]:
    """Return the values of the example file that do not fit their setting."""
    invalid: list[str] = []
    for name, value in {**_commented_values(text), **_active_values(text)}.items():
        field = Settings.model_fields.get(name)
        if field is None:
            continue
        try:
            _parse(field, value)
        except ValidationError:
            invalid.append(f"{name} is set to '{value}'")
    return invalid


def test_the_allow_list_of_undocumented_settings_is_valid() -> None:
    """The allow list may only name settings that exist."""
    unknown = sorted(UNDOCUMENTED - set(_defaults()))

    assert unknown == [], f"undocumented settings that do not exist: {unknown}"


def test_every_value_of_the_example_matches_the_default() -> None:
    """The example file must not drift away from the defaults."""
    mismatches = _mismatches(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert mismatches == [], f".env.example does not match the defaults: {mismatches}"


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


def test_every_setting_is_documented_in_the_example() -> None:
    """A new setting has to be documented in the example file."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    undocumented = sorted(set(_defaults()) - _documented_keys(text) - UNDOCUMENTED)

    assert undocumented == [], f"settings missing in .env.example: {undocumented}"


def test_no_setting_is_set_twice() -> None:
    """A setting that is set twice in the example file is a mistake."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    keys = [
        match.group("key")
        for match in (ACTIVE_SETTING.match(line.strip()) for line in text.splitlines())
        if match is not None
    ]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})

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


def test_the_checks_catch_a_drift() -> None:
    """The guard has to fail when the example file drifts away."""
    stale = "AIRLLM_OLLAMA_PORT=11434\n"
    assert _mismatches(stale) == ["port is set to '11434'"]
    assert _invalid_values(stale) == []

    invalid = "AIRLLM_OLLAMA_PORT=80OO\n"
    assert _invalid_values(invalid) == ["port is set to '80OO'"]
    assert _mismatches(invalid) == []

    unknown = "AIRLLM_OLLAMA_PORTS=8000\n"
    assert _mismatches(unknown) == ["'ports' is not a setting of the service"]

    missing = "AIRLLM_OLLAMA_HOST=127.0.0.1\n"
    assert "port" in set(_defaults()) - _documented_keys(missing) - UNDOCUMENTED

    commented = "# AIRLLM_OLLAMA_PORT=11434\n"
    assert _mismatches(commented) == []
    assert _invalid_values(commented) == []
    assert "port" in _documented_keys(commented)
