"""Guard against a drift between ``.env.example`` and the settings defaults."""

from __future__ import annotations

import io
import pathlib
import re

from dotenv import dotenv_values
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from airllm_ollama.config import Settings

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"

# The version comes from the package metadata and would have to be updated on
# every release, so it is not part of the example file.
UNDOCUMENTED = {"version"}

ACTIVE_SETTING = re.compile(r"^(?:export\s+)?(?P<key>AIRLLM_OLLAMA_[A-Z_]+)=")
COMMENTED_SETTING = re.compile(r"^#\s*(?:export\s+)?(?P<key>AIRLLM_OLLAMA_[A-Z_]+)=")


def _defaults() -> dict[str, object]:
    """Return the default of every setting."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in Settings.model_fields.items()
    }


def _field(name: str) -> FieldInfo:
    """Return the field of a setting."""
    return Settings.model_fields[name]


def _parse(field: FieldInfo, value: str) -> object:
    """Parse a value of the example file with the type of the setting."""
    annotation = field.annotation
    assert annotation is not None
    return TypeAdapter(annotation).validate_python(value)


def _active_values(text: str) -> dict[str, str]:
    """Return the settings the given text sets explicitly."""
    values = dotenv_values(stream=io.StringIO(text))
    return {
        key.removeprefix("AIRLLM_OLLAMA_").lower(): value or "" for key, value in values.items()
    }


def _commented_values(text: str) -> dict[str, str]:
    """Return the settings the given text mentions as a commented out line."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = COMMENTED_SETTING.match(line.strip())
        if match is None:
            continue
        key = match.group("key").removeprefix("AIRLLM_OLLAMA_").lower()
        values[key] = line.strip().split("=", 1)[1].strip()
    return values


def _documented_keys(text: str) -> set[str]:
    """Return every setting the given text mentions, active or commented out."""
    return set(_active_values(text)) | set(_commented_values(text))


def _mismatches(text: str) -> list[str]:
    """Return the active values that differ from the default of the setting."""
    mismatches: list[str] = []
    for name, value in _active_values(text).items():
        if name not in Settings.model_fields:
            mismatches.append(f"'{name}' is not a setting of the service")
            continue
        field = _field(name)
        if _parse(field, value) != field.get_default(call_default_factory=True):
            mismatches.append(f"{name} is set to '{value}'")
    return mismatches


def _invalid_values(text: str) -> list[str]:
    """Return the values of the example file that do not fit their setting."""
    invalid: list[str] = []
    for name, value in {**_commented_values(text), **_active_values(text)}.items():
        if name not in Settings.model_fields:
            continue
        try:
            _parse(_field(name), value)
        except ValueError:
            invalid.append(f"{name} is set to '{value}'")
    return invalid


def test_every_value_of_the_example_matches_the_default() -> None:
    """The example file must not drift away from the defaults."""
    mismatches = _mismatches(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert mismatches == []


def test_every_value_of_the_example_fits_its_setting() -> None:
    """A typo in a value has to be caught as well."""
    invalid = _invalid_values(ENV_EXAMPLE.read_text(encoding="utf-8"))

    assert invalid == []


def test_every_key_of_the_example_names_a_setting() -> None:
    """A renamed setting must not survive in the example file."""
    defaults = _defaults()
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    for name in _documented_keys(text):
        assert name in defaults, f"'{name}' is not a setting of the service"


def test_every_setting_is_documented_in_the_example() -> None:
    """A new setting has to be documented in the example file."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    undocumented = set(_defaults()) - _documented_keys(text) - UNDOCUMENTED

    assert undocumented == set()


def test_no_setting_is_set_twice() -> None:
    """A setting that is set twice in the example file is a mistake."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    keys = [
        match.group("key")
        for match in (ACTIVE_SETTING.match(line.strip()) for line in text.splitlines())
        if match is not None
    ]

    assert len(keys) == len(set(keys))


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
    text = "AIRLLM_OLLAMA_PORT=11434\n"

    assert _mismatches(text) == ["port is set to '11434'"]
    assert _invalid_values(text) == []
