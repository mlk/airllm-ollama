"""Guard against a drift between ``.env.example`` and the settings defaults."""

from __future__ import annotations

import io
import pathlib
import re

from dotenv import dotenv_values
from pydantic import TypeAdapter

from airllm_ollama.config import Settings

ENV_EXAMPLE = pathlib.Path(__file__).parent.parent / ".env.example"

# The version comes from the package metadata and would have to be updated on
# every release, so it is not part of the example file.
UNDOCUMENTED = {"version"}

ACTIVE_SETTING = re.compile(r"^(?P<key>AIRLLM_OLLAMA_[A-Z_]+)=")
COMMENTED_SETTING = re.compile(r"^#\s*(?P<key>AIRLLM_OLLAMA_[A-Z_]+)=")


def _defaults() -> dict[str, object]:
    """Return the default of every setting."""
    return {name: field.default for name, field in Settings.model_fields.items()}


def _active_values(text: str) -> dict[str, str]:
    """Return the settings the given text sets explicitly."""
    values = dotenv_values(stream=io.StringIO(text))
    return {
        key.removeprefix("AIRLLM_OLLAMA_").lower(): value or "" for key, value in values.items()
    }


def _documented_keys(text: str) -> set[str]:
    """Return every setting the given text mentions, active or commented out."""
    keys = set(_active_values(text))
    for line in text.splitlines():
        match = COMMENTED_SETTING.match(line.strip())
        if match is not None:
            keys.add(match.group("key").removeprefix("AIRLLM_OLLAMA_").lower())
    return keys


def test_every_value_of_the_example_matches_the_default() -> None:
    """The example file must not drift away from the defaults."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    fields = Settings.model_fields

    for name, value in _active_values(text).items():
        assert name in fields, f"'{name}' is not a setting of the service"
        field = fields[name]
        annotation = field.annotation
        assert annotation is not None
        parsed = TypeAdapter(annotation).validate_python(value)
        assert parsed == field.default, (
            f".env.example sets {name} to '{value}', the default is '{field.default}'"
        )


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
