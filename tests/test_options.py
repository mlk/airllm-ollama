"""Unit tests for the translation of Ollama options."""

from __future__ import annotations

from airllm_ollama.options import apply_stop_sequences, map_options, parse_keep_alive
from airllm_ollama.schemas.generate import GenerateOptions


def test_map_options_without_options() -> None:
    assert map_options(None) == {}


def test_map_options_translates_the_supported_options() -> None:
    mapped = map_options(
        GenerateOptions(
            temperature=0.2,
            top_p=0.8,
            top_k=20,
            num_predict=64,
            num_ctx=4096,
            repeat_penalty=1.05,
        )
    )

    assert mapped["temperature"] == 0.2
    assert mapped["top_p"] == 0.8
    assert mapped["top_k"] == 20
    assert mapped["max_new_tokens"] == 64
    assert mapped["max_length"] == 4096
    assert mapped["repetition_penalty"] == 1.05
    assert mapped["do_sample"] is True


def test_map_options_ignores_unsupported_options() -> None:
    mapped = map_options(
        GenerateOptions(
            mirostat=2,
            mirostat_tau=0.5,
            presence_penalty=1.0,
            frequency_penalty=0.5,
            seed=42,
            min_p=0.05,
        )
    )

    assert mapped == {}


def test_map_options_enables_sampling_for_every_sampling_option() -> None:
    assert map_options(GenerateOptions(temperature=0.1))["do_sample"] is True
    assert map_options(GenerateOptions(top_p=0.1))["do_sample"] is True
    assert map_options(GenerateOptions(top_k=5))["do_sample"] is True
    assert "do_sample" not in map_options(GenerateOptions(num_predict=5))


def test_apply_stop_sequences_without_stop_sequences() -> None:
    assert apply_stop_sequences("text", None) == "text"
    assert apply_stop_sequences("text", []) == "text"


def test_apply_stop_sequences_cuts_at_the_first_match() -> None:
    assert apply_stop_sequences("before STOP after", ["STOP"]) == "before "
    assert apply_stop_sequences("before END after", ["STOP", "END"]) == "before "
    assert apply_stop_sequences("before STOP after", ["MISSING"]) == "before STOP after"


def test_apply_stop_sequences_ignores_empty_sequences() -> None:
    assert apply_stop_sequences("text", [""]) == "text"


def test_parse_keep_alive() -> None:
    assert parse_keep_alive(None) is None
    assert parse_keep_alive("0") == 0
    assert parse_keep_alive("30s") == 30
    assert parse_keep_alive("5m") == 300
    assert parse_keep_alive("1h") == 3600
    assert parse_keep_alive("1.5m") == 90
    assert parse_keep_alive("  2 m ") == 120


def test_parse_keep_alive_ignores_unparsable_values() -> None:
    assert parse_keep_alive("forever") is None
    assert parse_keep_alive("5d") is None
    assert parse_keep_alive("") is None
