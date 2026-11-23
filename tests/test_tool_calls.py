"""Unit tests for the extraction of tool calls from the generated text."""

from __future__ import annotations

from airllm_ollama.tool_calls import extract_tool_calls


def test_a_plain_answer_has_no_tool_calls() -> None:
    text, calls = extract_tool_calls("The sky is blue because of Rayleigh scattering.")

    assert text == "The sky is blue because of Rayleigh scattering."
    assert calls == []


def test_a_tool_call_is_extracted_and_removed_from_the_text() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    )

    assert text == ""
    assert len(calls) == 1
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == {"city": "Paris"}


def test_a_tool_call_with_text_around_it_keeps_the_surrounding_text() -> None:
    text, calls = extract_tool_calls(
        'Let me check. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}'
        "</tool_call> Done."
    )

    assert text == "Let me check.  Done."
    assert [call.function.name for call in calls] == ["get_weather"]


def test_several_tool_calls_are_all_extracted() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"n": 1}}</tool_call>'
    )

    assert text == ""
    assert [call.function.name for call in calls] == ["a", "b"]


def test_a_tool_call_without_arguments_defaults_to_an_empty_object() -> None:
    text, calls = extract_tool_calls('<tool_call>{"name": "list_files"}</tool_call>')

    assert text == ""
    assert calls[0].function.arguments == {}


def test_arguments_encoded_as_a_json_string_are_parsed() -> None:
    _, calls = extract_tool_calls(
        '<tool_call>{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}</tool_call>'
    )

    assert calls[0].function.arguments == {"city": "Paris"}


def test_a_tool_call_with_invalid_json_is_left_in_the_text() -> None:
    raw = "<tool_call>{not json}</tool_call>"

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_tool_call_without_a_name_is_left_in_the_text() -> None:
    raw = '<tool_call>{"arguments": {}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_tool_call_with_non_object_arguments_is_left_in_the_text() -> None:
    raw = '<tool_call>{"name": "f", "arguments": [1, 2]}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_tool_call_with_a_string_that_is_not_json_is_left_in_the_text() -> None:
    raw = '<tool_call>{"name": "f", "arguments": "not json"}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_tool_call_spanning_several_lines_is_found() -> None:
    payload = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    text, calls = extract_tool_calls(f"<tool_call>\n{payload}\n</tool_call>")

    assert text == ""
    assert calls[0].function.arguments == {"city": "Paris"}


def test_a_valid_tool_call_among_invalid_ones_is_still_extracted() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{broken}</tool_call><tool_call>{"name": "f", "arguments": {}}</tool_call>'
    )

    assert text == "<tool_call>{broken}</tool_call>"
    assert [call.function.name for call in calls] == ["f"]


def test_a_tool_call_with_a_non_object_body_is_left_in_the_text() -> None:
    raw = "<tool_call>[1, 2]</tool_call>"

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_tool_call_with_null_arguments_gets_empty_arguments() -> None:
    text, calls = extract_tool_calls('<tool_call>{"name": "f", "arguments": null}</tool_call>')

    assert text == ""
    assert calls[0].function.arguments == {}


def test_a_closing_marker_inside_a_json_string_does_not_end_the_call() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{"name": "save", "arguments": {"text": "</tool_call>"}}</tool_call>'
    )

    assert text == ""
    assert calls[0].function.name == "save"
    assert calls[0].function.arguments == {"text": "</tool_call>"}


def test_a_non_finite_number_is_rejected() -> None:
    raw = '<tool_call>{"name": "f", "arguments": {"n": NaN}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_out_of_range_float_is_rejected() -> None:
    raw = '<tool_call>{"name": "f", "arguments": {"n": 1e999}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_non_finite_number_in_a_json_encoded_arguments_string_is_rejected() -> None:
    raw = '<tool_call>{"name": "f", "arguments": "{\\"n\\": Infinity}"}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_unterminated_tool_call_is_left_in_the_text() -> None:
    raw = '<tool_call>{"name": "f", "arguments": {}}'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_closing_marker_without_a_valid_body_is_skipped() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>garbage</tool_call><tool_call>{"name": "f", "arguments": {}}</tool_call>'
    )

    assert text == "<tool_call>garbage</tool_call>"
    assert [call.function.name for call in calls] == ["f"]


def test_a_lone_surrogate_in_the_arguments_is_rejected() -> None:
    raw = '<tool_call>{"name": "f", "arguments": {"text": "\\ud800"}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_lone_surrogate_in_the_name_is_rejected() -> None:
    raw = '<tool_call>{"name": "\\ud800", "arguments": {}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_surrogate_pair_is_accepted() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{"name": "f", "arguments": {"text": "\\ud83d\\ude00"}}</tool_call>'
    )

    assert text == ""
    assert calls[0].function.arguments == {"text": "😀"}
