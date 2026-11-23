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


def test_a_nested_marker_inside_a_malformed_block_is_not_a_call() -> None:
    raw = (
        '<tool_call>{"arguments": <tool_call>'
        '{"name": "delete_file", "arguments": {}}</tool_call>}</tool_call>'
    )

    text, calls = extract_tool_calls(raw)

    assert calls == []
    assert text == raw


def test_an_independent_call_after_a_malformed_block_is_still_parsed() -> None:
    _, calls = extract_tool_calls(
        '<tool_call>{"broken": <tool_call>{"name": "delete_file"}</tool_call>}</tool_call>'
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_lone_surrogate_in_a_json_encoded_arguments_string_is_rejected() -> None:
    raw = '<tool_call>{"name": "f", "arguments": "{\\"text\\": \\"\\ud800\\"}"}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_surrogate_pair_in_a_json_encoded_arguments_string_is_accepted() -> None:
    text, calls = extract_tool_calls(
        '<tool_call>{"name": "f", "arguments": "{\\"text\\": \\"\\ud83d\\ude00\\"}"}</tool_call>'
    )

    assert text == ""
    assert calls[0].function.arguments == {"text": "😀"}


def test_an_inline_code_example_is_not_a_call() -> None:
    raw = 'Example: `<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>` — use it.'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_fenced_code_example_is_not_a_call() -> None:
    raw = '```json\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n```'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_unterminated_fence_swallows_the_example() -> None:
    raw = '```\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_call_after_a_code_example_is_still_parsed() -> None:
    raw = (
        'Like `<tool_call>{"name": "example"}</tool_call>`. '
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_marker_inside_a_json_string_inside_a_malformed_block_is_not_a_call() -> None:
    raw = (
        '<tool_call>{"arguments":{"x":"</tool_call>"},'
        '"bad":<tool_call>{"name":"delete_file"}</tool_call>}</tool_call>'
        ' <tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_deeply_nested_argument_is_rejected_without_crashing() -> None:
    depth = 4000
    arguments = '{"a":' * depth + "1" + "}" * depth
    raw = '<tool_call>{"name": "f", "arguments": ' + arguments + "}</tool_call>"

    text, calls = extract_tool_calls(raw)

    assert calls == []
    assert text == raw


def test_a_deeply_nested_valid_call_is_accepted() -> None:
    depth = 100
    arguments = '{"a":' * depth + "1" + "}" * depth
    raw = '<tool_call>{"name": "f", "arguments": ' + arguments + "}</tool_call>"

    _, calls = extract_tool_calls(raw)

    assert len(calls) == 1


def test_a_double_backtick_example_is_not_a_call() -> None:
    raw = 'Example: ``<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>``'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_tilde_fenced_example_is_not_a_call() -> None:
    raw = '~~~\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n~~~'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_valid_call_whose_arguments_contain_a_fence_is_not_cut_short() -> None:
    body = (
        '<tool_call>{"name": "save", "arguments": {"text": "look: ```\\nsome code\\n```"}}'
        "</tool_call>\n"
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>'
    )

    text, calls = extract_tool_calls(body)

    assert [call.function.name for call in calls] == ["save", "delete_file"]
    assert text == ""


def test_a_longer_closing_fence_is_required() -> None:
    raw = '````\n```\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n````'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_inline_code_span_may_span_lines() -> None:
    raw = 'see `example\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>` done'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_call_after_a_multiline_code_span_is_still_parsed() -> None:
    raw = (
        'see `example\n<tool_call>{"name": "deleted"}</tool_call>` done. '
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_an_indented_code_block_is_not_a_call() -> None:
    raw = 'Example:\n\n    <tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_tab_indented_code_block_is_not_a_call() -> None:
    raw = 'Example:\n\n\t<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_indented_line_inside_a_paragraph_is_not_a_code_block() -> None:
    raw = 'Use it like this:\n    <tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_call_after_an_indented_code_block_is_still_parsed() -> None:
    raw = (
        'Example:\n\n    <tool_call>{"name": "example"}</tool_call>\n\n'
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_an_indented_closing_fence_does_not_close_the_fence() -> None:
    raw = '~~~text\n    ~~~\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n~~~'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_blockquote_fence_is_not_closed_by_an_unquoted_fence() -> None:
    raw = (
        "> ```\n"
        "> sample\n"
        "```\n"
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n'
        "~~~\n"
    )

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_indented_code_block_after_a_heading_is_not_a_call() -> None:
    raw = '# Example\n    <tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_indented_code_block_inside_a_blockquote_is_not_a_call() -> None:
    raw = '> Example\n>\n>     <tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_entity_before_an_inline_code_span_does_not_expose_the_call() -> None:
    raw = 'Fish &amp; chips: `<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>`'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_call_after_a_think_block_is_parsed() -> None:
    raw = (
        "<think>Need the weather for the user.</think>\n"
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_call_inside_a_think_block_is_not_executed() -> None:
    raw = (
        '<think>Maybe <tool_call>{"name": "delete_file", "arguments": {}}</tool_call>'
        "</think>\nDone."
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_bare_carriage_returns_do_not_break_the_scanner() -> None:
    raw = 'Hi\r\r<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_crlf_fence_is_recognized() -> None:
    raw = '```\r\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\r\n```\r\n'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_think_block_with_a_quoted_closing_delimiter_is_not_executable() -> None:
    raw = (
        "<think>Example: `</think>`. "
        'Never call <tool_call>{"name": "delete_file", "arguments": {}}</tool_call></think>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_an_unmatched_backtick_in_one_paragraph_does_not_hide_another() -> None:
    raw = (
        'unmatched `\n\nExample: `<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>`.'
    )

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_html_comment_example_is_not_a_call() -> None:
    raw = 'Example: <!-- <tool_call>{"name": "delete_file", "arguments": {}}</tool_call> -->'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_code_tag_example_is_not_a_call() -> None:
    raw = 'Example: <code><tool_call>{"name": "delete_file", "arguments": {}}</tool_call></code>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_escaped_opening_marker_is_not_a_call() -> None:
    raw = 'Example: \\<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_double_backslash_before_the_marker_is_an_escaped_backslash() -> None:
    raw = 'Example: \\\\<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_think_marker_inside_a_call_does_not_swallow_the_next_call() -> None:
    raw = (
        '<tool_call>{"name": "save", "arguments": {"text": "<think>"}}</tool_call>'
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["save", "get_weather"]


def test_a_heading_boundary_stops_a_code_span() -> None:
    raw = (
        "# unmatched `\n"
        '# Example: `<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>`'
    )

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_unterminated_html_comment_is_not_a_call() -> None:
    raw = '<!-- example\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_call_after_a_closed_html_block_is_parsed() -> None:
    raw = '<div>\n</div>\n\n<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>\n'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_call_inside_an_html_block_is_not_executed() -> None:
    # CommonMark: an HTML block continues until a blank line, so the call is
    # part of the markup and is not executed.
    raw = '<div>\n</div>\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n'

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_an_h1_block_example_is_not_a_call() -> None:
    raw = '<h1>\n<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\n</h1>'

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_nested_div_example_is_not_a_call() -> None:
    raw = (
        "<div><div>example</div>"
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call></div>'
    )

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_a_backtick_inside_an_html_attribute_is_not_a_code_span() -> None:
    raw = (
        'Link <a title="`">link</a>\n'
        '`<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>`'
    )

    text, calls = extract_tool_calls(raw)

    assert text == raw
    assert calls == []


def test_an_unterminated_div_in_a_fence_keeps_the_real_call() -> None:
    raw = '```\n<div>\n```\n\n<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_html_tag_inside_a_call_argument_keeps_the_next_call() -> None:
    raw = (
        '<tool_call>{"name": "save", "arguments": {"text": "<div>"}}</tool_call>\n\n'
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["save", "get_weather"]


def test_a_think_block_with_a_commented_closing_delimiter_is_not_executable() -> None:
    raw = (
        "<think>Example: <!-- </think> --> "
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call> </think>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_malformed_call_with_an_escaped_closing_marker_swallows_the_next_block() -> None:
    raw = (
        "<tool_call>bad \\</tool_call> "
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call> </tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_two_calls_with_backticks_in_their_arguments_are_both_parsed() -> None:
    raw = (
        '<tool_call>{"name": "save", "arguments": {"text": "`"}}</tool_call>'
        '<tool_call>{"name": "save", "arguments": {"text": "`"}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["save", "save"]


def test_a_call_inside_an_unterminated_script_is_not_executed() -> None:
    raw = (
        "Example: <script>var x = 1;\n"
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_call_after_a_closed_script_is_parsed() -> None:
    raw = (
        "Example: <script>var x = 1;</script>\n"
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_think_block_spanning_a_blank_line_is_not_executed() -> None:
    raw = (
        "<think>\nThinking.\n\n"
        'Maybe <tool_call>{"name": "delete_file", "arguments": {}}</tool_call> </think>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_call_inside_a_single_quoted_attribute_is_not_executed() -> None:
    raw = (
        'Example: <a title=\'<tool_call>{"name": "delete_file", '
        '"arguments": {}}</tool_call>\'>link</a>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_call_inside_an_attribute_containing_a_greater_than_sign() -> None:
    raw = (
        'Example: <a title=\'a > b <tool_call>{"name": "delete_file", '
        '"arguments": {}}</tool_call>\'>link</a>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_comment_opener_inside_call_arguments_does_not_swallow_the_next_call() -> None:
    raw = (
        '<tool_call>{"name": "save", "arguments": {"text": "<!--"}}</tool_call>'
        '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'
    )

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["save", "get_weather"]


def test_a_call_after_an_inline_code_example_is_executed() -> None:
    raw = 'Example: `<div>`\n\n<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_call_after_an_escaped_tag_is_executed() -> None:
    raw = 'Example: \\<div>\n\n<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_call_inside_a_void_element_attribute_is_not_executed() -> None:
    raw = 'Example: <img alt=\'<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>\'>'

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_comment_opener_inside_backticks_does_not_swallow_the_call() -> None:
    raw = 'Example: `<!--`\n\n<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_think_closing_tag_inside_a_tag_attribute_does_not_close_the_block() -> None:
    raw = (
        "<think><a title='</think>'>link</a> "
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call></think>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_call_spanned_by_an_apparent_tag_and_backticks_is_not_executed() -> None:
    raw = 'Example: `<img alt=\'` then `<tool_call>{"name":"delete_file"}</tool_call>`\'>'

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_think_delimiter_inside_a_script_does_not_close_the_block() -> None:
    raw = (
        '<think><script>const text="</think>";</script> '
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call></think>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_think_delimiter_inside_a_style_element_does_not_close_the_block() -> None:
    raw = (
        '<think><style>.x::after { content: "</think>" }</style> '
        '<tool_call>{"name": "delete_file", "arguments": {}}</tool_call></think>'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_backtick_inside_a_link_destination_is_not_a_code_span() -> None:
    raw = '[link](https://example.com/`) `<tool_call>{"name":"delete_file"}</tool_call>`'

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_backtick_inside_a_reference_link_definition_is_not_a_code_span() -> None:
    raw = (
        '[link][x]\n\n[x]: https://example.com/`\n\n`<tool_call>{"name":"delete_file"}</tool_call>`'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_call_after_a_link_destination_is_executed() -> None:
    raw = '[link](https://example.com/`) and then <tool_call>{"name":"get_weather"}</tool_call>'

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_many_unterminated_tags_do_not_hang() -> None:
    raw = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>' + "<a " * 2000

    _, calls = extract_tool_calls(raw)

    assert [call.function.name for call in calls] == ["get_weather"]


def test_a_stray_closing_paren_is_not_a_link() -> None:
    raw = 'Syntax ]( example: `<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>`'

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_an_unquoted_attribute_with_slashes_terminates() -> None:
    raw = (
        "Example: <a href=https://example.com>link</a> "
        '`<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>`'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []


def test_a_link_destination_with_nested_parentheses_masks_the_backtick() -> None:
    raw = (
        "[link](https://example.com/a_(b)/`) "
        '`<tool_call>{"name": "delete_file", "arguments": {}}</tool_call>`'
    )

    _, calls = extract_tool_calls(raw)

    assert calls == []
