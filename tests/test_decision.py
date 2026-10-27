"""Unit tests for the decision models."""

from __future__ import annotations

import pytest

from airllm_ollama.decision import (
    best,
    build_prompt,
    confidence,
    normalize,
    score,
)

TEMPLATE = "Context: {state}\nChoices: {choices}\nSelected Option:"


def test_the_prompt_carries_the_state_and_the_candidates() -> None:
    """The prompt is the template with its placeholders filled in."""
    prompt = build_prompt(TEMPLATE, "a ticket", ["billing", "bug"])

    assert prompt == "Context: a ticket\nChoices: ['billing', 'bug']\nSelected Option:"


def test_the_newlines_of_the_template_are_expanded() -> None:
    """The template stays a single line in the environment."""
    prompt = build_prompt("Context: {state}\\nChoices: {choices}", "a ticket", ["a"])

    assert prompt == "Context: a ticket\nChoices: ['a']"


def test_the_probabilities_are_normalized() -> None:
    """The probabilities of the candidates sum to one."""
    normalized = normalize({"a": 0.2, "b": 0.3})

    assert normalized == {"a": 0.4, "b": 0.6}


def test_normalizing_nothing_leaves_the_probabilities_alone() -> None:
    """The probabilities of a model that answered nothing stay empty."""
    assert normalize({"a": 0.0}) == {"a": 0.0}


def test_the_best_candidate_is_the_most_likely_one() -> None:
    """The candidate with the highest probability wins."""
    assert best({"a": 0.2, "b": 0.8}) == "b"


def test_a_tie_is_won_by_the_first_candidate() -> None:
    """The order of the candidates breaks a tie."""
    assert best({"a": 0.5, "b": 0.5}) == "a"


def test_the_confidence_measures_the_concentration() -> None:
    """One dominant candidate has a high confidence."""
    assert confidence({"a": 0.99, "b": 0.01}) > 0.9


def test_uniform_probabilities_have_no_confidence() -> None:
    """The uniform distribution does not favour a candidate."""
    assert confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)


def test_a_single_candidate_is_certain() -> None:
    """There is nothing to choose from."""
    assert confidence({"a": 1.0}) == 1.0


def test_the_score_is_the_weighted_average_of_the_levels() -> None:
    """The score is the average of the zero-based levels."""
    assert score({"low": 0.5, "high": 0.5}, ["low", "high"]) == pytest.approx(0.5)
    assert score({"low": 0.0, "high": 1.0}, ["low", "high"]) == pytest.approx(1.0)


def test_the_score_follows_the_order_of_the_levels() -> None:
    """The levels of the scale decide the order, not the order of the answers."""
    assert score({"high": 1.0, "low": 0.0}, ["low", "high"]) == pytest.approx(1.0)
    assert score({"low": 1.0, "high": 0.0}, ["low", "high"]) == pytest.approx(0.0)


def test_a_level_without_a_probability_counts_as_zero() -> None:
    """A level the model did not answer for does not change the score."""
    assert score({"low": 1.0}, ["low", "high"]) == pytest.approx(0.0)


def test_the_confidence_stays_between_zero_and_one() -> None:
    """The rounding of the probabilities cannot push it out of the range."""
    assert 0.0 <= confidence({"a": 0.3333, "b": 0.3333, "c": 0.3334}) <= 1.0


def test_the_candidates_may_be_formatted_as_json() -> None:
    """A template that needs a JSON array can ask for one."""
    prompt = build_prompt("Context: {state}\nChoices: {choices_json}", "a ticket", ["a", "b"])

    assert prompt == 'Context: a ticket\nChoices: ["a", "b"]'


def test_a_distribution_without_any_probability_has_no_confidence() -> None:
    """A candidate the model did not answer for says nothing."""
    assert confidence({"a": 0.0, "b": 0.0}) == 0.0


def test_a_state_may_contain_the_placeholders() -> None:
    """The state is inserted last, so its content is not replaced again."""
    prompt = build_prompt(TEMPLATE, "a {choices} ticket", ["billing"])

    assert prompt == "Context: a {choices} ticket\nChoices: ['billing']\nSelected Option:"
