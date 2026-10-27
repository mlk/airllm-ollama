"""The decision models: questions about a state, answered with probabilities.

Ollama answers typed questions about a piece of text with a decision model
(``POST /v1/systemone``): a *choice* between labelled options, a *noul*
(yes or no) or a *score* on a scale. The model is asked once — a forward pass
without the autoregressive loop — and the probabilities of the candidates are
read from the logits of the last position.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

# The question types Ollama knows.
CHOICE = "choice"
NOUL = "noul"
SCORE = "score"

# The two candidates of a yes/no question.
FALSE = "false"
TRUE = "true"


def build_prompt(template: str, state: str, candidates: Sequence[str]) -> str:
    r"""Build the prompt of a question.

    Args:
        template: The prompt template of the model, with the placeholders
            ``{state}``, ``{choices}`` and ``{choices_json}``. The newlines are
            written as ``\n``.
        state: The text the question is about.
        candidates: The candidates the model chooses from.

    Returns:
        The prompt of the question.
    """
    prompt = template.replace("\\n", "\n")
    prompt = prompt.replace("{state}", state)
    prompt = prompt.replace("{choices_json}", json.dumps(list(candidates)))
    return prompt.replace("{choices}", str(list(candidates)))


def normalize(probabilities: Mapping[str, float]) -> dict[str, float]:
    """Normalize the probabilities of the candidates.

    Ollama reports the probabilities normalized over the candidates, so that
    they sum to one.

    Args:
        probabilities: The probability of every candidate.

    Returns:
        The normalized probabilities.
    """
    total = sum(probabilities.values())
    if total <= 0:
        return dict(probabilities)
    return {name: value / total for name, value in probabilities.items()}


def best(probabilities: Mapping[str, float]) -> str:
    """Return the candidate with the highest probability.

    A tie is won by the candidate that comes first.

    Args:
        probabilities: The probability of every candidate.

    Returns:
        The name of the candidate.
    """
    return max(probabilities, key=lambda name: probabilities[name])


def confidence(probabilities: Mapping[str, float]) -> float:
    """Return how strongly the probabilities favour one candidate.

    The concentration of the distribution: ``1 - H(p) / ln(N)``, where ``H(p)``
    is its entropy and ``N`` the number of candidates. Zero means uniform
    probabilities, values near one mean that one candidate dominates. It is not
    a calibrated correctness.

    Args:
        probabilities: The probability of every candidate.

    Returns:
        The confidence of the answer.
    """
    values = list(probabilities.values())
    if len(values) < 2:
        return 1.0
    entropy = -sum(value * math.log(value) for value in values if value > 0)
    concentration = 1.0 - entropy / math.log(len(values))
    # The rounding of the probabilities can push the value out of the range.
    return max(0.0, min(1.0, concentration))


def score(probabilities: Mapping[str, float], levels: Sequence[str]) -> float:
    """Return the probability weighted average of the levels.

    The levels are numbered from zero, so the answer is a number between zero
    and the number of levels minus one.

    Args:
        probabilities: The probability of every level.
        levels: The levels of the scale, from the lowest to the highest.

    Returns:
        The score of the answer.
    """
    return sum(index * probabilities.get(level, 0.0) for index, level in enumerate(levels))
