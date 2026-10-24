"""Doubles for the AirLLM runtime, shared by the test-suite.

They mimic the parts of AirLLM the engine talks to: a tokenizer and a model
with a ``generate`` method.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import cast

TOKEN_IDS = [11, 22, 33]


class FakeTokenizer:
    """Tokenizer double recording the prompts it was asked to tokenize."""

    def __init__(self, template: str | None = None) -> None:
        """Create a tokenizer, optionally with a chat template."""
        self.prompts: list[str] = []
        self.tokenizations: list[dict[str, object]] = []
        self.decoded: list[object] = []
        self.template = template
        self.renderings: list[list[dict[str, str]]] = []

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, object]:
        """Return fake token ids for ``text``."""
        self.prompts.append(text)
        self.tokenizations.append(dict(kwargs))
        ids = self.token_ids(text, **kwargs)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def token_ids(self, text: str, **kwargs: object) -> list[int]:
        """Return the token ids of ``text``.

        Args:
            text: The text to tokenize.
            **kwargs: The options of the tokenization.

        Returns:
            The token ids of the text.
        """
        return list(TOKEN_IDS)

    def batch_decode(self, sequences: object, **kwargs: object) -> list[str]:
        """Return a deterministic decoding of ``sequences``."""
        self.decoded.append(sequences)
        return [f"decoded:{sequences}"]

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        """Render the messages with the chat template of the tokenizer."""
        if self.template is None:
            raise ValueError("this model has no chat template")
        if isinstance(messages, list):
            self.renderings.append(
                [
                    {
                        "role": str(message.get("role", "")),
                        "content": str(message.get("content", "")),
                    }
                    for message in messages
                    if isinstance(message, dict)
                ]
            )
        return self.template


class BosTokenizer(FakeTokenizer):
    """Tokenizer double that adds a BOS token, like the real ones do.

    A chat template renders the BOS token of the model itself, so the double
    counts the BOS tokens of the text and adds another one when the tokenizer
    is allowed to add the special tokens.
    """

    BOS = 1
    BOS_TEXT = "<|begin_of_text|>"

    def token_ids(self, text: str, **kwargs: object) -> list[int]:
        """Return the token ids of ``text`` with the BOS token of the text.

        Args:
            text: The text to tokenize.
            **kwargs: The options of the tokenization.

        Returns:
            The token ids of the text.
        """
        ids = [self.BOS] * text.count(self.BOS_TEXT)
        if kwargs.get("add_special_tokens") is not False:
            ids.append(self.BOS)
        ids.extend(TOKEN_IDS)
        return ids


class FakeModel:
    """AirLLM model double recording how it was called."""

    def __init__(self, output: object = None, chunks: list[str] | None = None) -> None:
        """Create a model returning ``output`` (token ids by default).

        Args:
            output: The value the generation returns.
            chunks: The text pieces the model streams to a streamer, if any.
        """
        self.tokenizer = FakeTokenizer()
        self.output = list(TOKEN_IDS) if output is None else output
        self.chunks = chunks or []
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def generate(self, *args: object, **kwargs: object) -> object:
        """Record the call, stream the chunks and return the canned output."""
        self.calls.append((args, kwargs))
        streamer = kwargs.get("streamer")
        if streamer is not None:
            put = cast("Callable[[str], None] | None", getattr(streamer, "put", None))
            if put is not None:
                for chunk in self.chunks:
                    put(chunk)
        return self.output


class RecordingFactory:
    """Model factory double counting how often the model was created."""

    def __init__(self, model: FakeModel | None = None) -> None:
        """Create a factory returning ``model``."""
        self.model = model or FakeModel()
        self.calls: list[object] = []

    def __call__(self, settings: object) -> FakeModel:
        """Record the settings and return the canned model."""
        self.calls.append(settings)
        return self.model


class FakeFactory:
    """Model factory that always returns the same fake model."""

    def __init__(self, output: object = None) -> None:
        """Create a factory with a canned model."""
        self.model = FakeModel(output)
        self.calls: list[object] = []

    def __call__(self, settings: object) -> FakeModel:
        """Record the settings and return the canned model."""
        self.calls.append(settings)
        return self.model
