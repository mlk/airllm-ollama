"""Doubles for the AirLLM runtime, shared by the test-suite.

They mimic the parts of AirLLM the engine talks to: a tokenizer and a model
with a ``generate`` method.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any, cast

TOKEN_IDS = [11, 22, 33]


class FakeInputs(dict[str, object]):
    """The tokenized inputs of the fake tokenizer."""

    def to(self, device: str) -> FakeInputs:
        """Pretend to move the inputs to a device."""
        return self


class FakeLogits:
    """The logits of a fake forward pass."""

    def __init__(self, probabilities: dict[int, float] | None = None) -> None:
        """Create the logits of the fake pass.

        Args:
            probabilities: The probability of every token.
        """
        self.probabilities = probabilities or {}

    def __getitem__(self, index: object) -> FakeLogits:
        """Return the last position, or the probability of a single token."""
        if isinstance(index, tuple):
            return self
        token = index if isinstance(index, int) else 0
        return FakeLogits({token: self.probabilities.get(token, 0.0)})

    def softmax(self, dim: int) -> FakeLogits:
        """Return the softmax of the logits."""
        return self

    def __float__(self) -> float:
        """Return the probability of the token."""
        return next(iter(self.probabilities.values()), 0.0)


class FakeOutput:
    """The output of a fake forward pass."""

    def __init__(self, probabilities: dict[int, float] | None = None) -> None:
        """Create the output of the fake pass.

        Args:
            probabilities: The probability of every token.
        """
        self.logits = FakeLogits(probabilities or {})


class FakeTokenizer:
    """Tokenizer double recording the prompts it was asked to tokenize."""

    def __init__(self, template: str | None = None) -> None:
        """Create a tokenizer, optionally with a chat template."""
        self.prompts: list[str] = []
        self.tokenizations: list[dict[str, object]] = []
        self.decoded: list[object] = []
        self.template = template
        self.renderings: list[list[dict[str, object]]] = []
        self.template_kwargs: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> FakeInputs:
        """Return fake token ids for ``text``."""
        self.prompts.append(text)
        self.tokenizations.append(dict(kwargs))
        ids = self.token_ids(text, **kwargs)
        return FakeInputs({"input_ids": ids, "attention_mask": [1] * len(ids)})

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
        rows = sequences if isinstance(sequences, list) else [sequences]
        return [f"decoded:{list(row) if isinstance(row, list) else row}" for row in rows]

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        """Render the messages with the chat template of the tokenizer."""
        if self.template is None:
            raise ValueError("this model has no chat template")
        if isinstance(messages, list):
            self.renderings.append(
                [dict(message) for message in messages if isinstance(message, dict)]
            )
        self.template_kwargs.append(dict(kwargs))
        return self.template


class EchoTokenizer(FakeTokenizer):
    """Tokenizer double whose template echoes the messages and the tool names.

    Used when a test needs the template to preserve the tool exchange — a
    template that stays silent about tool calls or tool results is treated
    as dropping them.
    """

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        """Render the messages by echoing their contents."""
        if self.template is None:
            raise ValueError("this model has no chat template")
        if isinstance(messages, list):
            self.renderings.append(
                [dict(message) for message in messages if isinstance(message, dict)]
            )
        message_list: list[dict[str, Any]] = [
            message
            for message in (messages if isinstance(messages, list) else [])
            if isinstance(message, dict)
        ]
        tools = kwargs.get("tools")
        tool_list: list[dict[str, Any]] = [
            tool for tool in (tools if isinstance(tools, list) else []) if isinstance(tool, dict)
        ]
        echoed = [
            str(message.get("content", ""))
            + "".join(
                str(call.get("function", {}).get("arguments", ""))
                for call in message.get("tool_calls", [])
                if isinstance(call, dict)
            )
            + str(message.get("tool_name", ""))
            for message in message_list
        ]
        echoed.extend(str(tool.get("function", {}).get("name", "")) for tool in tool_list)
        self.template_kwargs.append(dict(kwargs))
        return "<​|im_start|>assistant\n" + "\n".join(echoed)


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


class SystemRoleTokenizer(FakeTokenizer):
    """Tokenizer double whose template refuses the system role.

    Templates without support for the system role (Gemma, older Mistral
    versions) raise — the rendering has to ask again with the system message
    merged into the first user turn.
    """

    def apply_chat_template(self, messages: object, **kwargs: object) -> str:
        """Refuse the messages when they carry a system message."""
        if isinstance(messages, list) and any(
            isinstance(message, dict) and message.get("role") == "system" for message in messages
        ):
            raise ValueError("this template does not support the system role")
        return super().apply_chat_template(messages, **kwargs)


class FakeModel:
    """AirLLM model double recording how it was called."""

    def __init__(self, output: object = None, chunks: list[str] | None = None) -> None:
        """Create a model returning ``output`` (prompt ids + generated ids).

        The default output is the prompt tokens followed by fresh tokens,
        the way a real backend returns the whole sequence.
        """
        self.tokenizer = FakeTokenizer()
        self.output = output
        self.chunks = chunks or []
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.forwards: list[dict[str, object]] = []

    def generate(self, *args: object, **kwargs: object) -> object:
        """Record the call, stream the chunks and return the canned output."""
        self.calls.append((args, kwargs))
        streamer = kwargs.get("streamer")
        if streamer is not None:
            put = cast("Callable[[str], None] | None", getattr(streamer, "put", None))
            if put is not None:
                for chunk in self.chunks:
                    put(chunk)
        if self.output is not None:
            return self.output
        first = args[0] if args else None
        if isinstance(first, dict):
            prompt = list(first.get("input_ids", []))
        elif isinstance(first, (list, tuple)):
            prompt = list(first)
        else:
            prompt = list(TOKEN_IDS)
        return [*prompt, *TOKEN_IDS]

    def forward(self, *args: object, **kwargs: object) -> object:
        """Record the forward pass and return the canned output."""
        self.forwards.append(dict(kwargs))
        return self.output


class AbortableModel(FakeModel):
    """Model double that streams a chunk and waits for the abort signal.

    It mimics the generation loop of ``transformers``: the model produces a
    token, checks the stopping criteria and produces the next one. The double
    stops as soon as the criterion is aborted, the way a real generation does.
    """

    def generate(self, *args: object, **kwargs: object) -> object:
        """Stream a chunk and wait until the generation is aborted."""
        self.calls.append((args, kwargs))
        streamer = kwargs.get("streamer")
        put = cast("Callable[[str], None] | None", getattr(streamer, "put", None))
        if put is not None:
            put("hello")
        for criterion in cast("list[object]", kwargs.get("stopping_criteria") or []):
            abort = getattr(criterion, "abort", None)
            if abort is not None:
                cast("threading.Event", abort).wait(timeout=5)
        return self.output


class BrokenModel(FakeModel):
    """Model double whose generation fails, like a broken checkpoint does."""

    def generate(self, *args: object, **kwargs: object) -> object:
        """Record the call and fail."""
        self.calls.append((args, kwargs))
        raise ValueError("[load_npz] Input must be a zip file")

    def forward(self, *args: object, **kwargs: object) -> object:
        """Record the forward pass and fail."""
        self.forwards.append(dict(kwargs))
        raise ValueError("[load_npz] Input must be a zip file")


class FailingStream:
    """An iterator that fails as soon as it is read.

    It stands in for a generation that fails while the model is producing the
    answer — a broken checkpoint, for example.
    """

    def __init__(self, error: Exception) -> None:
        """Create the stream.

        Args:
            error: The error the stream raises.
        """
        self._error = error

    def __iter__(self) -> Iterator[object]:
        """Return the iterator that fails."""
        return self

    def __next__(self) -> object:
        """Raise the error of the generation."""
        raise self._error


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
