"""Exception hierarchy for the AirLLM Ollama wrapper."""

from __future__ import annotations


class AirLLMError(Exception):
    """Base class for every error raised by this package."""


class ModelMismatchError(AirLLMError):
    """Raised when a client requests a model this server does not serve."""


class ModelInitializationError(AirLLMError):
    """Raised when the AirLLM model cannot be loaded or is unavailable."""


class GenerationError(AirLLMError):
    """Raised when a generation fails because of the model or the checkpoint.

    The checkpoint can be broken in a way that only shows up while the model
    generates — a missing layer shard, a corrupt weight file — so the failure
    is reported like a model that cannot be loaded.
    """
