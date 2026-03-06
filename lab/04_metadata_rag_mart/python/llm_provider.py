#!/usr/bin/env python3
"""
llm_provider.py — Unified LLM interface for Lab 04

Supports multiple LLM providers behind a common interface:
  - OpenAILLMProvider:     OpenAI (gpt-5.2, gpt-5-mini)
  - AnthropicLLMProvider:  Anthropic (claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5)
  - FakeLLMProvider:       Deterministic strings (testing)

Usage:
    from llm_provider import get_llm_provider
    provider = get_llm_provider("claude-sonnet-4-6")
    response = provider.chat(system="You are an analyst.", user="Explain VaR.")
"""

from abc import ABC, abstractmethod

import config


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, system: str, user: str, max_tokens: int = 400,
             temperature: float = 0.1) -> str:
        """Send a chat completion request and return the response text."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""


class OpenAILLMProvider(LLMProvider):
    """OpenAI chat completions (gpt-5.2, gpt-5-mini)."""

    def __init__(self, model: str = "gpt-5.2"):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def chat(self, system: str, user: str, max_tokens: int = 400,
             temperature: float = 0.1) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model


class AnthropicLLMProvider(LLMProvider):
    """Anthropic chat completions (Claude models).

    Uses the Anthropic API format where system message is a top-level parameter.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self._model = model
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def chat(self, system: str, user: str, max_tokens: int = 400,
             temperature: float = 0.1) -> str:
        message = self._client.messages.create(
            model=self._model,
            system=system,
            messages=[
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return message.content[0].text

    @property
    def model_name(self) -> str:
        return self._model


class FakeLLMProvider(LLMProvider):
    """Deterministic string responses for API-free testing."""

    def chat(self, system: str, user: str, max_tokens: int = 400,
             temperature: float = 0.1) -> str:
        # Return a deterministic response that includes context from the user message
        user_preview = user[:80].replace("\n", " ")
        return (
            f"Based on the metadata search, the relevant tables and columns have been identified. "
            f"The data can be joined to answer the question. "
            f"(FakeLLM response for: {user_preview})"
        )

    @property
    def model_name(self) -> str:
        return "fake"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "openai": OpenAILLMProvider,
    "anthropic": AnthropicLLMProvider,
    "fake": FakeLLMProvider,
}


def get_llm_provider(model_name: str = None) -> LLMProvider:
    """Create an LLM provider for the given model name.

    Args:
        model_name: One of the keys in config.LLM_MODELS, or None for default.

    Returns:
        An LLMProvider instance.
    """
    if model_name is None:
        model_name = config.CHAT_MODEL

    if model_name == "fake":
        return FakeLLMProvider()

    model_info = config.LLM_MODELS.get(model_name)
    if model_info is None:
        raise ValueError(
            f"Unknown LLM model '{model_name}'. "
            f"Available: {list(config.LLM_MODELS.keys())}"
        )

    provider_name = model_info["provider"]
    provider_cls = _PROVIDER_MAP.get(provider_name)
    if provider_cls is None:
        raise ValueError(f"Unknown provider '{provider_name}' for model '{model_name}'")

    if provider_cls == FakeLLMProvider:
        return provider_cls()
    return provider_cls(model=model_name)
