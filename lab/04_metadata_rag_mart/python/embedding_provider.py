#!/usr/bin/env python3
"""
embedding_provider.py — Unified embedding interface for Lab 04

Supports multiple embedding providers behind a common interface:
  - VoyageEmbeddingProvider: Voyage AI (voyage-finance-2)
  - OpenAIEmbeddingProvider: OpenAI (text-embedding-3-small/large)
  - OllamaEmbeddingProvider: Ollama self-hosted (mxbai-embed-large)
  - FakeEmbeddingProvider:   Deterministic random vectors (testing)

Usage:
    from embedding_provider import get_embedding_provider
    provider = get_embedding_provider("voyage-finance-2")
    vectors = provider.embed_texts(["hello world"])
    query_vec = provider.embed_query("search query")
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List

import config


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """Embed a batch of texts. input_type is 'document' or 'query'."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text (convenience wrapper)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embeddings (voyage-finance-2)."""

    def __init__(self, model: str = "voyage-finance-2"):
        import voyageai
        self._model = model
        self._client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    def embed_texts(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        result = self._client.embed(texts, model=self._model, input_type=input_type)
        return result.embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text], input_type="query")[0]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return config.EMBEDDING_MODELS.get(self._model, {}).get("dimension", 1024)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings (text-embedding-3-small/large) with dimension reduction to 1024."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self._model = model
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def embed_texts(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        resp = self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=1024,
        )
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text], input_type="query")[0]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return 1024


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama self-hosted embeddings (mxbai-embed-large)."""

    def __init__(self, model: str = None):
        import ollama as _ollama
        self._ollama = _ollama
        self._model = model or config.OLLAMA_EMBED_MODEL
        self._endpoint = config.OLLAMA_ENDPOINT

    def embed_texts(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        try:
            client = self._ollama.Client(host=self._endpoint)
            resp = client.embed(model=self._model, input=texts)
            embeddings = resp.get("embeddings", resp.get("embedding", []))
            # Truncate or pad to 1024 dimensions
            result = []
            for emb in embeddings:
                if len(emb) > 1024:
                    emb = emb[:1024]
                elif len(emb) < 1024:
                    emb = emb + [0.0] * (1024 - len(emb))
                result.append(emb)
            return result
        except ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self._endpoint}. "
                f"Is the Ollama service running? Error: {e}"
            )

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text], input_type="query")[0]

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return 1024


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic random vectors for API-free testing."""

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _make_vector(self) -> List[float]:
        vec = [self._rng.gauss(0, 1) for _ in range(1024)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec]

    def embed_texts(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        return [self._make_vector() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._make_vector()

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimension(self) -> int:
        return 1024


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "voyage": VoyageEmbeddingProvider,
    "openai": OpenAIEmbeddingProvider,
    "ollama": OllamaEmbeddingProvider,
    "fake": FakeEmbeddingProvider,
}


def get_embedding_provider(model_name: str = None) -> EmbeddingProvider:
    """Create an embedding provider for the given model name.

    Args:
        model_name: One of the keys in config.EMBEDDING_MODELS, or None for default.

    Returns:
        An EmbeddingProvider instance.
    """
    if model_name is None:
        model_name = config.EMBEDDING_MODEL

    if model_name == "fake":
        return FakeEmbeddingProvider()

    model_info = config.EMBEDDING_MODELS.get(model_name)
    if model_info is None:
        raise ValueError(
            f"Unknown embedding model '{model_name}'. "
            f"Available: {list(config.EMBEDDING_MODELS.keys())}"
        )

    provider_name = model_info["provider"]
    provider_cls = _PROVIDER_MAP.get(provider_name)
    if provider_cls is None:
        raise ValueError(f"Unknown provider '{provider_name}' for model '{model_name}'")

    if provider_cls == FakeEmbeddingProvider:
        return provider_cls()
    return provider_cls(model=model_name)
