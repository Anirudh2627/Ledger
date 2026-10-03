"""LLM provider abstraction: mock (offline) and OpenAI-compatible (HTTP)."""

from ledger.providers.mock import MockLLMProvider
from ledger.providers.openai_compat import (
    OpenAICompatEmbedder,
    OpenAICompatProvider,
    ProviderError,
)
from ledger.providers.registry import build_embedder, build_provider

__all__ = [
    "MockLLMProvider",
    "OpenAICompatEmbedder",
    "OpenAICompatProvider",
    "ProviderError",
    "build_embedder",
    "build_provider",
]
