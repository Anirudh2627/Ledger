"""Provider registry: build providers/embedders from configuration."""

from __future__ import annotations

from ledger.config.settings import EmbedderConfig, ProviderConfig
from ledger.core.interfaces import LLMProvider
from ledger.providers.mock import MockLLMProvider
from ledger.providers.openai_compat import OpenAICompatEmbedder, OpenAICompatProvider, ProviderError
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


def build_provider(config: ProviderConfig, *, seed: int = 1337) -> LLMProvider:
    """Instantiate the LLM provider described by ``config``."""
    if config.name == "mock":
        logger.info("using mock provider (model=%s, deterministic offline mode)", config.model)
        return MockLLMProvider(model=config.model, seed=seed)
    if config.name == "openai_compat":
        if not config.base_url:
            raise ProviderError("openai_compat provider requires base_url")
        logger.info(
            "using openai_compat provider (model=%s, base_url=%s)", config.model, config.base_url
        )
        return OpenAICompatProvider(config)
    raise ProviderError(f"unknown provider: {config.name}")  # pragma: no cover


def build_embedder(config: EmbedderConfig) -> object:
    """Instantiate an embedder from config.

    Returns either the offline hashing embedder or the HTTP embedder; both
    satisfy the ``Embedder`` protocol from :mod:`ledger.systems.rag.embeddings`.
    """
    # Imported lazily to keep providers independent from the RAG package.
    from ledger.systems.rag.embeddings import HashingTfIdfEmbedder

    if config.name == "hashing_tfidf":
        return HashingTfIdfEmbedder(dim=config.dim)
    if config.name == "openai_compat":
        return OpenAICompatEmbedder(config)
    raise ProviderError(f"unknown embedder: {config.name}")  # pragma: no cover


__all__ = [
    "MockLLMProvider",
    "OpenAICompatEmbedder",
    "OpenAICompatProvider",
    "ProviderError",
    "build_embedder",
    "build_provider",
]
