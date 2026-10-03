"""Sample RAG system under test - a reference implementation, deliberately small.

The RAG system exists to demonstrate Ledger, not to be a production RAG
framework: documents -> chunking -> embeddings -> retrieval -> versioned
prompt -> LLM generation. Every stage is injected/replaceable, and every
knob that a real team would tune (chunk size, overlap, top-k, prompt version,
model, embedder) is config-driven - which is precisely what makes it useful
as a regression-testing target.
"""

from __future__ import annotations

import time
from typing import Any

from ledger.config.settings import RagSystemConfig
from ledger.core.interfaces import LLMProvider, SystemUnderTest
from ledger.core.models import CompletionRequest, RetrievedChunk, SystemOutput
from ledger.dataset.schema import TestCase
from ledger.providers.openai_compat import OpenAICompatEmbedder
from ledger.systems.base import register_system
from ledger.systems.rag.chunking import chunk_corpus, load_corpus
from ledger.systems.rag.embeddings import Embedder, HashingTfIdfEmbedder
from ledger.systems.rag.prompts import get_prompt_template
from ledger.systems.rag.retrieval import VectorIndex
from ledger.utils.logging import get_logger

logger = get_logger(__name__)


def build_embedder_from_config(config: RagSystemConfig) -> Embedder:
    """Instantiate the configured embedder (offline hashing or HTTP)."""
    if config.embedder.name == "hashing_tfidf":
        return HashingTfIdfEmbedder(dim=config.embedder.dim)
    return OpenAICompatEmbedder(config.embedder)  # type: ignore[return-value]


@register_system("rag")
class RAGSystem(SystemUnderTest):
    """Retrieve-then-generate system under test."""

    def __init__(
        self,
        config: RagSystemConfig,
        provider: LLMProvider,
        *,
        embedder: Embedder | None = None,
        version: str = "rag",
        seed: int = 1337,
    ) -> None:
        self.config = config
        self.provider = provider
        self.version = version
        self.seed = seed
        self.prompt_template = get_prompt_template(config.prompt_version)
        self.embedder = embedder or build_embedder_from_config(config)

        started = time.perf_counter()
        documents = load_corpus(config.corpus_dir, tuple(config.file_extensions))
        chunks = chunk_corpus(
            documents, chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )
        self.index = VectorIndex(chunks, self.embedder)
        self._build_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "RAG index ready: %d docs, %d chunks in %.0fms (embedder=%s, top_k=%d, prompt=%s)",
            len(documents),
            len(chunks),
            self._build_ms,
            type(self.embedder).__name__,
            config.top_k,
            config.prompt_version,
        )

    # -- SystemUnderTest API ---------------------------------------------------

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a question (exposed for retrieval metrics)."""
        return self.index.search(question, self.config.top_k)

    def generate(self, test_case: TestCase) -> SystemOutput:
        chunks = self.retrieve(test_case.question)
        messages = self.prompt_template.build(
            test_case.question,
            chunks,
            system_instruction=self.config.system_instruction,
            max_context_chars=self.config.max_context_chars,
        )
        request = CompletionRequest(
            messages=messages,
            purpose="generation",
            temperature=self.provider_temperature(),
            seed=self.seed,
        )
        started = time.perf_counter()
        response = self.provider.complete(request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return SystemOutput(
            answer=response.text.strip(),
            retrieved_context=chunks,
            model_name=response.model or self.provider_name(),
            prompt_version=self.config.prompt_version,
            latency_ms=round(latency_ms + (response.latency_ms or 0.0), 2),
            metadata={
                "top_k": self.config.top_k,
                "chunk_size": self.config.chunk_size,
                "n_chunks_retrieved": len(chunks),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "rag",
            "version": self.version,
            "corpus_dir": self.config.corpus_dir,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "top_k": self.config.top_k,
            "prompt_version": self.config.prompt_version,
            "embedder": type(self.embedder).__name__,
            "provider": self.provider_name(),
            "index_size": self.index.size,
            "index_build_ms": round(self._build_ms, 1),
        }

    # -- helpers ----------------------------------------------------------------

    def provider_name(self) -> str:
        return getattr(self.provider, "name", "unknown")

    def provider_temperature(self) -> float | None:
        return getattr(getattr(self.provider, "_config", None), "temperature", None)
