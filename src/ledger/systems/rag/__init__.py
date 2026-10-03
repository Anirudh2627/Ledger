"""Sample RAG system under test (reference implementation for Ledger)."""

from ledger.systems.rag.chunking import Chunk, Document, chunk_corpus, chunk_document, load_corpus
from ledger.systems.rag.embeddings import Embedder, HashingTfIdfEmbedder
from ledger.systems.rag.prompts import available_prompt_versions, get_prompt_template
from ledger.systems.rag.retrieval import VectorIndex
from ledger.systems.rag.system import RAGSystem

__all__ = [
    "Chunk",
    "Document",
    "Embedder",
    "HashingTfIdfEmbedder",
    "RAGSystem",
    "VectorIndex",
    "available_prompt_versions",
    "chunk_corpus",
    "chunk_document",
    "get_prompt_template",
    "load_corpus",
]
