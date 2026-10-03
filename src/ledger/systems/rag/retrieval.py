"""Vector retrieval index (cosine similarity over numpy)."""

from __future__ import annotations

import numpy as np

from ledger.core.models import RetrievedChunk
from ledger.systems.rag.chunking import Chunk
from ledger.systems.rag.embeddings import Embedder


class VectorIndex:
    """Brute-force cosine top-k index.

    At golden-dataset scale (hundreds to low thousands of chunks) brute force
    is exact, dependency-free and fast; an ANN index would be premature
    complexity here. The embedder is injected so the same index serves both
    the offline hashing embedder and HTTP embedding APIs.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self._chunks = list(chunks)
        self._embedder = embedder
        texts = [c.text for c in self._chunks]
        embedder.fit(texts)
        self._matrix = embedder.encode(texts)
        if self._matrix.shape[0] != len(self._chunks):
            raise RuntimeError("embedder returned wrong number of vectors")

    @property
    def size(self) -> int:
        return len(self._chunks)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Return the top-k chunks for ``query`` (cosine similarity).

        Ties break deterministically by chunk position.
        """
        if not self._chunks:
            return []
        q = self._embedder.encode([query])[0]
        scores = self._matrix @ q
        k = min(k, len(self._chunks))
        # lexsort: primary key = -score, secondary = index (stable, deterministic)
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [
            RetrievedChunk(
                doc_id=self._chunks[i].doc_id,
                chunk_index=self._chunks[i].chunk_index,
                text=self._chunks[i].text,
                score=float(scores[i]),
            )
            for i in order
        ]
