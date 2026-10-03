"""Embedding backends for the sample RAG system.

Two implementations of the same tiny ``Embedder`` protocol:

* :class:`HashingTfIdfEmbedder` - fully offline, deterministic hashed
  bag-of-n-grams with corpus-level IDF weighting. No model downloads, no
  network, bit-stable across machines. Lexical, not semantic: it is a
  deliberate baseline that keeps the *evaluation framework* the star of the
  project while still exercising the real retrieve->generate loop.
* The OpenAI-compatible HTTP embedder lives in
  :mod:`ledger.providers.openai_compat` and can be selected via config for
  semantic retrieval.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np

from ledger.utils.text import STOPWORDS, tokenize


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps text to fixed-size vectors."""

    dim: int

    def fit(self, texts: list[str]) -> None:
        """Optionally adapt to a corpus (no-op for stateless embedders)."""
        ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float matrix (rows should be L2-normalized)."""
        ...


def _token_ngrams(text: str) -> list[str]:
    """Content words, word bigrams, and character 3-4 grams.

    Stopwords are dropped before feature construction: they would otherwise
    form rare bigrams ("does_helios") whose high IDF weights inject noise
    into both queries and documents. Word bigrams give the lexical embedder
    phrase-level discrimination ("single sign" matches SSO text even when
    individual words are common).
    """
    words = [w for w in tokenize(text) if w not in STOPWORDS]
    features: list[str] = list(words)
    features.extend(f"{a}_{b}" for a, b in pairwise(words))
    for word in words:
        if len(word) >= 4:
            features.extend(word[i : i + 3] for i in range(len(word) - 2))
            features.extend(word[i : i + 4] for i in range(len(word) - 3))
    return features


class HashingTfIdfEmbedder:
    """Deterministic hashed TF-IDF embedder (the hashing trick)."""

    def __init__(self, dim: int = 512) -> None:
        if dim < 32:
            raise ValueError("dim must be >= 32")
        self.dim = dim
        self._idf: dict[str, float] | None = None

    # -- Embedder protocol ----------------------------------------------------

    def fit(self, texts: list[str]) -> None:
        """Compute IDF weights over the (chunk) corpus."""
        doc_freq: Counter[str] = Counter()
        for text in texts:
            doc_freq.update(set(_token_ngrams(text)))
        n_docs = max(len(texts), 1)
        self._idf = {
            feature: math.log((1.0 + n_docs) / (1.0 + count)) + 1.0
            for feature, count in doc_freq.items()
        }

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            counts = Counter(_token_ngrams(text))
            for feature, count in counts.items():
                bucket = self._bucket(feature)
                sign = self._sign(feature)
                weight = count * self._idf_weight(feature)
                matrix[row, bucket] += sign * weight
            norm = np.linalg.norm(matrix[row])
            if norm > 0:
                matrix[row] /= norm
        return matrix

    # -- internals -------------------------------------------------------------

    def _digest(self, feature: str) -> bytes:
        return hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()

    def _bucket(self, feature: str) -> int:
        return int.from_bytes(self._digest(feature)[:8], "little") % self.dim

    def _sign(self, feature: str) -> float:
        return 1.0 if self._digest(feature)[8] % 2 == 0 else -1.0

    def _idf_weight(self, feature: str) -> float:
        if self._idf is None:
            return 1.0
        return self._idf.get(feature, math.log(2.0) + 1.0)
