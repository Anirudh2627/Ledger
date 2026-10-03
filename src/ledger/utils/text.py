"""Text normalization and lightweight lexical utilities.

All functions are deterministic and dependency-free so that metrics computed
from them are reproducible across runs and platforms.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import unicodedata
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

#: Common English stopwords used only for *lexical overlap* heuristics.
STOPWORDS = frozenset(
    """a an and are as at be by can could do does for from had has have how i if in into is
    it its me my no not of on or our so than that the their them then there these they this
    to was we were what when which who why will with you your please tell""".split()
)


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace and punctuation runs."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    """Tokenize into lowercase alphanumeric words."""
    tokens = _WORD_RE.findall(normalize_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a conservative regex.

    Intentionally simple and deterministic; not an NLP-grade sentence splitter.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between two strings (SQuAD-style), in ``[0, 1]``."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity of content-token sets, in ``[0, 1]``."""
    set_a = set(tokenize(a, remove_stopwords=True))
    set_b = set(tokenize(b, remove_stopwords=True))
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def sequence_similarity(a: str, b: str) -> float:
    """Normalized longest-common-subsequence style similarity via difflib."""
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def truncate(text: str, max_chars: int, *, suffix: str = " [...]") -> str:
    """Truncate text for reports/logs without breaking mid-word if avoidable."""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - len(suffix)].rstrip()
    return cut + suffix


def sha256_text(text: str) -> str:
    """Hex SHA-256 digest of a string (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike) -> str:
    """Hex SHA-256 digest of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()
