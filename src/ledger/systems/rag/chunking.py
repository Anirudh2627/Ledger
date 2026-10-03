"""Corpus loading, sentence-aware chunking for the sample RAG system."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ledger.utils.logging import get_logger
from ledger.utils.text import split_sentences

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(frozen=True)
class Document:
    doc_id: str  # stable identifier (file name)
    text: str


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    heading: str = ""


def load_corpus(
    corpus_dir: str | Path, extensions: tuple[str, ...] = (".md", ".txt")
) -> list[Document]:
    """Load every matching file in ``corpus_dir`` as a document.

    Files are processed in sorted order so chunk ids - and therefore
    retrieval - are fully deterministic.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(doc_id=path.name, text=text))
    if not documents:
        raise ValueError(f"no documents with extensions {extensions} found in {root}")
    logger.info("loaded corpus: %d documents from %s", len(documents), root)
    return documents


def chunk_document(
    doc: Document,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split a document into overlapping, sentence-aligned chunks.

    The nearest preceding markdown heading is attached to each chunk (both as
    metadata and as a prefix inside the chunk text) because heading context
    materially improves retrieval and answer grounding.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Track headings and build (heading, sentence-unit) pairs. Markdown table
    # rows are kept as atomic units (they have no sentence boundaries); prose
    # lines are accumulated and split into sentences.
    units: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            units.extend((current_heading, s) for s in split_sentences(" ".join(buffer)))
            buffer.clear()

    for line in doc.text.splitlines():
        stripped = line.strip()
        heading_match = _HEADING_RE.match(stripped)
        if heading_match:
            flush_buffer()
            current_heading = heading_match.group(2).strip()
        elif stripped.startswith("|"):
            flush_buffer()
            if not re.fullmatch(r"\|[\s\-:|]+\|", stripped):  # skip separator rows
                units.append((current_heading, stripped))
        else:
            buffer.append(stripped)
    flush_buffer()

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_heading = units[0][0] if units else ""
    current_len = 0

    def flush() -> None:
        nonlocal current_len
        if not current_sentences:
            return
        prefix = f"{current_heading}.\n" if current_heading else ""
        text = prefix + "\n".join(current_sentences)
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_index=len(chunks),
                text=text.strip(),
                heading=current_heading,
            )
        )
        current_len = 0

    for heading, sentence in units:
        sentence = sentence.strip()
        if not sentence:
            continue
        if heading != current_heading and current_sentences:
            flush()
            current_sentences = []
            current_heading = heading
        if current_len + len(sentence) + 1 > chunk_size and current_sentences:
            # Keep overlap: retain trailing sentences up to chunk_overlap chars.
            overlap_sentences: list[str] = []
            overlap_len = 0
            for previous in reversed(current_sentences):
                if overlap_len + len(previous) + 1 > chunk_overlap:
                    break
                overlap_sentences.insert(0, previous)
                overlap_len += len(previous) + 1
            flush()
            current_sentences = overlap_sentences
            current_len = overlap_len
        current_sentences.append(sentence)
        current_len += len(sentence) + 1
    flush()

    if not chunks:  # document shorter than one chunk
        chunks = [Chunk(doc_id=doc.doc_id, chunk_index=0, text=doc.text)]
    return chunks


def chunk_corpus(
    documents: list[Document], *, chunk_size: int = 900, chunk_overlap: int = 120
) -> list[Chunk]:
    """Chunk every document, preserving document order."""
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    logger.debug("corpus chunked into %d chunks", len(chunks))
    return chunks
