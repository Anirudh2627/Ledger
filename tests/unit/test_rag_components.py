"""Unit tests: sample RAG system components (chunking, embeddings, retrieval, prompts)."""

from __future__ import annotations

import numpy as np
import pytest

from ledger.config.settings import RagSystemConfig
from ledger.providers.mock import MockLLMProvider
from ledger.systems.rag.chunking import Chunk, Document, chunk_document, load_corpus
from ledger.systems.rag.embeddings import HashingTfIdfEmbedder
from ledger.systems.rag.prompts import available_prompt_versions, get_prompt_template
from ledger.systems.rag.retrieval import VectorIndex
from ledger.systems.rag.system import RAGSystem
from tests.helpers import make_case

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# corpus loading + chunking
# ---------------------------------------------------------------------------


def test_load_corpus_sorted_and_filtered(mini_corpus_dir) -> None:
    docs = load_corpus(mini_corpus_dir)
    assert [d.doc_id for d in docs] == ["gizmo-pricing.md", "gizmo-support.md"]


def test_load_corpus_missing_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


def test_load_corpus_empty_dir(tmp_path) -> None:
    (tmp_path / "corpus").mkdir()
    with pytest.raises(ValueError, match="no documents"):
        load_corpus(tmp_path / "corpus")


def test_chunk_short_document_single_chunk() -> None:
    doc = Document(doc_id="d.md", text="# Title\n\nOne short sentence.")
    chunks = chunk_document(doc, chunk_size=900, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].heading == "Title"
    assert chunks[0].text.startswith("Title.")


def test_chunk_respects_size_budget() -> None:
    text = "# H\n\n" + " ".join(f"Sentence number {i} is here." for i in range(80))
    doc = Document(doc_id="d.md", text=text)
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    longest_sentence = max(len(s) for s in text.split(". ")) + 1
    for chunk in chunks:
        # a chunk may overshoot by at most one sentence (sentences are atomic)
        assert len(chunk.text) <= 300 + longest_sentence + len("H. ")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_overlap_repeats_tail_sentences() -> None:
    sentences = [f"Unique sentence alpha{i} beta." for i in range(30)]
    doc = Document(doc_id="d.md", text="# H\n\n" + " ".join(sentences))
    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=80)
    assert len(chunks) >= 2
    first_tail = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_tail & second_words  # overlap shares vocabulary
    # a specific sentence from the end of chunk 0 reappears in chunk 1
    shared = [s for s in sentences if s in chunks[0].text and s in chunks[1].text]
    assert shared, "expected overlap sentences shared between consecutive chunks"


def test_chunk_table_rows_are_atomic() -> None:
    text = (
        "# Plans\n\n"
        "| Plan | Price |\n"
        "|---|---|\n"
        "| Starter | $49 |\n"
        "| Growth | $299 |\n\n"
        "Prose after the table.\n"
    )
    doc = Document(doc_id="d.md", text=text)
    chunks = chunk_document(doc, chunk_size=900, chunk_overlap=0)
    joined = "\n".join(c.text for c in chunks)
    assert "| Starter | $49 |" in joined  # row kept intact
    assert "| Growth | $299 |" in joined
    assert "|---|---|" not in joined  # separator rows dropped


def test_chunk_overlap_validation() -> None:
    doc = Document(doc_id="d.md", text="text")
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size=100, chunk_overlap=100)


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------


def test_hashing_embedder_shape_and_normalization() -> None:
    emb = HashingTfIdfEmbedder(dim=128)
    emb.fit(["hello world", "another document about tests"])
    matrix = emb.encode(["hello world", "something else entirely"])
    assert matrix.shape == (2, 128)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0)


def test_hashing_embedder_deterministic() -> None:
    texts = ["alpha beta gamma", "delta epsilon"]
    a = HashingTfIdfEmbedder(dim=64)
    b = HashingTfIdfEmbedder(dim=64)
    a.fit(texts)
    b.fit(texts)
    assert np.array_equal(a.encode(texts), b.encode(texts))


def test_hashing_embedder_similar_texts_closer() -> None:
    corpus = [
        "the growth plan includes 800 compute credits",
        "the starter plan includes 100 compute credits",
        "webhooks are signed with hmac sha256 signatures",
    ]
    emb = HashingTfIdfEmbedder(dim=256)
    emb.fit(corpus)
    vectors = emb.encode(corpus)
    q = emb.encode(["how many compute credits does the growth plan include"])[0]
    sims = vectors @ q
    assert float(np.argmax(sims)) == 0.0  # growth-plan chunk is closest


def test_hashing_embedder_min_dim() -> None:
    with pytest.raises(ValueError):
        HashingTfIdfEmbedder(dim=8)


# ---------------------------------------------------------------------------
# retrieval index
# ---------------------------------------------------------------------------


def _index() -> VectorIndex:
    chunks = [
        Chunk(doc_id="a.md", chunk_index=0, text="alpha beta gamma delta"),
        Chunk(doc_id="b.md", chunk_index=0, text="epsilon zeta eta theta"),
        Chunk(doc_id="c.md", chunk_index=0, text="gamma delta omega sigma"),
    ]
    emb = HashingTfIdfEmbedder(dim=128)
    return VectorIndex(chunks, emb)


def test_index_search_top_k_and_ordering() -> None:
    index = _index()
    results = index.search("alpha beta gamma", k=2)
    assert len(results) == 2
    assert results[0].doc_id == "a.md"
    assert all(r.score is not None for r in results)
    scores = [float(r.score or 0.0) for r in results]
    assert scores == sorted(scores, reverse=True)


def test_index_search_k_larger_than_size() -> None:
    index = _index()
    results = index.search("gamma", k=99)
    assert len(results) == 3


def test_index_deterministic_ties() -> None:
    index = _index()
    first = index.search("completely unrelated tokens xyz", k=3)
    second = index.search("completely unrelated tokens xyz", k=3)
    assert [r.doc_id for r in first] == [r.doc_id for r in second]


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def test_prompt_versions_registered() -> None:
    versions = available_prompt_versions()
    assert {"v0_sloppy", "v1_grounding", "v2_concise_cited"} <= set(versions)
    with pytest.raises(KeyError):
        get_prompt_template("v999_nope")


def test_prompt_build_structure() -> None:
    template = get_prompt_template("v1_grounding")
    from tests.helpers import make_chunk

    messages = template.build(
        "What is the price?",
        [make_chunk("pricing.md", 0, "A Gizmo costs $10.")],
    )
    assert [m.role for m in messages] == ["system", "user"]
    assert "<context>" in messages[1].content
    assert "[source: pricing.md #0]" in messages[1].content
    assert "Question: What is the price?" in messages[1].content


def test_prompt_context_truncation() -> None:
    from tests.helpers import make_chunk

    template = get_prompt_template("v1_grounding")
    big = make_chunk("d.md", 0, "word " * 2000)
    messages = template.build("q?", [big], max_context_chars=500)
    assert "[...]" in messages[1].content
    assert len(messages[1].content) < 1200


def test_prompt_system_instruction_override() -> None:
    template = get_prompt_template("v1_grounding")
    messages = template.build("q?", [], system_instruction="CUSTOM OVERRIDING INSTRUCTION")
    assert messages[0].content == "CUSTOM OVERRIDING INSTRUCTION"


# ---------------------------------------------------------------------------
# RAGSystem end-to-end (mock provider)
# ---------------------------------------------------------------------------


def _system(corpus_dir, **overrides) -> RAGSystem:
    config = RagSystemConfig.model_validate(
        {
            "kind": "rag",
            "corpus_dir": str(corpus_dir),
            "chunk_size": overrides.pop("chunk_size", 300),
            "chunk_overlap": overrides.pop("chunk_overlap", 30),
            "top_k": overrides.pop("top_k", 3),
            "prompt_version": overrides.pop("prompt_version", "v1_grounding"),
            "embedder": {"name": "hashing_tfidf", "dim": 256},
            **overrides,
        }
    )
    return RAGSystem(config, MockLLMProvider(), version="test-rag")


def test_rag_system_generates_grounded_answer(mini_corpus_dir) -> None:
    system = _system(mini_corpus_dir)
    case = make_case("c1", question="How much does a standard Gizmo cost per unit?")
    output = system.generate(case)
    assert "$10" in output.answer
    assert len(output.retrieved_context) == 3
    assert output.prompt_version == "v1_grounding"
    assert output.model_name == "mock-llm-v1"
    assert output.error is None
    assert output.latency_ms is not None and output.latency_ms >= 0


def test_rag_system_refuses_unknown_question(mini_corpus_dir) -> None:
    system = _system(mini_corpus_dir)
    case = make_case("c2", question="What is the airspeed velocity of an unladen swallow?")
    output = system.generate(case)
    assert "enough information" in output.answer.lower()


def test_rag_system_describe(mini_corpus_dir) -> None:
    system = _system(mini_corpus_dir, top_k=2)
    description = system.describe()
    assert description["kind"] == "rag"
    assert description["top_k"] == 2
    assert description["index_size"] > 0
    assert description["embedder"] == "HashingTfIdfEmbedder"


def test_rag_system_unknown_prompt_version(mini_corpus_dir) -> None:
    with pytest.raises(KeyError):
        _system(mini_corpus_dir, prompt_version="v404_missing")


def test_rag_system_deterministic(mini_corpus_dir) -> None:
    a = _system(mini_corpus_dir)
    b = _system(mini_corpus_dir)
    case = make_case("c3", question="What are the Gizmo support channels?")
    out_a = a.generate(case)
    out_b = b.generate(case)
    assert out_a.answer == out_b.answer
    assert [c.doc_id for c in out_a.retrieved_context] == [
        c.doc_id for c in out_b.retrieved_context
    ]
