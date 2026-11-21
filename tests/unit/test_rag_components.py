"""Unit tests: sample RAG system components (chunking, embeddings, retrieval, prompts)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ledger.config.settings import EmbedderConfig, RagSystemConfig
from ledger.providers.mock import MockLLMProvider
from ledger.providers.openai_compat import OpenAICompatEmbedder, ProviderError, _headers
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


class _EmbeddingResponse:
    def __init__(self, data: dict[str, object] | Exception) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


class _EmbeddingClient:
    response: dict[str, object] | Exception | None = None

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> "_EmbeddingClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def post(
        self,
        endpoint: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> _EmbeddingResponse:
        texts = json["input"]
        assert isinstance(texts, list)
        if type(self).response is not None:
            return _EmbeddingResponse(type(self).response)
        items = [
            {
                "index": index,
                "embedding": [float(len(text)), 1.0] + [0.0] * 30,
            }
            for index, text in reversed(list(enumerate(texts)))
        ]
        return _EmbeddingResponse({"data": items})


def _remote_embedder(batch_size: int = 64) -> OpenAICompatEmbedder:
    config = EmbedderConfig(
        name="openai_compat",
        dim=32,
        base_url="http://embedding.test/v1",
        batch_size=batch_size,
    )
    return OpenAICompatEmbedder(config, api_key="test-key")


def test_remote_embedder_returns_numpy_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)
    matrix = _remote_embedder().encode(["one", "three"])

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 32)
    assert np.issubdtype(matrix.dtype, np.number)


def test_remote_embedder_empty_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)
    matrix = _remote_embedder().encode([])

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (0, 32)
    assert np.issubdtype(matrix.dtype, np.number)


def test_remote_embedder_fit_exists_and_is_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)
    embedder = _remote_embedder()

    embedder.fit(["alpha", "beta"])
    assert embedder.dim == 32


def test_remote_embedder_multiple_batches_preserve_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)
    embedder = _remote_embedder(batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    matrix = embedder.encode(texts)

    assert matrix.shape == (5, 32)
    lengths = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = lengths / np.sqrt(lengths**2 + 1.0)
    assert np.allclose(matrix[:, 0], expected)


@pytest.mark.parametrize(
    ("embedding", "include_embedding", "message"),
    [
        (None, False, "missing 'embedding'"),
        (None, True, "malformed"),
        ([], True, "empty"),
        (1.0, True, "malformed"),
        (True, True, "malformed"),
        ("not-a-list", True, "malformed"),
        (["bad", 0.0] + [0.0] * 30, True, "numeric"),
        ([1.0] * 31, True, "dimension"),
        ([float("nan")] + [0.0] * 31, True, "finite"),
        ([float("inf")] + [0.0] * 31, True, "finite"),
        ([float("-inf")] + [0.0] * 31, True, "finite"),
    ],
)
def test_remote_embedder_rejects_invalid_vectors(
    monkeypatch: pytest.MonkeyPatch,
    embedding: object,
    include_embedding: bool,
    message: str,
) -> None:
    item: dict[str, object] = {"index": 0}
    if include_embedding:
        item["embedding"] = embedding
    monkeypatch.setattr(_EmbeddingClient, "response", {"data": [item]})
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)
    monkeypatch.setattr("ledger.providers.openai_compat.time.sleep", lambda delay: None)

    with pytest.raises(ProviderError, match=message):
        _remote_embedder().encode(["one"])


def test_remote_embedder_rejects_ragged_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {
        "data": [
            {"index": 0, "embedding": [1.0] * 32},
            {"index": 1, "embedding": [1.0] * 31},
        ]
    }
    monkeypatch.setattr(_EmbeddingClient, "response", data)
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)

    with pytest.raises(ProviderError, match="dimension|ragged"):
        _remote_embedder().encode(["one", "two"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "malformed embedding response"),
        ({"data": None}, "malformed embedding response"),
        ({"data": {}}, "malformed embedding response"),
        ({"data": ["bad-item"]}, "malformed embedding response"),
        ({"data": [{}]}, "malformed embedding response"),
        ({"data": [{"index": "0", "embedding": [1.0] * 32}]}, "index"),
        ({"data": [{"index": 0.0, "embedding": [1.0] * 32}]}, "index"),
        ({"data": [{"index": True, "embedding": [1.0] * 32}]}, "index"),
    ],
)
def test_remote_embedder_rejects_malformed_responses(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    message: str,
) -> None:
    monkeypatch.setattr(_EmbeddingClient, "response", payload)
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)
    monkeypatch.setattr("ledger.providers.openai_compat.time.sleep", lambda delay: None)

    with pytest.raises(ProviderError, match=message):
        _remote_embedder().encode(["one"])


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([{"index": 0, "embedding": [1.0] * 32}], "response count"),
        (
            [
                {"index": 0, "embedding": [1.0] * 32},
                {"index": 1, "embedding": [1.0] * 32},
                {"index": 0, "embedding": [1.0] * 32},
            ],
            "response count",
        ),
        (
            [
                {"index": 0, "embedding": [1.0] * 32},
                {"index": 0, "embedding": [1.0] * 32},
            ],
            "duplicate|unique",
        ),
        (
            [
                {"index": 0, "embedding": [1.0] * 32},
                {"index": 2, "embedding": [1.0] * 32},
            ],
            "missing|index",
        ),
        (
            [
                {"index": 0, "embedding": [1.0] * 32},
                {"index": -1, "embedding": [1.0] * 32},
            ],
            "out of range",
        ),
        (
            [
                {"index": 0, "embedding": [1.0] * 32},
                {"index": 2.0, "embedding": [1.0] * 32},
            ],
            "integer|index",
        ),
        (
            [
                {"index": 0, "embedding": [1.0] * 32},
                {"index": True, "embedding": [1.0] * 32},
            ],
            "integer|index",
        ),
    ],
)
def test_remote_embedder_rejects_invalid_indexes_or_counts(
    monkeypatch: pytest.MonkeyPatch,
    items: list[dict[str, object]],
    message: str,
) -> None:
    monkeypatch.setattr(_EmbeddingClient, "response", {"data": items})
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)

    with pytest.raises(ProviderError, match=message):
        _remote_embedder().encode(["one", "two"])


def test_remote_embedder_rejects_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _EmbeddingClient,
        "response",
        json.JSONDecodeError("invalid JSON", "{bad", 1),
    )
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)

    with pytest.raises(ProviderError, match="malformed embedding response"):
        _remote_embedder().encode(["one"])


def test_remote_embedder_restores_out_of_order_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0] + [0.0] * 30},
            {"index": 0, "embedding": [1.0, 0.0] + [0.0] * 30},
        ]
    }
    monkeypatch.setattr(_EmbeddingClient, "response", data)
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)

    matrix = _remote_embedder().encode(["first", "second"])

    assert matrix.shape == (2, 32)
    assert np.allclose(matrix[0], [1.0, 0.0] + [0.0] * 30)
    assert np.allclose(matrix[1], [0.0, 1.0] + [0.0] * 30)


def test_remote_embedder_preserves_zero_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {
        "data": [
            {"index": 0, "embedding": [0.0] * 32},
        ]
    }
    monkeypatch.setattr(_EmbeddingClient, "response", data)
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _EmbeddingClient)

    matrix = _remote_embedder().encode(["zero"])

    assert matrix.shape == (1, 32)
    assert np.array_equal(matrix, np.zeros((1, 32), dtype=np.float64))


def test_api_headers_use_bearer_token_and_preserve_custom_headers() -> None:
    headers = _headers("abc123xyz", {"X-Custom-Header": "custom-value"})

    expected_authorization = "".join(["Bearer", " ", "abc123xyz"])
    assert headers["Authorization"] == expected_authorization

    assert headers["Content-Type"] == "application/json"
    assert headers["X-Custom-Header"] == "custom-value"

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
