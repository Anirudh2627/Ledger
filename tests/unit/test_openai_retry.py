"""Unit tests: OpenAI-compatible HTTP retry behavior."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from ledger.config.settings import EmbedderConfig, ProviderConfig
from ledger.core.models import ChatMessage, CompletionRequest
from ledger.providers.openai_compat import (
    OpenAICompatEmbedder,
    OpenAICompatProvider,
    ProviderError,
)

pytestmark = pytest.mark.unit


class _SequencedClient:
    responses: list[httpx.Response] = []

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> "_SequencedClient":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def post(
        self,
        endpoint: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
    ) -> httpx.Response:
        return type(self).responses.pop(0)


def _http_response(
    status: int,
    *,
    headers: dict[str, str] | None = None,
    json: dict[str, object] | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", "http://provider.test/v1/test")
    return httpx.Response(
        status,
        headers=headers,
        json=json,
        request=request,
    )


def _chat_request() -> CompletionRequest:
    return CompletionRequest(
        messages=[
            ChatMessage(role="user", content="hello"),
        ],
        purpose="generation",
    )


def _chat_provider(max_retries: int = 1) -> OpenAICompatProvider:
    config = ProviderConfig(
        name="openai_compat",
        model="test-model",
        base_url="http://provider.test/v1",
        max_retries=max_retries,
    )
    return OpenAICompatProvider(config)


def _embedding(batch_size: int = 64) -> OpenAICompatEmbedder:
    config = EmbedderConfig(
        name="openai_compat",
        model="test-model",
        dim=32,
        base_url="http://provider.test/v1",
        batch_size=batch_size,
    )
    return OpenAICompatEmbedder(config)


def test_chat_retry_after_uses_response_header(monkeypatch: pytest.MonkeyPatch) -> None:
    rate_limited = _http_response(
        429,
        headers={"Retry-After": "5"},
        json={"error": {"message": "rate limited"}},
    )
    success = _http_response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "test-model",
        },
    )
    monkeypatch.setattr(_SequencedClient, "responses", [rate_limited, success])
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _SequencedClient)
    sleeps: list[float] = []
    monkeypatch.setattr("ledger.providers.openai_compat.time.sleep", sleeps.append)

    response = _chat_provider().complete(_chat_request())

    assert response.text == "ok"
    assert sleeps == [5.0]


def test_embedding_retry_after_uses_response_header(monkeypatch: pytest.MonkeyPatch) -> None:
    rate_limited = _http_response(
        429,
        headers={"Retry-After": "3"},
        json={"error": {"message": "rate limited"}},
    )
    success = _http_response(
        200,
        json={
            "data": [
                {"index": 0, "embedding": [1.0] + [0.0] * 31},
            ]
        },
    )
    monkeypatch.setattr(_SequencedClient, "responses", [rate_limited, success])
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _SequencedClient)
    sleeps: list[float] = []
    monkeypatch.setattr("ledger.providers.openai_compat.time.sleep", sleeps.append)

    matrix = _embedding().encode(["hello"])

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (1, 32)
    assert sleeps == [3.0]


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        (None, 0.5),
        ("not-a-number", 0.5),
        ("60", 20.0),
    ],
)
def test_chat_retry_after_fallback_and_cap(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str | None,
    expected: float,
) -> None:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    rate_limited = _http_response(
        429,
        headers=headers,
        json={"error": {"message": "rate limited"}},
    )
    success = _http_response(
        200,
        json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "model": "test-model",
        },
    )
    monkeypatch.setattr(_SequencedClient, "responses", [rate_limited, success])
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _SequencedClient)
    sleeps: list[float] = []
    monkeypatch.setattr("ledger.providers.openai_compat.time.sleep", sleeps.append)

    _chat_provider().complete(_chat_request())

    assert sleeps == [expected]


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [
        (None, 0.5),
        ("not-a-number", 0.5),
        ("60", 20.0),
    ],
)
def test_embedding_retry_after_fallback_and_cap(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str | None,
    expected: float,
) -> None:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    rate_limited = _http_response(
        429,
        headers=headers,
        json={"error": {"message": "rate limited"}},
    )
    success = _http_response(
        200,
        json={
            "data": [
                {"index": 0, "embedding": [1.0] + [0.0] * 31},
            ]
        },
    )
    monkeypatch.setattr(_SequencedClient, "responses", [rate_limited, success])
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _SequencedClient)
    sleeps: list[float] = []
    monkeypatch.setattr("ledger.providers.openai_compat.time.sleep", sleeps.append)

    _embedding().encode(["hello"])

    assert sleeps == [expected]


@pytest.mark.parametrize("provider_kind", ["chat", "embedding"])
def test_non_retryable_status_does_not_sleep(
    monkeypatch: pytest.MonkeyPatch,
    provider_kind: str,
) -> None:
    bad_request = _http_response(
        400,
        headers={"Retry-After": "5"},
        json={"error": {"message": "bad request"}},
    )
    monkeypatch.setattr(_SequencedClient, "responses", [bad_request])
    monkeypatch.setattr("ledger.providers.openai_compat.httpx.Client", _SequencedClient)
    sleeps: list[float] = []
    monkeypatch.setattr("ledger.providers.openai_compat.time.sleep", sleeps.append)

    with pytest.raises(ProviderError):
        if provider_kind == "chat":
            _chat_provider().complete(_chat_request())
        else:
            _embedding().encode(["hello"])

    assert sleeps == []
