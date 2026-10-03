"""OpenAI-compatible HTTP provider (chat completions + embeddings).

Works against any API speaking the OpenAI chat-completions protocol:
OpenAI, Azure OpenAI (via its compatible endpoint), vLLM, Ollama, LiteLLM
proxy, TGI, Together, Groq, OpenRouter, local llama.cpp servers, etc.

Robustness features:
  * bounded retries with deterministic exponential backoff for transport
    errors, 429 and 5xx responses (honours ``Retry-After`` when present);
  * graceful fallback when a server rejects ``response_format``;
  * timeouts on every request;
  * API keys only from the environment (never from config files).
"""

from __future__ import annotations

import time

import httpx

from ledger.config.settings import EmbedderConfig, ProviderConfig
from ledger.core.interfaces import LLMProvider
from ledger.core.models import CompletionRequest, CompletionResponse, CompletionUsage
from ledger.utils.logging import get_logger

logger = get_logger(__name__)

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_BACKOFF_BASE_S = 0.5
_MAX_BACKOFF_S = 15.0
_MAX_RETRY_AFTER_S = 20.0


class ProviderError(RuntimeError):
    """Raised when an LLM API call ultimately fails after retries."""


def _backoff_seconds(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(max(retry_after, 0.0), _MAX_RETRY_AFTER_S)
    return min(_BACKOFF_BASE_S * (2**attempt), _MAX_BACKOFF_S)


def _headers(api_key: str | None, extra: dict[str, str]) -> dict[str, str]:
    headers = {"Content-Type": "application/json", **extra}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class OpenAICompatProvider(LLMProvider):
    """Chat-completion provider for any OpenAI-compatible endpoint."""

    name = "openai_compat"

    def __init__(self, config: ProviderConfig, *, api_key: str | None = None) -> None:
        self._config = config
        self._api_key = api_key if api_key is not None else config.resolved_api_key()
        if not config.base_url:
            raise ProviderError(
                "provider.base_url is required for openai_compat (e.g. https://api.openai.com/v1)"
            )
        if config.name != "mock" and not self._api_key:
            logger.warning(
                "no API key resolved for provider (api_key_env=%r); requests will be "
                "unauthenticated - fine for some local servers, fatal for hosted APIs",
                config.api_key_env,
            )
        self._endpoint = config.base_url.rstrip("/") + "/chat/completions"

    # -- LLMProvider API -----------------------------------------------------

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        payload = self._payload(request)
        attempts = 0
        last_error: Exception | None = None
        allow_json_mode = request.response_format == "json"

        while attempts <= self._config.max_retries:
            attempts += 1
            body = dict(payload)
            if allow_json_mode:
                body["response_format"] = {"type": "json_object"}
            try:
                data = self._post(body)
                return self._parse_response(data, attempts)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                detail = exc.response.text[:400]
                # Some servers don't support response_format: retry once without it.
                if status == 400 and allow_json_mode and "response_format" in detail:
                    logger.warning("server rejected response_format=json_object; retrying without")
                    allow_json_mode = False
                    attempts -= 1
                    continue
                last_error = ProviderError(f"HTTP {status}: {detail}")
                if status not in _RETRYABLE_STATUS:
                    raise last_error from exc
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last_error = ProviderError(f"transport error: {exc}")

            if attempts > self._config.max_retries:
                break
            wait = _backoff_seconds(
                attempts - 1, self._retry_after_hint(getattr(last_error, "response", None))
            )
            logger.warning(
                "provider attempt %d/%d failed (%s); retrying in %.1fs",
                attempts,
                self._config.max_retries + 1,
                last_error,
                wait,
            )
            time.sleep(wait)

        assert last_error is not None
        raise last_error

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _retry_after_hint(exc_or_response: object) -> float | None:
        response = getattr(exc_or_response, "response", None)
        header = getattr(response, "headers", {}).get("Retry-After") if response else None
        try:
            return float(header) if header else None
        except (TypeError, ValueError):
            return None

    def _payload(self, request: CompletionRequest) -> dict[str, object]:
        config = self._config
        payload: dict[str, object] = {
            "model": config.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": config.temperature
            if request.temperature is None
            else request.temperature,
            "max_tokens": config.max_tokens if request.max_tokens is None else request.max_tokens,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        return payload

    def _post(self, body: dict[str, object]) -> dict[str, object]:
        with httpx.Client(timeout=self._config.timeout_s) as client:
            response = client.post(
                self._endpoint,
                json=body,
                headers=_headers(self._api_key, self._config.extra_headers),
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _parse_response(data: dict[str, object], attempts: int) -> CompletionResponse:
        try:
            choices = data["choices"]
            assert isinstance(choices, list) and choices, "empty choices"
            message = choices[0]["message"]
            text = message.get("content") or ""
            finish = choices[0].get("finish_reason")
        except (KeyError, TypeError, AssertionError) as exc:
            raise ProviderError(f"malformed completion response: {exc}") from exc

        usage_raw = data.get("usage") or {}
        usage = None
        if isinstance(usage_raw, dict) and usage_raw:
            usage = CompletionUsage(
                prompt_tokens=usage_raw.get("prompt_tokens"),
                completion_tokens=usage_raw.get("completion_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
            )
        return CompletionResponse(
            text=str(text),
            model=str(data.get("model") or ""),
            finish_reason=str(finish) if finish else None,
            latency_ms=0.0,  # measured by caller if needed
            usage=usage,
            attempts=attempts,
        )


class OpenAICompatEmbedder:
    """Embedding client for OpenAI-compatible ``/embeddings`` endpoints."""

    name = "openai_compat"

    def __init__(self, config: EmbedderConfig, *, api_key: str | None = None) -> None:
        self._config = config
        self._api_key = api_key if api_key is not None else config.resolved_api_key()
        if not config.base_url:
            raise ProviderError("embedder.base_url is required for openai_compat embedder")
        self._endpoint = config.base_url.rstrip("/") + "/embeddings"

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts (batched), returning L2-normalized vectors."""
        import numpy as np

        vectors: list[list[float]] = []
        batch_size = self._config.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._encode_batch(batch))
        if not vectors:
            return []
        matrix = np.asarray(vectors, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).tolist()

    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(self._config_batch_attempts()):
            try:
                with httpx.Client(timeout=self._config.timeout_s) as client:
                    response = client.post(
                        self._endpoint,
                        json={"model": self._config.model, "input": batch},
                        headers=_headers(self._api_key, {}),
                    )
                    response.raise_for_status()
                    data = response.json()
                items = sorted(data["data"], key=lambda d: d["index"])
                return [list(map(float, item["embedding"])) for item in items]
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError, TypeError) as exc:
                last_error = ProviderError(f"embedding request failed: {exc}")
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                    if status not in _RETRYABLE_STATUS:
                        raise last_error from exc
                if attempt < self._config_batch_attempts() - 1:
                    time.sleep(_backoff_seconds(attempt, None))
        assert last_error is not None
        raise last_error

    def _config_batch_attempts(self) -> int:
        return 3
