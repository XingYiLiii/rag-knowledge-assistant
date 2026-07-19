"""Unit tests for the OpenAI-compatible embedding provider."""

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import ApplicationError
from app.rag.embeddings import OpenAICompatibleEmbeddingProvider, create_embedding_provider
from app.rag.providers import EmbeddingProvider


@dataclass
class FakeEmbeddingItem:
    """Small response item matching the OpenAI embeddings response shape."""

    index: int
    embedding: list[float]


@dataclass
class FakeEmbeddingResponse:
    """Small response container matching the OpenAI embeddings response shape."""

    data: list[FakeEmbeddingItem]


class FakeEmbeddingsResource:
    """Record create calls and return a configurable fake response or error."""

    def __init__(self, response: FakeEmbeddingResponse | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeEmbeddingResponse:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeClient:
    """Minimal OpenAI-compatible client fake."""

    def __init__(self, response: FakeEmbeddingResponse | Exception) -> None:
        self.embeddings = FakeEmbeddingsResource(response)


def test_factory_reads_openai_compatible_settings() -> None:
    """Base URL, API key, model, and timeout are passed to the compatible client."""
    received: dict[str, Any] = {}
    client = FakeClient(FakeEmbeddingResponse([FakeEmbeddingItem(0, [0.1, 0.2])]))

    def client_factory(**kwargs: Any) -> FakeClient:
        received.update(kwargs)
        return client

    settings = Settings(
        embedding_base_url="https://embedding.example/v1",
        embedding_api_key="test-secret-key",
        embedding_model="test-embedding-model",
        embedding_timeout_seconds=12.5,
    )
    provider = create_embedding_provider(settings, client_factory=client_factory)

    assert isinstance(provider, EmbeddingProvider)
    assert received == {
        "api_key": "test-secret-key",
        "base_url": "https://embedding.example/v1",
        "timeout": 12.5,
    }
    assert provider.embed_query("query") == [0.1, 0.2]
    assert client.embeddings.calls == [{"model": "test-embedding-model", "input": ["query"]}]


def test_embed_documents_preserves_response_index_order() -> None:
    """Batch embedding results are returned in the same order as the input texts."""
    client = FakeClient(
        FakeEmbeddingResponse([FakeEmbeddingItem(1, [0.3]), FakeEmbeddingItem(0, [0.1, 0.2])])
    )
    provider = OpenAICompatibleEmbeddingProvider(client=client, model="embedding-model")

    vectors = provider.embed_documents(["first", "second"])

    assert vectors == [[0.1, 0.2], [0.3]]
    assert client.embeddings.calls == [{"model": "embedding-model", "input": ["first", "second"]}]


def test_empty_batch_does_not_call_provider() -> None:
    """Empty input is handled locally without an unnecessary network request."""
    client = FakeClient(FakeEmbeddingResponse([]))
    provider = OpenAICompatibleEmbeddingProvider(client=client, model="embedding-model")

    assert provider.embed_documents([]) == []
    assert client.embeddings.calls == []


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (TimeoutError("test-secret-key timed out"), "EMBEDDING_TIMEOUT"),
        (OSError("test-secret-key unavailable"), "EMBEDDING_NETWORK_ERROR"),
    ],
)
def test_network_errors_are_safe_and_do_not_log_api_key(
    error: Exception, expected_code: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Transport failures expose stable errors without exposing sensitive provider input."""
    client = FakeClient(error)
    provider = OpenAICompatibleEmbeddingProvider(client=client, model="embedding-model")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ApplicationError) as raised_error:
            provider.embed_query("query")

    assert raised_error.value.code == expected_code
    assert "test-secret-key" not in raised_error.value.message
    assert "test-secret-key" not in caplog.text


def test_missing_api_key_returns_safe_configuration_error() -> None:
    """Test environments do not need a real key and missing configuration stays safe."""
    settings = Settings(embedding_api_key=None)

    with pytest.raises(ApplicationError) as raised_error:
        create_embedding_provider(settings)

    assert raised_error.value.code == "EMBEDDING_CONFIGURATION_ERROR"
