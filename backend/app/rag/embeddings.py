"""OpenAI-compatible implementation of the embedding provider contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.core.config import Settings, get_settings
from app.core.exceptions import ApplicationError
from app.rag.providers import EmbeddingProvider


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Adapt any OpenAI-compatible embeddings endpoint to the shared contract."""

    def __init__(self, *, client: Any, model: str) -> None:
        self._client = client
        self._model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch while preserving the input ordering."""
        input_texts = list(texts)
        if not input_texts:
            return []
        try:
            response = self._client.embeddings.create(model=self._model, input=input_texts)
        except (APITimeoutError, TimeoutError) as exc:
            raise ApplicationError(
                code="EMBEDDING_TIMEOUT",
                message="The embedding provider request timed out.",
                status_code=504,
            ) from exc
        except (APIConnectionError, OSError) as exc:
            raise ApplicationError(
                code="EMBEDDING_NETWORK_ERROR",
                message="The embedding provider could not be reached.",
                status_code=503,
            ) from exc
        except APIError as exc:
            raise ApplicationError(
                code="EMBEDDING_PROVIDER_ERROR",
                message="The embedding provider request failed.",
                status_code=502,
            ) from exc

        try:
            response_items = sorted(response.data, key=lambda item: item.index)
            vectors = [list(item.embedding) for item in response_items]
        except (AttributeError, TypeError) as exc:
            raise ApplicationError(
                code="EMBEDDING_RESPONSE_INVALID",
                message="The embedding provider returned an invalid response.",
                status_code=502,
            ) from exc
        if len(vectors) != len(input_texts):
            raise ApplicationError(
                code="EMBEDDING_RESPONSE_INVALID",
                message="The embedding provider returned an unexpected number of vectors.",
                status_code=502,
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query through the same batch API contract."""
        return self.embed_documents([text])[0]


def create_embedding_provider(
    settings: Settings | None = None,
    *,
    client_factory: Callable[..., Any] = OpenAI,
) -> EmbeddingProvider:
    """Build the configured provider without exposing vendor details to callers."""
    resolved_settings = settings or get_settings()
    if (
        resolved_settings.embedding_api_key is None
        or not resolved_settings.embedding_api_key.get_secret_value()
    ):
        raise ApplicationError(
            code="EMBEDDING_CONFIGURATION_ERROR",
            message="An embedding API key must be configured.",
            status_code=500,
        )
    client = client_factory(
        api_key=resolved_settings.embedding_api_key.get_secret_value(),
        base_url=resolved_settings.embedding_base_url,
        timeout=resolved_settings.embedding_timeout_seconds,
    )
    return OpenAICompatibleEmbeddingProvider(client=client, model=resolved_settings.embedding_model)
