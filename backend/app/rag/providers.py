"""Provider-agnostic interfaces for RAG model integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingProvider(ABC):
    """Stable embedding contract consumed by application business logic."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector for each source document text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return one vector for a retrieval query."""
