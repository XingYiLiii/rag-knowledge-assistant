"""Provider-agnostic interfaces for RAG model integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict


class ChatMessage(TypedDict):
    """Provider-neutral message passed to a chat completion provider."""

    role: str
    content: str


@dataclass(frozen=True)
class ChatUsage:
    """Token usage reported by a chat completion provider."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class ChatResult:
    """Normalized non-streaming chat completion result."""

    content: str
    model: str
    usage: ChatUsage


class EmbeddingProvider(ABC):
    """Stable embedding contract consumed by application business logic."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector for each source document text."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Return one vector for a retrieval query."""


class ChatProvider(ABC):
    """Stable chat contract consumed by future application business logic."""

    @abstractmethod
    def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        """Return a normalized assistant response for the supplied messages."""
