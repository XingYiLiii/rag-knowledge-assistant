"""Application service that orchestrates one grounded RAG answer."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationError
from app.database.models import KnowledgeBase
from app.rag.citations import Citation, build_citations
from app.rag.context_builder import ContextBuilder, ContextBuildResult
from app.rag.prompts import build_grounded_messages
from app.rag.providers import ChatMessage, ChatProvider, ChatResult
from app.rag.retriever import RAGRetriever, RetrievedChunk

logger = logging.getLogger("app")

RetrieverFactory = Callable[[UUID], RAGRetriever]
ContextBuilderFactory = Callable[[], ContextBuilder]
ChatProviderFactory = Callable[[], ChatProvider]

NO_RELEVANT_KNOWLEDGE_ANSWER = "No relevant knowledge was found in the selected knowledge base."


@dataclass(frozen=True)
class ChatServiceResult:
    """Service-layer output for a completed or intentionally skipped RAG answer."""

    answer: str
    model: str | None
    latency_ms: float
    used_chunks: int
    citations: list[Citation]


class ChatService:
    """Coordinate retrieval, grounding, citations, and non-streaming model generation."""

    def __init__(
        self,
        session: Session,
        *,
        retriever_factory: RetrieverFactory,
        context_builder_factory: ContextBuilderFactory,
        chat_provider_factory: ChatProviderFactory,
    ) -> None:
        self._session = session
        self._retriever_factory = retriever_factory
        self._context_builder_factory = context_builder_factory
        self._chat_provider_factory = chat_provider_factory

    def answer(self, *, knowledge_base_id: UUID, question: str) -> ChatServiceResult:
        """Generate one answer and citations from evidence in the selected knowledge base."""
        self._ensure_knowledge_base_exists(knowledge_base_id)
        started_at = time.perf_counter()
        chunks = self._retrieve(knowledge_base_id, question)
        if not chunks:
            return self._no_knowledge_result(started_at)

        context_result = self._build_context(chunks, knowledge_base_id)
        if not context_result.chunks or not context_result.text:
            return self._no_knowledge_result(started_at)

        citations = build_citations(context_result.chunks)
        messages = build_grounded_messages(question=question, context=context_result.text)
        chat_result = self._generate(messages, knowledge_base_id)
        return ChatServiceResult(
            answer=chat_result.content,
            model=chat_result.model,
            latency_ms=_elapsed_ms(started_at),
            used_chunks=len(context_result.chunks),
            citations=citations,
        )

    def _ensure_knowledge_base_exists(self, knowledge_base_id: UUID) -> None:
        if self._session.get(KnowledgeBase, knowledge_base_id) is None:
            raise ApplicationError(
                code="KNOWLEDGE_BASE_NOT_FOUND",
                message="Knowledge base was not found.",
                status_code=404,
            )

    def _retrieve(self, knowledge_base_id: UUID, question: str) -> list[RetrievedChunk]:
        try:
            return self._retriever_factory(knowledge_base_id).retrieve(question)
        except ApplicationError:
            raise
        except Exception as exc:
            logger.error(
                "rag_chat.retrieval_failed", extra={"knowledge_base_id": str(knowledge_base_id)}
            )
            raise ApplicationError(
                code="RETRIEVAL_FAILED",
                message="Knowledge retrieval could not be completed.",
                status_code=502,
            ) from exc

    def _build_context(
        self,
        chunks: list[RetrievedChunk],
        knowledge_base_id: UUID,
    ) -> ContextBuildResult:
        try:
            return self._context_builder_factory().build(chunks)
        except Exception as exc:
            logger.error(
                "rag_chat.context_build_failed", extra={"knowledge_base_id": str(knowledge_base_id)}
            )
            raise ApplicationError(
                code="CONTEXT_BUILD_FAILED",
                message="Answer context could not be prepared.",
                status_code=500,
            ) from exc

    def _generate(self, messages: list[ChatMessage], knowledge_base_id: UUID) -> ChatResult:
        try:
            return self._chat_provider_factory().chat(messages)
        except ApplicationError:
            raise
        except Exception as exc:
            logger.error(
                "rag_chat.generation_failed", extra={"knowledge_base_id": str(knowledge_base_id)}
            )
            raise ApplicationError(
                code="CHAT_GENERATION_FAILED",
                message="The answer could not be generated.",
                status_code=502,
            ) from exc

    @staticmethod
    def _no_knowledge_result(started_at: float) -> ChatServiceResult:
        return ChatServiceResult(
            answer=NO_RELEVANT_KNOWLEDGE_ANSWER,
            model=None,
            latency_ms=_elapsed_ms(started_at),
            used_chunks=0,
            citations=[],
        )


def _elapsed_ms(started_at: float) -> float:
    """Return stable, non-negative request latency rounded for API output."""
    return round(max(0.0, (time.perf_counter() - started_at) * 1000), 2)
