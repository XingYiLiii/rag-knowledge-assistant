"""Business operations for Knowledge Base management and statistics."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationError
from app.database.models import Document, DocumentStatus, KnowledgeBase
from app.rag.vector_store import ChromaVectorStore

VectorStoreFactory = Callable[[UUID], ChromaVectorStore]


@dataclass(frozen=True)
class KnowledgeBaseResult:
    """Service-layer representation of a knowledge base response."""

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    document_count: int


@dataclass(frozen=True)
class KnowledgeBaseStatsResult:
    """Aggregated document and vector-store statistics."""

    document_count: int
    ready_document_count: int
    processing_document_count: int
    failed_document_count: int
    total_chunk_count: int
    vector_count: int


class KnowledgeBaseService:
    """Coordinate knowledge-base persistence without leaking ORM logic to routes."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, name: str, description: str | None) -> KnowledgeBaseResult:
        """Create and return a new knowledge base."""
        knowledge_base = KnowledgeBase(name=name, description=description)
        self._session.add(knowledge_base)
        self._commit_or_raise_name_conflict()
        self._session.refresh(knowledge_base)
        return self._to_result(knowledge_base, document_count=0)

    def list(self) -> list[KnowledgeBaseResult]:
        """List knowledge bases with a correlated document count."""
        document_count = (
            select(func.count(Document.id))
            .where(Document.knowledge_base_id == KnowledgeBase.id)
            .correlate(KnowledgeBase)
            .scalar_subquery()
        )
        statement = select(KnowledgeBase, document_count.label("document_count")).order_by(
            KnowledgeBase.created_at.desc()
        )
        rows = self._session.execute(statement).all()
        return [self._to_result(knowledge_base, count) for knowledge_base, count in rows]

    def get(self, knowledge_base_id: UUID) -> KnowledgeBaseResult:
        """Return one knowledge base or raise a safe 404 error."""
        knowledge_base = self._get_entity_or_raise(knowledge_base_id)
        document_count = self._session.scalar(
            select(func.count(Document.id)).where(Document.knowledge_base_id == knowledge_base.id)
        )
        return self._to_result(knowledge_base, document_count or 0)

    def stats(
        self,
        knowledge_base_id: UUID,
        *,
        vector_store_factory: VectorStoreFactory,
    ) -> KnowledgeBaseStatsResult:
        """Return document lifecycle counts alongside the isolated vector count."""
        self._get_entity_or_raise(knowledge_base_id)
        statement = select(
            func.count(Document.id),
            func.coalesce(func.sum(case((Document.status == DocumentStatus.READY, 1), else_=0)), 0),
            func.coalesce(
                func.sum(case((Document.status == DocumentStatus.PROCESSING, 1), else_=0)), 0
            ),
            func.coalesce(
                func.sum(case((Document.status == DocumentStatus.FAILED, 1), else_=0)), 0
            ),
            func.coalesce(func.sum(Document.chunk_count), 0),
        ).where(Document.knowledge_base_id == knowledge_base_id)
        document_count, ready_count, processing_count, failed_count, total_chunk_count = (
            self._session.execute(statement).one()
        )
        return KnowledgeBaseStatsResult(
            document_count=document_count,
            ready_document_count=ready_count,
            processing_document_count=processing_count,
            failed_document_count=failed_count,
            total_chunk_count=total_chunk_count,
            vector_count=vector_store_factory(knowledge_base_id).count(),
        )

    def update(
        self,
        knowledge_base_id: UUID,
        *,
        name: str | None,
        description: str | None,
        updated_fields: set[str],
    ) -> KnowledgeBaseResult:
        """Apply the explicitly supplied fields and return the updated knowledge base."""
        knowledge_base = self._get_entity_or_raise(knowledge_base_id)
        if "name" in updated_fields:
            knowledge_base.name = name  # type: ignore[assignment]
        if "description" in updated_fields:
            knowledge_base.description = description

        self._commit_or_raise_name_conflict()
        self._session.refresh(knowledge_base)
        return self.get(knowledge_base.id)

    def delete(self, knowledge_base_id: UUID) -> None:
        """Delete a knowledge base and rely on ORM/database cascades for related data."""
        knowledge_base = self._get_entity_or_raise(knowledge_base_id)
        self._session.delete(knowledge_base)
        self._session.commit()

    def _get_entity_or_raise(self, knowledge_base_id: UUID) -> KnowledgeBase:
        knowledge_base = self._session.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise ApplicationError(
                code="KNOWLEDGE_BASE_NOT_FOUND",
                message="Knowledge base was not found.",
                status_code=404,
            )
        return knowledge_base

    def _commit_or_raise_name_conflict(self) -> None:
        try:
            self._session.commit()
        except IntegrityError:
            self._session.rollback()
            raise ApplicationError(
                code="KNOWLEDGE_BASE_NAME_CONFLICT",
                message="A knowledge base with this name already exists.",
                status_code=409,
            ) from None

    @staticmethod
    def _to_result(knowledge_base: KnowledgeBase, document_count: int) -> KnowledgeBaseResult:
        return KnowledgeBaseResult(
            id=knowledge_base.id,
            name=knowledge_base.name,
            description=knowledge_base.description,
            created_at=knowledge_base.created_at,
            updated_at=knowledge_base.updated_at,
            document_count=document_count,
        )
