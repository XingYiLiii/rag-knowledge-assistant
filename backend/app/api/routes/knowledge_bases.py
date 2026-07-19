"""Knowledge Base management endpoints."""

from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.api.routes.documents import get_vector_store_factory
from app.rag.vector_store import ChromaVectorStore
from app.schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseRead,
    KnowledgeBaseStatsRead,
    KnowledgeBaseUpdate,
)
from app.services.knowledge_base_service import KnowledgeBaseService

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])
VectorStoreFactory = Callable[[UUID], ChromaVectorStore]


def get_knowledge_base_service(
    session: Annotated[Session, Depends(get_database_session)],
) -> KnowledgeBaseService:
    """Build a request-scoped service from the injected database session."""
    return KnowledgeBaseService(session)


@router.post("", response_model=KnowledgeBaseRead, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> KnowledgeBaseRead:
    """Create a knowledge base."""
    return service.create(name=payload.name, description=payload.description)


@router.get("", response_model=list[KnowledgeBaseRead])
def list_knowledge_bases(
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> list[KnowledgeBaseRead]:
    """List knowledge bases with document counts."""
    return service.list()


@router.get("/{knowledge_base_id}/stats", response_model=KnowledgeBaseStatsRead)
def get_knowledge_base_stats(
    knowledge_base_id: UUID,
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
    vector_store_factory: Annotated[VectorStoreFactory, Depends(get_vector_store_factory)],
) -> KnowledgeBaseStatsRead:
    """Return document lifecycle totals and the isolated Chroma vector count."""
    return service.stats(knowledge_base_id, vector_store_factory=vector_store_factory)


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseRead)
def get_knowledge_base(
    knowledge_base_id: UUID,
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> KnowledgeBaseRead:
    """Get one knowledge base."""
    return service.get(knowledge_base_id)


@router.patch("/{knowledge_base_id}", response_model=KnowledgeBaseRead)
def update_knowledge_base(
    knowledge_base_id: UUID,
    payload: KnowledgeBaseUpdate,
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> KnowledgeBaseRead:
    """Update the explicitly provided knowledge base fields."""
    return service.update(
        knowledge_base_id,
        name=payload.name,
        description=payload.description,
        updated_fields=set(payload.model_fields_set),
    )


@router.delete("/{knowledge_base_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge_base(
    knowledge_base_id: UUID,
    service: Annotated[KnowledgeBaseService, Depends(get_knowledge_base_service)],
) -> Response:
    """Delete a knowledge base and rely on ORM/database cascades for related data."""
    service.delete(knowledge_base_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
