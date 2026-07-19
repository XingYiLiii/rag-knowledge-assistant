"""Document upload and management endpoints."""

from collections.abc import Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.core.config import get_settings
from app.rag.pipeline import DocumentIngestionRunner, create_document_ingestion_runner
from app.rag.vector_store import ChromaVectorStore
from app.schemas.document import DocumentPage, DocumentRead
from app.services.document_service import DocumentService

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}/documents", tags=["documents"])
status_router = APIRouter(prefix="/documents", tags=["documents"])
VectorStoreFactory = Callable[[UUID], ChromaVectorStore]


def get_vector_store_factory() -> VectorStoreFactory:
    """Build a factory so services stay independent of Chroma construction details."""
    settings = get_settings()
    return lambda knowledge_base_id: ChromaVectorStore(
        knowledge_base_id=knowledge_base_id,
        persist_directory=settings.chroma_persist_directory,
    )


def get_document_service(
    session: Annotated[Session, Depends(get_database_session)],
    vector_store_factory: Annotated[VectorStoreFactory, Depends(get_vector_store_factory)],
) -> DocumentService:
    """Build a request-scoped document management service."""
    settings = get_settings()
    return DocumentService(
        session,
        storage_directory=settings.upload_directory,
        max_file_size=settings.max_upload_file_size,
        vector_store_factory=vector_store_factory,
    )


def get_document_ingestion_runner() -> DocumentIngestionRunner:
    """Build the background runner outside the request database session."""
    return create_document_ingestion_runner()


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    knowledge_base_id: UUID,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="PDF, DOCX, Markdown, or TXT document")],
    service: Annotated[DocumentService, Depends(get_document_service)],
    ingestion_runner: Annotated[DocumentIngestionRunner, Depends(get_document_ingestion_runner)],
) -> DocumentRead:
    """Store an upload and schedule independent background ingestion."""
    document = await service.upload(knowledge_base_id, file)
    background_tasks.add_task(ingestion_runner.run, document.id)
    return document


@router.get("", response_model=DocumentPage)
def list_documents(
    knowledge_base_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> DocumentPage:
    """List paginated documents belonging to one knowledge base."""
    return service.list(knowledge_base_id, page=page, page_size=page_size)


@status_router.get("/{document_id}", response_model=DocumentRead)
def get_document_status(
    document_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentRead:
    """Return document ingestion status and metadata."""
    return service.get(document_id)


@status_router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> Response:
    """Remove database, local-file, and vector resources for one document."""
    service.delete(document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@status_router.post("/{document_id}/reindex", response_model=DocumentRead)
def reindex_document(
    document_id: UUID,
    background_tasks: BackgroundTasks,
    service: Annotated[DocumentService, Depends(get_document_service)],
    ingestion_runner: Annotated[DocumentIngestionRunner, Depends(get_document_ingestion_runner)],
) -> DocumentRead:
    """Reset a ready document and schedule a new ingestion pass."""
    document = service.prepare_reindex(document_id)
    background_tasks.add_task(ingestion_runner.run, document.id)
    return document
