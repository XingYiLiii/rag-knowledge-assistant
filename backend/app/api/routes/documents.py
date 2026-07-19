"""Document upload endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.core.config import get_settings
from app.rag.pipeline import DocumentIngestionRunner, create_document_ingestion_runner
from app.schemas.document import DocumentRead
from app.services.document_service import DocumentService

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}/documents", tags=["documents"])
status_router = APIRouter(prefix="/documents", tags=["documents"])


def get_document_service(
    session: Annotated[Session, Depends(get_database_session)],
) -> DocumentService:
    """Build a request-scoped document service from application settings."""
    settings = get_settings()
    return DocumentService(
        session,
        storage_directory=settings.upload_directory,
        max_file_size=settings.max_upload_file_size,
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


@status_router.get("/{document_id}", response_model=DocumentRead)
def get_document_status(
    document_id: UUID,
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentRead:
    """Return document ingestion status and safe failure details."""
    return service.get(document_id)
