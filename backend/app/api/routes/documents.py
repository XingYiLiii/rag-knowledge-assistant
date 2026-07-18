"""Document upload endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.core.config import get_settings
from app.schemas.document import DocumentRead
from app.services.document_service import DocumentService

router = APIRouter(prefix="/knowledge-bases/{knowledge_base_id}/documents", tags=["documents"])


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


@router.post("", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    knowledge_base_id: UUID,
    file: Annotated[UploadFile, File(description="PDF, DOCX, Markdown, or TXT document")],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentRead:
    """Store an uploaded document and create its pending processing record."""
    return await service.upload(knowledge_base_id, file)
