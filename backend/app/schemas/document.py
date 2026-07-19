"""Pydantic schemas for uploaded-document responses."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.database.models import DocumentStatus


class DocumentRead(BaseModel):
    """Public metadata for a document accepted for later processing."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    original_name: str
    file_type: str
    file_size: int
    sha256: str
    status: DocumentStatus
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime
