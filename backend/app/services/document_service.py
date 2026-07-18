"""Business operations for secure local document uploads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationError
from app.database.models import Document, DocumentStatus, KnowledgeBase

SUPPORTED_FILE_TYPES = {".pdf": "pdf", ".docx": "docx", ".md": "markdown", ".txt": "txt"}


@dataclass(frozen=True)
class DocumentResult:
    """Service-layer representation of one accepted upload."""

    id: UUID
    knowledge_base_id: UUID
    original_name: str
    file_type: str
    file_size: int
    sha256: str
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime


class DocumentService:
    """Persist validated uploads without exposing file-system operations to routes."""

    def __init__(self, session: Session, *, storage_directory: Path, max_file_size: int) -> None:
        self._session = session
        self._storage_directory = storage_directory.resolve()
        self._max_file_size = max_file_size

    async def upload(self, knowledge_base_id: UUID, uploaded_file: UploadFile) -> DocumentResult:
        """Validate, store, and record an uploaded document atomically as far as possible."""
        self._get_knowledge_base_or_raise(knowledge_base_id)
        original_name, suffix, file_type = self._validate_filename(uploaded_file.filename)
        content = await uploaded_file.read(self._max_file_size + 1)
        await uploaded_file.close()
        self._validate_content_size(len(content))

        sha256 = hashlib.sha256(content).hexdigest()
        self._raise_if_duplicate(knowledge_base_id, sha256)

        stored_name = f"{uuid4().hex}{suffix}"
        relative_path = Path(str(knowledge_base_id)) / stored_name
        destination = self._resolve_destination(relative_path)
        file_written = False

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as output_file:
                output_file.write(content)
            file_written = True

            document = Document(
                knowledge_base_id=knowledge_base_id,
                original_name=original_name,
                stored_name=stored_name,
                storage_path=relative_path.as_posix(),
                file_type=file_type,
                file_size=len(content),
                sha256=sha256,
                status=DocumentStatus.PENDING,
            )
            self._session.add(document)
            self._session.commit()
            self._session.refresh(document)
            return self._to_result(document)
        except (OSError, SQLAlchemyError) as exc:
            self._session.rollback()
            if file_written:
                destination.unlink(missing_ok=True)
            raise ApplicationError(
                code="DOCUMENT_UPLOAD_FAILED",
                message="The document could not be uploaded.",
                status_code=500,
            ) from exc

    def _get_knowledge_base_or_raise(self, knowledge_base_id: UUID) -> KnowledgeBase:
        knowledge_base = self._session.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is None:
            raise ApplicationError(
                code="KNOWLEDGE_BASE_NOT_FOUND",
                message="Knowledge base was not found.",
                status_code=404,
            )
        return knowledge_base

    @staticmethod
    def _validate_filename(filename: str | None) -> tuple[str, str, str]:
        original_name = Path((filename or "").replace("\\", "/")).name
        suffix = Path(original_name).suffix.lower()
        file_type = SUPPORTED_FILE_TYPES.get(suffix)
        if not original_name or file_type is None:
            raise ApplicationError(
                code="UNSUPPORTED_FILE_TYPE",
                message="Only PDF, DOCX, Markdown, and TXT files are supported.",
                status_code=400,
            )
        return original_name, suffix, file_type

    def _validate_content_size(self, file_size: int) -> None:
        if file_size == 0:
            raise ApplicationError(
                code="EMPTY_FILE",
                message="Uploaded files must not be empty.",
                status_code=400,
            )
        if file_size > self._max_file_size:
            raise ApplicationError(
                code="FILE_TOO_LARGE",
                message="The uploaded file exceeds the configured size limit.",
                status_code=413,
            )

    def _raise_if_duplicate(self, knowledge_base_id: UUID, sha256: str) -> None:
        statement = select(Document.id).where(
            Document.knowledge_base_id == knowledge_base_id,
            Document.sha256 == sha256,
        )
        if self._session.scalar(statement) is not None:
            raise ApplicationError(
                code="DUPLICATE_DOCUMENT",
                message="This file has already been uploaded to the knowledge base.",
                status_code=409,
            )

    def _resolve_destination(self, relative_path: Path) -> Path:
        destination = (self._storage_directory / relative_path).resolve()
        if not destination.is_relative_to(self._storage_directory):
            raise ApplicationError(
                code="INVALID_STORAGE_PATH",
                message="The document storage path is invalid.",
                status_code=400,
            )
        return destination

    @staticmethod
    def _to_result(document: Document) -> DocumentResult:
        return DocumentResult(
            id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            original_name=document.original_name,
            file_type=document.file_type,
            file_size=document.file_size,
            sha256=document.sha256,
            status=document.status,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
