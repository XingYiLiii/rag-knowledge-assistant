"""Business operations for secure document upload and lifecycle management."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationError
from app.database.models import Document, DocumentStatus, KnowledgeBase
from app.rag.vector_store import ChromaVectorStore

SUPPORTED_FILE_TYPES = {".pdf": "pdf", ".docx": "docx", ".md": "markdown", ".txt": "txt"}
VectorStoreFactory = Callable[[UUID], ChromaVectorStore]


@dataclass(frozen=True)
class DocumentResult:
    """Service-layer representation of a document response."""

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


@dataclass(frozen=True)
class DocumentPageResult:
    """Service-layer representation of a paginated document list."""

    items: list[DocumentResult]
    page: int
    page_size: int
    total: int


class DocumentService:
    """Coordinate document persistence, storage cleanup, and vector lifecycle operations."""

    def __init__(
        self,
        session: Session,
        *,
        storage_directory: Path,
        max_file_size: int,
        vector_store_factory: VectorStoreFactory,
    ) -> None:
        self._session = session
        self._storage_directory = storage_directory.resolve()
        self._max_file_size = max_file_size
        self._vector_store_factory = vector_store_factory

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

    def list(self, knowledge_base_id: UUID, *, page: int, page_size: int) -> DocumentPageResult:
        """List documents belonging to exactly one existing knowledge base."""
        self._get_knowledge_base_or_raise(knowledge_base_id)
        total = self._session.scalar(
            select(func.count(Document.id)).where(Document.knowledge_base_id == knowledge_base_id)
        )
        statement = (
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.created_at.desc(), Document.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        documents = self._session.scalars(statement).all()
        return DocumentPageResult(
            items=[self._to_result(document) for document in documents],
            page=page,
            page_size=page_size,
            total=total or 0,
        )

    def get(self, document_id: UUID) -> DocumentResult:
        """Return document processing status or a safe 404 error."""
        return self._to_result(self._get_document_or_raise(document_id))

    def delete(self, document_id: UUID) -> None:
        """Delete document vectors, staged local file, and database row with compensation."""
        document = self._get_document_or_raise(document_id)
        staged_file = self._stage_local_file(document)
        vector_store = self._vector_store_factory(document.knowledge_base_id)
        try:
            vector_store.delete_documents(document_id=document.id)
        except Exception as exc:
            self._restore_staged_file(staged_file)
            raise ApplicationError(
                code="DOCUMENT_DELETE_FAILED",
                message="The document could not be deleted.",
                status_code=500,
            ) from exc

        try:
            self._session.delete(document)
            self._session.commit()
        except SQLAlchemyError as exc:
            self._session.rollback()
            self._restore_staged_file(staged_file)
            raise ApplicationError(
                code="DOCUMENT_DELETE_FAILED",
                message="The document could not be deleted.",
                status_code=500,
            ) from exc

        if staged_file is not None:
            staged_file.unlink(missing_ok=True)

    def prepare_reindex(self, document_id: UUID) -> DocumentResult:
        """Remove old vectors and return a ready document to the pending ingestion state."""
        document = self._get_document_or_raise(document_id)
        if document.status != DocumentStatus.READY:
            raise ApplicationError(
                code="DOCUMENT_REINDEX_NOT_ALLOWED",
                message="Only ready documents can be reindexed.",
                status_code=409,
            )
        try:
            self._vector_store_factory(document.knowledge_base_id).delete_documents(
                document_id=document.id
            )
        except Exception as exc:
            raise ApplicationError(
                code="DOCUMENT_REINDEX_FAILED",
                message="Existing document vectors could not be removed.",
                status_code=500,
            ) from exc

        document.status = DocumentStatus.PENDING
        document.chunk_count = 0
        document.error_message = None
        self._session.commit()
        self._session.refresh(document)
        return self._to_result(document)

    def _get_document_or_raise(self, document_id: UUID) -> Document:
        document = self._session.get(Document, document_id)
        if document is None:
            raise ApplicationError(
                code="DOCUMENT_NOT_FOUND",
                message="Document was not found.",
                status_code=404,
            )
        return document

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

    def _stage_local_file(self, document: Document) -> Path | None:
        if not document.storage_path:
            return None
        source = self._resolve_destination(Path(document.storage_path))
        if not source.exists():
            return None
        staged_file = source.with_name(f".{source.name}.deleting-{uuid4().hex}")
        try:
            source.replace(staged_file)
        except OSError as exc:
            raise ApplicationError(
                code="DOCUMENT_DELETE_FAILED",
                message="The document file could not be removed.",
                status_code=500,
            ) from exc
        return staged_file

    @staticmethod
    def _restore_staged_file(staged_file: Path | None) -> None:
        if staged_file is None or not staged_file.exists():
            return
        original_name = staged_file.name.split(".deleting-", maxsplit=1)[0].lstrip(".")
        staged_file.replace(staged_file.with_name(original_name))

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
            chunk_count=document.chunk_count,
            error_message=document.error_message,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
