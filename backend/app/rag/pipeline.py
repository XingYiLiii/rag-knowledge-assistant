"""Background document ingestion orchestration."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from uuid import UUID

from langchain_core.documents import Document as LangChainDocument
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.database.models import Document, DocumentStatus
from app.database.session import SessionLocal
from app.rag.embeddings import create_embedding_provider
from app.rag.loaders import load_document
from app.rag.providers import EmbeddingProvider
from app.rag.splitter import split_documents
from app.rag.vector_store import ChromaVectorStore

DocumentLoader = Callable[[Document], list[LangChainDocument]]
DocumentSplitter = Callable[[Sequence[LangChainDocument]], list[LangChainDocument]]
EmbeddingProviderFactory = Callable[[], EmbeddingProvider]
VectorStoreFactory = Callable[[UUID], ChromaVectorStore]


class DocumentIngestionPipeline:
    """Move one uploaded document through parsing, chunking, embedding, and vector storage."""

    def __init__(
        self,
        session: Session,
        *,
        loader: DocumentLoader,
        splitter: DocumentSplitter,
        embedding_provider_factory: EmbeddingProviderFactory,
        vector_store_factory: VectorStoreFactory,
    ) -> None:
        self._session = session
        self._loader = loader
        self._splitter = splitter
        self._embedding_provider_factory = embedding_provider_factory
        self._vector_store_factory = vector_store_factory

    def ingest(self, document_id: UUID) -> bool:
        """Process a pending document once, returning whether this call acquired processing."""
        if not self._mark_processing(document_id):
            return False

        vector_store: ChromaVectorStore | None = None
        try:
            document = self._require_document(document_id)
            source_documents = self._loader(document)
            chunks = self._splitter(source_documents)
            if not chunks:
                raise ValueError("Document processing produced no chunks.")
            embedding_provider = self._embedding_provider_factory()
            embeddings = embedding_provider.embed_documents(
                [chunk.page_content for chunk in chunks]
            )
            if len(embeddings) != len(chunks):
                raise ValueError("Embedding provider returned an unexpected vector count.")
            vector_store = self._vector_store_factory(document.knowledge_base_id)
            vector_store.add_documents(chunks, embeddings)
        except Exception as exc:
            if vector_store is not None:
                vector_store.delete_documents(document_id=document_id)
            self._mark_failed(document_id, exc)
            return False

        document = self._require_document(document_id)
        document.status = DocumentStatus.READY
        document.chunk_count = len(chunks)
        document.error_message = None
        self._session.commit()
        return True

    def _mark_processing(self, document_id: UUID) -> bool:
        result = self._session.execute(
            update(Document)
            .where(Document.id == document_id, Document.status == DocumentStatus.PENDING)
            .values(status=DocumentStatus.PROCESSING, error_message=None)
        )
        self._session.commit()
        return result.rowcount == 1

    def _mark_failed(self, document_id: UUID, error: Exception) -> None:
        self._session.rollback()
        document = self._session.get(Document, document_id)
        if document is None:
            return
        document.status = DocumentStatus.FAILED
        document.error_message = _error_summary(error)
        self._session.commit()

    def _require_document(self, document_id: UUID) -> Document:
        document = self._session.get(Document, document_id)
        if document is None:
            raise ValueError("Document no longer exists.")
        return document


class DocumentIngestionRunner:
    """Create an isolated database session when FastAPI runs a background task."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        pipeline_factory: Callable[[Session], DocumentIngestionPipeline],
    ) -> None:
        self._session_factory = session_factory
        self._pipeline_factory = pipeline_factory

    def run(self, document_id: UUID) -> None:
        """Run ingestion with a new session independent of the request lifecycle."""
        with self._session_factory() as session:
            self._pipeline_factory(session).ingest(document_id)


def create_document_ingestion_runner(settings: Settings | None = None) -> DocumentIngestionRunner:
    """Build the production background runner from application settings."""
    resolved_settings = settings or get_settings()

    def pipeline_factory(session: Session) -> DocumentIngestionPipeline:
        return DocumentIngestionPipeline(
            session,
            loader=lambda document: load_document(
                document,
                storage_directory=resolved_settings.upload_directory,
            ),
            splitter=split_documents,
            embedding_provider_factory=lambda: create_embedding_provider(resolved_settings),
            vector_store_factory=lambda knowledge_base_id: ChromaVectorStore(
                knowledge_base_id=knowledge_base_id,
                persist_directory=resolved_settings.chroma_persist_directory,
            ),
        )

    return DocumentIngestionRunner(session_factory=SessionLocal, pipeline_factory=pipeline_factory)


def _error_summary(error: Exception) -> str:
    """Persist a useful but non-sensitive error summary for document status inspection."""
    message = getattr(error, "message", None)
    if isinstance(message, str) and message:
        return message[:500]
    return f"Ingestion failed: {type(error).__name__}."
