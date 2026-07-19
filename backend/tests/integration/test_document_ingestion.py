"""Integration tests for background document ingestion orchestration."""

from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document as LangChainDocument
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.api.routes.documents import get_document_ingestion_runner
from app.core.config import get_settings
from app.database.models import Document, KnowledgeBase
from app.main import create_app
from app.rag.pipeline import DocumentIngestionPipeline


@dataclass
class FakeEmbeddingProvider:
    """Record embedding requests without calling an external API."""

    calls: list[list[str]] = field(default_factory=list)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass
class FakeVectorStore:
    """Record writes and cleanup calls without starting Chroma."""

    fail_on_add: bool = False
    added: list[tuple[list[LangChainDocument], list[list[float]]]] = field(default_factory=list)
    deleted_document_ids: list[UUID | str | None] = field(default_factory=list)

    def add_documents(
        self,
        documents: Sequence[LangChainDocument],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if self.fail_on_add:
            raise RuntimeError("vector write failed")
        self.added.append((list(documents), [list(embedding) for embedding in embeddings]))

    def delete_documents(self, *, document_id: UUID | str | None = None) -> None:
        self.deleted_document_ids.append(document_id)


class IngestionRunnerFake:
    """BackgroundTasks-compatible runner backed by a test pipeline."""

    def __init__(self, pipeline: DocumentIngestionPipeline) -> None:
        self.pipeline = pipeline
        self.calls: list[UUID] = []

    def run(self, document_id: UUID) -> None:
        self.calls.append(document_id)
        self.pipeline.ingest(document_id)


def _make_chunk(document: Document) -> LangChainDocument:
    return LangChainDocument(
        page_content="chunk content",
        metadata={
            "document_id": str(document.id),
            "knowledge_base_id": str(document.knowledge_base_id),
            "original_filename": document.original_name,
            "file_type": document.file_type,
            "chunk_index": 0,
            "chunk_hash": "a" * 64,
        },
    )


@pytest.fixture()
def knowledge_base(db_session: Session) -> KnowledgeBase:
    """Persist a target knowledge base for each ingestion test."""
    entity = KnowledgeBase(name="Ingestion target")
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)
    return entity


def _build_client(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: IngestionRunnerFake,
) -> TestClient:
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_directory", tmp_path / "uploads")
    app = create_app()

    def override_database_session() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_document_ingestion_runner] = lambda: runner
    client = TestClient(app)
    client._ingestion_app = app  # type: ignore[attr-defined]
    return client


def test_upload_runs_background_ingestion_and_exposes_ready_status(
    db_session: Session,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful background task transitions a new document from pending to ready."""
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore()
    pipeline = DocumentIngestionPipeline(
        db_session,
        loader=lambda document: [_make_chunk(document)],
        splitter=lambda documents: list(documents),
        embedding_provider_factory=lambda: embedding_provider,
        vector_store_factory=lambda _: vector_store,  # type: ignore[arg-type]
    )
    runner = IngestionRunnerFake(pipeline)
    client = _build_client(db_session, tmp_path, monkeypatch, runner)

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents",
        files={"file": ("source.txt", b"source content", "text/plain")},
    )

    assert response.status_code == 201
    document_id = UUID(response.json()["id"])
    status_response = client.get(f"/api/v1/documents/{document_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "ready"
    assert status_response.json()["chunk_count"] == 1
    assert status_response.json()["error_message"] is None
    assert runner.calls == [document_id]
    assert embedding_provider.calls == [["chunk content"]]
    assert len(vector_store.added) == 1

    assert pipeline.ingest(document_id) is False
    assert len(embedding_provider.calls) == 1


def test_ingestion_failure_marks_document_failed_with_safe_summary(
    db_session: Session,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader errors persist a safe failed state without exposing raw error details."""
    pipeline = DocumentIngestionPipeline(
        db_session,
        loader=lambda _: (_ for _ in ()).throw(RuntimeError("sensitive parser detail")),
        splitter=lambda documents: list(documents),
        embedding_provider_factory=FakeEmbeddingProvider,
        vector_store_factory=lambda _: FakeVectorStore(),  # type: ignore[arg-type]
    )
    client = _build_client(db_session, tmp_path, monkeypatch, IngestionRunnerFake(pipeline))

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents",
        files={"file": ("source.txt", b"source content", "text/plain")},
    )

    status_response = client.get(f"/api/v1/documents/{response.json()['id']}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "failed"
    assert status_response.json()["error_message"] == "Ingestion failed: RuntimeError."


def test_vector_store_failure_cleans_up_document_vectors(
    db_session: Session,
    knowledge_base: KnowledgeBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vector write failures trigger scoped cleanup and a failed document status."""
    vector_store = FakeVectorStore(fail_on_add=True)
    pipeline = DocumentIngestionPipeline(
        db_session,
        loader=lambda document: [_make_chunk(document)],
        splitter=lambda documents: list(documents),
        embedding_provider_factory=FakeEmbeddingProvider,
        vector_store_factory=lambda _: vector_store,  # type: ignore[arg-type]
    )
    client = _build_client(db_session, tmp_path, monkeypatch, IngestionRunnerFake(pipeline))

    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents",
        files={"file": ("source.txt", b"source content", "text/plain")},
    )
    document_id = UUID(response.json()["id"])

    status_response = client.get(f"/api/v1/documents/{document_id}")
    assert status_response.json()["status"] == "failed"
    assert vector_store.deleted_document_ids == [document_id]
