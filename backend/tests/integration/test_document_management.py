"""Integration tests for document management and knowledge-base statistics."""

from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.api.routes.documents import get_document_ingestion_runner, get_vector_store_factory
from app.core.config import get_settings
from app.database.models import Document, DocumentStatus, KnowledgeBase
from app.main import create_app


@dataclass
class FakeVectorStore:
    """In-memory vector-store substitute for management API tests."""

    vector_count: int = 0
    deleted_document_ids: list[UUID | str | None] = field(default_factory=list)

    def delete_documents(self, *, document_id: UUID | str | None = None) -> None:
        self.deleted_document_ids.append(document_id)

    def count(self) -> int:
        return self.vector_count


class IngestionRunnerFake:
    """Background runner that completes reindexing without invoking external services."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self.calls: list[UUID] = []

    def run(self, document_id: UUID) -> None:
        self.calls.append(document_id)
        document = self._session.get(Document, document_id)
        assert document is not None
        document.status = DocumentStatus.READY
        document.chunk_count = 1
        self._session.commit()


@pytest.fixture()
def knowledge_base(db_session: Session) -> KnowledgeBase:
    """Persist the knowledge base used by management tests."""
    entity = KnowledgeBase(name="Document management target")
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)
    return entity


def _document(
    knowledge_base: KnowledgeBase,
    *,
    name: str,
    status: DocumentStatus = DocumentStatus.READY,
    chunk_count: int = 1,
) -> Document:
    return Document(
        knowledge_base_id=knowledge_base.id,
        original_name=name,
        stored_name=f"stored-{name}",
        storage_path=f"{knowledge_base.id}/stored-{name}",
        file_type="txt",
        file_size=10,
        sha256=(name.encode().hex() + "a" * 64)[:64],
        status=status,
        chunk_count=chunk_count,
    )


@pytest.fixture()
def client(
    db_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TestClient, FakeVectorStore, IngestionRunnerFake], None, None]:
    """Create a client with no real vector store, embedding provider, or runner."""
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_directory", tmp_path / "uploads")
    vector_store = FakeVectorStore()
    runner = IngestionRunnerFake(db_session)
    app = create_app()

    def override_database_session() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_vector_store_factory] = lambda: lambda _: vector_store
    app.dependency_overrides[get_document_ingestion_runner] = lambda: runner
    yield TestClient(app), vector_store, runner
    app.dependency_overrides.clear()


def test_lists_documents_with_pagination(
    client: tuple[TestClient, FakeVectorStore, IngestionRunnerFake],
    db_session: Session,
    knowledge_base: KnowledgeBase,
) -> None:
    """Document list responses are knowledge-base scoped and paginated."""
    other_knowledge_base = KnowledgeBase(name="Other target")
    db_session.add(other_knowledge_base)
    db_session.flush()
    db_session.add_all(
        [
            _document(knowledge_base, name="one.txt"),
            _document(knowledge_base, name="two.txt"),
            _document(knowledge_base, name="three.txt"),
            _document(other_knowledge_base, name="other.txt"),
        ]
    )
    db_session.commit()

    response = client[0].get(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents?page=2&page_size=2"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 2
    assert payload["page_size"] == 2
    assert payload["total"] == 3
    assert len(payload["items"]) == 1
    assert payload["items"][0]["original_name"] in {"one.txt", "two.txt", "three.txt"}


def test_document_detail_and_delete_cleans_all_resources(
    client: tuple[TestClient, FakeVectorStore, IngestionRunnerFake],
    db_session: Session,
    knowledge_base: KnowledgeBase,
) -> None:
    """Deletion removes the database record, staged local file, and document vectors."""
    document = _document(knowledge_base, name="remove.txt")
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)
    storage_path = get_settings().upload_directory / document.storage_path
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_path.write_text("content", encoding="utf-8")

    detail_response = client[0].get(f"/api/v1/documents/{document.id}")
    delete_response = client[0].delete(f"/api/v1/documents/{document.id}")

    assert detail_response.status_code == 200
    assert detail_response.json()["original_name"] == "remove.txt"
    assert delete_response.status_code == 204
    assert db_session.get(Document, document.id) is None
    assert not storage_path.exists()
    assert client[1].deleted_document_ids == [document.id]


def test_reindex_resets_document_and_does_not_duplicate_vectors(
    client: tuple[TestClient, FakeVectorStore, IngestionRunnerFake],
    db_session: Session,
    knowledge_base: KnowledgeBase,
) -> None:
    """Ready documents clear old vectors before exactly one background reindex run."""
    document = _document(knowledge_base, name="reindex.txt", chunk_count=3)
    db_session.add(document)
    db_session.commit()
    db_session.refresh(document)

    response = client[0].post(f"/api/v1/documents/{document.id}/reindex")

    assert response.status_code == 200
    refreshed = db_session.get(Document, document.id)
    assert refreshed is not None
    assert refreshed.status == DocumentStatus.READY
    assert refreshed.chunk_count == 1
    assert client[1].deleted_document_ids == [document.id]
    assert client[2].calls == [document.id]


def test_knowledge_base_stats_aggregate_document_and_vector_counts(
    client: tuple[TestClient, FakeVectorStore, IngestionRunnerFake],
    db_session: Session,
    knowledge_base: KnowledgeBase,
) -> None:
    """Stats combine database lifecycle totals with the isolated vector-store count."""
    client[1].vector_count = 4
    db_session.add_all(
        [
            _document(knowledge_base, name="ready.txt", chunk_count=3),
            _document(
                knowledge_base,
                name="processing.txt",
                status=DocumentStatus.PROCESSING,
                chunk_count=1,
            ),
            _document(
                knowledge_base,
                name="failed.txt",
                status=DocumentStatus.FAILED,
                chunk_count=0,
            ),
        ]
    )
    db_session.commit()

    response = client[0].get(f"/api/v1/knowledge-bases/{knowledge_base.id}/stats")

    assert response.status_code == 200
    assert response.json() == {
        "document_count": 3,
        "ready_document_count": 1,
        "processing_document_count": 1,
        "failed_document_count": 1,
        "total_chunk_count": 4,
        "vector_count": 4,
    }
