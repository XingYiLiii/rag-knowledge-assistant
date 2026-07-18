"""Integration tests for secure document uploads."""

from collections.abc import Generator
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.core.config import get_settings
from app.database.models import Document, KnowledgeBase
from app.main import create_app


@pytest.fixture()
def client(
    db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[TestClient, None, None]:
    """Create a client isolated from local database and upload storage."""
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_directory", tmp_path / "uploads")
    monkeypatch.setattr(settings, "max_upload_file_size", 32)
    app = create_app()

    def override_database_session() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_database_session] = override_database_session
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def knowledge_base(db_session: Session) -> KnowledgeBase:
    """Persist a knowledge base available for upload tests."""
    entity = KnowledgeBase(name="Upload target")
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)
    return entity


@pytest.mark.parametrize(
    ("filename", "content", "expected_type"),
    [("guide.pdf", b"pdf", "pdf"), ("notes.txt", b"notes", "txt")],
)
def test_upload_supported_document(
    client: TestClient,
    knowledge_base: KnowledgeBase,
    filename: str,
    content: bytes,
    expected_type: str,
    db_session: Session,
) -> None:
    """PDF and TXT uploads are stored with pending document records."""
    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents",
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["knowledge_base_id"] == str(knowledge_base.id)
    assert payload["original_name"] == filename
    assert payload["file_type"] == expected_type
    assert payload["file_size"] == len(content)
    assert payload["status"] == "pending"
    assert len(payload["id"]) == 36

    document = db_session.get(Document, UUID(payload["id"]))
    assert document is not None
    assert document.storage_path is not None
    stored_file = get_settings().upload_directory / document.storage_path
    assert stored_file.is_file()
    assert stored_file.name != filename


def test_rejects_unsupported_file_type(client: TestClient, knowledge_base: KnowledgeBase) -> None:
    """Unsupported extensions return a safe business error."""
    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents",
        files={"file": ("script.exe", b"binary", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_rejects_empty_file(client: TestClient, knowledge_base: KnowledgeBase) -> None:
    """Empty uploads do not create database records."""
    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_FILE"


def test_rejects_oversized_file(client: TestClient, knowledge_base: KnowledgeBase) -> None:
    """Files over the configured limit are rejected before storage."""
    response = client.post(
        f"/api/v1/knowledge-bases/{knowledge_base.id}/documents",
        files={"file": ("large.txt", b"x" * 33, "text/plain")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_missing_knowledge_base_returns_safe_404(client: TestClient) -> None:
    """Upload requests keep the existing knowledge-base 404 contract."""
    response = client.post(
        f"/api/v1/knowledge-bases/{uuid4()}/documents",
        files={"file": ("notes.txt", b"notes", "text/plain")},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "KNOWLEDGE_BASE_NOT_FOUND"


def test_rejects_duplicate_file(
    client: TestClient, knowledge_base: KnowledgeBase, db_session: Session
) -> None:
    """The same SHA-256 cannot be stored twice in one knowledge base."""
    url = f"/api/v1/knowledge-bases/{knowledge_base.id}/documents"
    first_response = client.post(files={"file": ("notes.txt", b"same", "text/plain")}, url=url)
    duplicate_response = client.post(files={"file": ("copy.txt", b"same", "text/plain")}, url=url)

    assert first_response.status_code == 201
    assert duplicate_response.status_code == 409
    assert duplicate_response.json()["error"]["code"] == "DUPLICATE_DOCUMENT"
    assert db_session.scalar(select(func.count(Document.id))) == 1
