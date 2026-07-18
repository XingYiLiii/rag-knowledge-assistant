"""Integration tests for Knowledge Base CRUD endpoints."""

from collections.abc import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.database.models import Conversation, Document, KnowledgeBase, Message, MessageRole
from app.main import create_app


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Create an API client whose database dependency uses the test session."""
    app = create_app()

    def override_database_session() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_database_session] = override_database_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_list_and_get_knowledge_base(client: TestClient) -> None:
    """Knowledge bases can be created, listed, and fetched with their count."""
    create_response = client.post(
        "/api/v1/knowledge-bases",
        json={"name": "Engineering", "description": "Platform team documents"},
    )

    assert create_response.status_code == 201
    created = create_response.json()
    assert created["name"] == "Engineering"
    assert created["description"] == "Platform team documents"
    assert created["document_count"] == 0

    list_response = client.get("/api/v1/knowledge-bases")
    assert list_response.status_code == 200
    assert list_response.json() == [created]

    get_response = client.get(f"/api/v1/knowledge-bases/{created['id']}")
    assert get_response.status_code == 200
    assert get_response.json() == created


def test_update_knowledge_base(client: TestClient) -> None:
    """Name and description updates persist through the service layer."""
    create_response = client.post("/api/v1/knowledge-bases", json={"name": "Original"})
    knowledge_base_id = create_response.json()["id"]

    update_response = client.patch(
        f"/api/v1/knowledge-bases/{knowledge_base_id}",
        json={"name": "Renamed", "description": "Updated description"},
    )

    assert update_response.status_code == 200
    assert update_response.json()["name"] == "Renamed"
    assert update_response.json()["description"] == "Updated description"


def test_delete_knowledge_base_cascades_related_records(
    client: TestClient, db_session: Session
) -> None:
    """Deleting a knowledge base removes related documents, conversations, and messages."""
    knowledge_base = KnowledgeBase(name="Cascade target")
    knowledge_base.documents.append(
        Document(
            original_name="source.txt",
            stored_name="source.txt",
            file_type="txt",
            file_size=10,
            sha256="a" * 64,
        )
    )
    conversation = Conversation(title="Cascade conversation")
    conversation.messages.append(Message(role=MessageRole.USER, content="Question"))
    knowledge_base.conversations.append(conversation)
    db_session.add(knowledge_base)
    db_session.commit()

    response = client.delete(f"/api/v1/knowledge-bases/{knowledge_base.id}")

    assert response.status_code == 204
    assert db_session.scalar(select(func.count(Document.id))) == 0
    assert db_session.scalar(select(func.count(Conversation.id))) == 0
    assert db_session.scalar(select(func.count(Message.id))) == 0


def test_missing_knowledge_base_returns_safe_404(client: TestClient) -> None:
    """Missing resources use the existing uniform application error response."""
    response = client.get(f"/api/v1/knowledge-bases/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "KNOWLEDGE_BASE_NOT_FOUND",
        "message": "Knowledge base was not found.",
    }
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/api/v1/knowledge-bases", {"name": "   "}),
        ("patch", f"/api/v1/knowledge-bases/{uuid4()}", {}),
    ],
)
def test_invalid_payload_returns_validation_error(
    client: TestClient,
    method: str,
    path: str,
    payload: dict[str, str],
) -> None:
    """Schema validation failures use the global safe validation response."""
    response = getattr(client, method)(path, json=payload)

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "VALIDATION_ERROR",
        "message": "Request validation failed.",
    }
