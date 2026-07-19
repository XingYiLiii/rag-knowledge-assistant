"""Integration tests for public Chat API input safeguards."""

from collections.abc import Generator
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.main import create_app
from app.schemas.chat import MAX_QUESTION_LENGTH


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Create a client with the isolated test database dependency."""
    app = create_app()

    def override_database_session() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_database_session] = override_database_session
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_rejects_questions_exceeding_the_safe_length_limit(client: TestClient) -> None:
    """Oversized prompts are rejected at request validation before any RAG dependency runs."""
    response = client.post(
        "/api/v1/chat",
        json={
            "knowledge_base_id": str(uuid4()),
            "question": "x" * (MAX_QUESTION_LENGTH + 1),
        },
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "VALIDATION_ERROR",
        "message": "Request validation failed.",
    }
