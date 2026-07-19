"""Integration tests for persisted RAG conversation history."""

from collections.abc import Generator
from dataclasses import dataclass
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.api.routes.chat import get_chat_service
from app.database.models import Conversation, KnowledgeBase, Message
from app.main import create_app
from app.rag.context_builder import ContextBuildResult
from app.rag.providers import ChatResult, ChatUsage
from app.rag.retriever import RetrievedChunk
from app.services.chat_service import ChatService


@dataclass
class FakeRetriever:
    """Return one local chunk without embedding or vector-store calls."""

    chunk: RetrievedChunk

    def retrieve(self, _: str) -> list[RetrievedChunk]:
        return [self.chunk]


@dataclass
class FakeContextBuilder:
    """Return deterministic context containing the same local source chunk."""

    chunk: RetrievedChunk

    def build(self, _: list[RetrievedChunk]) -> ContextBuildResult:
        return ContextBuildResult(text="[1]\nDeployment instructions", chunks=[self.chunk])


class FakeChatProvider:
    """Produce a deterministic assistant answer without an external LLM."""

    def chat(self, _: list[dict[str, str]]) -> ChatResult:
        return ChatResult(
            content="Use Docker Compose for deployment.",
            model="test-model",
            usage=ChatUsage(total_tokens=12),
        )


@pytest.fixture()
def knowledge_base(db_session: Session) -> KnowledgeBase:
    """Persist the primary knowledge base used by history tests."""
    entity = KnowledgeBase(name="History target")
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)
    return entity


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Create an API client whose chat dependencies use fully local fakes."""
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        document_name="deployment.pdf",
        content="Use Docker Compose for deployment.",
        metadata={"page_number": 4},
        score=0.91,
        rank=1,
    )
    app = create_app()

    def override_database_session() -> Generator[Session, None, None]:
        yield db_session

    def override_chat_service() -> ChatService:
        return ChatService(
            db_session,
            retriever_factory=lambda _: FakeRetriever(chunk),
            context_builder_factory=lambda: FakeContextBuilder(chunk),
            chat_provider_factory=FakeChatProvider,
        )

    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_chat_service] = override_chat_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def _create_conversation(client: TestClient, knowledge_base_id: UUID) -> dict[str, str]:
    response = client.post(
        "/api/v1/conversations", json={"knowledge_base_id": str(knowledge_base_id)}
    )
    assert response.status_code == 201
    return response.json()


def test_chat_persists_user_assistant_messages_and_citation_snapshot(
    client: TestClient,
    db_session: Session,
    knowledge_base: KnowledgeBase,
) -> None:
    """Conversation-scoped chat saves the question, answer, and current citation snapshot."""
    conversation = _create_conversation(client, knowledge_base.id)

    response = client.post(
        "/api/v1/chat",
        json={
            "knowledge_base_id": str(knowledge_base.id),
            "conversation_id": conversation["id"],
            "question": "How should I deploy?",
        },
    )

    assert response.status_code == 200
    messages = db_session.scalars(
        select(Message)
        .where(Message.conversation_id == UUID(conversation["id"]))
        .order_by(Message.created_at.asc())
    ).all()
    assert [message.role.value for message in messages] == ["user", "assistant"]
    assert messages[0].content == "How should I deploy?"
    assert messages[1].content == "Use Docker Compose for deployment."
    assert messages[1].citations_json == [
        {
            "citation_id": 1,
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "document_name": "deployment.pdf",
            "page_number": 4,
            "section_title": None,
            "score": 0.91,
            "matched_text": "Use Docker Compose for deployment.",
        }
    ]

    history_response = client.get(f"/api/v1/conversations/{conversation['id']}/messages")
    assert history_response.status_code == 200
    assert [item["role"] for item in history_response.json()["messages"]] == ["user", "assistant"]
    assert history_response.json()["messages"][1]["citations"] == messages[1].citations_json


def test_list_is_scoped_to_knowledge_base_and_chat_rejects_cross_base_conversation(
    client: TestClient,
    db_session: Session,
    knowledge_base: KnowledgeBase,
) -> None:
    """Conversation IDs cannot be reused with a different knowledge base."""
    other_knowledge_base = KnowledgeBase(name="Other history target")
    db_session.add(other_knowledge_base)
    db_session.commit()
    db_session.refresh(other_knowledge_base)
    own_conversation = _create_conversation(client, knowledge_base.id)
    other_conversation = _create_conversation(client, other_knowledge_base.id)

    list_response = client.get(f"/api/v1/knowledge-bases/{knowledge_base.id}/conversations")
    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()] == [own_conversation["id"]]

    response = client.post(
        "/api/v1/chat",
        json={
            "knowledge_base_id": str(knowledge_base.id),
            "conversation_id": other_conversation["id"],
            "question": "How should I deploy?",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "CONVERSATION_NOT_FOUND",
        "message": "Conversation was not found.",
    }
    assert (
        db_session.scalar(
            select(func.count(Message.id)).where(
                Message.conversation_id == UUID(other_conversation["id"])
            )
        )
        == 0
    )


def test_deleting_conversation_cascades_messages_without_deleting_knowledge_base(
    client: TestClient,
    db_session: Session,
    knowledge_base: KnowledgeBase,
) -> None:
    """Conversation deletion removes only its message history through existing ORM cascade."""
    conversation = _create_conversation(client, knowledge_base.id)
    client.post(
        "/api/v1/chat",
        json={
            "knowledge_base_id": str(knowledge_base.id),
            "conversation_id": conversation["id"],
            "question": "How should I deploy?",
        },
    )

    delete_response = client.delete(f"/api/v1/conversations/{conversation['id']}")

    assert delete_response.status_code == 204
    assert db_session.get(Conversation, UUID(conversation["id"])) is None
    assert (
        db_session.scalar(
            select(func.count(Message.id)).where(
                Message.conversation_id == UUID(conversation["id"])
            )
        )
        == 0
    )
    assert db_session.get(KnowledgeBase, knowledge_base.id) is not None
