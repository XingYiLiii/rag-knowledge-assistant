"""Integration tests for the non-streaming end-to-end RAG chat endpoint."""

from collections.abc import Generator
from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.api.routes.chat import get_chat_service
from app.core.exceptions import ApplicationError
from app.database.models import KnowledgeBase
from app.main import create_app
from app.rag.context_builder import ContextBuildResult
from app.rag.providers import ChatResult, ChatUsage
from app.rag.retriever import RetrievedChunk
from app.services.chat_service import NO_RELEVANT_KNOWLEDGE_ANSWER, ChatService


@dataclass
class FakeRetriever:
    """Retriever substitute that makes the received query observable."""

    chunks: list[RetrievedChunk]
    error: Exception | None = None
    queries: list[str] = field(default_factory=list)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        return self.chunks


@dataclass
class FakeContextBuilder:
    """Context builder substitute that keeps the test fully local."""

    result: ContextBuildResult
    calls: list[list[RetrievedChunk]] = field(default_factory=list)

    def build(self, chunks: list[RetrievedChunk]) -> ContextBuildResult:
        self.calls.append(chunks)
        return self.result


@dataclass
class FakeChatProvider:
    """Chat provider substitute that records messages without calling an LLM."""

    result: ChatResult | None = None
    error: Exception | None = None
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@pytest.fixture()
def knowledge_base(db_session: Session) -> KnowledgeBase:
    """Persist the knowledge base used by chat endpoint tests."""
    entity = KnowledgeBase(name="Chat target")
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)
    return entity


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        document_name="guide.txt",
        content="The supported deployment uses Docker Compose.",
        metadata={"original_filename": "guide.txt"},
        score=0.9,
        rank=1,
    )


@pytest.fixture()
def chat_collaborators() -> tuple[FakeRetriever, FakeContextBuilder, FakeChatProvider]:
    """Create deterministic fakes for every RAG execution collaborator."""
    chunk = _chunk()
    return (
        FakeRetriever(chunks=[chunk]),
        FakeContextBuilder(ContextBuildResult(text="[1]\ncontent", chunks=[chunk])),
        FakeChatProvider(
            result=ChatResult(
                content="Use Docker Compose for deployment.",
                model="test-model",
                usage=ChatUsage(total_tokens=12),
            )
        ),
    )


@pytest.fixture()
def client(
    db_session: Session,
    chat_collaborators: tuple[FakeRetriever, FakeContextBuilder, FakeChatProvider],
) -> Generator[TestClient, None, None]:
    """Create an API client with database and all RAG collaborators overridden."""
    retriever, context_builder, chat_provider = chat_collaborators
    app = create_app()

    def override_database_session() -> Generator[Session, None, None]:
        yield db_session

    def override_chat_service() -> ChatService:
        return ChatService(
            db_session,
            retriever_factory=lambda _: retriever,
            context_builder_factory=lambda: context_builder,
            chat_provider_factory=lambda: chat_provider,
        )

    app.dependency_overrides[get_database_session] = override_database_session
    app.dependency_overrides[get_chat_service] = override_chat_service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_chat_runs_the_grounded_rag_flow(
    client: TestClient,
    knowledge_base: KnowledgeBase,
    chat_collaborators: tuple[FakeRetriever, FakeContextBuilder, FakeChatProvider],
) -> None:
    """The endpoint chains retrieval, context construction, prompting, and generation."""
    response = client.post(
        "/api/v1/chat",
        json={"knowledge_base_id": str(knowledge_base.id), "question": "How should I deploy?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Use Docker Compose for deployment."
    assert response.json()["model"] == "test-model"
    assert response.json()["used_chunks"] == 1
    assert response.json()["latency_ms"] >= 0
    retriever, context_builder, chat_provider = chat_collaborators
    assert retriever.queries == ["How should I deploy?"]
    assert context_builder.calls == [[_chunk()]]
    assert chat_provider.calls[0][0]["role"] == "system"
    assert "How should I deploy?" in chat_provider.calls[0][1]["content"]


def test_chat_without_retrieval_results_skips_llm(
    client: TestClient,
    knowledge_base: KnowledgeBase,
    chat_collaborators: tuple[FakeRetriever, FakeContextBuilder, FakeChatProvider],
) -> None:
    """No retrieval evidence returns a clear local answer without invoking the LLM."""
    retriever, _, chat_provider = chat_collaborators
    retriever.chunks = []

    response = client.post(
        "/api/v1/chat",
        json={"knowledge_base_id": str(knowledge_base.id), "question": "Unknown topic"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": NO_RELEVANT_KNOWLEDGE_ANSWER,
        "model": None,
        "latency_ms": response.json()["latency_ms"],
        "used_chunks": 0,
        "citations": [],
    }
    assert chat_provider.calls == []


def test_chat_retrieval_failure_uses_safe_error_response(
    client: TestClient,
    knowledge_base: KnowledgeBase,
    chat_collaborators: tuple[FakeRetriever, FakeContextBuilder, FakeChatProvider],
) -> None:
    """Unexpected retrieval failures are converted without exposing internal details."""
    retriever, _, chat_provider = chat_collaborators
    retriever.error = RuntimeError("internal retrieval failure")

    response = client.post(
        "/api/v1/chat",
        json={"knowledge_base_id": str(knowledge_base.id), "question": "How should I deploy?"},
    )

    assert response.status_code == 502
    assert response.json()["error"] == {
        "code": "RETRIEVAL_FAILED",
        "message": "Knowledge retrieval could not be completed.",
    }
    assert chat_provider.calls == []


def test_chat_provider_failure_uses_safe_error_response(
    client: TestClient,
    knowledge_base: KnowledgeBase,
    chat_collaborators: tuple[FakeRetriever, FakeContextBuilder, FakeChatProvider],
) -> None:
    """Provider errors propagate through the existing uniform error handler."""
    _, _, chat_provider = chat_collaborators
    chat_provider.error = ApplicationError(
        code="CHAT_PROVIDER_ERROR",
        message="The chat provider request failed.",
        status_code=502,
    )

    response = client.post(
        "/api/v1/chat",
        json={"knowledge_base_id": str(knowledge_base.id), "question": "How should I deploy?"},
    )

    assert response.status_code == 502
    assert response.json()["error"] == {
        "code": "CHAT_PROVIDER_ERROR",
        "message": "The chat provider request failed.",
    }


def test_chat_rejects_a_missing_knowledge_base(client: TestClient) -> None:
    """A missing knowledge base is checked before retrieval or generation begins."""
    response = client.post(
        "/api/v1/chat",
        json={"knowledge_base_id": str(uuid4()), "question": "How should I deploy?"},
    )

    assert response.status_code == 404
    assert response.json()["error"] == {
        "code": "KNOWLEDGE_BASE_NOT_FOUND",
        "message": "Knowledge base was not found.",
    }
