"""Integration coverage for citations returned by the RAG chat endpoint."""

from collections.abc import Generator
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.api.routes.chat import get_chat_service
from app.database.models import KnowledgeBase
from app.main import create_app
from app.rag.context_builder import ContextBuildResult
from app.rag.providers import ChatResult, ChatUsage
from app.rag.retriever import RetrievedChunk
from app.services.chat_service import ChatService


@dataclass
class FakeRetriever:
    """Return a fixed set of retrieved chunks without external dependencies."""

    chunks: list[RetrievedChunk]

    def retrieve(self, _: str) -> list[RetrievedChunk]:
        return self.chunks


@dataclass
class FakeContextBuilder:
    """Deliberately retain only the selected chunk to model context budget filtering."""

    result: ContextBuildResult

    def build(self, _: list[RetrievedChunk]) -> ContextBuildResult:
        return self.result


@dataclass
class FakeChatProvider:
    """Return a local answer while recording that a prompt was received."""

    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(
            content="The deployment guide describes the supported process.",
            model="test-model",
            usage=ChatUsage(total_tokens=15),
        )


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    document_name: str,
    content: str,
    score: float,
    metadata: dict[str, object],
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        content=content,
        metadata=metadata,
        score=score,
        rank=1,
    )


@pytest.fixture()
def knowledge_base(db_session: Session) -> KnowledgeBase:
    """Persist the knowledge base used by citation API tests."""
    entity = KnowledgeBase(name="Citation target")
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)
    return entity


@pytest.fixture()
def client(db_session: Session) -> Generator[tuple[TestClient, FakeChatProvider], None, None]:
    """Build a local API client with Retriever, Context Builder, and Provider fakes."""
    selected = _chunk(
        chunk_id="selected-chunk",
        document_id="document-pdf",
        document_name="deployment.pdf",
        content="Use Docker Compose for deployment.",
        score=0.92,
        metadata={"page_number": 4},
    )
    omitted = _chunk(
        chunk_id="omitted-chunk",
        document_id="document-md",
        document_name="notes.md",
        content="This chunk does not enter the context.",
        score=0.8,
        metadata={"section_title": "Notes"},
    )
    retriever = FakeRetriever([selected, omitted])
    context_builder = FakeContextBuilder(ContextBuildResult(text="[1]\ncontent", chunks=[selected]))
    chat_provider = FakeChatProvider()
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
    yield TestClient(app), chat_provider
    app.dependency_overrides.clear()


def test_chat_returns_only_selected_context_citations(
    client: tuple[TestClient, FakeChatProvider],
    knowledge_base: KnowledgeBase,
) -> None:
    """Citation IDs and payloads map to the exact sources inserted into prompt context."""
    response = client[0].post(
        "/api/v1/chat",
        json={"knowledge_base_id": str(knowledge_base.id), "question": "How do I deploy?"},
    )

    assert response.status_code == 200
    assert response.json()["used_chunks"] == 1
    assert response.json()["citations"] == [
        {
            "citation_id": 1,
            "chunk_id": "selected-chunk",
            "document_id": "document-pdf",
            "document_name": "deployment.pdf",
            "page_number": 4,
            "section_title": None,
            "score": 0.92,
            "matched_text": "Use Docker Compose for deployment.",
        }
    ]
    assert "How do I deploy?" in client[1].calls[0][1]["content"]
