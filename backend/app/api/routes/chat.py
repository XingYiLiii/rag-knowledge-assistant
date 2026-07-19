"""Non-streaming grounded RAG chat endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.core.config import get_settings
from app.rag.chat_provider import create_chat_provider
from app.rag.context_builder import ContextBuilder
from app.rag.retriever import create_retriever
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_service(
    session: Annotated[Session, Depends(get_database_session)],
) -> ChatService:
    """Build a request-scoped RAG chat service from configured collaborators."""
    settings = get_settings()
    return ChatService(
        session,
        retriever_factory=lambda knowledge_base_id: create_retriever(knowledge_base_id, settings),
        context_builder_factory=lambda: ContextBuilder(max_length=settings.context_max_length),
        chat_provider_factory=lambda: create_chat_provider(settings),
    )


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    """Answer one question using evidence retrieved from the selected knowledge base."""
    result = service.answer(
        knowledge_base_id=payload.knowledge_base_id,
        question=payload.question,
    )
    return ChatResponse(
        answer=result.answer,
        model=result.model,
        latency_ms=result.latency_ms,
        used_chunks=result.used_chunks,
    )
