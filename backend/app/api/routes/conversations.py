"""Conversation history management endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_database_session
from app.schemas.conversation import (
    ConversationCreate,
    ConversationMessagesRead,
    ConversationRead,
    MessageRead,
)
from app.services.conversation_service import ConversationService

router = APIRouter(tags=["conversations"])


def get_conversation_service(
    session: Annotated[Session, Depends(get_database_session)],
) -> ConversationService:
    """Build a request-scoped conversation history service."""
    return ConversationService(session)


@router.post("/conversations", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
def create_conversation(
    payload: ConversationCreate,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ConversationRead:
    """Create a conversation for one existing knowledge base."""
    return ConversationRead.model_validate(service.create(payload.knowledge_base_id))


@router.get(
    "/knowledge-bases/{knowledge_base_id}/conversations", response_model=list[ConversationRead]
)
def list_conversations(
    knowledge_base_id: UUID,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> list[ConversationRead]:
    """List conversations scoped to the requested knowledge base."""
    return [ConversationRead.model_validate(item) for item in service.list(knowledge_base_id)]


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessagesRead)
def list_messages(
    conversation_id: UUID,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> ConversationMessagesRead:
    """Return chronologically ordered persisted messages for one conversation."""
    conversation, messages = service.messages(conversation_id)
    return ConversationMessagesRead(
        conversation=ConversationRead.model_validate(conversation),
        messages=[MessageRead.model_validate(message) for message in messages],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: UUID,
    service: Annotated[ConversationService, Depends(get_conversation_service)],
) -> Response:
    """Delete only the selected conversation and its cascading message history."""
    service.delete(conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
