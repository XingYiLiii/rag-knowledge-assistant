"""Pydantic schemas for persisted RAG conversations and messages."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.database.models import MessageRole


class ConversationCreate(BaseModel):
    """Create one conversation scoped to a knowledge base."""

    knowledge_base_id: UUID


class ConversationRead(BaseModel):
    """Public conversation metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    created_at: datetime
    updated_at: datetime


class MessageRead(BaseModel):
    """Persisted chat message with its immutable citation snapshot when applicable."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: MessageRole
    content: str
    citations: list[dict[str, object]] | None
    created_at: datetime


class ConversationMessagesRead(BaseModel):
    """Conversation metadata and messages returned together for history display."""

    conversation: ConversationRead
    messages: list[MessageRead]
