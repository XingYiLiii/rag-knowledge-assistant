"""Persistence operations for knowledge-base-scoped conversation history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import ApplicationError
from app.database.models import Conversation, KnowledgeBase, Message


@dataclass(frozen=True)
class ConversationResult:
    """Service-layer conversation metadata."""

    id: UUID
    knowledge_base_id: UUID
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class MessageResult:
    """Service-layer snapshot of one persisted user or assistant message."""

    id: UUID
    role: str
    content: str
    citations: list[dict[str, object]] | None
    created_at: datetime


class ConversationService:
    """Manage conversations without exposing ORM entities to API routes."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, knowledge_base_id: UUID) -> ConversationResult:
        """Create an empty conversation after confirming its knowledge base exists."""
        self._ensure_knowledge_base_exists(knowledge_base_id)
        conversation = Conversation(knowledge_base_id=knowledge_base_id)
        self._session.add(conversation)
        self._session.commit()
        self._session.refresh(conversation)
        return self._to_conversation_result(conversation)

    def list(self, knowledge_base_id: UUID) -> list[ConversationResult]:
        """List conversations belonging only to the requested knowledge base."""
        self._ensure_knowledge_base_exists(knowledge_base_id)
        conversations = self._session.scalars(
            select(Conversation)
            .where(Conversation.knowledge_base_id == knowledge_base_id)
            .order_by(Conversation.updated_at.desc())
        ).all()
        return [self._to_conversation_result(conversation) for conversation in conversations]

    def messages(self, conversation_id: UUID) -> tuple[ConversationResult, list[MessageResult]]:
        """Return one conversation and messages ordered from oldest to newest."""
        conversation = self._get_conversation_or_raise(conversation_id)
        messages = self._session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        ).all()
        return self._to_conversation_result(conversation), [
            self._to_message_result(message) for message in messages
        ]

    def delete(self, conversation_id: UUID) -> None:
        """Delete a conversation and rely on ORM cascade for its messages only."""
        conversation = self._get_conversation_or_raise(conversation_id)
        self._session.delete(conversation)
        self._session.commit()

    def _ensure_knowledge_base_exists(self, knowledge_base_id: UUID) -> None:
        if self._session.get(KnowledgeBase, knowledge_base_id) is None:
            raise ApplicationError(
                code="KNOWLEDGE_BASE_NOT_FOUND",
                message="Knowledge base was not found.",
                status_code=404,
            )

    def _get_conversation_or_raise(self, conversation_id: UUID) -> Conversation:
        conversation = self._session.get(Conversation, conversation_id)
        if conversation is None:
            raise ApplicationError(
                code="CONVERSATION_NOT_FOUND",
                message="Conversation was not found.",
                status_code=404,
            )
        return conversation

    @staticmethod
    def _to_conversation_result(conversation: Conversation) -> ConversationResult:
        return ConversationResult(
            id=conversation.id,
            knowledge_base_id=conversation.knowledge_base_id,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    @staticmethod
    def _to_message_result(message: Message) -> MessageResult:
        return MessageResult(
            id=message.id,
            role=message.role.value,
            content=message.content,
            citations=message.citations_json,
            created_at=message.created_at,
        )
