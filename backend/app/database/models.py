"""Core SQLAlchemy ORM models for the application domain."""

from __future__ import annotations

from enum import Enum, StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin


class DocumentStatus(StrEnum):
    """Allowed lifecycle states for an uploaded document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class MessageRole(StrEnum):
    """Allowed chat message roles."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def _enum_values(enum_class: type[Enum]) -> list[str]:
    """Persist enum values instead of their Python member names."""
    return [member.value for member in enum_class]


class KnowledgeBase(TimestampMixin, Base):
    """A logical collection of documents sharing retrieval settings."""

    __tablename__ = "knowledge_bases"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunk_size: Mapped[int] = mapped_column(Integer, default=1000, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=200, nullable=False)

    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint("chunk_overlap < chunk_size", name="ck_knowledge_bases_chunk_overlap"),
    )


class Document(TimestampMixin, Base):
    """Metadata and processing state for one uploaded knowledge document."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        SqlEnum(
            DocumentStatus,
            name="document_status",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        default=DocumentStatus.PENDING,
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")

    __table_args__ = (Index("ix_documents_knowledge_base_sha256", "knowledge_base_id", "sha256"),)


class Conversation(TimestampMixin, Base):
    """A chat session scoped to one knowledge base."""

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="conversations")
    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Message(TimestampMixin, Base):
    """One user, assistant, or system message and its answer metadata."""

    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[MessageRole] = mapped_column(
        SqlEnum(
            MessageRole,
            name="message_role",
            native_enum=False,
            create_constraint=True,
            validate_strings=True,
            values_callable=_enum_values,
        ),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations_json: Mapped[list[dict[str, object]] | None] = mapped_column(JSON, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
