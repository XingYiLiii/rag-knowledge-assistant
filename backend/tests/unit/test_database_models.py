"""Tests for ORM model structure, relationships, and SQLite constraints."""

from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from app.database.models import (
    Conversation,
    Document,
    DocumentStatus,
    KnowledgeBase,
    Message,
    MessageRole,
)
from app.database.session import engine


def test_test_database_isolated_from_runtime_database(db_engine) -> None:
    """Temporary test databases never point at the configured runtime database."""
    assert db_engine.url.database != engine.url.database


def test_all_core_tables_are_created(db_engine) -> None:
    """The metadata creates all four v1 domain tables."""
    table_names = set(inspect(db_engine).get_table_names())

    assert {"knowledge_bases", "documents", "conversations", "messages"}.issubset(table_names)


def test_models_persist_relationships_and_citations(db_session: Session) -> None:
    """Knowledge bases own documents and conversations, which own messages."""
    knowledge_base = KnowledgeBase(name="Engineering Handbook")
    document = Document(
        original_name="handbook.pdf",
        stored_name="a1b2c3.pdf",
        file_type="pdf",
        file_size=1024,
        sha256="a" * 64,
        status=DocumentStatus.READY,
    )
    conversation = Conversation(title="Leave policy")
    message = Message(
        role=MessageRole.ASSISTANT,
        content="Employees receive annual leave according to policy.",
        citations_json=[
            {
                "document_id": str(uuid4()),
                "matched_text": "Annual leave policy",
            }
        ],
        model_name="test-model",
        latency_ms=42,
    )
    knowledge_base.documents.append(document)
    knowledge_base.conversations.append(conversation)
    conversation.messages.append(message)
    db_session.add(knowledge_base)
    db_session.commit()

    persisted_document = db_session.get(Document, document.id)
    persisted_conversation = db_session.get(Conversation, conversation.id)
    persisted_message = db_session.get(Message, message.id)

    assert persisted_document is not None
    assert persisted_document.knowledge_base_id == knowledge_base.id
    assert persisted_conversation is not None
    assert persisted_conversation.knowledge_base_id == knowledge_base.id
    assert persisted_message is not None
    assert persisted_message.conversation_id == conversation.id
    assert persisted_message.citations_json == message.citations_json


def test_deleting_knowledge_base_cascades_to_related_records(db_session: Session) -> None:
    """Deleting a knowledge base removes its documents, conversations, and messages."""
    knowledge_base = KnowledgeBase(name="Temporary Knowledge Base")
    knowledge_base.documents.append(
        Document(
            original_name="temporary.txt",
            stored_name="temporary.txt",
            file_type="txt",
            file_size=10,
            sha256="b" * 64,
        )
    )
    conversation = Conversation(title="Temporary conversation")
    conversation.messages.append(Message(role=MessageRole.USER, content="Temporary question"))
    knowledge_base.conversations.append(conversation)
    db_session.add(knowledge_base)
    db_session.commit()

    db_session.delete(knowledge_base)
    db_session.commit()

    assert db_session.query(Document).count() == 0
    assert db_session.query(Conversation).count() == 0
    assert db_session.query(Message).count() == 0


def test_document_status_rejects_invalid_values(db_session: Session) -> None:
    """Document status values are constrained to the declared enum."""
    knowledge_base = KnowledgeBase(name="Status Validation")
    invalid_document = Document(
        original_name="invalid.txt",
        stored_name="invalid.txt",
        file_type="txt",
        file_size=1,
        sha256="c" * 64,
        status="not-a-valid-status",
    )
    knowledge_base.documents.append(invalid_document)
    db_session.add(knowledge_base)

    with pytest.raises(StatementError):
        db_session.flush()
