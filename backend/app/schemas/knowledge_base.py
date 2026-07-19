"""Pydantic schemas for Knowledge Base management APIs."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KnowledgeBaseCreate(BaseModel):
    """Payload for creating a knowledge base."""

    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Normalize a name and reject values containing only whitespace."""
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Name must not be blank.")
        return normalized_value


class KnowledgeBaseUpdate(BaseModel):
    """Payload for updating a knowledge base."""

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        """Normalize a supplied name and reject whitespace-only values."""
        if value is None:
            return None
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Name must not be blank.")
        return normalized_value

    @model_validator(mode="after")
    def validate_update_fields(self) -> "KnowledgeBaseUpdate":
        """Require at least one field so PATCH requests are meaningful."""
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided.")
        return self


class KnowledgeBaseRead(BaseModel):
    """Public representation of a knowledge base and its document count."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime
    document_count: int


class KnowledgeBaseStatsRead(BaseModel):
    """Document-processing and vector-store statistics for one knowledge base."""

    document_count: int
    ready_document_count: int
    processing_document_count: int
    failed_document_count: int
    total_chunk_count: int
    vector_count: int
