"""Pydantic schemas for one non-streaming RAG chat request and response."""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """A question scoped to exactly one knowledge base."""

    knowledge_base_id: UUID
    conversation_id: UUID | None = None
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject whitespace-only questions after normalizing surrounding whitespace."""
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Question must not be blank.")
        return normalized_value


class CitationRead(BaseModel):
    """A source snapshot for one chunk used to ground the returned answer."""

    citation_id: int = Field(ge=1)
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    section_title: str | None
    score: float
    matched_text: str


class ChatResponse(BaseModel):
    """The final answer and safe request-level RAG execution metadata."""

    answer: str
    model: str | None
    latency_ms: float = Field(ge=0)
    used_chunks: int = Field(ge=0)
    citations: list[CitationRead]
