"""Build stable, source-traceable citation snapshots for grounded answers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class Citation:
    """A source snapshot for one chunk included in the answer context."""

    citation_id: int
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    section_title: str | None
    score: float
    matched_text: str


def build_citations(chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    """Create citations in the exact order used by the bounded prompt context."""
    return [
        Citation(
            citation_id=index,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page_number=_page_number(chunk.metadata.get("page_number")),
            section_title=_optional_text(chunk.metadata.get("section_title")),
            score=chunk.score,
            matched_text=chunk.content,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _page_number(value: object) -> int | None:
    """Normalize a loader-provided page number without raising for optional metadata."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _optional_text(value: object) -> str | None:
    """Keep non-empty textual metadata as a source locator."""
    if isinstance(value, str) and value.strip():
        return value
    return None
