"""Build bounded, source-separated LLM context from retrieved chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import get_settings
from app.rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class ContextBuildResult:
    """Bounded context text and the chunks represented by its stable source numbers."""

    text: str
    chunks: list[RetrievedChunk]


class ContextBuilder:
    """Format high-confidence chunks into a bounded, citation-numbered context block."""

    def __init__(self, *, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero.")
        self._max_length = max_length

    def build(self, chunks: Sequence[RetrievedChunk]) -> ContextBuildResult:
        """Keep highest-scored chunks first while respecting the configured character budget."""
        parts: list[str] = []
        included_chunks: list[RetrievedChunk] = []
        current_length = 0

        for chunk in sorted(chunks, key=lambda item: (-item.score, item.rank)):
            source_number = len(included_chunks) + 1
            source_text = _format_source(source_number, chunk)
            separator_length = 2 if parts else 0
            remaining_length = self._max_length - current_length - separator_length
            if remaining_length <= 0:
                break
            if len(source_text) > remaining_length:
                source_text = _truncate_source(source_number, chunk, remaining_length)
            if not source_text:
                break
            parts.append(source_text)
            included_chunks.append(chunk)
            current_length += separator_length + len(source_text)

        return ContextBuildResult(text="\n\n".join(parts), chunks=included_chunks)


def build_context(
    chunks: Sequence[RetrievedChunk], *, max_length: int | None = None
) -> ContextBuildResult:
    """Build context using explicit length or the configured application default."""
    settings = get_settings()
    return ContextBuilder(
        max_length=max_length if max_length is not None else settings.context_max_length
    ).build(chunks)


def _format_source(source_number: int, chunk: RetrievedChunk, *, content: str | None = None) -> str:
    metadata = chunk.metadata
    lines = [f"[{source_number}]", f"来源文件：{chunk.document_name}"]
    if page_number := metadata.get("page_number"):
        lines.append(f"页码：{page_number}")
    if section_title := metadata.get("section_title"):
        lines.append(f"章节：{section_title}")
    lines.extend(["内容：", content if content is not None else chunk.content])
    return "\n".join(lines)


def _truncate_source(source_number: int, chunk: RetrievedChunk, max_length: int) -> str:
    """Truncate only source content, retaining source metadata when it fits the budget."""
    header = _format_source(source_number, chunk, content="")
    available_content_length = max_length - len(header)
    if available_content_length <= 0:
        return ""
    truncated_content = chunk.content[: max(0, available_content_length - 1)].rstrip()
    if len(truncated_content) < len(chunk.content):
        truncated_content += "…"
    return _format_source(source_number, chunk, content=truncated_content)
