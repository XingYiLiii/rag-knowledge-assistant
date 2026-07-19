"""Text cleaning and metadata-preserving LangChain document chunking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


def clean_text(text: str) -> str:
    """Normalize meaningless whitespace while preserving paragraphs and code indentation."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    previous_line_blank = True

    for line in normalized.split("\n"):
        if not line.strip():
            if not previous_line_blank:
                cleaned_lines.append("")
            previous_line_blank = True
            continue
        cleaned_lines.append(line.rstrip())
        previous_line_blank = False

    return "\n".join(cleaned_lines).strip()


class DocumentChunker:
    """Split cleaned LangChain documents with stable traceability metadata."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: Sequence[Document]) -> list[Document]:
        """Clean source documents and return non-empty chunks in deterministic order."""
        chunks: list[Document] = []
        for source_document in documents:
            cleaned_content = clean_text(source_document.page_content)
            if not cleaned_content:
                continue
            source_metadata = dict(source_document.metadata)
            for chunk_content in self._splitter.split_text(cleaned_content):
                if not chunk_content.strip():
                    continue
                metadata = {
                    **source_metadata,
                    "chunk_index": len(chunks),
                }
                metadata["chunk_hash"] = _chunk_hash(chunk_content, source_metadata)
                chunks.append(Document(page_content=chunk_content, metadata=metadata))
        return chunks


def split_documents(
    documents: Sequence[Document],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split LangChain documents using explicit values or configured application defaults."""
    settings = get_settings()
    return DocumentChunker(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
    ).split(documents)


def _chunk_hash(chunk_content: str, source_metadata: dict[str, object]) -> str:
    canonical_payload = json.dumps(
        {"content": chunk_content, "metadata": source_metadata},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()
