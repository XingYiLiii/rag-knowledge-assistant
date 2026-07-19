"""Unit tests for text cleaning and LangChain document chunking."""

from langchain_core.documents import Document

from app.rag.splitter import DocumentChunker, clean_text, split_documents

SOURCE_METADATA = {
    "document_id": "document-1",
    "knowledge_base_id": "knowledge-base-1",
    "original_filename": "architecture.md",
    "file_type": "markdown",
    "section_title": "Architecture",
}


def test_clean_text_preserves_paragraphs_and_code_indentation() -> None:
    """Whitespace normalization retains meaningful layout and code indentation."""
    source = "  First paragraph.  \n\n\n    code_block()\n\nSecond paragraph.   "

    assert clean_text(source) == "First paragraph.\n\n    code_block()\n\nSecond paragraph."


def test_long_text_is_split_with_ordered_metadata() -> None:
    """Long text produces bounded chunks that retain source metadata."""
    source = Document(page_content="alpha " * 30, metadata=SOURCE_METADATA)

    chunks = DocumentChunker(chunk_size=50, chunk_overlap=10).split([source])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 50 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["document_id"] == "document-1" for chunk in chunks)
    assert all(chunk.metadata["section_title"] == "Architecture" for chunk in chunks)


def test_short_text_does_not_create_empty_chunks() -> None:
    """Short non-empty input stays as one useful chunk."""
    chunks = DocumentChunker(chunk_size=100, chunk_overlap=10).split(
        [Document(page_content="Short text.", metadata=SOURCE_METADATA)]
    )

    assert [chunk.page_content for chunk in chunks] == ["Short text."]


def test_chinese_and_english_text_are_both_preserved() -> None:
    """Character-based splitting supports Chinese and English without external services."""
    source = Document(
        page_content="中文内容用于知识库检索。 English content for retrieval.",
        metadata=SOURCE_METADATA,
    )

    chunks = DocumentChunker(chunk_size=20, chunk_overlap=3).split([source])

    combined_text = "".join(chunk.page_content for chunk in chunks)
    assert "中文内容" in combined_text
    assert "English" in combined_text


def test_empty_text_produces_no_chunks() -> None:
    """Whitespace-only source documents cannot produce meaningless chunks."""
    chunks = DocumentChunker(chunk_size=100, chunk_overlap=10).split(
        [Document(page_content=" \n\n \t ", metadata=SOURCE_METADATA)]
    )

    assert chunks == []


def test_chunk_overlap_is_applied() -> None:
    """Adjacent chunks share configured text when a separator-free string is split."""
    source = Document(page_content="abcdefghijklmnop", metadata=SOURCE_METADATA)

    chunks = DocumentChunker(chunk_size=8, chunk_overlap=3).split([source])

    assert [chunk.page_content for chunk in chunks] == ["abcdefgh", "fghijklm", "klmnop"]


def test_chunk_hash_is_stable_for_identical_input() -> None:
    """The same source content and metadata always produce the same chunk hashes."""
    source = Document(page_content="stable content " * 5, metadata=SOURCE_METADATA)
    chunker = DocumentChunker(chunk_size=20, chunk_overlap=2)

    first_hashes = [chunk.metadata["chunk_hash"] for chunk in chunker.split([source])]
    second_hashes = [chunk.metadata["chunk_hash"] for chunk in chunker.split([source])]

    assert first_hashes == second_hashes
    assert all(len(chunk_hash) == 64 for chunk_hash in first_hashes)


def test_split_documents_uses_explicit_configuration() -> None:
    """Callers can override application defaults without mutating global configuration."""
    chunks = split_documents(
        [Document(page_content="abcdefghij", metadata=SOURCE_METADATA)],
        chunk_size=5,
        chunk_overlap=1,
    )

    assert [chunk.page_content for chunk in chunks] == ["abcde", "efghi", "ij"]
