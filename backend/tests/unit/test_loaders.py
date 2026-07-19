"""Unit tests for local multi-format document loaders."""

from pathlib import Path
from uuid import UUID

import pytest
from langchain_core.documents import Document as LangChainDocument

from app.core.exceptions import ApplicationError
from app.database.models import Document
from app.rag.loaders import load_document

FIXTURES_DIRECTORY = Path(__file__).parents[1] / "fixtures"
DOCUMENT_ID = UUID("11111111-1111-1111-1111-111111111111")
KNOWLEDGE_BASE_ID = UUID("22222222-2222-2222-2222-222222222222")


def _stored_document(filename: str, file_type: str) -> Document:
    return Document(
        id=DOCUMENT_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        original_name=filename,
        stored_name=filename,
        storage_path=filename,
        file_type=file_type,
        file_size=1,
        sha256="a" * 64,
    )


@pytest.mark.parametrize(
    ("filename", "file_type", "expected_text"),
    [
        ("sample.pdf", "pdf", "PDF fixture text"),
        ("sample.docx", "docx", "DOCX fixture text"),
        ("sample.md", "markdown", "The API uses a layered FastAPI design."),
        ("sample.txt", "txt", "Plain text fixture"),
    ],
)
def test_loads_supported_file_types(filename: str, file_type: str, expected_text: str) -> None:
    """Each supported source format becomes a LangChain document with source metadata."""
    documents = load_document(
        _stored_document(filename, file_type), storage_directory=FIXTURES_DIRECTORY
    )

    assert documents
    assert all(isinstance(document, LangChainDocument) for document in documents)
    assert expected_text in "\n".join(document.page_content for document in documents)
    assert documents[0].metadata["document_id"] == str(DOCUMENT_ID)
    assert documents[0].metadata["knowledge_base_id"] == str(KNOWLEDGE_BASE_ID)
    assert documents[0].metadata["original_filename"] == filename
    assert documents[0].metadata["file_type"] == file_type


def test_pdf_loader_keeps_page_number() -> None:
    """PDF output contains a one-based source page number."""
    documents = load_document(
        _stored_document("sample.pdf", "pdf"), storage_directory=FIXTURES_DIRECTORY
    )

    assert documents[0].metadata["page_number"] == 1


def test_markdown_loader_keeps_section_title() -> None:
    """Markdown headings are retained as section metadata."""
    documents = load_document(
        _stored_document("sample.md", "markdown"), storage_directory=FIXTURES_DIRECTORY
    )

    assert documents[0].metadata["section_title"] == "Architecture"


def test_corrupted_file_raises_parse_error(tmp_path: Path) -> None:
    """Invalid PDF content cannot masquerade as a successful parse."""
    (tmp_path / "broken.pdf").write_bytes(b"not a valid PDF")

    with pytest.raises(ApplicationError, match="could not be parsed") as error:
        load_document(_stored_document("broken.pdf", "pdf"), storage_directory=tmp_path)

    assert error.value.code == "DOCUMENT_PARSE_FAILED"


def test_unsupported_type_raises_clear_error() -> None:
    """Unsupported document records fail with an explicit application error."""
    with pytest.raises(ApplicationError, match="not supported") as error:
        load_document(
            _stored_document("sample.bin", "binary"), storage_directory=FIXTURES_DIRECTORY
        )

    assert error.value.code == "UNSUPPORTED_DOCUMENT_TYPE"
