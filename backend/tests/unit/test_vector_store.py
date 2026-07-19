"""Unit tests for persistent, knowledge-base-isolated Chroma storage."""

from pathlib import Path
from uuid import UUID

from langchain_core.documents import Document

from app.rag.vector_store import ChromaVectorStore

KNOWLEDGE_BASE_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
KNOWLEDGE_BASE_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
DOCUMENT_A = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_B = UUID("22222222-2222-2222-2222-222222222222")


def _chunk(
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    chunk_index: int,
    chunk_hash: str,
    text: str,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "document_id": str(document_id),
            "knowledge_base_id": str(knowledge_base_id),
            "original_filename": "source.txt",
            "file_type": "txt",
            "chunk_index": chunk_index,
            "chunk_hash": chunk_hash,
        },
    )


def test_add_documents_persists_after_client_restart(tmp_path: Path) -> None:
    """Stored vectors are available from a newly created persistent client."""
    document = _chunk(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        document_id=DOCUMENT_A,
        chunk_index=0,
        chunk_hash="a" * 64,
        text="first chunk",
    )
    store = ChromaVectorStore(knowledge_base_id=KNOWLEDGE_BASE_A, persist_directory=tmp_path)

    store.add_documents([document], [[0.1, 0.2, 0.3]])

    restarted_store = ChromaVectorStore(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        persist_directory=tmp_path,
    )
    assert restarted_store.count() == 1


def test_collections_are_isolated_by_knowledge_base(tmp_path: Path) -> None:
    """Different knowledge bases cannot read each other's vectors."""
    store_a = ChromaVectorStore(knowledge_base_id=KNOWLEDGE_BASE_A, persist_directory=tmp_path)
    store_b = ChromaVectorStore(knowledge_base_id=KNOWLEDGE_BASE_B, persist_directory=tmp_path)
    store_a.add_documents(
        [
            _chunk(
                knowledge_base_id=KNOWLEDGE_BASE_A,
                document_id=DOCUMENT_A,
                chunk_index=0,
                chunk_hash="a" * 64,
                text="knowledge base A",
            )
        ],
        [[1.0, 0.0]],
    )

    assert store_a.count() == 1
    assert store_b.count() == 0
    assert store_b.similarity_search([1.0, 0.0]) == []


def test_duplicate_chunk_hash_upserts_instead_of_duplicating(tmp_path: Path) -> None:
    """Stable vector IDs make repeated writes idempotent."""
    store = ChromaVectorStore(knowledge_base_id=KNOWLEDGE_BASE_A, persist_directory=tmp_path)
    original = _chunk(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        document_id=DOCUMENT_A,
        chunk_index=0,
        chunk_hash="a" * 64,
        text="original text",
    )
    updated = _chunk(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        document_id=DOCUMENT_A,
        chunk_index=0,
        chunk_hash="a" * 64,
        text="updated text",
    )

    store.add_documents([original], [[1.0, 0.0]])
    store.add_documents([updated], [[1.0, 0.0]])

    assert store.count() == 1
    assert store.similarity_search([1.0, 0.0])[0].page_content == "updated text"


def test_delete_by_document_id_updates_count(tmp_path: Path) -> None:
    """Deleting one source document removes only its corresponding vectors."""
    store = ChromaVectorStore(knowledge_base_id=KNOWLEDGE_BASE_A, persist_directory=tmp_path)
    first_chunk = _chunk(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        document_id=DOCUMENT_A,
        chunk_index=0,
        chunk_hash="a" * 64,
        text="first document",
    )
    second_chunk = _chunk(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        document_id=DOCUMENT_B,
        chunk_index=0,
        chunk_hash="b" * 64,
        text="second document",
    )
    store.add_documents([first_chunk, second_chunk], [[1.0, 0.0], [0.0, 1.0]])

    store.delete_documents(document_id=DOCUMENT_A)

    assert store.count() == 1
    result = store.similarity_search([0.0, 1.0])
    assert [document.metadata["document_id"] for document in result] == [str(DOCUMENT_B)]
