"""Unit tests for metadata-aware vector retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import UUID

from langchain_core.documents import Document

from app.rag.retriever import RAGRetriever

KNOWLEDGE_BASE_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
KNOWLEDGE_BASE_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


@dataclass
class FakeEmbeddingProvider:
    """Record query embedding calls without invoking a model API."""

    queries: list[str] = field(default_factory=list)

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


@dataclass
class FakeVectorStore:
    """Return preconfigured document-distance pairs and record query arguments."""

    candidates: list[tuple[Document, float]]
    queries: list[tuple[list[float], int]] = field(default_factory=list)

    def similarity_search_with_distances(
        self, query_embedding: Sequence[float], *, limit: int
    ) -> list[tuple[Document, float]]:
        self.queries.append((list(query_embedding), limit))
        return self.candidates


def _chunk(content: str, chunk_hash: str, document_id: str = "document-1") -> Document:
    return Document(
        page_content=content,
        metadata={
            "chunk_hash": chunk_hash,
            "document_id": document_id,
            "original_filename": "architecture.md",
            "file_type": "markdown",
            "knowledge_base_id": str(KNOWLEDGE_BASE_A),
        },
    )


def test_retrieve_embeds_query_and_returns_ranked_results() -> None:
    """Query embedding and vector distances produce descending ranked chunks."""
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore(
        [(_chunk("lower relevance", "b" * 64), 0.4), (_chunk("higher relevance", "a" * 64), 0.1)]
    )
    requested_knowledge_bases: list[UUID] = []
    retriever = RAGRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        vector_store_factory=lambda knowledge_base_id: (
            requested_knowledge_bases.append(knowledge_base_id) or vector_store  # type: ignore[return-value]
        ),
        top_k=3,
        score_threshold=0.0,
    )

    results = retriever.retrieve("How is retrieval designed?")

    assert embedding_provider.queries == ["How is retrieval designed?"]
    assert requested_knowledge_bases == [KNOWLEDGE_BASE_A]
    assert vector_store.queries == [([0.1, 0.2], 3)]
    assert [result.content for result in results] == ["higher relevance", "lower relevance"]
    assert [result.rank for result in results] == [1, 2]
    assert results[0].chunk_id == "a" * 64
    assert results[0].document_name == "architecture.md"


def test_retrieve_applies_threshold_deduplication_and_top_k() -> None:
    """Low-score and duplicate chunks are removed before the final top-k result."""
    vector_store = FakeVectorStore(
        [
            (_chunk("duplicate content", "a" * 64), 0.05),
            (_chunk(" duplicate   content ", "b" * 64), 0.1),
            (_chunk("second result", "c" * 64), 0.2),
            (_chunk("below threshold", "d" * 64), 0.8),
        ]
    )
    retriever = RAGRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
        vector_store_factory=lambda _: vector_store,  # type: ignore[arg-type]
        top_k=2,
        score_threshold=0.5,
    )

    results = retriever.retrieve("query")

    assert [result.content for result in results] == ["duplicate content", "second result"]
    assert [result.rank for result in results] == [1, 2]
    assert vector_store.queries[0][1] == 2


def test_retrieve_supports_per_call_top_k_and_empty_results() -> None:
    """Per-call top-k is forwarded, and empty knowledge bases return an empty list."""
    vector_store = FakeVectorStore([])
    retriever = RAGRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_B,
        embedding_provider=FakeEmbeddingProvider(),  # type: ignore[arg-type]
        vector_store_factory=lambda _: vector_store,  # type: ignore[arg-type]
        top_k=5,
        score_threshold=0.0,
    )

    assert retriever.retrieve("query", top_k=1) == []
    assert vector_store.queries == [([0.1, 0.2], 1)]


def test_blank_query_skips_embedding_and_vector_search() -> None:
    """Whitespace-only queries do not call downstream model or vector components."""
    embedding_provider = FakeEmbeddingProvider()
    vector_store = FakeVectorStore([])
    retriever = RAGRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_A,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        vector_store_factory=lambda _: vector_store,  # type: ignore[arg-type]
        top_k=3,
        score_threshold=0.0,
    )

    assert retriever.retrieve("   ") == []
    assert embedding_provider.queries == []
    assert vector_store.queries == []
