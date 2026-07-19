"""Metadata-aware retrieval over one knowledge base's vector collection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from app.core.config import Settings, get_settings
from app.rag.embeddings import create_embedding_provider
from app.rag.providers import EmbeddingProvider
from app.rag.vector_store import ChromaVectorStore

VectorStoreFactory = Callable[[UUID], ChromaVectorStore]


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked, source-traceable chunk returned by retrieval."""

    chunk_id: str
    document_id: str
    document_name: str
    content: str
    metadata: dict[str, object]
    score: float
    rank: int


class RAGRetriever:
    """Embed a query and apply retrieval policy over an isolated vector store."""

    def __init__(
        self,
        *,
        knowledge_base_id: UUID,
        embedding_provider: EmbeddingProvider,
        vector_store_factory: VectorStoreFactory,
        top_k: int,
        score_threshold: float,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self._knowledge_base_id = knowledge_base_id
        self._embedding_provider = embedding_provider
        self._vector_store_factory = vector_store_factory
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return deduplicated chunks above threshold, sorted from highest relevance."""
        if not query.strip():
            return []
        effective_top_k = top_k if top_k is not None else self._top_k
        if effective_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self._embedding_provider.embed_query(query)
        candidates = self._vector_store_factory(
            self._knowledge_base_id
        ).similarity_search_with_distances(
            query_embedding,
            limit=effective_top_k,
        )
        filtered_candidates = sorted(
            (
                (document, 1.0 - distance)
                for document, distance in candidates
                if 1.0 - distance >= self._score_threshold
            ),
            key=lambda candidate: candidate[1],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        seen_content: set[str] = set()
        for document, score in filtered_candidates:
            deduplication_key = " ".join(document.page_content.split())
            if not deduplication_key or deduplication_key in seen_content:
                continue
            seen_content.add(deduplication_key)
            metadata = dict(document.metadata)
            results.append(
                RetrievedChunk(
                    chunk_id=str(metadata.get("chunk_hash", "")),
                    document_id=str(metadata.get("document_id", "")),
                    document_name=str(metadata.get("original_filename", "")),
                    content=document.page_content,
                    metadata=metadata,
                    score=score,
                    rank=len(results) + 1,
                )
            )
            if len(results) == effective_top_k:
                break
        return results


def create_retriever(knowledge_base_id: UUID, settings: Settings | None = None) -> RAGRetriever:
    """Build the configured retriever for a single knowledge base collection."""
    resolved_settings = settings or get_settings()
    return RAGRetriever(
        knowledge_base_id=knowledge_base_id,
        embedding_provider=create_embedding_provider(resolved_settings),
        vector_store_factory=lambda knowledge_base: ChromaVectorStore(
            knowledge_base_id=knowledge_base,
            persist_directory=resolved_settings.chroma_persist_directory,
        ),
        top_k=resolved_settings.retrieval_top_k,
        score_threshold=resolved_settings.retrieval_score_threshold,
    )
