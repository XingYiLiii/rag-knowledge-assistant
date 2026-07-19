"""Persistent, knowledge-base-isolated Chroma vector storage."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from uuid import UUID

import chromadb
from langchain_core.documents import Document


class ChromaVectorStore:
    """Encapsulate Chroma operations for one knowledge base collection."""

    def __init__(self, *, knowledge_base_id: UUID, persist_directory: Path) -> None:
        self._knowledge_base_id = knowledge_base_id
        self._persist_directory = persist_directory.resolve()
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_directory))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name(knowledge_base_id),
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Upsert chunk text, metadata, and precomputed embeddings into Chroma."""
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length.")
        if not documents:
            return

        ids = [self._vector_id(document) for document in documents]
        metadatas = [self._prepare_metadata(document) for document in documents]
        self._collection.upsert(
            ids=ids,
            documents=[document.page_content for document in documents],
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=metadatas,
        )

    def delete_documents(self, *, document_id: UUID | str | None = None) -> None:
        """Delete one source document's vectors, or all vectors in this knowledge base."""
        where = {"knowledge_base_id": str(self._knowledge_base_id)}
        if document_id is not None:
            where = {"document_id": str(document_id)}
        self._collection.delete(where=where)

    def similarity_search(
        self, query_embedding: Sequence[float], *, limit: int = 4
    ) -> list[Document]:
        """Run raw vector similarity search without generating query embeddings."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=limit,
            include=["documents", "metadatas"],
        )
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        return [
            Document(page_content=page_content, metadata=metadata or {})
            for page_content, metadata in zip(documents, metadatas, strict=True)
            if page_content is not None
        ]

    def similarity_search_with_distances(
        self, query_embedding: Sequence[float], *, limit: int = 4
    ) -> list[tuple[Document, float]]:
        """Return raw Chroma cosine distances without applying retrieval policy."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        results = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        return [
            (Document(page_content=page_content, metadata=metadata or {}), float(distance))
            for page_content, metadata, distance in zip(
                documents, metadatas, distances, strict=True
            )
            if page_content is not None and distance is not None
        ]

    def count(self) -> int:
        """Return the number of vectors in this knowledge base collection."""
        return self._collection.count()

    @staticmethod
    def _collection_name(knowledge_base_id: UUID) -> str:
        return f"knowledge_base_{knowledge_base_id.hex}"

    @staticmethod
    def _vector_id(document: Document) -> str:
        try:
            chunk_hash = str(document.metadata["chunk_hash"])
        except KeyError as exc:
            raise ValueError("Chunk metadata must include chunk_hash.") from exc
        return chunk_hash

    def _prepare_metadata(self, document: Document) -> dict[str, str | int | float | bool]:
        metadata = {
            key: _as_chroma_metadata(value)
            for key, value in document.metadata.items()
            if value is not None
        }
        metadata["knowledge_base_id"] = str(self._knowledge_base_id)
        required_keys = {
            "document_id",
            "knowledge_base_id",
            "original_filename",
            "file_type",
            "chunk_index",
            "chunk_hash",
        }
        missing_keys = required_keys.difference(metadata)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Chunk metadata is missing required fields: {missing}.")
        if metadata["knowledge_base_id"] != str(self._knowledge_base_id):
            raise ValueError("Chunk knowledge_base_id does not match this vector store.")
        return metadata


def _as_chroma_metadata(value: object) -> str | int | float | bool:
    """Convert metadata to Chroma's supported scalar values deterministically."""
    if isinstance(value, str | int | float | bool):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
