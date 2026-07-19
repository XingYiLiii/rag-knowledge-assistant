"""Unit tests for citation construction from selected prompt-context chunks."""

from app.rag.citations import build_citations
from app.rag.context_builder import ContextBuilder, ContextBuildResult
from app.rag.retriever import RetrievedChunk


def _chunk(
    *,
    chunk_id: str,
    document_id: str,
    document_name: str,
    score: float,
    rank: int,
    metadata: dict[str, object],
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        content=f"Matched text for {chunk_id}.",
        metadata=metadata,
        score=score,
        rank=rank,
    )


def test_citations_follow_the_exact_context_order_and_preserve_sources() -> None:
    """Citation IDs match Context Builder's stable score-ordered source numbering."""
    lower_score = _chunk(
        chunk_id="chunk-low",
        document_id="document-a",
        document_name="guide.md",
        score=0.6,
        rank=2,
        metadata={"section_title": "Deployment"},
    )
    higher_score = _chunk(
        chunk_id="chunk-high",
        document_id="document-b",
        document_name="manual.pdf",
        score=0.9,
        rank=1,
        metadata={"page_number": 3},
    )

    context = ContextBuilder(max_length=10_000).build([lower_score, higher_score])
    citations = build_citations(context.chunks)

    assert [citation.citation_id for citation in citations] == [1, 2]
    assert [citation.chunk_id for citation in citations] == ["chunk-high", "chunk-low"]
    assert citations[0].document_name == "manual.pdf"
    assert citations[0].page_number == 3
    assert citations[1].section_title == "Deployment"
    assert citations[1].page_number is None


def test_citations_only_include_chunks_that_entered_the_bounded_context() -> None:
    """Chunks rejected by context budget never appear in answer citation snapshots."""
    selected = _chunk(
        chunk_id="selected",
        document_id="document-a",
        document_name="selected.txt",
        score=0.9,
        rank=1,
        metadata={},
    )
    omitted = _chunk(
        chunk_id="omitted",
        document_id="document-b",
        document_name="omitted.txt",
        score=0.8,
        rank=2,
        metadata={},
    )

    context = ContextBuildResult(text="[1]\nselected", chunks=[selected])
    citations = build_citations(context.chunks)

    assert omitted not in context.chunks
    assert [citation.chunk_id for citation in citations] == ["selected"]
    assert citations[0].matched_text == selected.content
