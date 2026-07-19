"""Unit tests for source-separated, bounded context construction."""

from app.rag.context_builder import ContextBuilder
from app.rag.retriever import RetrievedChunk


def _chunk(*, rank: int, score: float, content: str, name: str = "guide.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"hash-{rank}",
        document_id="document-1",
        document_name=name,
        content=content,
        metadata={"page_number": 3, "section_title": "Installation"},
        score=score,
        rank=rank,
    )


def test_context_numbers_sources_in_score_order() -> None:
    """Higher-score chunks appear first with stable, sequential source numbers."""
    result = ContextBuilder(max_length=1000).build(
        [
            _chunk(rank=2, score=0.7, content="second evidence"),
            _chunk(rank=1, score=0.9, content="first evidence"),
        ]
    )

    assert result.text.index("[1]") < result.text.index("[2]")
    assert result.text.index("first evidence") < result.text.index("second evidence")
    assert [chunk.rank for chunk in result.chunks] == [1, 2]
    assert "来源文件：guide.pdf" in result.text
    assert "页码：3" in result.text
    assert "章节：Installation" in result.text


def test_context_truncates_to_length_budget_with_high_score_priority() -> None:
    """A bounded context retains the higher-score source and truncates only its content."""
    result = ContextBuilder(max_length=70).build(
        [
            _chunk(rank=2, score=0.4, content="low score evidence should be excluded"),
            _chunk(rank=1, score=0.9, content="high score evidence that is intentionally long"),
        ]
    )

    assert len(result.text) <= 70
    assert "[1]" in result.text
    assert "[2]" not in result.text
    assert result.chunks == [
        _chunk(rank=1, score=0.9, content="high score evidence that is intentionally long")
    ]


def test_context_uses_retrieval_rank_for_equal_scores() -> None:
    """Equal scores preserve the original retriever rank deterministically."""
    result = ContextBuilder(max_length=1000).build(
        [
            _chunk(rank=2, score=0.8, content="rank two"),
            _chunk(rank=1, score=0.8, content="rank one"),
        ]
    )

    assert result.text.index("rank one") < result.text.index("rank two")
