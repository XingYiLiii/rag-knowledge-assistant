"""Tests for the fixed, reproducible RAG evaluation dataset and scoring contract."""

from evals.evaluate import DEFAULT_DATASET_PATH, evaluate_cases, load_dataset


def test_eval_dataset_has_valid_required_fields() -> None:
    """Every benchmark item is a valid non-empty question with source expectations."""
    dataset = load_dataset(DEFAULT_DATASET_PATH)

    assert 10 <= len(dataset) <= 20
    assert all(case["question"].strip() for case in dataset)
    assert all("expected_document" in case for case in dataset)
    assert all(isinstance(case["expected_keywords"], list) for case in dataset)
    assert any(case.get("is_unanswerable") for case in dataset)
    assert any("expected_documents" in case for case in dataset)


def test_eval_metrics_are_deterministic_for_fixed_results() -> None:
    """Metric calculations use only supplied document names and never invoke RAG services."""
    dataset = [
        {
            "question": "Known question",
            "expected_document": "source.md",
            "expected_keywords": ["source"],
        },
        {
            "question": "Unknown question",
            "expected_document": None,
            "expected_keywords": [],
            "is_unanswerable": True,
        },
    ]
    results = [
        {
            "question": "Known question",
            "retrieved_documents": ["source.md"],
            "citation_documents": ["source.md"],
        },
        {
            "question": "Unknown question",
            "retrieved_documents": [],
            "citation_documents": [],
        },
    ]

    report = evaluate_cases(dataset, results, top_k=4)

    assert report == {
        "total_cases": 2,
        "top_k": 4,
        "answerable_cases": 1,
        "unanswerable_cases": 1,
        "recall_at_k": 1.0,
        "citation_source_accuracy": 1.0,
        "unanswerable_empty_retrieval_rate": 1.0,
    }
