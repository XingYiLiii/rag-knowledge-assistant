"""Reproducibly evaluate RAG retrieval and citation-source outcomes without an LLM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

DEFAULT_DATASET_PATH = Path(__file__).with_name("dataset.json")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate the fixed evaluation dataset before it is used for scoring."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON array.")
    for index, case in enumerate(payload):
        _validate_case(case, index)
    return payload


def evaluate_cases(
    dataset: Sequence[dict[str, Any]],
    results: Sequence[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, int | float]:
    """Calculate deterministic metrics from one result record per fixed dataset question."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")
    result_by_question = _index_results(results)
    answerable_cases = [case for case in dataset if not case.get("is_unanswerable", False)]
    unanswerable_cases = [case for case in dataset if case.get("is_unanswerable", False)]

    recall_hits = 0
    citation_hits = 0
    empty_unanswerable_hits = 0
    for case in dataset:
        result = result_by_question.get(case["question"])
        if result is None:
            raise ValueError(f"Missing result for question: {case['question']}")
        retrieved_documents = _document_names(result, "retrieved_documents")[:top_k]
        citation_documents = _document_names(result, "citation_documents")[:top_k]
        if case.get("is_unanswerable", False):
            empty_unanswerable_hits += int(not retrieved_documents)
            continue
        expected_documents = set(case.get("expected_documents", [case["expected_document"]]))
        recall_hits += int(bool(expected_documents.intersection(retrieved_documents)))
        citation_hits += int(bool(expected_documents.intersection(citation_documents)))

    return {
        "total_cases": len(dataset),
        "top_k": top_k,
        "answerable_cases": len(answerable_cases),
        "unanswerable_cases": len(unanswerable_cases),
        "recall_at_k": _rate(recall_hits, len(answerable_cases)),
        "citation_source_accuracy": _rate(citation_hits, len(answerable_cases)),
        "unanswerable_empty_retrieval_rate": _rate(
            empty_unanswerable_hits,
            len(unanswerable_cases),
        ),
    }


def read_results(path: Path) -> list[dict[str, Any]]:
    """Read externally captured retrieval and citation documents without changing system state."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation results must be a JSON array.")
    return payload


def retrieve_results(
    knowledge_base_id: UUID, dataset: Sequence[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Run the existing read-only retriever; this path calls embeddings but never an LLM."""
    from app.rag.retriever import create_retriever

    retriever = create_retriever(knowledge_base_id)
    results: list[dict[str, Any]] = []
    for case in dataset:
        chunks = retriever.retrieve(case["question"], top_k=top_k)
        document_names = [chunk.document_name for chunk in chunks]
        results.append(
            {
                "question": case["question"],
                "retrieved_documents": document_names,
                "citation_documents": document_names,
            }
        )
    return results


def _validate_case(case: object, index: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"Dataset item {index} must be an object.")
    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Dataset item {index} must contain a non-empty question.")
    if "expected_document" not in case:
        raise ValueError(f"Dataset item {index} must contain expected_document.")
    if not isinstance(case.get("expected_keywords"), list):
        raise ValueError(f"Dataset item {index} must contain expected_keywords as a list.")
    if case.get("is_unanswerable", False):
        if case["expected_document"] is not None:
            raise ValueError(f"Unanswerable dataset item {index} must not expect a document.")
    elif not isinstance(case["expected_document"], str) or not case["expected_document"].strip():
        raise ValueError(f"Answerable dataset item {index} must name an expected document.")


def _index_results(results: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed_results: dict[str, dict[str, Any]] = {}
    for result in results:
        question = result.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Each result must contain a non-empty question.")
        if question in indexed_results:
            raise ValueError(f"Duplicate result for question: {question}")
        _document_names(result, "retrieved_documents")
        _document_names(result, "citation_documents")
        indexed_results[question] = result
    return indexed_results


def _document_names(result: dict[str, Any], field: str) -> list[str]:
    value = result.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Result field {field} must be a list of document names.")
    return value


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def main() -> int:
    """Run an offline replay or a read-only retriever evaluation and print a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--top-k", type=int, default=4)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-file", type=Path)
    source.add_argument("--knowledge-base-id", type=UUID)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results = (
        read_results(args.results_file)
        if args.results_file is not None
        else retrieve_results(args.knowledge_base_id, dataset, args.top_k)
    )
    print(json.dumps(evaluate_cases(dataset, results, top_k=args.top_k), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
