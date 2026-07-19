# RAG Design and Evaluation

## Retrieval evaluation

`backend/evals/dataset.json` is a fixed 12-question benchmark. It includes answerable questions, unanswerable questions, and one multi-document source case. Each answerable case names an expected document and expected keywords; keywords are retained for qualitative review and future extensions, while the current automated metrics use source names only.

Run an offline, reproducible evaluation against a captured result file:

```bash
cd backend
python evals/evaluate.py --results-file path/to/results.json --top-k 4
```

The result file is a JSON array. Each item must contain the benchmark `question`, `retrieved_documents`, and `citation_documents` as arrays of source filenames. This replay mode does not call an embedding API, LLM, vector store, or database.

For a read-only evaluation of an existing knowledge base, use:

```bash
python evals/evaluate.py --knowledge-base-id <UUID> --top-k 4
```

This mode reuses `RAGRetriever`, so it may call the configured embedding provider and reads the configured Chroma collection. It does not call a chat model or modify SQLite, uploads, or vectors. In this mode citation candidates are the chunks selected by retrieval; use captured Chat API citation results when evaluating final answer citations separately.

## Metrics

- **Recall@K**: fraction of answerable benchmark cases where at least one expected source appears in the first K retrieved documents.
- **Citation Source Accuracy**: fraction of answerable cases where at least one expected source appears in the supplied citation sources.
- **Unanswerable Empty Retrieval Rate**: fraction of unanswerable cases for which no document is retrieved.

The script prints only values derived from the provided dataset and result records. It does not claim model quality, judge answer wording, optimize parameters, or infer metrics for cases that were not evaluated.

## Limitations

The benchmark is intentionally lightweight and document-name based. It does not measure semantic answer correctness, ranking quality beyond source presence, hallucination rate, or prompt-injection resilience. Those concerns need larger domain-specific datasets and separate evaluation methods.
