# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - 2026-07-19

### Features

- Knowledge Base management, document upload, document lifecycle management, and statistics.
- Background document ingestion for PDF, DOCX, Markdown, and TXT files.
- Configurable text chunking, OpenAI-compatible embeddings, and persistent Chroma vector storage.
- Knowledge-base-isolated RAG retrieval with bounded context and grounded prompts.
- Traceable citations and persisted conversation/message history with citation snapshots.
- Jinja2 and native JavaScript Web UI for knowledge-base management, chat, citations, and history.
- Docker Compose deployment with persistent SQLite, upload, and Chroma volumes.
- GitHub Actions CI for Ruff linting, format checks, and pytest.

### Testing

- 105 tests passed during v1.0.0 release verification.
- Ruff lint and formatting checks passed during release verification.

### Known limitations

- No user authentication.
- No authorization or permission system.
- No SSE or streaming responses.
- Single-host deployment target.
