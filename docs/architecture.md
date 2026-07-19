# Architecture

```text
Browser
  -> FastAPI
  -> Service Layer
  -> RAG Pipeline
  -> Vector Store + LLM
```

## Components

### Browser

The Jinja2 Web UI provides knowledge-base management, document operations, Chat, Citation display, and conversation history. Its native JavaScript calls only the documented `/api/v1` endpoints.

### FastAPI

FastAPI exposes API routers for health, knowledge bases, documents, conversations, and Chat. It also renders the Web UI pages and mounts static CSS and JavaScript assets.

### Service Layer

Services coordinate database lifecycle rules without leaking ORM operations to routes. `KnowledgeBaseService`, `DocumentService`, `ConversationService`, and `ChatService` are the primary application services.

### RAG Pipeline

The ingestion pipeline parses uploaded files, cleans and splits content, creates embeddings, writes vectors to Chroma, and updates document status. The answer path retrieves chunks, builds bounded context, constructs the grounded Prompt, invokes the Chat Provider, and creates Citation snapshots.

### Vector Store + LLM

Chroma collections are isolated by knowledge base. OpenAI-compatible providers support configurable Embedding and Chat endpoints. SQLite stores application metadata, document status, conversations, messages, and Citation snapshots.

## Data boundaries

- SQLite stores relational application data.
- Local storage holds uploaded source files.
- Chroma stores Chunk vectors and metadata.
- Retrieved knowledge remains untrusted context and never becomes a System Message.
- Conversation history persists answer-time Citation snapshots so historical sources do not require a later vector query.
