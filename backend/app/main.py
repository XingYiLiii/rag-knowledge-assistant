"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.router import api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG Knowledge Assistant API",
        version="0.1.0",
        description="API for the RAG Knowledge Assistant portfolio project.",
    )
    app.include_router(api_router)
    return app


app = create_app()
