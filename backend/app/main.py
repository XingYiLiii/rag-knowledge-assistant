"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.api.router import api_router
from app.core.config import get_settings
from app.core.error_handlers import (
    application_error_handler,
    request_validation_error_handler,
    unhandled_exception_handler,
)
from app.core.exceptions import ApplicationError
from app.core.logging import configure_logging
from app.core.middleware import RequestIDMiddleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="API for the RAG Knowledge Assistant portfolio project.",
        debug=settings.app_debug,
    )
    app.add_middleware(RequestIDMiddleware)
    app.add_exception_handler(ApplicationError, application_error_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.include_router(api_router)
    return app


app = create_app()
