"""Global exception handlers with a consistent, safe response shape."""

import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.exceptions import ApplicationError

logger = logging.getLogger("app")


def _request_id(request: Request) -> str | None:
    """Get the request identifier without assuming middleware has already run."""
    return getattr(request.state, "request_id", None)


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    """Build the public error schema without internal exception data."""
    request_id = _request_id(request)
    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message},
            "request_id": request_id,
        },
    )
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


async def application_error_handler(request: Request, exc: ApplicationError) -> JSONResponse:
    """Return known application errors in the public error schema."""
    logger.warning(
        "request.application_error",
        extra={
            "request_id": _request_id(request),
            "error_code": exc.code,
            "status_code": exc.status_code,
        },
    )
    return _error_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
    )


async def request_validation_error_handler(
    request: Request,
    _: RequestValidationError,
) -> JSONResponse:
    """Hide validation internals while retaining a stable public error response."""
    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="VALIDATION_ERROR",
        message="Request validation failed.",
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic response for unexpected failures without leaking details."""
    logger.error(
        "request.unhandled_exception",
        extra={
            "request_id": _request_id(request),
            "error_type": type(exc).__name__,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )
    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred.",
    )
