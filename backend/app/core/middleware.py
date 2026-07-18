"""HTTP middleware shared by all application routes."""

import logging
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("app")


def _request_id_from_header(value: str | None) -> str:
    """Reuse a valid client request ID or create a server-generated UUID."""
    if value:
        try:
            return str(UUID(value))
        except ValueError:
            pass
    return str(uuid4())


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a safe request identifier to every request, response, and log entry."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process a request while recording its identifier and duration."""
        request_id = _request_id_from_header(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started_at = perf_counter()

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request.completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return response
