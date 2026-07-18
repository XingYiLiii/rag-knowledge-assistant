"""Unit tests for structured logging."""

import json
import logging

from app.core.logging import JsonFormatter


def test_json_formatter_serializes_safe_context_fields() -> None:
    """Structured logs include tracing metadata without exception payloads."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="request.completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "request-123"
    record.status_code = 200

    payload = json.loads(formatter.format(record))

    assert payload["message"] == "request.completed"
    assert payload["request_id"] == "request-123"
    assert payload["status_code"] == 200
    assert "timestamp" in payload
