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


def test_json_formatter_redacts_credential_values_in_messages() -> None:
    """Credential-like text is removed even if it accidentally reaches a log message."""
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Authorization: Bearer secret-token API_KEY=secret-api-key",
        args=(),
        exc_info=None,
    )

    rendered = formatter.format(record)

    assert "secret-token" not in rendered
    assert "secret-api-key" not in rendered
    assert "[REDACTED]" in rendered
