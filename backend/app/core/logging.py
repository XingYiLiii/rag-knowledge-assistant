"""Structured application logging utilities."""

import json
import logging
import re
import sys
from typing import Any

_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)([^\s,;]+)"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*)([^\s,;]+)"),
)


class JsonFormatter(logging.Formatter):
    """Format application logs as machine-readable JSON without sensitive payloads."""

    context_fields = (
        "request_id",
        "method",
        "path",
        "status_code",
        "duration_ms",
        "error_code",
        "error_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Serialize whitelisted metadata and a redacted log message into JSON."""
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_values(record.getMessage()),
        }
        for field in self.context_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


def redact_sensitive_values(message: str) -> str:
    """Redact common credential representations if they reach a log message."""
    redacted_message = message
    for pattern in _SENSITIVE_VALUE_PATTERNS:
        redacted_message = pattern.sub(r"\1[REDACTED]", redacted_message)
    return redacted_message


def configure_logging() -> None:
    """Configure an idempotent JSON logger for application events."""
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if any(isinstance(handler.formatter, JsonFormatter) for handler in logger.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
