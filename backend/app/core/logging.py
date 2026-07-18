"""Structured application logging utilities."""

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format application logs as machine-readable JSON without exception payloads."""

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
        """Serialize safe log metadata into one JSON record."""
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self.context_fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        return json.dumps(payload, ensure_ascii=False, default=str)


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
