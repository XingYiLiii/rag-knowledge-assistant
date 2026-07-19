"""OpenAI-compatible, non-streaming chat provider implementation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from app.core.config import Settings, get_settings
from app.core.exceptions import ApplicationError
from app.rag.providers import ChatMessage, ChatProvider, ChatResult, ChatUsage

logger = logging.getLogger("app")


class OpenAICompatibleChatProvider(ChatProvider):
    """Adapt OpenAI-compatible chat completions to the provider-neutral contract."""

    def __init__(self, *, client: Any, model: str, temperature: float) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def chat(self, messages: Sequence[ChatMessage]) -> ChatResult:
        """Call a non-streaming chat completion without logging prompt content."""
        started_at = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                temperature=self._temperature,
            )
        except (APITimeoutError, TimeoutError) as exc:
            raise ApplicationError(
                code="CHAT_TIMEOUT",
                message="The chat provider request timed out.",
                status_code=504,
            ) from exc
        except (APIConnectionError, OSError) as exc:
            raise ApplicationError(
                code="CHAT_NETWORK_ERROR",
                message="The chat provider could not be reached.",
                status_code=503,
            ) from exc
        except APIError as exc:
            raise ApplicationError(
                code="CHAT_PROVIDER_ERROR",
                message="The chat provider request failed.",
                status_code=502,
            ) from exc

        result = _parse_chat_response(response, fallback_model=self._model)
        logger.info(
            "chat_provider.completed",
            extra={
                "model": result.model,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            },
        )
        return result


def create_chat_provider(
    settings: Settings | None = None,
    *,
    client_factory: Callable[..., Any] = OpenAI,
) -> ChatProvider:
    """Build the configured provider without exposing vendor implementation details."""
    resolved_settings = settings or get_settings()
    if (
        resolved_settings.llm_api_key is None
        or not resolved_settings.llm_api_key.get_secret_value()
    ):
        raise ApplicationError(
            code="CHAT_CONFIGURATION_ERROR",
            message="An LLM API key must be configured.",
            status_code=500,
        )
    client = client_factory(
        api_key=resolved_settings.llm_api_key.get_secret_value(),
        base_url=resolved_settings.llm_base_url,
        timeout=resolved_settings.llm_timeout_seconds,
    )
    return OpenAICompatibleChatProvider(
        client=client,
        model=resolved_settings.llm_model,
        temperature=resolved_settings.llm_temperature,
    )


def _parse_chat_response(response: Any, *, fallback_model: str) -> ChatResult:
    """Normalize the minimal OpenAI-compatible response shape without exposing payloads."""
    try:
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise TypeError
        usage = getattr(response, "usage", None)
        return ChatResult(
            content=content,
            model=str(getattr(response, "model", fallback_model)),
            usage=ChatUsage(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
        )
    except (AttributeError, IndexError, TypeError) as exc:
        raise ApplicationError(
            code="CHAT_RESPONSE_INVALID",
            message="The chat provider returned an invalid response.",
            status_code=502,
        ) from exc
