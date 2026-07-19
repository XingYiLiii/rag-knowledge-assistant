"""Unit tests for the OpenAI-compatible chat provider."""

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import ApplicationError
from app.rag.chat_provider import OpenAICompatibleChatProvider, create_chat_provider
from app.rag.providers import ChatProvider


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]
    model: str
    usage: FakeUsage


class FakeCompletions:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class FakeClient:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.chat = type("Chat", (), {"completions": FakeCompletions(response)})()


MESSAGES = [
    {"role": "system", "content": "system rule"},
    {"role": "user", "content": "secret user question"},
]


def _response() -> FakeResponse:
    return FakeResponse(
        choices=[FakeChoice(message=FakeMessage(content="assistant answer"))],
        model="response-model",
        usage=FakeUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_factory_reads_config_and_chat_parses_usage() -> None:
    """The factory forwards compatible configuration and returns normalized usage."""
    received: dict[str, Any] = {}
    client = FakeClient(_response())

    def client_factory(**kwargs: Any) -> FakeClient:
        received.update(kwargs)
        return client

    provider = create_chat_provider(
        Settings(
            llm_base_url="https://chat.example/v1",
            llm_api_key="test-secret-key",
            llm_model="chat-model",
            llm_timeout_seconds=12.5,
            llm_temperature=0.3,
        ),
        client_factory=client_factory,
    )
    result = provider.chat(MESSAGES)

    assert isinstance(provider, ChatProvider)
    assert received == {
        "api_key": "test-secret-key",
        "base_url": "https://chat.example/v1",
        "timeout": 12.5,
    }
    assert client.chat.completions.calls == [
        {"model": "chat-model", "messages": MESSAGES, "temperature": 0.3}
    ]
    assert result.content == "assistant answer"
    assert result.model == "response-model"
    assert result.usage.total_tokens == 15


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (TimeoutError("test-secret-key timeout"), "CHAT_TIMEOUT"),
        (OSError("test-secret-key network"), "CHAT_NETWORK_ERROR"),
    ],
)
def test_transport_errors_are_safe_and_do_not_log_sensitive_content(
    error: Exception, expected_code: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Provider errors expose safe codes without logging the key or full prompt."""
    provider = OpenAICompatibleChatProvider(
        client=FakeClient(error),
        model="chat-model",
        temperature=0.2,
    )

    with caplog.at_level(logging.INFO):
        with pytest.raises(ApplicationError) as raised_error:
            provider.chat(MESSAGES)

    assert raised_error.value.code == expected_code
    assert "test-secret-key" not in raised_error.value.message
    assert "test-secret-key" not in caplog.text
    assert "secret user question" not in caplog.text


def test_provider_api_error_is_converted_to_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider API errors use a stable error code without exposing payloads."""

    class FakeProviderError(Exception):
        pass

    monkeypatch.setattr("app.rag.chat_provider.APIError", FakeProviderError)
    provider = OpenAICompatibleChatProvider(
        client=FakeClient(FakeProviderError("test-secret-key provider failure")),
        model="chat-model",
        temperature=0.2,
    )

    with pytest.raises(ApplicationError) as raised_error:
        provider.chat(MESSAGES)

    assert raised_error.value.code == "CHAT_PROVIDER_ERROR"
    assert "test-secret-key" not in raised_error.value.message


def test_missing_llm_key_returns_safe_configuration_error() -> None:
    """Tests and local startup do not require a real LLM API key."""
    with pytest.raises(ApplicationError) as raised_error:
        create_chat_provider(Settings(llm_api_key=None))

    assert raised_error.value.code == "CHAT_CONFIGURATION_ERROR"
