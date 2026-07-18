"""Tests for environment-based application settings."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings


def test_settings_read_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings validate values loaded from environment variables."""
    monkeypatch.setenv("APP_NAME", "Test API")
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_DEBUG", "true")
    monkeypatch.setenv("APP_HOST", "127.0.0.1")
    monkeypatch.setenv("APP_PORT", "9000")

    settings = Settings(_env_file=None)

    assert settings.app_name == "Test API"
    assert settings.app_version == "9.9.9"
    assert settings.app_env == "test"
    assert settings.app_debug is True
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 9000


def test_settings_reject_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Port values must remain inside the valid TCP port range."""
    monkeypatch.setenv("APP_PORT", "70000")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_returns_a_cached_instance() -> None:
    """Repeated calls reuse one process-level settings instance."""
    get_settings.cache_clear()

    assert get_settings() is get_settings()

    get_settings.cache_clear()
