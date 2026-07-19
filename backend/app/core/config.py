"""Environment-based application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or a local .env file."""

    app_name: str = "RAG Knowledge Assistant API"
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    database_url: str = "sqlite:///./data/rag_knowledge_assistant.db"
    upload_directory: Path = Path("./data/uploads")
    max_upload_file_size: int = Field(default=10 * 1024 * 1024, ge=1)
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_timeout_seconds: float = Field(default=20.0, gt=0)
    chroma_persist_directory: Path = Path("./data/chroma")
    retrieval_top_k: int = Field(default=4, gt=0, le=100)
    retrieval_score_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    context_max_length: int = Field(default=6000, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the current process."""
    return Settings()
