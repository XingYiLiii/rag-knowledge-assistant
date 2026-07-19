"""Static checks for the Docker deployment contract when Docker is unavailable in tests."""

from pathlib import Path


def test_compose_config_persists_runtime_data_and_checks_health() -> None:
    """Compose declares isolated runtime volumes and probes the public health endpoint."""
    repository_root = Path(__file__).resolve().parents[3]
    compose_file = (repository_root / "docker-compose.yml").read_text(encoding="utf-8")

    assert "8000:8000" in compose_file
    assert "DATABASE_URL: sqlite:////app/data/sqlite/rag_knowledge_assistant.db" in compose_file
    assert "sqlite_data:/app/data/sqlite" in compose_file
    assert "uploads_data:/app/data/uploads" in compose_file
    assert "chroma_data:/app/data/chroma" in compose_file
    assert "/api/v1/health" in compose_file


def test_dockerfile_uses_a_non_root_runtime_user() -> None:
    """The image installs dependencies before switching to the dedicated application user."""
    repository_root = Path(__file__).resolve().parents[3]
    dockerfile = (repository_root / "backend" / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert "pip install --no-cache-dir ." in dockerfile
    assert "USER app" in dockerfile
    assert '"app.main:app"' in dockerfile


def test_container_runtime_paths_can_be_overridden_by_environment(
    monkeypatch,
) -> None:
    """Settings accepts POSIX container paths without relying on host-specific locations."""
    from app.core.config import Settings

    monkeypatch.setenv("DATABASE_URL", "sqlite:////app/data/sqlite/runtime.db")
    monkeypatch.setenv("UPLOAD_DIRECTORY", "/app/data/uploads")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma")

    settings = Settings(_env_file=None)

    assert settings.database_url == "sqlite:////app/data/sqlite/runtime.db"
    assert settings.upload_directory.as_posix() == "/app/data/uploads"
    assert settings.chroma_persist_directory.as_posix() == "/app/data/chroma"
