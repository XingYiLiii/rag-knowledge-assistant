"""Tests for SQLite database path initialization."""

from pathlib import Path

from sqlalchemy import text

from app.database.session import create_database_engine


def test_file_based_sqlite_engine_creates_missing_parent_directory(tmp_path: Path) -> None:
    """Creating a file-based SQLite engine prepares its missing parent directory."""
    database_path = tmp_path / "nested" / "runtime" / "assistant.db"
    database_engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")

    try:
        with database_engine.connect() as connection:
            assert connection.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        database_engine.dispose()

    assert database_path.parent.is_dir()
    assert database_path.is_file()
