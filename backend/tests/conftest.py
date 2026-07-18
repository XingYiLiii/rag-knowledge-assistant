"""Shared database fixtures that isolate tests from the development database."""

from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.database.base import Base
from app.database.session import create_database_engine


@pytest.fixture()
def db_engine(tmp_path: Path) -> Generator[Engine, None, None]:
    """Create a temporary SQLite database dedicated to one test."""
    database_path = tmp_path / "test_rag_knowledge_assistant.db"
    database_engine = create_database_engine(f"sqlite:///{database_path.as_posix()}")

    Base.metadata.create_all(database_engine)
    yield database_engine
    Base.metadata.drop_all(database_engine)
    database_engine.dispose()


@pytest.fixture()
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    """Provide an isolated SQLAlchemy session for one test."""
    with Session(db_engine, expire_on_commit=False) as session:
        yield session
        session.rollback()
