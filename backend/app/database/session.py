"""SQLAlchemy engine, sessions, and database initialization helpers."""

from collections.abc import Generator
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.database.base import Base


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _: Any) -> None:
    """Enable foreign-key enforcement for SQLite connections."""
    dbapi_connection.execute("PRAGMA foreign_keys=ON")


def create_database_engine(database_url: str) -> Engine:
    """Create an engine that works with SQLite now and PostgreSQL later."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    database_engine = create_engine(
        database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    if database_engine.dialect.name == "sqlite":
        event.listen(database_engine, "connect", _enable_sqlite_foreign_keys)
    return database_engine


engine = create_database_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db(database_engine: Engine | None = None) -> None:
    """Create the current ORM tables for the configured database."""
    import app.database.models  # noqa: F401

    Base.metadata.create_all(bind=database_engine or engine)


def get_db() -> Generator[Session, None, None]:
    """Provide one database session for a FastAPI request."""
    with SessionLocal() as session:
        yield session
