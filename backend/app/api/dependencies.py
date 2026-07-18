"""Shared FastAPI dependencies."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.database.session import get_db


def get_database_session() -> Generator[Session, None, None]:
    """Expose the database session dependency under an API-focused name."""
    yield from get_db()
