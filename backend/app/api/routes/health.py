"""Health-check endpoint."""

from fastapi import APIRouter, status

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Return the liveness state of the API."""
    settings = get_settings()
    return {
        "status": "ok",
        "service": "rag-knowledge-assistant",
        "version": settings.app_version,
    }
