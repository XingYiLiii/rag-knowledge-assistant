"""Health-check endpoint."""

from fastapi import APIRouter, status

router = APIRouter(tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> dict[str, str]:
    """Return the liveness state of the API."""
    return {
        "status": "ok",
        "service": "rag-knowledge-assistant",
        "version": "0.1.0",
    }
