"""Integration tests for the public health endpoint."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint_returns_service_metadata() -> None:
    """The service exposes a stable, versioned health-check response."""
    client = TestClient(create_app())

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "rag-knowledge-assistant",
        "version": "0.1.0",
    }
