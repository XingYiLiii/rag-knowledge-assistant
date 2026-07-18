"""Integration tests for request identifiers and global exception handling."""

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.core.exceptions import ApplicationError
from app.main import create_app


def test_every_response_includes_a_request_id() -> None:
    """Successful responses always include a generated request identifier."""
    response = TestClient(create_app()).get("/api/v1/health")

    assert response.status_code == 200
    assert UUID(response.headers["X-Request-ID"])


def test_client_request_id_is_preserved() -> None:
    """A valid client-generated UUID is returned in the response headers."""
    request_id = str(uuid4())
    client = TestClient(create_app())

    response = client.get("/api/v1/health", headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id


def test_application_error_uses_the_standard_response_schema() -> None:
    """Known errors expose only their explicitly safe attributes."""
    app = create_app()

    @app.get("/_test/application-error")
    def raise_application_error() -> None:
        raise ApplicationError(code="TEST_ERROR", message="A safe error message.", status_code=418)

    response = TestClient(app).get("/_test/application-error")

    assert response.status_code == 418
    assert response.json()["error"] == {
        "code": "TEST_ERROR",
        "message": "A safe error message.",
    }
    assert UUID(response.json()["request_id"])
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_unhandled_error_hides_internal_details() -> None:
    """Unexpected exceptions never expose exception text or stack traces to clients."""
    app = create_app()

    @app.get("/_test/unhandled-error")
    def raise_unhandled_error() -> None:
        raise RuntimeError("secret-api-key-should-not-be-exposed")

    response = TestClient(app, raise_server_exceptions=False).get("/_test/unhandled-error")

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "INTERNAL_SERVER_ERROR",
        "message": "An unexpected error occurred.",
    }
    assert "secret-api-key" not in response.text
    assert "Traceback" not in response.text
    assert UUID(response.headers["X-Request-ID"])
