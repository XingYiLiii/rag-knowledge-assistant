"""Integration tests for the server-rendered knowledge management UI."""

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app


def test_knowledge_base_management_page_renders_key_elements() -> None:
    """The landing page serves the list view, creation form, and static asset references."""
    client = TestClient(create_app())

    response = client.get("/")

    assert response.status_code == 200
    assert "RAG Knowledge Assistant" in response.text
    assert 'id="create-knowledge-base-form"' in response.text
    assert 'id="knowledge-base-list"' in response.text
    assert "/static/styles.css" in response.text


def test_knowledge_base_detail_page_renders_document_management_controls() -> None:
    """The detail template exposes upload and document-list hooks for the client script."""
    knowledge_base_id = uuid4()
    client = TestClient(create_app())

    response = client.get(f"/knowledge-bases/{knowledge_base_id}")

    assert response.status_code == 200
    assert f'data-knowledge-base-id="{knowledge_base_id}"' in response.text
    assert 'id="upload-document-form"' in response.text
    assert 'id="document-list"' in response.text
    assert "/static/app.js" in response.text


def test_web_static_assets_are_available() -> None:
    """The application mounts the scripts used for document-management actions."""
    client = TestClient(create_app())

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "loadKnowledgeBases" in response.text
    assert "reindexDocument" in response.text
    assert "deleteDocument" in response.text
