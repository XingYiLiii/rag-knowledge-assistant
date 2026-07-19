"""Integration tests for server-rendered chat and history web pages."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_chat_page_renders_chat_inputs_and_citation_region() -> None:
    """The chat page includes the API-backed question controls and source display hooks."""
    client = TestClient(create_app())

    response = client.get("/chat")

    assert response.status_code == 200
    assert 'id="chat-knowledge-base"' in response.text
    assert 'id="chat-conversation"' in response.text
    assert 'id="chat-question"' in response.text
    assert 'id="chat-messages"' in response.text
    assert "可展开的引用快照" in response.text


def test_history_page_renders_knowledge_base_and_message_regions() -> None:
    """The history page includes the controls used to load saved conversations and messages."""
    client = TestClient(create_app())

    response = client.get("/history")

    assert response.status_code == 200
    assert 'id="history-knowledge-base"' in response.text
    assert 'id="conversation-list"' in response.text
    assert 'id="history-messages"' in response.text
    assert "引用快照" in response.text


def test_chat_script_contains_safe_citation_and_history_rendering() -> None:
    """The client script uses safe DOM APIs and calls the existing history endpoints."""
    client = TestClient(create_app())

    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "renderCitation" in response.text
    assert "createElement" in response.text
    assert "innerHTML" not in response.text
    assert "/conversations" in response.text
    assert 'pending: "等待处理"' in response.text
    assert 'processing: "处理中"' in response.text
    assert 'ready: "已完成"' in response.text
    assert 'failed: "失败"' in response.text
