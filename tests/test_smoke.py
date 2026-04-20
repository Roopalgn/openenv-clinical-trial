"""Smoke tests for the Clinical Trial Designer environment API."""

from starlette.testclient import TestClient

from server.app import app

client = TestClient(app)


def test_ping_returns_200() -> None:
    """GET /ping should return HTTP 200 with status ok."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
