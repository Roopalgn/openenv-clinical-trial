"""Tests for global exception handling and request validation (Requirements 15.1, 15.5)."""

from unittest.mock import patch

from starlette.testclient import TestClient

from server.app import app, _manager

client = TestClient(app, raise_server_exceptions=False)


def test_malformed_step_body_returns_422() -> None:
    """POST /step with a malformed body must return 422 with field-level errors."""
    response = client.post("/step", json={"not_a_valid_field": "garbage"})
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_step_missing_required_fields_returns_422() -> None:
    """POST /step with an empty body must return 422."""
    response = client.post("/step", json={})
    assert response.status_code == 422


def test_step_invalid_action_type_returns_422() -> None:
    """POST /step with an unknown action_type must return 422."""
    response = client.post(
        "/step",
        json={
            "action_type": "not_a_real_action",
            "parameters": {},
            "justification": "test",
            "confidence": 0.5,
        },
    )
    assert response.status_code == 422


def test_unhandled_exception_returns_500_json() -> None:
    """Any unhandled exception in an endpoint must return 500 JSON, not crash."""
    # Patch get_state to raise a non-RuntimeError (unexpected) exception.
    # The /state route only catches RuntimeError → anything else hits the global handler.
    with patch.object(_manager, "get_state", side_effect=ValueError("unexpected boom")):
        response = client.get("/state")
    assert response.status_code == 500
    body = response.json()
    assert "detail" in body
    assert body["detail"] == "Internal server error"


def test_step_no_active_episode_returns_400() -> None:
    """POST /step before /reset must return 400 with a descriptive message."""
    # Ensure no active episode by creating a fresh manager state
    _manager._state = None  # type: ignore[attr-defined]
    _manager._latent = None  # type: ignore[attr-defined]

    response = client.post(
        "/step",
        json={
            "action_type": "enroll_patients",
            "parameters": {},
            "justification": "test",
            "confidence": 0.5,
        },
    )
    assert response.status_code == 400
    assert "detail" in response.json()
