"""
Tests for EpisodeManager.step() compliance wiring (Task 10).

Requirements 10.1, 10.4:
  - Invalid actions return negative r_validity and leave latent state unchanged
  - No unhandled exceptions escape from step()
"""

from __future__ import annotations

import pytest

from models import ActionType, TrialAction
from server.episode_manager import EpisodeManager


def _make_action(action_type: ActionType, **params) -> TrialAction:
    return TrialAction(
        action_type=action_type,
        parameters=params,
        justification="test",
        confidence=0.5,
    )


@pytest.fixture()
def manager() -> EpisodeManager:
    em = EpisodeManager()
    em.reset()
    return em


class TestInvalidActionReturnsNegativeRValidity:
    """Requirement 10.1: invalid actions → negative r_validity, latent unchanged."""

    def test_invalid_action_r_validity_negative(
        self, manager: EpisodeManager
    ) -> None:
        # SUBMIT_TO_FDA_REVIEW not permitted in literature_review phase
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        _, reward, _, _ = manager.step(action)
        assert reward.r_validity < 0, "r_validity must be negative for invalid action"

    def test_invalid_action_state_unchanged(
        self, manager: EpisodeManager
    ) -> None:
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        history_before = list(manager._latent.action_history)
        step_before = len(history_before)

        manager.step(action)

        assert len(manager._latent.action_history) == step_before, (
            "action_history must not change on invalid action"
        )

    def test_invalid_action_rule_violations_populated(
        self, manager: EpisodeManager
    ) -> None:
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        obs, _, _, info = manager.step(action)
        assert len(obs.rule_violations) > 0
        assert len(info["violations"]) > 0

    def test_invalid_action_done_is_false(
        self, manager: EpisodeManager
    ) -> None:
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        _, _, done, _ = manager.step(action)
        assert done is False

    def test_invalid_action_info_contains_action_valid_false(
        self, manager: EpisodeManager
    ) -> None:
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        _, _, _, info = manager.step(action)
        assert info.get("action_valid") is False


class TestValidActionAdvancesState:
    """Valid actions should advance action_history."""

    def test_valid_action_in_literature_review_phase(self) -> None:
        em = EpisodeManager()
        em.reset()
        # Default phase is literature_review — SET_PRIMARY_ENDPOINT is permitted
        action = _make_action(ActionType.SET_PRIMARY_ENDPOINT)
        history_before = len(em._latent.action_history)

        em.step(action)

        assert len(em._latent.action_history) == history_before + 1

    def test_valid_action_r_validity_not_negative(self) -> None:
        em = EpisodeManager()
        em.reset()
        action = _make_action(ActionType.SET_PRIMARY_ENDPOINT)
        _, reward, _, _ = em.step(action)
        assert reward.r_validity >= 0.0


class TestNoUnhandledExceptions:
    """Requirement 10.4: step() must never raise unhandled exceptions."""

    def test_step_without_reset_raises_runtime_error(self) -> None:
        em = EpisodeManager()
        action = _make_action(ActionType.SET_PRIMARY_ENDPOINT)
        with pytest.raises(RuntimeError, match="No active episode"):
            em.step(action)

    def test_multiple_invalid_steps_do_not_raise(
        self, manager: EpisodeManager
    ) -> None:
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        for _ in range(5):
            _, reward, _, _ = manager.step(action)
            assert reward.r_validity < 0
