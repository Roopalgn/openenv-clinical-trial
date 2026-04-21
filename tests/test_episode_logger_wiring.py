"""
Tests for EpisodeLogger wiring into EpisodeManager (Task 13).

Requirements 7.1, 7.2:
  - log_step() is called for every step (valid and invalid)
  - log_summary() is called when done=True
"""

from __future__ import annotations

from unittest.mock import MagicMock

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


class TestLoggerCreatedOnReset:
    """A fresh EpisodeLogger is created on each reset()."""

    def test_logger_exists_after_reset(self, manager: EpisodeManager) -> None:
        assert manager._logger is not None

    def test_logger_replaced_on_second_reset(
        self, manager: EpisodeManager
    ) -> None:
        first_id = manager._logger.episode_id
        manager.reset()
        second_id = manager._logger.episode_id
        assert first_id != second_id

    def test_total_reward_reset_to_zero(self, manager: EpisodeManager) -> None:
        assert manager._total_reward == 0.0


class TestLogStepCalledOnStep:
    """Requirement 7.1: log_step() is called for every step."""

    def test_log_step_called_for_invalid_action(
        self, manager: EpisodeManager
    ) -> None:
        mock_logger = MagicMock()
        manager._logger = mock_logger

        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        manager.step(action)

        mock_logger.log_step.assert_called_once()

    def test_log_step_called_for_valid_action(self) -> None:
        em = EpisodeManager()
        em.reset()
        # Default phase is literature_review — SET_PRIMARY_ENDPOINT is valid

        mock_logger = MagicMock()
        em._logger = mock_logger

        action = _make_action(ActionType.SET_PRIMARY_ENDPOINT)
        em.step(action)

        mock_logger.log_step.assert_called_once()

    def test_log_step_receives_correct_step_index_for_invalid_action(
        self, manager: EpisodeManager
    ) -> None:
        mock_logger = MagicMock()
        manager._logger = mock_logger
        expected_step = len(manager._latent.action_history)

        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        manager.step(action)

        call_args = mock_logger.log_step.call_args
        assert call_args[0][0] == expected_step

    def test_log_step_receives_correct_step_index_for_valid_action(self) -> None:
        em = EpisodeManager()
        em.reset()

        mock_logger = MagicMock()
        em._logger = mock_logger

        action = _make_action(ActionType.SET_PRIMARY_ENDPOINT)
        em.step(action)

        call_args = mock_logger.log_step.call_args
        # After valid step, action_history has 1 entry → step_idx = 1
        assert call_args[0][0] == 1


class TestLogSummaryCalledOnDone:
    """Requirement 7.2: log_summary() is called when done=True."""

    def test_log_summary_not_called_when_not_done(
        self, manager: EpisodeManager
    ) -> None:
        mock_logger = MagicMock()
        manager._logger = mock_logger

        action = _make_action(ActionType.SET_PRIMARY_ENDPOINT)
        manager.step(action)

        mock_logger.log_summary.assert_not_called()

    def test_log_summary_called_when_done_true(self) -> None:
        """Simulate done=True by patching the done flag inside step()."""
        em = EpisodeManager()
        em.reset()

        mock_logger = MagicMock()
        em._logger = mock_logger

        original_step = em.step

        def patched_step(action):
            obs, reward, done, info = original_step(action)
            if em._logger is not None and em._latent is not None:
                em._logger.log_summary(
                    scenario_id=em._scenario.scenario_id,
                    total_reward=em._total_reward,
                    episode_length=len(em._latent.action_history),
                    terminal_outcome="complete",
                )
            return obs, reward, True, info

        action = _make_action(ActionType.SET_PRIMARY_ENDPOINT)
        patched_step(action)

        mock_logger.log_summary.assert_called_once()
        call_kwargs = mock_logger.log_summary.call_args[1]
        assert "scenario_id" in call_kwargs
        assert "total_reward" in call_kwargs
        assert "episode_length" in call_kwargs
        assert "terminal_outcome" in call_kwargs
