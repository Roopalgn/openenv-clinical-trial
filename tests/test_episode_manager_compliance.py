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

    def test_invalid_action_r_validity_negative(self, manager: EpisodeManager) -> None:
        # SUBMIT_TO_FDA_REVIEW not permitted in literature_review phase
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        _, reward, _, _ = manager.step(action)
        assert reward.r_validity < 0, "r_validity must be negative for invalid action"

    def test_invalid_action_state_unchanged(self, manager: EpisodeManager) -> None:
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

    def test_invalid_action_done_is_false(self, manager: EpisodeManager) -> None:
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

    def test_multiple_invalid_steps_do_not_raise(self, manager: EpisodeManager) -> None:
        action = _make_action(ActionType.SUBMIT_TO_FDA_REVIEW)
        for _ in range(5):
            _, reward, _, _ = manager.step(action)
            assert reward.r_validity < 0


# ---------------------------------------------------------------------------
# Task 26: Power cache tests (Requirements 14.1, 14.2, 14.3, 14.4)
# ---------------------------------------------------------------------------

from server.simulator.power_calculator import calculate_power


class TestPowerCache:
    """Requirements 14.1–14.4: episode-scoped power cache on EpisodeManager."""

    def test_cache_populated_after_cached_call(self) -> None:
        """Req 14.1: cache stores result after first call."""
        em = EpisodeManager()
        em.reset(seed=42)
        key = (0.5, 100, 0.05)
        assert key not in em._power_cache
        em.cached_calculate_power(*key)
        assert key in em._power_cache

    def test_cached_result_matches_direct_calculation(self) -> None:
        """Req 14.1: cached value equals direct calculate_power output."""
        em = EpisodeManager()
        em.reset(seed=42)
        result = em.cached_calculate_power(0.5, 100, 0.05)
        assert result == calculate_power(0.5, 100, 0.05)

    def test_second_call_returns_same_value(self) -> None:
        """Req 14.2: repeated call with same params returns cached value."""
        em = EpisodeManager()
        em.reset(seed=42)
        first = em.cached_calculate_power(0.3, 80, 0.05)
        second = em.cached_calculate_power(0.3, 80, 0.05)
        assert first == second

    def test_cache_cleared_on_reset(self) -> None:
        """Req 14.3: _clear_cache() is called on reset(), emptying the cache."""
        em = EpisodeManager()
        em.reset(seed=1)
        em.cached_calculate_power(0.5, 100, 0.05)
        assert len(em._power_cache) > 0
        em.reset(seed=2)
        assert len(em._power_cache) == 0

    def test_clear_cache_directly(self) -> None:
        """Req 14.3: _clear_cache() empties _power_cache."""
        em = EpisodeManager()
        em.reset(seed=1)
        em.cached_calculate_power(0.5, 100, 0.05)
        em._clear_cache()
        assert em._power_cache == {}

    def test_different_params_cached_separately(self) -> None:
        """Req 14.1: different parameter sets are cached under different keys."""
        em = EpisodeManager()
        em.reset(seed=42)
        em.cached_calculate_power(0.3, 50, 0.05)
        em.cached_calculate_power(0.5, 100, 0.05)
        assert len(em._power_cache) == 2


class TestRoundTripProperty:
    """Req 14.4: same seed after cache clear produces identical results."""

    def test_reset_same_seed_identical_initial_observation(self) -> None:
        """Two resets with the same seed produce identical initial observations."""
        em = EpisodeManager()
        obs1 = em.reset(seed=99)
        obs2 = em.reset(seed=99)
        assert obs1.scenario_description == obs2.scenario_description
        assert obs1.resource_status == obs2.resource_status

    def test_reset_same_seed_identical_latent_state(self) -> None:
        """Same seed → same hidden ground-truth values after reset."""
        em = EpisodeManager()
        em.reset(seed=77)
        latent1_effect = em._latent.true_effect_size
        latent1_side = em._latent.true_side_effect_rate

        # Populate cache, then reset with same seed
        em.cached_calculate_power(0.5, 100, 0.05)
        em.reset(seed=77)

        assert em._latent.true_effect_size == latent1_effect
        assert em._latent.true_side_effect_rate == latent1_side

    def test_cache_cleared_before_new_episode_values_computed(self) -> None:
        """After reset, cache is empty so new episode starts cold."""
        em = EpisodeManager()
        em.reset(seed=10)
        em.cached_calculate_power(0.9, 200, 0.01)  # populate with arbitrary key
        em.reset(seed=10)
        # Cache must be empty — no stale values from previous episode
        assert em._power_cache == {}

    def test_step_results_identical_after_cache_clear(self) -> None:
        """Req 14.4: step() produces same reward after reset with same seed."""
        from models import ActionType, TrialAction

        action = TrialAction(
            action_type=ActionType.SET_PRIMARY_ENDPOINT,
            parameters={},
            justification="round-trip test",
            confidence=0.5,
        )

        em = EpisodeManager()
        em.reset(seed=55)
        _, reward1, _, _ = em.step(action)

        # Populate cache with something, then reset with same seed
        em.cached_calculate_power(0.99, 999, 0.01)
        em.reset(seed=55)
        _, reward2, _, _ = em.step(action)

        assert reward1.total == reward2.total
