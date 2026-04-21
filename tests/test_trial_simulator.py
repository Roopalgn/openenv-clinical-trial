"""
Tests for simulate_trial edge cases — Task 27.

Requirements 15.2, 15.3, 15.4:
  - budget_remaining <= 0  → failure_reason="budget_exhausted"
  - time_remaining_days <= 0 → failure_reason="time_exhausted"
  - patients_enrolled == 0 at terminal → failure_reason="no_enrollment"
  - Each edge case returns a valid TrialResult
"""

from __future__ import annotations

from models import ActionType, TrialAction, TrialLatentState, TrialResult
from server.simulator.trial_simulator import simulate_trial


def _make_latent(**overrides) -> TrialLatentState:
    """Build a minimal valid TrialLatentState with sensible defaults."""
    defaults = dict(
        true_effect_size=0.5,
        true_side_effect_rate=0.1,
        true_responder_population="all",
        true_responder_criteria=[],
        true_dose_response={1.0: 0.5},
        true_mechanism="inhibition",
        placebo_response_rate=0.1,
        dropout_rate=0.05,
        site_variability=0.05,
        measurement_noise=0.05,
        budget_remaining=500_000.0,
        time_remaining_days=180,
        patients_enrolled=50,
        phase_i_complete=True,
        mtd_identified=True,
        effect_estimated=True,
        protocol_submitted=True,
        interim_complete=True,
        trial_complete=False,
        adverse_events=0,
        episode_phase="phase_iii",
        action_history=[],
        seed=42,
    )
    defaults.update(overrides)
    return TrialLatentState(**defaults)


def _make_action() -> TrialAction:
    return TrialAction(
        action_type=ActionType.RUN_PRIMARY_ANALYSIS,
        parameters={},
        justification="test",
        confidence=0.9,
    )


class TestBudgetExhausted:
    """Requirement 15.2: budget_remaining <= 0 → failure_reason='budget_exhausted'."""

    def test_zero_budget_returns_failure(self) -> None:
        latent = _make_latent(budget_remaining=0.0)
        result = simulate_trial(latent, _make_action())
        assert isinstance(result, TrialResult)
        assert result.failure_reason == "budget_exhausted"
        assert result.success is False

    def test_negative_budget_returns_failure(self) -> None:
        latent = _make_latent(budget_remaining=-1.0)
        result = simulate_trial(latent, _make_action())
        assert result.failure_reason == "budget_exhausted"
        assert result.success is False

    def test_budget_exhausted_result_is_valid_trial_result(self) -> None:
        latent = _make_latent(budget_remaining=0.0)
        result = simulate_trial(latent, _make_action())
        assert isinstance(result, TrialResult)
        assert 0.0 <= result.p_value <= 1.0
        assert 0.0 <= result.power <= 1.0
        assert 0.0 <= result.adverse_event_rate <= 1.0
        assert len(result.confidence_interval) == 2


class TestTimeExhausted:
    """Requirement 15.3: time_remaining_days <= 0 → failure_reason='time_exhausted'."""

    def test_zero_time_returns_failure(self) -> None:
        latent = _make_latent(time_remaining_days=0)
        result = simulate_trial(latent, _make_action())
        assert isinstance(result, TrialResult)
        assert result.failure_reason == "time_exhausted"
        assert result.success is False

    def test_negative_time_returns_failure(self) -> None:
        latent = _make_latent(time_remaining_days=-5)
        result = simulate_trial(latent, _make_action())
        assert result.failure_reason == "time_exhausted"
        assert result.success is False

    def test_time_exhausted_result_is_valid_trial_result(self) -> None:
        latent = _make_latent(time_remaining_days=0)
        result = simulate_trial(latent, _make_action())
        assert isinstance(result, TrialResult)
        assert 0.0 <= result.p_value <= 1.0
        assert 0.0 <= result.power <= 1.0
        assert 0.0 <= result.adverse_event_rate <= 1.0
        assert len(result.confidence_interval) == 2


class TestNoEnrollment:
    """Requirement 15.4: patients_enrolled == 0 at terminal → failure_reason='no_enrollment'."""

    def test_zero_patients_at_terminal_returns_failure(self) -> None:
        latent = _make_latent(patients_enrolled=0, trial_complete=True)
        result = simulate_trial(latent, _make_action())
        assert isinstance(result, TrialResult)
        assert result.failure_reason == "no_enrollment"
        assert result.success is False

    def test_zero_patients_non_terminal_does_not_trigger(self) -> None:
        # patients_enrolled == 0 but trial_complete=False → normal path
        latent = _make_latent(patients_enrolled=0, trial_complete=False)
        result = simulate_trial(latent, _make_action())
        # Should NOT be no_enrollment (non-terminal)
        assert result.failure_reason != "no_enrollment"

    def test_no_enrollment_result_is_valid_trial_result(self) -> None:
        latent = _make_latent(patients_enrolled=0, trial_complete=True)
        result = simulate_trial(latent, _make_action())
        assert isinstance(result, TrialResult)
        assert 0.0 <= result.p_value <= 1.0
        assert 0.0 <= result.power <= 1.0
        assert 0.0 <= result.adverse_event_rate <= 1.0
        assert len(result.confidence_interval) == 2


class TestNormalPath:
    """Sanity check: valid state with resources → no failure_reason."""

    def test_normal_state_no_failure_reason(self) -> None:
        latent = _make_latent()
        result = simulate_trial(latent, _make_action())
        assert result.failure_reason is None

    def test_budget_priority_over_time(self) -> None:
        # Both budget and time exhausted — budget check fires first
        latent = _make_latent(budget_remaining=0.0, time_remaining_days=0)
        result = simulate_trial(latent, _make_action())
        assert result.failure_reason == "budget_exhausted"
