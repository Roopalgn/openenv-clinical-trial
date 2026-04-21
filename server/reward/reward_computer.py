"""
Reward computer — computes all eight RewardBreakdown components.

Requirements 6.1–6.6:
  - compute_reward accepts TrialAction, TrialLatentState, TrialResult
  - All eight keys are computed
  - Invalid actions → r_validity < 0, r_penalty < 0
  - Terminal success → r_terminal_success > 0
  - Terminal calibration → r_terminal_calibration > 0
  - Deterministic given same inputs
"""

from __future__ import annotations

from models import (
    RewardBreakdown,
    TrialAction,
    TrialLatentState,
    TrialResult,
)
from server.phase_detector import compute_phase_ordering_reward
from server.rules.fda_rules import check_fda_compliance

# Reward magnitude constants
_VALIDITY_VALID = 1.0
_VALIDITY_INVALID = -1.0
_PENALTY_INVALID = -0.5
_TERMINAL_SUCCESS = 10.0
_TERMINAL_CALIBRATION = 5.0
_INFO_GAIN_BASE = 0.5
_EFFICIENCY_SCALE = 2.0
_NOVELTY_BASE = 0.2


def compute_reward(
    action: TrialAction,
    latent: TrialLatentState,
    result: TrialResult,
    phase_history: list[str] | None = None,
    initial_budget: float = 1_000_000.0,
) -> RewardBreakdown:
    """Compute all eight reward components for a single step.

    Deterministic: given the same (action, latent, result) inputs,
    always returns the same RewardBreakdown.

    Args:
        action: The agent's action.
        latent: Hidden ground-truth + episode tracking state.
        result: The simulated trial result.
        phase_history: List of phase names from previous steps (for r_ordering).
        initial_budget: The scenario's starting budget (used for efficiency reward).
            Defaults to 1_000_000 for backwards compatibility but should be set
            to the scenario's actual budget_usd.

    Returns:
        A RewardBreakdown with all eight keys populated.
    """
    compliance = check_fda_compliance(action, latent)

    r_validity = _VALIDITY_VALID if compliance.valid else _VALIDITY_INVALID
    r_penalty = (
        _PENALTY_INVALID * len(compliance.violations) if not compliance.valid else 0.0
    )
    r_ordering = compute_phase_ordering_reward(action, phase_history or [])
    r_info_gain = _info_gain_reward(action, result)
    r_efficiency = _efficiency_reward(latent, initial_budget)
    r_novelty = _novelty_reward(action, latent)
    r_terminal_success = _terminal_success_reward(latent, result)
    r_terminal_calibration = _terminal_calibration_reward(latent, result)

    return RewardBreakdown(
        r_validity=r_validity,
        r_ordering=r_ordering,
        r_info_gain=r_info_gain,
        r_efficiency=r_efficiency,
        r_novelty=r_novelty,
        r_penalty=r_penalty,
        r_terminal_success=r_terminal_success,
        r_terminal_calibration=r_terminal_calibration,
    )


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _info_gain_reward(action: TrialAction, result: TrialResult) -> float:
    """Reward for information-gathering actions that produce useful results."""
    from models import ActionType

    info_actions = {
        ActionType.ESTIMATE_EFFECT_SIZE,
        ActionType.OBSERVE_SAFETY_SIGNAL,
        ActionType.RUN_INTERIM_ANALYSIS,
        ActionType.RUN_DOSE_ESCALATION,
        ActionType.ADD_BIOMARKER_STRATIFICATION,
    }
    if action.action_type not in info_actions:
        return 0.0
    return _INFO_GAIN_BASE * result.power


def _efficiency_reward(latent: TrialLatentState, initial_budget: float = 1_000_000.0) -> float:
    """Reward proportional to remaining budget (encourages frugality)."""
    if initial_budget <= 0:
        return 0.0
    budget_fraction = min(max(latent.budget_remaining / initial_budget, 0.0), 1.0)
    return _EFFICIENCY_SCALE * budget_fraction


def _novelty_reward(action: TrialAction, latent: TrialLatentState) -> float:
    """Small bonus for action types not yet used in this episode."""
    if action.action_type.value not in latent.action_history:
        return _NOVELTY_BASE
    return 0.0


def _terminal_success_reward(latent: TrialLatentState, result: TrialResult) -> float:
    """Positive reward when the episode ends with a successful trial (req 6.4)."""
    if latent.trial_complete and result.success and result.failure_reason is None:
        return _TERMINAL_SUCCESS
    return 0.0


def _terminal_calibration_reward(
    latent: TrialLatentState, result: TrialResult
) -> float:
    """Positive reward when uncertainty estimate is well-calibrated (req 6.5).

    A narrow, accurate CI earns the full bonus; a wide or inaccurate CI earns less.
    """
    if not latent.trial_complete:
        return 0.0

    ci_low, ci_high = result.confidence_interval
    ci_width = ci_high - ci_low
    ci_centre = (ci_low + ci_high) / 2.0
    true_effect = latent.true_effect_size

    centre_error = abs(ci_centre - true_effect)
    calibration_score = max(0.0, 1.0 - centre_error)
    width_penalty = min(ci_width, 1.0)
    calibration_score *= 1.0 - width_penalty * 0.5

    return _TERMINAL_CALIBRATION * calibration_score
