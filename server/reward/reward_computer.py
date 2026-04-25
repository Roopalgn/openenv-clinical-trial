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

import math

from models import (
    RewardBreakdown,
    TrialAction,
    TrialLatentState,
    TrialResult,
)
from server.phase_detector import compute_phase_ordering_reward
from server.rules.fda_rules import check_fda_compliance

# Reward magnitude constants — V3
# Tuned for GRPO: minimise free per-step floor so within-group variance is high.
# Good valid action: ~1.0 | Mediocre valid: ~0.1 | Invalid: ~-2.5
_VALIDITY_VALID = 0.05
_VALIDITY_INVALID = -2.0
_PENALTY_INVALID = -0.5
_TERMINAL_SUCCESS = 4.0
_TERMINAL_CALIBRATION = 2.0
_INFO_GAIN_BASE = 1.0
_EFFICIENCY_SCALE = 0.3
_NOVELTY_BASE = 0.1


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

    # Add milestone completion bonus to info_gain (progressive learning signal)
    r_info_gain += _milestone_reward(action, latent)

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
    if not math.isfinite(result.power):
        return 0.0
    # Base info gain proportional to power (how useful the experiment was)
    base = _INFO_GAIN_BASE * max(result.power, 0.1)  # floor at 0.1 so info actions always get something
    return base


# Milestone completion bonus constants
_MILESTONE_BONUS = 1.0  # Moderate bonus for reaching new milestones


def _milestone_reward(action: TrialAction, latent: TrialLatentState) -> float:
    """Bonus for actions that complete key milestones for the FIRST time only.

    Provides intermediate reward signal that GRPO needs to learn the
    correct action sequence. Only fires once per milestone to prevent
    degenerate "repeat the same action" exploit.
    """
    from models import ActionType

    # Count how many times this action has been taken (TransitionEngine already
    # appended it, so count >= 1 for current action).
    action_count = latent.action_history.count(action.action_type.value)

    bonus = 0.0
    # Phase I completion (dose escalation) — first time only
    if (action.action_type == ActionType.RUN_DOSE_ESCALATION
            and latent.phase_i_complete and action_count == 1):
        bonus += _MILESTONE_BONUS
    # Effect size estimation — first time only
    if (action.action_type == ActionType.ESTIMATE_EFFECT_SIZE
            and latent.effect_estimated and action_count == 1):
        bonus += _MILESTONE_BONUS * 0.5
    # Interim analysis completion — first time only
    if (action.action_type == ActionType.RUN_INTERIM_ANALYSIS
            and latent.interim_complete and action_count == 1):
        bonus += _MILESTONE_BONUS
    # Protocol submission — first time only
    if (action.action_type == ActionType.SUBMIT_TO_FDA_REVIEW
            and latent.protocol_submitted and action_count == 1):
        bonus += _MILESTONE_BONUS * 0.5
    # Primary analysis (trial complete) — first time only
    if (action.action_type == ActionType.RUN_PRIMARY_ANALYSIS
            and latent.trial_complete and action_count == 1):
        bonus += _MILESTONE_BONUS * 1.5
    # Patient enrollment — first time only (not per-repeat)
    if (action.action_type == ActionType.ENROLL_PATIENTS
            and latent.patients_enrolled > 0 and action_count == 1):
        bonus += _MILESTONE_BONUS * 0.3
    return bonus


def _efficiency_reward(
    latent: TrialLatentState,
    initial_budget: float = 1_000_000.0,
) -> float:
    """Reward proportional to remaining budget — ONLY at terminal.

    Changed from per-step to terminal-only to eliminate the massive constant
    baseline that was drowning out discriminative reward components.
    Previously gave ~1.9 per step × 15 steps = ~28.5 free reward per episode.
    """
    if not latent.trial_complete:
        return 0.0
    if initial_budget <= 0:
        return 0.0
    budget_fraction = min(max(latent.budget_remaining / initial_budget, 0.0), 1.0)
    return _EFFICIENCY_SCALE * budget_fraction


def _novelty_reward(action: TrialAction, latent: TrialLatentState) -> float:
    """Small bonus for action types not yet used in this episode."""
    # TransitionEngine appends the current action before compute_reward runs,
    # so exclude the last entry to check novelty correctly.
    prior_history = latent.action_history[:-1] if latent.action_history else []
    if action.action_type.value not in prior_history:
        return _NOVELTY_BASE
    return 0.0


def _terminal_success_reward(latent: TrialLatentState, result: TrialResult) -> float:
    """Positive reward when the episode ends with a successful trial (req 6.4)."""
    if latent.trial_complete and result.success and result.failure_reason is None:
        return _TERMINAL_SUCCESS
    if latent.trial_complete:
        return -1.0
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
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        return 0.0
    ci_width = ci_high - ci_low
    ci_centre = (ci_low + ci_high) / 2.0
    true_effect = latent.true_effect_size

    centre_error = abs(ci_centre - true_effect)
    calibration_score = max(0.0, 1.0 - centre_error)
    width_penalty = min(ci_width, 1.0)
    calibration_score *= 1.0 - width_penalty * 0.5

    return _TERMINAL_CALIBRATION * calibration_score
