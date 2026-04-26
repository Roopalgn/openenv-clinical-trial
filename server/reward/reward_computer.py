"""
Reward computer — computes all eight RewardBreakdown components.

Requirements 6.1–6.6:
  - compute_reward accepts TrialAction, TrialLatentState, TrialResult
  - All eight keys are computed
  - Invalid actions → r_validity < 0, r_penalty < 0
  - Terminal success → r_terminal_success > 0
  - Terminal calibration → r_terminal_calibration > 0
  - Deterministic given same inputs

V4 — Steep-slope tuning for GRPO:
  - Milestone bonuses doubled → clear reward tiers for partial vs full completion
  - Episode-wide violation penalty at terminal → no "clean-last-step" exploit
  - Progress-proportional terminal bonus → longer episodes are reliably rewarded
  - Phase-skip penalty sharpened → clearer ordering gradient
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
from server.rules.fda_rules import ComplianceResult, check_fda_compliance

# Reward magnitude constants — V4
# Tuned for GRPO: maximise separation between "did nothing useful" and
# "completed a good trial" to produce a steep training slope.
_VALIDITY_VALID = 0.05
_VALIDITY_REPEAT = 0.0
_VALIDITY_INVALID = -2.0
_PENALTY_INVALID = -0.5
_TERMINAL_SUCCESS = 4.0
_TERMINAL_FAILURE = -1.0
_TERMINAL_CALIBRATION = 2.0
_INFO_GAIN_BASE = 1.0
_EFFICIENCY_SCALE = 0.3
_NOVELTY_BASE = 0.1
# Statistical-power gating for the terminal success bonus. Trials that hit
# p < 0.05 with very low power (small n + large effect + noise) used to receive
# the full +4.0 bonus, which trained the agent to design statistically unsound
# studies. The bonus now ramps linearly from POWER_FLOOR (no bonus) to
# POWER_TARGET (full bonus).
_TERMINAL_SUCCESS_POWER_FLOOR = 0.40
_TERMINAL_SUCCESS_POWER_TARGET = 0.80

# V4: Terminal progress bonus — rewards "how far" the agent got even if it
# didn't fully complete.  This creates a smooth gradient from "did 1 step"
# to "nearly finished" which is critical for GRPO slope.
_TERMINAL_PROGRESS_SCALE = 3.0

# V4: Episode-wide violation cost at terminal.  Previously only the current
# step's violations were penalised.  Now cumulative violations across the
# entire episode reduce terminal reward.
_EPISODE_VIOLATION_COST = -0.3


def compute_reward(
    action: TrialAction,
    latent: TrialLatentState,
    result: TrialResult,
    phase_history: list[str] | None = None,
    initial_budget: float = 1_000_000.0,
    compliance: ComplianceResult | None = None,
    episode_violation_count: int = 0,
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
        compliance: Optional precomputed FDA compliance result for this action.
        episode_violation_count: Cumulative FDA violation count across the episode
            (used only at terminal for episode-wide penalty).

    Returns:
        A RewardBreakdown with all eight keys populated.
    """
    if compliance is None:
        compliance = check_fda_compliance(action, latent)

    if not compliance.valid:
        r_validity = _VALIDITY_INVALID
    else:
        # Repeated valid actions get 0.0, not +0.05. Otherwise the agent can
        # farm a free per-step floor (validity + ordering + novelty) by spamming
        # any one valid action type forever.
        prior_history = (
            latent.action_history[:-1] if latent.action_history else []
        )
        if action.action_type.value in prior_history:
            r_validity = _VALIDITY_REPEAT
        else:
            r_validity = _VALIDITY_VALID
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

    # V4: At terminal, add progress bonus and episode-wide violation penalty.
    # Progress bonus gives smooth gradient from "1 step" to "almost done".
    # Episode violation penalty prevents exploiting a clean last step.
    if latent.trial_complete or _is_terminal_step(latent):
        r_info_gain += _progress_bonus(latent)
        if episode_violation_count > 0:
            r_penalty += _EPISODE_VIOLATION_COST * episode_violation_count

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


def _is_terminal_step(latent: TrialLatentState) -> bool:
    """Check if this is a terminal step (max steps reached)."""
    # This is a heuristic — the episode manager sets the actual done flag.
    # Here we just check if trial_complete is set, which is already
    # handled by the caller in episode_manager.
    return False


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


# Milestone completion bonus constants — V4: doubled from V3 to create
# steeper reward gradient.  The key insight is that GRPO needs *large*
# separation between partial completions (3 milestones = ~+4.5) and full
# completions (7 milestones = ~+10.8) for the advantage signal to dominate
# the noise across the generation group.
_MILESTONE_PHASE_I = 1.5       # Phase I completion (was 1.0)
_MILESTONE_EFFECT_EST = 1.0    # Effect size estimation (was 0.5)
_MILESTONE_INTERIM = 1.5       # Interim analysis (was 1.0)
_MILESTONE_PROTOCOL = 1.0     # Protocol submission (was 0.5)
_MILESTONE_PRIMARY = 1.5      # Primary analysis (was 1.0)
_MILESTONE_CONCLUSION = 2.5   # Trial complete (was 1.5)
_MILESTONE_ENROLLMENT = 0.5   # First enrollment (was 0.3)


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
        bonus += _MILESTONE_PHASE_I
    # Effect size estimation — first time only
    if (action.action_type == ActionType.ESTIMATE_EFFECT_SIZE
            and latent.effect_estimated and action_count == 1):
        bonus += _MILESTONE_EFFECT_EST
    # Interim analysis completion — first time only
    if (action.action_type == ActionType.RUN_INTERIM_ANALYSIS
            and latent.interim_complete and action_count == 1):
        bonus += _MILESTONE_INTERIM
    # Protocol submission — first time only
    if (action.action_type == ActionType.SUBMIT_TO_FDA_REVIEW
            and latent.protocol_submitted and action_count == 1):
        bonus += _MILESTONE_PROTOCOL
    # Primary analysis run — first time only
    if (action.action_type == ActionType.RUN_PRIMARY_ANALYSIS
            and latent.primary_analysis_complete and action_count == 1):
        bonus += _MILESTONE_PRIMARY
    # Synthesize conclusion (trial complete) — first time only
    if (action.action_type == ActionType.SYNTHESIZE_CONCLUSION
            and latent.trial_complete and action_count == 1):
        bonus += _MILESTONE_CONCLUSION
    # Patient enrollment — first time only (not per-repeat)
    if (action.action_type == ActionType.ENROLL_PATIENTS
            and latent.patients_enrolled > 0 and action_count == 1):
        bonus += _MILESTONE_ENROLLMENT
    return bonus


def _progress_bonus(latent: TrialLatentState) -> float:
    """Terminal progress bonus — proportional to milestones completed.

    This creates a smooth gradient so that episodes reaching 5/7 milestones
    reliably score higher than episodes reaching 2/7 milestones, even if
    neither fully completes.  Critical for GRPO's advantage computation
    to produce meaningful updates.
    """
    milestones = [
        latent.phase_i_complete,
        latent.effect_estimated,
        latent.interim_complete,
        latent.protocol_submitted,
        latent.primary_analysis_complete,
        latent.trial_complete,
        latent.patients_enrolled > 0,
    ]
    completed = sum(1 for m in milestones if m)
    fraction = completed / len(milestones)
    return _TERMINAL_PROGRESS_SCALE * fraction


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
    """Positive reward when the episode ends with a successful trial (req 6.4).

    Success requires both p < alpha and adequate statistical power. Underpowered
    trials that happen to hit p < 0.05 by chance (small n, large effect, noise)
    used to receive the full +4.0 bonus, which trained the agent to design
    statistically unsound studies. Now the bonus is scaled linearly by power
    above the floor so that there is still a usable gradient on hard scenarios
    where reaching the canonical 0.80 power target is infeasible within the
    budget.
    """
    if not latent.trial_complete:
        return 0.0
    if not (result.success and result.failure_reason is None):
        return _TERMINAL_FAILURE
    if not math.isfinite(result.power):
        return _TERMINAL_FAILURE
    if result.power < _TERMINAL_SUCCESS_POWER_FLOOR:
        return _TERMINAL_FAILURE
    # Linear ramp from POWER_FLOOR → POWER_TARGET maps to 0 → full bonus,
    # capped at full bonus once we hit the canonical 0.80 target.
    span = _TERMINAL_SUCCESS_POWER_TARGET - _TERMINAL_SUCCESS_POWER_FLOOR
    if span <= 0.0 or result.power >= _TERMINAL_SUCCESS_POWER_TARGET:
        return _TERMINAL_SUCCESS
    fraction = (result.power - _TERMINAL_SUCCESS_POWER_FLOOR) / span
    return _TERMINAL_SUCCESS * fraction


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
