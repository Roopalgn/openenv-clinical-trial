"""
Phase Detector — classifies TrialActions into clinical workflow phases.

Clinical workflow phase order:
  literature_review → hypothesis → design → enrollment →
  monitoring → analysis → submission

Phase-order bonus: +0.1 for correct order (no regression, no skips)
Skip penalty: -0.3 per skipped phase

Requirements: 8.5, 9.4
"""

from __future__ import annotations

from models import ActionType, TrialAction

# Ordered list of clinical workflow phases
PHASE_ORDER: list[str] = [
    "literature_review",
    "hypothesis",
    "design",
    "enrollment",
    "monitoring",
    "analysis",
    "submission",
]

# Reward constants — V3: near-zero for same-phase, small bonus for advancing
PHASE_BONUS: float = 0.1
PHASE_SKIP_PENALTY: float = -0.3

# Mapping from ActionType to phase name.
# Kept aligned with TransitionEngine so ordering rewards match the
# episode-phase progression used by rule checks.
# literature_review has no direct action — used as default for unknown.
_ACTION_TO_PHASE: dict[ActionType, str] = {
    # hypothesis
    ActionType.SET_PRIMARY_ENDPOINT: "hypothesis",
    ActionType.ESTIMATE_EFFECT_SIZE: "hypothesis",
    # design
    ActionType.SET_SAMPLE_SIZE: "design",
    ActionType.SET_INCLUSION_CRITERIA: "design",
    ActionType.SET_EXCLUSION_CRITERIA: "design",
    ActionType.SET_DOSING_SCHEDULE: "design",
    ActionType.SET_CONTROL_ARM: "design",
    ActionType.SET_RANDOMIZATION_RATIO: "design",
    ActionType.SET_BLINDING: "design",
    ActionType.ADD_BIOMARKER_STRATIFICATION: "design",
    ActionType.REQUEST_PROTOCOL_AMENDMENT: "design",
    # enrollment
    ActionType.ENROLL_PATIENTS: "enrollment",
    ActionType.RUN_DOSE_ESCALATION: "enrollment",
    ActionType.OBSERVE_SAFETY_SIGNAL: "enrollment",
    ActionType.MODIFY_SAMPLE_SIZE: "enrollment",
    # monitoring
    ActionType.RUN_INTERIM_ANALYSIS: "monitoring",
    # analysis
    ActionType.RUN_PRIMARY_ANALYSIS: "analysis",
    ActionType.SYNTHESIZE_CONCLUSION: "analysis",
    # submission
    ActionType.SUBMIT_TO_FDA_REVIEW: "submission",
}


def detect_phase(action: TrialAction, history: list[str]) -> tuple[str, bool]:
    """Classify a TrialAction into a clinical workflow phase.

    Args:
        action: The agent's action for this step.
        history: List of phase names (strings) from previous steps in the episode.

    Returns:
        A tuple of (phase_name, phase_order_correct) where:
          - phase_name is the detected phase string
          - phase_order_correct is True iff the phase transition is valid
            (no regression, no skipped phases)
    """
    phase_name = _ACTION_TO_PHASE.get(action.action_type, "literature_review")

    if not history:
        # First action — any phase is valid
        return phase_name, True

    last_phase = history[-1]
    last_idx = PHASE_ORDER.index(last_phase) if last_phase in PHASE_ORDER else 0
    current_idx = PHASE_ORDER.index(phase_name) if phase_name in PHASE_ORDER else 0

    # Regression: going backwards is not correct
    if current_idx < last_idx:
        return phase_name, False

    # Skipped phases: any phase between last+1 and current-1 (exclusive) is a skip
    skipped = current_idx - last_idx - 1
    if skipped > 0:
        return phase_name, False

    # Staying in same phase or advancing by exactly one — correct
    return phase_name, True


def compute_phase_ordering_reward(action: TrialAction, history: list[str]) -> float:
    """Compute the r_ordering reward component using phase detection.

    Returns:
        +PHASE_BONUS if phase order is correct.
        PHASE_SKIP_PENALTY * num_skipped_phases if phases were skipped.
        0.0 if there is a regression (going backwards).
    """
    phase_name = _ACTION_TO_PHASE.get(action.action_type, "literature_review")

    if not history:
        return PHASE_BONUS

    last_phase = history[-1]
    last_idx = PHASE_ORDER.index(last_phase) if last_phase in PHASE_ORDER else 0
    current_idx = PHASE_ORDER.index(phase_name) if phase_name in PHASE_ORDER else 0

    if current_idx < last_idx:
        # Regression — no bonus, no skip penalty
        return 0.0

    skipped = current_idx - last_idx - 1
    if skipped > 0:
        return PHASE_SKIP_PENALTY * skipped

    return PHASE_BONUS
