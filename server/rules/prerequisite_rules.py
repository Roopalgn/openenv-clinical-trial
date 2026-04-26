"""
Prerequisite rule checks for clinical trial actions.

Provides check_prerequisites which returns a list of violation strings.
"""

from __future__ import annotations

from models import ActionType, TrialAction, TrialLatentState

# Maps each ActionType to the action(s) that must appear in action_history first.
_HISTORY_PREREQUISITES: dict[ActionType, list[ActionType]] = {
    ActionType.SET_DOSING_SCHEDULE: [ActionType.SET_PRIMARY_ENDPOINT],
    ActionType.SET_CONTROL_ARM: [ActionType.SET_PRIMARY_ENDPOINT],
    ActionType.SET_RANDOMIZATION_RATIO: [ActionType.SET_CONTROL_ARM],
    ActionType.SET_BLINDING: [ActionType.SET_RANDOMIZATION_RATIO],
    ActionType.RUN_DOSE_ESCALATION: [ActionType.SET_DOSING_SCHEDULE],
    ActionType.ADD_BIOMARKER_STRATIFICATION: [ActionType.SET_INCLUSION_CRITERIA],
}


def check_prerequisites(action: TrialAction, latent: TrialLatentState) -> list[str]:
    """Return a list of prerequisite violation strings for *action* given *latent*.

    Returns an empty list when all prerequisites are satisfied.
    Does NOT mutate *latent*.
    """
    violations: list[str] = []

    # History-based prerequisites
    required_actions = _HISTORY_PREREQUISITES.get(action.action_type, [])
    for required in required_actions:
        if required.value not in latent.action_history:
            violations.append(
                f"Action '{action.action_type.value}' requires '{required.value}' "
                f"to have been performed first, but it is not in the action history."
            )

    # State-flag prerequisites
    if action.action_type == ActionType.REQUEST_PROTOCOL_AMENDMENT:
        if not latent.protocol_submitted:
            violations.append(
                "Cannot request protocol amendment: protocol has not been submitted "
                "(protocol_submitted=False)."
            )

    return violations
