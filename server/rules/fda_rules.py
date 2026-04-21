"""
FDA compliance rule engine.

Provides TRANSITION_TABLE, ComplianceResult, and check_fda_compliance.
"""

from __future__ import annotations

from pydantic import BaseModel

from models import ActionType, TrialAction, TrialLatentState
from server.rules.prerequisite_rules import check_prerequisites

# Maps each episode phase to the set of permitted ActionTypes.
TRANSITION_TABLE: dict[str, set[ActionType]] = {
    "literature_review": {
        ActionType.SET_PRIMARY_ENDPOINT,
        ActionType.OBSERVE_SAFETY_SIGNAL,
        ActionType.ESTIMATE_EFFECT_SIZE,
    },
    "hypothesis": {
        ActionType.SET_PRIMARY_ENDPOINT,
        ActionType.SET_SAMPLE_SIZE,
        ActionType.SET_INCLUSION_CRITERIA,
        ActionType.SET_EXCLUSION_CRITERIA,
        ActionType.ESTIMATE_EFFECT_SIZE,
    },
    "design": {
        ActionType.SET_SAMPLE_SIZE,
        ActionType.SET_INCLUSION_CRITERIA,
        ActionType.SET_EXCLUSION_CRITERIA,
        ActionType.SET_DOSING_SCHEDULE,
        ActionType.SET_CONTROL_ARM,
        ActionType.SET_RANDOMIZATION_RATIO,
        ActionType.SET_BLINDING,
        ActionType.ADD_BIOMARKER_STRATIFICATION,
        ActionType.REQUEST_PROTOCOL_AMENDMENT,
        ActionType.ENROLL_PATIENTS,
    },
    "enrollment": {
        ActionType.ENROLL_PATIENTS,
        ActionType.RUN_DOSE_ESCALATION,
        ActionType.OBSERVE_SAFETY_SIGNAL,
        ActionType.MODIFY_SAMPLE_SIZE,
        ActionType.ADD_BIOMARKER_STRATIFICATION,
        ActionType.REQUEST_PROTOCOL_AMENDMENT,
    },
    "monitoring": {
        ActionType.RUN_INTERIM_ANALYSIS,
        ActionType.OBSERVE_SAFETY_SIGNAL,
        ActionType.MODIFY_SAMPLE_SIZE,
        ActionType.REQUEST_PROTOCOL_AMENDMENT,
    },
    "analysis": {
        ActionType.RUN_PRIMARY_ANALYSIS,
        ActionType.ESTIMATE_EFFECT_SIZE,
        ActionType.SYNTHESIZE_CONCLUSION,
    },
    "submission": {
        ActionType.SUBMIT_TO_FDA_REVIEW,
        ActionType.REQUEST_PROTOCOL_AMENDMENT,
        ActionType.SYNTHESIZE_CONCLUSION,
    },
}


class ComplianceResult(BaseModel):
    """Result of an FDA compliance check."""

    valid: bool
    violations: list[str]


def check_fda_compliance(
    action: TrialAction, latent: TrialLatentState
) -> ComplianceResult:
    """Check whether *action* is compliant given the current *latent* state.

    Does NOT mutate *latent*.

    Returns a ComplianceResult with valid=True and empty violations when all
    checks pass, or valid=False with descriptive violation messages otherwise.
    """
    violations: list[str] = []

    # 1. Transition table check — episode_phase lives in latent state
    permitted = TRANSITION_TABLE.get(latent.episode_phase, set())
    if action.action_type not in permitted:
        violations.append(
            f"Action '{action.action_type.value}' is not permitted in episode "
            f"phase '{latent.episode_phase}'. Permitted actions: "
            f"{sorted(a.value for a in permitted) if permitted else '[]'}."
        )

    # 2. FDA hard rules
    if action.action_type == ActionType.SET_SAMPLE_SIZE:
        sample_size = action.parameters.get("sample_size")
        if sample_size is not None and sample_size < 30:
            violations.append(
                f"Sample size {sample_size} is below the regulatory minimum of 30."
            )

    if action.action_type == ActionType.SUBMIT_TO_FDA_REVIEW:
        if not latent.protocol_submitted:
            violations.append(
                "Cannot submit to FDA review: protocol has not been submitted yet "
                "(protocol_submitted=False)."
            )
        if not latent.phase_i_complete:
            violations.append(
                "Cannot submit to FDA review: Phase I has not been completed "
                "(phase_i_complete=False)."
            )

    if action.action_type == ActionType.RUN_PRIMARY_ANALYSIS:
        if not latent.interim_complete:
            violations.append(
                "Cannot run primary analysis: interim analysis has not been completed "
                "(interim_complete=False)."
            )

    if action.action_type == ActionType.RUN_INTERIM_ANALYSIS:
        if latent.patients_enrolled <= 0:
            violations.append(
                "Cannot run interim analysis: no patients are enrolled "
                "(patients_enrolled=0)."
            )

    if action.action_type == ActionType.MODIFY_SAMPLE_SIZE:
        if ActionType.SET_SAMPLE_SIZE.value not in latent.action_history:
            violations.append(
                "Cannot modify sample size: SET_SAMPLE_SIZE has not been performed "
                "in this episode."
            )

    if action.action_type == ActionType.SYNTHESIZE_CONCLUSION:
        if not latent.trial_complete:
            violations.append(
                "Cannot synthesize conclusion: trial is not complete "
                "(trial_complete=False)."
            )

    # 3. Prerequisite checks
    prerequisite_violations = check_prerequisites(action, latent)
    violations.extend(prerequisite_violations)

    return ComplianceResult(valid=len(violations) == 0, violations=violations)
