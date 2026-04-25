"""
TransitionEngine — mutates TrialLatentState per action.

Follows the Bio Experiment pattern: TransitionEngine updates hidden state,
OutputGenerator produces noisy observations from it. Agent never sees clean
hidden values.

Key responsibilities:
  - Enroll patients (ENROLL_PATIENTS)
  - Spend budget and advance time
  - Record adverse events
  - Set milestone flags (phase_i_complete, mtd_identified, effect_estimated,
    protocol_submitted, interim_complete, trial_complete)
  - Degrade data quality on soft violations
"""

from __future__ import annotations

import random

from models import ActionType, TrialAction, TrialLatentState


class TransitionEngine:
    """Mutates TrialLatentState in response to agent actions.

    All state transitions are deterministic given the same seed and action
    sequence (reproducibility requirement 9.2).
    """

    # Cost and time constants (per action type)
    _ACTION_COSTS: dict[ActionType, float] = {
        ActionType.SET_PRIMARY_ENDPOINT: 5_000.0,
        ActionType.SET_SAMPLE_SIZE: 2_000.0,
        ActionType.SET_INCLUSION_CRITERIA: 3_000.0,
        ActionType.SET_EXCLUSION_CRITERIA: 3_000.0,
        ActionType.SET_DOSING_SCHEDULE: 10_000.0,
        ActionType.SET_CONTROL_ARM: 5_000.0,
        ActionType.SET_RANDOMIZATION_RATIO: 2_000.0,
        ActionType.SET_BLINDING: 4_000.0,
        ActionType.RUN_DOSE_ESCALATION: 50_000.0,
        ActionType.OBSERVE_SAFETY_SIGNAL: 15_000.0,
        ActionType.ESTIMATE_EFFECT_SIZE: 20_000.0,
        ActionType.RUN_INTERIM_ANALYSIS: 30_000.0,
        ActionType.MODIFY_SAMPLE_SIZE: 5_000.0,
        ActionType.ADD_BIOMARKER_STRATIFICATION: 25_000.0,
        ActionType.SUBMIT_TO_FDA_REVIEW: 100_000.0,
        ActionType.REQUEST_PROTOCOL_AMENDMENT: 15_000.0,
        ActionType.RUN_PRIMARY_ANALYSIS: 50_000.0,
        ActionType.SYNTHESIZE_CONCLUSION: 10_000.0,
        ActionType.ENROLL_PATIENTS: 0.0,  # cost computed per patient
    }

    _ACTION_TIME_DAYS: dict[ActionType, int] = {
        ActionType.SET_PRIMARY_ENDPOINT: 7,
        ActionType.SET_SAMPLE_SIZE: 3,
        ActionType.SET_INCLUSION_CRITERIA: 5,
        ActionType.SET_EXCLUSION_CRITERIA: 5,
        ActionType.SET_DOSING_SCHEDULE: 14,
        ActionType.SET_CONTROL_ARM: 7,
        ActionType.SET_RANDOMIZATION_RATIO: 3,
        ActionType.SET_BLINDING: 5,
        ActionType.RUN_DOSE_ESCALATION: 90,
        ActionType.OBSERVE_SAFETY_SIGNAL: 30,
        ActionType.ESTIMATE_EFFECT_SIZE: 45,
        ActionType.RUN_INTERIM_ANALYSIS: 60,
        ActionType.MODIFY_SAMPLE_SIZE: 7,
        ActionType.ADD_BIOMARKER_STRATIFICATION: 30,
        ActionType.SUBMIT_TO_FDA_REVIEW: 180,
        ActionType.REQUEST_PROTOCOL_AMENDMENT: 30,
        ActionType.RUN_PRIMARY_ANALYSIS: 90,
        ActionType.SYNTHESIZE_CONCLUSION: 14,
        ActionType.ENROLL_PATIENTS: 0,  # time computed per patient
    }

    # Cost per patient enrolled (varies by disease area complexity)
    _COST_PER_PATIENT: float = 10_000.0
    _DAYS_PER_PATIENT: float = 2.0

    def __init__(self) -> None:
        """Initialize the TransitionEngine."""
        pass

    def apply_transition(
        self, latent: TrialLatentState, action: TrialAction
    ) -> TrialLatentState:
        """Apply *action* to *latent* and return the updated state.

        Does NOT mutate the input latent state — returns a new copy with
        updated fields.

        Args:
            latent: Current hidden state.
            action: Agent action to apply.

        Returns:
            Updated TrialLatentState with mutated fields.
        """
        # Create a mutable copy
        updated = latent.model_copy(deep=True)

        # Update action history
        updated.action_history.append(action.action_type.value)

        # Compute step-specific RNG
        step_index = len(updated.action_history)
        rng = random.Random(latent.seed ^ step_index)

        # --- Budget and time consumption ---
        base_cost = self._ACTION_COSTS.get(action.action_type, 0.0)
        base_time = self._ACTION_TIME_DAYS.get(action.action_type, 0)

        if action.action_type == ActionType.ENROLL_PATIENTS:
            n_patients = max(int(action.parameters.get("n_patients", 0)), 0)
            base_cost = n_patients * self._COST_PER_PATIENT
            base_time = int(n_patients * self._DAYS_PER_PATIENT)
            updated.patients_enrolled += n_patients

        updated.budget_remaining -= base_cost
        updated.time_remaining_days -= base_time

        # --- Milestone flag updates ---
        if action.action_type == ActionType.RUN_DOSE_ESCALATION:
            updated.phase_i_complete = True
            updated.mtd_identified = True

        if action.action_type == ActionType.ESTIMATE_EFFECT_SIZE:
            updated.effect_estimated = True

        if action.action_type == ActionType.SUBMIT_TO_FDA_REVIEW:
            updated.protocol_submitted = True

        if action.action_type == ActionType.RUN_INTERIM_ANALYSIS:
            updated.interim_complete = True

        if action.action_type == ActionType.RUN_PRIMARY_ANALYSIS:
            updated.trial_complete = True

        # --- Soft violation: degrade data quality ---
        # If action confidence is low (< 0.5), increase measurement noise
        if action.confidence < 0.5:
            degradation = 0.05 * (0.5 - action.confidence)
            updated.measurement_noise = min(
                updated.measurement_noise + degradation, 0.5
            )

        # If budget is negative (soft violation), degrade site variability
        if updated.budget_remaining < 0:
            updated.site_variability = min(updated.site_variability + 0.03, 0.5)

        # If time is negative (soft violation), increase dropout rate
        if updated.time_remaining_days < 0:
            updated.dropout_rate = min(updated.dropout_rate * 1.15, 0.8)

        # --- Phase progression (G23) ---
        # Advance episode_phase based on the action taken so the phase detector
        # and rule engine see a moving phase rather than a stuck "literature_review".
        # Phase names must match TRANSITION_TABLE keys in fda_rules.py.
        _PHASE_TRANSITIONS: dict[ActionType, str] = {
            ActionType.SET_PRIMARY_ENDPOINT: "hypothesis",
            ActionType.ESTIMATE_EFFECT_SIZE: "hypothesis",
            ActionType.SET_SAMPLE_SIZE: "design",
            ActionType.SET_INCLUSION_CRITERIA: "design",
            ActionType.SET_EXCLUSION_CRITERIA: "design",
            ActionType.SET_DOSING_SCHEDULE: "design",
            ActionType.SET_CONTROL_ARM: "design",
            ActionType.SET_RANDOMIZATION_RATIO: "design",
            ActionType.SET_BLINDING: "design",
            ActionType.ADD_BIOMARKER_STRATIFICATION: "design",
            ActionType.REQUEST_PROTOCOL_AMENDMENT: "design",
            ActionType.ENROLL_PATIENTS: "enrollment",
            ActionType.RUN_DOSE_ESCALATION: "enrollment",
            ActionType.OBSERVE_SAFETY_SIGNAL: "enrollment",
            ActionType.MODIFY_SAMPLE_SIZE: "enrollment",
            ActionType.RUN_INTERIM_ANALYSIS: "monitoring",
            ActionType.RUN_PRIMARY_ANALYSIS: "analysis",
            ActionType.SYNTHESIZE_CONCLUSION: "analysis",
            ActionType.SUBMIT_TO_FDA_REVIEW: "submission",
        }
        # Only advance — never go backwards
        _PHASE_ORDER = [
            "literature_review",
            "hypothesis",
            "design",
            "enrollment",
            "monitoring",
            "analysis",
            "submission",
        ]
        target_phase = _PHASE_TRANSITIONS.get(action.action_type)
        if target_phase is not None:
            try:
                current_idx = _PHASE_ORDER.index(updated.episode_phase)
                target_idx = _PHASE_ORDER.index(target_phase)
                if target_idx > current_idx:
                    updated.episode_phase = target_phase
            except ValueError:
                updated.episode_phase = target_phase

        # --- Adverse event recording (stochastic) ---
        # On certain actions, record adverse events based on true_side_effect_rate
        if action.action_type in {
            ActionType.ENROLL_PATIENTS,
            ActionType.OBSERVE_SAFETY_SIGNAL,
            ActionType.RUN_DOSE_ESCALATION,
        }:
            # For ENROLL_PATIENTS, scale AEs with number of patients
            n_exposed = 1
            if action.action_type == ActionType.ENROLL_PATIENTS:
                n_exposed = max(action.parameters.get("n_patients", 1), 1)
            # Each exposed patient has independent AE chance
            ae_count = sum(
                1 for _ in range(n_exposed)
                if rng.random() < updated.true_side_effect_rate
            )
            if ae_count > 0:
                updated.adverse_events += ae_count
                updated.site_variability = min(
                    updated.site_variability + 0.02 * ae_count, 0.5
                )

        return updated
