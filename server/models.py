"""
Pydantic models for the Clinical Trial Designer environment.
All field names follow the naming contract defined in §5.1–5.4 of the project roadmap.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    SET_PRIMARY_ENDPOINT = "set_primary_endpoint"
    SET_SAMPLE_SIZE = "set_sample_size"
    SET_INCLUSION_CRITERIA = "set_inclusion_criteria"
    SET_EXCLUSION_CRITERIA = "set_exclusion_criteria"
    SET_DOSING_SCHEDULE = "set_dosing_schedule"
    SET_CONTROL_ARM = "set_control_arm"
    SET_RANDOMIZATION_RATIO = "set_randomization_ratio"
    SET_BLINDING = "set_blinding"
    RUN_DOSE_ESCALATION = "run_dose_escalation"
    OBSERVE_SAFETY_SIGNAL = "observe_safety_signal"
    ESTIMATE_EFFECT_SIZE = "estimate_effect_size"
    RUN_INTERIM_ANALYSIS = "run_interim_analysis"
    MODIFY_SAMPLE_SIZE = "modify_sample_size"
    ADD_BIOMARKER_STRATIFICATION = "add_biomarker_stratification"
    SUBMIT_TO_FDA_REVIEW = "submit_to_fda_review"
    REQUEST_PROTOCOL_AMENDMENT = "request_protocol_amendment"
    RUN_PRIMARY_ANALYSIS = "run_primary_analysis"
    SYNTHESIZE_CONCLUSION = "synthesize_conclusion"


class TrialAction(BaseModel):
    """Represents a single agent action in the trial design episode."""

    action_type: ActionType
    parameters: dict[str, Any]
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)


class TrialObservation(BaseModel):
    """Observable state returned to the agent after each step."""

    step_index: int
    episode_phase: str
    budget_remaining: float
    time_remaining_days: int
    patients_enrolled: int
    last_action_valid: bool
    violation_messages: list[str]
    observed_effect_estimate: float | None
    observed_side_effect_rate: float | None
    phase_i_complete: bool
    interim_complete: bool
    protocol_submitted: bool
    scenario_id: str
    curriculum_tier: int


class TrialState(BaseModel):
    """Full internal episode state including hidden ground-truth fields.

    Field names follow the §5.4 naming contract.
    """

    # Naming contract §5.4 fields (required)
    true_effect_size: float  # hidden — not exposed in TrialObservation
    true_side_effect_rate: float
    true_responder_population: str
    placebo_response_rate: float
    dropout_rate: float
    budget_remaining: float
    time_remaining_days: int
    patients_enrolled: int
    # Episode tracking
    step_index: int
    episode_phase: str
    scenario_id: str
    curriculum_tier: int
    action_history: list[str]  # action_type strings
    phase_i_complete: bool
    interim_complete: bool
    protocol_submitted: bool
    trial_complete: bool
    seed: int


class TrialLatentState(BaseModel):
    """Hidden ground-truth parameters not exposed in TrialObservation."""

    true_effect_size: float
    true_side_effect_rate: float
    true_responder_population: str
    placebo_response_rate: float
    dropout_rate: float
    site_variability: float
    measurement_noise: float
    true_dose_response: dict[float, float]
    true_mechanism: str
    true_responder_criteria: list[str]


class TrialResult(BaseModel):
    """Outcome of a simulated trial."""

    p_value: float
    success: bool  # p_value < alpha
    power: float
    adverse_event_rate: float
    confidence_interval: tuple[float, float]
    failure_reason: str | None  # "budget_exhausted", "time_exhausted", "no_enrollment"


class ScenarioConfig(BaseModel):
    """Scenario-specific configuration parameters."""

    scenario_id: str  # one of the four named IDs
    curriculum_tier: int
    disease_area: str
    effect_size_range: tuple[float, float]
    side_effect_rate_range: tuple[float, float]
    placebo_response_range: tuple[float, float]
    dropout_rate_range: tuple[float, float]
    budget_usd: float
    time_budget_days: int
    min_sample_size: int
    description: str


class RewardBreakdown(BaseModel):
    """Decomposed reward components for a single step."""

    r_validity: float
    r_ordering: float
    r_info_gain: float
    r_efficiency: float
    r_novelty: float
    r_penalty: float
    r_terminal_success: float
    r_terminal_calibration: float

    @property
    def total(self) -> float:
        return sum(
            [
                self.r_validity,
                self.r_ordering,
                self.r_info_gain,
                self.r_efficiency,
                self.r_novelty,
                self.r_penalty,
                self.r_terminal_success,
                self.r_terminal_calibration,
            ]
        )
