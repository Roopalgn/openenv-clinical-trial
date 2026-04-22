"""
Tests for server/phase_detector.py

Validates Requirements 8.5 and 9.4:
  - detect_phase classifies actions into correct clinical workflow phases
  - phase_order_correct is True for valid transitions, False for regressions/skips
  - compute_phase_ordering_reward returns correct bonus/penalty values
"""

from __future__ import annotations

import pytest

from models import ActionType, TrialAction
from server.phase_detector import (
    PHASE_BONUS,
    PHASE_ORDER,
    PHASE_SKIP_PENALTY,
    compute_phase_ordering_reward,
    detect_phase,
)


def _action(action_type: ActionType) -> TrialAction:
    return TrialAction(
        action_type=action_type,
        parameters={},
        justification="test",
        confidence=0.5,
    )


# ---------------------------------------------------------------------------
# Phase mapping tests
# ---------------------------------------------------------------------------


class TestPhaseMapping:
    def test_hypothesis_actions(self):
        for at in [
            ActionType.ESTIMATE_EFFECT_SIZE,
            ActionType.ADD_BIOMARKER_STRATIFICATION,
        ]:
            phase, _ = detect_phase(_action(at), [])
            assert phase == "hypothesis", f"{at} should map to hypothesis"

    def test_design_actions(self):
        design_actions = [
            ActionType.SET_PRIMARY_ENDPOINT,
            ActionType.SET_SAMPLE_SIZE,
            ActionType.SET_INCLUSION_CRITERIA,
            ActionType.SET_EXCLUSION_CRITERIA,
            ActionType.SET_DOSING_SCHEDULE,
            ActionType.SET_CONTROL_ARM,
            ActionType.SET_RANDOMIZATION_RATIO,
            ActionType.SET_BLINDING,
            ActionType.REQUEST_PROTOCOL_AMENDMENT,
        ]
        for at in design_actions:
            phase, _ = detect_phase(_action(at), [])
            assert phase == "design", f"{at} should map to design"

    def test_enrollment_action(self):
        phase, _ = detect_phase(_action(ActionType.ENROLL_PATIENTS), [])
        assert phase == "enrollment"

    def test_monitoring_actions(self):
        monitoring_actions = [
            ActionType.RUN_DOSE_ESCALATION,
            ActionType.OBSERVE_SAFETY_SIGNAL,
            ActionType.RUN_INTERIM_ANALYSIS,
            ActionType.MODIFY_SAMPLE_SIZE,
        ]
        for at in monitoring_actions:
            phase, _ = detect_phase(_action(at), [])
            assert phase == "monitoring", f"{at} should map to monitoring"

    def test_analysis_actions(self):
        for at in [ActionType.RUN_PRIMARY_ANALYSIS, ActionType.SYNTHESIZE_CONCLUSION]:
            phase, _ = detect_phase(_action(at), [])
            assert phase == "analysis", f"{at} should map to analysis"

    def test_submission_action(self):
        phase, _ = detect_phase(_action(ActionType.SUBMIT_TO_FDA_REVIEW), [])
        assert phase == "submission"


# ---------------------------------------------------------------------------
# Phase order correctness tests
# ---------------------------------------------------------------------------


class TestPhaseOrderCorrectness:
    def test_empty_history_always_correct(self):
        for at in ActionType:
            _, correct = detect_phase(_action(at), [])
            assert correct is True, f"Empty history should always be correct for {at}"

    def test_same_phase_is_correct(self):
        _, correct = detect_phase(_action(ActionType.SET_SAMPLE_SIZE), ["design"])
        assert correct is True

    def test_advance_one_phase_is_correct(self):
        _, correct = detect_phase(_action(ActionType.ENROLL_PATIENTS), ["design"])
        assert correct is True

    def test_regression_is_incorrect(self):
        # Going from enrollment back to design
        _, correct = detect_phase(_action(ActionType.SET_SAMPLE_SIZE), ["enrollment"])
        assert correct is False

    def test_skip_one_phase_is_incorrect(self):
        # Jumping from hypothesis to enrollment (skipping design)
        _, correct = detect_phase(_action(ActionType.ENROLL_PATIENTS), ["hypothesis"])
        assert correct is False

    def test_skip_multiple_phases_is_incorrect(self):
        # Jumping from design to analysis (skipping enrollment + monitoring)
        _, correct = detect_phase(_action(ActionType.RUN_PRIMARY_ANALYSIS), ["design"])
        assert correct is False

    def test_valid_full_sequence(self):
        """Walk through the full phase sequence and verify all transitions are correct."""
        history: list[str] = []
        sequence = [
            ActionType.ESTIMATE_EFFECT_SIZE,  # hypothesis
            ActionType.SET_PRIMARY_ENDPOINT,  # design
            ActionType.ENROLL_PATIENTS,  # enrollment
            ActionType.RUN_DOSE_ESCALATION,  # monitoring
            ActionType.RUN_PRIMARY_ANALYSIS,  # analysis
            ActionType.SUBMIT_TO_FDA_REVIEW,  # submission
        ]
        for at in sequence:
            phase, correct = detect_phase(_action(at), history)
            assert (
                correct is True
            ), f"Expected correct order for {at} with history {history}"
            history.append(phase)


# ---------------------------------------------------------------------------
# PHASE_ORDER constant
# ---------------------------------------------------------------------------


class TestPhaseOrderConstant:
    def test_phase_order_has_seven_phases(self):
        assert len(PHASE_ORDER) == 7

    def test_phase_order_sequence(self):
        assert PHASE_ORDER == [
            "literature_review",
            "hypothesis",
            "design",
            "enrollment",
            "monitoring",
            "analysis",
            "submission",
        ]


# ---------------------------------------------------------------------------
# compute_phase_ordering_reward tests
# ---------------------------------------------------------------------------


class TestComputePhaseOrderingReward:
    def test_empty_history_returns_bonus(self):
        reward = compute_phase_ordering_reward(_action(ActionType.SET_SAMPLE_SIZE), [])
        assert reward == PHASE_BONUS

    def test_correct_advance_returns_bonus(self):
        reward = compute_phase_ordering_reward(
            _action(ActionType.ENROLL_PATIENTS), ["design"]
        )
        assert reward == PHASE_BONUS

    def test_same_phase_returns_bonus(self):
        reward = compute_phase_ordering_reward(
            _action(ActionType.SET_SAMPLE_SIZE), ["design"]
        )
        assert reward == PHASE_BONUS

    def test_regression_returns_zero(self):
        reward = compute_phase_ordering_reward(
            _action(ActionType.SET_SAMPLE_SIZE), ["enrollment"]
        )
        assert reward == 0.0

    def test_skip_one_phase_returns_single_penalty(self):
        # hypothesis → enrollment skips design (1 skip)
        reward = compute_phase_ordering_reward(
            _action(ActionType.ENROLL_PATIENTS), ["hypothesis"]
        )
        assert reward == pytest.approx(PHASE_SKIP_PENALTY * 1)

    def test_skip_two_phases_returns_double_penalty(self):
        # design → monitoring skips enrollment (1 skip)
        # design → analysis skips enrollment + monitoring (2 skips)
        reward = compute_phase_ordering_reward(
            _action(ActionType.RUN_PRIMARY_ANALYSIS), ["design"]
        )
        assert reward == pytest.approx(PHASE_SKIP_PENALTY * 2)

    def test_constants_values(self):
        assert PHASE_BONUS == 0.2
        assert PHASE_SKIP_PENALTY == -0.3
