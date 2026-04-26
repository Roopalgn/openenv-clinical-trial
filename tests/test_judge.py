"""
Tests for server/judge.py — TrialJudge multi-layer verification.

Covers:
  - Layer 1 programmatic checks (power, p-value, FDA compliance, budget)
  - Layer 2 persona selection (junior/senior/principal)
  - Layer 2 LLM path (mocked) and stub fallback
  - Overconfidence penalty
  - Hint generation for junior persona
  - No unhandled exceptions on any valid input (req 10.4)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from models import ActionType, TrialAction, TrialLatentState, TrialResult, TrialState
from server.judge import JudgeResult, TrialJudge, _select_persona

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_latent(**overrides) -> TrialLatentState:
    defaults = dict(
        true_effect_size=0.8,
        true_side_effect_rate=0.05,
        true_responder_population="all",
        true_responder_criteria=[],
        true_dose_response={},
        true_mechanism="unknown",
        placebo_response_rate=0.1,
        dropout_rate=0.05,
        site_variability=0.0,
        measurement_noise=0.0,
        budget_remaining=500_000.0,
        time_remaining_days=300,
        patients_enrolled=200,
        phase_i_complete=True,
        mtd_identified=True,
        effect_estimated=True,
        protocol_submitted=True,
        interim_complete=True,
        trial_complete=True,
        adverse_events=0,
        episode_phase="analysis",
        action_history=["run_primary_analysis"],
        seed=42,
    )
    defaults.update(overrides)
    return TrialLatentState(**defaults)


def _make_state(difficulty: float = 0.3) -> TrialState:
    return TrialState(
        episode_id="test-ep",
        step_count=5,
        difficulty=difficulty,
        scenario_id="solid_tumor_chemo",
        curriculum_tier="0",
        curriculum_stats={},
        action_diversity=0.8,
        phase_compliance_rate=1.0,
        is_resolved=False,
    )


def _make_action(
    action_type: ActionType = ActionType.RUN_PRIMARY_ANALYSIS,
    confidence: float = 0.5,
    **params,
) -> TrialAction:
    return TrialAction(
        action_type=action_type,
        parameters=params,
        justification="test",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Persona selection
# ---------------------------------------------------------------------------


def test_persona_junior():
    assert _select_persona(0.0) == "junior"
    assert _select_persona(0.39) == "junior"


def test_persona_senior():
    assert _select_persona(0.4) == "senior"
    assert _select_persona(0.7) == "senior"


def test_persona_principal():
    assert _select_persona(0.71) == "principal"
    assert _select_persona(1.0) == "principal"


# ---------------------------------------------------------------------------
# Layer 1: budget check
# ---------------------------------------------------------------------------


def test_budget_exhausted_fails():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(), latent)
    assert not result.passed
    assert any("budget" in v.lower() for v in result.violations)


def test_budget_negative_fails():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=-100.0)
    result = judge.verify(_make_action(), _make_state(), latent)
    assert not result.passed
    assert any("budget" in v.lower() for v in result.violations)


def test_budget_positive_passes_budget_check():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=1.0)
    # Other checks may still fail, but budget violation should not be present
    result = judge.verify(_make_action(), _make_state(), latent)
    assert not any("budget" in v.lower() for v in result.violations)


# ---------------------------------------------------------------------------
# Layer 1: power check
# ---------------------------------------------------------------------------


def test_low_power_fails():
    judge = TrialJudge()
    # Very small effect + few patients → low power
    latent = _make_latent(true_effect_size=0.01, patients_enrolled=10)
    result = judge.verify(_make_action(), _make_state(), latent)
    assert not result.passed
    assert any("power" in v.lower() for v in result.violations)


def test_sufficient_power_no_power_violation():
    judge = TrialJudge()
    # Large effect + many patients → high power
    latent = _make_latent(true_effect_size=1.5, patients_enrolled=500)
    result = judge.verify(_make_action(), _make_state(), latent)
    assert not any("power" in v.lower() for v in result.violations)


# ---------------------------------------------------------------------------
# Layer 1: p-value check
# ---------------------------------------------------------------------------


def test_nonsignificant_pvalue_fails():
    judge = TrialJudge()
    # Zero effect → p-value = 1.0
    latent = _make_latent(true_effect_size=0.0, patients_enrolled=100)
    result = judge.verify(_make_action(), _make_state(), latent)
    assert not result.passed
    assert any("p-value" in v.lower() for v in result.violations)


def test_significant_pvalue_no_pvalue_violation():
    judge = TrialJudge()
    # Large effect + many patients → very small p-value
    latent = _make_latent(true_effect_size=2.0, patients_enrolled=1000)
    result = judge.verify(_make_action(), _make_state(), latent)
    assert not any("p-value" in v.lower() for v in result.violations)


# ---------------------------------------------------------------------------
# Layer 1: FDA compliance
# ---------------------------------------------------------------------------


def test_fda_violation_propagated():
    judge = TrialJudge()
    # Action not permitted in current phase
    latent = _make_latent(episode_phase="literature_review")
    action = _make_action(action_type=ActionType.SUBMIT_TO_FDA_REVIEW)
    result = judge.verify(action, _make_state(), latent)
    assert not result.passed
    assert len(result.violations) > 0


# ---------------------------------------------------------------------------
# Overconfidence penalty
# ---------------------------------------------------------------------------


def test_overconfidence_penalty_applied_when_high_confidence_and_wrong():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)  # guaranteed violation
    action = _make_action(confidence=0.9)
    result = judge.verify(action, _make_state(), latent)
    assert not result.passed
    assert result.overconfidence_penalty < 0.0


def test_no_overconfidence_penalty_when_low_confidence():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)  # violation present
    action = _make_action(confidence=0.5)
    result = judge.verify(action, _make_state(), latent)
    assert result.overconfidence_penalty == 0.0


def test_no_overconfidence_penalty_when_passed():
    judge = TrialJudge()
    # Use large effect + many patients to pass power/p-value, valid phase/action
    latent = _make_latent(
        true_effect_size=2.0,
        patients_enrolled=1000,
        budget_remaining=500_000.0,
        episode_phase="analysis",
        interim_complete=True,
        trial_complete=True,
    )
    action = _make_action(action_type=ActionType.RUN_PRIMARY_ANALYSIS, confidence=0.95)
    result = judge.verify(action, _make_state(), latent)
    if result.passed:
        assert result.overconfidence_penalty == 0.0


def test_overconfidence_penalty_scales_with_violation_count():
    judge = TrialJudge()
    # Budget violation is an actionable violation that triggers overconfidence.
    # Power/p-value violations are excluded from penalty (M7 fix).
    latent = _make_latent(
        budget_remaining=0.0,
        true_effect_size=0.0,
        patients_enrolled=1,
    )
    action = _make_action(confidence=0.9)
    result = judge.verify(action, _make_state(), latent)
    # Only budget violation counts for overconfidence penalty now
    assert result.overconfidence_penalty <= -0.5  # at least 1 actionable violation × -0.5


# ---------------------------------------------------------------------------
# Layer 2: persona in result
# ---------------------------------------------------------------------------


def test_junior_persona_in_result():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(difficulty=0.2), latent)
    assert result.persona == "junior"


def test_senior_persona_in_result():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(difficulty=0.5), latent)
    assert result.persona == "senior"


def test_principal_persona_in_result():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(difficulty=0.9), latent)
    assert result.persona == "principal"


# ---------------------------------------------------------------------------
# Layer 2: hints
# ---------------------------------------------------------------------------


def test_junior_gets_hint_on_failure():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(difficulty=0.2), latent)
    assert not result.passed
    assert result.hint is not None and len(result.hint) > 0


def test_senior_no_hint_on_failure():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(difficulty=0.5), latent)
    assert result.hint is None


def test_principal_no_hint_on_failure():
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(difficulty=0.9), latent)
    assert result.hint is None


def test_junior_gets_hint_on_pass():
    judge = TrialJudge()
    latent = _make_latent(
        true_effect_size=2.0,
        patients_enrolled=1000,
        budget_remaining=500_000.0,
        episode_phase="analysis",
        interim_complete=True,
        trial_complete=True,
    )
    action = _make_action(action_type=ActionType.RUN_PRIMARY_ANALYSIS)
    result = judge.verify(action, _make_state(difficulty=0.2), latent)
    if result.passed:
        assert result.hint is not None


# ---------------------------------------------------------------------------
# JudgeResult model
# ---------------------------------------------------------------------------


def test_judge_result_is_pydantic_model():
    result = JudgeResult(
        passed=True,
        violations=[],
        feedback="ok",
        hint=None,
        overconfidence_penalty=0.0,
        persona="senior",
    )
    assert result.passed is True
    assert result.persona == "senior"
    assert result.llm_used is False  # default


# ---------------------------------------------------------------------------
# Layer 2: llm_used flag — stub path
# ---------------------------------------------------------------------------


def test_llm_used_false_when_no_llm_configured():
    """llm_used is False when JUDGE_LLM_MODEL is not set (stub path)."""
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)
    result = judge.verify(_make_action(), _make_state(), latent)
    assert result.llm_used is False


# ---------------------------------------------------------------------------
# Layer 2: LLM path (mocked)
# ---------------------------------------------------------------------------


def test_llm_path_used_when_configured():
    """When JUDGE_LLM_MODEL and JUDGE_LLM_API_KEY are set, llm_used=True."""
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)

    # Mock the OpenAI client response
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{"feedback": "LLM feedback text.", "hint": "LLM hint."}'
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with (
        patch("server.config.settings") as mock_settings,
        patch("server.judge.OpenAI", return_value=mock_client, create=True),
    ):
        mock_settings.judge_llm_model = "gpt-4o-mini"
        mock_settings.judge_llm_api_key = "sk-test"
        mock_settings.judge_llm_base_url = None

        # Patch the import inside _call_llm
        with patch.dict(
            "sys.modules", {"openai": MagicMock(OpenAI=lambda **kw: mock_client)}
        ):
            result = judge.verify(_make_action(), _make_state(difficulty=0.2), latent)

    assert result.llm_used is True


def test_llm_fallback_to_stub_on_import_error():
    """When openai is not importable, falls back to stub without raising."""
    judge = TrialJudge()
    latent = _make_latent(budget_remaining=0.0)

    with (
        patch("server.config.settings") as mock_settings,
        patch.dict("sys.modules", {"openai": None}),
    ):
        mock_settings.judge_llm_model = "gpt-4o-mini"
        mock_settings.judge_llm_api_key = "sk-test"
        mock_settings.judge_llm_base_url = None

        # Should not raise — falls back to stub
        result = judge.verify(_make_action(), _make_state(), latent)
        assert isinstance(result, JudgeResult)
        assert result.feedback  # stub still produces feedback
        assert result.llm_used is False


def test_verify_uses_supplied_trial_result_for_stat_checks():
    """When a TrialResult is supplied, judge stats should come from that result."""
    judge = TrialJudge()
    latent = _make_latent(
        true_effect_size=0.01,
        patients_enrolled=20,
        trial_complete=True,
    )
    action = _make_action()
    supplied_result = TrialResult(
        p_value=0.01,
        success=True,
        power=0.92,
        adverse_event_rate=0.1,
        confidence_interval=(0.0, 0.2),
        failure_reason=None,
    )

    result = judge.verify(action, _make_state(), latent, result=supplied_result)

    assert not any("power" in violation.lower() for violation in result.violations)
    assert not any("p-value" in violation.lower() for violation in result.violations)


# ---------------------------------------------------------------------------
# Req 10.4: no unhandled exceptions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action_type",
    list(ActionType),
)
def test_no_exception_for_any_action_type(action_type):
    """TrialJudge.verify must never raise for any valid action type (req 10.4)."""
    judge = TrialJudge()
    latent = _make_latent()
    state = _make_state()
    action = TrialAction(
        action_type=action_type,
        parameters={},
        justification="test",
        confidence=0.5,
    )
    # Must not raise
    result = judge.verify(action, state, latent)
    assert isinstance(result, JudgeResult)
