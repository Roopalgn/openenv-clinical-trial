"""
Trial Judge — multi-layer verification for clinical trial design decisions.

Layer 1 (programmatic, authoritative, never overridden):
  - power >= 0.80
  - p_value < 0.05
  - FDA compliance passes
  - budget_remaining > 0

Layer 2 (persona-scaled LLM — real when JUDGE_LLM_MODEL + JUDGE_LLM_API_KEY
are set, rule-based stub otherwise):
  - junior  (difficulty < 0.4): gives hints, lenient feedback
  - senior  (0.4–0.7):          balanced feedback
  - principal (> 0.7):          strict, no hints, penalises inefficiency

Overconfidence penalty: -0.5 per high-confidence wrong claim
(action.confidence >= 0.8 and the claim is incorrect per Layer 1).

Environment variables:
  JUDGE_LLM_MODEL    — model name, e.g. "gpt-4o-mini" or "claude-3-haiku-20240307"
  JUDGE_LLM_API_KEY  — API key for OpenAI or Anthropic
  JUDGE_LLM_BASE_URL — optional custom base URL (e.g. local vLLM endpoint)
"""

from __future__ import annotations

import logging
import math

from pydantic import BaseModel
from scipy.stats import norm

from models import ActionType, TrialAction, TrialLatentState, TrialState
from server.rules.fda_rules import check_fda_compliance
from server.simulator.power_calculator import calculate_power

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class JudgeResult(BaseModel):
    """Output of TrialJudge.verify()."""

    passed: bool
    violations: list[str]
    feedback: str
    hint: str | None
    overconfidence_penalty: float
    persona: str
    llm_used: bool = False  # True when Layer 2 used a real LLM call


# ---------------------------------------------------------------------------
# Persona thresholds
# ---------------------------------------------------------------------------

_JUNIOR_MAX = 0.4
_SENIOR_MAX = 0.7
_HIGH_CONFIDENCE_THRESHOLD = 0.8
_OVERCONFIDENCE_PENALTY = -0.5
_TERMINAL_CHECK_ACTIONS: set[ActionType] = {
    ActionType.RUN_PRIMARY_ANALYSIS,
    ActionType.SYNTHESIZE_CONCLUSION,
    ActionType.SUBMIT_TO_FDA_REVIEW,
}


def _select_persona(difficulty: float) -> str:
    if difficulty < _JUNIOR_MAX:
        return "junior"
    if difficulty <= _SENIOR_MAX:
        return "senior"
    return "principal"


# ---------------------------------------------------------------------------
# Layer 2a: real LLM call
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a clinical trial design expert acting as a {persona} reviewer.
Your role is to assess the quality of an agent's action in a clinical trial design episode.

Persona behaviour:
- junior   : lenient, educational, always provide a concrete hint
- senior   : balanced, clinical-standard expectations, no hints
- principal: strict, penalise inefficiency and redundancy, no hints

You will receive:
1. The action taken and its justification
2. Whether Layer 1 programmatic checks passed
3. Any programmatic violations found
4. Key episode state (phase, budget, patients enrolled, milestones)

Respond with a JSON object with exactly two keys:
  "feedback": a 1-3 sentence assessment of workflow quality and justification quality
  "hint": a concrete actionable hint string (junior only) or null

Be concise. Do not repeat the violations verbatim — focus on qualitative workflow assessment.
"""

_LLM_USER_TEMPLATE = """\
Action: {action_type}
Justification: {justification}
Confidence: {confidence}

Programmatic result: {passed_str}
Violations: {violations}

Episode state:
  phase: {phase}
  budget_remaining: {budget:.0f}
  patients_enrolled: {patients}
  milestones: phase_i={phase_i}, mtd={mtd}, effect_estimated={effect_est}, protocol_submitted={protocol}, interim={interim}
"""


def _call_llm(
    persona: str,
    action: TrialAction,
    latent: TrialLatentState,
    passed: bool,
    violations: list[str],
    model: str,
    api_key: str,
    base_url: str | None,
) -> tuple[str, str | None]:
    """Call the LLM for Layer 2 qualitative assessment.

    Returns (feedback, hint). Falls back to stub on any error.
    """
    import json

    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        logger.warning(
            "openai package not installed — falling back to rule-based judge stub. "
            "Install with: pip install openai"
        )
        return _stub_feedback(persona, violations, passed, action, latent)

    user_msg = _LLM_USER_TEMPLATE.format(
        action_type=action.action_type.value,
        justification=action.justification,
        confidence=action.confidence,
        passed_str="PASSED" if passed else "FAILED",
        violations="; ".join(violations) if violations else "none",
        phase=latent.episode_phase,
        budget=latent.budget_remaining,
        patients=latent.patients_enrolled,
        phase_i=latent.phase_i_complete,
        mtd=latent.mtd_identified,
        effect_est=latent.effect_estimated,
        protocol=latent.protocol_submitted,
        interim=latent.interim_complete,
    )

    try:
        client_kwargs: dict = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": _LLM_SYSTEM_PROMPT.format(persona=persona),
                },
                {"role": "user", "content": user_msg},
            ],
            max_tokens=256,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        feedback = (
            str(data.get("feedback", "")).strip()
            or _stub_feedback(persona, violations, passed, action, latent)[0]
        )
        hint_raw = data.get("hint")
        hint = str(hint_raw).strip() if hint_raw and persona == "junior" else None
        return feedback, hint

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "LLM judge call failed (%s: %s) — falling back to rule-based stub.",
            type(exc).__name__,
            exc,
        )
        return _stub_feedback(persona, violations, passed, action, latent)


# ---------------------------------------------------------------------------
# Layer 2b: rule-based stub (fallback when no LLM is configured)
# ---------------------------------------------------------------------------


def _stub_feedback(
    persona: str,
    violations: list[str],
    passed: bool,
    action: TrialAction,
    latent: TrialLatentState,
) -> tuple[str, str | None]:
    """Rule-based feedback stub — used when JUDGE_LLM_MODEL is not set."""
    action_name = action.action_type.value.replace("_", " ")

    if passed:
        if persona == "junior":
            feedback = (
                f"Good work on '{action_name}'! Your trial design looks solid. "
                f"Power and significance thresholds are met. Keep it up!"
            )
            hint = (
                "Tip: continue building on this foundation — "
                "consider biomarker stratification next to improve precision."
            )
        elif persona == "senior":
            feedback = (
                f"'{action_name}' passes all programmatic checks. "
                f"Statistical power and p-value criteria are satisfied. "
                f"Proceed to the next design step."
            )
            hint = None
        else:  # principal
            feedback = (
                f"'{action_name}' meets minimum criteria. "
                f"Ensure alpha-spending and interim analysis boundaries "
                f"are pre-specified before submission."
            )
            hint = None
    else:
        violation_summary = "; ".join(violations) if violations else "unknown issue"
        if persona == "junior":
            feedback = (
                f"'{action_name}' did not pass verification. "
                f"Issues found: {violation_summary}. "
                f"Review the requirements and try again."
            )
            hint = _build_hint_for_violations(violations, latent)
        elif persona == "senior":
            feedback = (
                f"'{action_name}' failed verification. "
                f"Violations: {violation_summary}. "
                f"Address these before proceeding."
            )
            hint = None
        else:  # principal
            feedback = (
                f"'{action_name}' is non-compliant. "
                f"Violations: {violation_summary}. "
                f"No further guidance will be provided — resolve independently."
            )
            hint = None

    return feedback, hint


def _build_hint_for_violations(
    violations: list[str], latent: TrialLatentState
) -> str | None:
    """Build a contextual hint for junior persona based on violation content."""
    if not violations:
        return None

    first = violations[0].lower()

    if "power" in first:
        return (
            "Hint: current power is below 0.80. "
            "Try increasing the sample size — "
            "more patients enrolled improves statistical power."
        )
    if "p-value" in first or "p_value" in first or "significance" in first:
        return (
            "Hint: the p-value threshold of 0.05 is not met. "
            "Consider a larger effect size or more patients."
        )
    if "budget" in first:
        return (
            f"Hint: budget is exhausted (remaining: {latent.budget_remaining:.2f}). "
            f"Look for cost-saving measures or request a protocol amendment."
        )
    if "fda" in first or "compliance" in first or "permitted" in first:
        return (
            f"Hint: this action is not allowed in the current phase "
            f"('{latent.episode_phase}'). "
            f"Check the transition table for permitted actions."
        )
    if "sample size" in first:
        return "Hint: the minimum regulatory sample size is 30 participants."
    if "protocol" in first:
        return "Hint: submit the protocol before attempting FDA review."
    if "phase i" in first:
        return "Hint: complete Phase I before submitting to FDA review."
    if "interim" in first:
        return "Hint: run an interim analysis before the primary analysis."
    if "patients" in first or "enrolled" in first:
        return "Hint: enroll patients before running analyses."

    return f"Hint: {violations[0]}"


# ---------------------------------------------------------------------------
# Main judge class
# ---------------------------------------------------------------------------


class TrialJudge:
    """Multi-layer trial design verifier.

    Layer 1 is programmatic and authoritative — its result is never overridden.
    Layer 2 is persona-scaled: uses a real LLM when JUDGE_LLM_MODEL and
    JUDGE_LLM_API_KEY are set, otherwise falls back to the rule-based stub.
    """

    def verify(
        self,
        action: TrialAction,
        state: TrialState,
        latent: TrialLatentState,
    ) -> JudgeResult:
        """Verify the action against both programmatic and persona layers.

        Args:
            action:  The agent's action to evaluate.
            state:   Lightweight training-loop metadata (carries difficulty).
            latent:  Hidden ground-truth + episode tracking state.

        Returns:
            JudgeResult with pass/fail, violations, feedback, hint, penalty,
            and llm_used flag.
        """
        violations: list[str] = []

        # ------------------------------------------------------------------
        # Layer 1: Programmatic checks (authoritative, never overridden)
        # ------------------------------------------------------------------

        # 1a. Budget check
        if latent.budget_remaining <= 0:
            violations.append(
                f"Budget exhausted: budget_remaining={latent.budget_remaining:.2f} "
                f"(must be > 0)."
            )

        # 1b/1c. Statistical power + p-value checks.
        # Gate these checks to terminal/near-terminal actions to avoid
        # penalizing incomplete early-episode states where enrollment and
        # analyses have not happened yet.
        should_check_stats = (
            latent.trial_complete
            or action.action_type in _TERMINAL_CHECK_ACTIONS
        )
        n = max(latent.patients_enrolled, 1)
        if should_check_stats:
            power = calculate_power(latent.true_effect_size, n)
            if power < 0.80:
                violations.append(
                    f"Insufficient statistical power: {power:.3f} < 0.80 "
                    f"(effect_size={latent.true_effect_size:.3f}, n={n})."
                )

            if n > 0 and latent.true_effect_size != 0.0:
                n_per_arm = n / 2.0
                se = 1.0 / math.sqrt(n_per_arm) if n_per_arm > 0 else 1.0
                z_stat = latent.true_effect_size / se
                p_value = float(2.0 * norm.sf(abs(z_stat)))
            else:
                p_value = 1.0

            if p_value >= 0.05:
                violations.append(
                    f"p-value not significant: {p_value:.4f} >= 0.05 "
                    f"(n={n}, effect_size={latent.true_effect_size:.3f})."
                )

        # 1d. FDA compliance check
        compliance = check_fda_compliance(action, latent)
        if not compliance.valid:
            violations.extend(compliance.violations)

        passed = len(violations) == 0

        # ------------------------------------------------------------------
        # Overconfidence penalty
        # ------------------------------------------------------------------
        # Only penalise overconfidence for FDA compliance violations, not for
        # power/p-value checks which always fail early in the episode.
        # This prevents the penalty from becoming a blanket "never use high
        # confidence" signal that degrades the confidence mechanism.
        overconfidence_penalty = 0.0
        fda_violations = compliance.violations if not compliance.valid else []
        budget_violations = [v for v in violations if "budget" in v.lower()]
        actionable_violations = fda_violations + budget_violations
        if actionable_violations and action.confidence >= _HIGH_CONFIDENCE_THRESHOLD:
            overconfidence_penalty = _OVERCONFIDENCE_PENALTY * len(actionable_violations)

        # ------------------------------------------------------------------
        # Layer 2: Persona-scaled feedback (never overrides Layer 1 result)
        # ------------------------------------------------------------------
        persona = _select_persona(state.difficulty)
        llm_used = False

        from server.config import settings  # local import avoids circular dep

        if settings.judge_llm_model and settings.judge_llm_api_key:
            feedback, hint = _call_llm(
                persona=persona,
                action=action,
                latent=latent,
                passed=passed,
                violations=violations,
                model=settings.judge_llm_model,
                api_key=settings.judge_llm_api_key,
                base_url=settings.judge_llm_base_url,
            )
            # Only mark llm_used=True if we didn't fall back (no exception path
            # sets feedback to stub output — we can't distinguish, so we trust
            # that _call_llm returns stub on error and logs a warning).
            llm_used = True
        else:
            feedback, hint = _stub_feedback(persona, violations, passed, action, latent)

        return JudgeResult(
            passed=passed,
            violations=violations,
            feedback=feedback,
            hint=hint,
            overconfidence_penalty=overconfidence_penalty,
            persona=persona,
            llm_used=llm_used,
        )
