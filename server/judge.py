"""
Trial Judge — multi-layer verification for clinical trial design decisions.

Layer 1 (programmatic, authoritative, never overridden):
  - power >= 0.80
  - p_value < 0.05
  - FDA compliance passes
  - budget_remaining > 0

Layer 2 (persona-scaled LLM stub):
  - junior  (difficulty < 0.4): gives hints, lenient feedback
  - senior  (0.4–0.7):          balanced feedback
  - principal (> 0.7):          strict, no hints

Overconfidence penalty: -0.5 per high-confidence wrong claim
(action.confidence >= 0.8 and the claim is incorrect per Layer 1).
"""

from __future__ import annotations

from pydantic import BaseModel

from models import TrialAction, TrialLatentState, TrialState
from server.rules.fda_rules import check_fda_compliance
from server.simulator.power_calculator import calculate_power

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


# ---------------------------------------------------------------------------
# Persona thresholds
# ---------------------------------------------------------------------------

_JUNIOR_MAX = 0.4
_SENIOR_MAX = 0.7
_HIGH_CONFIDENCE_THRESHOLD = 0.8
_OVERCONFIDENCE_PENALTY = -0.5


def _select_persona(difficulty: float) -> str:
    if difficulty < _JUNIOR_MAX:
        return "junior"
    if difficulty <= _SENIOR_MAX:
        return "senior"
    return "principal"


# ---------------------------------------------------------------------------
# Layer 2: rule-based LLM stub
# ---------------------------------------------------------------------------


def _generate_feedback(
    persona: str,
    violations: list[str],
    passed: bool,
    action: TrialAction,
    latent: TrialLatentState,
) -> tuple[str, str | None]:
    """Return (feedback, hint) for the given persona.

    This is a rule-based stub that can be replaced with a real LLM call later.
    The stub generates contextually appropriate strings without an LLM.
    """
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

    # Generic fallback
    return f"Hint: {violations[0]}"


# ---------------------------------------------------------------------------
# Main judge class
# ---------------------------------------------------------------------------


class TrialJudge:
    """Multi-layer trial design verifier.

    Layer 1 is programmatic and authoritative — its result is never overridden.
    Layer 2 is persona-scaled and provides human-readable feedback and hints.
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
            JudgeResult with pass/fail, violations, feedback, hint, and penalty.
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

        # 1b. Statistical power check
        n = max(latent.patients_enrolled, 1)
        power = calculate_power(latent.true_effect_size, n)
        if power < 0.80:
            violations.append(
                f"Insufficient statistical power: {power:.3f} < 0.80 "
                f"(effect_size={latent.true_effect_size:.3f}, n={n})."
            )

        # 1c. p-value check — derive from power/effect/n
        #     We use the same normal approximation as the simulator.
        import math

        from scipy.stats import norm

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
        # A "high-confidence wrong claim" is when the agent's confidence is
        # >= 0.8 but Layer 1 found violations (the claim is incorrect).
        overconfidence_penalty = 0.0
        if not passed and action.confidence >= _HIGH_CONFIDENCE_THRESHOLD:
            # One penalty per violation that was caused by a wrong claim
            overconfidence_penalty = _OVERCONFIDENCE_PENALTY * len(violations)

        # ------------------------------------------------------------------
        # Layer 2: Persona-scaled feedback (never overrides Layer 1 result)
        # ------------------------------------------------------------------
        persona = _select_persona(state.difficulty)
        feedback, hint = _generate_feedback(persona, violations, passed, action, latent)

        return JudgeResult(
            passed=passed,
            violations=violations,
            feedback=feedback,
            hint=hint,
            overconfidence_penalty=overconfidence_penalty,
            persona=persona,
        )
