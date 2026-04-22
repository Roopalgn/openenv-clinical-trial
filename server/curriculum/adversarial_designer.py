"""
AdversarialDesigner — Expert-tier scenario generation for the curriculum.

Activates when difficulty > 0.80 (expert tier). Analyses episode history to
identify failure patterns (weak spots) and generates targeted ScenarioConfig
instances with compound challenges:
  - small effect size  (0.10–0.20)
  - hidden subgroup    (e.g. BRCA1+ responders)
  - high dropout       (0.25–0.35)

All generated scenarios are validated as solvable within the step budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import ScenarioConfig

# ── Constants ─────────────────────────────────────────────────────────────────

EXPERT_DIFFICULTY_THRESHOLD: float = 0.80

# Compound-challenge parameter ranges
_SMALL_EFFECT_RANGE: tuple[float, float] = (0.10, 0.20)
_HIGH_DROPOUT_RANGE: tuple[float, float] = (0.25, 0.35)
_HIDDEN_SUBGROUP_LABEL: str = "BRCA1+"

# Minimum steps required to complete all mandatory trial phases:
#   phase-I safety, dose-escalation, interim analysis, primary analysis,
#   FDA submission, conclusion synthesis  → 6 phases × 2 actions each = 12
_MIN_STEPS_REQUIRED: int = 12

# Default step budget for adversarial scenarios
_DEFAULT_MAX_STEPS: int = 30


# ── Failure-pattern keys ──────────────────────────────────────────────────────

_KEY_SMALL_EFFECT = "small_effect_failures"
_KEY_HIGH_DROPOUT = "high_dropout_failures"
_KEY_HIDDEN_SUBGROUP = "hidden_subgroup_failures"
_KEY_TOTAL = "total_episodes"


# ── AdversarialDesigner ───────────────────────────────────────────────────────


@dataclass
class AdversarialDesigner:
    """Generates adversarial ScenarioConfigs targeting an agent's weak spots.

    Usage::

        designer = AdversarialDesigner()
        if difficulty > EXPERT_DIFFICULTY_THRESHOLD:
            designer.analyze_failures(episode_history)
            weak_spots = designer.get_weak_spots()
            scenario = designer.generate_scenario(weak_spots)
    """

    # Internal accumulator for failure patterns
    _weak_spots: dict[str, int] = field(
        default_factory=lambda: {
            _KEY_SMALL_EFFECT: 0,
            _KEY_HIGH_DROPOUT: 0,
            _KEY_HIDDEN_SUBGROUP: 0,
            _KEY_TOTAL: 0,
        }
    )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_failures(self, episode_history: list[dict]) -> dict:
        """Scan episode history and update internal weak-spot counters.

        Each entry in *episode_history* is expected to be a dict with at least:
          - ``success`` (bool)
          - ``scenario_id`` (str, optional)
          - ``true_effect_size`` (float, optional)
          - ``dropout_rate`` (float, optional)
          - ``used_biomarker_stratification`` (bool, optional)

        Args:
            episode_history: List of episode result dicts (most recent last).

        Returns:
            The updated weak-spot summary dict (same as ``get_weak_spots()``).
        """
        # Reset counters before re-scanning the full history
        self._weak_spots = {
            _KEY_SMALL_EFFECT: 0,
            _KEY_HIGH_DROPOUT: 0,
            _KEY_HIDDEN_SUBGROUP: 0,
            _KEY_TOTAL: 0,
        }

        for ep in episode_history:
            self._weak_spots[_KEY_TOTAL] += 1
            if ep.get("success", True):
                # Only count failures
                continue

            effect = ep.get("true_effect_size")
            if effect is not None and effect <= 0.25:
                self._weak_spots[_KEY_SMALL_EFFECT] += 1

            dropout = ep.get("dropout_rate")
            if dropout is not None and dropout >= 0.20:
                self._weak_spots[_KEY_HIGH_DROPOUT] += 1

            used_biomarker = ep.get("used_biomarker_stratification", False)
            if not used_biomarker:
                self._weak_spots[_KEY_HIDDEN_SUBGROUP] += 1

        return self.get_weak_spots()

    def generate_scenario(self, weak_spots: dict) -> ScenarioConfig:
        """Create a targeted ScenarioConfig with compound challenges.

        The scenario always combines:
          - small effect size (0.10–0.20)
          - hidden subgroup (BRCA1+ responders)
          - high dropout (0.25–0.35)

        The description is tailored to emphasise the most prominent weak spot.

        Args:
            weak_spots: Failure-pattern summary (from ``get_weak_spots()``).

        Returns:
            A validated ``ScenarioConfig`` solvable within the step budget.

        Raises:
            ValueError: If the generated scenario cannot be solved within the
                step budget (should never happen with the current defaults).
        """
        # Determine the dominant weak spot for the description
        dominant = self._dominant_weak_spot(weak_spots)

        description = (
            f"Adversarial expert scenario targeting '{dominant}'. "
            "Compound challenges: small effect size (0.10–0.20), "
            "hidden BRCA1+ responder subgroup, and high dropout (0.25–0.35). "
            "Agent must identify the biomarker subgroup and power the study "
            "despite high attrition and a subtle treatment signal."
        )

        scenario = ScenarioConfig(
            scenario_id="adversarial_expert",
            curriculum_tier=4,
            disease_area="oncology",
            effect_size_range=_SMALL_EFFECT_RANGE,
            side_effect_rate_range=(0.15, 0.30),
            placebo_response_range=(0.10, 0.20),
            dropout_rate_range=_HIGH_DROPOUT_RANGE,
            budget_usd=25_000_000.0,
            time_budget_days=1_080,
            min_sample_size=200,
            description=description,
        )

        self._validate_solvable(scenario)
        return scenario

    def get_weak_spots(self) -> dict:
        """Return the current failure-pattern summary.

        Returns:
            Dict with keys:
              - ``small_effect_failures`` (int)
              - ``high_dropout_failures`` (int)
              - ``hidden_subgroup_failures`` (int)
              - ``total_episodes`` (int)
        """
        return dict(self._weak_spots)

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _dominant_weak_spot(weak_spots: dict) -> str:
        """Return the name of the most frequent failure pattern."""
        candidates = {
            "small_effect": weak_spots.get(_KEY_SMALL_EFFECT, 0),
            "high_dropout": weak_spots.get(_KEY_HIGH_DROPOUT, 0),
            "hidden_subgroup": weak_spots.get(_KEY_HIDDEN_SUBGROUP, 0),
        }
        return max(candidates, key=lambda k: candidates[k])

    @staticmethod
    def _validate_solvable(scenario: ScenarioConfig) -> None:
        """Assert the scenario can be solved within the step budget.

        Solvability criterion: the time budget must accommodate at least
        ``_MIN_STEPS_REQUIRED`` distinct trial phases.  We use
        ``time_budget_days`` as a proxy — each phase takes at most
        ``time_budget_days / _MIN_STEPS_REQUIRED`` days.

        Args:
            scenario: The scenario to validate.

        Raises:
            ValueError: If the scenario is not solvable within the step budget.
        """
        days_per_step = scenario.time_budget_days / _MIN_STEPS_REQUIRED
        if days_per_step < 1:
            raise ValueError(
                f"Scenario '{scenario.scenario_id}' is not solvable within the "
                f"step budget: time_budget_days={scenario.time_budget_days} is "
                f"too small for {_MIN_STEPS_REQUIRED} required phases."
            )
