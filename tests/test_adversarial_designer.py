"""
Tests for server/curriculum/adversarial_designer.py

Verifies:
  - analyze_failures correctly counts weak spots
  - generate_scenario produces a valid ScenarioConfig with compound challenges
  - get_weak_spots returns the current failure summary
  - Solvability validation rejects impossible scenarios
  - Expert-tier activation threshold constant is correct
"""

import pytest

from server.curriculum.adversarial_designer import (
    EXPERT_DIFFICULTY_THRESHOLD,
    AdversarialDesigner,
    _HIGH_DROPOUT_RANGE,
    _SMALL_EFFECT_RANGE,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_episode(
    success: bool,
    effect: float | None = None,
    dropout: float | None = None,
    used_biomarker: bool = True,
) -> dict:
    ep: dict = {"success": success}
    if effect is not None:
        ep["true_effect_size"] = effect
    if dropout is not None:
        ep["dropout_rate"] = dropout
    ep["used_biomarker_stratification"] = used_biomarker
    return ep


# ── Expert threshold ──────────────────────────────────────────────────────────


def test_expert_difficulty_threshold():
    assert EXPERT_DIFFICULTY_THRESHOLD == 0.80


# ── analyze_failures ──────────────────────────────────────────────────────────


def test_analyze_failures_empty_history():
    d = AdversarialDesigner()
    result = d.analyze_failures([])
    assert result["total_episodes"] == 0
    assert result["small_effect_failures"] == 0
    assert result["high_dropout_failures"] == 0
    assert result["hidden_subgroup_failures"] == 0


def test_analyze_failures_only_successes():
    """Successful episodes should not increment any failure counter."""
    history = [
        _make_episode(True, effect=0.10, dropout=0.30, used_biomarker=False),
        _make_episode(True, effect=0.15, dropout=0.28, used_biomarker=False),
    ]
    d = AdversarialDesigner()
    result = d.analyze_failures(history)
    assert result["total_episodes"] == 2
    assert result["small_effect_failures"] == 0
    assert result["high_dropout_failures"] == 0
    assert result["hidden_subgroup_failures"] == 0


def test_analyze_failures_small_effect():
    """Failed episodes with effect ≤ 0.25 increment small_effect_failures."""
    history = [
        _make_episode(False, effect=0.10),
        _make_episode(False, effect=0.25),  # boundary — still counts
        _make_episode(False, effect=0.30),  # above threshold — does not count
    ]
    d = AdversarialDesigner()
    result = d.analyze_failures(history)
    assert result["small_effect_failures"] == 2


def test_analyze_failures_high_dropout():
    """Failed episodes with dropout ≥ 0.20 increment high_dropout_failures."""
    history = [
        _make_episode(False, dropout=0.20),  # boundary — counts
        _make_episode(False, dropout=0.30),
        _make_episode(False, dropout=0.15),  # below threshold
    ]
    d = AdversarialDesigner()
    result = d.analyze_failures(history)
    assert result["high_dropout_failures"] == 2


def test_analyze_failures_hidden_subgroup():
    """Failed episodes without biomarker stratification increment hidden_subgroup_failures."""
    history = [
        _make_episode(False, used_biomarker=False),
        _make_episode(False, used_biomarker=False),
        _make_episode(False, used_biomarker=True),  # used biomarker — does not count
    ]
    d = AdversarialDesigner()
    result = d.analyze_failures(history)
    assert result["hidden_subgroup_failures"] == 2


def test_analyze_failures_resets_on_each_call():
    """Calling analyze_failures twice should reset counters, not accumulate."""
    d = AdversarialDesigner()
    history = [_make_episode(False, effect=0.10)]
    d.analyze_failures(history)
    result = d.analyze_failures(history)  # second call with same history
    assert result["small_effect_failures"] == 1  # not 2


def test_analyze_failures_missing_optional_fields():
    """Episodes without optional fields should not crash."""
    history = [{"success": False}]
    d = AdversarialDesigner()
    result = d.analyze_failures(history)
    assert result["total_episodes"] == 1
    # No optional fields → only hidden_subgroup counted (used_biomarker defaults False)
    assert result["hidden_subgroup_failures"] == 1
    assert result["small_effect_failures"] == 0
    assert result["high_dropout_failures"] == 0


# ── generate_scenario ─────────────────────────────────────────────────────────


def test_generate_scenario_returns_scenario_config():
    from models import ScenarioConfig

    d = AdversarialDesigner()
    weak_spots = d.get_weak_spots()
    scenario = d.generate_scenario(weak_spots)
    assert isinstance(scenario, ScenarioConfig)


def test_generate_scenario_compound_challenges():
    """Generated scenario must have small effect, high dropout, expert tier."""
    d = AdversarialDesigner()
    scenario = d.generate_scenario(d.get_weak_spots())

    # Small effect size
    assert scenario.effect_size_range == _SMALL_EFFECT_RANGE

    # High dropout
    assert scenario.dropout_rate_range == _HIGH_DROPOUT_RANGE

    # Expert tier
    assert scenario.curriculum_tier == 4


def test_generate_scenario_description_mentions_brca1():
    d = AdversarialDesigner()
    scenario = d.generate_scenario(d.get_weak_spots())
    assert "BRCA1" in scenario.description


def test_generate_scenario_solvable():
    """Generated scenario must pass solvability validation (no ValueError)."""
    d = AdversarialDesigner()
    # Should not raise
    scenario = d.generate_scenario(d.get_weak_spots())
    assert scenario.time_budget_days >= 12  # at least 12 days for 12 phases


def test_generate_scenario_dominant_weak_spot_in_description():
    """Description should mention the dominant weak spot."""
    d = AdversarialDesigner()
    history = [_make_episode(False, effect=0.10)] * 5  # many small-effect failures
    d.analyze_failures(history)
    scenario = d.generate_scenario(d.get_weak_spots())
    assert "small_effect" in scenario.description


# ── get_weak_spots ────────────────────────────────────────────────────────────


def test_get_weak_spots_initial_state():
    d = AdversarialDesigner()
    ws = d.get_weak_spots()
    assert ws == {
        "small_effect_failures": 0,
        "high_dropout_failures": 0,
        "hidden_subgroup_failures": 0,
        "total_episodes": 0,
    }


def test_get_weak_spots_returns_copy():
    """Mutating the returned dict should not affect internal state."""
    d = AdversarialDesigner()
    ws = d.get_weak_spots()
    ws["small_effect_failures"] = 999
    assert d.get_weak_spots()["small_effect_failures"] == 0


# ── Solvability validation ────────────────────────────────────────────────────


def test_validate_solvable_raises_for_tiny_budget():
    """A scenario with time_budget_days < _MIN_STEPS_REQUIRED should raise."""
    from server.curriculum.adversarial_designer import AdversarialDesigner, _MIN_STEPS_REQUIRED
    from models import ScenarioConfig

    tiny_scenario = ScenarioConfig(
        scenario_id="too_small",
        curriculum_tier=4,
        disease_area="oncology",
        effect_size_range=(0.10, 0.20),
        side_effect_rate_range=(0.15, 0.30),
        placebo_response_range=(0.10, 0.20),
        dropout_rate_range=(0.25, 0.35),
        budget_usd=1_000.0,
        time_budget_days=_MIN_STEPS_REQUIRED - 1,  # one day too few
        min_sample_size=10,
        description="too small",
    )
    with pytest.raises(ValueError, match="not solvable"):
        AdversarialDesigner._validate_solvable(tiny_scenario)


# ── Integration: full adversarial workflow ────────────────────────────────────


def test_full_adversarial_workflow():
    """Simulate the full expert-tier adversarial workflow."""
    difficulty = 0.85  # above expert threshold
    assert difficulty > EXPERT_DIFFICULTY_THRESHOLD

    history = [
        _make_episode(False, effect=0.12, dropout=0.28, used_biomarker=False),
        _make_episode(False, effect=0.18, dropout=0.31, used_biomarker=False),
        _make_episode(True, effect=0.50, dropout=0.10, used_biomarker=True),
        _make_episode(False, effect=0.15, dropout=0.26, used_biomarker=False),
    ]

    designer = AdversarialDesigner()
    designer.analyze_failures(history)
    weak_spots = designer.get_weak_spots()

    assert weak_spots["total_episodes"] == 4
    assert weak_spots["small_effect_failures"] == 3
    assert weak_spots["high_dropout_failures"] == 3
    assert weak_spots["hidden_subgroup_failures"] == 3

    scenario = designer.generate_scenario(weak_spots)
    assert scenario.scenario_id == "adversarial_expert"
    assert scenario.curriculum_tier == 4
    assert scenario.effect_size_range == _SMALL_EFFECT_RANGE
    assert scenario.dropout_rate_range == _HIGH_DROPOUT_RANGE
