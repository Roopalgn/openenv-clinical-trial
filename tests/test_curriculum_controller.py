"""
Tests for server/curriculum/controller.py

Verifies:
  - advance_curriculum mastery logic (70% → graduate, 90% → fast-track)
  - select_scenario tier mapping
  - Edge cases (empty history, max tier, clamping)
"""

import numpy as np

from server.curriculum.controller import (
    MAX_TIER,
    EpisodeMetrics,
    advance_curriculum,
    select_scenario,
)
from server.curriculum.scenarios import (
    AUTOIMMUNE_BIOLOGIC,
    CNS_DEPRESSION,
    RARE_DISEASE_ORPHAN,
    SOLID_TUMOR_CHEMO,
    WARMUP,
)

# ── advance_curriculum tests ──────────────────────────────────────────────────


def test_advance_curriculum_empty_history():
    """Empty history → stay at current tier."""
    metrics = EpisodeMetrics(success=True, episode_history=[])
    assert advance_curriculum(0, metrics) == 0
    assert advance_curriculum(2, metrics) == 2


def test_advance_curriculum_no_mastery():
    """Below 70% success → stay at current tier."""
    # 6/10 = 60% → no graduation
    history = [True, False, True, False, True, False, True, False, True, False]
    metrics = EpisodeMetrics(success=False, episode_history=history)
    assert advance_curriculum(1, metrics) == 1


def test_advance_curriculum_normal_graduation():
    """70%+ rolling success → advance one tier."""
    # 7/10 = 70% → graduate
    history = [True, True, True, True, True, True, True, False, False, False]
    metrics = EpisodeMetrics(success=False, episode_history=history)
    assert advance_curriculum(0, metrics) == 1
    assert advance_curriculum(2, metrics) == 3


def test_advance_curriculum_fast_track():
    """90%+ success after ≥3 episodes → skip one tier (advance by 2)."""
    # 9/10 = 90% → fast-track
    history = [True, True, True, True, True, True, True, True, True, False]
    metrics = EpisodeMetrics(success=False, episode_history=history)
    assert advance_curriculum(0, metrics) == 2  # skip tier 1
    assert advance_curriculum(1, metrics) == 3  # skip tier 2


def test_advance_curriculum_fast_track_requires_min_episodes():
    """Fast-track requires at least 3 episodes."""
    # 2 episodes, 100% success → not enough for fast-track
    history = [True, True]
    metrics = EpisodeMetrics(success=True, episode_history=history)
    # Should not fast-track (only 2 episodes), but 100% ≥ 70% → normal graduate
    assert advance_curriculum(0, metrics) == 1

    # 3 episodes, 100% success → fast-track
    history = [True, True, True]
    metrics = EpisodeMetrics(success=True, episode_history=history)
    assert advance_curriculum(0, metrics) == 2


def test_advance_curriculum_max_tier_clamp():
    """Cannot advance beyond MAX_TIER (4)."""
    history = [True] * 10  # 100% success
    metrics = EpisodeMetrics(success=True, episode_history=history)
    assert advance_curriculum(MAX_TIER, metrics) == MAX_TIER
    assert advance_curriculum(MAX_TIER - 1, metrics) == MAX_TIER  # fast-track clamped


def test_advance_curriculum_rolling_window():
    """Only the most recent 10 episodes count for rolling rate."""
    # 20 episodes: first 10 are all False, last 10 are 9 True + 1 False
    # Rolling window (last 10) = 9/10 = 90% → fast-track
    history = [False] * 10 + [True] * 9 + [False]
    metrics = EpisodeMetrics(success=False, episode_history=history)
    assert advance_curriculum(0, metrics) == 2


def test_advance_curriculum_exactly_70_percent():
    """Exactly 70% success → should graduate."""
    history = [True] * 7 + [False] * 3
    metrics = EpisodeMetrics(success=False, episode_history=history)
    assert advance_curriculum(1, metrics) == 2


def test_advance_curriculum_exactly_90_percent():
    """Exactly 90% success after ≥3 episodes → fast-track."""
    history = [True] * 9 + [False]
    metrics = EpisodeMetrics(success=False, episode_history=history)
    assert advance_curriculum(0, metrics) == 2


# ── select_scenario tests ─────────────────────────────────────────────────────


def test_select_scenario_tier_mapping():
    """Each tier maps to the correct ScenarioConfig."""
    rng = np.random.default_rng(42)
    assert select_scenario(0, rng) == WARMUP
    assert select_scenario(1, rng) == SOLID_TUMOR_CHEMO
    assert select_scenario(2, rng) == AUTOIMMUNE_BIOLOGIC
    assert select_scenario(3, rng) == CNS_DEPRESSION
    assert select_scenario(4, rng) == RARE_DISEASE_ORPHAN


def test_select_scenario_clamping():
    """Out-of-range tiers are clamped to [MIN_TIER, MAX_TIER]."""
    rng = np.random.default_rng(42)
    # Below MIN_TIER → clamp to 0
    assert select_scenario(-1, rng) == WARMUP
    assert select_scenario(-100, rng) == WARMUP
    # Above MAX_TIER → clamp to 4
    assert select_scenario(5, rng) == RARE_DISEASE_ORPHAN
    assert select_scenario(100, rng) == RARE_DISEASE_ORPHAN


def test_select_scenario_deterministic():
    """Same tier + rng seed → same scenario (currently deterministic anyway)."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert select_scenario(2, rng1) == select_scenario(2, rng2)


# ── Integration test: full curriculum progression ─────────────────────────────


def test_full_curriculum_progression():
    """Simulate a full curriculum progression from tier 0 → 4."""
    tier = 0
    history: list[bool] = []

    # Tier 0 → 1 (normal graduation at 70%)
    for _ in range(7):
        history.append(True)
    for _ in range(3):
        history.append(False)
    metrics = EpisodeMetrics(success=False, episode_history=history)
    tier = advance_curriculum(tier, metrics)
    assert tier == 1

    # Tier 1 → 3 (fast-track at 90%)
    history = [True] * 9 + [False]
    metrics = EpisodeMetrics(success=False, episode_history=history)
    tier = advance_curriculum(tier, metrics)
    assert tier == 3

    # Tier 3 → 4 (normal graduation)
    history = [True] * 7 + [False] * 3
    metrics = EpisodeMetrics(success=False, episode_history=history)
    tier = advance_curriculum(tier, metrics)
    assert tier == 4

    # Tier 4 → 4 (max tier, cannot advance)
    history = [True] * 10
    metrics = EpisodeMetrics(success=True, episode_history=history)
    tier = advance_curriculum(tier, metrics)
    assert tier == 4
