"""
Curriculum controller for the Clinical Trial Designer environment.

Exposes:
  - advance_curriculum(tier, metrics) -> int
  - select_scenario(tier, rng) -> ScenarioConfig

5-tier mastery logic:
  Tier 0: warmup
  Tier 1: beginner
  Tier 2: intermediate
  Tier 3: advanced
  Tier 4: expert

Graduation rules:
  - 70% rolling success rate over recent episodes → advance one tier
  - 90% success rate after at least 3 episodes → fast-track (skip one tier)
  - Max tier is 4 (expert); cannot advance beyond.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from models import ScenarioConfig
from server.curriculum.scenarios import (
    AUTOIMMUNE_BIOLOGIC,
    CNS_DEPRESSION,
    RARE_DISEASE_ORPHAN,
    SOLID_TUMOR_CHEMO,
    WARMUP,
)

# ── Constants ────────────────────────────────────────────────────────────────

MIN_TIER: int = 0
MAX_TIER: int = 4

MASTERY_THRESHOLD: float = 0.70  # 70% rolling success → graduate
FAST_TRACK_THRESHOLD: float = 0.90  # 90% success after ≥3 episodes → skip tier
FAST_TRACK_MIN_EPISODES: int = 3

# Rolling window size for success-rate calculation
ROLLING_WINDOW: int = 10

# Tier → ScenarioConfig mapping (one canonical scenario per tier)
_TIER_SCENARIO: dict[int, ScenarioConfig] = {
    0: WARMUP,
    1: SOLID_TUMOR_CHEMO,
    2: AUTOIMMUNE_BIOLOGIC,
    3: CNS_DEPRESSION,
    4: RARE_DISEASE_ORPHAN,
}

TIER_NAMES: dict[int, str] = {
    0: "warmup",
    1: "beginner",
    2: "intermediate",
    3: "advanced",
    4: "expert",
}


# ── EpisodeMetrics ────────────────────────────────────────────────────────────


@dataclass
class EpisodeMetrics:
    """Performance metrics for a completed episode.

    Attributes:
        success: Whether the episode ended in a successful trial outcome.
        episode_history: Rolling list of recent success booleans (most recent
            episode appended last).  The controller uses the last
            ``ROLLING_WINDOW`` entries to compute the rolling success rate.
            Callers should append the current episode's ``success`` value
            *before* passing this object to ``advance_curriculum``.
    """

    success: bool
    episode_history: list[bool] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────


def advance_curriculum(tier: int, metrics: EpisodeMetrics) -> int:
    """Return the updated curriculum tier after evaluating episode metrics.

    Args:
        tier: Current curriculum tier (0–4).
        metrics: Performance metrics for the just-completed episode.
            ``metrics.episode_history`` must already include the current
            episode's success value as its last element.

    Returns:
        The new curriculum tier.  May be the same tier (not yet mastered),
        ``tier + 1`` (normal graduation), or ``tier + 2`` (fast-track skip).
        Never exceeds ``MAX_TIER``.
    """
    if tier >= MAX_TIER:
        return MAX_TIER

    history: Sequence[bool] = metrics.episode_history
    n_episodes = len(history)

    if n_episodes == 0:
        return tier

    # Use the most recent ROLLING_WINDOW episodes for the rolling rate
    window = list(history[-ROLLING_WINDOW:])
    rolling_rate = sum(window) / len(window)

    # Fast-track: 90%+ success after at least 3 episodes → skip one tier
    if n_episodes >= FAST_TRACK_MIN_EPISODES and rolling_rate >= FAST_TRACK_THRESHOLD:
        new_tier = min(tier + 2, MAX_TIER)
        return new_tier

    # Normal graduation: 70%+ rolling success → advance one tier
    if rolling_rate >= MASTERY_THRESHOLD:
        return min(tier + 1, MAX_TIER)

    return tier


def select_scenario(tier: int, rng: np.random.Generator) -> ScenarioConfig:
    """Select a ScenarioConfig appropriate for the given curriculum tier.

    At tier 0 (warmup) the solid_tumor_chemo scenario is returned with an
    inflated effect size (already encoded in the WARMUP ScenarioConfig).

    Args:
        tier: Current curriculum tier (0–4).  Values outside [0, 4] are
            clamped to the valid range.
        rng: A seeded ``numpy.random.Generator`` used for any stochastic
            selection.  Currently each tier maps to exactly one scenario, so
            ``rng`` is accepted for API consistency and future extensibility
            (e.g. sampling from a pool of scenarios at the same tier).

    Returns:
        The ``ScenarioConfig`` for the given tier.
    """
    clamped_tier = max(MIN_TIER, min(tier, MAX_TIER))
    return _TIER_SCENARIO[clamped_tier]
