"""
Tests for NoiseModel — seeded domain randomization (Task 12).

Requirements 9.1, 9.2:
  - NoiseModel produces a randomized ScenarioConfig from a seed
  - Same seed → identical output (idempotence)
"""

from __future__ import annotations

import pytest

from models import ScenarioConfig
from server.noise_model import NoiseModel


@pytest.fixture()
def base_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="solid_tumor_chemo",
        curriculum_tier=0,
        disease_area="NSCLC",
        effect_size_range=(0.3, 0.7),
        side_effect_rate_range=(0.05, 0.20),
        placebo_response_range=(0.10, 0.25),
        dropout_rate_range=(0.05, 0.15),
        budget_usd=1_000_000.0,
        time_budget_days=365,
        min_sample_size=100,
        description="Test scenario",
    )


class TestNoiseModelIdempotence:
    """Requirement 9.2: same seed → same randomized config."""

    def test_same_seed_same_budget(self, base_scenario: ScenarioConfig) -> None:
        r1 = NoiseModel(seed=42).randomize(base_scenario)
        r2 = NoiseModel(seed=42).randomize(base_scenario)
        assert r1.budget_usd == r2.budget_usd

    def test_same_seed_same_time(self, base_scenario: ScenarioConfig) -> None:
        r1 = NoiseModel(seed=42).randomize(base_scenario)
        r2 = NoiseModel(seed=42).randomize(base_scenario)
        assert r1.time_budget_days == r2.time_budget_days

    def test_same_seed_same_dropout_range(
        self, base_scenario: ScenarioConfig
    ) -> None:
        r1 = NoiseModel(seed=42).randomize(base_scenario)
        r2 = NoiseModel(seed=42).randomize(base_scenario)
        assert r1.dropout_rate_range == r2.dropout_rate_range

    def test_same_seed_same_placebo_range(
        self, base_scenario: ScenarioConfig
    ) -> None:
        r1 = NoiseModel(seed=42).randomize(base_scenario)
        r2 = NoiseModel(seed=42).randomize(base_scenario)
        assert r1.placebo_response_range == r2.placebo_response_range

    def test_different_seeds_different_budget(
        self, base_scenario: ScenarioConfig
    ) -> None:
        r1 = NoiseModel(seed=1).randomize(base_scenario)
        r2 = NoiseModel(seed=2).randomize(base_scenario)
        assert r1.budget_usd != r2.budget_usd


class TestNoiseModelRanges:
    """Domain randomization stays within specified bounds."""

    def test_budget_within_30_percent(self, base_scenario: ScenarioConfig) -> None:
        for seed in range(50):
            result = NoiseModel(seed=seed).randomize(base_scenario)
            lo = base_scenario.budget_usd * 0.70
            hi = base_scenario.budget_usd * 1.30
            assert lo <= result.budget_usd <= hi, (
                f"seed={seed}: budget {result.budget_usd} outside [{lo}, {hi}]"
            )

    def test_time_within_20_percent(self, base_scenario: ScenarioConfig) -> None:
        for seed in range(50):
            result = NoiseModel(seed=seed).randomize(base_scenario)
            lo = base_scenario.time_budget_days * 0.80
            hi = base_scenario.time_budget_days * 1.20
            assert lo <= result.time_budget_days <= hi, (
                f"seed={seed}: time {result.time_budget_days} outside [{lo}, {hi}]"
            )

    def test_dropout_range_valid(self, base_scenario: ScenarioConfig) -> None:
        for seed in range(50):
            result = NoiseModel(seed=seed).randomize(base_scenario)
            lo, hi = result.dropout_rate_range
            assert 0.0 <= lo <= hi <= 1.0, (
                f"seed={seed}: dropout range [{lo}, {hi}] invalid"
            )

    def test_placebo_range_valid(self, base_scenario: ScenarioConfig) -> None:
        for seed in range(50):
            result = NoiseModel(seed=seed).randomize(base_scenario)
            lo, hi = result.placebo_response_range
            assert 0.0 <= lo <= hi <= 1.0, (
                f"seed={seed}: placebo range [{lo}, {hi}] invalid"
            )

    def test_non_randomized_fields_unchanged(
        self, base_scenario: ScenarioConfig
    ) -> None:
        result = NoiseModel(seed=99).randomize(base_scenario)
        assert result.scenario_id == base_scenario.scenario_id
        assert result.curriculum_tier == base_scenario.curriculum_tier
        assert result.disease_area == base_scenario.disease_area
        assert result.effect_size_range == base_scenario.effect_size_range
        assert result.side_effect_rate_range == base_scenario.side_effect_rate_range
        assert result.min_sample_size == base_scenario.min_sample_size

    def test_time_budget_at_least_one_day(
        self, base_scenario: ScenarioConfig
    ) -> None:
        for seed in range(50):
            result = NoiseModel(seed=seed).randomize(base_scenario)
            assert result.time_budget_days >= 1


class TestEpisodeManagerNoiseIntegration:
    """NoiseModel is wired into EpisodeManager.reset() correctly."""

    def test_same_seed_same_budget_remaining(self) -> None:
        from server.episode_manager import EpisodeManager

        em1 = EpisodeManager()
        em1.reset(seed=7)

        em2 = EpisodeManager()
        em2.reset(seed=7)

        assert em1._latent.budget_remaining == em2._latent.budget_remaining

    def test_same_seed_same_time_remaining(self) -> None:
        from server.episode_manager import EpisodeManager

        em1 = EpisodeManager()
        em1.reset(seed=7)

        em2 = EpisodeManager()
        em2.reset(seed=7)

        assert em1._latent.time_remaining_days == em2._latent.time_remaining_days

    def test_same_seed_same_latent_effect_size(self) -> None:
        from server.episode_manager import EpisodeManager

        em1 = EpisodeManager()
        em1.reset(seed=123)

        em2 = EpisodeManager()
        em2.reset(seed=123)

        assert em1._latent.true_effect_size == em2._latent.true_effect_size
        assert em1._latent.placebo_response_rate == em2._latent.placebo_response_rate

    def test_no_seed_still_initializes(self) -> None:
        from server.episode_manager import EpisodeManager

        em = EpisodeManager()
        em.reset()  # no seed — should use random seed
        assert em._latent.seed >= 0
        assert em._latent.budget_remaining > 0
