"""
NoiseModel — seeded domain randomization for ScenarioConfig parameters.

Applies uniform noise to scenario parameters to prevent the agent from
overfitting to fixed scenario values. Uses a seeded numpy.Generator so
the same seed always produces the same randomized config (reproducibility).

Domain randomization ranges:
  - budget_usd:        ±30%
  - time_budget_days:  ±20%
  - dropout_rate_range: ±15%
  - placebo_response_range: ±20%
"""

from __future__ import annotations

import numpy as np

from models import ScenarioConfig


class NoiseModel:
    """Applies seeded domain randomization to a ScenarioConfig.

    Args:
        seed: Integer seed for the numpy Generator. Same seed → same output.
    """

    # Randomization magnitudes (fraction of original value)
    _BUDGET_NOISE: float = 0.30
    _TIME_NOISE: float = 0.20
    _DROPOUT_NOISE: float = 0.15
    _PLACEBO_NOISE: float = 0.20

    def __init__(self, seed: int) -> None:
        self._seed = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def randomize(self, config: ScenarioConfig) -> ScenarioConfig:
        """Return a new ScenarioConfig with domain-randomized parameters.

        The original config is not mutated. The returned config has the same
        scenario_id, curriculum_tier, disease_area, and other non-randomized
        fields as the input.

        Args:
            config: The base ScenarioConfig to randomize.

        Returns:
            A new ScenarioConfig with perturbed budget, time, dropout, and
            placebo response parameters.
        """
        # budget ±30%
        budget_factor = 1.0 + self._rng.uniform(-self._BUDGET_NOISE, self._BUDGET_NOISE)
        new_budget = max(0.0, config.budget_usd * budget_factor)

        # time ±20%
        time_factor = 1.0 + self._rng.uniform(-self._TIME_NOISE, self._TIME_NOISE)
        new_time = max(1, round(config.time_budget_days * time_factor))

        # dropout ±15% — perturb both ends of the range symmetrically
        dropout_lo, dropout_hi = config.dropout_rate_range
        dropout_factor = 1.0 + self._rng.uniform(
            -self._DROPOUT_NOISE, self._DROPOUT_NOISE
        )
        new_dropout_lo = float(np.clip(dropout_lo * dropout_factor, 0.0, 1.0))
        new_dropout_hi = float(np.clip(dropout_hi * dropout_factor, 0.0, 1.0))
        # Ensure lo ≤ hi after clipping
        if new_dropout_lo > new_dropout_hi:
            new_dropout_lo, new_dropout_hi = new_dropout_hi, new_dropout_lo

        # placebo ±20% — perturb both ends of the range symmetrically
        placebo_lo, placebo_hi = config.placebo_response_range
        placebo_factor = 1.0 + self._rng.uniform(
            -self._PLACEBO_NOISE, self._PLACEBO_NOISE
        )
        new_placebo_lo = float(np.clip(placebo_lo * placebo_factor, 0.0, 1.0))
        new_placebo_hi = float(np.clip(placebo_hi * placebo_factor, 0.0, 1.0))
        if new_placebo_lo > new_placebo_hi:
            new_placebo_lo, new_placebo_hi = new_placebo_hi, new_placebo_lo

        return config.model_copy(
            update={
                "budget_usd": new_budget,
                "time_budget_days": new_time,
                "dropout_rate_range": (new_dropout_lo, new_dropout_hi),
                "placebo_response_range": (new_placebo_lo, new_placebo_hi),
            }
        )
