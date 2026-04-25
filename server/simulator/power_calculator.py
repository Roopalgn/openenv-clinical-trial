"""
Power calculator using scipy.stats.norm.

Pure function — no side effects. Caching is handled by EpisodeManager.
"""

from __future__ import annotations

import math

from scipy.stats import norm


def calculate_power(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """Calculate statistical power for a two-sample t-test.

    Uses the normal approximation: power = P(Z > z_alpha - delta) where
    delta = effect_size * sqrt(n_per_arm).

    Args:
        effect_size: Cohen's d (standardised effect size).
        n: Total sample size (both arms combined).
        alpha: Type I error rate (default 0.05, two-tailed).

    Returns:
        Power as a float in [0, 1].
    """
    if n <= 0:
        return 0.0
    if effect_size == 0.0:
        return alpha  # power equals alpha when there is no effect

    # Two-tailed critical value
    z_alpha = norm.ppf(1.0 - alpha / 2.0)

    # Non-centrality parameter for two-sample test with equal group sizes
    n_per_arm = n / 2.0
    ncp = abs(effect_size) * math.sqrt(n_per_arm)

    # Power = P(reject H0 | H1 true)
    power = norm.sf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    return float(min(max(power, 0.0), 1.0))
