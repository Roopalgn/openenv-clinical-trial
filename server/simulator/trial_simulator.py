"""
Trial simulator — produces a TrialResult from TrialLatentState and TrialAction.

Handles edge cases:
  - budget_remaining <= 0  → failure_reason="budget_exhausted"
  - time_remaining_days <= 0 → failure_reason="time_exhausted"
  - patients_enrolled == 0 at terminal → failure_reason="no_enrollment"
"""

from __future__ import annotations

import math
import random

from models import TrialAction, TrialLatentState, TrialResult
from server.simulator.power_calculator import calculate_power


def simulate_trial(
    latent: TrialLatentState,
    action: TrialAction,
) -> TrialResult:
    """Simulate a trial step and return a TrialResult.

    Uses the latent ground-truth parameters to compute realistic outcomes.
    Deterministic given the same inputs (uses latent.seed for RNG).

    Args:
        latent: Hidden ground-truth + episode tracking state.
        action: The action taken by the agent.

    Returns:
        A TrialResult reflecting the simulated outcome.
    """
    # --- Edge-case checks (requirements 15.2, 15.3, 15.4) ---
    if latent.budget_remaining <= 0:
        return TrialResult(
            p_value=1.0,
            success=False,
            power=0.0,
            adverse_event_rate=latent.true_side_effect_rate,
            confidence_interval=(0.0, 0.0),
            failure_reason="budget_exhausted",
        )

    if latent.time_remaining_days <= 0:
        return TrialResult(
            p_value=1.0,
            success=False,
            power=0.0,
            adverse_event_rate=latent.true_side_effect_rate,
            confidence_interval=(0.0, 0.0),
            failure_reason="time_exhausted",
        )

    if latent.patients_enrolled == 0 and latent.trial_complete:
        return TrialResult(
            p_value=1.0,
            success=False,
            power=0.0,
            adverse_event_rate=latent.true_side_effect_rate,
            confidence_interval=(0.0, 0.0),
            failure_reason="no_enrollment",
        )

    # --- Normal simulation path ---
    step_index = len(latent.action_history)
    rng = random.Random(latent.seed ^ step_index)

    effect_size = latent.true_effect_size
    n = max(latent.patients_enrolled, 1)
    alpha = 0.05

    power = calculate_power(effect_size, n, alpha)

    noise = rng.gauss(
        0.0,
        latent.measurement_noise if latent.measurement_noise > 0 else 0.05,
    )
    observed_effect = effect_size + noise

    n_per_arm = n / 2.0
    if n_per_arm > 0 and effect_size != 0.0:
        se = 1.0 / math.sqrt(n_per_arm)
        z_stat = observed_effect / se if se > 0 else 0.0
        from scipy.stats import norm
        p_value = float(2.0 * norm.sf(abs(z_stat)))
    else:
        p_value = 1.0

    p_value = min(max(p_value, 0.0), 1.0)
    success = p_value < alpha

    if n_per_arm > 0:
        from scipy.stats import norm as _norm
        z_95 = _norm.ppf(0.975)
        se = 1.0 / math.sqrt(n_per_arm)
        ci_low = observed_effect - z_95 * se
        ci_high = observed_effect + z_95 * se
    else:
        ci_low, ci_high = 0.0, 0.0

    ae_noise = rng.gauss(
        0.0,
        latent.site_variability if latent.site_variability > 0 else 0.01,
    )
    adverse_event_rate = min(
        max(latent.true_side_effect_rate + ae_noise, 0.0), 1.0
    )

    return TrialResult(
        p_value=p_value,
        success=success,
        power=power,
        adverse_event_rate=adverse_event_rate,
        confidence_interval=(ci_low, ci_high),
        failure_reason=None,
    )
