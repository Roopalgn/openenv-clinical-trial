"""
Potential-based reward shaping.

φ(s) = milestone_completion × budget_efficiency
Shaping bonus = γ · (φ(s') − φ(s))
"""

from __future__ import annotations

from models import TrialLatentState

GAMMA: float = 0.99  # discount factor for potential-based shaping


def _milestone_completion(latent: TrialLatentState) -> float:
    """Fraction of key milestones completed (0.0 – 1.0)."""
    milestones = [
        latent.phase_i_complete,
        latent.interim_complete,
        latent.protocol_submitted,
        latent.primary_analysis_complete,
        latent.trial_complete,
    ]
    return sum(1.0 for m in milestones if m) / len(milestones)


def _budget_efficiency(
    latent: TrialLatentState, initial_budget: float = 1_000_000.0
) -> float:
    """Fraction of budget remaining (0.0 – 1.0)."""
    if initial_budget <= 0:
        return 0.0
    return min(max(latent.budget_remaining / initial_budget, 0.0), 1.0)


def potential(latent: TrialLatentState, initial_budget: float = 1_000_000.0) -> float:
    """φ(s) = milestone_completion × budget_efficiency."""
    return _milestone_completion(latent) * _budget_efficiency(latent, initial_budget)


def shaping_bonus(
    latent: TrialLatentState,
    next_latent: TrialLatentState,
    initial_budget: float = 1_000_000.0,
    gamma: float = GAMMA,
) -> float:
    """Return γ · (φ(s') − φ(s)).

    Args:
        latent: Hidden state before the action (s).
        next_latent: Hidden state after the action (s').
        initial_budget: Starting budget used to normalise budget_efficiency.
        gamma: Discount factor (default 0.99).

    Returns:
        Scalar shaping bonus.
    """
    phi_s = potential(latent, initial_budget)
    phi_s_prime = potential(next_latent, initial_budget)
    return gamma * (phi_s_prime - phi_s)
