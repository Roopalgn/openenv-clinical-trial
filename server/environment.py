"""
OpenEnv Environment subclass wrapping EpisodeManager for direct (non-HTTP) training.

Bypasses HTTP entirely — same behavior as the FastAPI routes but without
network overhead. Used by train.py and eval_compare.py.
"""

from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment as BaseEnvironment

from models import TrialAction, TrialObservation, TrialState
from server.episode_manager import EpisodeManager


class Environment(BaseEnvironment[TrialAction, TrialObservation, TrialState]):
    """Clinical Trial Designer environment for direct (non-HTTP) training use.

    Wraps EpisodeManager and conforms to the OpenEnv Environment contract.
    reset() and step() call EpisodeManager directly, bypassing HTTP entirely.
    Behavior is identical to the /reset and /step FastAPI routes.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._manager = EpisodeManager()
        self._last_obs: TrialObservation | None = None

    # ------------------------------------------------------------------
    # OpenEnv abstract interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TrialObservation:
        """Initialize a new episode and return the initial observation.

        Args:
            seed: Optional random seed for reproducibility.
            episode_id: Ignored — EpisodeManager generates its own UUID.

        Returns:
            Initial TrialObservation for the new episode.
        """
        self._reset_rubric()
        obs = self._manager.reset(seed=seed)
        self._last_obs = obs
        return self._apply_transform(obs)

    def step(
        self,
        action: TrialAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrialObservation:
        """Advance the episode by one step.

        Args:
            action: TrialAction to execute.
            timeout_s: Ignored — local execution has no network timeout.

        Returns:
            TrialObservation after the action (reward embedded in obs.reward).

        Raises:
            RuntimeError: If no active episode exists (call reset() first).
        """
        obs, _reward_breakdown, _done, _info = self._manager.step(action)
        self._last_obs = obs
        return self._apply_transform(obs)

    @property
    def state(self) -> TrialState:
        """Return the current TrialState (training-loop metadata)."""
        return self._manager.get_state()

    # ------------------------------------------------------------------
    # Extended helpers for training scripts
    # ------------------------------------------------------------------

    def step_full(
        self,
        action: TrialAction,
    ) -> tuple[TrialObservation, dict[str, Any], bool, dict[str, Any]]:
        """Advance the episode and return the full (obs, reward, done, info) tuple.

        Mirrors the FastAPI /step response structure for training scripts that
        need the reward breakdown and done flag alongside the observation.

        Args:
            action: TrialAction to execute.

        Returns:
            Tuple of (observation, reward_dict, done, info).
        """
        obs, reward_breakdown, done, info = self._manager.step(action)
        self._last_obs = obs
        return self._apply_transform(obs), reward_breakdown.model_dump(), done, info
