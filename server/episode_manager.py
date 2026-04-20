"""
EpisodeManager — central orchestrator for the Clinical Trial Designer environment.

This module provides skeleton stubs for reset(), step(), and get_state().
No real logic is implemented here; all methods return properly typed model
instances with placeholder/default values. Logic will be wired in later pushes.
"""

from __future__ import annotations

from models import (
    RewardBreakdown,
    TrialAction,
    TrialLatentState,
    TrialObservation,
    TrialResult,
    TrialState,
)


class EpisodeManager:
    """Orchestrates the reset/step lifecycle for a single clinical trial episode.

    Holds the active TrialState, TrialLatentState, and power cache.
    All state mutation flows through this class; both the FastAPI routes and
    the OpenEnv Environment subclass delegate to it.
    """

    def __init__(self) -> None:
        self._state: TrialState | None = None
        self._latent: TrialLatentState | None = None
        self._power_cache: dict[tuple[float, int, float], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> TrialObservation:
        """Initialize a new episode and return the initial TrialObservation.

        Stub flow (no real logic yet):
        1. Select scenario via CurriculumController (stub)
        2. Generate TrialLatentState via hidden-state generator (stub)
        3. Initialize TrialState from ScenarioConfig (stub)
        4. Clear power cache
        5. Log episode start (stub)
        6. Return initial TrialObservation

        Args:
            seed: Optional integer seed for reproducible episode generation.

        Returns:
            A TrialObservation with default/placeholder values.
        """
        resolved_seed = seed if seed is not None else 0

        # Stub: placeholder TrialLatentState
        self._latent = TrialLatentState(
            true_effect_size=0.5,
            true_side_effect_rate=0.1,
            true_responder_population="all",
            placebo_response_rate=0.2,
            dropout_rate=0.05,
            site_variability=0.0,
            measurement_noise=0.0,
            true_dose_response={},
            true_mechanism="unknown",
            true_responder_criteria=[],
        )

        # Stub: placeholder TrialState
        self._state = TrialState(
            true_effect_size=0.5,
            true_side_effect_rate=0.1,
            true_responder_population="all",
            placebo_response_rate=0.2,
            dropout_rate=0.05,
            budget_remaining=1_000_000.0,
            time_remaining_days=365,
            patients_enrolled=0,
            step_index=0,
            episode_phase="setup",
            scenario_id="solid_tumor_chemo",
            curriculum_tier=0,
            action_history=[],
            phase_i_complete=False,
            interim_complete=False,
            protocol_submitted=False,
            trial_complete=False,
            seed=resolved_seed,
        )

        self._clear_cache()

        return self._observation_from_state(self._state)

    def step(
        self, action: TrialAction
    ) -> tuple[TrialObservation, RewardBreakdown, bool, dict]:
        """Advance the episode by one step and return the result.

        Stub flow (no real logic yet):
        1. Validate active episode exists (400 if not)
        2. Call check_fda_compliance(action, state) → compliance result (stub)
        3. Call simulate_trial(state, action, latent) → TrialResult (stub)
        4. Call compute_reward(action, state, result, latent) → RewardBreakdown (stub)
        5. Advance TrialState (stub)
        6. Check terminal condition (stub)
        7. Log step to JSONL (stub)
        8. Return (TrialObservation, RewardBreakdown, done, info)

        Args:
            action: The TrialAction submitted by the agent.

        Returns:
            A 4-tuple of (TrialObservation, RewardBreakdown, done, info).

        Raises:
            RuntimeError: If no active episode exists (no reset() called yet).
        """
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() before step().")

        # Stub: advance step index
        self._state = self._state.model_copy(
            update={
                "step_index": self._state.step_index + 1,
                "action_history": (
                    self._state.action_history + [action.action_type.value]
                ),
            }
        )

        # Stub: placeholder TrialResult
        _result = TrialResult(
            p_value=0.05,
            success=False,
            power=0.8,
            adverse_event_rate=0.1,
            confidence_interval=(0.0, 1.0),
            failure_reason=None,
        )

        # Stub: zero-valued RewardBreakdown
        reward = RewardBreakdown(
            r_validity=0.0,
            r_ordering=0.0,
            r_info_gain=0.0,
            r_efficiency=0.0,
            r_novelty=0.0,
            r_penalty=0.0,
            r_terminal_success=0.0,
            r_terminal_calibration=0.0,
        )

        # Stub: episode never terminates
        done = False
        info: dict = {"step_index": self._state.step_index}

        obs = self._observation_from_state(self._state)
        return obs, reward, done, info

    def get_state(self) -> TrialState:
        """Return the current full TrialState for the active episode.

        Returns:
            The current TrialState.

        Raises:
            RuntimeError: If no active episode exists.
        """
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() before get_state().")
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear_cache(self) -> None:
        """Clear the power calculation cache. Called on reset()."""
        self._power_cache.clear()

    def _observation_from_state(self, state: TrialState) -> TrialObservation:
        """Build a TrialObservation from the current TrialState.

        Only exposes the fields that are visible to the agent (i.e. excludes
        hidden ground-truth fields such as true_effect_size).
        """
        return TrialObservation(
            step_index=state.step_index,
            episode_phase=state.episode_phase,
            budget_remaining=state.budget_remaining,
            time_remaining_days=state.time_remaining_days,
            patients_enrolled=state.patients_enrolled,
            last_action_valid=True,
            violation_messages=[],
            observed_effect_estimate=None,
            observed_side_effect_rate=None,
            phase_i_complete=state.phase_i_complete,
            interim_complete=state.interim_complete,
            protocol_submitted=state.protocol_submitted,
            scenario_id=state.scenario_id,
            curriculum_tier=state.curriculum_tier,
        )
