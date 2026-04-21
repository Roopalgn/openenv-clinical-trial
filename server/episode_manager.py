"""
EpisodeManager — central orchestrator for the Clinical Trial Designer environment.

Manages the reset/step lifecycle. TrialLatentState holds all hidden ground truth
and episode tracking. TrialState is the lightweight metadata visible to the
training loop. TrialObservation is the noisy agent-facing view.
"""

from __future__ import annotations

import random
import uuid

from models import (
    RewardBreakdown,
    ScenarioConfig,
    TrialAction,
    TrialLatentState,
    TrialObservation,
    TrialResult,
    TrialState,
)
from server.logger import EpisodeLogger
from server.noise_model import NoiseModel
from server.rules.fda_rules import check_fda_compliance

# Default scenario used until CurriculumController is fully wired (Push 3).
_DEFAULT_SCENARIO = ScenarioConfig(
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
    description="Solid tumor chemotherapy — find EGFR+ subgroup",
)

_MAX_STEPS = 100


class EpisodeManager:
    """Orchestrates the reset/step lifecycle for a single clinical trial episode.

    TrialLatentState is the authoritative hidden state (ground truth + tracking).
    TrialState is the lightweight metadata snapshot for the training loop.
    """

    def __init__(self) -> None:
        self._latent: TrialLatentState | None = None
        self._state: TrialState | None = None
        self._power_cache: dict[tuple[float, int, float], float] = {}
        self._logger: EpisodeLogger | None = None
        self._total_reward: float = 0.0
        self._episode_id: str = ""
        self._difficulty: float = 0.0
        self._scenario: ScenarioConfig | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> TrialObservation:
        """Initialize a new episode and return the initial TrialObservation."""
        resolved_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        self._episode_id = str(uuid.uuid4())

        # Step 1: Select scenario (stub — CurriculumController wired in Push 3)
        scenario = _DEFAULT_SCENARIO
        self._scenario = scenario

        # Step 2: Apply domain randomization via NoiseModel (req 9.1, 9.2)
        noise_model = NoiseModel(seed=resolved_seed)
        randomized = noise_model.randomize(scenario)

        # Sample concrete hidden values from randomized ranges
        effect_lo, effect_hi = randomized.effect_size_range
        side_lo, side_hi = randomized.side_effect_rate_range
        placebo_lo, placebo_hi = randomized.placebo_response_range
        dropout_lo, dropout_hi = randomized.dropout_rate_range

        rng = noise_model._rng
        true_effect = float(rng.uniform(effect_lo, effect_hi))
        true_side = float(rng.uniform(side_lo, side_hi))
        true_placebo = float(rng.uniform(placebo_lo, placebo_hi))
        true_dropout = float(rng.uniform(dropout_lo, dropout_hi))

        # Step 3: Build TrialLatentState — holds ALL hidden + tracking state
        self._latent = TrialLatentState(
            true_effect_size=true_effect,
            true_side_effect_rate=true_side,
            true_responder_population="all",
            true_responder_criteria=[],
            true_dose_response={},
            true_mechanism="unknown",
            placebo_response_rate=true_placebo,
            dropout_rate=true_dropout,
            site_variability=0.0,
            measurement_noise=0.0,
            budget_remaining=randomized.budget_usd,
            time_remaining_days=randomized.time_budget_days,
            patients_enrolled=0,
            phase_i_complete=False,
            mtd_identified=False,
            effect_estimated=False,
            protocol_submitted=False,
            interim_complete=False,
            trial_complete=False,
            episode_phase="literature_review",
            action_history=[],
            seed=resolved_seed,
        )

        # Step 4: Build lightweight TrialState for training loop
        self._state = self._state_from_latent(self._latent, randomized)

        self._clear_cache()

        # Step 5: Fresh logger and reward accumulator
        self._logger = EpisodeLogger(
            curriculum_tier=randomized.curriculum_tier
        )
        self._total_reward = 0.0
        self._difficulty = 0.0

        return self._observation_from_latent(self._latent, randomized)

    def step(
        self, action: TrialAction
    ) -> tuple[TrialObservation, RewardBreakdown, bool, dict]:
        """Advance the episode by one step."""
        if self._latent is None or self._scenario is None:
            raise RuntimeError("No active episode. Call reset() before step().")

        try:
            # Check FDA compliance against latent state (req 10.1, 10.4)
            compliance = check_fda_compliance(action, self._latent)

            if not compliance.valid:
                reward = RewardBreakdown(
                    r_validity=-1.0,
                    r_ordering=0.0,
                    r_info_gain=0.0,
                    r_efficiency=0.0,
                    r_novelty=0.0,
                    r_penalty=0.0,
                    r_terminal_success=0.0,
                    r_terminal_calibration=0.0,
                )
                done = False
                info: dict = {
                    "step_index": len(self._latent.action_history),
                    "action_valid": False,
                    "violations": compliance.violations,
                }
                obs = self._observation_from_latent(
                    self._latent,
                    self._scenario,
                    rule_violations=compliance.violations,
                )
                if self._logger is not None:
                    self._logger.log_step(
                        len(self._latent.action_history), action, obs, reward, done
                    )
                return obs, reward, done, info

            # Valid action: advance latent state
            self._latent = self._latent.model_copy(
                update={
                    "action_history": (
                        self._latent.action_history + [action.action_type.value]
                    ),
                }
            )

            # Stub TrialResult
            _result = TrialResult(
                p_value=0.05,
                success=False,
                power=0.8,
                adverse_event_rate=0.1,
                confidence_interval=(0.0, 1.0),
                failure_reason=None,
            )

            # Stub reward
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

            step_idx = len(self._latent.action_history)
            done = step_idx >= _MAX_STEPS or self._latent.trial_complete
            info = {"step_index": step_idx, "action_valid": True}

            obs = self._observation_from_latent(self._latent, self._scenario)

            # Update training-loop TrialState
            self._state = self._state_from_latent(self._latent, self._scenario)

            # Accumulate reward and log step (req 7.1)
            self._total_reward += sum(reward.model_dump().values())
            if self._logger is not None:
                self._logger.log_step(step_idx, action, obs, reward, done)

            # Log summary on episode end (req 7.2)
            if done and self._logger is not None:
                self._logger.log_summary(
                    scenario_id=self._scenario.scenario_id,
                    total_reward=self._total_reward,
                    episode_length=step_idx,
                    terminal_outcome=(
                        "success" if self._latent.trial_complete else "timeout"
                    ),
                )

            return obs, reward, done, info

        except RuntimeError:
            raise
        except Exception as exc:  # req 10.4: no unhandled exceptions
            reward = RewardBreakdown(
                r_validity=-1.0,
                r_ordering=0.0,
                r_info_gain=0.0,
                r_efficiency=0.0,
                r_novelty=0.0,
                r_penalty=0.0,
                r_terminal_success=0.0,
                r_terminal_calibration=0.0,
            )
            step_idx = len(self._latent.action_history) if self._latent else 0
            info = {
                "step_index": step_idx,
                "action_valid": False,
                "violations": [f"Internal error: {exc}"],
            }
            obs = self._observation_from_latent(
                self._latent,
                self._scenario,
                rule_violations=[f"Internal error: {exc}"],
            )
            return obs, reward, False, info

    def get_state(self) -> TrialState:
        """Return the current TrialState (training-loop metadata)."""
        if self._state is None:
            raise RuntimeError("No active episode. Call reset() before get_state().")
        return self._state

    def get_latent(self) -> TrialLatentState:
        """Return the full hidden TrialLatentState (for offline debugging)."""
        if self._latent is None:
            raise RuntimeError("No active episode. Call reset() before get_latent().")
        return self._latent

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _clear_cache(self) -> None:
        self._power_cache.clear()

    def _state_from_latent(
        self, latent: TrialLatentState, scenario: ScenarioConfig
    ) -> TrialState:
        """Build the lightweight TrialState from latent state."""
        step_count = len(latent.action_history)
        unique_actions = len(set(latent.action_history))
        action_diversity = (
            unique_actions / step_count if step_count > 0 else 0.0
        )
        return TrialState(
            episode_id=self._episode_id,
            step_count=step_count,
            difficulty=self._difficulty,
            scenario_id=scenario.scenario_id,
            curriculum_tier=str(scenario.curriculum_tier),
            curriculum_stats={},
            action_diversity=action_diversity,
            phase_compliance_rate=0.0,  # wired in Push 3 with PhaseDetector
            is_resolved=latent.trial_complete,
        )

    def _observation_from_latent(
        self,
        latent: TrialLatentState,
        scenario: ScenarioConfig,
        rule_violations: list[str] | None = None,
    ) -> TrialObservation:
        """Build a TrialObservation from latent state — noisy, agent-facing."""
        return TrialObservation(
            scenario_description=scenario.description,
            phase_data={
                "episode_phase": latent.episode_phase,
                "observed_effect_estimate": None,
                "observed_side_effect_rate": None,
                "phase_i_complete": latent.phase_i_complete,
                "interim_complete": latent.interim_complete,
                "protocol_submitted": latent.protocol_submitted,
            },
            resource_status={
                "budget_remaining": latent.budget_remaining,
                "time_remaining_days": latent.time_remaining_days,
                "patients_enrolled": latent.patients_enrolled,
            },
            rule_violations=rule_violations or [],
            available_actions=[],  # wired in Push 3 with TransitionEngine
            steps_taken=len(latent.action_history),
            max_steps=_MAX_STEPS,
            hint="",  # populated by TrialJudge at junior difficulty (Push 3)
            done=latent.trial_complete,
            reward=0.0,  # filled in by step() after reward computation
        )
