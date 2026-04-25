"""
EpisodeManager — central orchestrator for the Clinical Trial Designer environment.

Manages the reset/step lifecycle. TrialLatentState holds all hidden ground truth
and episode tracking. TrialState is the lightweight metadata visible to the
training loop. TrialObservation is the noisy agent-facing view.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

import numpy as np

from models import (
    EpisodeTranscript,
    RewardBreakdown,
    ScenarioConfig,
    TrialAction,
    TrialLatentState,
    TrialObservation,
    TrialState,
)
from server.config import settings
from server.curriculum.adversarial_designer import (
    EXPERT_DIFFICULTY_THRESHOLD,
    AdversarialDesigner,
)
from server.curriculum.controller import (
    EpisodeMetrics,
    advance_curriculum,
    select_scenario,
)
from server.judge import TrialJudge
from server.logger import EpisodeLogger
from server.noise_model import NoiseModel
from server.phase_detector import detect_phase
from server.reward.reward_computer import compute_reward
from server.reward.shaping import shaping_bonus
from server.rules.fda_rules import check_fda_compliance
from server.simulator.output_generator import OutputGenerator
from server.simulator.power_calculator import calculate_power
from server.simulator.transition_engine import TransitionEngine
from server.simulator.trial_simulator import simulate_trial

_MAX_STEPS = 100


def _latent_biology_from_scenario(scenario: ScenarioConfig) -> dict:
    """Return scenario-specific hidden biology values for TrialLatentState."""
    scenario_id = scenario.scenario_id
    if scenario_id in {"solid_tumor_chemo", "solid_tumor_chemo_warmup"}:
        return {
            "true_responder_population": "EGFR+",
            "true_responder_criteria": ["EGFR_mutation"],
            "true_dose_response": {
                50.0: 0.08,
                100.0: 0.19,
                150.0: 0.31,
                200.0: 0.29,
                250.0: 0.22,
            },
            "true_mechanism": "EGFR tyrosine kinase inhibition",
        }
    if scenario_id == "autoimmune_biologic":
        return {
            "true_responder_population": "all",
            "true_responder_criteria": ["RA_diagnosis"],
            "true_dose_response": {
                50.0: 0.12,
                100.0: 0.28,
                150.0: 0.38,
                200.0: 0.42,
                250.0: 0.35,
                300.0: 0.22,
                400.0: 0.10,
            },
            "true_mechanism": "IL-6 receptor blockade",
        }
    if scenario_id == "cns_depression":
        return {
            "true_responder_population": "severe_trd",
            "true_responder_criteria": ["MADRS>=35"],
            "true_dose_response": {
                25.0: 0.06,
                50.0: 0.14,
                75.0: 0.18,
                100.0: 0.17,
                150.0: 0.12,
            },
            "true_mechanism": "NMDA pathway modulation",
        }
    if scenario_id == "rare_disease_orphan":
        return {
            "true_responder_population": "all",
            "true_responder_criteria": ["Morquio_A_confirmed"],
            "true_dose_response": {
                0.5: 0.40,
                1.0: 0.80,
                2.0: 1.20,
                3.0: 1.15,
                4.0: 0.95,
            },
            "true_mechanism": "enzyme replacement therapy",
        }
    # Fallback for generated or unknown scenarios.
    return {
        "true_responder_population": "all",
        "true_responder_criteria": [],
        "true_dose_response": {},
        "true_mechanism": "unknown",
    }


def _phase_order_correct_at(phase: str, prior_history: list[str]) -> bool:
    """Return True if `phase` is a valid next phase given `prior_history`."""
    from server.phase_detector import PHASE_ORDER

    if not prior_history:
        return True
    last = prior_history[-1]
    last_idx = PHASE_ORDER.index(last) if last in PHASE_ORDER else 0
    current_idx = PHASE_ORDER.index(phase) if phase in PHASE_ORDER else 0
    return current_idx >= last_idx and (current_idx - last_idx) <= 1


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
        self._phase_history: list[str] = []
        self._noise_model: NoiseModel | None = None
        self._curriculum_tier: int = settings.curriculum_start_tier
        self._episode_history: list[bool] = []  # rolling success history for curriculum
        self._episode_outcomes: list[dict] = []  # richer history for adversarial analysis
        self._adversarial_designer: AdversarialDesigner = AdversarialDesigner()
        self._transition_engine: TransitionEngine = TransitionEngine()
        self._judge: TrialJudge = TrialJudge()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> TrialObservation:
        """Initialize a new episode and return the initial TrialObservation.

        Seeded resets are reproducible: same seed → same scenario selection
        and initial TrialLatentState (Req 8.5, 9.4).
        """
        if seed is not None:
            resolved_seed = seed
        elif settings.default_seed is not None:
            resolved_seed = settings.default_seed
        else:
            resolved_seed = random.randint(0, 2**31 - 1)
        self._episode_id = str(uuid.uuid4())

        # Step 1: Select scenario via CurriculumController (Req 8.3, 8.5)
        # Use a seeded RNG so scenario selection is reproducible for same seed.
        scenario_rng = np.random.default_rng(resolved_seed)

        # At expert tier (difficulty > 0.80), use AdversarialDesigner to generate
        # a targeted scenario based on the agent's tracked weak spots.
        current_difficulty = self._curriculum_tier / 4.0
        if current_difficulty > EXPERT_DIFFICULTY_THRESHOLD and self._episode_history:
            weak_spots = self._adversarial_designer.analyze_failures(
                self._episode_outcomes
            )
            scenario = self._adversarial_designer.generate_scenario(weak_spots)
        else:
            scenario = select_scenario(self._curriculum_tier, scenario_rng)
        self._scenario = scenario

        # Step 2: Apply domain randomization via NoiseModel (Req 9.1, 9.2)
        # NoiseModel is seeded so same seed → same randomized config.
        noise_model = NoiseModel(seed=resolved_seed)
        self._noise_model = noise_model
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
        latent_biology = _latent_biology_from_scenario(scenario)

        # Step 3: Build TrialLatentState — holds ALL hidden + tracking state
        self._latent = TrialLatentState(
            true_effect_size=true_effect,
            true_side_effect_rate=true_side,
            true_responder_population=latent_biology["true_responder_population"],
            true_responder_criteria=latent_biology["true_responder_criteria"],
            true_dose_response=latent_biology["true_dose_response"],
            true_mechanism=latent_biology["true_mechanism"],
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
            adverse_events=0,
            episode_phase="literature_review",
            action_history=[],
            seed=resolved_seed,
        )

        # Step 4: Build lightweight TrialState for training loop
        self._state = self._state_from_latent(self._latent, randomized)

        # Step 5: Clear power cache (Req 14.3)
        self._clear_cache()
        self._phase_history = []

        # Step 6: Fresh logger (episode_id matches this episode), reward accumulator
        self._logger = EpisodeLogger(
            episode_id=self._episode_id,
            curriculum_tier=randomized.curriculum_tier,
        )
        self._total_reward = 0.0
        # Difficulty scales linearly with curriculum tier: tier 0 → 0.0, tier 4 → 1.0
        self._difficulty = scenario.curriculum_tier / 4.0

        # Step 7: Return initial TrialObservation via OutputGenerator
        output_gen = OutputGenerator(noise_model)
        return output_gen.generate(
            latent=self._latent,
            trial_state=self._state,
            steps_taken=0,
            max_steps=_MAX_STEPS,
            rule_violations=[],
            done=False,
            reward=0.0,
            scenario_description=scenario.description,
            hint="",
        )

    def step(
        self, action: TrialAction
    ) -> tuple[TrialObservation, RewardBreakdown, bool, dict]:
        """Advance the episode by one step.

        Full pipeline (Req 8.5, 9.4, 7.1):
          1. Validate active episode
          2. check_fda_compliance → ComplianceResult
          3. TransitionEngine.apply_transition() mutates TrialLatentState
          4. OutputGenerator.generate() produces noisy TrialObservation
          5. compute_reward() → RewardBreakdown
          6. PhaseDetector.detect_phase() classifies action
          7. TrialJudge.verify() for hint/feedback
          8. Check terminal condition
          9. Log full EpisodeTranscript to JSONL
          10. Return (obs, reward_breakdown, done, info)
        """
        if self._latent is None or self._scenario is None:
            raise RuntimeError("No active episode. Call reset() before step().")

        try:
            # Step 1: Check FDA compliance (read-only, does not mutate state)
            compliance = check_fda_compliance(action, self._latent)

            if not compliance.valid:
                reward = RewardBreakdown(
                    r_validity=-1.0,
                    r_ordering=0.0,
                    r_info_gain=0.0,
                    r_efficiency=0.0,
                    r_novelty=0.0,
                    r_penalty=-0.5 * len(compliance.violations),
                    r_terminal_success=0.0,
                    r_terminal_calibration=0.0,
                )
                done = False
                step_idx = len(self._latent.action_history)
                info: dict = {
                    "step_index": step_idx,
                    "action_valid": False,
                    "violations": compliance.violations,
                }
                # Build observation without mutating latent
                noise_model = self._noise_model or NoiseModel(seed=self._latent.seed)
                output_gen = OutputGenerator(noise_model)
                obs = output_gen.generate(
                    latent=self._latent,
                    trial_state=self._state
                    or self._state_from_latent(self._latent, self._scenario),
                    steps_taken=step_idx,
                    max_steps=_MAX_STEPS,
                    rule_violations=compliance.violations,
                    done=False,
                    reward=reward.total,
                    scenario_description=self._scenario.description,
                    hint="",
                )
                # Log invalid step
                if self._logger is not None:
                    self._logger.log_step(step_idx, action, obs, reward, done)
                return obs, reward, done, info

            # Step 2: TransitionEngine mutates TrialLatentState
            latent_before = self._latent  # snapshot for shaping bonus (Issue #1)
            updated_latent = self._transition_engine.apply_transition(
                self._latent, action
            )
            self._latent = updated_latent

            # Step 3: Detect phase and update phase history
            phase_name, phase_order_correct = detect_phase(action, self._phase_history)
            self._phase_history = self._phase_history + [phase_name]

            # Step 4: Simulate trial result for reward computation
            result = simulate_trial(
                self._latent, action, power_fn=self.cached_calculate_power
            )

            # Step 5: Compute reward (all 8 components)
            reward = compute_reward(
                action=action,
                latent=self._latent,
                result=result,
                phase_history=self._phase_history[:-1],  # history before this step
                initial_budget=float(self._scenario.budget_usd),
            )

            # Add potential-based shaping bonus: γ·(φ(s') − φ(s))
            initial_budget = float(self._scenario.budget_usd)
            shaped_bonus = shaping_bonus(
                latent=latent_before,
                next_latent=self._latent,
                initial_budget=initial_budget,
            )
            reward = reward.model_copy(
                update={"r_ordering": reward.r_ordering + shaped_bonus}
            )

            # Step 6: TrialJudge verification (hint + overconfidence penalty)
            self._state = self._state_from_latent(self._latent, self._scenario)
            judge_result = self._judge.verify(action, self._state, self._latent)
            hint = judge_result.hint or ""

            # Apply overconfidence penalty to r_penalty
            if judge_result.overconfidence_penalty != 0.0:
                reward = reward.model_copy(
                    update={
                        "r_penalty": (
                            reward.r_penalty + judge_result.overconfidence_penalty
                        )
                    }
                )

            # Step 7: Check terminal condition
            step_idx = len(self._latent.action_history)
            done = step_idx >= _MAX_STEPS or self._latent.trial_complete

            # Step 8: Generate noisy observation via OutputGenerator
            noise_model = self._noise_model or NoiseModel(seed=self._latent.seed)
            output_gen = OutputGenerator(noise_model)
            obs = output_gen.generate(
                latent=self._latent,
                trial_state=self._state,
                steps_taken=step_idx,
                max_steps=_MAX_STEPS,
                rule_violations=[],
                done=done,
                reward=reward.total,
                scenario_description=self._scenario.description,
                hint=hint,
            )

            # Step 9: Accumulate total reward
            self._total_reward += reward.total

            # Step 10: Log full EpisodeTranscript record to JSONL (Req 7.1)
            transcript = EpisodeTranscript(
                episode_id=self._episode_id,
                step=step_idx,
                action=action,
                observation=obs,
                reward_breakdown=reward.model_dump(),
                total_reward=reward.total,
                phase_detected=phase_name,
                phase_order_correct=phase_order_correct,
                hidden_state_snapshot=self._latent,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            if self._logger is not None:
                self._logger.log_step(step_idx, action, obs, reward, done)
                # Also write the full EpisodeTranscript as a separate JSONL record
                self._logger._append_jsonl(
                    {"type": "transcript", **transcript.model_dump(mode="json")}
                )

            # Log summary on episode end (Req 7.2)
            if done and self._logger is not None:
                self._logger.log_summary(
                    scenario_id=self._scenario.scenario_id,
                    total_reward=self._total_reward,
                    episode_length=step_idx,
                    terminal_outcome=(
                        "success" if self._latent.trial_complete else "timeout"
                    ),
                )

            # Advance curriculum tier at episode end (Issue #2)
            if done:
                episode_success = self._latent.trial_complete
                self._episode_history.append(episode_success)
                self._episode_outcomes.append(
                    {
                        "success": episode_success,
                        "scenario_id": self._scenario.scenario_id,
                        "true_effect_size": self._latent.true_effect_size,
                        "dropout_rate": self._latent.dropout_rate,
                    }
                )
                metrics = EpisodeMetrics(
                    success=episode_success,
                    episode_history=self._episode_history,
                )
                self._curriculum_tier = advance_curriculum(
                    self._curriculum_tier, metrics
                )

            info = {
                "step_index": step_idx,
                "action_valid": True,
                "phase_detected": phase_name,
                "phase_order_correct": phase_order_correct,
                "judge_passed": judge_result.passed,
                "judge_feedback": judge_result.feedback,
                "judge_hint": hint,
                "overconfidence_penalty": judge_result.overconfidence_penalty,
            }

            return obs, reward, done, info

        except RuntimeError:
            raise
        except Exception as exc:  # Req 10.4: no unhandled exceptions
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
            noise_model = self._noise_model or NoiseModel(
                seed=self._latent.seed if self._latent else 0
            )
            output_gen = OutputGenerator(noise_model)
            obs = (
                output_gen.generate(
                    latent=self._latent,
                    trial_state=self._state
                    or TrialState(
                        episode_id=self._episode_id,
                        step_count=step_idx,
                        difficulty=self._difficulty,
                        scenario_id=self._scenario.scenario_id
                        if self._scenario
                        else "",
                        curriculum_tier="0",
                        curriculum_stats={},
                        action_diversity=0.0,
                        phase_compliance_rate=0.0,
                        is_resolved=False,
                    ),
                    steps_taken=step_idx,
                    max_steps=_MAX_STEPS,
                    rule_violations=[f"Internal error: {exc}"],
                    done=False,
                    reward=reward.total,
                    scenario_description=(
                        self._scenario.description if self._scenario else ""
                    ),
                    hint="",
                )
                if self._latent is not None
                else TrialObservation(
                    scenario_description="",
                    phase_data={},
                    resource_status={},
                    rule_violations=[f"Internal error: {exc}"],
                    available_actions=[],
                    steps_taken=step_idx,
                    max_steps=_MAX_STEPS,
                    hint="",
                    done=False,
                    reward=0.0,
                )
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

    def cached_calculate_power(
        self, effect_size: float, n: int, alpha: float = 0.05
    ) -> float:
        """Return cached power; compute and store on first call for this key.

        Satisfies Req 14.1 and 14.2: results are cached by (effect_size, n, alpha)
        within a single episode and reused without recomputation on subsequent calls.
        """
        key = (effect_size, n, alpha)
        if key not in self._power_cache:
            self._power_cache[key] = calculate_power(effect_size, n, alpha)
        return self._power_cache[key]

    def _state_from_latent(
        self, latent: TrialLatentState, scenario: ScenarioConfig
    ) -> TrialState:
        """Build the lightweight TrialState from latent state."""
        step_count = len(latent.action_history)
        unique_actions = len(set(latent.action_history))
        action_diversity = unique_actions / step_count if step_count > 0 else 0.0

        # Compute phase compliance rate from phase history
        phase_steps = len(self._phase_history)
        if phase_steps > 0:
            correct_count = sum(
                1
                for i, ph in enumerate(self._phase_history)
                if _phase_order_correct_at(ph, self._phase_history[:i])
            )
            phase_compliance_rate = correct_count / phase_steps
        else:
            phase_compliance_rate = 0.0

        return TrialState(
            episode_id=self._episode_id,
            step_count=step_count,
            difficulty=self._difficulty,
            scenario_id=scenario.scenario_id,
            curriculum_tier=str(scenario.curriculum_tier),
            curriculum_stats={},
            action_diversity=action_diversity,
            phase_compliance_rate=phase_compliance_rate,
            is_resolved=latent.trial_complete,
        )
