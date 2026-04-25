"""
OutputGenerator — produces a noisy TrialObservation from a TrialLatentState.

Follows the Bio Experiment pattern: TransitionEngine updates hidden state,
OutputGenerator produces noisy observations from it. Agent never sees clean
hidden values.

Key responsibilities:
  - Inject measurement noise and site variability via NoiseModel's seeded RNG
  - Populate phase_data with noisy (not raw) experimental results
  - Populate resource_status from latent state resource fields
  - Populate available_actions based on current milestone flags and phase
  - Never expose true_effect_size, true_side_effect_rate, or other hidden values
    directly — always add noise before returning to the agent
"""

from __future__ import annotations

import numpy as np

from models import ActionType, TrialLatentState, TrialObservation, TrialState
from server.noise_model import NoiseModel
from server.rules.fda_rules import TRANSITION_TABLE
from server.rules.prerequisite_rules import _HISTORY_PREREQUISITES


class OutputGenerator:
    """Produces a noisy TrialObservation from a TrialLatentState.

    The agent never sees clean hidden values — all experimental results are
    perturbed by measurement noise and site variability before being returned.

    Args:
        noise_model: Seeded NoiseModel used to draw observation noise.
    """

    def __init__(self, noise_model: NoiseModel) -> None:
        self._noise_model = noise_model

    def generate(
        self,
        latent: TrialLatentState,
        trial_state: TrialState,
        *,
        steps_taken: int,
        max_steps: int,
        rule_violations: list[str],
        done: bool,
        reward: float,
        scenario_description: str,
        hint: str = "",
    ) -> TrialObservation:
        """Generate a noisy TrialObservation from the current latent state.

        Args:
            latent: Updated hidden state from TransitionEngine.
            trial_state: Episode metadata (difficulty, curriculum tier, etc.).
            steps_taken: Number of steps taken so far in the episode.
            max_steps: Maximum steps allowed in the episode.
            rule_violations: List of rule violation strings from this step.
            done: Whether the episode is finished.
            reward: Reward signal for this step.
            scenario_description: Human-readable scenario description.
            hint: Optional hint string (only populated at junior difficulty).

        Returns:
            A TrialObservation with noisy phase_data, resource_status, and
            available_actions. Raw hidden values are never included.
        """
        rng = self._noise_model.rng

        phase_data = self._build_phase_data(latent, rng)
        resource_status = self._build_resource_status(latent)
        available_actions = self._build_available_actions(latent)

        return TrialObservation(
            scenario_description=scenario_description,
            phase_data=phase_data,
            resource_status=resource_status,
            rule_violations=rule_violations,
            available_actions=available_actions,
            steps_taken=steps_taken,
            max_steps=max_steps,
            hint=hint,
            done=done,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_phase_data(
        self,
        latent: TrialLatentState,
        rng: "np.random.Generator",
    ) -> dict:
        """Build noisy phase_data dict — never exposes raw hidden values.

        Measurement noise (latent.measurement_noise) is applied to effect-size
        estimates. Site variability (latent.site_variability) is applied to
        adverse-event-rate estimates.
        """
        import numpy as np  # local import to keep module-level deps minimal

        noise_std = max(latent.measurement_noise, 1e-6)
        site_std = max(latent.site_variability, 1e-6)

        phase_data: dict = {
            "current_phase": latent.episode_phase,
            "patients_enrolled": latent.patients_enrolled,
            # Milestones — these are observable flags, not hidden values
            "phase_i_complete": latent.phase_i_complete,
            "mtd_identified": latent.mtd_identified,
            "effect_estimated": latent.effect_estimated,
            "protocol_submitted": latent.protocol_submitted,
            "interim_complete": latent.interim_complete,
            "trial_complete": latent.trial_complete,
        }

        # Noisy effect-size estimate — only available after ESTIMATE_EFFECT_SIZE
        if latent.effect_estimated:
            noisy_effect = float(latent.true_effect_size + rng.normal(0.0, noise_std))
            phase_data["observed_effect_size"] = round(noisy_effect, 4)

            # Noisy confidence interval width (derived from noise level)
            ci_half_width = float(rng.normal(noise_std * 2, noise_std * 0.5))
            ci_half_width = max(ci_half_width, 0.01)
            phase_data["effect_size_ci"] = (
                round(noisy_effect - ci_half_width, 4),
                round(noisy_effect + ci_half_width, 4),
            )

        # Noisy adverse-event rate — only available after OBSERVE_SAFETY_SIGNAL
        # or RUN_DOSE_ESCALATION
        if (
            latent.phase_i_complete
            or ActionType.OBSERVE_SAFETY_SIGNAL.value in latent.action_history
        ):
            noisy_ae_rate = float(
                latent.true_side_effect_rate + rng.normal(0.0, site_std)
            )
            noisy_ae_rate = float(np.clip(noisy_ae_rate, 0.0, 1.0))
            phase_data["observed_adverse_event_rate"] = round(noisy_ae_rate, 4)

        # Noisy placebo response — only available after interim or primary analysis
        if latent.interim_complete or latent.trial_complete:
            noisy_placebo = float(
                latent.placebo_response_rate + rng.normal(0.0, noise_std)
            )
            noisy_placebo = float(np.clip(noisy_placebo, 0.0, 1.0))
            phase_data["observed_placebo_response"] = round(noisy_placebo, 4)

        # Noisy dose-response curve — only available after Phase I
        if latent.phase_i_complete and latent.true_dose_response:
            noisy_dose_response: dict[str, float] = {}
            for dose, response in latent.true_dose_response.items():
                noisy_resp = float(response + rng.normal(0.0, noise_std))
                noisy_resp = float(np.clip(noisy_resp, 0.0, 1.0))
                noisy_dose_response[str(dose)] = round(noisy_resp, 4)
            phase_data["observed_dose_response"] = noisy_dose_response

        # Dropout rate estimate — noisy, only after enrollment begins
        if latent.patients_enrolled > 0:
            noisy_dropout = float(
                latent.dropout_rate + rng.normal(0.0, noise_std * 0.5)
            )
            noisy_dropout = float(np.clip(noisy_dropout, 0.0, 1.0))
            phase_data["observed_dropout_rate"] = round(noisy_dropout, 4)

        # Responder population hint — only after biomarker stratification
        if ActionType.ADD_BIOMARKER_STRATIFICATION.value in latent.action_history:
            # Reveal population label but NOT the true criteria (hidden)
            phase_data["responder_population_hint"] = latent.true_responder_population

        return phase_data

    def _build_resource_status(self, latent: TrialLatentState) -> dict:
        """Build resource_status from latent state resource fields."""
        return {
            "budget_remaining": latent.budget_remaining,
            "time_remaining_days": latent.time_remaining_days,
            "patients_enrolled": latent.patients_enrolled,
        }

    def _build_available_actions(self, latent: TrialLatentState) -> list[str]:
        """Return the list of valid action strings given current milestone flags.

        Filters the phase-permitted actions through prerequisite checks so the
        agent only sees actions it can actually take right now.
        """
        phase_permitted: set[ActionType] = TRANSITION_TABLE.get(
            latent.episode_phase, set()
        )

        available: list[str] = []
        for action_type in sorted(phase_permitted, key=lambda a: a.value):
            if self._prerequisites_met(action_type, latent):
                available.append(action_type.value)

        return available

    def _prerequisites_met(
        self, action_type: ActionType, latent: TrialLatentState
    ) -> bool:
        """Return True if all prerequisites for *action_type* are satisfied."""
        # History-based prerequisites
        required_actions = _HISTORY_PREREQUISITES.get(action_type, [])
        for required in required_actions:
            if required.value not in latent.action_history:
                return False

        # State-flag prerequisites (mirrors prerequisite_rules.py logic)
        if action_type == ActionType.REQUEST_PROTOCOL_AMENDMENT:
            if not latent.protocol_submitted:
                return False

        if action_type == ActionType.SUBMIT_TO_FDA_REVIEW:
            # Only require phase_i_complete — protocol_submitted is set BY this
            # action itself, so checking it here would create a Catch-22.
            if not latent.phase_i_complete:
                return False

        if action_type == ActionType.RUN_PRIMARY_ANALYSIS:
            if not latent.interim_complete:
                return False

        if action_type == ActionType.RUN_INTERIM_ANALYSIS:
            if latent.patients_enrolled <= 0:
                return False

        if action_type == ActionType.MODIFY_SAMPLE_SIZE:
            if ActionType.SET_SAMPLE_SIZE.value not in latent.action_history:
                return False

        if action_type == ActionType.SYNTHESIZE_CONCLUSION:
            if not latent.trial_complete:
                return False

        return True
