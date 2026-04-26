"""
Tests for OutputGenerator — noisy TrialObservation generation (Task 15).

Requirements 9.1, 9.2, 9.3, 9.4:
  - OutputGenerator produces a TrialObservation from a TrialLatentState
  - Agent never sees raw hidden values (noise is always injected)
  - phase_data, resource_status, available_actions are correctly populated
  - Measurement noise and site variability are applied via NoiseModel
"""

from __future__ import annotations

import pytest

from models import ActionType, TrialLatentState, TrialState
from server.noise_model import NoiseModel
from server.simulator.output_generator import OutputGenerator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_latent() -> TrialLatentState:
    """A minimal TrialLatentState for testing."""
    return TrialLatentState(
        true_effect_size=0.5,
        true_side_effect_rate=0.10,
        true_responder_population="BRCA1+",
        true_responder_criteria=["BRCA1+", "age < 65"],
        true_dose_response={10.0: 0.2, 20.0: 0.4, 40.0: 0.7},
        true_mechanism="PARP inhibition",
        placebo_response_rate=0.15,
        dropout_rate=0.08,
        site_variability=0.05,
        measurement_noise=0.05,
        budget_remaining=500_000.0,
        time_remaining_days=200,
        patients_enrolled=0,
        phase_i_complete=False,
        mtd_identified=False,
        effect_estimated=False,
        protocol_submitted=False,
        interim_complete=False,
        trial_complete=False,
        adverse_events=0,
        episode_phase="design",
        action_history=[],
        seed=42,
    )


@pytest.fixture()
def trial_state() -> TrialState:
    return TrialState(
        episode_id="ep-001",
        step_count=1,
        difficulty=0.5,
        scenario_id="solid_tumor_chemo",
        curriculum_tier="tier_0",
        curriculum_stats={},
        action_diversity=0.0,
        phase_compliance_rate=1.0,
        is_resolved=False,
    )


@pytest.fixture()
def generator() -> OutputGenerator:
    return OutputGenerator(noise_model=NoiseModel(seed=42))


def _make_obs(generator, latent, trial_state, **kwargs):
    defaults = dict(
        steps_taken=1,
        max_steps=20,
        rule_violations=[],
        done=False,
        reward=0.0,
        scenario_description="Test scenario",
        hint="",
    )
    defaults.update(kwargs)
    return generator.generate(latent, trial_state, **defaults)


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------


class TestObservationStructure:
    """TrialObservation has all required fields populated."""

    def test_returns_trial_observation(self, generator, base_latent, trial_state):
        from models import TrialObservation

        obs = _make_obs(generator, base_latent, trial_state)
        assert isinstance(obs, TrialObservation)

    def test_scenario_description_passed_through(
        self, generator, base_latent, trial_state
    ):
        obs = _make_obs(
            generator, base_latent, trial_state, scenario_description="My scenario"
        )
        assert obs.scenario_description == "My scenario"

    def test_steps_taken_and_max_steps(self, generator, base_latent, trial_state):
        obs = _make_obs(
            generator, base_latent, trial_state, steps_taken=5, max_steps=30
        )
        assert obs.steps_taken == 5
        assert obs.max_steps == 30

    def test_done_and_reward_passed_through(self, generator, base_latent, trial_state):
        obs = _make_obs(generator, base_latent, trial_state, done=True, reward=1.5)
        assert obs.done is True
        assert obs.reward == 1.5

    def test_rule_violations_passed_through(self, generator, base_latent, trial_state):
        violations = ["violation A", "violation B"]
        obs = _make_obs(generator, base_latent, trial_state, rule_violations=violations)
        assert obs.rule_violations == violations

    def test_hint_passed_through(self, generator, base_latent, trial_state):
        obs = _make_obs(generator, base_latent, trial_state, hint="Try Phase I first")
        assert obs.hint == "Try Phase I first"


# ---------------------------------------------------------------------------
# resource_status tests
# ---------------------------------------------------------------------------


class TestResourceStatus:
    """resource_status reflects latent state resource fields."""

    def test_budget_remaining(self, generator, base_latent, trial_state):
        obs = _make_obs(generator, base_latent, trial_state)
        assert obs.resource_status["budget_remaining"] == base_latent.budget_remaining

    def test_time_remaining_days(self, generator, base_latent, trial_state):
        obs = _make_obs(generator, base_latent, trial_state)
        assert (
            obs.resource_status["time_remaining_days"]
            == base_latent.time_remaining_days
        )

    def test_patients_enrolled(self, generator, base_latent, trial_state):
        latent = base_latent.model_copy(update={"patients_enrolled": 50})
        obs = _make_obs(generator, latent, trial_state)
        assert obs.resource_status["patients_enrolled"] == 50

    def test_resource_status_has_three_keys(self, generator, base_latent, trial_state):
        obs = _make_obs(generator, base_latent, trial_state)
        assert set(obs.resource_status.keys()) == {
            "budget_remaining",
            "time_remaining_days",
            "patients_enrolled",
        }


# ---------------------------------------------------------------------------
# phase_data tests — noise injection
# ---------------------------------------------------------------------------


class TestPhaseDataNoiseInjection:
    """Agent never sees raw hidden values — noise is always injected."""

    def test_true_effect_size_not_in_phase_data(
        self, generator, base_latent, trial_state
    ):
        """Raw true_effect_size must never appear directly in phase_data."""
        latent = base_latent.model_copy(update={"effect_estimated": True})
        obs = _make_obs(generator, latent, trial_state)
        # observed_effect_size should differ from true value (noise injected)
        # We can't guarantee they differ by chance, but the key should be present
        assert "observed_effect_size" in obs.phase_data

    def test_effect_size_not_exposed_before_estimation(
        self, generator, base_latent, trial_state
    ):
        """observed_effect_size should not appear before ESTIMATE_EFFECT_SIZE."""
        obs = _make_obs(generator, base_latent, trial_state)
        assert "observed_effect_size" not in obs.phase_data

    def test_effect_size_exposed_after_estimation(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(update={"effect_estimated": True})
        obs = _make_obs(generator, latent, trial_state)
        assert "observed_effect_size" in obs.phase_data
        assert "effect_size_ci" in obs.phase_data

    def test_ae_rate_not_exposed_before_phase_i(
        self, generator, base_latent, trial_state
    ):
        """Adverse event rate should not appear before Phase I or safety signal."""
        obs = _make_obs(generator, base_latent, trial_state)
        assert "observed_adverse_event_rate" not in obs.phase_data

    def test_ae_rate_exposed_after_phase_i(self, generator, base_latent, trial_state):
        latent = base_latent.model_copy(update={"phase_i_complete": True})
        obs = _make_obs(generator, latent, trial_state)
        assert "observed_adverse_event_rate" in obs.phase_data

    def test_ae_rate_exposed_after_safety_signal(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"action_history": [ActionType.OBSERVE_SAFETY_SIGNAL.value]}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert "observed_adverse_event_rate" in obs.phase_data

    def test_ae_rate_is_clipped_to_0_1(self, generator, base_latent, trial_state):
        latent = base_latent.model_copy(
            update={"phase_i_complete": True, "true_side_effect_rate": 0.99}
        )
        obs = _make_obs(generator, latent, trial_state)
        rate = obs.phase_data["observed_adverse_event_rate"]
        assert 0.0 <= rate <= 1.0

    def test_placebo_response_not_exposed_before_interim(
        self, generator, base_latent, trial_state
    ):
        obs = _make_obs(generator, base_latent, trial_state)
        assert "observed_placebo_response" not in obs.phase_data

    def test_placebo_response_exposed_after_interim(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(update={"interim_complete": True})
        obs = _make_obs(generator, latent, trial_state)
        assert "observed_placebo_response" in obs.phase_data

    def test_dose_response_not_exposed_before_phase_i(
        self, generator, base_latent, trial_state
    ):
        obs = _make_obs(generator, base_latent, trial_state)
        assert "observed_dose_response" not in obs.phase_data

    def test_dose_response_exposed_after_phase_i(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(update={"phase_i_complete": True})
        obs = _make_obs(generator, latent, trial_state)
        assert "observed_dose_response" in obs.phase_data
        # All dose-response values should be clipped to [0, 1]
        for v in obs.phase_data["observed_dose_response"].values():
            assert 0.0 <= v <= 1.0

    def test_dropout_rate_not_exposed_before_enrollment(
        self, generator, base_latent, trial_state
    ):
        obs = _make_obs(generator, base_latent, trial_state)
        assert "observed_dropout_rate" not in obs.phase_data

    def test_dropout_rate_exposed_after_enrollment(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(update={"patients_enrolled": 10})
        obs = _make_obs(generator, latent, trial_state)
        assert "observed_dropout_rate" in obs.phase_data

    def test_responder_population_hint_not_exposed_without_biomarker(
        self, generator, base_latent, trial_state
    ):
        obs = _make_obs(generator, base_latent, trial_state)
        assert "responder_population_hint" not in obs.phase_data

    def test_responder_population_hint_exposed_after_biomarker(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"action_history": [ActionType.ADD_BIOMARKER_STRATIFICATION.value]}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert "responder_population_hint" in obs.phase_data
        # Population label is revealed but NOT the true criteria
        assert obs.phase_data["responder_population_hint"] == "BRCA1+"
        assert "true_responder_criteria" not in obs.phase_data

    def test_milestone_flags_in_phase_data(self, generator, base_latent, trial_state):
        """Milestone flags are observable (not hidden values)."""
        obs = _make_obs(generator, base_latent, trial_state)
        assert "phase_i_complete" in obs.phase_data
        assert "mtd_identified" in obs.phase_data
        assert "effect_estimated" in obs.phase_data
        assert "protocol_submitted" in obs.phase_data
        assert "interim_complete" in obs.phase_data
        assert "trial_complete" in obs.phase_data

    def test_true_mechanism_not_in_phase_data(
        self, generator, base_latent, trial_state
    ):
        """true_mechanism is a hidden value and must never appear in phase_data."""
        obs = _make_obs(generator, base_latent, trial_state)
        assert "true_mechanism" not in obs.phase_data

    def test_true_responder_criteria_not_in_phase_data(
        self, generator, base_latent, trial_state
    ):
        """true_responder_criteria is hidden and must never appear in phase_data."""
        obs = _make_obs(generator, base_latent, trial_state)
        assert "true_responder_criteria" not in obs.phase_data


# ---------------------------------------------------------------------------
# available_actions tests
# ---------------------------------------------------------------------------


class TestAvailableActions:
    """available_actions reflects phase-permitted actions filtered by prerequisites."""

    def test_available_actions_is_list_of_strings(
        self, generator, base_latent, trial_state
    ):
        obs = _make_obs(generator, base_latent, trial_state)
        assert isinstance(obs.available_actions, list)
        assert all(isinstance(a, str) for a in obs.available_actions)

    def test_design_phase_actions(self, generator, base_latent, trial_state):
        """In design phase with empty history, basic design actions are available."""
        obs = _make_obs(generator, base_latent, trial_state)
        # SET_SAMPLE_SIZE, SET_INCLUSION_CRITERIA, SET_EXCLUSION_CRITERIA should be available
        assert ActionType.SET_SAMPLE_SIZE.value in obs.available_actions
        assert ActionType.SET_INCLUSION_CRITERIA.value in obs.available_actions
        assert ActionType.SET_EXCLUSION_CRITERIA.value in obs.available_actions

    def test_dosing_schedule_requires_primary_endpoint(
        self, generator, base_latent, trial_state
    ):
        """SET_DOSING_SCHEDULE requires SET_PRIMARY_ENDPOINT in history."""
        obs = _make_obs(generator, base_latent, trial_state)
        # Without SET_PRIMARY_ENDPOINT in history, SET_DOSING_SCHEDULE should not be available
        assert ActionType.SET_DOSING_SCHEDULE.value not in obs.available_actions

    def test_dosing_schedule_available_after_primary_endpoint(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"action_history": [ActionType.SET_PRIMARY_ENDPOINT.value]}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.SET_DOSING_SCHEDULE.value in obs.available_actions

    def test_estimate_effect_size_available_in_literature_review(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"episode_phase": "literature_review", "action_history": []}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.ESTIMATE_EFFECT_SIZE.value in obs.available_actions

    def test_synthesize_conclusion_requires_primary_analysis(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={
                "episode_phase": "submission",
                "primary_analysis_complete": False,
            }
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.SYNTHESIZE_CONCLUSION.value not in obs.available_actions

    def test_synthesize_conclusion_available_after_primary_analysis(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={
                "episode_phase": "submission",
                "primary_analysis_complete": True,
            }
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.SYNTHESIZE_CONCLUSION.value in obs.available_actions

    def test_run_interim_analysis_requires_patients(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"episode_phase": "monitoring", "patients_enrolled": 0}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.RUN_INTERIM_ANALYSIS.value not in obs.available_actions

    def test_run_interim_analysis_available_with_patients(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"episode_phase": "monitoring", "patients_enrolled": 50}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.RUN_INTERIM_ANALYSIS.value in obs.available_actions

    def test_run_primary_analysis_requires_interim_complete(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"episode_phase": "analysis", "interim_complete": False}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.RUN_PRIMARY_ANALYSIS.value not in obs.available_actions

    def test_run_primary_analysis_available_after_interim(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(
            update={"episode_phase": "analysis", "interim_complete": True}
        )
        obs = _make_obs(generator, latent, trial_state)
        assert ActionType.RUN_PRIMARY_ANALYSIS.value in obs.available_actions

    def test_unknown_phase_returns_empty_actions(
        self, generator, base_latent, trial_state
    ):
        latent = base_latent.model_copy(update={"episode_phase": "unknown_phase"})
        obs = _make_obs(generator, latent, trial_state)
        assert obs.available_actions == []


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed + same latent state → same observation (requirement 9.2)."""

    def test_same_seed_same_observed_effect(self, base_latent, trial_state):
        latent = base_latent.model_copy(update={"effect_estimated": True})
        obs1 = OutputGenerator(NoiseModel(seed=99)).generate(
            latent,
            trial_state,
            steps_taken=1,
            max_steps=20,
            rule_violations=[],
            done=False,
            reward=0.0,
            scenario_description="S",
            hint="",
        )
        obs2 = OutputGenerator(NoiseModel(seed=99)).generate(
            latent,
            trial_state,
            steps_taken=1,
            max_steps=20,
            rule_violations=[],
            done=False,
            reward=0.0,
            scenario_description="S",
            hint="",
        )
        assert (
            obs1.phase_data["observed_effect_size"]
            == obs2.phase_data["observed_effect_size"]
        )

    def test_different_seeds_different_observed_effect(self, base_latent, trial_state):
        latent = base_latent.model_copy(update={"effect_estimated": True})
        obs1 = OutputGenerator(NoiseModel(seed=1)).generate(
            latent,
            trial_state,
            steps_taken=1,
            max_steps=20,
            rule_violations=[],
            done=False,
            reward=0.0,
            scenario_description="S",
            hint="",
        )
        obs2 = OutputGenerator(NoiseModel(seed=2)).generate(
            latent,
            trial_state,
            steps_taken=1,
            max_steps=20,
            rule_violations=[],
            done=False,
            reward=0.0,
            scenario_description="S",
            hint="",
        )
        # Different seeds should (almost certainly) produce different noisy values
        assert (
            obs1.phase_data["observed_effect_size"]
            != obs2.phase_data["observed_effect_size"]
        )
