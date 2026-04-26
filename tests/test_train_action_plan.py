"""Tests for full-action-plan GRPO reward rollouts."""

from __future__ import annotations

import json

from server.environment import Environment
from train import (
    INVALID_SEQUENCE_REWARD,
    PLAN_PARSE_FAILURE_REWARD,
    parse_action_plan,
    rollout_action_plan_reward,
)


def _plan(actions: list[dict]) -> str:
    return json.dumps({"actions": actions})


def test_legacy_two_knob_design_is_not_accepted_as_full_plan() -> None:
    text = '{"sample_size": 60, "add_biomarker": false}'
    assert parse_action_plan(text) is None


def test_plan_parser_requires_action_list() -> None:
    assert parse_action_plan("not json") is None
    assert parse_action_plan('{"actions": []}') is None


def test_invalid_first_action_scores_parse_failure_band() -> None:
    env = Environment()
    text = _plan([{"action_type": "synthesize_conclusion", "parameters": {}}])

    reward = rollout_action_plan_reward(env, text, seed=42, max_steps=12)

    assert reward == INVALID_SEQUENCE_REWARD


def test_short_incomplete_plan_stays_near_low_reward_band() -> None:
    env = Environment()
    text = _plan(
        [
            {
                "action_type": "set_primary_endpoint",
                "parameters": {"endpoint": "overall_survival"},
            }
        ]
    )

    reward = rollout_action_plan_reward(env, text, seed=42, max_steps=12)

    assert PLAN_PARSE_FAILURE_REWARD < reward < 0.0


def test_complete_valid_plan_reaches_positive_reward_band() -> None:
    env = Environment()
    text = _plan(
        [
            {
                "action_type": "set_primary_endpoint",
                "parameters": {"endpoint": "overall_survival"},
            },
            {"action_type": "set_sample_size", "parameters": {"sample_size": 260}},
            {"action_type": "set_inclusion_criteria", "parameters": {}},
            {"action_type": "set_dosing_schedule", "parameters": {}},
            {"action_type": "set_control_arm", "parameters": {}},
            {"action_type": "enroll_patients", "parameters": {"n_patients": 260}},
            {"action_type": "run_dose_escalation", "parameters": {}},
            {"action_type": "run_interim_analysis", "parameters": {}},
            {"action_type": "run_primary_analysis", "parameters": {}},
        ]
    )

    reward = rollout_action_plan_reward(env, text, seed=42, max_steps=12)

    assert reward > 5.0
