"""
Quick reward range verification — shows the reward tiers that GRPO will see.

Run with: python debug_reward.py
"""

import json
from server.environment import Environment
from train import rollout_action_plan_reward

def _plan(actions):
    return json.dumps({"actions": actions})


def main():
    env = Environment()
    
    print("=" * 60)
    print("REWARD RANGE VERIFICATION (V4)")
    print("=" * 60)
    
    # Tier 1: Parse failure
    r_fail = rollout_action_plan_reward(env, "not json at all", seed=42, max_steps=15)
    print(f"\n1. Parse failure:                    {r_fail:+.3f}")
    
    # Tier 2: Single valid action
    r_single = rollout_action_plan_reward(env, _plan([
        {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}}
    ]), seed=42, max_steps=15)
    print(f"2. Single valid action:              {r_single:+.3f}")
    
    # Tier 3: Few design actions (3 steps)
    r_design = rollout_action_plan_reward(env, _plan([
        {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}},
        {"action_type": "set_sample_size", "parameters": {"sample_size": 240}},
        {"action_type": "set_inclusion_criteria", "parameters": {}},
    ]), seed=42, max_steps=15)
    print(f"3. Three design actions:             {r_design:+.3f}")
    
    # Tier 4: Design + enrollment (6 steps)
    r_enroll = rollout_action_plan_reward(env, _plan([
        {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}},
        {"action_type": "set_sample_size", "parameters": {"sample_size": 240}},
        {"action_type": "set_inclusion_criteria", "parameters": {}},
        {"action_type": "set_dosing_schedule", "parameters": {}},
        {"action_type": "set_control_arm", "parameters": {}},
        {"action_type": "enroll_patients", "parameters": {"n_patients": 240}},
    ]), seed=42, max_steps=15)
    print(f"4. Design + enrollment (6 steps):    {r_enroll:+.3f}")
    
    # Tier 5: Through dose escalation (8 steps)
    r_phase1 = rollout_action_plan_reward(env, _plan([
        {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}},
        {"action_type": "set_sample_size", "parameters": {"sample_size": 240}},
        {"action_type": "set_inclusion_criteria", "parameters": {}},
        {"action_type": "set_dosing_schedule", "parameters": {}},
        {"action_type": "set_control_arm", "parameters": {}},
        {"action_type": "enroll_patients", "parameters": {"n_patients": 240}},
        {"action_type": "run_dose_escalation", "parameters": {}},
        {"action_type": "estimate_effect_size", "parameters": {}},
    ]), seed=42, max_steps=15)
    print(f"5. Through Phase I + estimation:     {r_phase1:+.3f}")
    
    # Tier 6: Through interim (10 steps, no terminal)
    r_interim = rollout_action_plan_reward(env, _plan([
        {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}},
        {"action_type": "set_sample_size", "parameters": {"sample_size": 240}},
        {"action_type": "set_inclusion_criteria", "parameters": {}},
        {"action_type": "set_dosing_schedule", "parameters": {}},
        {"action_type": "set_control_arm", "parameters": {}},
        {"action_type": "enroll_patients", "parameters": {"n_patients": 240}},
        {"action_type": "run_dose_escalation", "parameters": {}},
        {"action_type": "estimate_effect_size", "parameters": {}},
        {"action_type": "run_interim_analysis", "parameters": {}},
        {"action_type": "run_primary_analysis", "parameters": {}},
    ]), seed=42, max_steps=15)
    print(f"6. Through primary analysis:         {r_interim:+.3f}")
    
    # Tier 7: Full completion (11 steps)
    r_full = rollout_action_plan_reward(env, _plan([
        {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}},
        {"action_type": "set_sample_size", "parameters": {"sample_size": 260}},
        {"action_type": "set_inclusion_criteria", "parameters": {}},
        {"action_type": "set_dosing_schedule", "parameters": {}},
        {"action_type": "set_control_arm", "parameters": {}},
        {"action_type": "enroll_patients", "parameters": {"n_patients": 260}},
        {"action_type": "run_dose_escalation", "parameters": {}},
        {"action_type": "estimate_effect_size", "parameters": {}},
        {"action_type": "run_interim_analysis", "parameters": {}},
        {"action_type": "run_primary_analysis", "parameters": {}},
        {"action_type": "synthesize_conclusion", "parameters": {}},
    ]), seed=42, max_steps=15)
    print(f"7. Full completion (11 steps):       {r_full:+.3f}")
    
    print(f"\n{'=' * 60}")
    print("REWARD SEPARATION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Parse fail -> Single valid:   {r_single - r_fail:+.3f}")
    print(f"  Single -> 3 design:           {r_design - r_single:+.3f}")
    print(f"  3 design -> + enrollment:     {r_enroll - r_design:+.3f}")
    print(f"  + enrollment -> + Phase I:    {r_phase1 - r_enroll:+.3f}")
    print(f"  + Phase I -> + primary:       {r_interim - r_phase1:+.3f}")
    print(f"  + primary -> full completion: {r_full - r_interim:+.3f}")
    print(f"\n  Total range: [{r_fail:.1f}, {r_full:.1f}] = {r_full - r_fail:.1f} points")
    print(f"\n  GRPO needs: reward_std > 0 and increasing mean")
    print(f"  This range gives GRPO {r_full - r_fail:.1f}x the signal of single-step")


if __name__ == "__main__":
    main()
