"""
Clinical Trial GRPO Training — V3 (Full-Episode Evaluation)

Fixes from V2:
- Full-episode evaluation (agent plans full action sequence, not single-step)
- Cumulative reward over complete trial episode gives GRPO the full [-3, +15] range
- Diverse prompts from different env seeds with observation-rich context
- Better generation settings for parseability (lower temperature, longer completions)

Usage on Colab:
  1. Upload this file or clone the repo
  2. Run: !python train_colab_v2.py --dry-run   (validate pipeline)
  3. Run: !python train_colab_v2.py              (real training)
"""

import argparse
import csv
import json
import os
import random
import re
from datetime import datetime, timezone

import requests

# === CONFIG ===
ENV_URL = "https://roopalgn-openenv-clinical-trial.hf.space"
SYSTEM_PROMPT = """You are an expert clinical trial designer.
Given the current trial state, plan the COMPLETE sequence of actions to design a successful trial.
Return a JSON object with an "actions" list. Each action must have action_type, parameters, justification, and confidence.

Example response:
{"actions": [
  {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "overall_survival"}, "justification": "OS is gold standard", "confidence": 0.9},
  {"action_type": "set_sample_size", "parameters": {"sample_size": 240}, "justification": "Powered for 0.80 at expected effect", "confidence": 0.85},
  {"action_type": "set_inclusion_criteria", "parameters": {"criteria": "adults 18-75"}, "justification": "Standard population", "confidence": 0.8},
  {"action_type": "set_dosing_schedule", "parameters": {"schedule": "daily"}, "justification": "Standard dosing", "confidence": 0.8},
  {"action_type": "set_control_arm", "parameters": {"control": "placebo"}, "justification": "RCT standard", "confidence": 0.9},
  {"action_type": "enroll_patients", "parameters": {"n_patients": 240}, "justification": "Full enrollment", "confidence": 0.85},
  {"action_type": "run_dose_escalation", "parameters": {}, "justification": "Phase I safety", "confidence": 0.8},
  {"action_type": "run_interim_analysis", "parameters": {}, "justification": "Check futility/efficacy", "confidence": 0.75},
  {"action_type": "run_primary_analysis", "parameters": {}, "justification": "Final statistical test", "confidence": 0.8},
  {"action_type": "synthesize_conclusion", "parameters": {}, "justification": "Complete trial report", "confidence": 0.85}
]}

You MUST include ALL phases: design -> enrollment -> analysis -> conclusion.
Aim for 8-12 actions. Return ONLY the JSON object, no other text."""

# Maximum steps to execute per episode during reward evaluation
MAX_EPISODE_STEPS = 20


def env_reset(seed=None):
    payload = {"seed": seed} if seed is not None else {}
    resp = requests.post(f"{ENV_URL}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type, parameters=None, justification="", confidence=0.5):
    payload = {
        "action_type": action_type,
        "parameters": parameters or {},
        "justification": justification,
        "confidence": confidence,
    }
    resp = requests.post(f"{ENV_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_action_plan(text):
    """Extract a list of actions from model output.
    
    Handles:
    - {"actions": [...]} format (full plan)
    - Single action {"action_type": ...} format  
    - Fenced code blocks ```json ... ```
    - Partial/malformed JSON recovery
    """
    # Try fenced code blocks first
    candidates = []
    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(text)
    
    for candidate in candidates:
        # Find JSON objects
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        
        json_str = candidate[start:end + 1]
        
        # Try standard JSON parse
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            # Try ast.literal_eval for Python-style dicts
            try:
                import ast
                parsed = ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                continue
        
        if not isinstance(parsed, dict):
            continue
            
        # Full plan format: {"actions": [...]}
        if "actions" in parsed and isinstance(parsed["actions"], list):
            actions = []
            for a in parsed["actions"]:
                if isinstance(a, dict) and "action_type" in a:
                    actions.append(_normalize_action(a))
            if actions:
                return actions
        
        # Single action format: {"action_type": "..."}
        if "action_type" in parsed:
            return [_normalize_action(parsed)]
    
    # Last resort: try to find individual action objects
    action_pattern = re.findall(r'\{[^{}]*"action_type"\s*:\s*"[^"]+?"[^{}]*\}', text)
    if action_pattern:
        actions = []
        for match in action_pattern:
            try:
                parsed = json.loads(match)
                actions.append(_normalize_action(parsed))
            except json.JSONDecodeError:
                continue
        if actions:
            return actions
    
    return []  # Parse failure


def _normalize_action(action_dict):
    """Normalize an action dict to have required fields."""
    at = str(action_dict.get("action_type", "")).strip()
    params = action_dict.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    justification = str(action_dict.get("justification", ""))
    confidence = float(action_dict.get("confidence", 0.7))
    confidence = max(0.0, min(1.0, confidence))
    
    # Handle n_patients / sample_size normalization
    if at == "enroll_patients" and "n_patients" not in params:
        if "sample_size" in params:
            params["n_patients"] = params.pop("sample_size")
        else:
            params["n_patients"] = 240  # default
    if at == "set_sample_size" and "sample_size" not in params:
        if "n_patients" in params:
            params["sample_size"] = params.pop("n_patients")
        else:
            params["sample_size"] = 240
    
    return {
        "action_type": at,
        "parameters": params,
        "justification": justification,
        "confidence": confidence,
    }


def extract_reward(result):
    """Extract total reward from step result."""
    reward = result.get("reward", 0.0)
    if isinstance(reward, dict):
        # Use canonical reward keys, not raw sum
        canonical_keys = [
            "r_validity", "r_ordering", "r_info_gain", "r_efficiency",
            "r_novelty", "r_penalty", "r_terminal_success", "r_terminal_calibration"
        ]
        total = 0.0
        for key in canonical_keys:
            if key in reward:
                total += float(reward[key])
        return total if total != 0.0 else float(sum(float(v) for v in reward.values()))
    return float(reward)


def full_episode_reward(model_response, seed):
    """Score a model's full action plan by executing it against the environment.
    
    This gives GRPO access to the full reward range [-3, +15]:
    - Parse failure → -3.0
    - 1-2 valid actions → ~-1 to +1 (mostly ordering + validity)
    - Partial plan (5-8 actions) → ~+2 to +5 (some milestones)
    - Complete plan (12-15 actions) → ~+5 to +15 (milestones + terminal)
    
    This is the key change from V2: single-step gave [-2.5, +0.25] range,
    which had almost no gradient once the model learned valid JSON.
    """
    try:
        # Parse the model's action plan
        actions = parse_action_plan(model_response)
        
        if not actions:
            return -3.0  # Parse failure
        
        # Reset the environment
        obs = env_reset(seed=seed)
        
        # Execute each action in order, accumulating reward
        total_reward = 0.0
        steps_taken = 0
        
        for action in actions[:MAX_EPISODE_STEPS]:
            try:
                result = env_step(
                    action["action_type"],
                    action.get("parameters", {}),
                    action.get("justification", ""),
                    action.get("confidence", 0.7),
                )
                step_reward = extract_reward(result)
                total_reward += step_reward
                steps_taken += 1
                
                # Check if episode ended
                obs_data = result.get("observation", result)
                if isinstance(obs_data, dict) and obs_data.get("done", False):
                    break
                    
            except requests.exceptions.HTTPError as e:
                # Action was rejected by the server — penalty but continue
                total_reward += -1.0
                steps_taken += 1
                continue
            except Exception as e:
                print(f"  Step error: {e}")
                total_reward += -0.5
                steps_taken += 1
                continue
        
        # Bonus for plan completeness: longer valid plans that reach later phases
        # This helps differentiate "3 design actions" from "15-action full plan"
        if steps_taken == 0:
            return -3.0
        
        return total_reward
        
    except Exception as e:
        print(f"Episode reward error: {e}")
        return -3.0


def build_prompt(obs):
    """Build an observation-rich prompt from env state."""
    available = obs.get("available_actions", ["set_primary_endpoint"])
    phase = obs.get("phase_data", {}).get("current_phase", "unknown")
    return (
        f"You are designing a clinical trial.\n\n"
        f"Scenario: {obs.get('scenario_description', '')}\n"
        f"Current phase: {phase}\n"
        f"Resources: {json.dumps(obs.get('resource_status', {}))}\n"
        f"Available starting actions: {available}\n"
        f"Max steps: {obs.get('max_steps', 100)}\n"
        f"Hint: {obs.get('hint', '')}\n\n"
        f"Plan the COMPLETE sequence of actions for a successful trial.\n"
        f"Include ALL phases: design → enrollment → analysis → conclusion.\n"
        f"Return ONLY a JSON object with an 'actions' list (12-15 actions)."
    )


def generate_prompts(n_prompts, base_seed=42):
    """Generate diverse prompts by resetting env with different seeds."""
    prompts = []
    seeds = []
    for i in range(n_prompts):
        seed = base_seed + i
        try:
            obs = env_reset(seed=seed)
            prompt_text = build_prompt(obs)
            prompts.append(prompt_text)
            seeds.append(seed)
        except Exception as e:
            print(f"Prompt generation failed for seed {seed}: {e}")
    return prompts, seeds


def run_dry_run(n_episodes=5, base_seed=42):
    """Validate pipeline: test that full-episode rewards are discriminative."""
    print("=== DRY RUN: Testing full-episode reward discrimination ===")
    os.makedirs("outputs/grpo_v3", exist_ok=True)

    results = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        
        # Test 1: Good plan (full sequence)
        good_plan = json.dumps({"actions": [
            {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}, "justification": "gold standard", "confidence": 0.9},
            {"action_type": "set_sample_size", "parameters": {"sample_size": 240}, "justification": "adequate power", "confidence": 0.85},
            {"action_type": "set_inclusion_criteria", "parameters": {"criteria": "adults"}, "justification": "standard", "confidence": 0.8},
            {"action_type": "set_exclusion_criteria", "parameters": {"criteria": "prior tx"}, "justification": "clean", "confidence": 0.8},
            {"action_type": "set_dosing_schedule", "parameters": {"schedule": "daily"}, "justification": "standard", "confidence": 0.8},
            {"action_type": "set_control_arm", "parameters": {"control": "placebo"}, "justification": "RCT", "confidence": 0.9},
            {"action_type": "set_randomization_ratio", "parameters": {"ratio": "1:1"}, "justification": "equal", "confidence": 0.9},
            {"action_type": "set_blinding", "parameters": {"blinding": "double"}, "justification": "bias control", "confidence": 0.9},
            {"action_type": "enroll_patients", "parameters": {"n_patients": 240}, "justification": "full enrollment", "confidence": 0.85},
            {"action_type": "run_dose_escalation", "parameters": {}, "justification": "Phase I", "confidence": 0.8},
            {"action_type": "estimate_effect_size", "parameters": {}, "justification": "quantify", "confidence": 0.7},
            {"action_type": "observe_safety_signal", "parameters": {}, "justification": "safety", "confidence": 0.8},
            {"action_type": "run_interim_analysis", "parameters": {}, "justification": "interim check", "confidence": 0.75},
            {"action_type": "run_primary_analysis", "parameters": {}, "justification": "final test", "confidence": 0.8},
            {"action_type": "synthesize_conclusion", "parameters": {}, "justification": "complete", "confidence": 0.85},
        ]})
        reward_good = full_episode_reward(good_plan, seed)
        
        # Test 2: Minimal plan (just 2 design actions)
        minimal_plan = json.dumps({"actions": [
            {"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}, "justification": "test", "confidence": 0.5},
            {"action_type": "set_sample_size", "parameters": {"sample_size": 50}, "justification": "test", "confidence": 0.5},
        ]})
        reward_minimal = full_episode_reward(minimal_plan, seed)
        
        # Test 3: Parse failure
        reward_fail = full_episode_reward("I don't know how to design a trial", seed)
        
        delta = reward_good - reward_minimal
        print(f"  Ep {ep+1}: good={reward_good:.3f} minimal={reward_minimal:.3f} fail={reward_fail:.3f} delta={delta:.3f}")
        results.append(delta)

    avg_delta = sum(results) / len(results)
    print(f"\nAvg reward delta (good - minimal): {avg_delta:.3f}")
    if avg_delta > 2.0:
        print("✓ Rewards are highly discriminative. Ready for training.")
    elif avg_delta > 0.5:
        print("~ Rewards are moderately discriminative.")
    else:
        print("✗ Reward discrimination too low. Check server deployment.")
    return avg_delta


def run_training(args):
    """Full GRPO training loop with full-episode evaluation."""
    print("=== GRPO TRAINING V3 (Full-Episode) ===")

    # Generate diverse prompts
    print(f"Generating {args.episodes} diverse prompts...")
    prompts, seeds = generate_prompts(args.episodes, args.seed)
    print(f"Generated {len(prompts)} prompts from {len(set(seeds))} unique seeds")

    # Load model
    from unsloth import FastLanguageModel

    MODEL_PRESETS = {
        "1.5b": {"model_name": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", "lora_r": 8, "seq_len": 2048},
        "3b": {"model_name": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit", "lora_r": 16, "seq_len": 3072},
        "7b": {"model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "lora_r": 32, "seq_len": 4096},
    }
    preset = MODEL_PRESETS[args.model_size]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=preset["model_name"],
        max_seq_length=preset["seq_len"],
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=preset["lora_r"],
        lora_alpha=preset["lora_r"] * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded: {preset['model_name']}")
    model.print_trainable_parameters()

    # Build dataset with chat-formatted prompts
    from datasets import Dataset

    chat_prompts = []
    for prompt_text in prompts:
        chat_prompts.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt_text},
            ]
        })
    train_dataset = Dataset.from_list(chat_prompts)

    # Reward function — full episode per completion
    seed_counter = [0]

    def reward_func(completions, **kwargs):
        rewards = []
        for completion in completions:
            # Extract text from various TRL completion formats
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            elif isinstance(completion, dict):
                text = completion.get("content", str(completion))
            else:
                text = str(completion)
            
            seed = args.seed + 10000 + seed_counter[0]
            seed_counter[0] += 1
            reward = full_episode_reward(text, seed)
            rewards.append(reward)
        return rewards

    # Configure GRPO — V3 settings tuned for full-episode evaluation
    import torch
    from trl import GRPOConfig, GRPOTrainer

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = GRPOConfig(
        output_dir="checkpoints/grpo_v3",
        num_generations=args.num_generations,
        # V3: Longer completions needed for full action plans (15 actions × ~50 tokens)
        max_completion_length=512,
        # V3: Lower temperature for more structured/parseable output
        temperature=0.5,
        learning_rate=5e-6,
        num_train_epochs=1,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        max_steps=args.episodes,
        warmup_steps=2,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=1,
        save_steps=max(1, args.episodes // 4),
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        reward_funcs=[reward_func],
    )

    print(f"Training: {args.episodes} steps, {args.num_generations} generations/step")
    print(f"Full-episode evaluation with up to {MAX_EPISODE_STEPS} steps per completion")
    start_time = datetime.now(timezone.utc)
    trainer.train()
    end_time = datetime.now(timezone.utc)

    # Save artifacts
    os.makedirs("outputs/grpo_v3", exist_ok=True)
    reward_rows = []
    for idx, log in enumerate(trainer.state.log_history, start=1):
        if "reward" not in log:
            continue
        reward_rows.append({
            "step": int(log.get("step", idx)),
            "reward": float(log.get("reward", 0.0)),
            "reward_std": float(log.get("reward_std", 0.0)),
            "loss": float(log.get("loss", log.get("training_loss", 0.0)) or 0.0),
        })

    if reward_rows:
        csv_path = "outputs/grpo_v3/reward_log.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "reward", "reward_std", "loss"])
            writer.writeheader()
            writer.writerows(reward_rows)

        rewards = [r["reward"] for r in reward_rows]
        import numpy as np
        if len(rewards) > 1:
            z = np.polyfit(range(len(rewards)), rewards, 1)
            slope = z[0]
        else:
            slope = 0.0

        summary = {
            "model_size": args.model_size,
            "episodes": args.episodes,
            "num_generations": args.num_generations,
            "evaluation_mode": "full_episode",
            "max_episode_steps": MAX_EPISODE_STEPS,
            "mean_reward": float(np.mean(rewards)),
            "final_reward": float(rewards[-1]),
            "max_reward": float(max(rewards)),
            "min_reward": float(min(rewards)),
            "slope": float(slope),
            "runtime_seconds": (end_time - start_time).total_seconds(),
            "completed_at": end_time.isoformat(),
        }
        with open("outputs/grpo_v3/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults: mean={summary['mean_reward']:.3f}, slope={slope:.4f}")
        print(f"Saved to outputs/grpo_v3/")
    else:
        print("No reward rows captured.")

    # Save model
    model.save_pretrained("checkpoints/grpo_v3/final")
    tokenizer.save_pretrained("checkpoints/grpo_v3/final")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline only")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-size", choices=["1.5b", "3b", "7b"], default="1.5b")
    parser.add_argument("--num-generations", type=int, default=8)
    args = parser.parse_args()

    # Test connection
    print(f"Testing connection to {ENV_URL}...")
    try:
        r = requests.get(f"{ENV_URL}/ping", timeout=10)
        print(f"Ping: {r.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect: {e}")
        exit(1)

    if args.dry_run:
        run_dry_run(n_episodes=5, base_seed=args.seed)
    else:
        run_training(args)
