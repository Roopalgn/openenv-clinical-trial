"""
Clinical Trial GRPO Training — V2 (Single-Step Evaluation)

Fixes from V1:
- Single-step evaluation (no multi-step noise from random actions)
- Observation-rich prompts (available_actions, phase, resources)
- Diverse prompts from different env seeds
- Proper reward variance for GRPO

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
SYSTEM_PROMPT = """You are a clinical trial designer.
Given the current trial state and available actions, choose the BEST next action.
Return exactly ONE valid JSON object:
{"action_type": "<from available_actions>", "parameters": {}, "justification": "why", "confidence": 0.8}
Do not add any text outside the JSON."""


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


def parse_action(text, available_actions=None):
    """Extract JSON action from model output."""
    candidates = []
    fenced = re.findall(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    candidates.append(text)

    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start:end + 1])
            at = str(parsed.get("action_type", "")).strip()
            params = parsed.get("parameters", {})
            if not isinstance(params, dict):
                params = {}
            if available_actions and at not in available_actions:
                at = available_actions[0]
            elif not at:
                at = "set_primary_endpoint"
            return {"action_type": at, "parameters": params}
        except Exception:
            continue

    if available_actions:
        return {"action_type": available_actions[0], "parameters": {}}
    return {"action_type": "set_primary_endpoint", "parameters": {}}


def build_prompt(obs):
    """Build an observation-rich prompt from env state."""
    available = obs.get("available_actions", ["set_primary_endpoint"])
    phase = obs.get("phase_data", {}).get("current_phase", "unknown")
    return (
        f"You are designing a clinical trial.\n\n"
        f"Scenario: {obs.get('scenario_description', '')}\n"
        f"Current phase: {phase}\n"
        f"Resources: {json.dumps(obs.get('resource_status', {}))}\n"
        f"Available actions: {available}\n"
        f"Steps taken: {obs.get('steps_taken', 0)}/{obs.get('max_steps', 100)}\n"
        f"Hint: {obs.get('hint', '')}\n\n"
        f"Choose ONE action from: {available}\n"
        f"Return ONLY JSON: "
        '{"action_type": "<from list above>", "parameters": {}, "justification": "...", "confidence": 0.8}'
    )


def extract_reward(result):
    """Extract total reward from step result."""
    reward = result.get("reward", 0.0)
    if isinstance(reward, dict):
        return float(sum(float(v) for v in reward.values()))
    return float(reward)


def single_step_reward(model_response, seed):
    """Score a single model output with one env step. No multi-step noise."""
    try:
        obs = env_reset(seed=seed)
        available = obs.get("available_actions", ["set_primary_endpoint"])
        action = parse_action(model_response, available)
        result = env_step(action["action_type"], action.get("parameters", {}))
        return extract_reward(result)
    except Exception as e:
        print(f"Reward error: {e}")
        return -2.0


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
    """Validate pipeline: test rewards are discriminative."""
    print("=== DRY RUN: Testing reward discrimination ===")
    os.makedirs("outputs/grpo_v2", exist_ok=True)

    results = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        obs = env_reset(seed=seed)
        available = obs.get("available_actions", ["set_primary_endpoint"])

        # Test 1: valid action (first available)
        valid_action = available[0]
        r1 = env_step(valid_action)
        reward_valid = extract_reward(r1)

        # Reset and test 2: invalid action
        obs = env_reset(seed=seed)
        r2 = env_step("synthesize_conclusion")  # almost always invalid early
        reward_invalid = extract_reward(r2)

        delta = reward_valid - reward_invalid
        results.append(delta)
        print(f"  Ep {ep+1}: valid={reward_valid:.3f} invalid={reward_invalid:.3f} delta={delta:.3f}")

    avg_delta = sum(results) / len(results)
    print(f"\nAvg reward delta (valid - invalid): {avg_delta:.3f}")
    if avg_delta > 0.5:
        print("✓ Rewards are discriminative. Ready for training.")
    else:
        print("✗ Reward discrimination too low. Check server deployment.")
    return avg_delta


def run_training(args):
    """Full GRPO training loop."""
    print("=== GRPO TRAINING (Single-Step) ===")

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

    # Reward function — single step per completion
    seed_counter = [0]

    def reward_func(completions, **kwargs):
        rewards = []
        for completion in completions:
            if isinstance(completion, list):
                text = completion[-1]["content"] if completion else ""
            else:
                text = str(completion)
            seed = args.seed + 10000 + seed_counter[0]
            seed_counter[0] += 1
            reward = single_step_reward(text, seed)
            rewards.append(reward)
        return rewards

    # Configure GRPO
    import torch
    from trl import GRPOConfig, GRPOTrainer

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = GRPOConfig(
        output_dir="checkpoints/grpo_v2",
        num_generations=args.num_generations,
        max_completion_length=256,
        temperature=0.7,
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
    start_time = datetime.now(timezone.utc)
    trainer.train()
    end_time = datetime.now(timezone.utc)

    # Save artifacts
    os.makedirs("outputs/grpo_v2", exist_ok=True)
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
        csv_path = "outputs/grpo_v2/reward_log.csv"
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
            "mean_reward": float(np.mean(rewards)),
            "final_reward": float(rewards[-1]),
            "max_reward": float(max(rewards)),
            "min_reward": float(min(rewards)),
            "slope": float(slope),
            "runtime_seconds": (end_time - start_time).total_seconds(),
            "completed_at": end_time.isoformat(),
        }
        with open("outputs/grpo_v2/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults: mean={summary['mean_reward']:.3f}, slope={slope:.4f}")
        print(f"Saved to outputs/grpo_v2/")
    else:
        print("No reward rows captured.")

    # Save model
    model.save_pretrained("checkpoints/grpo_v2/final")
    tokenizer.save_pretrained("checkpoints/grpo_v2/final")
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
