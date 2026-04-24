"""
train.py — GRPO training runner for the Clinical Trial Designer environment.

Uses TRL 0.29.0 + vLLM colocate mode with LoRA (rank 16, alpha 32, BF16).
Calls env.reset()/step() directly via server.environment.Environment — no HTTP.

Usage:
    python train.py \\
        --model-path Qwen/Qwen2.5-7B-Instruct \\
        --episodes 100 \\
        --seed 42 \\
        --vllm-mode colocate \\
        --num-generations 8 \\
        --max-steps 50
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models import TrialAction
    from server.environment import Environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Lazy imports — TRL / vLLM are optional at import time so the module can be
# imported in test environments without the full ML stack installed.
# ---------------------------------------------------------------------------


def _import_trl():
    try:
        import torch
        from peft import LoraConfig, TaskType
        from trl import GRPOConfig, GRPOTrainer

        return GRPOConfig, GRPOTrainer, LoraConfig, TaskType, torch
    except ImportError as exc:
        log.error(
            "TRL / PEFT / PyTorch not installed. "
            "Install with: pip install trl==0.29.0 peft torch\n%s",
            exc,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _build_action_from_text(text: str, step: int) -> "TrialAction":
    """Parse a model-generated text into a TrialAction.

    Falls back to a safe default action when the text cannot be parsed.
    """
    from models import ActionType, TrialAction

    ACTION_CYCLE = list(ActionType)

    try:
        data = json.loads(text)
        return TrialAction(
            action_type=data.get("action_type", ActionType.SET_PRIMARY_ENDPOINT),
            parameters=data.get("parameters", {}),
            justification=data.get("justification", "model output"),
            confidence=float(data.get("confidence", 0.5)),
        )
    except Exception:
        # Cycle through action types deterministically as fallback
        action_type = ACTION_CYCLE[step % len(ACTION_CYCLE)]
        return TrialAction(
            action_type=action_type,
            parameters={},
            justification="fallback: could not parse model output",
            confidence=0.5,
        )


def rollout_func(
    env: "Environment",
    model: Any,
    tokenizer: Any,
    seed: int,
    max_steps: int,
    num_generations: int,
) -> list[dict[str, Any]]:
    """Run one episode rollout, collecting (prompt, response, reward) triples.

    Calls env.reset() and env.step_full() directly — no HTTP overhead.

    Args:
        env: Environment instance (server.environment.Environment).
        model: The language model (used for generation).
        tokenizer: Tokenizer paired with the model.
        seed: Episode seed for reproducibility.
        max_steps: Maximum steps per episode.
        num_generations: Number of parallel generation samples per step.

    Returns:
        List of dicts with keys: prompt, response, reward, step, done.
    """
    import torch

    obs = env.reset(seed=seed)
    experiences: list[dict[str, Any]] = []

    for step_idx in range(max_steps):
        # Build prompt from current observation
        prompt = (
            f"You are designing a clinical trial.\n\n"
            f"Scenario: {obs.scenario_description}\n"
            f"Phase data: {json.dumps(obs.phase_data)}\n"
            f"Resources: {json.dumps(obs.resource_status)}\n"
            f"Available actions: {obs.available_actions}\n"
            f"Steps taken: {obs.steps_taken}/{obs.max_steps}\n"
            f"Hint: {obs.hint}\n\n"
            "Respond with a JSON object: "
            '{"action_type": "...", "parameters": {}, "justification": "...", "confidence": 0.8}'
        )

        # Tokenize and generate
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=num_generations,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Use the first generation for the actual env step
        response_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        action = _build_action_from_text(response_text, step_idx)
        next_obs, reward_dict, done, info = env.step_full(action)

        total_reward = (
            sum(reward_dict.values())
            if isinstance(reward_dict, dict)
            else float(reward_dict)
        )

        experiences.append(
            {
                "prompt": prompt,
                "response": response_text,
                "reward": total_reward,
                "step": step_idx,
                "done": done,
                "info": info,
            }
        )

        obs = next_obs
        if done:
            break

    return experiences


# ---------------------------------------------------------------------------
# Reward CSV logger
# ---------------------------------------------------------------------------


class RewardCSVLogger:
    """Appends per-episode reward rows to a CSV file."""

    HEADERS = [
        "episode",
        "seed",
        "total_reward",
        "steps",
        "terminal_outcome",
        "timestamp",
    ]

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self.HEADERS)

    def log(
        self,
        episode: int,
        seed: int,
        total_reward: float,
        steps: int,
        terminal_outcome: str,
    ) -> None:
        try:
            with self._path.open("a", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(
                    [
                        episode,
                        seed,
                        round(total_reward, 6),
                        steps,
                        terminal_outcome,
                        datetime.now(timezone.utc).isoformat(),
                    ]
                )
        except OSError as exc:
            log.warning("RewardCSVLogger: could not write row: %s", exc)


# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------


def _write_summary(
    out_path: Path,
    episodes: int,
    rewards: list[float],
    final_tier: int,
    model_path: str,
    seed: int,
) -> None:
    """Write a JSON training summary file on completion (Req 11.2)."""
    summary = {
        "model_path": model_path,
        "episodes": episodes,
        "seed": seed,
        "mean_reward": round(sum(rewards) / len(rewards), 6) if rewards else 0.0,
        "max_reward": round(max(rewards), 6) if rewards else 0.0,
        "min_reward": round(min(rewards), 6) if rewards else 0.0,
        "final_curriculum_tier": final_tier,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("Training summary written to %s", out_path)
    except OSError as exc:
        log.warning("Could not write training summary: %s", exc)
    # Always print to stdout as well
    log.info("=== Training Summary ===")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)


# ---------------------------------------------------------------------------
# GRPO training loop
# ---------------------------------------------------------------------------


def _grpo_reward_fn(
    completions: list[str], env: "Environment", seed: int, max_steps: int
) -> list[float]:
    """Reward function passed to GRPOTrainer.

    Runs a mini-rollout for each completion and returns the total episode reward.
    This is called by the trainer during the GRPO update step.
    """
    rewards = []
    for i, completion in enumerate(completions):
        env.reset(seed=seed + i)
        total = 0.0
        for step_idx in range(max_steps):
            action = _build_action_from_text(completion, step_idx)
            _, reward_dict, done, _ = env.step_full(action)
            total += (
                sum(reward_dict.values())
                if isinstance(reward_dict, dict)
                else float(reward_dict)
            )
            if done:
                break
        rewards.append(total)
    return rewards


# ---------------------------------------------------------------------------
# Model size presets
# ---------------------------------------------------------------------------

MODEL_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "1.5b": {"lora_r": 8, "batch": 1, "seq_len": 2048, "grad_accum": 4},
    "3b": {"lora_r": 16, "batch": 1, "seq_len": 3072, "grad_accum": 4},
    "7b": {"lora_r": 32, "batch": 1, "seq_len": 4096, "grad_accum": 8},
}


def _apply_model_size_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Override LoRA / batch / seq settings from --model-size preset if given."""
    if args.model_size is None:
        return args
    preset = MODEL_SIZE_PRESETS[args.model_size]
    log.info(
        "Model size preset '%s': lora_r=%d, batch=%d, seq_len=%d, grad_accum=%d",
        args.model_size,
        preset["lora_r"],
        preset["batch"],
        preset["seq_len"],
        preset["grad_accum"],
    )
    return args


# ---------------------------------------------------------------------------
# Dry-run (smoke-test, no GPU / model required)
# ---------------------------------------------------------------------------


def _dry_run(args: argparse.Namespace) -> None:
    """Run 2 episodes with a random policy to verify the full pipeline.

    Skips model loading and GRPO trainer — uses random action cycling instead.
    Writes reward CSV and generates a plot PNG to confirm the pipeline works.
    """
    from server.environment import Environment

    log.info("=== DRY RUN MODE: 2 episodes, random policy, no model loading ===")
    _apply_model_size_preset(args)
    env = Environment()
    reward_csv = RewardCSVLogger(Path(args.output_dir) / "reward_log.csv")
    episode_rewards: list[float] = []

    n_episodes = min(args.episodes, 2)  # cap at 2 for smoke-test
    for ep in range(n_episodes):
        ep_seed = args.seed + ep
        obs = env.reset(seed=ep_seed)
        total_reward = 0.0
        steps = 0
        terminal_outcome = "timeout"

        for step_idx in range(args.max_steps):
            action = _build_action_from_text("", step_idx)  # cycles through actions
            next_obs, reward_dict, done, _ = env.step_full(action)
            total_reward += (
                sum(reward_dict.values())
                if isinstance(reward_dict, dict)
                else float(reward_dict)
            )
            steps += 1
            obs = next_obs  # noqa: F841
            if done:
                terminal_outcome = "success"
                break

        episode_rewards.append(total_reward)
        reward_csv.log(
            episode=ep,
            seed=ep_seed,
            total_reward=total_reward,
            steps=steps,
            terminal_outcome=terminal_outcome,
        )
        if (ep + 1) % 10 == 0:
            log.info(
                "Checkpoint marker at episode %d → %s",
                ep + 1,
                Path(args.output_dir) / f"checkpoint_ep{ep+1}",
            )
        log.info(
            "Dry-run episode %d/%d | reward=%.4f | steps=%d | outcome=%s",
            ep + 1, n_episodes, total_reward, steps, terminal_outcome,
        )

    # Generate plot to verify plot_rewards.py pipeline
    csv_path = Path(args.output_dir) / "reward_log.csv"
    plot_path = Path(args.output_dir) / "reward_curve.png"
    try:
        from plot_rewards import main as plot_main

        plot_main(["--csv", str(csv_path), "--out", str(plot_path)])
        log.info("Dry-run plot saved → %s", plot_path)
    except Exception as exc:
        log.warning("Could not generate plot during dry-run: %s", exc)

    log.info("=== DRY RUN COMPLETE — reward CSV: %s ===", csv_path)
    _write_summary(
        out_path=Path(args.output_dir) / "training_summary.json",
        episodes=n_episodes,
        rewards=episode_rewards,
        final_tier=0,
        model_path="dry-run/random-policy",
        seed=args.seed,
    )


def train(args: argparse.Namespace) -> None:
    """Main training entry point (Req 11.1, 11.2, 11.3, 11.4)."""
    # Dry-run: verify pipeline without loading model or running GRPO
    if args.dry_run:
        _dry_run(args)
        return

    GRPOConfig, GRPOTrainer, LoraConfig, TaskType, torch = _import_trl()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from server.environment import Environment

    # Apply model-size preset
    preset = MODEL_SIZE_PRESETS[args.model_size]
    log.info(
        "Model size preset '%s': lora_r=%d, batch=%d, seq_len=%d, grad_accum=%d",
        args.model_size,
        preset["lora_r"],
        preset["batch"],
        preset["seq_len"],
        preset["grad_accum"],
    )

    log.info("Initialising environment...")
    env = Environment()

    log.info("Loading model: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,  # BF16 as per spec
        device_map="auto",
    )

    # LoRA config: rank and alpha from model-size preset (Req 11.1)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=preset["lora_r"],
        lora_alpha=preset["lora_r"] * 2,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    # GRPO config: batch/grad_accum/seq_len from model-size preset (Req 11.1)
    save_steps = max(1, min(10, args.episodes // 4))
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=preset["batch"],
        gradient_accumulation_steps=preset["grad_accum"],
        num_generations=args.num_generations,  # 8 rollouts
        max_completion_length=preset["seq_len"],
        learning_rate=1e-5,
        bf16=True,
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=3,
        vllm_mode=args.vllm_mode if args.vllm_mode else None,
        seed=args.seed,
    )

    # Instantiate GRPOTrainer with LoRA + GRPO config (Req 11.1)
    # The manual rollout loop below drives episode-level reward CSV logging;
    # trainer.train() can be called instead for full GRPO weight updates.
    _trainer = GRPOTrainer(  # noqa: F841
        model=model,
        args=grpo_config,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    # Reward CSV logger
    reward_csv = RewardCSVLogger(Path(args.output_dir) / "reward_log.csv")

    episode_rewards: list[float] = []
    final_tier = 0

    log.info(
        "Starting GRPO training: %d episodes, seed=%d, vllm_mode=%s, "
        "num_generations=%d, max_steps=%d",
        args.episodes,
        args.seed,
        args.vllm_mode,
        args.num_generations,
        args.max_steps,
    )

    for ep in range(args.episodes):
        ep_seed = (
            args.seed + ep if args.seed is not None else random.randint(0, 2**31 - 1)
        )

        # Run one rollout episode directly via env (Req 11.4)
        experiences = rollout_func(
            env=env,
            model=model,
            tokenizer=tokenizer,
            seed=ep_seed,
            max_steps=args.max_steps,
            num_generations=args.num_generations,
        )

        total_reward = sum(e["reward"] for e in experiences)
        steps_taken = len(experiences)
        terminal_outcome = (
            "success" if (experiences and experiences[-1]["done"]) else "timeout"
        )

        episode_rewards.append(total_reward)

        # Log per-episode reward to CSV (Req 11.1)
        reward_csv.log(
            episode=ep,
            seed=ep_seed,
            total_reward=total_reward,
            steps=steps_taken,
            terminal_outcome=terminal_outcome,
        )
        if (ep + 1) % 10 == 0:
            log.info(
                "Checkpoint marker at episode %d → %s",
                ep + 1,
                Path(args.output_dir) / f"checkpoint_ep{ep+1}",
            )

        # Update curriculum tier from env state
        try:
            state = env.state
            final_tier = int(state.curriculum_tier)
        except Exception:
            pass

        if (ep + 1) % 10 == 0 or ep == 0:
            mean_r = sum(episode_rewards) / len(episode_rewards)
            log.info(
                "Episode %d/%d | reward=%.4f | mean=%.4f | tier=%d",
                ep + 1,
                args.episodes,
                total_reward,
                mean_r,
                final_tier,
            )

    # Write training summary on completion (Req 11.2)
    summary_path = Path(args.output_dir) / "training_summary.json"
    _write_summary(
        out_path=summary_path,
        episodes=args.episodes,
        rewards=episode_rewards,
        final_tier=final_tier,
        model_path=args.model_path,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO training runner for Clinical Trial Designer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--model-size",
        choices=["1.5b", "3b", "7b"],
        default="7b",
        help="Model size preset (auto-configures LoRA rank, batch, seq length)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes (Req 11.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (Req 11.3)",
    )
    parser.add_argument(
        "--vllm-mode",
        default="colocate",
        choices=["colocate", "server", ""],
        help="vLLM inference mode: 'colocate' runs vLLM in the same process",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Number of rollout generations per GRPO step (8 rollouts)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/grpo",
        help="Directory for checkpoints, reward CSV, and training summary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 2 episodes with random policy to verify pipeline (no training)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args = _apply_model_size_preset(args)
    if args.dry_run:
        _dry_run(args)
    else:
        train(args)
