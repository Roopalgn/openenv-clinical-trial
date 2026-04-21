"""
eval_compare.py — Evaluation script comparing base (random) policy vs trained checkpoint.

Runs both policies through N episodes using identical seeds and outputs a comparison
report with success rate, mean reward, mean episode length, and scenario breakdown per tier.

Usage:
    python eval_compare.py \\
        --model-path ./outputs/grpo/checkpoint-100 \\
        --episodes 50 \\
        --seed 42 \\
        --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models import TrialAction, TrialObservation
    from server.environment import Environment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("eval_compare")


# ---------------------------------------------------------------------------
# Lazy imports — ML stack is optional at import time
# ---------------------------------------------------------------------------


def _import_ml_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        return AutoModelForCausalLM, AutoTokenizer, torch
    except ImportError as exc:
        log.error(
            "Transformers / PyTorch not installed. "
            "Install with: pip install transformers torch\n%s",
            exc,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Episode result tracking
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Tracks the outcome of a single evaluation episode."""

    episode_id: int
    seed: int
    total_reward: float
    episode_length: int
    success: bool
    scenario_id: str
    curriculum_tier: str


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Base policy that selects random valid actions."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def select_action(self, obs: "TrialObservation", step: int) -> "TrialAction":
        """Select a random action from available actions."""
        from models import ActionType, TrialAction

        # Use available_actions if provided, otherwise cycle through all actions
        if obs.available_actions:
            action_str = self._rng.choice(obs.available_actions)
            try:
                action_type = ActionType(action_str)
            except ValueError:
                action_type = ActionType.SET_PRIMARY_ENDPOINT
        else:
            action_types = list(ActionType)
            action_type = self._rng.choice(action_types)

        return TrialAction(
            action_type=action_type,
            parameters={},
            justification="random policy selection",
            confidence=self._rng.uniform(0.3, 0.7),
        )


class TrainedPolicy:
    """Trained model policy that generates actions via LLM."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        AutoModelForCausalLM, AutoTokenizer, torch = _import_ml_stack()

        log.info("Loading trained model from: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self.torch = torch

    def select_action(self, obs: "TrialObservation", step: int) -> "TrialAction":
        """Generate action from model output."""
        from models import ActionType, TrialAction

        # Build prompt from observation
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
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Parse response into TrialAction
        try:
            data = json.loads(response_text)
            return TrialAction(
                action_type=data.get("action_type", ActionType.SET_PRIMARY_ENDPOINT),
                parameters=data.get("parameters", {}),
                justification=data.get("justification", "model output"),
                confidence=float(data.get("confidence", 0.5)),
            )
        except Exception:
            # Fallback to safe default
            action_types = list(ActionType)
            action_type = action_types[step % len(action_types)]
            return TrialAction(
                action_type=action_type,
                parameters={},
                justification="fallback: could not parse model output",
                confidence=0.5,
            )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(
    env: "Environment",
    policy: RandomPolicy | TrainedPolicy,
    seed: int,
    max_steps: int,
    episode_id: int,
) -> EpisodeResult:
    """Run a single episode with the given policy.

    Args:
        env: Environment instance.
        policy: Policy to use for action selection.
        seed: Episode seed for reproducibility.
        max_steps: Maximum steps per episode.
        episode_id: Episode number for tracking.

    Returns:
        EpisodeResult with episode metrics.
    """
    obs = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    for step_idx in range(max_steps):
        action = policy.select_action(obs, step_idx)
        obs, reward_dict, done, info = env.step_full(action)

        step_reward = (
            sum(reward_dict.values())
            if isinstance(reward_dict, dict)
            else float(reward_dict)
        )
        total_reward += step_reward
        steps += 1

        if done:
            break

    # Extract episode metadata
    state = env.state
    success = state.is_resolved and total_reward > 0

    return EpisodeResult(
        episode_id=episode_id,
        seed=seed,
        total_reward=total_reward,
        episode_length=steps,
        success=success,
        scenario_id=state.scenario_id,
        curriculum_tier=state.curriculum_tier,
    )


# ---------------------------------------------------------------------------
# Comparison report generation
# ---------------------------------------------------------------------------


def generate_comparison_report(
    base_results: list[EpisodeResult],
    trained_results: list[EpisodeResult],
) -> dict[str, Any]:
    """Generate comparison report from episode results (Req 12.2).

    Args:
        base_results: Results from random policy.
        trained_results: Results from trained policy.

    Returns:
        Dictionary with comparison metrics.
    """

    def compute_metrics(results: list[EpisodeResult]) -> dict[str, Any]:
        if not results:
            return {
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_episode_length": 0.0,
                "total_episodes": 0,
            }

        success_count = sum(1 for r in results if r.success)
        total_reward = sum(r.total_reward for r in results)
        total_length = sum(r.episode_length for r in results)

        return {
            "success_rate": round(success_count / len(results), 4),
            "mean_reward": round(total_reward / len(results), 4),
            "mean_episode_length": round(total_length / len(results), 2),
            "total_episodes": len(results),
        }

    def scenario_breakdown(results: list[EpisodeResult]) -> dict[str, Any]:
        """Compute per-tier scenario breakdown (Req 12.2)."""
        tier_groups: dict[str, list[EpisodeResult]] = {}
        for r in results:
            tier = r.curriculum_tier
            if tier not in tier_groups:
                tier_groups[tier] = []
            tier_groups[tier].append(r)

        breakdown = {}
        for tier, tier_results in sorted(tier_groups.items()):
            tier_success = sum(1 for r in tier_results if r.success)
            tier_reward = sum(r.total_reward for r in tier_results)
            breakdown[tier] = {
                "episodes": len(tier_results),
                "success_rate": round(tier_success / len(tier_results), 4),
                "mean_reward": round(tier_reward / len(tier_results), 4),
            }

        return breakdown

    base_metrics = compute_metrics(base_results)
    trained_metrics = compute_metrics(trained_results)

    report = {
        "base_policy": base_metrics,
        "trained_policy": trained_metrics,
        "improvement": {
            "success_rate_delta": round(
                trained_metrics["success_rate"] - base_metrics["success_rate"], 4
            ),
            "mean_reward_delta": round(
                trained_metrics["mean_reward"] - base_metrics["mean_reward"], 4
            ),
            "mean_episode_length_delta": round(
                trained_metrics["mean_episode_length"]
                - base_metrics["mean_episode_length"],
                2,
            ),
        },
        "scenario_breakdown": {
            "base_policy": scenario_breakdown(base_results),
            "trained_policy": scenario_breakdown(trained_results),
        },
    }

    return report


def print_comparison_report(report: dict[str, Any]) -> None:
    """Print comparison report to console in human-readable format."""
    log.info("=" * 70)
    log.info("EVALUATION COMPARISON REPORT")
    log.info("=" * 70)

    log.info("\nBase Policy (Random):")
    for key, value in report["base_policy"].items():
        log.info("  %s: %s", key, value)

    log.info("\nTrained Policy:")
    for key, value in report["trained_policy"].items():
        log.info("  %s: %s", key, value)

    log.info("\nImprovement:")
    for key, value in report["improvement"].items():
        log.info("  %s: %s", key, value)

    log.info("\nScenario Breakdown by Tier:")
    log.info("\n  Base Policy:")
    for tier, metrics in report["scenario_breakdown"]["base_policy"].items():
        log.info("    Tier %s: %s", tier, metrics)

    log.info("\n  Trained Policy:")
    for tier, metrics in report["scenario_breakdown"]["trained_policy"].items():
        log.info("    Tier %s: %s", tier, metrics)

    log.info("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    """Main evaluation entry point (Req 12.1, 12.2, 12.3, 12.4)."""
    from server.environment import Environment

    log.info("Initializing environment...")
    env = Environment()

    # Generate episode seeds (Req 12.3 — same seeds for both policies)
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    episode_seeds = [base_seed + i for i in range(args.episodes)]

    # Initialize policies
    log.info("Initializing base (random) policy...")
    base_policy = RandomPolicy(seed=base_seed)

    log.info("Initializing trained policy from: %s", args.model_path)
    trained_policy = TrainedPolicy(model_path=args.model_path)

    # Run base policy episodes
    log.info("Running %d episodes with base policy...", args.episodes)
    base_results: list[EpisodeResult] = []
    for i, seed in enumerate(episode_seeds):
        result = run_episode(env, base_policy, seed, args.max_steps, i)
        base_results.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            log.info(
                "  Base episode %d/%d | reward=%.4f | length=%d | success=%s",
                i + 1,
                args.episodes,
                result.total_reward,
                result.episode_length,
                result.success,
            )

    # Run trained policy episodes (Req 12.3 — same seeds)
    log.info("Running %d episodes with trained policy...", args.episodes)
    trained_results: list[EpisodeResult] = []
    for i, seed in enumerate(episode_seeds):
        result = run_episode(env, trained_policy, seed, args.max_steps, i)
        trained_results.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            log.info(
                "  Trained episode %d/%d | reward=%.4f | length=%d | success=%s",
                i + 1,
                args.episodes,
                result.total_reward,
                result.episode_length,
                result.success,
            )

    # Generate and output comparison report (Req 12.2)
    report = generate_comparison_report(base_results, trained_results)
    print_comparison_report(report)

    # Write report to JSON file
    output_path = Path(args.output_dir) / "eval_comparison_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("\nComparison report written to: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation script comparing base vs trained policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model checkpoint (Req 12.4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of evaluation episodes per policy (Req 12.4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (Req 12.3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs/eval",
        help="Directory for evaluation report output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(args)
