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

PLAN_PARSE_FAILURE_REWARD = -3.0
INVALID_SEQUENCE_REWARD = -3.0
INCOMPLETE_PLAN_PENALTY = -3.0
MAX_INCOMPLETE_PROGRESS_BONUS = 2.4
DEFAULT_TRAIN_CURRICULUM_TIER = 3
DEFAULT_FREEZE_CURRICULUM = True

REQUIRED_ACTION_ORDER: tuple[str, ...] = (
    "set_primary_endpoint",
    "set_sample_size",
    "set_inclusion_criteria",
    "set_dosing_schedule",
    "set_control_arm",
    "enroll_patients",
    "run_dose_escalation",
    "run_interim_analysis",
    "run_primary_analysis",
)

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


def _build_action_from_text(text: str, step: int, available_actions: list[str] | None = None) -> "TrialAction":
    """Parse a model-generated text into a TrialAction.

    Falls back to a safe default action when the text cannot be parsed.
    If available_actions is provided, validates that the parsed action is among them.
    """
    from models import ActionType, TrialAction

    # Try to extract JSON from the text (handle markdown code blocks, etc.)
    json_text = text.strip()
    # Strip markdown code fences
    if "```" in json_text:
        parts = json_text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                json_text = part
                break
    # Find the first { ... } block
    start = json_text.find("{")
    end = json_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_text = json_text[start:end + 1]

    try:
        data = json.loads(json_text)
        action_type_str = data.get("action_type", "set_primary_endpoint")
        # Validate against available actions if provided
        if available_actions and action_type_str not in available_actions:
            # Pick the first available action instead of an invalid one
            action_type_str = available_actions[0] if available_actions else "set_primary_endpoint"
        return TrialAction(
            action_type=action_type_str,
            parameters=data.get("parameters", {}),
            justification=data.get("justification", "model output"),
            confidence=min(max(float(data.get("confidence", 0.7)), 0.0), 1.0),
        )
    except Exception:
        # Fallback: pick a valid action from available_actions, or safe default
        if available_actions:
            fallback_type = available_actions[step % len(available_actions)]
        else:
            fallback_type = "set_primary_endpoint"
        return TrialAction(
            action_type=fallback_type,
            parameters={},
            justification="fallback: could not parse model output",
            confidence=0.5,
        )


def _extract_json_payload(text: str) -> Any | None:
    """Extract the first JSON object/list from model text."""
    text = text.strip()
    candidates = []
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part:
                candidates.append(part)
    candidates.append(text)

    for candidate in candidates:
        object_start = candidate.find("{")
        object_end = candidate.rfind("}")
        list_start = candidate.find("[")
        list_end = candidate.rfind("]")

        spans = []
        if object_start != -1 and object_end > object_start:
            spans.append((object_start, object_end + 1))
        if list_start != -1 and list_end > list_start:
            spans.append((list_start, list_end + 1))

        for start, end in sorted(spans, key=lambda span: span[0]):
            try:
                return json.loads(candidate[start:end])
            except json.JSONDecodeError:
                continue
    return None


def _coerce_bounded_int(value: Any, *, minimum: int, maximum: int) -> int | None:
    try:
        coerced = int(float(value))
    except (TypeError, ValueError):
        return None
    return max(minimum, min(maximum, coerced))


def _coerce_confidence(value: Any, default: float = 0.7) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return min(max(confidence, 0.0), 1.0)


def _normalise_action_spec(raw_action: Any) -> "TrialAction | None":
    from models import ActionType, TrialAction

    if not isinstance(raw_action, dict):
        return None

    action_type = raw_action.get("action_type")
    valid_actions = {action.value for action in ActionType}
    if action_type not in valid_actions:
        return None

    parameters = raw_action.get("parameters", {})
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        return None

    parameters = dict(parameters)
    if action_type == ActionType.SET_SAMPLE_SIZE.value:
        sample_size = _coerce_bounded_int(
            parameters.get("sample_size"), minimum=30, maximum=500
        )
        if sample_size is None:
            return None
        parameters["sample_size"] = sample_size

    if action_type == ActionType.ENROLL_PATIENTS.value:
        n_patients = _coerce_bounded_int(
            parameters.get("n_patients"), minimum=0, maximum=500
        )
        if n_patients is None:
            return None
        parameters["n_patients"] = n_patients

    return TrialAction(
        action_type=action_type,
        parameters=parameters,
        justification=str(raw_action.get("justification", "model action plan")),
        confidence=_coerce_confidence(raw_action.get("confidence", 0.7)),
    )


def parse_action_plan(text: str, max_actions: int = 12) -> list["TrialAction"] | None:
    """Parse a full model-generated action plan.

    Accepted shape:
      {"actions": [{"action_type": "...", "parameters": {...}}, ...]}
    A bare JSON list of action objects is accepted for convenience.
    """
    payload = _extract_json_payload(text)
    if isinstance(payload, dict):
        raw_actions = payload.get("actions")
    elif isinstance(payload, list):
        raw_actions = payload
    else:
        return None

    if not isinstance(raw_actions, list) or not raw_actions:
        return None

    actions: list["TrialAction"] = []
    for raw_action in raw_actions[:max_actions]:
        action = _normalise_action_spec(raw_action)
        if action is None:
            return None
        actions.append(action)
    return actions


def plan_progress_bonus(actions: list["TrialAction"]) -> float:
    """Reward longer valid prefixes without letting incomplete plans look solved."""
    matched = 0
    for action in actions:
        if matched >= len(REQUIRED_ACTION_ORDER):
            break
        if action.action_type.value == REQUIRED_ACTION_ORDER[matched]:
            matched += 1
    if matched <= 1:
        return 0.0
    progress = matched / len(REQUIRED_ACTION_ORDER)
    return MAX_INCOMPLETE_PROGRESS_BONUS * progress


def _observation_to_plan_prompt(obs: Any) -> str:
    """Build the full-episode planning prompt used by GRPO."""
    return (
        "You are designing a clinical trial. Produce a complete ordered action plan, not just one action.\n\n"
        f"Scenario: {obs.scenario_description}\n"
        f"Current phase: {obs.phase_data.get('current_phase', 'unknown')}\n"
        f"Phase data: {json.dumps(obs.phase_data)}\n"
        f"Resources: {json.dumps(obs.resource_status)}\n"
        f"Currently available actions: {obs.available_actions}\n\n"
        "The environment will execute exactly the actions you list. Invalid, unparsable, or incomplete plans receive low reward.\n"
        f"Minimum action_type order for completion: {list(REQUIRED_ACTION_ORDER)}.\n"
        "For higher reward, include one useful information action when legal, such as add_biomarker_stratification or estimate_effect_size.\n"
        "Set sample_size and enroll_patients to the same integer between 120 and 420, use 9 to 11 actions, and end with run_primary_analysis.\n"
        "Respond with ONLY valid JSON: {\"actions\": [action objects]}. Each action object needs action_type and parameters."
    )


def rollout_action_plan_reward(
    env: "Environment",
    completion: str,
    seed: int,
    max_steps: int,
    curriculum_tier: int = DEFAULT_TRAIN_CURRICULUM_TIER,
    freeze_curriculum: bool = DEFAULT_FREEZE_CURRICULUM,
) -> float:
    """Score one completion by executing exactly its proposed episode plan."""
    actions = parse_action_plan(completion, max_actions=max_steps)
    if actions is None:
        return PLAN_PARSE_FAILURE_REWARD

    try:
        _obs = env.reset(
            seed=seed,
            curriculum_tier=curriculum_tier,
            freeze_curriculum=freeze_curriculum,
        )
        total = 0.0
        done = False

        for action in actions[:max_steps]:
            _obs, reward_dict, done, info = env.step_full(action)
            if not info.get("action_valid", True):
                return INVALID_SEQUENCE_REWARD
            total += (
                sum(reward_dict.values())
                if isinstance(reward_dict, dict)
                else float(reward_dict)
            )
            if done:
                return total

        if not done:
            return total + plan_progress_bonus(actions) + INCOMPLETE_PLAN_PENALTY
        return total
    except Exception as exc:
        log.debug("Action-plan rollout failed: %s", exc)
        return PLAN_PARSE_FAILURE_REWARD


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
        available = obs.available_actions
        prompt = (
            f"You are designing a clinical trial. Choose ONE action from the available actions.\n\n"
            f"Scenario: {obs.scenario_description}\n"
            f"Current phase: {obs.phase_data.get('current_phase', 'unknown')}\n"
            f"Phase data: {json.dumps(obs.phase_data)}\n"
            f"Resources: {json.dumps(obs.resource_status)}\n"
            f"Available actions: {available}\n"
            f"Steps taken: {obs.steps_taken}/{obs.max_steps}\n"
            f"Hint: {obs.hint}\n\n"
            "Respond with ONLY a JSON object. Choose action_type from the available actions list above:\n"
            '{"action_type": "<one of available actions>", "parameters": {}, "justification": "...", "confidence": 0.8}'
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

        action = _build_action_from_text(response_text, step_idx, available)
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
    completions: list[str],
    env: "Environment",
    seed: int,
    max_steps: int,
    curriculum_tier: int = DEFAULT_TRAIN_CURRICULUM_TIER,
    freeze_curriculum: bool = DEFAULT_FREEZE_CURRICULUM,
) -> list[float]:
    """Reward function passed to GRPOTrainer.

    Each completion must contain the full ordered action plan. The environment
    executes exactly that plan, so the model earns the low reward band until it
    learns valid JSON, clinical ordering, enrollment, and terminal analysis.
    """
    rewards = []
    for i, completion in enumerate(completions):
        rewards.append(
            rollout_action_plan_reward(
                env=env,
                completion=completion,
                seed=seed + i,
                max_steps=max_steps,
                curriculum_tier=curriculum_tier,
                freeze_curriculum=freeze_curriculum,
            )
        )
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
            available = obs.available_actions
            action = _build_action_from_text("", step_idx, available)  # uses available actions
            next_obs, reward_dict, done, _ = env.step_full(action)
            total_reward += (
                sum(reward_dict.values())
                if isinstance(reward_dict, dict)
                else float(reward_dict)
            )
            steps += 1
            obs = next_obs
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

    # Reward CSV logger — populated via a callback during trainer.train()
    reward_csv = RewardCSVLogger(Path(args.output_dir) / "reward_log.csv")
    episode_rewards: list[float] = []
    final_tier = 0

    # Closure passed to GRPOTrainer as the reward function (Req 11.1, 11.4)
    def _reward_fn(completions: list[str], **kwargs) -> list[float]:
        ep = len(episode_rewards)
        ep_seed = args.seed + ep if args.seed is not None else random.randint(0, 2**31 - 1)
        rewards = _grpo_reward_fn(
            completions=completions,
            env=env,
            seed=ep_seed,
            max_steps=args.max_steps,
            curriculum_tier=args.train_curriculum_tier,
            freeze_curriculum=not args.no_freeze_curriculum,
        )
        total_reward = sum(rewards)
        episode_rewards.append(total_reward)
        reward_csv.log(
            episode=ep,
            seed=ep_seed,
            total_reward=total_reward,
            steps=args.max_steps,
            terminal_outcome="grpo",
        )
        nonlocal final_tier
        try:
            final_tier = int(env.state.curriculum_tier)
        except Exception:
            pass
        if ep % 10 == 0:
            mean_r = sum(episode_rewards) / len(episode_rewards)
            log.info(
                "Episode %d | reward=%.4f | mean=%.4f | tier=%d",
                ep + 1, total_reward, mean_r, final_tier,
            )
        return rewards

    # Build a diverse prompt dataset — each prompt uses a different scenario seed
    # so the model sees different observations during training
    from datasets import Dataset
    prompt_list = []
    for ep_idx in range(args.episodes):
        ep_seed = args.seed + ep_idx if args.seed is not None else random.randint(0, 2**31 - 1)
        obs = env.reset(
            seed=ep_seed,
            curriculum_tier=args.train_curriculum_tier,
            freeze_curriculum=not args.no_freeze_curriculum,
        )
        prompt_text = _observation_to_plan_prompt(obs)
        prompt_list.append(prompt_text)
    prompt_dataset = Dataset.from_dict({"prompt": prompt_list})

    log.info(
        "Starting GRPO training: %d episodes, seed=%d, vllm_mode=%s, "
        "num_generations=%d, max_steps=%d",
        args.episodes,
        args.seed,
        args.vllm_mode,
        args.num_generations,
        args.max_steps,
    )

    # Instantiate and run GRPOTrainer (Req 11.1)
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        peft_config=lora_config,
        tokenizer=tokenizer,
        reward_funcs=_reward_fn,
        train_dataset=prompt_dataset,
    )
    trainer.train()

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
    parser.add_argument(
        "--train-curriculum-tier",
        type=int,
        default=DEFAULT_TRAIN_CURRICULUM_TIER,
        help="Fixed curriculum tier used for stationary GRPO reward rollouts",
    )
    parser.add_argument(
        "--no-freeze-curriculum",
        action="store_true",
        help="Allow adaptive curriculum changes during GRPO reward rollouts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args = _apply_model_size_preset(args)
    if args.dry_run:
        _dry_run(args)
    else:
        train(args)
