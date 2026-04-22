# Model Training Runbook — GRPO on Clinical Trial Designer

> **Audience:** Both team members (Roopal + Suyash) for onsite training April 25–26. This is the step-by-step procedure to run GRPO training with HuggingFace H100 compute credits.

## Pre-Training Checklist

Before arriving onsite, verify:

- [ ] Environment server runs locally: `docker compose up` → `/ping` responds
- [ ] `openenv.yaml` passes OpenEnv CLI validation
- [ ] All 4 scenarios complete at Warmup tier without errors
- [ ] Reward CSV logs correctly from a 5-episode dry run
- [ ] Episode JSONL transcripts contain full action/observation history
- [ ] `eval_compare.py --policy scripted --episodes 10 --tier warmup` produces valid JSON
- [ ] `plot_rewards.py` generates PNG from sample reward CSV
- [ ] Git repo clean, `main` branch up to date with all merged code

---

## Training Configuration

### GRPO Hyperparameters

> **Inspired by:** KubeSRE's GRPO config (8 generations, vLLM colocate). Bio Experiment's training loop. TRL 0.29+ GRPOTrainer defaults.

```python
GRPO_CONFIG = {
    # Model
    "model_name": "Qwen/Qwen2.5-7B-Instruct",     # Or "unsloth/Qwen2.5-7B-bnb-4bit" for Unsloth
    "max_seq_length": 4096,                          # Sufficient for multi-step episodes

    # LoRA
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],

    # GRPO
    "num_generations": 8,            # Rollouts per prompt (GRPO group size)
    "max_completion_length": 512,    # Max tokens per action response
    "temperature": 0.7,             # Sampling temperature for rollouts
    "top_p": 0.95,

    # Training
    "learning_rate": 5e-6,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_steps": 500,               # ~500 GRPO steps × 8 rollouts = 4000 episodes
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,

    # vLLM colocate
    "use_vllm": True,
    "vllm_gpu_utilization": 0.7,    # Leave 30% for training

    # Logging
    "logging_steps": 1,
    "save_steps": 50,
    "output_dir": "checkpoints/grpo_clinical_trial",
    "report_to": "none",            # Or "wandb" if available

    # Reward
    "reward_weights": None,          # We compute scalar reward externally
}
```

### System Prompt (Push 5 — Improved)

> **Rationale:** The original system prompt listed actions without explaining *when* or *why* to use them. The improved version gives the agent a decision framework — phase-by-phase guidance, parameter hints, and a clear success definition. This reduces early-episode random exploration and accelerates Phase I → Phase II transition learning.

```python
SYSTEM_PROMPT = """You are a clinical trial designer. You are given a disease scenario 
and must design a complete Phase I/II clinical trial that detects the drug's true effect.

## YOUR GOAL
Design a trial where:
1. The primary analysis achieves statistical significance (p < 0.05)
2. Statistical power is adequate (≥ 0.80)
3. The correct patient population is identified
4. Budget and time constraints are respected
5. All FDA regulations are followed

## WORKFLOW (follow this order — skipping phases is penalized)

### Phase I — Safety & Dose-Finding (do this FIRST)
- `run_dose_escalation`: Test increasing doses to find the maximum tolerated dose (MTD).
  Parameters: {"dose_mg": <int>, "cohort_size": 3|6}. Start low (50mg), escalate in steps.
  Run at least 3 dose levels before concluding Phase I.
- `observe_safety_signal`: Check adverse events at the current dose.
  Use after each dose escalation to monitor safety before going higher.
- `estimate_effect_size`: Get a noisy estimate of the drug's effect from Phase I data.
  Requires at least 1 dose escalation. This estimate guides your Phase II sample size.

### Phase II — Efficacy Trial Design (based on Phase I findings)
- `set_primary_endpoint`: Define what the trial measures (e.g., "progression_free_survival").
- `set_sample_size`: Set N based on estimated effect size and desired power.
  Use the power formula: larger effect → smaller N needed. Underpowered trials fail.
- `set_inclusion_criteria`: Define who enters the trial.
  KEY INSIGHT: If Phase I suggests a subgroup benefits more (e.g., "EGFR+"),
  restricting to that subgroup dramatically increases power with fewer patients.
- `set_exclusion_criteria`: Define who is excluded (safety, confounders).
- `set_dosing_schedule`: Choose dose and frequency based on Phase I MTD finding.
- `set_control_arm`: Set comparator ("placebo" or "standard_of_care").
- `set_randomization_ratio`: Set treatment:control ratio (e.g., "1:1" or "2:1").
- `set_blinding`: Set blinding level ("double_blind" recommended).

### Regulatory
- `submit_to_fda_review`: Submit protocol for approval. Requires endpoint + sample size set.
- `request_protocol_amendment`: Fix protocol issues after a failed FDA review. Costs time/budget.

### Monitoring (after FDA approval)
- `run_interim_analysis`: Check early results. Can stop for futility or reduce sample size.
- `modify_sample_size`: Adjust N based on interim results (requires interim first).
- `add_biomarker_stratification`: Analyze by subgroup. Use when you suspect a hidden responder population.

### Analysis & Conclusion
- `run_primary_analysis`: Run the final statistical test. This determines success/failure.
- `synthesize_conclusion`: Summarize findings. Include effect estimate, confidence interval,
  identified subgroups, and mechanism hypothesis.

## DECISION TIPS
- Phase I data is noisy. Run multiple dose levels, not just one.
- If the effect size estimate is small, look for subgroups before committing to a large N.
- Budget is limited. Enriching for responders (smaller N, bigger effect) beats enrolling everyone.
- Overconfidence is penalized: don't claim high confidence without data to support it.
- You can amend the protocol if FDA review fails, but it costs time and budget.

Respond with a JSON action: {"action_type": "<action>", "parameters": {...}}"""
```

### Environment Connection

```python
ENV_CONFIG = {
    "env_url": "http://localhost:8000",     # Local Docker or HF Space URL
    "reset_endpoint": "/reset",
    "step_endpoint": "/step",
    "state_endpoint": "/state",
    "timeout_seconds": 30,
}
```

---

## Training Procedure

### Phase 1: Setup (15 minutes)

```bash
# 1. Clone repo and install dependencies
git clone https://github.com/Roopalgn/openenv-clinical-trial.git
cd openenv-clinical-trial

# 2. Install training dependencies
pip install trl>=0.29.0 unsloth peft accelerate bitsandbytes
pip install vllm>=0.4.0 scipy numpy requests

# 3. Start environment server
docker compose up -d
curl http://localhost:8000/ping  # Should return {"status": "ok"}

# 4. Run smoke test
python train.py --max-steps 2 --dry-run
```

### Phase 2: Baseline Collection (20 minutes)

```bash
# Random baseline
python eval_compare.py --policy random --episodes 50 --tier warmup --seed 42 \
    --output results/random_warmup.json

# Scripted baseline
python eval_compare.py --policy scripted --episodes 50 --tier warmup --seed 42 \
    --output results/scripted_warmup.json

# Generate baseline plots
python plot_rewards.py --input results/random_warmup.json --output results/random_baseline.png
python plot_rewards.py --input results/scripted_warmup.json --output results/scripted_baseline.png
```

### Phase 3: GRPO Training (2–4 hours)

```bash
# Full training run
python train.py \
    --model unsloth/Qwen2.5-7B-bnb-4bit \
    --env-url http://localhost:8000 \
    --lora-r 16 \
    --lora-alpha 32 \
    --num-generations 8 \
    --max-steps 500 \
    --lr 5e-6 \
    --output-dir checkpoints/grpo_clinical_trial \
    --reward-csv results/rewards.csv \
    --curriculum-log results/curriculum.jsonl \
    --transcript-log results/transcripts.jsonl \
    --seed 42
```

**Monitor during training:**

```bash
# In a separate terminal — watch reward trend
tail -f results/rewards.csv | python -c "
import sys, csv
reader = csv.DictReader(sys.stdin)
for row in reader:
    ep = row['episode']
    tier = row['tier']
    reward = row['total_reward']
    success = row['success']
    print(f'Episode {ep} | Tier {tier} | Reward {reward} | Success {success}')
"
```

### Phase 4: Post-Training Evaluation (30 minutes)

```bash
# Trained model evaluation across all tiers
for tier in warmup beginner intermediate advanced expert; do
    python eval_compare.py \
        --policy trained \
        --checkpoint checkpoints/grpo_clinical_trial \
        --episodes 50 \
        --tier $tier \
        --seed 42 \
        --output results/trained_${tier}.json
done

# Generate comparison
python eval_compare.py --compare \
    results/random_warmup.json \
    results/scripted_warmup.json \
    results/trained_warmup.json \
    --output results/comparison_report.md

# Generate reward curve plots
python plot_rewards.py --input results/rewards.csv --output results/reward_curve.png
```

### Phase 5: Pitch Asset Generation (30 minutes)

```bash
# Best episode transcript for demo replay
python -c "
import json
with open('results/transcripts.jsonl') as f:
    episodes = [json.loads(line) for line in f]
best = max(episodes, key=lambda e: e['total_reward'])
print(f'Best episode: #{best[\"episode_id\"]} | Reward: {best[\"total_reward\"]:.2f}')
with open('results/best_episode.json', 'w') as f:
    json.dump(best, f, indent=2)
"

# Worst episode transcript (for before/after contrast)
python -c "
import json
with open('results/transcripts.jsonl') as f:
    episodes = [json.loads(line) for line in f]
worst = min(episodes, key=lambda e: e['total_reward'])
with open('results/worst_episode.json', 'w') as f:
    json.dump(worst, f, indent=2)
"

# Push checkpoint to HF Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/grpo_clinical_trial',
    repo_id='Roopalgn/clinical-trial-designer-grpo',
    repo_type='model',
)
print('Checkpoint uploaded to HF Hub')
"
```

---

## Expected Training Timeline

| Time | Activity | Expected Output |
|------|----------|----------------|
| T+0:00 | Setup + environment health check | Server running, smoke test passes |
| T+0:15 | Baseline collection | `random_warmup.json`, `scripted_warmup.json` |
| T+0:35 | GRPO training starts | `rewards.csv` begins populating |
| T+1:00 | ~125 episodes done | Agent should be Warmup→Beginner transition |
| T+2:00 | ~250 episodes done | Agent should be Beginner→Intermediate |
| T+3:00 | ~375 episodes done | Agent should be Intermediate→Advanced |
| T+4:00 | ~500 episodes done | Agent at Advanced, possibly Expert |
| T+4:15 | Post-training evaluation | `comparison_report.md`, reward curve PNGs |
| T+4:45 | Pitch asset generation | Best/worst episodes, HF Hub upload |

---

## Expected Outputs

### Success Indicators

| Metric | Expected at T+4:00 | Minimum Acceptable |
|--------|--------------------|--------------------|
| Warmup success rate | 80–90% | ≥ 60% |
| Beginner success rate | 55–70% | ≥ 40% |
| Intermediate success rate | 35–50% | ≥ 25% |
| Advanced success rate | 20–35% | ≥ 10% |
| Reward trend slope | > 0.1/episode | > 0 |
| Final rolling avg reward | +6 to +10 | > +3 |
| Curriculum tier reached | Advanced or Expert | ≥ Intermediate |

### Files Generated

```
results/
├── rewards.csv              # Per-episode reward breakdown (500+ rows)
├── curriculum.jsonl         # Tier advancement log
├── transcripts.jsonl        # Full episode transcripts
├── random_warmup.json       # Random baseline results
├── scripted_warmup.json     # Scripted baseline results
├── trained_warmup.json      # Trained model results (per tier)
├── trained_beginner.json
├── trained_intermediate.json
├── trained_advanced.json
├── trained_expert.json
├── comparison_report.md     # Side-by-side comparison table
├── reward_curve.png         # Training reward scatter + rolling avg
├── random_baseline.png      # Random policy reward distribution
├── scripted_baseline.png    # Scripted policy reward distribution
├── best_episode.json        # Highest-reward episode transcript
└── worst_episode.json       # Lowest-reward episode transcript
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All rewards ≈ 0 | Per-step components too small | Increase `w_validity`, `w_ordering` weights |
| All rewards similar (low variance) | Terminal rewards too small | Increase `r_terminal_success` magnitude |
| Agent repeats same action | Repeat penalty too low | Increase repeat penalty to -0.3 |
| Agent skips Phase I | `r_info_gain` too low | Increase `w_info_gain` or data quality weight in φ(s) |
| Reward plateau at tier 1 | Curriculum not advancing | Check `should_advance()` — may need W=8 instead of W=10 |
| OOM on H100 | Model too large for LoRA + vLLM | Reduce `vllm_gpu_utilization` to 0.5, or use 4-bit quantization |
| Slow rollouts | Environment latency | Check Docker container resource limits, reduce `max_completion_length` |
| Training diverges (reward drops) | Learning rate too high | Reduce to 1e-6, increase `warmup_ratio` to 0.10 |

---

## Checkpoint Strategy

- Save every 50 steps: `checkpoints/grpo_clinical_trial/step_050/`, `step_100/`, etc.
- Keep the best checkpoint by validation reward (manual check after training)
- Upload final checkpoint to HuggingFace Hub for demo

---

## Reward Weight Tuning Guide (Push 5)

> **Purpose:** After baseline collection, use the diagnostic patterns below to decide whether reward weights need adjustment before the full training run. This saves hours of training on a miscalibrated reward.

### Diagnostic Procedure

After collecting 50 scripted-baseline episodes, run:

```bash
python -c "
import csv
from collections import defaultdict

totals = defaultdict(list)
with open('results/rewards.csv') as f:
    for row in csv.DictReader(f):
        for k, v in row.items():
            if k.startswith('r_'):
                totals[k].append(float(v))

print('Component   | Mean    | Std     | Min     | Max')
print('-' * 55)
for k in sorted(totals):
    vals = totals[k]
    import statistics
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0
    print(f'{k:12s} | {mean:7.3f} | {std:7.3f} | {min(vals):7.3f} | {max(vals):7.3f}')
"
```

### Decision Matrix

| Diagnostic Pattern | Problem | Adjustment |
|-------------------|---------|------------|
| `r_info_gain` mean < 0.05, agent skips Phase I | Agent not incentivized to gather data | Increase `w_info_gain` from 1.0 → 1.5 |
| `r_ordering` mean ≈ 0 with many skips | Phase-order penalty too weak for GRPO | Increase skip penalty from -0.3 → -0.5 per phase |
| `r_validity` mean > 0.25 for all episodes | Validity is "free reward" — not discriminating | Reduce `w_validity` from 1.0 → 0.6 |
| `r_terminal_success` std < 1.0 | Terminal reward doesn't separate good/bad | Increase success reward to +7.0 base (from +5.0) |
| `r_shaping` dominates (>40% of total) | Shaped reward hacking — agent optimizes φ not task | Reduce γ from 0.99 → 0.95, or cap shaping at ±0.5/step |
| `r_penalty` mean < -0.5 | Penalty overwhelms learning signal | Reduce penalty magnitude or cap at -0.1/step |
| `r_novelty` mean ≈ 0 after 5 steps | Agent ignores action diversity | Increase first-use bonus from +0.1 → +0.2 |
| `r_efficiency` near 0 for all episodes | Budget signal too weak | Increase weight or make budget depletion penalty harsher |
| Total reward variance < 3.0 | GRPO needs high variance for advantages | Increase terminal reward magnitudes (+success, -failure) |
| Total reward variance > 20.0 | Too noisy — GRPO updates are unstable | Reduce terminal reward magnitudes or add reward normalization |

### Recommended Starting Weights

Based on analysis of winner configurations (KubeSRE, Bio Experiment, VRAM):

```python
REWARD_WEIGHTS = {
    # Per-step (relative importance)
    "w_validity": 0.8,     # Slightly below 1.0 — compliance is baseline behavior
    "w_ordering": 1.0,     # Phase ordering is a key differentiator 
    "w_info_gain": 1.2,    # Most important per-step signal — drives data gathering
    "w_efficiency": 0.6,   # Secondary — important but shouldn't dominate
    "w_novelty": 0.5,      # Exploration bonus — diminishes over training
    "w_penalty": 1.0,      # Full weight for penalties
    
    # Shaping
    "gamma": 0.99,         # Discount factor for potential-based shaping
    "shaping_cap": 0.5,    # Max |r_shaping| per step to prevent shaping dominance
    
    # Terminal (absolute magnitudes)
    "r_success_base": 5.0,          # +5.0 flat for detecting true effect
    "r_success_efficiency_bonus": 2.0,  # Up to +2.0 for completing quickly
    "r_calibration_max": 5.0,       # Up to +5.0 for matching ground truth
    "r_power_max": 2.0,             # +2.0 for power ≥ 0.90
    "r_fda_bonus": 2.0,             # +2.0 for full FDA compliance
    "r_failure": -1.0,              # Failure penalty
    "r_timeout": -2.0,              # Timeout wipes accumulated reward
    "r_overconfidence": -0.5,       # Per high-confidence wrong claim
}
```

### When to Re-Tune During Training

Check reward CSV every ~100 episodes during training. If you see:

- **Plateau lasting >50 episodes:** Likely curriculum is stuck. Check `should_advance()` thresholds.
- **Reward cliff (sudden drop):** Tier advancement happened but agent can't handle new difficulty. May need intermediate hardening steps (see `adaptive_difficulty_spec.md`).
- **r_shaping growing while r_terminal flat:** Agent is gaming the shaping function. Reduce γ or cap shaping.
- **Success rate >90% at current tier for >30 episodes:** Curriculum advancement is too slow. Lower mastery window.
- Tag the checkpoint with training metadata: `{"episodes": N, "tier_reached": T, "avg_reward": R}`
