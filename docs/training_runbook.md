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

### System Prompt

```python
SYSTEM_PROMPT = """You are a clinical trial designer. You are given a disease scenario 
and must design a complete Phase I/II clinical trial.

Your goal: design a trial that correctly detects the drug's true effect and identifies 
the right patient population, while staying within budget and following FDA regulations.

You interact with the environment by choosing actions. Each action returns an observation 
with results or feedback. Available actions include:

PHASE I: run_dose_escalation, observe_safety_signal, estimate_effect_size
PHASE II DESIGN: set_primary_endpoint, set_sample_size, set_inclusion_criteria, 
    set_exclusion_criteria, set_dosing_schedule, set_control_arm, 
    set_randomization_ratio, set_blinding
REGULATORY: submit_to_fda_review, request_protocol_amendment
MONITORING: run_interim_analysis, modify_sample_size, add_biomarker_stratification
ANALYSIS: run_primary_analysis
CONCLUSION: synthesize_conclusion

Follow the clinical workflow: Phase I → Phase II Design → FDA Submission → 
Monitoring → Analysis → Conclusion. Skipping phases is penalized.

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
- Tag the checkpoint with training metadata: `{"episodes": N, "tier_reached": T, "avg_reward": R}`
