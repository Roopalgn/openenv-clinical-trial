# Onsite Roadmap — Apr 25–26 (Scaler Campus, Bangalore)

> Execute GRPO training on H100, generate deliverables, rehearse pitch.
> Historical execution plan retained for context; several P0 items below are now already implemented in the current codebase.
> **Deadline:** 5:00 PM April 26 — Google Form submission on campus.

---

## What We Arrive With

- HF Space live: `https://roopalgn-openenv-clinical-trial.hf.space`
- `train.py` with `--dry-run` + `--model-size` tested and working
- Random policy baseline: −61.6 reward
- Colab + Kaggle notebooks validated
- All deliverable templates with `[FILL ONSITE]` placeholders
- 249/249 tests passing

---

## Day 1 — Phase 0: Setup + P0 Fixes (Hours 0–2)

### Environment Setup (~15 min)

```bash
git clone https://github.com/Roopalgn/openenv-clinical-trial.git
cd openenv-clinical-trial
pip install -e ".[train]"
pip install matplotlib

export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE="/scratch/hf_cache"
export PYTHONUNBUFFERED=1
```

### Verify (~5 min)

```bash
curl https://roopalgn-openenv-clinical-trial.hf.space/ping
# → {"status":"ok"}

python train.py --dry-run --episodes 2 --model-size 1.5b --output-dir outputs/verify
cat outputs/verify/training_summary.json
rm -rf outputs/verify
```

### P0 Code Fixes (~2 hrs, Historical)

These were the Day-1 gap fixes that were prioritized for reward-signal quality.

| Fix | What | Who | How |
|-----|------|-----|-----|
| **G26** | Wire `trainer.train()` — currently `GRPOTrainer` is unused | Suyash | Remove `noqa: F841`, wire `rollout_func`, call `trainer.train()`. Keep manual loop for `--dry-run` only. |
| **G23** | Phase progression stuck at `"literature_review"` | Suyash | In `transition_engine.py`, update `latent.episode_phase` after each action based on action type. |
| **G24** | Latent biology hardcoded instead of from scenarios | Roopal | In `episode_manager.py` reset(), populate latent state from `ScenarioConfig` (responder population, dose-response, mechanism). |
| **G25** | Adversarial designer gets `None` values | Roopal | Pass actual `latent.true_effect_size` and `latent.dropout_rate` to `analyze_failures()`. |

**Gate:** `pytest tests/ -q` must pass before training starts.

---

## Day 1 — Phase 1: First Training (Hours 2–6)

### Model Selection (try in order, fall back on OOM)

| Priority | Model | VRAM | ~Time/20 ep |
|----------|-------|------|-------------|
| Start here | Qwen2.5-1.5B (`--model-size 1.5b`) | ~3 GB | ~1 hr |
| Scale up | Qwen2.5-3B (`--model-size 3b`) | ~6 GB | ~3 hrs |
| If ambitious | Qwen2.5-7B (`--model-size 7b`) | ~14 GB | ~5 hrs |

### Run 1: Fast Signal Check

```bash
python train.py \
    --model-size 1.5b \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 20 --seed 42 \
    --output-dir outputs/run1

# Check signal immediately:
python plot_rewards.py --csv outputs/run1/reward_log.csv --out results/reward_curve_run1.png
```

**Decision point:**
- **Positive slope** → proceed to Run 2 (scale up model or episodes)
- **Flat** → debug reward weights, try different curriculum tier, document in training_log.md
- **Negative** → check action parsing, reward component weights

### Run 2: Scale Up (NON-OPTIONAL — judges want overlay comparison)

```bash
python train.py \
    --model-size 3b \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --episodes 50 --seed 42 \
    --output-dir outputs/run2
```

### Key Learning From Winner Analysis

Winners (kube-sre-gym, Bio Experiment) all had:
1. **Multiple training runs on same plot** — shows iteration, not just one lucky run
2. **Co-evolution story** — bugs found during training that improved the environment
3. **Real `trainer.train()` calls** — not manual rollout loops

**Document EVERY bug/surprise** in `docs/training_log.md` — this IS the Statement 4 story.

---

## Day 1 — Phase 2: Eval + Artifacts (Hours 6–10)

```bash
# Base vs trained comparison
python eval_compare.py --model-path outputs/run2/checkpoint --episodes 20 --output-dir outputs/eval

# Overlay reward curves from all runs
python plot_rewards.py --csv outputs/run1/reward_log.csv --out results/reward_curve_run1.png
python plot_rewards.py --csv outputs/run2/reward_log.csv --out results/reward_curve_run2.png
```

### Extract Key Episodes

Look in `logs/transcripts.jsonl` for:
1. **Episode 1 failure** — random behavior, times out
2. **Mid-training breakthrough** — first correct phase ordering
3. **Late success** — agent enriches for subgroup, trial succeeds

### Push Checkpoint

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/run2/checkpoint',
    repo_id='Roopalgn/clinical-trial-designer-grpo',
    repo_type='model',
    commit_message='GRPO trained checkpoint from onsite H100'
)
"
```

---

## Day 2 — Phase 3: Deliverables (Hours 0–6)

### Roopal: Content
1. Fill `docs/mini_blog_draft.md` — replace all `[FILL ONSITE]` with real numbers
2. Publish mini-blog on HuggingFace (New Post → paste markdown → upload reward_curve.png)
   - Tags: `openenv grpo clinical-trial reinforcement-learning trl hackathon meta-pytorch`
3. Update `README.md` § Results with actual training metrics
4. Update `ARCHITECTURE.md` § Post-Training Results
5. Fill `docs/internal/pitch_notes.md` with real reward numbers

### Suyash: Artifacts
1. Generate final reward curve: `python plot_rewards.py --csv outputs/run2/reward_log.csv --out results/reward_curve.png`
2. Run final eval: `python eval_compare.py --model-path outputs/run2/checkpoint --episodes 20`
3. Commit everything:
   ```bash
   git add outputs/ results/ logs/
   git commit -m "feat: onsite training results"
   git push origin main
   ```
4. Verify HF Space still responds after code changes

---

## Day 2 — Phase 4: Pitch Prep (Hours 6–8)

### 3-Minute Structure

```
0:00–0:30  Hook: "Can a 7B model learn to design a clinical trial from scratch?"
           Problem: $2.6B per drug, 90% fail, partially observable
0:30–1:30  Environment: hidden state → 19 actions → 10-phase workflow → 5-tier curriculum
           Demo: show dashboard, architecture diagram
1:30–2:30  Results: reward curve, before/after episode, baseline comparison
           Key moment: agent discovered biomarker stratification on its own
2:30–3:00  Co-evolution: bugs found during training → env improved → self-improvement loop
```

### Likely Judge Questions

| Question | Answer |
|----------|--------|
| Why not real clinical data? | Grounded in rpact/scipy.stats power calcs + FDA ICH E9 rules. Not toy heuristics. |
| How does difficulty adapt? | Per-scenario mastery tracking, weak-spot targeting, adversarial compound challenges at expert tier. |
| What did the agent learn? | Show episode 1 vs episode N side by side. |
| How is this different from a chatbot? | Hidden ground truth, objective verification, no shortcut exploits. |
| Where did you train? | All training onsite on HF H100 credits. Pipeline validated pre-event with dry-runs. |

---

## Training-Failure Fallback

> **Trigger:** No learning signal after 2 hours of debugging.

1. Switch to Qwen2.5-1.5B + 4-bit quant
2. Run 5 dry-run episodes with annotated reward narrative
3. Document failure analysis in `docs/training_log.md` as Statement 4 evidence
4. **Even flat curves with analysis beat zero evidence**

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OOM on 7B | `--model-size 3b` or `1.5b` |
| HF Space down | `uvicorn server.app:app --port 8000` locally |
| No learning signal | Check reward variance with 10 dry-run episodes |
| Git push fails | `git pull --rebase origin main` then push |
| Checkpoint too large | Upload only LoRA adapter, not full model |

---

## Disk Budget

| Item | Size |
|------|------|
| Base model (1.5B, BF16) | ~3 GB |
| Base model (7B, BF16) | ~14 GB |
| LoRA checkpoint | ~50–200 MB |
| Reward CSV (100 ep) | ~10 KB |
| Transcripts JSONL | ~5–50 MB |
| **Total (conservative)** | **~20 GB** |
