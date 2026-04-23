# Onsite Checklist — Apr 25–26 (Scaler Campus)

> **Goal:** Execute ALL GRPO training on H100, generate deliverables, rehearse pitch.
> This is the step-by-step terminal sequence. Follow in order.

---

## Pre-Arrival (before leaving home)

- [ ] Government-issued ID + college/company ID packed
- [ ] Laptop charged, charger packed
- [ ] `HF_TOKEN` saved in a secure note (you'll need it onsite)
- [ ] This checklist open on phone as backup

---

## Phase 0: Setup (First 30 minutes)

### Clone and install

```bash
git clone https://github.com/Roopalgn/openenv-clinical-trial.git
cd openenv-clinical-trial
pip install -e ".[train]"
pip install matplotlib   # for reward curve plots
```

### Environment variables

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
```

### Verify environment server

```bash
# Option A: Use the live HF Space (simplest — no setup needed)
curl https://roopalgn-openenv-clinical-trial.hf.space/ping
# Should return: {"status":"ok"}

# Option B: Run locally (if HF Space is slow or down)
uvicorn server.app:app --host 0.0.0.0 --port 8000 &
curl http://localhost:8000/ping
```

### Verify pipeline (dry-run, ~30 seconds)

```bash
python train.py --dry-run --episodes 2 --model-size 1.5b --output-dir outputs/verify
# Should complete with reward CSV + JSON summary
cat outputs/verify/training_summary.json
rm -rf outputs/verify
```

---

## Phase 1: First Training Run (Hours 0–2)

### Model size fallback ladder

Try in this order. Move to next if OOM or too slow:

| Priority | Command | VRAM (BF16) | Expected time (20 ep) |
|----------|---------|-------------|----------------------|
| 1st | `--model-size 7b --model-path Qwen/Qwen2.5-7B-Instruct` | ~14 GB | ~5 hrs |
| 2nd | `--model-size 3b --model-path Qwen/Qwen2.5-3B-Instruct` | ~6 GB | ~3 hrs |
| **3rd (safe)** | `--model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | ~1 hr |

**Start with 1.5B for fast signal, then scale up if time allows.**

### Run 1: Fast signal check (1.5B, 20 episodes)

```bash
python train.py \
    --model-size 1.5b \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 20 \
    --seed 42 \
    --output-dir outputs/run1

# Monitor live:
tail -f outputs/run1/reward_log.csv
```

### After Run 1: Check signal

```bash
python plot_rewards.py --csv outputs/run1/reward_log.csv --out results/reward_curve_run1.png
cat outputs/run1/training_summary.json
```

**Decision point:**
- Positive slope → proceed to Run 2 (scale up model or episodes)
- Flat → debug reward weights, check `docs/training_log.md`, try different curriculum tier
- Negative → check `_build_action_from_text` parsing, reward component weights

---

## Phase 2: Scale Up (Hours 2–6)

### Run 2: Scale model OR episodes

```bash
# Option A: Scale to 3B (if 1.5B showed signal)
python train.py \
    --model-size 3b \
    --model-path Qwen/Qwen2.5-3B-Instruct \
    --episodes 50 \
    --seed 42 \
    --output-dir outputs/run2

# Option B: More episodes with 1.5B (if 3B OOMs or too slow)
python train.py \
    --model-size 1.5b \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 100 \
    --seed 42 \
    --output-dir outputs/run2
```

### After Run 2

```bash
python plot_rewards.py --csv outputs/run2/reward_log.csv --out results/reward_curve_run2.png
python eval_compare.py --base-only --episodes 10 --output-dir outputs/eval
# Then with trained checkpoint:
python eval_compare.py --model-path outputs/run2/checkpoint --episodes 10 --output-dir outputs/eval
```

---

## Phase 3: Eval + Artifacts (Hours 6–8)

### Generate eval comparison

```bash
python eval_compare.py \
    --model-path outputs/run2/checkpoint \
    --episodes 20 \
    --seed 42 \
    --output-dir outputs/eval
```

### Extract episode transcripts

Look in `logs/transcripts.jsonl` for:
1. **Episode 1 failure** — early random behavior
2. **Mid-training breakthrough** — first time agent follows phase order
3. **Late success** — agent enriches for subgroup, trial succeeds

### Push checkpoint to HF Hub

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
print('Uploaded!')
"
```

### Commit training artifacts to repo

```bash
git add outputs/ results/ logs/
git commit -m "feat: onsite training results (Run 1 + Run 2)"
git push origin main
```

---

## Phase 4: Deliverables (Hours 8–12)

### Roopal: Fill deliverable placeholders

1. Open `docs/mini_blog_draft.md` — replace all `[FILL ONSITE]` with real numbers
2. Open `docs/internal/pitch_notes.md` — replace all `[FILL ONSITE]` with real numbers
3. Update `README.md` § Results with actual metrics
4. Update `ARCHITECTURE.md` § Post-Training Results

### Suyash: Final artifacts

1. Generate final reward curve: `python plot_rewards.py --csv outputs/run2/reward_log.csv --out results/reward_curve.png`
2. Verify HF Space still responds after code changes
3. Run final `eval_compare.py` and save `results/eval_report.json`

### Publish mini-blog

1. Go to huggingface.co → New Post
2. Copy content from `docs/mini_blog_draft.md` (with real data filled in)
3. Upload `results/reward_curve.png` as inline image
4. Add tags: `openenv grpo clinical-trial reinforcement-learning trl hackathon meta-pytorch`

---

## Phase 5: Pitch Prep (Last 2 hours)

```
0:00–0:15  Hook: "Can a 7B model learn to design a clinical trial from scratch?"
0:15–0:40  Problem: $2.6B per drug, 90% fail, partially observable
0:40–1:30  Env demo: architecture → hidden state → 19 actions → curriculum
1:30–2:00  Reward: decomposed (8+7), potential shaping, math verification
2:00–2:40  Results: reward curve, before/after episode, baseline comparison
2:40–3:00  Close: "Agent discovered biomarker stratification on its own"
```

- [ ] Rehearse once with timer
- [ ] Prepare Q&A answers (see `docs/internal/pitch_notes.md`)
- [ ] Have dashboard.html open as backup demo

---

## Training-Failure Fallback

> **Trigger:** No learning signal (flat or negative reward curve) after 2 hours of debugging.

1. **Switch model:** Qwen2.5-1.5B + 4-bit quant for fastest iteration
   ```bash
   python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 20 --seed 42 --output-dir outputs/fallback
   ```
2. **Run 5 dry-run episodes** with detailed annotated reward narrative
   ```bash
   python train.py --dry-run --episodes 5 --output-dir outputs/dryrun_fallback
   ```
3. **Document the failure** in `docs/training_log.md` as Statement 4 co-evolution evidence — explain why it didn't converge and what you learned
4. **Minimum acceptable submission:** dry-run evidence + annotated reward evolution + failure analysis in `docs/training_log.md`
5. **Even flat curves with analysis beat zero evidence** — judges score effort and understanding, not just results

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OOM on 7B | Use `--model-size 3b` or `1.5b` |
| HF Space down | Run env locally: `uvicorn server.app:app --port 8000` |
| No learning signal | Check reward variance. Run `--dry-run` with 10 episodes, check if rewards differ. |
| `plot_rewards.py` fails | `pip install matplotlib` |
| Git push fails | `git pull --rebase origin main` then push again |
| Checkpoint too large for HF Hub | Upload only the LoRA adapter (small), not full model |

---

## Disk Budget

| Item | Size | Path |
|------|------|------|
| Base model (1.5B, BF16) | ~3 GB | cached by HF |
| Base model (7B, BF16) | ~14 GB | cached by HF |
| LoRA checkpoint | ~50–200 MB | `outputs/runN/checkpoint/` |
| Reward CSV (100 ep) | ~10 KB | `outputs/runN/reward_log.csv` |
| Transcripts JSONL | ~5–50 MB | `logs/transcripts.jsonl` |
| Reward curve PNG | ~200 KB | `results/reward_curve.png` |
| **Total (conservative)** | **~20 GB** | |
