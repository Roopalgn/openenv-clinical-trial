# Training Log — OpenEnv Clinical Trial Designer

> **Purpose:** Document every bug, surprise, and fix discovered during training.
> This IS the Statement 4 co-evolution story: environment improved *because* of the agent.
> **Fill this live during the short validation run first, then expand for the onsite judged runs.** Judges read this to evaluate self-improvement.

---

## Fast Fill Sheet (Use This First)

Fill these fields as soon as Suyash sends the first real run output. Everything below can be derived from this block plus the run artifacts.

| Field | Value |
|------|-------|
| Date | `2026-04-25` |
| Platform | `Colab GPU (Tesla T4)` |
| Commit hash trained from | `[FILL]` |
| Exact command used | `[FILL]` |
| Output directory | `outputs/grpo` |
| Runtime | `Duration not exported separately; completed_at=2026-04-25T12:32:29.445902+00:00` |
| Mean reward | `17.5157039642334` |
| Final reward | `17.29707145690918` |
| Best reward | `18.528873443603516` |
| Worst reward | `16.508432388305664` |
| Success rate | `TBD from longer eval` |
| Avg steps / episode | `TBD from training_summary.json` |
| Final curriculum tier | `TBD (not exported by notebook summary)` |
| Early bad episode ID | `[FILL]` |
| Best late episode ID | `[FILL]` |
| One learned behavior | `The policy maintained stable positive rewards after parser/precision fixes instead of collapsing to constant -2 outputs.` |
| One bug or surprise | `Upload initially failed with 401 until the model repo was explicitly created; auth was valid, but repo creation needed to be made explicit.` |

### Sources of Truth

- Metrics source of truth: `training_summary.json`
- Curve source of truth: reward PNG generated from `reward_log.csv`
- Qualitative source of truth: episode IDs and transcripts/logs from the run

### If The Short Run Fails

Capture these before doing anything else:

| Field | Value |
|------|-------|
| Failure stage | `[FILL: install / model download / CUDA / OOM / training crash / plotting / eval]` |
| Exact command used | `[FILL]` |
| Dry-run passed? | `[FILL]` |
| Tests passed? | `[FILL]` |
| Last relevant logs | `[FILL]` |
| Fix attempted | `[FILL]` |
| Why Colab/HF-credit strategy was reasonable | `[FILL]` |

---

## Run 1 — Short Validation Pass (Use for the current Colab run)

**Goal:** Prove the fixed training path works end to end and produce the first real artifacts without touching the submission Space.

**Date:** `2026-04-25`
**Platform:** `Colab GPU (Tesla T4)`
**Commit hash:** `[FILL]`
**Command:**
```bash
python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 20 --seed 42 --output-dir outputs/grpo
```

**Config:**
| Param | Value |
|-------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Precision | `fp16 on T4 (bf16 disabled)` |
| LoRA rank | `8` |
| Batch size | `8` |
| Seq length | `2048` |
| Grad accum | `1` |
| Seed | 42 |
| Episodes | `20` |

### Result Summary

| Metric | Value |
|--------|-------|
| Mean reward | `17.5157039642334` |
| Final reward (last ep) | `17.29707145690918` |
| Best reward | `18.528873443603516` |
| Worst reward | `16.508432388305664` |
| Success rate | `TBD` |
| Avg steps/episode | `Not exported by the notebook training summary; short eval used a 15-step cap` |
| Curriculum tier reached | `TBD (not exported by notebook summary)` |
| Runtime | `Completed at 2026-04-25T12:32:29.445902+00:00` |
| OOM? | `No` |
| Reward curve PNG | `outputs/grpo/reward_curve.png` |

### Reward Curve Signal

- [x] Positive slope (learning signal present, though shallow)
- [ ] Flat (reward weights need adjustment → see Bug #X below)
- [ ] Negative slope (reward too sparse / action parsing broken)

### Immediate Next Actions After This Run

- [ ] Fill the trained row in `README.md`
- [ ] Fill the metric block in `docs/mini_blog_draft.md`
- [ ] Fill the top fill-sheet in `docs/internal/pitch_notes.md`
- [ ] Select one early failure episode and one late better episode
- [ ] Record one real bug, surprise, or learning observation below

---

## Co-Evolution Bugs (Statement 4)

> Each bug found during training = evidence the environment self-improved.
> Minimum 2 required for Phase B gate. Aim for 4–5.

---

### Bug #1 — _(Short title, e.g. "Reward too generous at warmup tier")_

**Discovered:** Episode _(N)_, Run 1
**Symptom:**
> _(What did you observe? e.g. "Agent achieving +8.0 reward on episode 3 — too fast, no learning signal after ep 5")_

**Root Cause:**
> _(Why did it happen? e.g. "r_terminal_success weight of 8.0 dominates before agent learns workflow")_

**Fix Applied:**
> _(What changed? e.g. "Reduced r_terminal_success to 5.0 at warmup tier, added curriculum gate requiring 3 correct phase-order steps before terminal bonus")_

**Evidence Fix Worked:**
> _(What changed in reward curve after fix? e.g. "Reward slope increased from +0.02 to +0.18 per episode")_

---

### Bug #2 — _(Short title)_

**Discovered:** Episode _(N)_, Run _(1/2)_
**Symptom:**
> 

**Root Cause:**
> 

**Fix Applied:**
> 

**Evidence Fix Worked:**
> 

---

### Bug #3 — _(Short title)_

**Discovered:** Episode _(N)_, Run _(1/2)_
**Symptom:**
> 

**Root Cause:**
> 

**Fix Applied:**
> 

**Evidence Fix Worked:**
> 

---

### Bug #4 — _(Short title)_

**Discovered:** Episode _(N)_, Run _(1/2)_
**Symptom:**
> 

**Root Cause:**
> 

**Fix Applied:**
> 

**Evidence Fix Worked:**
> 

---

## Episode Transcripts (Key 3)

> For the short run, capturing just the first and best episode pair is enough. Expand to all 3 after the next larger run.

### Transcript A — Early Failure

**Episode ID:** `[FILL]`
**Scenario:** `[FILL]`
**Outcome:** `[FILL]`
**Total Reward:** `[FILL]`
**Narrative shell:** `Early in training the agent [FILL failure pattern], which led to [FILL outcome].`

---

### Transcript B — Best Available Late Episode

**Episode ID:** `[FILL]`
**Scenario:** `[FILL]`
**Outcome:** `[FILL]`
**Total Reward:** `[FILL]`
**Narrative shell:** `Later in training the agent [FILL improved workflow], which led to [FILL measurable outcome].`

---

## Run 2 — Onsite H100, Scale Up (50 episodes, 3B OR 100 episodes, 1.5B)

**Date:** `[FILL ONSITE]`
**Decision from Run 1:** `[FILL ONSITE: e.g. "Run 1 showed positive slope → scale to 3B" / "Run 1 flat → debug reward first"]`

**Command:**
```bash
# Option A: Scale model
python train.py --model-size 3b --model-path Qwen/Qwen2.5-3B-Instruct --episodes 50 --seed 42 --output-dir outputs/run2

# Option B: Scale episodes (same model)
python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 100 --seed 42 --output-dir outputs/run2
```

### Result Summary

| Metric | Value |
|--------|-------|
| Final reward (last ep) | `[FILL ONSITE]` |
| Best reward | `[FILL ONSITE]` |
| Success rate | `[FILL ONSITE]`% |
| Curriculum tier reached | `[FILL ONSITE]` |
| Improvement vs Run 1 | `[FILL ONSITE]` |

---

## Episode Transcripts (Key 3)

> Extract these from the full run after Run 2.
> These are shown to judges as before/after comparison.

### Transcript A — Episode 1 (Failure)

**Episode:** 1
**Scenario:** _(fill)_
**Outcome:** FAIL — _(fill reason, e.g. "Agent skipped dose escalation, submitted FDA review with no safety data")_
**Total Reward:** _(fill)_
**Key Moment:**
```
Step 3: Agent action → submit_to_fda_review
Observation: "ERROR: No Phase I safety data. FDA submission rejected."
Reward: r_validity=-0.5, r_ordering=-0.3
```

---

### Transcript B — Mid-training Breakthrough

**Episode:** _(fill, e.g. 8–12)_
**Scenario:** _(fill)_
**Outcome:** PARTIAL SUCCESS — _(fill, e.g. "Agent learned phase ordering but still underpowered")_
**Total Reward:** _(fill)_
**Key Moment:**
```
Step X: Agent action → _(fill)_
Observation: _(fill)_
```

---

### Transcript C — Late Episode Success

**Episode:** _(fill, e.g. 18–20)_
**Scenario:** _(fill)_
**Outcome:** SUCCESS — _(fill)_
**Total Reward:** _(fill)_
**Key Moment:**
```
Step X: Agent action → _(fill)_
Observation: _(fill)_
```

---

## Eval Compare Output (Run 1)

> Run: `python eval_compare.py --model-path outputs/run1/checkpoint --episodes 10 --output-dir outputs/eval`

| Policy | Success Rate | Avg Reward | Power ≥ 0.80 | Subgroup Found |
|--------|-------------|-----------|--------------|----------------|
| Random baseline | ~5% | `[FILL ONSITE]` | `[FILL ONSITE]` | `[FILL ONSITE]` |
| Run 1 checkpoint | `[FILL ONSITE]` | `[FILL ONSITE]` | `[FILL ONSITE]` | `[FILL ONSITE]` |
| Run 2 checkpoint | `[FILL ONSITE]` | `[FILL ONSITE]` | `[FILL ONSITE]` | `[FILL ONSITE]` |

---

## Key Observations for Pitch

> Fill after Run 1. These go directly into `docs/pitch_notes.md` and the mini-blog.

1. **What the agent learned:** `[FILL ONSITE]`
2. **What surprised us:** `[FILL ONSITE]`
3. **What we had to fix:** `[FILL ONSITE]`
4. **What the reward curve shows:** `[FILL ONSITE]`
