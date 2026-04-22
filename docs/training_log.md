# Training Log — OpenEnv Clinical Trial Designer

> **Purpose:** Document every bug, surprise, and fix discovered during training.
> This IS the Statement 4 co-evolution story: environment improved *because* of the agent.
> **Fill this live during Kaggle training runs.** Judges read this to evaluate self-improvement.

---

## Run 1 — Kaggle T4, Qwen2.5-1.5B-Instruct (20 episodes)

**Date:** _(fill)_
**Platform:** Kaggle (T4 / P100)
**Command:**
```bash
python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 20 --seed 42
```

**Config:**
| Param | Value |
|-------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Quantization | 4-bit (Unsloth) |
| LoRA rank | 8 |
| Batch size | 1 |
| Seq length | 2048 |
| Grad accum | 4 |
| Seed | 42 |
| Episodes | 20 |

### Result Summary

| Metric | Value |
|--------|-------|
| Final reward (ep 20) | _(fill)_ |
| Best reward | _(fill)_ |
| Worst reward | _(fill)_ |
| Success rate | _(fill)_% |
| Avg steps/episode | _(fill)_ |
| Curriculum tier reached | _(fill)_ |
| Runtime | _(fill)_ hrs |
| OOM? | No / Yes → fix: _(fill)_ |

### Reward Curve Signal

- [ ] Positive slope (learning signal present)
- [ ] Flat (reward weights need adjustment → see Bug #X below)
- [ ] Negative slope (reward too sparse / action parsing broken)

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

## Run 2 — Kaggle T4/P100 (50 episodes, 1.5B OR 20 episodes, 3B)

**Date:** _(fill)_
**Decision from Run 1:** _(e.g. "Run 1 showed positive slope → scale to 50 episodes" / "Run 1 flat → debug reward first")_

**Command:**
```bash
# Option A: Scale episodes (same model)
python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 50 --seed 42

# Option B: Scale model
python train.py --model-size 3b --model-path Qwen/Qwen2.5-3B-Instruct --episodes 20 --seed 42
```

### Result Summary

| Metric | Value |
|--------|-------|
| Final reward (last ep) | _(fill)_ |
| Best reward | _(fill)_ |
| Success rate | _(fill)_% |
| Curriculum tier reached | _(fill)_ |
| Improvement vs Run 1 | _(fill)_ |

---

## Episode Transcripts (Key 3)

> Extract these from `logs/transcripts.jsonl` after Run 1.
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

> Run: `python eval_compare.py --checkpoint logs/checkpoint_ep20/ --episodes 10`

| Policy | Success Rate | Avg Reward | Power ≥ 0.80 | Subgroup Found |
|--------|-------------|-----------|--------------|----------------|
| Random baseline | ~5% | -1.5 | 3% | 2% |
| Run 1 checkpoint | _(fill)_ | _(fill)_ | _(fill)_ | _(fill)_ |
| Run 2 checkpoint | _(fill, after Run 2)_ | _(fill)_ | _(fill)_ | _(fill)_ |

---

## Key Observations for Pitch

> Fill after Run 1. These go directly into `docs/pitch_notes.md` and the mini-blog.

1. **What the agent learned:** _(e.g. "By episode 15, agent reliably performs dose escalation before FDA submission — a behaviour that never appeared in episodes 1–5")_
2. **What surprised us:** _(e.g. "Agent discovered the EGFR+ enrichment strategy without being told it existed — found it by repeatedly running biomarker stratification actions")_
3. **What we had to fix:** _(e.g. "Phase skip penalty of -0.3 was too harsh at warmup tier — agent got stuck in local minimum avoiding regulatory actions entirely")_
4. **What the reward curve shows:** _(e.g. "Clear positive slope from ep 1 (-2.1) to ep 20 (+4.8). Breakpoint at ep 9 aligns with bug #2 fix applied mid-run")_
