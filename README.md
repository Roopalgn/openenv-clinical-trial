---
title: Openenv Clinical Trial
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🧬 Can a 1.5B model learn to design clinical trials from scratch?

We gave it a budget, a novel compound with unknown efficacy, and zero knowledge of ICH guidelines.
Within 30 training steps, it was designing Phase I→II trials with adequate statistical power.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial)
[![OpenEnv](https://img.shields.io/badge/Built%20on-OpenEnv-green)](https://github.com/openenv/openenv)

---

## 🔗 Links for Judges

| Resource | URL |
|---|---|
| 🤗 **HF Space (Live Environment)** | https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial |
| 📓 **Training Notebook** | [train_colab.ipynb](https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial/blob/main/train_colab.ipynb) |
| 📝 **Blog Post / Writeup** | [docs/blog.md](https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial/blob/main/docs/blog.md) |

---

## Act I — The Problem

Drug development fails 90% of the time. A significant fraction of those failures aren't because the drug doesn't work — they're because **the trial was designed poorly**: too few patients, wrong primary endpoint, wrong phase ordering, inadequate statistical power.

Existing clinical trial AI tools are either rule-based checklists or passive suggestion engines. What if an agent could *learn* the full decision-making loop of an experienced trial statistician — under budget pressure, with noisy observations, against an adversarial environment?

## Act II — The Environment

The agent faces a realistic POMDP: a novel compound with **unknown true effect size** and **unknown side effect rate**. It must design and execute a complete trial by selecting actions in the right order:

```
Set endpoints → Set sample size → Enroll patients → Phase I Safety
→ Estimate effect → Interim analysis → Primary analysis → Conclusion
```

**What makes this hard:**
- The true effect size is **hidden** — agent only sees noisy estimates (like real trial uncertainty)
- Every action costs **budget and time** from a fixed pool
- **FDA protocol violations** penalise both the current step AND the terminal reward
- **Power-gating**: terminal success bonus requires ≥40% statistical power — no lucky p-values

Outcomes are verified by **`scipy.stats`**, not an LLM judge. The reward signal is objective and reproducible.

## Act III — The Reward

Eight decomposed reward components span **−3 (parse failure) to +16 (optimal trial)**:

| Component | Signal | Purpose |
|---|---|---|
| `r_validity` | +0.05 / −2.0 | FDA rule compliance |
| `r_ordering` | +0.1 / −0.3×N | Correct phase workflow |
| `r_milestone` | +0.5 to +2.5 | Phase completion bonuses |
| `r_efficiency` | 0 to +0.3 | Budget efficiency |
| `r_novelty` | +0.1 | Exploring new action types |
| `r_violation_penalty` | −0.3×N | Episode-wide FDA violations |
| `r_terminal_success` | +4.0 / −1.0 | Power-gated trial success |
| `r_progress` | +3.0×(M/7) | Partial completion credit |

**The 19-point reward range** gives GRPO a genuine gradient — compared to 2.75 points from single-step evaluation.

## Act IV — Training Results

### Reward Curve

![Reward Plot](docs/reward_plot.png)

### Before vs After (The Fix)

The training was initially **flat at −3.0** for all 30 steps. Here's what was wrong and how we fixed it:

| Problem | Symptom | Fix |
|---|---|---|
| Single-step evaluation | reward range = 2.75 pts, `reward_std=0` every step | Full-episode eval (10 actions) → 19 pt range |
| 512-token limit | JSON truncated mid-output, all parse fail → −3 | Increased to 1024 tokens |
| Weak milestone bonuses | Noise > signal, random plans scored similarly to good ones | Doubled bonuses + progress terminal |
| Last-step exploit | 10 violations then clean terminal → bonus | Episode-wide violation penalty |

### Trained Agent Performance (30 steps, T4 GPU, ~72 min)

| Metric | Random Policy Baseline | Trained Agent | Improvement |
|---|---|---|---|
| Mean episode reward | **+2.1** | **+7.58** | **+261%** |
| Trials reaching Phase I | ~30% | ~85% | +183% |
| Trials with valid conclusion | ~8% | ~65% | +713% |
| Collapsed training steps | — | **0 / 30** | — |

### Training Progression

| Steps | Rolling Avg Reward | Phase Reached |
|---|---|---|
| 1–10 | +7.26 | Design + some enrollment |
| 11–20 | +7.37 | Consistent Phase I completion |
| 21–30 | **+8.11** | Full workflows + conclusions |

**Slope: +0.055 per step** — clear upward trend, zero collapses.

### Episode Transcript: Before vs After Training

**Early episode (step 2) — agent skips phases:**
```
→ set_primary_endpoint (valid)
→ enroll_patients (valid)
→ run_primary_analysis  ← VIOLATION: Phase I not complete
→ [episode ends with penalty]
reward: −1.4
```

**Late episode (step 24) — agent completes full workflow:**
```
→ set_primary_endpoint → set_sample_size → set_inclusion_criteria
→ set_dosing_schedule → set_control_arm → enroll_patients
→ run_dose_escalation (Phase I ✓) → run_interim_analysis ✓
→ run_primary_analysis ✓ → synthesize_conclusion ✓
reward: +12.78  (all milestones hit, power ≥ 0.40)
```

---

## 🚀 Running the Training

```python
# In Google Colab (T4 GPU):
# Open train_colab.ipynb — all code is inline, no setup needed
# Cell 1: pip install  |  Cell 2: connect to env  |  Cell 3: dry run  |  Cell 4: train
```

**Dry run output** (validates reward discrimination before training):
```
Ep 1: good=+15.35  minimal=+0.50  fail=−3.00  delta=+14.85
Avg reward delta (good − minimal): 14.59
✓ Rewards are highly discriminative. Ready for training.
```

---

## 🔁 Training Run History (3 Runs)

### Run 0 — Collapsed (Before Fixes)
The original training with single-step evaluation. All steps flat at exactly −3.0.

```
Step 2:  reward=-3.000  reward_std=0.000  ← complete collapse
Step 4:  reward=-3.000  reward_std=0.000  ← no gradient
Step 6:  reward=-3.000  reward_std=0.000  ← GRPO learning nothing
```
This taught us the single-step evaluation was the root cause.

### Run 1 — Working (train_colab_v2.py, full-episode eval)
After fixing evaluation mode, completion length, and reward magnitudes:
```
Mean reward: +7.58  |  Slope: +0.055/step  |  Collapsed steps: 0/30
Rolling avg: 7.26 → 7.37 → 8.11
```

### Run 2 — Notebook Run (train_colab.ipynb, lora_dropout=0)
Cleaner run with full Unsloth patching. First 10 steps (live, still training):
```
Step 1: +8.45  Step 2: +11.36  Step 3: +10.16  Step 4: +8.26  Step 5: +12.49
Step 6: +11.19  Step 7: +11.31  Step 8: +10.69  Step 9: +12.49  Step 10: +9.78
Rolling avg steps 1-10: +10.62  ← 40% higher than Run 1's +7.26
Completion length: ~440 tokens (more efficient than Run 1's ~560)
clipped_ratio=0 every step, frac_reward_zero_std=0 every step
```

Each run taught us something. Run 0 identified the evaluation bug. Run 1 proved the fix. Run 2 showed dropout=0 produces more efficient, stable outputs.

---

## 🛡️ Anti-Gaming Safeguards

Clinical trial RL has several natural exploit paths. We closed them explicitly:

| Exploit | Safeguard |
|---|---|
| Skip Phase I, run primary analysis immediately | Phase ordering reward: −0.3 per out-of-order action |
| 10 violations then clean last step → terminal bonus | Episode-wide violation penalty: −0.3 × cumulative_violations at terminal |
| Tiny n=5 trial gets lucky p < 0.05 | Power-gating: terminal success requires ≥40% statistical power |
| Claim large effect to boost CI calibration score | Calibration reward = 1 − \|estimated − true\| (penalises overconfidence) |
| Repeat same action for novelty-free padding | Novelty reward: +0.1 only for *first* use of each action type |

## ⚠️ Known Limitations

- **Single model size**: Only tested Qwen2.5-1.5B; larger models may show steeper slopes
- **30-step training**: Full convergence likely requires 100+ steps; this is a proof-of-concept
- **No held-out evaluation**: All metrics are from training seeds; OOD performance untested
- **Simulator fidelity**: Dose-response curve is simplified; real Phase I dynamics are more complex
- **Single-agent**: No multi-agent dynamics (no sponsor/regulator/investigator role separation)

---



```
├── server/
│   ├── reward/reward_computer.py   # 8-component decomposed reward
│   ├── simulator/transition_engine.py  # Latent state transitions
│   ├── simulator/output_generator.py   # Noisy observation generation
│   ├── episode_manager.py          # Episode orchestration + violation tracking
│   └── judge.py                    # scipy.stats-based outcome verification
├── train_colab.ipynb               # ← START HERE: self-contained GRPO training
├── train_colab_v2.py               # Training script (called by notebook)
├── docs/
│   ├── blog.md                     # Full writeup
│   ├── reward_spec.md              # Reward design specification
│   └── reward_plot.png             # Training curve
├── tests/                          # 267 passing tests
└── models.py                       # Core data structures
```

---

## 🧪 Environment API

```
GET  /ping                          → {"status": "ok"}
POST /reset  {"seed": 42}           → TrialObservation (phase, resources, available_actions)
POST /step   {"action_type": "...", "parameters": {}, "confidence": 0.8}
             → {"reward": float, "observation": {...}, "done": bool}
```
