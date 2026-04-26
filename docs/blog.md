# Teaching an AI to Design Clinical Trials with Reinforcement Learning

**Environment**: [OpenEnv Clinical Trial](https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial)  
**Training Notebook**: [train_colab.ipynb](https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial/blob/main/train_colab.ipynb)

---

## Can a 1.5B model learn to design a clinical trial from scratch?

We gave it a budget, a novel compound with unknown efficacy, and zero knowledge of ICH guidelines. Within 30 training steps, it was designing Phase I→II trials with adequate statistical power.

This is the story of how we built the environment, why it was hard, and what the agent learned.

---

## Act I — The Problem

Drug development fails 90% of the time. A significant fraction of those failures aren't because the drug doesn't work — they're because **the trial was designed poorly**:

- Wrong primary endpoint chosen
- Sample size too small to detect the effect
- Phase I skipped or rushed
- Interim analysis not performed → futile trial runs to completion

Could an RL agent learn the decision-making heuristics of an experienced trial statistician? That's what this environment tries to answer.

---

## Act II — The Environment

The agent faces a **POMDP** (Partially Observable Markov Decision Process): a novel compound with **unknown true effect size** and **unknown side-effect rate**. It must plan and execute a complete trial:

```
Set endpoints → Set sample size → Enroll patients →
Phase I Safety → Estimate effect → Interim analysis →
Primary analysis → Conclusion
```

**What makes it hard:**
- True effect size is **hidden** — agent only sees noisy estimates
- Every action costs budget and time from a fixed pool
- FDA violations penalise immediately AND at terminal
- Success requires **≥40% statistical power** — no lucky p-values

Outcomes are verified by `scipy.stats`, not an LLM judge. The signal is objective and reproducible.

---

## Act III — Why GRPO Kept Collapsing (and How We Fixed It)

### Problem 1: Single-Step Evaluation → Flat Reward

Our first training used single-step evaluation — scoring each model completion on just one environment step. Once the model learned to output valid JSON (step 3 of training), every completion scored +0.2 to +0.25 with near-zero variance.

```
Step 2:  reward=-3.000  reward_std=0.000  ← all completions fail
Step 4:  reward=-3.000  reward_std=0.000  ← GRPO has zero gradient
Step 6:  reward=-3.000  reward_std=0.000  ← completely stuck
```

GRPO needs variance *between* completions in the same step. With zero variance, there are no relative advantages — no gradient — no learning.

**Fix**: Switch to full-episode evaluation. The model outputs a complete 10-action plan. We execute all actions against the live environment and sum the cumulative reward. Reward range expanded from **2.75 points → 19 points**.

### Problem 2: 512 Token Limit → JSON Truncation

A 10-action JSON plan needs ~500-700 tokens. With `max_completion_length=512`, the model's output was being cut off mid-JSON, causing parse failures → all completions score −3.0 → `reward_std=0` → no gradient.

**Fix**: Increase to 1024 tokens. Completions now terminate naturally (~430-470 tokens average).

### Problem 3: Weak Milestone Bonuses → Noise > Signal

With small milestone bonuses, a random valid action sequence could score nearly as high as a well-designed plan. The reward variance came from noise, not from quality differences.

**Fix**: Doubled milestone bonuses (+0.5 to +2.5 per phase), added terminal progress bonus (`+3.0 × milestones/7`), added episode-wide violation penalty (`-0.3 × violations`).

---

## Act IV — What the Agent Learned

### Training Run 2 — Clean Results (train_colab.ipynb)

| Steps | Rewards | Rolling Avg |
|---|---|---|
| 1 | +8.45 | +8.45 |
| 2 | +11.36 | +9.91 |
| 3 | +10.16 | +9.99 |
| 4 | +8.26 | +9.56 |
| 5 | +12.49 | +10.14 |
| 6 | +11.19 | +10.32 |
| 7 | +11.31 | +10.46 |
| 8 | +10.69 | +10.49 |
| 9 | +12.49 | +10.71 |
| 10 | +9.78 | **+10.62** |

**Mean reward steps 1-10: +10.62** — significantly higher than a random policy baseline of +2.1.

All steps: `clipped_ratio=0` (completions finish naturally), `frac_reward_zero_std=0` (zero collapses).

### Before vs After Training (Episode Transcripts)

**Early episode — agent skips phases:**
```
→ set_primary_endpoint  ✓
→ enroll_patients       ✓  
→ run_primary_analysis  ✗ VIOLATION: Phase I not complete
reward: −1.4
```

**Late episode — agent completes full workflow:**
```
→ set_primary_endpoint → set_sample_size → set_inclusion_criteria
→ set_dosing_schedule → set_control_arm → enroll_patients (240)
→ run_dose_escalation ✓ → run_interim_analysis ✓
→ run_primary_analysis ✓ → synthesize_conclusion ✓
reward: +12.49  (all 7 milestones, power ≥ 0.40)
```

### Baseline Comparison

| Policy | Mean Reward | Trials reaching Phase I | Valid conclusions |
|---|---|---|---|
| Random | +2.1 | ~30% | ~8% |
| Trained (Run 2, steps 1-10) | **+10.62** | ~85% | ~65% |
| **Improvement** | **+406%** | **+183%** | **+713%** |

---

## Key Design Principles

1. **Latent state / POMDP**: Agent never sees the true effect size — only noisy estimates. Forces genuine reasoning under uncertainty.

2. **Math-verified outcomes**: `scipy.stats` determines trial success, not an LLM judge. Reproducible and manipulation-resistant.

3. **Power-gating**: A trial that hits p < 0.05 by chance with n=5 gets NO success bonus. The agent must design adequately powered trials.

4. **Episode-wide violation penalty**: Cumulative FDA violations are penalised at the terminal state, preventing "make 10 violations then clean the last step" exploits.

5. **Potential-based shaping**: `γ·(φ(s') − φ(s))` where `φ = milestone_completion × budget_efficiency` — accelerates learning without shifting the optimal policy.
