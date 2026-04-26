# Teaching an AI to Design Clinical Trials with Reinforcement Learning

**Environment**: [OpenEnv Clinical Trial](https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial)  
**Training**: [train_colab_v2.py](https://huggingface.co/spaces/Roopalgn/openenv-clinical-trial/blob/main/train_colab_v2.py)

---

## The Problem

Drug development fails 90% of the time. A significant fraction of those failures are not because the drug doesn't work — they're because the **trial was designed poorly**: too few patients, wrong endpoint, wrong phase ordering, inadequate statistical power.

Could an RL agent learn the decision-making heuristics of an experienced clinical trial statistician?

That's the question this environment tries to answer.

---

## What the Environment Does

The clinical trial environment presents the agent with a realistic scenario: a novel compound with unknown true effect size and side effect profile. The agent must design and execute a complete trial by selecting a sequence of actions:

```
Set endpoints → Set sample size → Set inclusion criteria →
Enroll patients → Run Phase I safety → Estimate effect size →
Run interim analysis → Run primary analysis → Synthesize conclusion
```

Each action costs budget and time. The trial's hidden ground truth (true effect size, side effect rate, dose-response curve) is never directly observable — the agent only sees **noisy observations** that mimic real measurement uncertainty.

### The Reward Signal

Eight decomposed reward components combine to give a total episode reward from −3 (complete failure) to +16 (optimal trial design):

- **Validity** (+0.05 per valid action, −2.0 for FDA violations)
- **Phase ordering** (correct workflow sequence rewarded, skipping penalised)
- **Milestone bonuses** (one-time bonuses of +0.5 to +2.5 per completed phase)
- **Terminal success** (power-gated: requires ≥40% statistical power to earn the +4.0 bonus)
- **Calibration** (CI accuracy vs hidden true effect size)
- **Progress** (+3.0 × fraction of milestones completed, even without full completion)

---

## The Training Challenge (and How We Solved It)

### Problem 1: Reward Collapse from Single-Step Evaluation

Our initial training used **single-step evaluation** — scoring each model completion with one environment step. The result: once the model learned to output valid JSON, all completions scored +0.2 to +0.25 with near-zero variance. GRPO had nothing to learn from.

**Fix**: Switch to **full-episode evaluation**. The model outputs a complete 10-action plan. We execute all actions against the live environment and sum the cumulative reward. This expands the reward range from 2.75 points to **19 points**, giving GRPO a genuine gradient to follow.

### Problem 2: Completion Truncation

With 512 max tokens, the model's 10-action JSON plan was being cut off mid-JSON, causing parse failures and a reward of −3.0 for every completion. `reward_std=0` → no gradient.

**Fix**: Increase `max_completion_length` from 512 to 1024 tokens. Completions now terminate naturally (average ~560 tokens).

### Problem 3: Weak Milestone Differentiation

With small milestone bonuses, a random valid action sequence could score nearly as high as a well-designed plan. Signal-to-noise ratio too low.

**Fix**: Doubled milestone bonuses, added terminal progress bonus (smooth gradient proportional to milestones reached), and added episode-wide violation penalty at terminal.

---

## Results

After 30 training steps on a T4 GPU:

| Metric | Value |
|---|---|
| Mean episode reward | **+7.58** |
| Training slope | **+0.055/step** |
| Collapsed steps | **0/30** |
| Rolling avg (steps 1-10) | +7.26 |
| Rolling avg (steps 21-30) | +8.11 |

![Reward Curve](reward_plot.png)

The reward started positive from step 1 (+12.05) because Qwen2.5-1.5B-Instruct already understands JSON and basic scientific reasoning. GRPO then reinforced the specific action sequences that complete the full trial workflow, producing the upward rolling average trend.

**Dry-run discrimination** (before training):
```
good plan:    +15.35  (full 10-action sequence)
minimal plan:  +0.50  (2 design actions only)  
parse failure: -3.00
delta: 14.85 points — ✓ highly discriminative
```

---

## Key Design Principles

1. **Latent state**: The agent never sees the true effect size — only noisy estimates. This forces the agent to learn *when* to gather more information vs. when to proceed.

2. **Math-verified outcomes**: Trial success is determined by `scipy.stats`, not an LLM judge. The reward signal is objective and reproducible.

3. **Power gating**: A trial that hits p < 0.05 by chance with tiny sample size gets NO success bonus. The agent must design adequately powered trials.

4. **Potential-based shaping**: γ·(φ(s') − φ(s)) where φ = milestone_completion × budget_efficiency. This preserves the optimal policy while accelerating learning.

---

## Running It Yourself

```python
# Dry run (validates reward discrimination)
!python train_colab_v2.py --dry-run

# Full training run
!python train_colab_v2.py --episodes 30 --model-size 1.5b --num-generations 6
```

The environment exposes a simple HTTP API at the HF Space, so the training script connects to the live environment for reward evaluation — no local server setup required.
