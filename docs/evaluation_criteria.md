# Evaluation Criteria & Acceptance Metrics

## Primary Metrics

### 1. Trial Success Rate

The percentage of episodes where the agent's trial design successfully detects the true drug effect (p < 0.05) with adequate statistical power (≥ 0.80).

| Difficulty Tier | Target Success Rate | Measurement |
|---|---|---|
| Warmup (0.0–0.25) | ≥ 60% | Large effect, homogeneous population |
| Beginner (0.25–0.40) | ≥ 45% | Medium effect, some noise |
| Intermediate (0.40–0.60) | ≥ 30% | Small effect, enrichment needed |
| Advanced (0.60–0.80) | ≥ 15% | Hidden subgroup, misleading signals |
| Expert (0.80–0.95) | ≥ 5% | Tiny effect, adaptive design required |

### 2. Reward Trend

Positive slope on the per-episode total reward over training. Measured via `plot_rewards.py` trend line.

- **Pass**: trend slope > 0 over 20+ episodes
- **Strong**: trend slope > 0.1/episode
- **Excellent**: rolling average at end > rolling average at start by ≥ 50%

### 3. Curriculum Progression

Agent advances through difficulty tiers as it masters easier scenarios.

- **Pass**: reaches Beginner tier within 15 episodes
- **Strong**: reaches Intermediate tier within 40 episodes
- **Excellent**: reaches Advanced tier within 80 episodes

## Secondary Metrics

### 4. Phase Workflow Compliance

Percentage of episodes where the agent follows the correct clinical trial phase order (literature review → hypothesis → design → enrollment → monitoring → analysis → submission) without skipping phases.

- **Target**: ≥ 70% of episodes follow correct phase order
- **Measured by**: `_detect_phase()` + phase-order bonus/penalty in reward

### 5. FDA Rule Pass Rate

Percentage of trials that pass all hard FDA constraint checks.

- **Target**: ≥ 80% of completed trials pass all checks
- **Measured by**: rule engine pass/fail count per episode

### 6. Statistical Power Adequacy

Percentage of trial designs where the agent achieves statistical power ≥ 0.80.

- **Target**: ≥ 50% of completed trials
- **Measured by**: `calculate_power()` on final trial design

### 7. Budget Efficiency

Percentage of trials that complete within budget.

- **Target**: ≥ 75% of completed trials
- **Measured by**: `budget_remaining ≥ 0` at episode end

### 8. Reproducibility

Same seed produces identical episode outcomes.

- **Test**: run 2 episodes with same seed, assert identical reward sequences
- **Measured by**: integration test in CI

## Tracking & Reporting

All metrics are logged per episode in the reward CSV and JSONL transcript files. The evaluation pipeline (`eval_compare.py`) computes all metrics for both base and trained models and outputs a comparison table.

```
Metric                  Base Model    Trained Model    Delta
Trial Success Rate      12%           48%              +36%
Avg Total Reward        -1.23         +4.56            +5.79
Phase Compliance        31%           74%              +43%
FDA Pass Rate           45%           87%              +42%
Avg Steps to Complete   62            48               -14
```
