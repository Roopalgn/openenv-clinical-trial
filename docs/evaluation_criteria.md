# Evaluation Criteria & Acceptance Metrics

> **Inspired by:** KubeSRE's reward curves across 3 training runs. Bio Experiment's trajectory datasets with success rate, mean reward, episode length. VRAM's training dashboard with loss, reward, and capability radar chart. All winners tracked multiple complementary metrics, not just reward.

## Primary Metrics

### 1. Trial Success Rate
> *Pattern from KubeSRE: per-episode success tracking across training runs*

The percentage of episodes where the agent's trial design successfully detects the true drug effect (p < 0.05) with adequate statistical power (≥ 0.80).

| Difficulty Tier | Target Success Rate | Measurement |
|---|---|---|
| Warmup (0.0–0.25) | ≥ 60% | Large effect, homogeneous population |
| Beginner (0.25–0.40) | ≥ 45% | Medium effect, some noise |
| Intermediate (0.40–0.60) | ≥ 30% | Small effect, enrichment needed |
| Advanced (0.60–0.80) | ≥ 15% | Hidden subgroup, misleading signals |
| Expert (0.80–0.95) | ≥ 5% | Tiny effect, adaptive design required |

### 2. Reward Trend
> *Pattern from all 3 winners: reward curves were the primary visual proof of learning*

Positive slope on the per-episode total reward over training. Measured via `plot_rewards.py` trend line.

- **Pass**: trend slope > 0 over 20+ episodes
- **Strong**: trend slope > 0.1/episode
- **Excellent**: rolling average at end > rolling average at start by ≥ 50%

Track per-component trends (like Bio Experiment's decomposed reward logging):
- `r_validity` trend: are FDA violations decreasing?
- `r_ordering` trend: is phase compliance improving?
- `r_info_gain` trend: is the agent learning to gather information?

### 3. Curriculum Progression

Agent advances through difficulty tiers as it masters easier scenarios.

- **Pass**: reaches Beginner tier within 15 episodes
- **Strong**: reaches Intermediate tier within 40 episodes
- **Excellent**: reaches Advanced tier within 80 episodes

## Secondary Metrics

### 4. Phase Workflow Compliance
> *Pattern from KubeSRE: triage→investigate→fix→verify compliance tracking*

Percentage of episodes where the agent follows the correct clinical trial phase order (literature review → hypothesis → design → enrollment → monitoring → analysis → submission) without skipping phases.

- **Target**: ≥ 70% of episodes follow correct phase order
- **Measured by**: `_detect_phase()` + phase-order bonus/penalty in reward

### 5. FDA Rule Pass Rate
> *Pattern from Bio Experiment: prerequisite rule pass rate as hard constraint metric*

Percentage of trials that pass all hard FDA constraint checks.

- **Target**: ≥ 80% of completed trials pass all checks
- **Measured by**: rule engine pass/fail count per episode

### 6. Statistical Power Adequacy

Percentage of trial designs where the agent achieves statistical power ≥ 0.80.

- **Target**: ≥ 50% of completed trials
- **Measured by**: `calculate_power()` on final trial design

### 7. Budget Efficiency
> *Pattern from Bio Experiment: resource efficiency as reward component and metric*

Percentage of trials that complete within budget.

- **Target**: ≥ 75% of completed trials
- **Measured by**: `budget_remaining ≥ 0` at episode end

### 8. Action Diversity
> *Pattern from VRAM: unique tool usage as evidence of genuine exploration vs. pattern repetition*

Ratio of unique actions used to total actions taken. Higher diversity suggests the agent is exploring intelligently rather than spamming the same action.

- **Target**: ≥ 0.60 diversity ratio (at least 12 of 19 actions used per episode)
- **Measured by**: `len(set(actions)) / len(actions)` per episode
- **Visualization**: Action usage heatmap (like VRAM's tool usage heatmap) comparing base vs trained model

### 9. Steps to Completion
> *Pattern from KubeSRE: efficiency-scaled resolution bonus — faster = higher reward*

Average steps to reach terminal state. Fewer steps (while maintaining quality) indicates efficiency.

- **Target**: trained model completes in ≥ 10 fewer steps than base model on same scenarios
- **Measured by**: `step_count` at `done=True`

### 10. Reproducibility
> *Pattern from Bio Experiment: seeded NoiseModel guarantees deterministic episodes*

Same seed produces identical episode outcomes.

- **Test**: run 2 episodes with same seed, assert identical reward sequences
- **Measured by**: integration test in CI

## Tracking & Reporting

> **Inspired by:** KubeSRE's reward curves across 3 training runs. Bio Experiment's trajectory datasets. VRAM's capability radar chart.

All metrics are logged per episode in the reward CSV and JSONL transcript files.

### Episode Transcript JSONL (from KubeSRE pattern)
Each step logged as one JSON line: action, observation, reward breakdown (all 8 components), phase detected, phase order correctness, hidden state snapshot for offline debugging.

### Reward CSV (from Bio Experiment pattern)
Per-episode: episode_id, scenario_id, tier, total_reward, each component reward, success, power, FDA_pass, steps, action_diversity.

### Curriculum Log
Per-episode: tier, scenario_id, difficulty, outcome, advancement decision, mastery stats.

### Base vs Trained Comparison Table (eval_compare.py)
> *Pattern from all winners: before/after is the proof that the environment teaches*

```
Metric                  Base Model    Trained Model    Delta
───────────────────────────────────────────────────────────────
Trial Success Rate      12%           48%              +36%
Avg Total Reward        -1.23         +4.56            +5.79
Phase Compliance        31%           74%              +43%
FDA Pass Rate           45%           87%              +42%
Avg Power (successful)  0.62          0.84             +0.22
Action Diversity        0.37          0.68             +0.31
Avg Steps to Complete   62            48               -14
Budget Efficiency       51%           78%              +27%
Subgroup Identified     8%            41%              +33%
```

### Capability Radar Chart (from VRAM pattern)
> *VRAM showed post-training capability profile expansion as a radar chart*

Visualize 6 axes comparing base vs trained:
1. Trial success rate
2. Phase compliance
3. FDA pass rate
4. Action diversity
5. Budget efficiency
6. Subgroup identification rate

### Visual Assets for Demo
1. **Reward curve**: per-episode scatter + rolling average + trend line (from `plot_rewards.py`)
2. **Component reward trends**: 8 subplots, one per reward component over training
3. **Action heatmap**: base vs trained action usage patterns (from VRAM)
4. **Capability radar**: 6-axis comparison (from VRAM)
5. **Curriculum progression**: bar chart showing tier advancement over episodes
