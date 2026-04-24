# Chapter 14: Evaluation and Comparison

## Why Evaluation Matters

Training an RL agent is like sending a student to school. Evaluation is the exam — it tells you if the student actually learned or just memorized the homework.

We need to answer:
1. "Is the trained agent better than a random agent?"
2. "Is it improving over time?"
3. "What does it still struggle with?"

## Evaluation vs. Training

During **training**, the agent explores (tries new things, makes mistakes):
- Temperature = 0.7 (random sampling)
- Domain randomization active
- Rewards used for weight updates

During **evaluation**, the agent exploits (does its best):
- Temperature = 0.0 or greedy (pick the best option)
- Fixed seeds (same scenarios for fair comparison)
- Rewards measured but NOT used for weight updates

## The Eval Compare Script

Our project includes `eval_compare.py`, which compares two agents:

```python
# From eval_compare.py — conceptual structure

def evaluate(agent, env, seeds, max_steps):
    """Run the agent on a fixed set of seeds and collect metrics."""
    results = []
    for seed in seeds:
        obs = env.reset(seed=seed)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.act(obs)  # Agent picks an action
            obs, reward, done, info = env.step_full(action)
            total_reward += reward
            steps += 1
            if done:
                break
        
        results.append({
            "seed": seed,
            "total_reward": total_reward,
            "steps": steps,
            "success": info.get("success", False),
        })
    
    return results

# Compare random agent vs trained agent on same 100 seeds
random_results = evaluate(random_agent, env, seeds=range(100), max_steps=50)
trained_results = evaluate(trained_agent, env, seeds=range(100), max_steps=50)
```

### Key Metrics

| Metric | What It Measures | Good Value |
|---|---|---|
| **Mean reward** | Average total episode reward | Higher is better (target: +8) |
| **Success rate** | % of trials that achieved p < 0.05 | Higher is better (target: 70%+) |
| **Mean steps** | Average steps to complete trial | Lower is better (efficiency) |
| **FDA violation rate** | % of steps with rule violations | Lower is better (target: <5%) |
| **Phase compliance** | % of steps with correct phase ordering | Higher is better (target: >90%) |

### The Evaluation Report

The output is saved as JSON:

```json
{
    "random_agent": {
        "mean_reward": -1.2,
        "success_rate": 0.05,
        "mean_steps": 50.0,
        "fda_violation_rate": 0.45,
        "phase_compliance": 0.32
    },
    "trained_agent": {
        "mean_reward": 8.7,
        "success_rate": 0.72,
        "mean_steps": 28.3,
        "fda_violation_rate": 0.03,
        "phase_compliance": 0.94
    }
}
```

This shows dramatic improvement: the trained agent succeeds 72% of the time vs. 5% for random, with far fewer violations and better efficiency.

## Reward Curves: Visualizing Progress

The reward curve is the most important training diagnostic:

```python
# From plot_rewards.py
def main(args):
    """Plot reward curves from the training CSV log."""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv(args.csv)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["episode"], df["total_reward"], alpha=0.3, label="Raw")
    
    # Rolling average smooths out noise
    rolling = df["total_reward"].rolling(window=20).mean()
    plt.plot(df["episode"], rolling, linewidth=2, label="20-episode rolling avg")
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.savefig(args.out)
```

### Reading the Reward Curve

```
Reward
  +14 │                                        ★ ★ ★
      │                                   ★ ★ ★     ★ ★
  +10 │                              ★ ★ ★
      │                         ★ ★ ★
   +6 │                    ★ ★ ★
      │               ★ ★ ★
   +2 │          ★ ★ ★
      │     ★ ★ ★
   -2 │★ ★ ★
      └─────────────────────────────────────────────────
       0    50   100   150   200   250   300
                        Episode

Phases:
[--- Random ---|--- Learning workflow ---|--- Efficient ---|--- Expert ---]
```

**What to look for:**
- **Upward trend:** Agent is improving (good!)
- **Plateau:** Agent has maxed out at current difficulty
- **Jumps:** Often coincide with curriculum tier changes
- **Drops:** Tier advancement introduced harder problems
- **High variance:** Agent hasn't stabilized yet (need more training)

## Seed-Based Reproducibility

Every number in our system is deterministic given the same seed:

```python
# Same seed → same episode → same hidden truth → same results
env.reset(seed=42)
# true_effect_size = 0.38 (ALWAYS for seed=42)
# budget = $11.3M (ALWAYS for seed=42)
# dropout = 0.12 (ALWAYS for seed=42)
```

This means:
1. **Fair comparison:** Agent A and Agent B face identical scenarios
2. **Debugging:** "Replay seed 42 to see what went wrong"
3. **Regression testing:** "After code changes, seed 42 should still give the same observation"

## What Good Performance Looks Like

### Warmup Tier (Easy)
A good agent should:
- Follow the basic workflow (design → enroll → analyze)
- Not violate FDA rules
- Complete the trial within budget
- Success rate: 80-100%

### Beginner Tier (Solid Tumor)
A good agent should:
- Identify appropriate sample sizes
- Run dose escalation in Phase I
- Complete trials with p < 0.05
- Success rate: 60-80%

### Intermediate Tier (Autoimmune)
A good agent should:
- Recognize non-monotonic dose-response
- Pick optimal dose (200mg, not MTD 300mg)
- Use biomarker stratification when available
- Success rate: 50-70%

### Advanced Tier (Depression)
A good agent should:
- Recognize high placebo response as a challenge
- Enrich for severe patients (higher effect)
- Use large sample sizes or adaptive designs
- Know when to stop futile trials early
- Success rate: 40-60%

### Expert Tier (Adversarial)
A good agent should:
- Handle compound challenges simultaneously
- Make efficient resource allocation decisions
- Produce well-calibrated effect estimates
- Navigate rare disease / orphan drug pathways
- Success rate: 30-50%

## Before/After Analysis

The episode transcripts enable detailed before/after analysis:

```
BEFORE TRAINING (Episode 1):
  Step 0: submit_to_fda_review  → VIOLATION (Phase I not done!)
  Step 1: run_primary_analysis  → VIOLATION (no interim!)
  Step 2: enroll_patients       → OK but no design done
  Step 3: set_sample_size(5)    → VIOLATION (below minimum 30!)
  ...
  Result: Timeout, reward = -2.0

AFTER TRAINING (Episode 250):
  Step 0: set_primary_endpoint(PFS)           → +3.4
  Step 1: set_sample_size(200)                → +3.3
  Step 2: set_inclusion_criteria(EGFR+)       → +3.2
  Step 3: set_dosing_schedule(150mg daily)     → +3.1
  Step 4: enroll_patients(50)                  → +2.5
  Step 5: run_dose_escalation                  → +2.7
  Step 6: estimate_effect_size                 → +2.6
  Step 7: add_biomarker_stratification         → +2.4
  Step 8: enroll_patients(100)                 → +1.8
  Step 9: run_interim_analysis                 → +2.9
  Step 10: run_primary_analysis                → +14.2 (trial succeeded!)
  ...
  Result: Success, reward = +48.5
```

The trained agent follows the proper workflow, respects prerequisites, and makes strategic decisions.

---

## Chapter 14 Glossary

| Keyword | Definition |
|---------|-----------|
| **Evaluation** | Testing the agent's performance without updating weights |
| **Training** | Teaching the agent by updating weights based on rewards |
| **Reward Curve** | Plot showing reward over training episodes |
| **Rolling Average** | Smoothed version of a noisy signal (average of last N values) |
| **Success Rate** | Percentage of episodes where the trial achieved p < 0.05 |
| **Seed-Based Reproducibility** | Same seed always produces the same episode |
| **Regression Testing** | Verifying that changes don't break existing behavior |
| **Before/After Analysis** | Comparing agent behavior at start vs end of training |
| **Plateau** | A period where performance stops improving |
| **Greedy (Temperature=0)** | Always picking the most likely output (no randomness) |
