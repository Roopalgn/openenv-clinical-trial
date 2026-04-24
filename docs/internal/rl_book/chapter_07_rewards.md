# Chapter 7: The Reward System — Teaching Right from Wrong

## Why Rewards Are the Most Important Part

In RL, the reward signal is **everything**. It's the only way the agent learns what's good and what's bad. A poorly designed reward leads to an agent that games the system. A well-designed reward produces genuinely intelligent behavior.

**Analogy:** Imagine you're training a puppy using treats:
- If you give treats for sitting AND jumping, the puppy gets confused
- If you only give treats after a 30-minute perfect behavior streak, the puppy can't figure out what earned the treat
- If you give treats for each small correct behavior, the puppy learns fast

Our reward system follows the third approach: clear, frequent, decomposed signals.

## The Two Types of Rewards

### Per-Step Rewards (After Each Action)
"How good was this specific action in this specific moment?"

### Terminal Rewards (At Episode End)
"How good was the entire trial design?"

```
Episode timeline:
Step 1:  r₁ = +1.2  (good action)
Step 2:  r₂ = +0.8  (decent action)
Step 3:  r₃ = -1.5  (FDA violation!)
...
Step 40: r₄₀ = +0.5 (final action)
         + r_terminal = +10.0  (trial succeeded!)

Total episode reward = Σ(step rewards) + terminal reward
                     = (sum of ~40 step rewards) + terminal
                     = typically -3 to +14
```

## The 8 Per-Step Reward Components

Every step produces 8 independently calculated reward values:

```python
class RewardBreakdown(BaseModel):
    r_validity: float            # Was the action valid? (FDA rules)
    r_ordering: float            # Did you follow the correct clinical workflow?
    r_info_gain: float           # Did you learn something useful?
    r_efficiency: float          # Are you managing resources well?
    r_novelty: float             # Are you trying new things?
    r_penalty: float             # Any penalties for violations?
    r_terminal_success: float    # Did the trial succeed? (only at episode end)
    r_terminal_calibration: float # Were your estimates accurate? (only at episode end)
```

Let's explain each one in detail.

### 1. r_validity — "Did You Follow the Rules?"

```
Valid action:   +1.0  ("Yes, you were allowed to do that")
Invalid action: -1.0  ("No, that violates FDA rules")
```

This is binary — either you followed the rules or you didn't. The FDA rule engine (Chapter 8) determines this.

**Why it matters:** In real trials, regulatory violations can shut down the entire study. The agent must learn to respect the rules.

### 2. r_ordering — "Did You Follow the Correct Workflow?"

Clinical trials have a natural progression:
```
literature_review → hypothesis → design → enrollment → monitoring → analysis → submission
```

```
Correct order:  +0.2  ("Good, you're following the proper sequence")
Skipped phase:  -0.3 × N  ("You skipped N phases — that's sloppy")
Regression:      0.0  ("You went backwards — no bonus")
```

**Example:**
```
Step 1: set_primary_endpoint → phase "design"       → +0.2 (first step, always correct)
Step 2: set_sample_size      → phase "design"       → +0.2 (same phase, correct)
Step 3: enroll_patients       → phase "enrollment"   → +0.2 (next phase, correct)  
Step 4: set_inclusion_criteria→ phase "design"       → 0.0  (went BACK to design — regression!)
Step 5: submit_to_fda_review  → phase "submission"   → -0.9 (skipped 3 phases! -0.3 × 3)
```

**Why it matters:** A real trial that skips from enrollment directly to FDA submission would be rejected. The agent must learn the proper workflow.

### 3. r_info_gain — "Did You Learn Something?"

Some actions generate data, others are just administrative:

```python
# Info-generating actions and their rewards (from reward_computer.py)
info_actions = {
    ESTIMATE_EFFECT_SIZE,        # "How big is the drug's effect?"
    OBSERVE_SAFETY_SIGNAL,       # "Are there side effects?"
    RUN_INTERIM_ANALYSIS,        # "Is the trial on track?"
    RUN_DOSE_ESCALATION,         # "What's the right dose?"
    ADD_BIOMARKER_STRATIFICATION # "Which patients respond best?"
}

# Reward = base × statistical power
# If current power is 0.72:
r_info_gain = 0.5 × 0.72 = 0.36
```

Administrative actions (set_sample_size, set_blinding) get 0.0 info gain — they're necessary but don't generate new data.

**Why it matters:** The agent should seek information before making decisions. Running an experiment teaches you something; filling out a form doesn't.

### 4. r_efficiency — "Are You Managing Resources?"

```python
def _efficiency_reward(latent, initial_budget):
    budget_fraction = latent.budget_remaining / initial_budget
    return 2.0 × budget_fraction
    # If 80% of budget remains: 2.0 × 0.80 = 1.60
    # If 20% of budget remains: 2.0 × 0.20 = 0.40
    # If budget is gone:        2.0 × 0.00 = 0.00
```

This continuously rewards having budget left. It doesn't penalize spending — it rewards efficiency.

**Why it matters:** Real trials have finite budgets. An agent that blows the budget on unnecessary tests is wasteful.

### 5. r_novelty — "Are You Trying New Things?"

```python
def _novelty_reward(action, latent):
    if action.action_type.value not in latent.action_history:
        return 0.2  # First time using this action type!
    return 0.0      # You've done this before
```

**Why it matters:** Without this, the agent might spam one action type repeatedly. The novelty bonus encourages exploring the full action space.

### 6. r_penalty — "How Bad Were Your Violations?"

```python
# Each FDA violation costs -0.5
r_penalty = -0.5 × len(violations)

# Examples:
# "Tried to submit FDA review without Phase I" → -0.5
# "Sample size below 30 AND no prerequisite met" → -1.0 (two violations)
```

### 7. r_terminal_success — "Did the Trial Succeed?" (End Only)

```python
def _terminal_success_reward(latent, result):
    if latent.trial_complete and result.success and result.failure_reason is None:
        return +10.0  # Trial succeeded!
    return 0.0        # Not terminal, or trial failed
```

This is the big prize: +10.0 for a successful trial. It only fires once, at the end of the episode.

### 8. r_terminal_calibration — "Were You Right About the Drug?" (End Only)

```python
def _terminal_calibration_reward(latent, result):
    if not latent.trial_complete:
        return 0.0
    
    # How close was the agent's conclusion to the actual truth?
    ci_centre = (ci_low + ci_high) / 2.0
    centre_error = abs(ci_centre - latent.true_effect_size)
    calibration_score = max(0.0, 1.0 - centre_error)
    
    # Penalize wide confidence intervals (uncertainty)
    width_penalty = min(ci_width, 1.0)
    calibration_score *= (1.0 - width_penalty * 0.5)
    
    return 5.0 × calibration_score
    # Best case: CI centered on true effect, narrow → +5.0
    # Worst case: CI far off or super wide → ~+0.0
```

**Why it matters:** A trial can "succeed" (p < 0.05) by luck. Calibration rewards the agent for actually understanding the drug — not just getting lucky.

## Typical Episode Rewards

| Episode Outcome | Typical Total | How It Happens |
|---|---|---|
| Expert success | +11 to +14 | Found subgroup + efficient + FDA pass + calibrated |
| Good success | +6 to +10 | Trial works, decent design |
| Marginal success | +2 to +5 | Barely significant (p≈0.04), wrong subgroup |
| Near miss | -1 to +1 | Good design but p=0.06 (just missed) |
| Clear failure | -2 to 0 | Wrong design, FDA rejection |
| Timeout | -2.0 flat | Ran out of steps without finishing |

**Why the high variance?** GRPO (our training algorithm) needs clear separation between good and bad episodes. If all episodes scored between 3.0 and 5.0, the agent couldn't tell what's good. The -3 to +14 range provides a strong learning signal.

## Reward Shaping: Giving the Agent Breadcrumbs

### The Problem with Sparse Rewards

Imagine if we ONLY gave a reward at the end of the episode:
- Episode succeeds → +10
- Episode fails → -2
- All 40-100 intermediate steps → 0.0 (no signal!)

The agent has no idea which of its 40+ actions led to success or failure. This is called the **credit assignment problem**.

### The Solution: Potential-Based Reward Shaping

We give extra "breadcrumb" rewards for making progress toward milestones:

```python
# From server/reward/shaping.py

def potential(latent, initial_budget):
    """φ(s) = milestone_completion × budget_efficiency"""
    milestones = [
        latent.phase_i_complete,
        latent.interim_complete, 
        latent.protocol_submitted,
        latent.trial_complete,
    ]
    milestone_fraction = sum(1 for m in milestones if m) / len(milestones)
    budget_efficiency = latent.budget_remaining / initial_budget
    return milestone_fraction × budget_efficiency

def shaping_bonus(latent, next_latent, initial_budget, gamma=0.99):
    """Bonus = γ × (φ(s') − φ(s))"""
    phi_before = potential(latent, initial_budget)
    phi_after = potential(next_latent, initial_budget)
    return gamma × (phi_after - phi_before)
```

**How it works:**
- Before completing Phase I: φ = 0/4 × 0.80 = 0.0
- After completing Phase I:  φ = 1/4 × 0.78 = 0.195
- Bonus: 0.99 × (0.195 - 0.0) = +0.193

The agent gets a positive reward just for reaching a milestone.

### Why "Potential-Based"?

This is mathematically important. The shaping formula `γ·(φ(s') - φ(s))` is called **potential-based shaping** because it satisfies a beautiful property:

**The optimal policy doesn't change.** Adding this shaping bonus doesn't change which strategy ultimately scores highest — it only changes how quickly the agent learns that strategy.

Think of it like putting trail markers on a hiking path. The markers don't change where the summit is — they just help you find it faster.

The key mathematical insight is that the shaping bonuses **telescope** over a full episode:

$$\sum_{t=0}^{T-1} \gamma \cdot (\varphi(s_{t+1}) - \varphi(s_t)) = \gamma \cdot \varphi(s_T) - \varphi(s_0)$$

The intermediate terms cancel out! So the total shaped reward only depends on the starting and ending states, not the path — meaning the optimal path is unchanged.

## The Overconfidence Penalty

A subtle but important feature:

```
If the agent says "I'm 80%+ confident this drug's effect is 0.45"
but the true effect is actually 0.20 (way off)
→ Penalty: -0.5

If the agent says "I'm 50% confident" (appropriately uncertain)
→ No penalty, even if wrong
```

From the judge component:
```python
_HIGH_CONFIDENCE_THRESHOLD = 0.8
_OVERCONFIDENCE_PENALTY = -0.5

# Penalize high-confidence wrong claims
if action.confidence >= 0.8 and not result.success:
    penalty = -0.5
```

**Why it matters:** We want the agent to be **calibrated** — meaning its confidence levels match reality. An agent that says "I'm 90% sure!" when it's actually 50/50 is dangerous. In real medicine, overconfidence kills.

## Why 8 Components? Why Not One Number?

> **Design Decision Box: Decomposed vs. Single Reward**
>
> We could compute one number: `reward = magical_formula(everything)`. But decomposed rewards have massive advantages:
>
> **Debuggability:** When the agent scores poorly, we can see exactly WHY:
> ```
> Episode 42: total_reward = -1.5
>   r_validity: -1.0   ← FDA violation (THIS is the problem)
>   r_ordering: +0.2   ← Correct workflow
>   r_info_gain: +0.36 ← Good information gathering
>   r_efficiency: +1.2  ← Efficient
>   r_novelty: +0.1    ← Some exploration
>   r_penalty: -2.3    ← Multiple violations
>   r_terminal: 0.0    ← Didn't reach terminal
> ```
>
> **Tuning:** We can adjust individual components without affecting others. If the agent is too reckless with the budget, we increase `r_efficiency`'s weight.
>
> **Verification:** Each component is computed by a simple, testable function. `r_validity` is just "did the FDA rule engine say valid?" — binary, no ambiguity.
>
> **The alternative** — an LLM judge scoring "how good was this trial design on a scale of 1-10?" — would be non-reproducible, expensive, and impossible to debug.

---

## Mathematical Summary

For those who want the formula:

$$R_{\text{episode}} = \sum_{t=1}^{T} r_{\text{step}_t} + r_{\text{terminal}}$$

Where each step reward is:

$$r_{\text{step}} = r_{\text{validity}} + r_{\text{ordering}} + r_{\text{info\_gain}} + r_{\text{efficiency}} + r_{\text{novelty}} + r_{\text{penalty}} + r_{\text{shaping}}$$

And the shaping bonus is:

$$r_{\text{shaping}} = \gamma \cdot (\varphi(s') - \varphi(s)), \quad \varphi(s) = \text{milestone\_completion} \times \text{budget\_efficiency}, \quad \gamma = 0.99$$

---

## Chapter 7 Glossary

| Keyword | Definition |
|---------|-----------|
| **Reward Signal** | A number telling the agent how good an action was |
| **Per-Step Reward** | Reward given after each individual action |
| **Terminal Reward** | Large reward given at the end of an episode |
| **Reward Decomposition** | Breaking reward into independently verifiable components |
| **Credit Assignment** | Figuring out which actions led to success or failure |
| **Sparse Reward** | Reward given only rarely (e.g., only at episode end) |
| **Dense Reward** | Reward given after every step |
| **Reward Shaping** | Adding extra signals to help the agent learn faster |
| **Potential-Based Shaping** | Shaping that preserves the optimal policy (γ·(φ(s')−φ(s))) |
| **Telescoping** | Intermediate shaping terms cancel out over full episode |
| **Calibration** | Agent's confidence matches actual accuracy |
| **Overconfidence Penalty** | Punishment for being very confident and wrong |
| **p-value** | Probability of seeing these results if the drug doesn't work |
| **Statistical Power** | Probability of detecting a real effect (target: ≥0.80) |
