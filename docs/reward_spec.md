# Reward Decomposition Specification


## Design Principles

1. **Decomposed** — every reward component is independently verifiable and debuggable
2. **Math-verified** — core success is determined by scipy.stats, not LLM
3. **High variance** — GRPO needs clear separation between good (-3) and great (+14) episodes
4. **Shaped** — potential-based shaping gives gradient toward progress without changing optimal policy
5. **Phase-aware** — correct workflow ordering is rewarded

## Reward Formula

### Per-Step Reward

Each step produces a reward from 8 components:

```
r_step = (
    w_validity   × r_validity      +    # FDA rule compliance
    w_ordering   × r_ordering      +    # Phase-order bonus/penalty
    w_info_gain  × r_info_gain     +    # Information gained from experiment
    w_efficiency × r_efficiency    +    # Budget/time efficiency
    w_novelty    × r_novelty       +    # Action diversity bonus
    w_penalty    × r_penalty       +    # Soft constraint violations
    r_shaping                            # Potential-based: γ·(φ(s') − φ(s))
)
```

### Terminal Reward

Fires once at `done=True` after trial simulation:

```
r_terminal = (
    r_terminal_success     +    # Core: did the trial detect the true effect?
    r_terminal_calibration +    # How well do agent's claims match hidden truth?
    r_terminal_power       +    # Statistical power adequacy
    r_terminal_fda         +    # Overall FDA compliance
    r_terminal_budget      +    # Budget efficiency
    r_terminal_futility    +    # Early futility detection bonus
    r_terminal_overconf         # Overconfidence penalty
)
```

### Total Episode Reward

```
R_episode = Σ(r_step_t for t in 1..T) + r_terminal
```

On **timeout** (step_count >= max_steps without reaching conclusion): wipe accumulated step rewards, set `R_episode = -2.0` total. This prevents reward farming from endless episodes.

---

## Per-Step Components

### 1. `r_validity` — FDA Rule Compliance


| Condition | Reward |
|-----------|--------|
| Action passes all applicable FDA rules | +0.3 |
| Action violates a soft FDA rule | -0.15 per violation |
| Action blocked by hard prerequisite | 0.0 (action doesn't execute, error returned) |

**Weight:** `w_validity = 1.0` (applied directly, not scaled)

**Verification:** `check_fda_compliance(action, state)` returns list of violated rules. Each rule is a boolean function — no LLM.

**FDA rules checked per step:**
```python
FDA_RULES = {
    "phase_ii_min_n": lambda state: state.sample_size >= 100 if state.in_phase_ii else True,
    "primary_endpoint_prespecified": lambda state: state.primary_endpoint is not None if state.submitting else True,
    "interim_alpha_spending": lambda state: state.alpha_spent <= state.alpha_budget if state.interim_running else True,
    "randomization_required": lambda state: state.randomization_set if state.in_phase_ii else True,
    "safety_monitoring": lambda state: state.safety_committee_set if state.patient_count > 0 else True,
    "informed_consent": lambda state: True,  # Always required, auto-checked
}
```

### 2. `r_ordering` — Phase-Order Scoring


| Condition | Reward |
|-----------|--------|
| Action is in correct or next phase | +0.2 |
| Action stays in current phase (continuation) | +0.2 |
| Action skips N phases ahead | -0.3 × N |

**Weight:** `w_ordering = 1.0` (applied directly)

**Verification:** `_detect_phase(action, history)` classifies action, compares against `PHASE_ORDER` map. See `phase_workflow.md` for full logic.

**Judge persona scaling:**
- Warmup (< 0.25): junior — allows 1 skip without penalty, gives hints
- Intermediate (0.25–0.60): senior — standard penalties
- Expert (> 0.60): principal — harsher penalties (-0.5/skip), efficiency penalties

### 3. `r_info_gain` — Information Gained


Measures how much the agent learned about hidden ground truth from its action.

| Action Type | Info Gain Calculation | Typical Range |
|------------|----------------------|---------------|
| `run_dose_escalation` | Bayesian update on effect_size posterior: `KL(posterior ‖ prior)` | +0.1 to +0.5 |
| `observe_safety_signal` | Information bits about true_side_effect_rate: `H(prior) - H(posterior)` | +0.05 to +0.3 |
| `estimate_effect_size` | Reduction in uncertainty: `1 - (posterior_std / prior_std)` | +0.2 to +0.6 |
| `run_interim_analysis` | Evidence for/against futility: `abs(log_likelihood_ratio)` | +0.1 to +0.4 |
| `add_biomarker_stratification` | Subgroup signal strength: `effect_in_subgroup / effect_overall` | +0.1 to +0.8 |
| Design actions (set_*) | 0.0 — design doesn't generate new data | 0.0 |
| `submit_to_fda_review` | 0.0 — regulatory, not experimental | 0.0 |

**Weight:** `w_info_gain = 1.0`

**Verification:** `OutputGenerator` produces noisy observations conditioned on `TrialLatentState`. The info gain is computed by comparing the agent's posterior belief distribution (inferred from its actions) against the hidden truth. Higher quality experimental design (correct dose levels, appropriate cohort size) yields sharper posteriors and higher info gain.

**Key insight:** This is the component that teaches the agent to *gather information* before making design decisions. Without it, the agent would skip Phase I entirely and guess at Phase II parameters.

### 4. `r_efficiency` — Budget/Time Efficiency


```python
def compute_efficiency(state):
    budget_ratio = state.budget_remaining / state.initial_budget
    time_ratio = state.time_remaining_days / state.initial_time
    # Reward efficient use without being too conservative
    return 0.1 * min(budget_ratio, time_ratio)
```

| Condition | Reward |
|-----------|--------|
| ≥ 30% budget remaining at trial completion | +0.1 |
| 10–30% budget remaining | +0.05 |
| < 10% budget remaining | 0.0 |
| Budget exhausted (trial cannot continue) | -0.2 |

**Weight:** `w_efficiency = 1.0`

### 5. `r_novelty` — Action Diversity Bonus


```python
def compute_novelty(action, history):
    if action.action_type not in [h.action_type for h in history]:
        return 0.1  # First time using this action
    return 0.0
```

Additionally, **repeat penalty**:
```python
def compute_repeat_penalty(action, history):
    recent = history[-3:]  # last 3 actions
    repeat_count = sum(1 for h in recent if h.action_type == action.action_type)
    return -0.15 * repeat_count  # -0.15 per recent repeat
```

**Weight:** `w_novelty = 1.0`

### 6. `r_penalty` — Soft Constraint Violations


| Violation | Penalty |
|-----------|---------|
| Redundant action (repeating completed step) | -0.15 |
| Unjustified high confidence (confidence > 0.8 without data) | -0.1 |
| Design action without Phase I data | -0.15 |
| Protocol amendment after trial started | -0.1 |

**Weight:** `w_penalty = 1.0`

### 7. `r_shaping` — Potential-Based Reward Shaping

See [Potential-Based Shaping Function](#potential-based-shaping-function) section below for full specification.

---

## Terminal Components

### `r_terminal_success` — Core Trial Outcome


```python
def compute_terminal_success(trial_result, ground_truth):
    if trial_result.p_value < ground_truth.alpha:
        # Efficiency bonus: fewer steps = higher reward
        step_ratio = 1.0 - (steps_taken / max_steps)
        return 5.0 + (2.0 * step_ratio)  # +5.0 to +7.0
    else:
        return -1.0  # Trial failed to detect true effect
```

| Outcome | Reward |
|---------|--------|
| Trial detects true effect (p < α) — efficient | +5.0 to +7.0 |
| Trial detects true effect (p < α) — slow | +5.0 |
| Trial fails to detect (p ≥ α) | -1.0 |

### `r_terminal_calibration` — Ground Truth Match


```python
def compute_terminal_calibration(agent_conclusion, ground_truth):
    score = 0.0

    # Did agent identify the correct responder population?
    if agent_conclusion.responder_population == ground_truth.true_responder_population:
        score += 3.0

    # Did agent identify the correct mechanism?
    if agent_conclusion.mechanism in ground_truth.true_mechanism:
        score += 1.0

    # Did agent estimate effect size within 30% of truth?
    relative_error = abs(agent_conclusion.effect_estimate - ground_truth.true_effect_size) / ground_truth.true_effect_size
    if relative_error <= 0.30:
        score += 1.0
    elif relative_error <= 0.50:
        score += 0.5

    return score  # 0.0 to +5.0
```

### `r_terminal_power` — Statistical Power

```python
def compute_terminal_power(trial_design, ground_truth):
    power = calculate_power(
        effect_size=ground_truth.true_effect_size,
        n=trial_design.sample_size,
        alpha=trial_design.alpha
    )
    if power >= 0.90:
        return +2.0  # Excellent
    elif power >= 0.80:
        return +1.5  # Adequate
    elif power >= 0.60:
        return 0.0   # Underpowered but not terrible
    else:
        return -2.0  # Severely underpowered
```

### `r_terminal_fda` — Final FDA Compliance

```python
def compute_terminal_fda(trial_design, rule_engine):
    violations = rule_engine.check_all(trial_design)
    passed = len([v for v in violations if v.passed])
    total = len(violations)
    if total == 0:
        return 0.0
    pass_rate = passed / total
    if pass_rate == 1.0:
        return +2.0  # Perfect compliance
    elif pass_rate >= 0.80:
        return +1.0
    else:
        return -1.0  # Too many violations
```

### `r_terminal_budget` — Budget Outcome

```python
def compute_terminal_budget(state):
    if state.budget_remaining >= 0:
        return +1.0
    else:
        return -0.5  # Over budget
```

### `r_terminal_futility` — Early Futility Detection

> *Bonus for stopping a doomed trial early instead of wasting resources*

```python
def compute_terminal_futility(agent_stopped_early, trial_result):
    if agent_stopped_early and trial_result.would_have_failed:
        return +1.0  # Smart early stop
    elif agent_stopped_early and trial_result.would_have_succeeded:
        return -1.5  # Stopped a successful trial — bad call
    else:
        return 0.0  # Ran to completion (normal)
```

### `r_terminal_overconf` — Overconfidence Penalty


```python
def compute_terminal_overconfidence(agent_conclusions, ground_truth):
    penalty = 0.0
    for claim in agent_conclusions:
        if claim.confidence >= 0.80 and not claim.matches(ground_truth):
            penalty -= 0.5
    return penalty  # 0.0 to -2.5 (up to 5 wrong high-confidence claims)
```

---

## Reward Variance Summary (for GRPO)

| Outcome | Typical Total Reward | Range |
|---------|---------------------|-------|
| **Expert success**: correct subgroup, FDA pass, efficient, good calibration | +11 to +14 | Best case |
| **Good success**: trial succeeds, partial calibration | +6 to +10 | Common success |
| **Marginal success**: p < 0.05 barely, wrong subgroup | +2 to +5 | Partial win |
| **Marginal failure**: ran full trial, p > 0.05 | -1 to +1 | Near miss |
| **Clear failure**: wrong design, FDA rejection | -2 to 0 | Bad episode |
| **Timeout**: exceeded max_steps | -2.0 (flat) | Worst case |
| **Catastrophic**: stopped a winning trial + overconfidence | -3 to -2 | Very rare |

GRPO with 8 rollouts needs the top and bottom to be clearly separated. A range of **-3 to +14** provides ample variance for advantage computation.

---

## Potential-Based Shaping Function

> **Rationale:** Potential-based shaping provides gradient toward progress without changing the optimal policy. This section defines f(s) for clinical trial state.

### Theory

Potential-based reward shaping adds a shaped reward without changing the optimal policy (Ng et al., 1999):

$$R_{\text{shaped}}(s, a, s') = R_{\text{original}}(s, a, s') + \gamma \cdot \varphi(s') - \varphi(s)$$

Where:
- $\varphi(s)$ is the potential function mapping state to a scalar
- $\gamma = 0.99$ is the discount factor
- The shaping terms telescope over a full episode, so the optimal policy is unchanged

The shaped reward gives the agent a "compass" — gradient toward higher-potential states — without distorting which policy is best.

### Our Potential Function

```python
def phi(state: TrialState) -> float:
    """
    Potential function for clinical trial state.
    Higher potential = closer to successful trial completion.

    φ(s) = w_milestone × milestone_fraction
          + w_budget   × budget_efficiency
          + w_phase    × phase_progress
          + w_data     × data_quality
    """
    # Milestone completion (0.0 to 1.0)
    milestones_completed = sum([
        state.phase_i_complete,           # Phase I done
        state.mtd_identified,             # Maximum tolerated dose found
        state.effect_estimated,           # Effect size estimated from Phase I
        state.primary_endpoint_set,       # Primary endpoint chosen
        state.sample_size_set,            # Sample size calculated
        state.inclusion_criteria_set,     # Inclusion criteria defined
        state.protocol_submitted,         # FDA submission done
        state.fda_approved,              # FDA approved protocol
        state.interim_complete,           # Interim analysis done
        state.primary_analysis_complete,  # Final analysis done
        state.conclusion_written,         # Conclusion synthesized
        state.trial_complete,            # Episode complete
    ])
    milestone_fraction = milestones_completed / 12.0

    # Budget efficiency (1.0 = full budget, 0.0 = broke)
    budget_efficiency = max(0.0, state.budget_remaining / state.initial_budget)

    # Phase progress (0.0 to 1.0) — how far through the workflow
    phase_progress = PHASE_ORDER.get(state.current_phase, 0) / 9.0

    # Data quality (0.0 to 1.0) — how much the agent knows about hidden truth
    # Based on posterior uncertainty reduction from Phase I experiments
    data_quality = 1.0 - (state.effect_size_posterior_std / state.effect_size_prior_std) \
                   if state.effect_size_prior_std > 0 else 0.0
    data_quality = max(0.0, min(1.0, data_quality))

    # Weighted combination
    w_milestone = 3.0
    w_budget    = 1.0
    w_phase     = 1.5
    w_data      = 2.0

    return (w_milestone * milestone_fraction
          + w_budget   * budget_efficiency
          + w_phase    * phase_progress
          + w_data     * data_quality)
```

### Why These Weights

| Component | Weight | Rationale |
|-----------|--------|-----------|
| `milestone_fraction` | 3.0 | Strongest gradient — completing milestones is the primary goal |
| `data_quality` | 2.0 | Second strongest — gathering information enables good design |
| `phase_progress` | 1.5 | Moderate — being in the right phase matters but less than milestones |
| `budget_efficiency` | 1.0 | Weakest — budget matters but shouldn't override progress |

### Shaped Reward Computation

```python
def compute_shaping(state_prev, state_next, gamma=0.99):
    """Potential-based shaping: γ·φ(s') − φ(s)"""
    return gamma * phi(state_next) - phi(state_prev)
```

### Properties

- **Start of episode:** φ(s₀) ≈ 1.0 (full budget, no progress)
- **Mid-episode (Phase I complete):** φ(s) ≈ 3.5 (milestones + data gained)
- **Near-completion:** φ(s) ≈ 6.5 (most milestones + good data + phase progress)
- **Telescoping:** Over a full episode, Σ γ·φ(s') − φ(s) ≈ γᵀ·φ(sᵀ) − φ(s₀). The total shaped contribution depends only on start and end states, not the path — so the optimal policy is unchanged.

---

## Weight Tuning Guidelines

These weights are initial values. Tuning should happen during Push 5 based on training diagnostics:

1. **If agent ignores Phase I:** increase `w_info_gain` and `w_data` in shaping
2. **If agent spams same action:** increase repeat penalty magnitude
3. **If agent skips phases:** increase `w_ordering` penalties
4. **If all episodes score ~0:** increase per-step component magnitudes (not enough signal)
5. **If all episodes score similarly:** increase terminal reward magnitudes (not enough variance)
6. **If agent over-optimizes shaping:** reduce shaping weights, increase terminal weight

> **Rule from comparison.md:** "Environment must fight back — too-easy rewards cause plateaus."

---

## Reward Key Contract

These keys are frozen per Section 5 of ROADMAP.md:

```python
REWARD_KEYS = [
    "r_validity",              # Per-step: FDA rule compliance
    "r_ordering",              # Per-step: Phase-order bonus/penalty
    "r_info_gain",             # Per-step: Information gained
    "r_efficiency",            # Per-step: Budget/time efficiency
    "r_novelty",               # Per-step: Action diversity + repeat penalty
    "r_penalty",               # Per-step: Soft constraint violations
    "r_terminal_success",      # Terminal: Trial detected true effect
    "r_terminal_calibration",  # Terminal: Agent claims match hidden truth
]
```

All 8 keys are logged per step in the reward CSV and JSONL transcript for independent debugging.