# Chapter 6: Hidden State — The Secret Truth

## The Fog of War

Imagine playing a strategy game where the map is hidden. You can only see the area around your units. You know enemies exist, but not where or how many. This is called the **fog of war**, and it's exactly what our agent faces.

In our clinical trial environment:
- **What exists** (hidden): The drug's true effect, true side effects, true responder population
- **What the agent sees**: Noisy measurements, partial estimates, incomplete data

This concept is formally called **partial observability**, and it's one of the things that makes our project genuinely challenging.

## The Two-Layer Architecture

Our project has a clean architectural separation between truth and perception:

```
┌─────────────────────────────────────────┐
│         TransitionEngine                 │
│    (Updates the REAL hidden state)       │
│                                         │
│  TrialLatentState:                      │
│    true_effect_size = 0.38              │  ← THE TRUTH
│    true_side_effect_rate = 0.22         │     (agent never sees this)
│    budget_remaining = $6,450,000        │
│    patients_enrolled = 120              │
│    phase_i_complete = True              │
│    ...                                  │
└────────────────┬────────────────────────┘
                 │
                 │ Hidden state flows DOWN
                 ▼
┌─────────────────────────────────────────┐
│         OutputGenerator                  │
│    (Creates NOISY observation from       │
│     the hidden state + noise model)     │
│                                         │
│  TrialObservation:                      │
│    observed_effect_size = 0.41 ± 0.12   │  ← NOISY VERSION
│    observed_adverse_rate = 0.19         │     (what agent actually sees)
│    budget_remaining = $6,450,000        │
│    patients_enrolled = 120              │
│    phase_i_complete = True              │
│    ...                                  │
└─────────────────────────────────────────┘
```

Notice:
- The **true effect size** is 0.38, but the agent sees 0.41 (with ±0.12 uncertainty)
- The **true adverse event rate** is 0.22, but the agent sees 0.19 (site variability adds noise)  
- Some values (budget, patients enrolled) are shown accurately — these are observable facts
- Milestone flags (phase_i_complete) are shown accurately — these are binary facts

> **Design Decision Box: Why Separate TransitionEngine and OutputGenerator?**
>
> This is called the **Bio Experiment Pattern** — it mirrors how real biology experiments work:
> 1. **Reality** evolves (cells divide, proteins fold) — TransitionEngine
> 2. **Measurement** introduces noise (instruments have limited precision) — OutputGenerator
>
> By keeping them separate:
> - We can test the transition logic without worrying about noise
> - We can adjust noise levels without touching the transition logic
> - It's architecturally clean: one module mutates truth, another creates perception

## The TransitionEngine: Updating Hidden Reality

When the agent takes an action, the TransitionEngine updates the hidden state. Here's what happens for each action type:

### Budget and Time Costs

Every action costs money and time:

```python
# From server/simulator/transition_engine.py
_ACTION_COSTS = {
    "set_primary_endpoint":        $5,000,
    "set_sample_size":             $2,000,
    "set_dosing_schedule":         $10,000,
    "run_dose_escalation":         $50,000,     # Expensive!
    "observe_safety_signal":       $15,000,
    "estimate_effect_size":        $20,000,
    "run_interim_analysis":        $30,000,
    "submit_to_fda_review":        $100,000,    # Most expensive
    "run_primary_analysis":        $50,000,
    "enroll_patients":             $10,000/patient,  # Scales with count
}

_ACTION_TIME_DAYS = {
    "set_primary_endpoint":        7 days,
    "run_dose_escalation":         90 days,     # Slow!
    "submit_to_fda_review":        180 days,    # Very slow
    "enroll_patients":             2 days/patient,
}
```

**Real-world grounding:** These numbers reflect actual clinical trial costs. FDA review really does take ~6 months. Dose escalation studies really do take ~3 months.

### Milestone Flags

Actions set milestone flags that unlock later actions:

```python
# When you run dose escalation:
if action.action_type == ActionType.RUN_DOSE_ESCALATION:
    latent.phase_i_complete = True
    latent.mtd_identified = True

# When you estimate effect size:
if action.action_type == ActionType.ESTIMATE_EFFECT_SIZE:
    latent.effect_estimated = True

# When you submit to FDA:
if action.action_type == ActionType.SUBMIT_TO_FDA_REVIEW:
    latent.protocol_submitted = True
```

These flags are important because they're **prerequisites** for other actions. You can't submit to FDA without completing Phase I. You can't run the primary analysis without a completed interim analysis.

### Soft Degradation

Here's a clever feature: when the agent makes questionable decisions, the data quality degrades:

```python
# Low confidence → more measurement noise (harder to read future results)
if action.confidence < 0.5:
    degradation_factor = 1.0 + (0.5 - action.confidence)
    latent.measurement_noise = min(latent.measurement_noise * degradation_factor, 0.5)

# Over budget → more site variability (sites cut corners)
if latent.budget_remaining < 0:
    latent.site_variability = min(latent.site_variability * 1.2, 0.5)

# Over time → more dropouts (patients leave)
if latent.time_remaining_days < 0:
    latent.dropout_rate = min(latent.dropout_rate * 1.15, 0.8)
```

This models real-world effects: underfunded trials have lower data quality, prolonged trials have higher dropout rates.

### Adverse Events

Some actions can cause adverse events (side effects in patients):

```python
# Stochastic adverse events based on true side effect rate
if action.action_type in {ENROLL_PATIENTS, OBSERVE_SAFETY_SIGNAL, RUN_DOSE_ESCALATION}:
    if rng.random() < latent.true_side_effect_rate:
        latent.adverse_events += 1
        latent.site_variability += 0.02  # Each AE increases data noise
```

## The OutputGenerator: Creating Noisy Observations

The OutputGenerator takes the clean hidden state and creates a noisy observation that the agent sees. Let's trace through what the agent gets to see:

### Effect Size (Noisy)

```python
# Agent can only see this AFTER running estimate_effect_size
if latent.effect_estimated:
    # True effect is 0.38, but we add noise
    noisy_effect = latent.true_effect_size + rng.normal(0.0, noise_std)
    # Agent sees: 0.41 (the true value was 0.38)
    phase_data["observed_effect_size"] = round(noisy_effect, 4)
    
    # Confidence interval also noisy
    ci_half_width = rng.normal(noise_std * 2, noise_std * 0.5)
    phase_data["effect_size_ci"] = (noisy_effect - ci_half_width, 
                                     noisy_effect + ci_half_width)
    # Agent sees CI: (0.29, 0.53) around the noisy estimate
```

### Adverse Event Rate (Noisy)

```python
# Only visible after Phase I or safety observation
if latent.phase_i_complete or "observe_safety_signal" in latent.action_history:
    noisy_ae_rate = latent.true_side_effect_rate + rng.normal(0.0, site_std)
    noisy_ae_rate = clip(noisy_ae_rate, 0.0, 1.0)  # Keep in valid range
    phase_data["observed_adverse_event_rate"] = round(noisy_ae_rate, 4)
```

### Available Actions (Filtered)

The agent sees which actions it CAN take right now:

```python
def _build_available_actions(self, latent):
    # Phase-permitted actions (from FDA transition table)
    phase_permitted = TRANSITION_TABLE[latent.episode_phase]
    
    # Further filtered by prerequisites
    available = []
    for action_type in phase_permitted:
        if self._prerequisites_met(action_type, latent):
            available.append(action_type.value)
    return available
```

This is important — the agent doesn't have to guess which actions are valid. The environment tells it.

### What the Agent NEVER Sees

Even late in the episode, certain things remain hidden:

| Hidden Value | Why It's Hidden |
|---|---|
| `true_effect_size` | This is what the trial is trying to determine |
| `true_side_effect_rate` | Must be estimated through observation |
| `true_responder_population` | Must be discovered through biomarker stratification |
| `true_dose_response` | Must be discovered through dose escalation |
| `true_mechanism` | Never revealed (like real life) |
| `measurement_noise` | Agent doesn't know how noisy its measurements are |
| `site_variability` | Agent doesn't know site quality |

## The NoiseModel: Domain Randomization

The NoiseModel ensures that even with the same scenario, each episode has different parameters:

```python
class NoiseModel:
    _BUDGET_NOISE: float = 0.30   # Budget varies ±30%
    _TIME_NOISE: float = 0.20     # Time varies ±20%
    _DROPOUT_NOISE: float = 0.15  # Dropout varies ±15%
    _PLACEBO_NOISE: float = 0.20  # Placebo response varies ±20%
    
    def randomize(self, config):
        # Budget: $10M × (1.0 + uniform(-0.30, +0.30))
        # Could be anywhere from $7M to $13M
        budget_factor = 1.0 + self._rng.uniform(-0.30, 0.30)
        new_budget = config.budget_usd * budget_factor
        
        # Same for time, dropout, placebo
        ...
        return config.copy(update={...})
```

> **Design Decision Box: Why Domain Randomization?**
>
> Without noise, the agent could **memorize** specific scenarios: "In solid_tumor_chemo, always use N=200." This is called **overfitting** — the model works perfectly on training data but fails on anything slightly different.
>
> Domain randomization forces the agent to **generalize**: "For an oncology trial with effect size ~0.3–0.5, I need N ≈ 150–250." This is like how driving instructors make students practice in rain, snow, and traffic — not just sunny parking lots.

### Reproducibility: Same Seed = Same Episode

Despite all this randomness, the same seed always produces the same episode:

```python
noise_model = NoiseModel(seed=42)
# seed=42 ALWAYS produces the same randomized budget, same hidden truth, 
# same noise patterns. This is crucial for:
# 1. Debugging: "Episode with seed=42 failed. Let me replay it."
# 2. Evaluation: "Compare agent A vs agent B on the same 100 seeds."
# 3. Testing: "This test should always produce these exact values."
```

This is achieved by using **seeded numpy random generators** (`np.random.default_rng(seed)`), which produce deterministic pseudo-random sequences.

## Information Flow: A Complete Trace

Let's trace the complete information flow for one specific action — the agent runs an interim analysis:

```
Agent sends: {"action_type": "run_interim_analysis", "parameters": {}, 
              "justification": "Check if drug effect is significant", "confidence": 0.7}

1. FDA CHECK (fda_rules.py):
   ├── Is "run_interim_analysis" permitted in current phase "monitoring"? → YES ✓
   ├── Are patients enrolled > 0? → YES (120 patients) ✓
   └── All prerequisites met? → YES ✓

2. TRANSITION ENGINE:
   ├── budget_remaining: $6,450,000 → $6,420,000 (-$30,000 for interim analysis)
   ├── time_remaining_days: 290 → 230 (-60 days for interim analysis)
   ├── interim_complete: False → True  ← milestone unlocked!
   └── action_history: [..., "run_interim_analysis"]

3. PHASE DETECTOR:
   ├── Action maps to phase "monitoring"
   └── Previous phase was "monitoring" → ✓ correct order

4. TRIAL SIMULATOR:
   ├── effect_size = 0.38, n = 120, alpha = 0.05
   ├── power = calculate_power(0.38, 120, 0.05) → 0.72
   ├── p_value = 0.031 ← based on noisy observed effect
   └── success = True (p < 0.05)

5. REWARD COMPUTER:
   ├── r_validity = +1.0 (action was valid)
   ├── r_ordering = +0.2 (correct phase order)
   ├── r_info_gain = +0.36 (info action × power = 0.5 × 0.72)
   ├── r_efficiency = +1.29 (good budget remaining)
   ├── r_novelty = +0.2 (first time doing interim analysis)
   ├── r_penalty = 0.0
   ├── r_terminal = 0.0 (not terminal yet)
   ├── + shaping bonus = +0.12 (milestone completion increased)
   └── TOTAL: +3.17

6. OUTPUT GENERATOR:
   ├── observed_effect_size: 0.41 ± 0.12 (noisy version of 0.38)
   ├── observed_placebo_response: 0.11 (noisy version of 0.09)
   ├── budget_remaining: $6,420,000
   └── available_actions: ["run_primary_analysis", "observe_safety_signal", ...]
         ↑ Now includes "run_primary_analysis" because interim_complete=True!

7. JUDGE:
   ├── Layer 1: power=0.72, p=0.031, FDA=pass → PASSED
   ├── Layer 2: "Good timing for interim analysis given enrollment level."
   └── No overconfidence penalty (confidence 0.7 < 0.8 threshold)

Agent receives: TrialObservation + reward=3.17 + done=False
```

This single step involved 7 different components working together. Yet to the agent, it just sees: "I ran an interim analysis. I got a reward of 3.17. Here's what the data looks like now."

---

## Chapter 6 Glossary

| Keyword | Definition |
|---------|-----------|
| **Partial Observability** | Agent can't see the full state, only noisy observations |
| **POMDP** | Partially Observable Markov Decision Process (formal framework) |
| **Latent State** | The true hidden state of the world |
| **Observation** | What the agent actually sees (noisy, incomplete) |
| **Fog of War** | Game analogy for partial observability |
| **Domain Randomization** | Adding random variation to prevent memorization |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Generalization** | Model works on new, unseen situations |
| **Seeded RNG** | Random number generator with a fixed starting seed (reproducible) |
| **TransitionEngine** | Component that updates the hidden state based on actions |
| **OutputGenerator** | Component that creates a noisy observation from hidden state |
| **Bio Experiment Pattern** | Architecture pattern separating reality evolution from measurement |
| **Soft Degradation** | Gradually worsening data quality due to poor decisions |
| **Milestone Flag** | Boolean indicating whether a key trial phase is complete |
