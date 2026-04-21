# Curriculum Progression Policy & Mastery Thresholds

> **Inspired by:** KubeSRE's 3-tier curriculum with automatic advancement (70% success → advance). Bio Experiment's progressively harder parameter ranges gated by mastery. VRAM's round-robin environment rotation ensuring broad coverage. All winners: curriculum is the primary mechanism for preventing reward plateau.

## Design Principles

1. **Automatic advancement** — agent promotion is triggered programmatically by performance windows, not manually (KubeSRE pattern)
2. **Per-scenario tracking** — advancement is tracked per scenario type, not globally (Bio pattern: per-organism mastery)
3. **Sliding window** — success rate is computed over the last W episodes per scenario, not lifetime (prevents history dilution)
4. **No demotion by default** — once advanced, the agent stays at the new tier (GRPO updates ensure policy doesn't catastrophically regress)
5. **Weak-spot targeting** — after advancement, the curriculum controller preferentially selects the scenario type with the lowest success rate (KubeSRE pattern: adversarial designer targets weak spots)

---

## Tier Definitions

| Tier | Difficulty Range | Episode Budget | Judge Persona | Noise Level | Advancement Threshold |
|------|-----------------|----------------|---------------|-------------|-----------------------|
| 0 — **Warmup** | 0.00–0.25 | 100 steps | Junior (hints) | Low (±10%) | 60% success over W=10 |
| 1 — **Beginner** | 0.25–0.40 | 90 steps | Junior→Senior | Medium (±20%) | 55% success over W=15 |
| 2 — **Intermediate** | 0.40–0.60 | 80 steps | Senior | High (±30%) | 45% success over W=20 |
| 3 — **Advanced** | 0.60–0.80 | 70 steps | Senior→Principal | Very High (±40%) | 35% success over W=25 |
| 4 — **Expert** | 0.80–0.95 | 60 steps | Principal | Extreme (±50%) | Terminal — no further tier |

### Why These Thresholds

- **Decreasing success requirement** — harder tiers have lower thresholds because scenarios are genuinely harder (smaller effects, tighter budgets, rarer subgroups). A 35% success rate at Advanced means the agent is reliably designing good trials under adversarial conditions.
- **Increasing window size** — harder tiers require more episodes to prove mastery, preventing lucky streaks from causing premature advancement.
- **KubeSRE precedent** — KubeSRE used ~70% success to advance across 3 tiers. Our 5-tier system with decreasing thresholds matches the same spirit while accommodating the deeper curriculum.

---

## Advancement Logic

### Per-Scenario Advancement

```python
class CurriculumController:
    """Tracks per-scenario mastery and controls tier advancement."""

    def __init__(self):
        self.tier = 0                          # Global tier (0-4)
        self.scenario_results = {              # Per-scenario sliding windows
            "solid_tumor_chemo": deque(maxlen=25),
            "autoimmune_biologic": deque(maxlen=25),
            "cns_depression": deque(maxlen=25),
            "rare_disease_orphan": deque(maxlen=25),
        }
        self.tier_config = TIER_CONFIG         # See below
        self.episodes_in_tier = 0              # Minimum episodes before advancement check

    def record_episode(self, scenario_id: str, success: bool):
        """Record episode outcome for a scenario."""
        self.scenario_results[scenario_id].append(success)
        self.episodes_in_tier += 1

    def should_advance(self) -> bool:
        """Check if agent should advance to next tier."""
        if self.tier >= 4:
            return False  # Already at Expert

        config = self.tier_config[self.tier]

        # Minimum episodes gate — prevent advancement in first few episodes
        if self.episodes_in_tier < config["min_episodes"]:
            return False

        # Check ALL scenario types have enough data
        for scenario_id, results in self.scenario_results.items():
            window = config["window_size"]
            if len(results) < window:
                return False  # Not enough data for this scenario

            recent = list(results)[-window:]
            success_rate = sum(recent) / len(recent)

            if success_rate < config["threshold"]:
                return False  # This scenario not mastered yet

        return True  # All scenarios meet threshold

    def advance(self):
        """Move to next tier. Reset episode counter."""
        self.tier = min(self.tier + 1, 4)
        self.episodes_in_tier = 0
        # Do NOT clear scenario_results — keep history for weak-spot targeting

    def get_next_scenario(self) -> str:
        """Select next scenario with weak-spot targeting."""
        return _select_scenario_weakspot(
            self.scenario_results,
            self.tier_config[self.tier]["window_size"],
            weakspot_probability=0.70,
        )
```

### Tier Configuration

```python
TIER_CONFIG = {
    0: {  # Warmup
        "threshold": 0.60,       # 60% success rate to advance
        "window_size": 10,       # Over last 10 episodes per scenario
        "min_episodes": 20,      # At least 20 episodes before checking (5 per scenario)
        "max_steps": 100,
        "noise_scale": 0.10,     # ±10% domain randomization
        "judge_persona": "junior",
        "hints_enabled": True,
    },
    1: {  # Beginner
        "threshold": 0.55,
        "window_size": 15,
        "min_episodes": 30,
        "max_steps": 90,
        "noise_scale": 0.20,
        "judge_persona": "junior_senior",
        "hints_enabled": False,
    },
    2: {  # Intermediate
        "threshold": 0.45,
        "window_size": 20,
        "min_episodes": 40,
        "max_steps": 80,
        "noise_scale": 0.30,
        "judge_persona": "senior",
        "hints_enabled": False,
    },
    3: {  # Advanced
        "threshold": 0.35,
        "window_size": 25,
        "min_episodes": 50,
        "max_steps": 70,
        "noise_scale": 0.40,
        "judge_persona": "senior_principal",
        "hints_enabled": False,
    },
    4: {  # Expert
        "threshold": None,        # Terminal tier — no advancement
        "window_size": 25,
        "min_episodes": None,
        "max_steps": 60,
        "noise_scale": 0.50,
        "judge_persona": "principal",
        "hints_enabled": False,
    },
}
```

---

## Scenario Selection — Weak-Spot Targeting

> **Inspired by:** KubeSRE's adversarial designer that targets the agent's weakest fault types. Bio Experiment's `design_followup` targeting least-explored organisms.

Instead of round-robin or random scenario selection, 70% of episodes target the agent's weakest scenario type:

```python
def _select_scenario_weakspot(
    scenario_results: dict,
    window_size: int,
    weakspot_probability: float = 0.70,
) -> str:
    """
    Select scenario with 70% probability of targeting weakest type.
    30% random for exploration diversity.
    """
    import random

    if random.random() > weakspot_probability:
        # 30%: random selection for coverage
        return random.choice(list(scenario_results.keys()))

    # 70%: target weakest scenario
    success_rates = {}
    for scenario_id, results in scenario_results.items():
        if len(results) < 3:
            # Not enough data — prioritize under-explored scenarios
            success_rates[scenario_id] = -1.0
        else:
            recent = list(results)[-window_size:]
            success_rates[scenario_id] = sum(recent) / len(recent)

    # Return scenario with lowest success rate
    return min(success_rates, key=success_rates.get)
```

### Why 70/30 Split

- **70% weak-spot:** Focuses training compute on scenarios the agent struggles with (KubeSRE: adversarial designer targets failure modes)
- **30% random:** Prevents catastrophic forgetting of mastered scenarios and ensures all types get periodic coverage
- **Bio precedent:** Bio Experiment used similar approach — `design_followup` targeted least-understood organisms rather than random selection

---

## Fast-Track Advancement

> **Inspired by:** KubeSRE: 90%+ success rate skips minimum episode requirement

If the agent achieves **≥ 90% success rate** across ALL scenarios in the current window, the `min_episodes` requirement is waived:

```python
def should_fast_track(self) -> bool:
    """Check if agent qualifies for fast-track advancement (skip min_episodes)."""
    if self.tier >= 4:
        return False

    config = self.tier_config[self.tier]
    window = config["window_size"]

    for scenario_id, results in self.scenario_results.items():
        if len(results) < window:
            return False
        recent = list(results)[-window:]
        if sum(recent) / len(recent) < 0.90:
            return False

    return True  # All scenarios ≥ 90% — skip min_episodes gate
```

This prevents a clearly-ready agent from being held back by minimum episode requirements. In training, this can save 30–50 episodes of wasted compute at easy tiers.

---

## Difficulty Scaling per Tier

When the curriculum controller selects a scenario, it applies tier-specific difficulty parameters via the `NoiseModel`:

### Noise Model Scaling

| Parameter | Warmup | Beginner | Intermediate | Advanced | Expert |
|-----------|--------|----------|--------------|----------|--------|
| `budget_multiplier` | 1.2 | 1.0 | 0.85 | 0.70 | 0.55 |
| `time_multiplier` | 1.2 | 1.0 | 0.85 | 0.70 | 0.55 |
| `effect_size_multiplier` | 1.5 | 1.0 | 0.80 | 0.60 | 0.40 |
| `dropout_multiplier` | 0.5 | 1.0 | 1.3 | 1.6 | 2.0 |
| `noise_sigma` | 0.05 | 0.08 | 0.12 | 0.18 | 0.25 |
| `placebo_boost` | 0.0 | 0.0 | +0.05 | +0.10 | +0.15 |
| `subgroup_prevalence_mult` | 1.5 | 1.0 | 0.80 | 0.60 | 0.40 |

### What Changes at Each Tier

| Tier | Environment Changes | Agent Impact |
|------|-------------------|--------------|
| **Warmup** | Large effects, generous budget/time, low noise, no hidden subgroups | Agent learns basic workflow: dose escalation → design → submit → analyze |
| **Beginner** | Standard parameters (matching scenario card base values) | Agent must estimate effect size accurately and set appropriate sample size |
| **Intermediate** | Higher noise, tighter budget, placebo boost begins masking signal | Agent must use biomarker stratification and design efficiently |
| **Advanced** | Misleading early signals, rare subgroups, tight constraints | Agent must handle ambiguity, use adaptive designs, make tough futility calls |
| **Expert** | Tiny effects, high dropout, extreme noise, minimal budget | Agent must be near-optimal: perfect Phase I, precise design, early futility detection |

---

## Interaction with Other Components

### → Reward Spec (`reward_spec.md`)

- Judge persona scaling changes penalty/bonus magnitudes per tier
- Terminal reward expectations shift: Warmup expects +5–7; Expert considers +3 a strong result
- `r_shaping` weights remain constant across tiers (shaping doesn't depend on difficulty)

### → Milestone Map (`milestone_map.md`)

- `MILESTONE_VELOCITY_TARGETS` define per-tier step budgets
- Faster milestone completion at harder tiers contributes to fast-track advancement signal
- Milestone fraction at episode end is the primary shaping component

### → Scenario Cards (`scenario_cards.md`)

- Each scenario card includes a curriculum scaling table showing per-tier parameter overrides
- The `NoiseModel` applies tier multipliers on top of scenario base values
- Cross-scenario difficulty matrix ensures tier 2 of `rare_disease_orphan` is harder than tier 2 of `solid_tumor_chemo`

### → Phase Workflow (`phase_workflow.md`)

- Judge persona escalates with tier: junior → senior → principal
- Phase-order skip penalty increases: -0.3 (tiers 0–2) → -0.5 (tiers 3–4)
- Redundancy penalty activates at tier 3+ (-0.1 to -0.15 per repeat)

---

## Curriculum State Logging

Every episode logs curriculum state for training diagnostics:

```python
CURRICULUM_LOG_SCHEMA = {
    "episode_id": int,
    "tier": int,                        # 0-4
    "tier_name": str,                   # "warmup", "beginner", etc.
    "scenario_id": str,
    "selection_method": str,            # "weakspot" or "random"
    "noise_params": dict,               # Applied NoiseModel parameters
    "success": bool,
    "total_reward": float,
    "milestones_completed": int,        # Out of 18
    "steps_taken": int,
    "per_scenario_success_rates": dict, # Current window success rates
    "should_advance": bool,             # Whether this episode triggered advancement
    "fast_track": bool,                 # Whether fast-track was eligible
}
```

This log powers the curriculum progression charts in the dashboard and is essential for debugging training plateaus.

---

## Expected Training Trajectory

Based on winner analysis (KubeSRE: ~200 episodes to converge, Bio: 150-step training loops):

| Phase | Episodes | Expected Tier | Key Behavior Changes |
|-------|----------|--------------|---------------------|
| 1–20 | Exploration | 0 (Warmup) | Agent learns action vocabulary, begins following phase order |
| 20–50 | Warmup mastery | 0→1 | Agent consistently completes warmup scenarios, discovers dose escalation→design→submit flow |
| 50–100 | Beginner→Intermediate | 1→2 | Agent learns biomarker stratification, sample size calculation aligned to effect estimate |
| 100–200 | Intermediate→Advanced | 2→3 | Agent handles noise, makes futility decisions, adapts designs mid-trial |
| 200–400 | Advanced→Expert | 3→4 | Agent near-optimal: precise Phase I, targeted enrollment, adaptive strategies |
| 400–800 | Expert refinement | 4 | Agent optimizes efficiency within expert scenarios, reward plateaus near +8–10 |

> **Compute note:** Post-training with HuggingFace H100 credits onsite April 25–26. Expect ~200–400 episodes per hour with vLLM colocate. Full trajectory (800 episodes) possible in 2–4 hours.
