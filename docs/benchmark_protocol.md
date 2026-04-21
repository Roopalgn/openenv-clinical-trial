# Benchmark Protocol — Random & Scripted Baselines


## Purpose

Baselines serve three critical roles:

1. **Environment validation** — a random policy should score near the bottom of the reward range (-2 to +1). If random scores well, the environment is too easy.
2. **Training proof** — the trained model must beat both baselines. The delta between scripted and trained is the proof of RL learning.
3. **Pitch material** — the 3-minute demo shows before/after contrast. Baselines are the "before."

---

## Baseline 1: Random Policy

### Description

At each step, the agent uniformly samples from the set of currently valid actions (actions not blocked by prerequisite hard constraints).

```python
class RandomPolicy:
    """Uniform random action selection from valid actions."""

    def select_action(self, observation: TrialObservation) -> TrialAction:
        valid_actions = observation.available_actions
        action_type = random.choice(valid_actions)
        params = self._random_params(action_type)
        return TrialAction(action_type=action_type, parameters=params)

    def _random_params(self, action_type: str) -> dict:
        """Generate random but valid parameters for the action type."""
        if action_type == "set_sample_size":
            return {"n": random.randint(30, 500)}
        elif action_type == "set_primary_endpoint":
            return {"endpoint": random.choice(["PFS", "OS", "ORR", "DFS"])}
        elif action_type == "set_dosing_schedule":
            return {"dose_mg": random.choice([50, 100, 150, 200, 250]),
                    "frequency": random.choice(["daily", "weekly", "biweekly"])}
        elif action_type == "set_inclusion_criteria":
            return {"criteria": random.choice(["all_comers", "biomarker_positive",
                                                "age_restricted", "severity_restricted"])}
        elif action_type == "set_control_arm":
            return {"arm": random.choice(["placebo", "standard_of_care", "active_comparator"])}
        elif action_type == "set_randomization_ratio":
            return {"ratio": random.choice(["1:1", "2:1", "3:1"])}
        elif action_type == "set_blinding":
            return {"blinding": random.choice(["double_blind", "single_blind", "open_label"])}
        elif action_type == "run_dose_escalation":
            return {"dose_mg": random.choice([50, 100, 150, 200, 250]),
                    "cohort_size": random.randint(3, 10)}
        elif action_type == "modify_sample_size":
            return {"delta_n": random.randint(-50, 100)}
        else:
            return {}
```

### Expected Performance

| Metric | Expected (Warmup) | Expected (Intermediate) | Expected (Expert) |
|--------|-------------------|------------------------|-------------------|
| Success Rate | 5–15% | 1–5% | < 1% |
| Avg Reward | -1.5 to +0.5 | -2.0 to -0.5 | -2.0 |
| Phase Compliance | 10–20% | 5–10% | 2–5% |
| Avg Steps | 80–100 (timeout) | 70–80 (timeout) | 55–60 (timeout) |
| Milestones Completed | 3–6 / 18 | 2–4 / 18 | 1–3 / 18 |
| FDA Pass Rate | 5–10% | 2–5% | < 1% |

### Why Random is Important

- If random achieves > 20% success at Warmup, the environment is **too easy** — increase noise or reduce rewards
- Random should almost always timeout (hit max_steps) because it doesn't follow the workflow
- Random occasionally stumbles into Phase I completion but rarely reaches Phase II design correctly
- This sets the **floor** — any trained model must beat this convincingly

---

## Baseline 2: Scripted Heuristic Policy

### Description

A hand-coded expert policy that follows the correct clinical trial workflow but makes fixed (not optimal) design choices. This represents "reasonable human behavior" — correct process, suboptimal parameters.

```python
class ScriptedPolicy:
    """Hard-coded clinical trial workflow with fixed parameters.
    
    Follows correct phase order but uses generic design choices:
    - Always uses 150mg dose (may not be optimal for all scenarios)
    - Always uses n=200 (may be over/under-powered)
    - Never uses biomarker stratification (misses subgroups)
    - Never does adaptive modifications
    """

    def __init__(self):
        self.step_count = 0
        self.phase_i_done = False
        self.effect_estimated = False
        self.design_set = False
        self.submitted = False
        self.interim_done = False
        self.analyzed = False

    def select_action(self, observation: TrialObservation) -> TrialAction:
        self.step_count += 1

        # Phase I: 3 dose levels
        if not self.phase_i_done:
            if self.step_count <= 3:
                dose = [50, 100, 150][self.step_count - 1]
                return TrialAction("run_dose_escalation",
                                   {"dose_mg": dose, "cohort_size": 6})
            elif self.step_count == 4:
                return TrialAction("observe_safety_signal", {})
            elif self.step_count == 5:
                self.phase_i_done = True
                return TrialAction("estimate_effect_size", {})

        # Phase II design: fixed choices
        if not self.effect_estimated:
            self.effect_estimated = True

        if not self.design_set:
            design_steps = [
                TrialAction("set_primary_endpoint", {"endpoint": "PFS"}),
                TrialAction("set_sample_size", {"n": 200}),
                TrialAction("set_inclusion_criteria", {"criteria": "all_comers"}),
                TrialAction("set_control_arm", {"arm": "placebo"}),
                TrialAction("set_randomization_ratio", {"ratio": "1:1"}),
                TrialAction("set_blinding", {"blinding": "double_blind"}),
                TrialAction("set_dosing_schedule",
                            {"dose_mg": 150, "frequency": "daily"}),
            ]
            design_idx = self.step_count - 6
            if design_idx < len(design_steps):
                return design_steps[design_idx]
            self.design_set = True

        # Regulatory
        if not self.submitted:
            self.submitted = True
            return TrialAction("submit_to_fda_review", {})

        # Monitoring (single interim)
        if not self.interim_done:
            self.interim_done = True
            return TrialAction("run_interim_analysis", {})

        # Analysis
        if not self.analyzed:
            self.analyzed = True
            return TrialAction("run_primary_analysis", {})

        # Conclusion
        return TrialAction("synthesize_conclusion", {})
```

### Expected Performance

| Metric | Expected (Warmup) | Expected (Intermediate) | Expected (Expert) |
|--------|-------------------|------------------------|-------------------|
| Success Rate | 50–70% | 20–35% | 5–15% |
| Avg Reward | +3.0 to +6.0 | +1.0 to +3.0 | -0.5 to +1.5 |
| Phase Compliance | 95–100% | 90–95% | 85–90% |
| Avg Steps | 15–18 | 15–18 | 15–18 |
| Milestones Completed | 14–16 / 18 | 12–14 / 18 | 10–12 / 18 |
| FDA Pass Rate | 80–90% | 70–80% | 50–60% |

### Why Scripted is Important

- Sets the **ceiling without RL learning** — proves that correct process alone isn't enough for hard scenarios
- The scripted policy fails at Intermediate+ because it never:
  - Discovers the optimal dose (uses fixed 150mg)
  - Identifies responder subgroups (uses `all_comers`)
  - Adapts sample size based on interim results
  - Detects futility early
- The **gap between scripted and trained** is the proof that RL adds value beyond heuristics
- This is the "before" in the before/after story arc (see `story_arc.md`)

---

## Baseline 3: Trained Model (Post-Training)

Evaluated after GRPO training onsite. Expected to beat scripted baseline by learning:

| Capability | Scripted Baseline | Trained Model (Expected) |
|-----------|-------------------|--------------------------|
| Dose optimization | Fixed 150mg | Discovers optimal dose per scenario |
| Subgroup identification | None — uses all_comers | Uses `add_biomarker_stratification` to find responders |
| Adaptive design | None | Adjusts sample size based on interim results |
| Futility detection | None — always runs to completion | Early stops doomed trials (+1.0 futility bonus) |
| Phase efficiency | Fixed 15–18 steps | Variable: 12 steps (easy) to 25 (complex) |
| Expert scenarios | 5–15% success | Target: 25–40% success |

---

## Benchmark Execution Protocol

### Step 1: Run Random Baseline

```bash
python eval_compare.py \
    --policy random \
    --episodes 100 \
    --scenarios all \
    --tier warmup \
    --seed 42 \
    --output results/random_warmup.json

# Repeat for each tier
python eval_compare.py --policy random --episodes 50 --tier beginner --seed 42 --output results/random_beginner.json
python eval_compare.py --policy random --episodes 50 --tier intermediate --seed 42 --output results/random_intermediate.json
python eval_compare.py --policy random --episodes 50 --tier advanced --seed 42 --output results/random_advanced.json
python eval_compare.py --policy random --episodes 50 --tier expert --seed 42 --output results/random_expert.json
```

### Step 2: Run Scripted Baseline

```bash
python eval_compare.py \
    --policy scripted \
    --episodes 100 \
    --scenarios all \
    --tier warmup \
    --seed 42 \
    --output results/scripted_warmup.json

# Repeat for each tier (same commands with --policy scripted)
```

### Step 3: Run Trained Model (Onsite)

```bash
python eval_compare.py \
    --policy trained \
    --checkpoint checkpoints/grpo_final/ \
    --episodes 100 \
    --scenarios all \
    --tier warmup \
    --seed 42 \
    --output results/trained_warmup.json

# Repeat for each tier
```

### Step 4: Generate Comparison Report

```bash
python eval_compare.py \
    --compare results/random_warmup.json results/scripted_warmup.json results/trained_warmup.json \
    --output results/comparison_report.md
```

---

## Output Schema

Each evaluation run produces a JSON file:

```json
{
    "policy": "random | scripted | trained",
    "tier": "warmup",
    "episodes": 100,
    "seed": 42,
    "timestamp": "2026-04-25T10:30:00Z",
    "aggregate": {
        "success_rate": 0.12,
        "avg_reward": -0.85,
        "avg_steps": 87.3,
        "avg_milestones": 4.2,
        "fda_pass_rate": 0.08,
        "phase_compliance_rate": 0.15,
        "avg_power": 0.45,
        "timeout_rate": 0.82
    },
    "per_scenario": {
        "solid_tumor_chemo": {
            "episodes": 25,
            "success_rate": 0.16,
            "avg_reward": -0.62,
            "avg_steps": 85.1
        },
        "autoimmune_biologic": { "..." : "..." },
        "cns_depression": { "..." : "..." },
        "rare_disease_orphan": { "..." : "..." }
    },
    "reward_distribution": {
        "min": -2.0,
        "p25": -1.8,
        "median": -1.2,
        "p75": 0.1,
        "max": 3.5
    }
}
```

---

## Reproducibility Requirements

1. **Seeded randomness** — all baselines use `--seed 42` (or any fixed seed) for reproducible runs
2. **NoiseModel seeding** — the `NoiseModel` uses the same `numpy.Generator` seed for identical scenario parameterization
3. **Same scenarios** — all three policies run through the same scenario sequence (deterministic from seed)
4. **Episode transcripts** — each run saves JSONL transcripts per episode for post-hoc analysis
5. **Versioned environment** — lock the environment code version before running baselines (git commit hash in output)


---

## Pitch Integration

The benchmark results directly feed into the 3-minute pitch (see `story_arc.md`):

- **Act 1 (Cold Start):** Show random policy transcript — chaos, no learning
- **Act 2 (First Light):** Show scripted baseline — correct process but blind to nuance
- **Act 3 (Fights Back):** Show trained model struggling with Intermediate scenarios
- **Act 4 (Mastery):** Show trained model beating scripted at Advanced — discovered subgroup, adapted design

| Pitch Metric | Random | Scripted | Trained | Delta |
|-------------|--------|----------|---------|-------|
| Warmup Success | 10% | 60% | 85% | +25% vs scripted |
| Intermediate Success | 3% | 25% | 50% | +25% vs scripted |
| Expert Success | <1% | 10% | 30% | +20% vs scripted |
| Avg Reward (Warmup) | -1.2 | +4.0 | +8.5 | +4.5 vs scripted |