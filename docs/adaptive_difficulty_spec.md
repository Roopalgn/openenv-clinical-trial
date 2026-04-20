# Adaptive Difficulty Specification (G4)

> **Inspired by:** KubeSRE's AdversarialDesigner that uses Claude to design incidents targeting agent weak spots. Bio Experiment's progressive noise amplification on mastered organisms. VRAM's dynamic capability thresholds. All winners: difficulty escalation is the primary mechanism for preventing reward plateau and demonstrating genuine improvement.

## Purpose

After the agent masters a scenario type (>70% success over a sliding window), the environment should **not** keep serving easy instances. Instead, it should randomize harder parameter ranges — tighter budgets, rarer diseases, compound endpoints, misleading Phase I signals — and preferentially target the agent's tracked weak spots.

This document specifies:
1. **Mastery detection** — when does the environment consider a scenario "mastered"?
2. **Parameter hardening** — what changes when a scenario gets harder?
3. **Weak-spot targeting** — how does the environment exploit the agent's failure patterns?
4. **Compound challenges** — how do multiple difficulty axes combine at expert tier?
5. **Solvability guarantee** — how do we ensure generated scenarios remain solvable?

---

## 1. Mastery Detection

### Per-Scenario Mastery

Mastery is tracked **per scenario type** × **per curriculum tier**, not globally. This prevents a strong performance on `solid_tumor_chemo` from masking weakness on `rare_disease_orphan`.

```python
def is_mastered(scenario_id: str, tier: int, results: deque, config: dict) -> bool:
    """Check if agent has mastered a scenario at the current tier."""
    window = config["mastery_window"]
    threshold = config["mastery_threshold"]

    if len(results) < window:
        return False

    recent = list(results)[-window:]
    success_rate = sum(recent) / len(recent)
    return success_rate >= threshold
```

### Mastery Thresholds by Tier

| Tier | Mastery Window | Success Threshold | Min Episodes Before Check |
|------|---------------|-------------------|--------------------------|
| 0 — Warmup | 10 | 70% | 15 |
| 1 — Beginner | 12 | 65% | 20 |
| 2 — Intermediate | 15 | 55% | 25 |
| 3 — Advanced | 20 | 45% | 30 |
| 4 — Expert | N/A | N/A (terminal) | N/A |

### Fast-Track Detection

If the agent achieves ≥90% success in a scenario within the first 5 episodes of a new tier, it is immediately flagged for parameter hardening **without waiting for the full mastery window**. This prevents over-training on easy scenarios.

```python
def fast_track_check(scenario_id: str, tier: int, results: deque) -> bool:
    """Skip mastery window if agent is clearly above tier."""
    if len(results) < 5:
        return False
    recent_5 = list(results)[-5:]
    return sum(recent_5) / 5 >= 0.90
```

---

## 2. Parameter Hardening

When a scenario is mastered, the `NoiseModel` shifts its randomization ranges to harder values. This happens **within the same tier** — it is not a tier advancement. The agent stays at, say, Tier 2, but the Tier 2 version of `solid_tumor_chemo` becomes harder.

### Hardening Axes

Each scenario has multiple axes that can be independently hardened:

| Axis | Easy Range | Hard Range | Effect |
|------|-----------|------------|--------|
| `effect_size_multiplier` | 1.0× | 0.5×–0.7× | True effect becomes harder to detect |
| `budget_multiplier` | 1.0× | 0.6×–0.8× | Less money for experiments |
| `time_multiplier` | 1.0× | 0.7×–0.85× | Fewer days to complete trial |
| `dropout_rate_add` | +0% | +5%–15% | More patients leave the trial |
| `placebo_boost` | +0% | +5%–10% | Placebo response masks drug effect |
| `noise_multiplier` | 1.0× | 1.3×–1.8× | Noisier measurements |
| `subgroup_prevalence_mult` | 1.0× | 0.5×–0.7× | Responder subgroup is rarer |
| `misleading_phase_i` | False | True | Early cohorts show contradictory signal |

### Hardening Schedule

When mastery is detected, hardening is applied **incrementally** — not all at once:

```python
HARDENING_SCHEDULE = {
    "step_1": {  # First mastery detection
        "effect_size_multiplier": 0.85,
        "budget_multiplier": 0.90,
        "noise_multiplier": 1.2,
    },
    "step_2": {  # Second consecutive mastery (still mastered after step_1)
        "effect_size_multiplier": 0.70,
        "budget_multiplier": 0.80,
        "dropout_rate_add": 0.05,
        "noise_multiplier": 1.4,
    },
    "step_3": {  # Third consecutive mastery
        "effect_size_multiplier": 0.60,
        "budget_multiplier": 0.70,
        "time_multiplier": 0.85,
        "dropout_rate_add": 0.10,
        "placebo_boost": 0.05,
        "noise_multiplier": 1.6,
    },
    "step_4_max": {  # Maximum hardening within tier
        "effect_size_multiplier": 0.50,
        "budget_multiplier": 0.60,
        "time_multiplier": 0.70,
        "dropout_rate_add": 0.15,
        "placebo_boost": 0.10,
        "noise_multiplier": 1.8,
        "subgroup_prevalence_mult": 0.70,
        "misleading_phase_i": True,
    },
}
```

### Per-Scenario Hardening Examples

#### `solid_tumor_chemo` — Hardening Progression

| Level | true_effect_size | budget | EGFR+ prevalence | noise | New Challenge |
|-------|-----------------|--------|------------------|-------|---------------|
| Base | 0.31 | $2.5M | 35% | 0.08 | Standard |
| Step 1 | 0.26 | $2.25M | 35% | 0.10 | Smaller effect, noisier |
| Step 2 | 0.22 | $2.0M | 35% | 0.11 | Budget pressure forces efficient design |
| Step 3 | 0.19 | $1.75M | 25% | 0.13 | Rarer subgroup, must enrich early |
| Step 4 | 0.16 | $1.5M | 20% | 0.14 | Near-impossible without perfect enrichment + adaptive design |

#### `autoimmune_biologic` — Hardening Progression

| Level | U-shape visibility | budget | dose_noise | New Challenge |
|-------|-------------------|--------|------------|---------------|
| Base | Clear U-shape | $1.8M | 0.10 | Standard dose-response |
| Step 1 | Slightly masked | $1.6M | 0.15 | Need more dose levels |
| Step 2 | Plateau looks flat | $1.4M | 0.20 | Must use interim to detect U-shape |
| Step 3 | Inverted signal at low N | $1.2M | 0.25 | Misleading Phase I |
| Step 4 | Minimal U-signal | $1.0M | 0.30 | Requires adaptive Bayesian approach |

#### `cns_depression` — Hardening Progression

| Level | placebo_response | true_effect | measurement_noise | New Challenge |
|-------|-----------------|-------------|-------------------|---------------|
| Base | 30% | 18% over placebo | 0.12 | High placebo masks effect |
| Step 1 | 35% | 16% | 0.15 | Even higher placebo |
| Step 2 | 38% | 14% | 0.18 | Signal barely above noise |
| Step 3 | 40% | 12% | 0.20 | Must use large N or biomarker enrichment |
| Step 4 | 42% | 10% | 0.22 | Requires run-in period + stringent inclusion criteria |

#### `rare_disease_orphan` — Hardening Progression

| Level | max_patients | effect_size (Cohen's d) | budget | New Challenge |
|-------|-------------|------------------------|--------|---------------|
| Base | 50 | 1.2 | $1.2M | Small N, large effect |
| Step 1 | 40 | 1.0 | $1.1M | Fewer patients |
| Step 2 | 35 | 0.9 | $1.0M | Must use adaptive design |
| Step 3 | 30 | 0.8 | $0.9M | Bayesian approach essential |
| Step 4 | 25 | 0.7 | $0.8M | N-of-1 or crossover design needed |

---

## 3. Weak-Spot Targeting

> **Pattern from KubeSRE's AdversarialDesigner:** After mastery, the environment actively targets the agent's weakest capabilities rather than uniformly sampling scenarios.

### Failure Pattern Analysis

The `AdversarialDesigner` maintains a failure log that tracks **why** episodes fail, not just **which** scenarios fail:

```python
class FailureAnalyzer:
    """Tracks failure patterns to inform adaptive difficulty."""

    def __init__(self):
        self.failure_log = []  # (scenario_id, failure_reason, difficulty_params)

    def record_failure(self, scenario_id: str, episode_result: dict):
        """Classify failure reason and log it."""
        reasons = []

        if episode_result["timeout"]:
            reasons.append("timeout")
        if not episode_result["fda_pass"]:
            reasons.append("fda_violation")
        if episode_result["power"] < 0.80:
            reasons.append("underpowered")
        if episode_result["p_value"] >= 0.05:
            reasons.append("no_significance")
        if not episode_result["correct_subgroup"]:
            reasons.append("missed_subgroup")
        if episode_result["budget_remaining"] < 0:
            reasons.append("budget_exhausted")
        if episode_result["phases_skipped"] > 0:
            reasons.append("workflow_violation")
        if episode_result["overconfidence_penalty"] < -0.5:
            reasons.append("overconfidence")

        self.failure_log.append({
            "scenario_id": scenario_id,
            "reasons": reasons,
            "params": episode_result["difficulty_params"],
        })

    def get_weak_spots(self, window: int = 30) -> dict:
        """Return failure reason frequencies over the last N episodes."""
        recent = self.failure_log[-window:]
        reason_counts = {}
        for entry in recent:
            for reason in entry["reasons"]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        total = len(recent) if recent else 1
        return {k: v / total for k, v in sorted(
            reason_counts.items(), key=lambda x: -x[1]
        )}
```

### Targeted Scenario Generation

Based on `get_weak_spots()`, the `AdversarialDesigner` generates scenario parameters that amplify the agent's weaknesses:

| Weak Spot | Targeted Hardening |
|-----------|-------------------|
| `timeout` | Reduce `max_steps` by 10%, increase scenario complexity |
| `fda_violation` | Add more prerequisite traps (actions that look valid but violate obscure FDA rules) |
| `underpowered` | Reduce effect size, force larger N requirement |
| `no_significance` | Increase noise, reduce true effect, add confounders |
| `missed_subgroup` | Make subgroup more hidden (lower prevalence, misleading overall signal) |
| `budget_exhausted` | Tighter budget, higher per-patient cost |
| `workflow_violation` | No change to scenario — this is a policy issue, not env difficulty |
| `overconfidence` | Scenarios where obvious answer is wrong (e.g., dose-response plateaus early) |

### Weak-Spot Probability Allocation

When selecting the next scenario, the curriculum controller uses a **70/30 split**:

- **70% of episodes:** Target the scenario with the lowest success rate (weak-spot targeting)
- **30% of episodes:** Uniform random across all scenarios (prevents catastrophic forgetting)

```python
def select_scenario_adaptive(
    scenario_results: dict,
    failure_analyzer: FailureAnalyzer,
    tier: int,
    rng: np.random.Generator,
) -> tuple[str, dict]:
    """Select scenario and parameters with adaptive difficulty."""

    # Pick scenario (70/30 weak-spot targeting)
    if rng.random() < 0.70:
        scenario_id = _weakest_scenario(scenario_results)
    else:
        scenario_id = rng.choice(list(scenario_results.keys()))

    # Get base parameters for this scenario + tier
    base_params = get_base_params(scenario_id, tier)

    # Apply hardening based on mastery level
    mastery_level = get_mastery_level(scenario_id, scenario_results)
    hardened_params = apply_hardening(base_params, mastery_level)

    # Apply targeted modifications based on weak spots
    weak_spots = failure_analyzer.get_weak_spots()
    targeted_params = apply_weak_spot_targeting(hardened_params, weak_spots)

    # Validate solvability
    assert is_solvable(scenario_id, targeted_params), \
        f"Generated params for {scenario_id} are not solvable!"

    return scenario_id, targeted_params
```

---

## 4. Compound Challenges (Expert Tier)

At Expert tier (difficulty > 0.80), the `AdversarialDesigner` generates **compound challenges** that combine multiple difficulty axes simultaneously:

### Compound Challenge Types

| Challenge Name | Axes Combined | Description |
|---------------|--------------|-------------|
| `needle_in_haystack` | small effect + hidden subgroup + high noise | Tiny effect only visible in a rare, hard-to-identify subgroup |
| `budget_crunch` | tight budget + high per-patient cost + multiple required analyses | Must complete trial with minimal resources |
| `time_bomb` | short timeline + high dropout + slow enrollment | Trial must complete before patients leave |
| `misleading_signal` | contradictory Phase I + dose-response anomaly + placebo noise | Phase I data actively misleads about optimal parameters |
| `regulatory_maze` | strict FDA rules + compound endpoints + amendment limits | Must navigate complex regulatory requirements |
| `everything_hard` | all axes at step_3+ | Final boss — only solvable with near-optimal policy |

### Compound Generation Logic

```python
COMPOUND_CHALLENGES = {
    "needle_in_haystack": {
        "effect_size_multiplier": 0.5,
        "subgroup_prevalence_mult": 0.5,
        "noise_multiplier": 1.8,
        "required_capability": "subgroup_identification",
    },
    "budget_crunch": {
        "budget_multiplier": 0.5,
        "cost_per_patient_mult": 1.5,
        "max_amendments": 1,
        "required_capability": "resource_efficiency",
    },
    "time_bomb": {
        "time_multiplier": 0.6,
        "dropout_rate_add": 0.20,
        "enrollment_rate_mult": 0.7,
        "required_capability": "adaptive_design",
    },
    "misleading_signal": {
        "misleading_phase_i": True,
        "dose_response_anomaly": True,
        "placebo_boost": 0.15,
        "required_capability": "robustness",
    },
    "regulatory_maze": {
        "strict_fda": True,
        "compound_endpoints": True,
        "max_amendments": 0,
        "required_capability": "protocol_design",
    },
}
```

---

## 5. Solvability Guarantee

> **Pattern from KubeSRE:** Every generated incident is validated as having a valid resolution path before being served to the agent.

Every hardened/compound scenario must pass a **solvability check** before the environment serves it:

### Solvability Criteria

```python
def is_solvable(scenario_id: str, params: dict) -> bool:
    """Verify that a scenario with given params can be solved within step budget."""

    # 1. Statistical power is achievable
    #    Given the effect size and max affordable sample size, power >= 0.80 must be possible
    max_n = params["budget"] / params["cost_per_patient"]
    min_power = calculate_power(
        effect_size=params["true_effect_size"],
        n=int(max_n),
        alpha=0.05
    )
    if min_power < 0.80:
        return False

    # 2. Time budget allows minimum workflow
    #    At least: 3 Phase I steps + 4 Phase II design + 1 regulatory + 1 analysis + 1 conclusion = 10 steps
    min_steps = 10
    if params.get("max_steps", 100) < min_steps:
        return False

    # 3. Budget covers minimum viable trial
    #    At least 20 patients for Phase I + smallest valid Phase II
    min_patients = 20 + calculate_min_n(params["true_effect_size"], alpha=0.05, power=0.80)
    if max_n < min_patients:
        return False

    # 4. Subgroup is detectable if present
    #    If subgroup enrichment is needed, the prevalence must allow detection
    if params.get("subgroup_prevalence_mult", 1.0) < 0.3:
        # Subgroup prevalence < 10% of base — may be undetectable
        return False

    return True
```

### Fallback

If a generated scenario fails solvability, the `AdversarialDesigner` relaxes the hardest axis by one step until solvable. Maximum 3 relaxation attempts before falling back to the previous hardening level.

---

## 6. Integration Points

### With Curriculum Controller (`curriculum_policy.md`)

- Mastery detection triggers parameter hardening (this spec)
- Tier advancement is separate — it requires mastery across **all** scenarios
- Hardening resets when the agent advances to a new tier (new tier brings its own base difficulty)

### With NoiseModel (`ARCHITECTURE.md`)

- NoiseModel applies the hardened parameters via its `apply_domain_randomization()` method
- Hardening multipliers are composed with the NoiseModel's per-tier noise scaling
- The NoiseModel seed ensures reproducibility even with adaptive parameters

### With Reward Computer (`reward_spec.md`)

- Harder scenarios **do not change** the reward structure — same components, same weights
- But harder scenarios naturally produce lower raw rewards (smaller effect → harder to detect → lower success rate)
- The potential-based shaping φ(s) adapts automatically because milestones are harder to achieve

### With Training Runbook (`training_runbook.md`)

- Adaptive difficulty is logged in `curriculum_log.csv`: `mastery_level`, `hardening_step`, `compound_challenge`
- Dashboard shows hardening progression as part of curriculum panel
- Troubleshooting: if reward plateaus despite tier advancement, check if all scenarios are stuck at same hardening level

### With Dashboard (`dashboard.html`)

- Dashboard Panel 3 (Curriculum Progression) shows hardening steps as sub-tier markers
- Dashboard Panel 4 (Scenario Breakdown) shows per-scenario success rates with hardening level annotations
- Weak-spot heatmap: visual display of `get_weak_spots()` output

---

## 7. Expected Behavior During Training

### Episode 1–50 (Warmup Tier)

- All scenarios at base difficulty
- Agent learns basic workflow ordering
- First mastery detections around episode 30–40 for easier scenarios
- Hardening step 1 applied to mastered scenarios

### Episode 50–150 (Beginner + Intermediate)

- Some scenarios at hardening step 2–3
- Weak-spot targeting kicks in: agent gets more of its weakest scenario
- Tier advancements at ~episode 60 (beginner) and ~episode 120 (intermediate)
- Hardening resets on tier advancement, then re-accumulates

### Episode 150–300 (Advanced)

- Scenarios at hardening step 2–4
- Weak-spot targeting focused on specific failure reasons (not just scenario types)
- Agent must demonstrate robustness across all difficulty variations

### Episode 300+ (Expert)

- Compound challenges activated
- All axes hardened to near-maximum
- Only agents with genuine understanding of trial design succeed
- This is the "demo range" — episodes from here show the strongest before/after contrast

---

## Summary

| Component | What | Who Implements |
|-----------|------|---------------|
| Mastery detection | Per-scenario sliding window + fast-track | Suyash (CurriculumController) |
| Parameter hardening | 4-step schedule per scenario × axis | Suyash (NoiseModel + AdversarialDesigner) |
| Weak-spot targeting | FailureAnalyzer + 70/30 scenario selection | Suyash (AdversarialDesigner) |
| Compound challenges | 5 named challenge types at Expert tier | Suyash (AdversarialDesigner) |
| Solvability guarantee | Statistical + budgetary check before serving | Suyash (AdversarialDesigner) |
| This spec document | Design, expected behavior, integration points | Roopal |
