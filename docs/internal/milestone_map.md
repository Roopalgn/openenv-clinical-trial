# Milestone Map — Long-Horizon Episode Phases

> **Inspired by:** Bio Experiment's 18 milestones for pipeline progression tracking. KubeSRE's phase-aware workflow (triage→investigate→fix→verify) with per-phase step budgets. All winners: milestones drive both the potential-based shaping function φ(s) and curriculum advancement logic.

## Overview

An episode spans **55–100 steps** across 3 macro-phases and **18 milestones**. Each milestone is a binary progress flag in `TrialLatentState` used for:

1. **Potential-based shaping** — φ(s) increases as milestones complete (see `reward_spec.md`)
2. **Prerequisite enforcement** — certain actions require prior milestones (see `phase_workflow.md`)
3. **Curriculum tracking** — milestone velocity indicates agent skill level
4. **Terminal reward** — milestone completion fraction contributes to `r_terminal_calibration`

---

## Milestone Definitions

### Phase I — Safety & Dose-Finding (Steps ~1–25)

| # | Milestone ID | Triggered When | Typical Step | Prerequisite |
|---|-------------|---------------|-------------|-------------|
| M1 | `scenario_reviewed` | Agent takes first action (implicit) | 1 | None |
| M2 | `hypothesis_formed` | Agent sets expected effect estimate or identifies target | 1–3 | None |
| M3 | `dose_escalation_started` | First `run_dose_escalation` action | 2–5 | None |
| M4 | `safety_signal_observed` | First `observe_safety_signal` after dose escalation | 5–10 | M3 |
| M5 | `mtd_identified` | Agent has escalated through ≥3 dose levels AND observed safety | 8–15 | M3 + M4 |
| M6 | `effect_estimated` | `estimate_effect_size` action completes | 10–20 | M3 (at least 1 dose escalation) |
| M7 | `phase_i_complete` | All of M3 + M4 + M5 + M6 achieved | 15–25 | M3 + M4 + M5 + M6 |

### Phase II — Efficacy & Design (Steps ~20–60)

| # | Milestone ID | Triggered When | Typical Step | Prerequisite |
|---|-------------|---------------|-------------|-------------|
| M8 | `primary_endpoint_set` | `set_primary_endpoint` action | 20–30 | None (but better after M6) |
| M9 | `sample_size_set` | `set_sample_size` action | 20–35 | M6 (need effect estimate for power calc) |
| M10 | `inclusion_criteria_set` | `set_inclusion_criteria` action | 20–35 | None |
| M11 | `control_arm_set` | `set_control_arm` action | 25–35 | None |
| M12 | `protocol_submitted` | `submit_to_fda_review` action completes | 30–45 | M8 + M9 |
| M13 | `fda_approved` | FDA review passes all hard constraints | 30–45 | M12 |
| M14 | `interim_complete` | `run_interim_analysis` completes | 40–60 | M13 |
| M15 | `biomarker_stratified` | `add_biomarker_stratification` used (optional but high-value) | 25–50 | M6 |

### Analysis & Conclusion (Steps ~50–100)

| # | Milestone ID | Triggered When | Typical Step | Prerequisite |
|---|-------------|---------------|-------------|-------------|
| M16 | `primary_analysis_complete` | `run_primary_analysis` completes | 50–80 | M13 |
| M17 | `conclusion_written` | `synthesize_conclusion` completes | 55–90 | M16 |
| M18 | `trial_complete` | Episode done signal fires | 55–100 | M17 |

---

## Milestone Flow Diagram

```
Phase I (Safety)                Phase II (Efficacy)               Analysis
────────────────               ───────────────────               ─────────

M1 scenario_reviewed           M8  primary_endpoint_set          M16 primary_analysis_complete
  │                              │                                  │
  ├─► M2 hypothesis_formed     M9  sample_size_set ◄── M6         M17 conclusion_written
  │                              │                                  │
  ├─► M3 dose_escalation       M10 inclusion_criteria_set         M18 trial_complete
  │     │                        │
  │     ├─► M4 safety_signal   M11 control_arm_set
  │     │     │                  │
  │     │     ├─► M5 mtd_id    M12 protocol_submitted ◄── M8+M9
  │     │                        │
  │     ├─► M6 effect_est      M13 fda_approved ◄── M12
  │                              │
  └─► M7 phase_i_complete      M14 interim_complete ◄── M13
       (= M3+M4+M5+M6)           │
                                M15 biomarker_stratified ◄── M6
                                    (optional, high-value)
```

---

## Step Budget by Phase

> *Pattern from KubeSRE: episode length scales with difficulty. Bio Experiment: up to 30 steps.*

| Phase | Min Steps | Typical Steps | Max Steps | Notes |
|-------|-----------|--------------|-----------|-------|
| Phase I | 10 | 15–25 | 30 | 6 dose cohorts × ~3 actions each + analysis |
| Phase II Design | 5 | 8–15 | 20 | 6–8 design parameters + FDA submission |
| Phase II Execution | 5 | 10–20 | 30 | Interim analysis, amendments, monitoring |
| Analysis | 3 | 5–10 | 15 | Primary analysis + conclusion |
| **Total** | **23** | **55–70** | **100** | |

### Max Steps by Curriculum Tier

| Tier | Max Steps | Rationale |
|------|-----------|-----------|
| Warmup | 100 | Generous — learning the workflow |
| Beginner | 90 | Slightly tighter |
| Intermediate | 80 | Must be more efficient |
| Advanced | 70 | Principal judge penalizes slow episodes |
| Expert | 60 | Efficiency is part of mastery |

---

## Milestone Velocity Metrics

Track how quickly the agent achieves milestones for curriculum evaluation:

```python
MILESTONE_VELOCITY_TARGETS = {
    # milestone_id: max_steps_to_achieve (by tier)
    "phase_i_complete": {
        "warmup": 30, "beginner": 25, "intermediate": 20, "advanced": 18, "expert": 15
    },
    "protocol_submitted": {
        "warmup": 50, "beginner": 45, "intermediate": 40, "advanced": 35, "expert": 30
    },
    "trial_complete": {
        "warmup": 100, "beginner": 90, "intermediate": 80, "advanced": 70, "expert": 60
    },
}
```

If agent completes milestones faster than the target for their tier, it contributes to fast-track curriculum advancement (KubeSRE: 90%+ success rate → skip min_episodes requirement).

---

## Integration with φ(s) Shaping Function

The milestone_fraction component of the potential function φ(s) uses these milestones:

```python
def milestone_fraction(state):
    """Fraction of 18 milestones completed. Drives the largest component of φ(s)."""
    completed = sum([
        state.scenario_reviewed,
        state.hypothesis_formed,
        state.dose_escalation_started,
        state.safety_signal_observed,
        state.mtd_identified,
        state.effect_estimated,
        state.phase_i_complete,
        state.primary_endpoint_set,
        state.sample_size_set,
        state.inclusion_criteria_set,
        state.control_arm_set,
        state.protocol_submitted,
        state.fda_approved,
        state.interim_complete,
        state.biomarker_stratified,
        state.primary_analysis_complete,
        state.conclusion_written,
        state.trial_complete,
    ])
    return completed / 18.0
```

This creates a **smooth gradient** through the episode — each milestone completion nudges φ(s) up by ~0.056, generating a small positive shaping reward via γ·(φ(s')−φ(s)). The agent learns that making progress is always better than stalling.

---

## Milestone Reset Behavior

On `env.reset()`:
- All milestones set to `False`
- M1 (`scenario_reviewed`) set to `True` immediately (agent sees the scenario)
- Step counter set to 0
- Budget and time set to scenario base values × NoiseModel multiplier

On timeout (step_count ≥ max_steps):
- Episode terminates with `done=True`
- Milestones freeze at current state
- Terminal reward computation uses milestone_fraction at time of timeout
- Total reward overridden to -2.0 (timeout penalty, from KubeSRE pattern)
