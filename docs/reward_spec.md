# Reward Decomposition Specification

## Design Principles

1. **Decomposed** — every component is independently verifiable and debuggable
2. **Math-verified** — core success is determined by scipy.stats, not LLM
3. **High variance** — GRPO needs clear separation between good (−3) and great (+15) episodes
4. **Milestone-driven** — first-time milestone bonuses provide progressive learning signal
5. **Phase-aware** — correct workflow ordering is rewarded
6. **Progress-proportional** — episodes reaching more milestones reliably score higher

---

## Per-Step Reward (6 components)

```
r_step = r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty
```

| Component | What It Measures | Reward | Verification |
|-----------|-----------------|--------|-------------|
| `r_validity` | FDA rule compliance | +0.05 first-time valid, 0.0 repeat valid, −2.0 invalid | Rule engine (binary) |
| `r_ordering` | Correct phase workflow | +0.1 correct, −0.3×N skip | Phase detection heuristic |
| `r_info_gain` | Information gain + milestone bonuses | +0.1 to +2.5 | Power × base + first-time milestone |
| `r_efficiency` | Budget efficiency (terminal only) | 0.0 to +0.3 | Math (remaining / initial budget) |
| `r_novelty` | Trying new action types | +0.1 first use | Action history check |
| `r_penalty` | Compliance violations | −0.5 per violation + episode-wide at terminal | Rule engine |

**Info gain by action type:**

| Action | Info Gain | Why |
|--------|----------|-----|
| `estimate_effect_size` | base × max(power, 0.1) | Power proportional to sample quality |
| `observe_safety_signal` | base × max(power, 0.1) | Safety data generation |
| `run_interim_analysis` | base × max(power, 0.1) | Mid-trial statistical check |
| `run_dose_escalation` | base × max(power, 0.1) | Dose-finding data |
| `add_biomarker_stratification` | base × max(power, 0.1) | Subgroup signal |
| Design actions (set_*) | 0.0 | Design doesn't generate data |

**Milestone bonuses (fire once per episode, added to r_info_gain):**

| Milestone | Bonus | Trigger |
|-----------|-------|---------|
| Phase I complete | +1.5 | First `run_dose_escalation` with phase_i_complete |
| Effect estimated | +1.0 | First `estimate_effect_size` with effect_estimated |
| Interim complete | +1.5 | First `run_interim_analysis` with interim_complete |
| Protocol submitted | +1.0 | First `submit_to_fda_review` with protocol_submitted |
| Primary analysis complete | +1.5 | First `run_primary_analysis` with primary_analysis_complete |
| Trial complete | +2.5 | First `synthesize_conclusion` with trial_complete |
| Patients enrolled | +0.5 | First `enroll_patients` with patients > 0 |

**Terminal progress bonus (added to r_info_gain at episode end):**

Fires at terminal (trial_complete or timeout), proportional to milestones reached:

```
progress_bonus = 3.0 × (milestones_completed / 7)
```

Where milestones are: phase_i_complete, effect_estimated, interim_complete, protocol_submitted, primary_analysis_complete, trial_complete, patients_enrolled > 0.

This creates a smooth gradient: 1 milestone → +0.43, 4 milestones → +1.71, 7 milestones → +3.0.

---

## Terminal Reward (2 components)

Fires when `trial_complete=True` after `synthesize_conclusion`. (`run_primary_analysis` alone sets `primary_analysis_complete` and unlocks `synthesize_conclusion` but no longer ends the episode — the agent must produce a conclusion.)

| Component | Condition | Reward |
|-----------|----------|--------|
| `r_terminal_success` | p < α, no failure, power ≥ 0.80 | +4.0 |
| | p < α, no failure, 0.40 ≤ power < 0.80 | linear ramp 0 → +4.0 |
| | Trial completes but fails (or power < 0.40) | −1.0 |
| `r_terminal_calibration` | CI accuracy vs true effect size | 0.0 to +2.0 |

The power-gated ramp prevents the agent from being rewarded for statistically unsound trials that hit p < 0.05 by chance on small n.

**Episode-wide violation penalty (at terminal):** −0.3 per cumulative FDA violation across the entire episode.  This prevents the "clean last step" exploit where an agent with 10 violations gets full terminal reward because only the last step was checked.

**Timeout:** If steps ≥ max_steps without completion, earned components are preserved and an additional timeout penalty is applied (`r_validity -= 0.5`, `r_penalty -= 1.5`).

---

## Total Episode Reward

$$R_{\text{episode}} = \sum_{t=1}^{T} r_{\text{step}_t}$$

(Terminal components are included in the step where trial_complete triggers.)

| Outcome | Typical Total | Range |
|---------|-------------|-------|
| Optimal design (high power + efficient + calibrated) | +12 to +18 | Best |
| Good design (trial succeeds, partial calibration) | +6 to +12 | Common |
| Partial plan (some milestones, no completion) | +1 to +5 | Near miss |
| Few valid actions only | −1 to +1 | Weak |
| Parse failure / invalid sequence | −3 | Worst |

---

## Potential-Based Shaping

$$R_{\text{shaped}} = R_{\text{original}} + \gamma \cdot (\varphi(s') - \varphi(s))$$

Where $\varphi(s)$ = milestone_completion_fraction × budget_efficiency, $\gamma$ = 0.99. Terms telescope over a full episode — optimal policy unchanged, learning speed improved. Shaping bonus is folded into `r_info_gain`.
