# Reward Decomposition Specification

## Design Principles

1. **Decomposed** — every component is independently verifiable and debuggable
2. **Math-verified** — core success is determined by scipy.stats, not LLM
3. **High variance** — GRPO needs clear separation between good (−3) and great (+15) episodes
4. **Milestone-driven** — first-time milestone bonuses provide progressive learning signal
5. **Phase-aware** — correct workflow ordering is rewarded

---

## Per-Step Reward (6 components)

```
r_step = r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty
```

| Component | What It Measures | Reward | Verification |
|-----------|-----------------|--------|-------------|
| `r_validity` | FDA rule compliance | +0.05 pass, −2.0 invalid | Rule engine (binary) |
| `r_ordering` | Correct phase workflow | +0.1 correct, −0.3×N skip | Phase detection heuristic |
| `r_info_gain` | Information gain + milestone bonuses | +0.1 to +1.5 | Power × base + first-time milestone |
| `r_efficiency` | Budget efficiency (terminal only) | 0.0 to +0.3 | Math (remaining / initial budget) |
| `r_novelty` | Trying new action types | +0.1 first use | Action history check |
| `r_penalty` | Compliance violations | −0.5 per violation | Rule engine |

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
| Phase I complete | +1.0 | First `run_dose_escalation` with phase_i_complete |
| Effect estimated | +0.5 | First `estimate_effect_size` with effect_estimated |
| Interim complete | +1.0 | First `run_interim_analysis` with interim_complete |
| Protocol submitted | +0.5 | First `submit_to_fda_review` with protocol_submitted |
| Trial complete | +1.5 | First `run_primary_analysis` with trial_complete |
| Patients enrolled | +0.3 | First `enroll_patients` with patients > 0 |

---

## Terminal Reward (2 components)

Fires when `trial_complete=True` after `run_primary_analysis`.

| Component | Condition | Reward |
|-----------|----------|--------|
| `r_terminal_success` | Trial succeeds (p < α, no failure) | +4.0 |
| | Trial completes but fails | −1.0 |
| `r_terminal_calibration` | CI accuracy vs true effect size | 0.0 to +2.0 |

**Timeout:** If steps ≥ max_steps without completion, earned components are preserved and an additional timeout penalty is applied (`r_validity -= 0.5`, `r_penalty -= 1.5`).

---

## Total Episode Reward

$$R_{\text{episode}} = \sum_{t=1}^{T} r_{\text{step}_t}$$

(Terminal components are included in the step where trial_complete triggers.)

| Outcome | Typical Total | Range |
|---------|-------------|-------|
| Optimal design (high power + efficient + calibrated) | +10 to +15 | Best |
| Good design (trial succeeds, partial calibration) | +5 to +10 | Common |
| Failed trial (p ≥ 0.05 or budget/time exceeded) | −1 to +3 | Near miss |
| Parse failure / invalid sequence | −3 | Worst |

---

## Potential-Based Shaping

$$R_{\text{shaped}} = R_{\text{original}} + \gamma \cdot (\varphi(s') - \varphi(s))$$

Where $\varphi(s)$ = milestone_completion_fraction × budget_efficiency, $\gamma$ = 0.99. Terms telescope over a full episode — optimal policy unchanged, learning speed improved. Shaping bonus is folded into `r_info_gain`.
