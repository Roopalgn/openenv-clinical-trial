# Reward Decomposition Specification

## Design Principles

1. **Decomposed** — every component is independently verifiable and debuggable
2. **Math-verified** — core success is determined by scipy.stats, not LLM
3. **High variance** — GRPO needs clear separation between good (−3) and great (+14) episodes
4. **Shaped** — potential-based shaping gives gradient without changing optimal policy
5. **Phase-aware** — correct workflow ordering is rewarded

---

## Per-Step Reward (8 components)

```
r_step = r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty + r_shaping
```

| Component | What It Measures | Reward | Verification |
|-----------|-----------------|--------|-------------|
| `r_validity` | FDA rule compliance | +0.3 pass, −0.15/violation | Rule engine (binary) |
| `r_ordering` | Correct phase workflow | +0.2 correct, −0.3×N skip | Phase detection heuristic |
| `r_info_gain` | Information from experiments | +0.1 to +0.8 | Bayesian update quality |
| `r_efficiency` | Budget/time efficiency | +0.1 to −0.2 | Math (cost / budget) |
| `r_novelty` | Trying new action types | +0.1 first use | Action history check |
| `r_penalty` | Soft violations (redundant, unjustified) | −0.1 to −0.15 each | Rule engine |
| `r_shaping` | Progress toward milestones | γ·(φ(s') − φ(s)) | Potential function |

**Info gain by action type:**

| Action | Info Gain | Why |
|--------|----------|-----|
| `run_dose_escalation` | +0.1 to +0.5 | KL divergence on effect_size posterior |
| `observe_safety_signal` | +0.05 to +0.3 | Entropy reduction on side_effect_rate |
| `estimate_effect_size` | +0.2 to +0.6 | Posterior std reduction |
| `add_biomarker_stratification` | +0.1 to +0.8 | Subgroup signal strength |
| Design actions (set_*) | 0.0 | Design doesn't generate data |

---

## Terminal Reward (7 components)

Fires once at `done=True` after trial simulation.

| Component | Condition | Reward |
|-----------|----------|--------|
| `r_terminal_success` | Trial detects true effect (p < α) | +5.0 to +7.0 (efficiency-scaled) |
| | Trial fails (p ≥ α) | −1.0 |
| `r_terminal_calibration` | Correct responder population + mechanism + effect estimate | +0.0 to +5.0 |
| `r_terminal_power` | Power ≥ 0.90 / ≥ 0.80 / ≥ 0.60 / < 0.60 | +2.0 / +1.5 / 0.0 / −2.0 |
| `r_terminal_fda` | All rules pass / ≥80% / <80% | +2.0 / +1.0 / −1.0 |
| `r_terminal_budget` | Under budget / over | +1.0 / −0.5 |
| `r_terminal_futility` | Smart early stop / stopped a winner | +1.0 / −1.5 |
| `r_terminal_overconf` | High-confidence wrong claims | −0.5 each (max −2.5) |

**Timeout:** If steps ≥ max_steps without conclusion → wipe step rewards, set R_episode = −2.0.

---

## Total Episode Reward

$$R_{\text{episode}} = \sum_{t=1}^{T} r_{\text{step}_t} + r_{\text{terminal}}$$

| Outcome | Typical Total | Range |
|---------|-------------|-------|
| Expert success (subgroup + FDA + efficient + calibrated) | +11 to +14 | Best |
| Good success (trial succeeds, partial calibration) | +6 to +10 | Common |
| Marginal success (p < 0.05 barely, wrong subgroup) | +2 to +5 | Partial |
| Marginal failure (ran trial, p > 0.05) | −1 to +1 | Near miss |
| Clear failure (wrong design, FDA rejection) | −2 to 0 | Bad |
| Timeout | −2.0 flat | Worst |

---

## Potential-Based Shaping

$$R_{\text{shaped}} = R_{\text{original}} + \gamma \cdot (\varphi(s') - \varphi(s))$$

Where $\varphi(s)$ = milestone_completion_fraction × budget_efficiency, $\gamma$ = 0.99. Terms telescope over a full episode — optimal policy unchanged, learning speed improved.
