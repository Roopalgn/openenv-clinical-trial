# Adaptive Difficulty Specification

## Purpose

After the agent masters a scenario (>70% success), the environment hardens parameters — tighter budgets, rarer subgroups, noisier data — and targets the agent's tracked weak spots.

---

## Mastery Detection

Tracked **per scenario × per tier**. A strong `solid_tumor_chemo` doesn't mask weakness on `rare_disease_orphan`.

| Tier | Window | Threshold | Min Episodes |
|------|--------|-----------|-------------|
| Warmup | 10 | 70% | 15 |
| Beginner | 12 | 65% | 20 |
| Intermediate | 15 | 55% | 25 |
| Advanced | 20 | 45% | 30 |
| Expert | N/A | N/A | N/A |

**Fast-track:** ≥90% in first 5 episodes → immediate hardening (skip mastery window).

---

## Parameter Hardening

Applied **incrementally** when mastery is detected, within the same tier:

| Axis | Easy | Step 1 | Step 2 | Step 3 |
|------|------|--------|--------|--------|
| Effect size multiplier | 1.0× | 0.85× | 0.70× | 0.60× |
| Budget multiplier | 1.0× | 0.90× | 0.80× | 0.70× |
| Noise multiplier | 1.0× | 1.2× | 1.4× | 1.6× |
| Dropout add | +0% | — | +5% | +8% |
| Placebo boost | +0% | — | — | +5% |
| Subgroup prevalence | 1.0× | — | — | 0.7× |
| Misleading Phase I | No | — | — | Yes |

---

## Weak-Spot Targeting

The adversarial designer (`adversarial_designer.py`) analyzes failure patterns:

| Failure Type | Counter | Hardening Response |
|-------------|---------|-------------------|
| Small effect missed | `small_effect_failures` | Reduce effect size further |
| High dropout | `high_dropout_failures` | Increase dropout rate |
| Subgroup missed | `missed_subgroup_failures` | Reduce subgroup prevalence |
| Budget exhausted | `budget_failures` | Tighten budget |

At Expert tier, the adversarial designer creates **compound challenges** combining 2–3 difficulty axes simultaneously.

---

## Solvability Guarantee

Every generated scenario must remain solvable — at least one action sequence achieves success:
- Effect size > 0 (drug works)
- Budget sufficient for n patients needed for 80% power
- Time sufficient for enrollment + follow-up
- At least one valid inclusion criteria identifies responders

If generated params fail solvability check → regenerate with relaxed constraints.
