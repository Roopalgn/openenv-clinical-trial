# Curriculum Progression Policy

## Design Principles

1. **Automatic advancement** — triggered by performance windows, not manually
2. **Per-scenario tracking** — tracked per scenario type, not globally
3. **Sliding window** — success rate over last W episodes (prevents history dilution)
4. **No demotion** — once advanced, stays at new tier
5. **Weak-spot targeting** — 70% of episodes target the agent's weakest scenario

---

## Tier Definitions

| Tier | Difficulty | Max Steps | Judge | Noise | Advancement |
|------|-----------|-----------|-------|-------|-------------|
| 0 — Warmup | 0.00–0.25 | 100 | Junior (hints) | ±10% | 60% over W=10 |
| 1 — Beginner | 0.25–0.40 | 90 | Junior→Senior | ±20% | 55% over W=15 |
| 2 — Intermediate | 0.40–0.60 | 80 | Senior | ±30% | 45% over W=20 |
| 3 — Advanced | 0.60–0.80 | 70 | Senior→Principal | ±40% | 35% over W=25 |
| 4 — Expert | 0.80–0.95 | 60 | Principal | ±50% | Terminal |

**Decreasing thresholds** because harder tiers have genuinely harder scenarios. 35% at Advanced = reliably designing good trials under adversarial conditions.

---

## Advancement Logic

Agent advances when ALL scenarios meet threshold over the sliding window:

1. Check `episodes_in_tier >= min_episodes` (prevents premature advancement)
2. For each scenario: compute success rate over last W episodes
3. If ALL scenarios ≥ threshold → advance

**Fast-track:** ≥90% success across all scenarios → skip `min_episodes` gate. Saves 30–50 episodes of wasted compute.

---

## Scenario Selection

- **70% weak-spot:** Selects scenario with lowest success rate (focuses compute on failures)
- **30% random:** Prevents catastrophic forgetting of mastered scenarios

---

## Noise Scaling by Tier

| Parameter | Warmup | Beginner | Intermediate | Advanced | Expert |
|-----------|--------|----------|-------------|----------|--------|
| Budget multiplier | 1.2× | 1.0× | 0.85× | 0.70× | 0.55× |
| Effect size multiplier | 1.5× | 1.0× | 0.80× | 0.60× | 0.40× |
| Dropout multiplier | 0.5× | 1.0× | 1.3× | 1.6× | 2.0× |
| Noise sigma | 0.05 | 0.08 | 0.12 | 0.18 | 0.25 |
| Placebo boost | +0% | +0% | +5% | +10% | +15% |
| Subgroup prevalence | 1.5× | 1.0× | 0.80× | 0.60× | 0.40× |

---

## What Changes at Each Tier

| Tier | Agent Must Learn |
|------|-----------------|
| Warmup | Basic workflow: dose escalation → design → submit → analyze |
| Beginner | Accurate effect estimation and appropriate sample sizing |
| Intermediate | Biomarker stratification and efficient design |
| Advanced | Handle ambiguity, adaptive designs, tough futility calls |
| Expert | Near-optimal everything: precise Phase I, early futility, minimal waste |
