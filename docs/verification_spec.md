# Multi-Layer Verification Specification

## Overview

| Layer | Name | When | What It Checks | Verdict |
|-------|------|------|---------------|---------|
| L1 | Programmatic Ground-Truth | Terminal | Agent claims vs `TrialLatentState` | Pass/Fail + score |
| L2 | Rule-Engine Constraints | Per-step + Terminal | FDA rules, prerequisites | Block / Penalty / Pass |
| L3 | LLM Judge (Optional) | Terminal only | Workflow quality, reasoning | 0.0–1.0 score |

**Design principle:** Programmatic (L1) is authoritative. Rule engine (L2) gates actions. LLM judge (L3) adds qualitative color but never overrides L1/L2.

---

## Layer 1: Programmatic Ground-Truth

Fires at episode end. Compares agent's trial design against hidden `TrialLatentState`.

**Checks:**

| Check | Method | Threshold | Reward Key |
|-------|--------|-----------|-----------|
| Statistical power | `calculate_power(effect_size, n, alpha)` | ≥ 0.80 | `r_terminal_power` |
| Type I error | Agent's alpha | ≤ 0.05 | — |
| Effect estimate accuracy | Relative error vs true value | ≤ 30% | `r_terminal_calibration` |
| Responder population | Exact match | — | `r_terminal_calibration` |
| Dose selection | Within ±50mg of optimal | — | — |
| Budget compliance | Estimated cost ≤ initial budget | — | `r_terminal_budget` |

---

## Layer 2: Rule-Engine Constraints

**Per-step (hard blocks):**

| Rule | Condition |
|------|----------|
| Phase II minimum n | sample_size ≥ 100 if in Phase II |
| Primary endpoint prespecified | endpoint set before submission |
| Interim alpha spending | alpha_spent ≤ alpha_budget |
| Randomization required | randomization set for Phase II |
| Safety monitoring | committee set if patients > 0 |
| Informed consent | Always required (auto-checked) |

Hard blocks return an error and don't execute the action. Soft violations apply −0.15 penalty.

---

## Layer 3: LLM Judge (Optional)

Terminal-only qualitative assessment. Scales with curriculum tier:

| Tier | Persona | Behavior |
|------|---------|---------|
| Warmup | Junior | Lenient scoring, gives improvement hints |
| Intermediate | Senior | Standard clinical review |
| Expert | Principal | Harsh on inefficiency, expects near-optimal decisions |

**Not used for core reward.** Adds qualitative feedback and a 0.0–1.0 quality score. The agent cannot game L3 because L1 controls the core terminal reward.

---

## Conflict Resolution

If L1 says "failed" but L3 says "good workflow":
- **L1 wins.** Trial objectively failed. L3 feedback noted but doesn't change reward.

If L2 blocks an action but L3 would approve it:
- **L2 wins.** Prerequisites are hard constraints. Action doesn't execute.
