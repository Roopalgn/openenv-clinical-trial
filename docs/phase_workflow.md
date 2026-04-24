# Clinical Trial Phase-Aware Workflow & Scoring

## 10-Phase Clinical Workflow

The agent learns this ordering through reward signal, not hard-coding.

| # | Phase | Description | Actions |
|---|-------|-------------|---------|
| 0 | `literature_review` | Understand disease and constraints | Review scenario |
| 1 | `hypothesis` | Form hypothesis about drug mechanism | Estimate expected effect |
| 2 | `phase_i_design` | Phase I safety/dose-finding | run_dose_escalation, observe_safety_signal |
| 3 | `phase_i_analysis` | Analyze Phase I results | estimate_effect_size |
| 4 | `phase_ii_design` | Design Phase II efficacy trial | set_primary_endpoint, set_sample_size, set_inclusion_criteria, etc. |
| 5 | `regulatory` | FDA review | submit_to_fda_review, request_protocol_amendment |
| 6 | `enrollment` | Enroll patients | (implicit after FDA approval) |
| 7 | `monitoring` | Interim analysis, adaptation | run_interim_analysis, modify_sample_size, add_biomarker_stratification |
| 8 | `analysis` | Final statistical test | run_primary_analysis |
| 9 | `conclusion` | Synthesize results | synthesize_conclusion |

---

## Phase-Order Scoring

| Condition | Reward |
|-----------|--------|
| Action in correct or next phase | +0.2 |
| Action stays in current phase | +0.2 |
| Action skips N phases ahead | −0.3 × N |

**Judge persona scaling by tier:**

| Tier | Persona | Forward Bonus | Skip Penalty | Extras |
|------|---------|-------------|-------------|--------|
| Warmup | Junior | +0.20 | −0.30/skip | Allows 1 skip free, gives hints |
| Beginner | Junior→Senior | +0.20 | −0.30/skip | Standard |
| Intermediate | Senior | +0.15 | −0.30/skip | Expects correct ordering |
| Advanced | Senior→Principal | +0.10 | −0.50/skip | Redundancy penalty −0.10 |
| Expert | Principal | +0.05 | −0.50/skip | Redundancy −0.15, efficiency penalty |

---

## Hard Prerequisites

These block the action entirely (not a reward signal — returns error):

| Action | Requires |
|--------|----------|
| `estimate_effect_size` | ≥1 `run_dose_escalation` |
| `set_sample_size` | `estimate_effect_size` |
| `submit_to_fda_review` | `set_primary_endpoint` + `set_sample_size` |
| `run_interim_analysis` | `submit_to_fda_review` passed |
| `run_primary_analysis` | `submit_to_fda_review` passed |
| `synthesize_conclusion` | `run_primary_analysis` |
| `modify_sample_size` | `run_interim_analysis` |
| `add_biomarker_stratification` | `estimate_effect_size` |

---

## Protocol Amendment

- `request_protocol_amendment` allows recovery from FDA review failure
- Costs time and budget (realistic consequence)
- Successful recovery: +0.3 recovery bonus
- Maximum 2 amendments per episode
