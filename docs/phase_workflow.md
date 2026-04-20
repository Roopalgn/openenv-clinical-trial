# Clinical Trial Phase-Aware Workflow & Scoring

> **Inspired by:** KubeSRE's triage → investigate → fix → verify phase-order bonus (+0.2) and skip penalty. Bio Experiment's prerequisite rules as hard constraints. KubeSRE's judge persona scaling (junior/senior/principal) with curriculum tier.

## Clinical Trial Workflow Phases

A clinical trial follows a strict sequential workflow. The agent should learn this ordering through reward signal, not hard-coding. Phase-order bonuses and skip penalties teach the agent to follow the correct diagnostic sequence — the same pattern that won 1st place (KubeSRE's triage → investigate → fix → verify).

Our workflow maps KubeSRE's 5-phase SRE pattern to clinical trial design:

| KubeSRE Phase | Our Phase | Why it maps |
|---|---|---|
| Triage | literature_review + hypothesis | Understand the problem before acting |
| Investigation | phase_i_design + phase_i_analysis | Gather data to inform fix |
| Mitigation | phase_ii_design + regulatory | Implement the solution design |
| Fix | enrollment + monitoring | Execute and monitor |
| Verification | analysis + conclusion | Verify the outcome |

### Phase Definitions

| # | Phase | Description | Typical Actions |
|---|---|---|---|
| 0 | `literature_review` | Understand the disease, existing treatments, and target population | Review scenario description, note constraints |
| 1 | `hypothesis` | Form a hypothesis about the drug's mechanism and target population | Estimate expected effect size, identify potential responders |
| 2 | `phase_i_design` | Design Phase I safety/dose-finding study | run_dose_escalation, observe_safety_signal |
| 3 | `phase_i_analysis` | Analyze Phase I results | estimate_effect_size, identify MTD |
| 4 | `phase_ii_design` | Design Phase II efficacy trial based on Phase I findings | set_primary_endpoint, set_sample_size, set_inclusion_criteria, set_exclusion_criteria, set_dosing_schedule, set_control_arm, set_randomization_ratio, set_blinding |
| 5 | `regulatory` | Submit protocol for FDA review | submit_to_fda_review, request_protocol_amendment |
| 6 | `enrollment` | Enroll patients, begin trial execution | (implicit — happens after FDA approval) |
| 7 | `monitoring` | Run interim analyses, monitor safety | run_interim_analysis, modify_sample_size, add_biomarker_stratification |
| 8 | `analysis` | Run final statistical analysis | run_primary_analysis |
| 9 | `conclusion` | Synthesize results and conclusions | synthesize_conclusion |

### Phase Detection Heuristic

```python
def _detect_phase(action: TrialAction, history: list) -> str:
    """Classify action into clinical workflow phase."""
    action_type = action.action_type

    # Phase I
    if action_type in ("run_dose_escalation", "observe_safety_signal"):
        return "phase_i_design"
    if action_type == "estimate_effect_size":
        return "phase_i_analysis"

    # Phase II design
    if action_type in ("set_primary_endpoint", "set_sample_size",
                        "set_inclusion_criteria", "set_exclusion_criteria",
                        "set_dosing_schedule", "set_control_arm",
                        "set_randomization_ratio", "set_blinding"):
        return "phase_ii_design"

    # Regulatory
    if action_type in ("submit_to_fda_review", "request_protocol_amendment"):
        return "regulatory"

    # Monitoring
    if action_type in ("run_interim_analysis", "modify_sample_size",
                        "add_biomarker_stratification"):
        return "monitoring"

    # Analysis
    if action_type == "run_primary_analysis":
        return "analysis"

    # Conclusion
    if action_type == "synthesize_conclusion":
        return "conclusion"

    return "literature_review"
```

### Phase Order Map

```python
PHASE_ORDER = {
    "literature_review": 0,
    "hypothesis": 1,
    "phase_i_design": 2,
    "phase_i_analysis": 3,
    "phase_ii_design": 4,
    "regulatory": 5,
    "enrollment": 6,
    "monitoring": 7,
    "analysis": 8,
    "conclusion": 9,
}
```

## Phase-Order Scoring Rules

### Bonus: Correct Phase Ordering (+0.2)

If the current action's phase is equal to or one step ahead of the highest phase seen so far, the agent gets +0.2 bonus. This rewards following the natural clinical trial workflow.

```python
def _is_phase_order_correct(current_phase: str, history: list) -> bool:
    current_order = PHASE_ORDER[current_phase]
    if not history:
        return current_order <= 2  # literature_review, hypothesis, or phase_i_design are valid starts
    past_phases = [_detect_phase(h["action"], history[:i]) for i, h in enumerate(history)]
    max_past_order = max(PHASE_ORDER[p] for p in past_phases)
    return current_order <= max_past_order + 1
```

### Penalty: Skipping Phases (-0.3)

If the agent jumps ahead by 2+ phases (e.g., going from Phase I directly to analysis without Phase II design), it gets -0.3 penalty per skipped phase. This penalizes shortcutting the clinical workflow.

```python
def _get_skipped_phases(current_phase: str, history: list) -> list[str]:
    current_order = PHASE_ORDER[current_phase]
    if current_order <= 2:
        return []
    past_phases = {_detect_phase(h["action"], history[:i]) for i, h in enumerate(history)}
    past_phases.add(current_phase)
    skipped = []
    for phase, order in PHASE_ORDER.items():
        if order < current_order and phase not in past_phases:
            skipped.append(phase)
    return skipped
```

### Examples

**Good workflow** (total phase bonus: +2.0):
```
Step 1: run_dose_escalation       → phase_i_design  (correct start, +0.2)
Step 2: observe_safety_signal     → phase_i_design  (same phase, +0.2)
Step 3: estimate_effect_size      → phase_i_analysis (+0.2)
Step 4: set_primary_endpoint      → phase_ii_design (+0.2)
Step 5: set_sample_size           → phase_ii_design (same phase, +0.2)
Step 6: set_inclusion_criteria    → phase_ii_design (same phase, +0.2)
Step 7: submit_to_fda_review      → regulatory      (+0.2)
Step 8: run_interim_analysis      → monitoring      (+0.2)
Step 9: run_primary_analysis      → analysis        (+0.2)
Step 10: synthesize_conclusion    → conclusion      (+0.2)
```

**Bad workflow** (total phase penalty: -0.9):
```
Step 1: set_sample_size           → phase_ii_design (skipped phase_i_design, phase_i_analysis → -0.6)
Step 2: run_primary_analysis      → analysis        (skipped regulatory, monitoring → -0.6)
Step 3: synthesize_conclusion     → conclusion      (correct ordering from analysis, +0.2)
Net: -0.6 + -0.6 + 0.2 = -1.0
```

## Prerequisite Hard Constraints

In addition to soft phase-order scoring, some actions have hard prerequisites enforced by the rule engine. These are not reward signals — they block the action entirely and return an error:

| Action | Prerequisite | Rationale |
|---|---|---|
| `estimate_effect_size` | At least 1 `run_dose_escalation` completed | Can't estimate without Phase I data |
| `set_sample_size` | `estimate_effect_size` completed | Need effect size to calculate power |
| `submit_to_fda_review` | `set_primary_endpoint` + `set_sample_size` completed | FDA requires these minimum protocol elements |
| `run_interim_analysis` | `submit_to_fda_review` passed | Can't do interim before FDA approval |
| `run_primary_analysis` | `submit_to_fda_review` passed | Can't analyze before approved protocol |
| `synthesize_conclusion` | `run_primary_analysis` completed | Can't conclude without analysis |
| `modify_sample_size` | `run_interim_analysis` completed | Adaptation requires interim look |
| `add_biomarker_stratification` | `estimate_effect_size` completed | Need data to stratify on |

## Integration with Reward

The phase-order bonus/penalty is one component of the per-step reward:

```
r_step = (
    w_validity  × r_validity     +       # FDA rule compliance
    w_ordering  × r_ordering     +       # Phase-order bonus/penalty (THIS DOC)
    w_info_gain × r_info_gain    +       # Information gained
    w_efficiency × r_efficiency  +       # Budget/time efficiency
    w_novelty   × r_novelty      +       # Action diversity bonus
    w_penalty   × r_penalty      +       # Soft constraint violations
    γ × (φ(s') − φ(s))                  # Potential-based shaping
)
```

Where `r_ordering` = +0.2 if correct order, -0.3 × len(skipped_phases) if phases were skipped.

## Judge Persona Scaling by Difficulty

> **Inspired by:** KubeSRE's judge persona system: junior (lenient, gives hints) at low difficulty, senior (standard) at medium, principal (strict, penalizes inefficiency) at expert.

The phase-order evaluation strictness scales with curriculum tier:

| Tier | Judge Persona | Phase-Order Behavior |
|---|---|---|
| Warmup (0.0–0.25) | **Junior** | Lenient: allows 1 phase skip without penalty. Includes hint in observation (e.g., "Consider running dose escalation before setting sample size"). Phase bonus still +0.2. |
| Beginner (0.25–0.40) | **Junior→Senior** | Standard: phase skip penalty -0.3/skip. No hints. Phase bonus +0.2. |
| Intermediate (0.40–0.60) | **Senior** | Strict: phase skip penalty -0.3/skip. Phase bonus reduced to +0.15 (expects correct ordering as baseline). |
| Advanced (0.60–0.80) | **Senior→Principal** | Very strict: phase skip penalty -0.5/skip. Phase bonus +0.1. Penalizes redundant actions within a phase (-0.1 for repeating already-completed actions). |
| Expert (0.80–0.95) | **Principal** | Harshest: phase skip penalty -0.5/skip. Phase bonus +0.05 (near-zero, expects perfection). Redundancy penalty -0.15. Efficiency penalty for taking >N steps in any phase. |

This mirrors KubeSRE's approach: the junior judge at warmup is lenient and gives hints, while the principal judge at expert actively punishes inefficiency.

## Protocol Amendment & Recovery

> **Inspired by:** KubeSRE's mitigation phase — agents can partially fix before full resolution. Bio Experiment's `design_followup` meta-action.

The `request_protocol_amendment` action allows recovery from earlier mistakes:
- If FDA review fails, agent can amend the protocol and resubmit
- Amendment costs time and budget (realistic consequence)
- Successful recovery after failure gets a +0.3 **recovery bonus** (like KubeSRE's mitigation credit)
- Maximum 2 amendments per episode (prevents infinite retry loops)
- This teaches the agent that mistakes are recoverable but costly — a key real-world lesson
