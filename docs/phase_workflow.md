# Clinical Trial Phase-Aware Workflow & Scoring


## Clinical Trial Workflow Phases

A clinical trial follows a strict sequential workflow. The agent should learn this ordering through reward signal, not hard-coding. Phase-order bonuses and skip penalties teach the agent to follow the correct diagnostic sequence.

Our workflow maps a sequential decision-making pattern to clinical trial design:

| Sequential Phase | Our Phase | Why it maps |
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


The phase-order evaluation strictness scales with curriculum tier:

| Tier | Judge Persona | Phase-Order Behavior |
|---|---|---|
| Warmup (0.0–0.25) | **Junior** | Lenient: allows 1 phase skip without penalty. Includes hint in observation (e.g., "Consider running dose escalation before setting sample size"). Phase bonus still +0.2. |
| Beginner (0.25–0.40) | **Junior→Senior** | Standard: phase skip penalty -0.3/skip. No hints. Phase bonus +0.2. |
| Intermediate (0.40–0.60) | **Senior** | Strict: phase skip penalty -0.3/skip. Phase bonus reduced to +0.15 (expects correct ordering as baseline). |
| Advanced (0.60–0.80) | **Senior→Principal** | Very strict: phase skip penalty -0.5/skip. Phase bonus +0.1. Penalizes redundant actions within a phase (-0.1 for repeating already-completed actions). |
| Expert (0.80–0.95) | **Principal** | Harshest: phase skip penalty -0.5/skip. Phase bonus +0.05 (near-zero, expects perfection). Redundancy penalty -0.15. Efficiency penalty for taking >N steps in any phase. |

This mirrors approach: the junior judge at warmup is lenient and gives hints, while the principal judge at expert actively punishes inefficiency.

## Protocol Amendment & Recovery


The `request_protocol_amendment` action allows recovery from earlier mistakes:
- If FDA review fails, agent can amend the protocol and resubmit
- Amendment costs time and budget (realistic consequence)
- Successful recovery after failure gets a +0.3 **recovery bonus** (like mitigation credit)
- Maximum 2 amendments per episode (prevents infinite retry loops)
- This teaches the agent that mistakes are recoverable but costly — a key real-world lesson

---

## Phase-Order Bonus/Penalty Lookup Table (Push 3 — G13)

> **Complete lookup table for `compute_ordering_reward()`**. Suyash implements this in `server/rewards.py`. Each cell shows the reward delta for transitioning from row-phase to column-phase.

### Transition Matrix (Tier 0–2: Standard Penalties)

From ↓ / To → | lit_rev | hypo | p1_des | p1_ana | p2_des | reg | enroll | mon | analysis | concl |
|---|---|---|---|---|---|---|---|---|---|---|
| **(start)** | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 | -0.9 | -1.2 | -1.5 | -1.8 | -2.1 |
| **lit_rev** | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 | -0.9 | -1.2 | -1.5 | -1.8 | -2.1 |
| **hypothesis** | +0.2 | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 | -0.9 | -1.2 | -1.5 | -1.8 |
| **phase_i_design** | +0.2 | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 | -0.9 | -1.2 | -1.5 | -1.8 |
| **phase_i_analysis** | -0.3 | +0.2 | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 | -0.9 | -1.2 | -1.5 |
| **phase_ii_design** | -0.6 | -0.3 | +0.2 | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 | -0.9 | -1.2 |
| **regulatory** | -0.9 | -0.6 | -0.3 | +0.2 | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 | -0.9 |
| **enrollment** | N/A | N/A | N/A | N/A | N/A | +0.2 | +0.2 | +0.2 | -0.3 | -0.6 |
| **monitoring** | N/A | N/A | N/A | N/A | N/A | +0.2 | +0.2 | +0.2 | +0.2 | -0.3 |
| **analysis** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | +0.2 | +0.2 | +0.2 |

**Reading the table:**
- **+0.2** = correct forward or same-phase transition (bonus)
- **-0.3 × N** = skipping N phases ahead (penalty scales with skip distance)
- **N/A** = backward transition from late phases — penalized as -0.3 per phase regressed (rare, only via `request_protocol_amendment`)
- Diagonal entries (same phase) = +0.2 (continuation is good, but redundancy penalty may also apply at tier 3+)

### Tier 3–4 Penalty Multiplier

At Advanced and Expert tiers, penalties are harsher:

| Parameter | Tier 0–2 | Tier 3 (Advanced) | Tier 4 (Expert) |
|-----------|----------|-------------------|-----------------|
| Forward bonus | +0.20 | +0.10 | +0.05 |
| Skip penalty (per phase) | -0.30 | -0.50 | -0.50 |
| Same-phase continuation | +0.20 | +0.10 | +0.05 |
| Redundancy penalty | 0.00 | -0.10 | -0.15 |
| Phase efficiency penalty | 0.00 | 0.00 | -0.05/step over budget |
| Backward regression | -0.30/phase | -0.50/phase | -0.50/phase |

### Phase Step Budgets (Expert Tier Only)

At Expert tier, spending too many steps in a single phase incurs an efficiency penalty:

| Phase | Max Steps Allowed | Penalty per Extra Step |
|-------|-------------------|----------------------|
| `literature_review` | 2 | -0.05 |
| `hypothesis` | 2 | -0.05 |
| `phase_i_design` | 8 | -0.05 |
| `phase_i_analysis` | 3 | -0.05 |
| `phase_ii_design` | 10 | -0.05 |
| `regulatory` | 3 | -0.05 |
| `monitoring` | 5 | -0.05 |
| `analysis` | 3 | -0.05 |
| `conclusion` | 2 | -0.05 |

---

## Expanded `_detect_phase()` Patterns (Push 3 — G13)

The Push 1 `_detect_phase()` covers the main action types. This expanded version adds pattern matching for edge cases, multi-action sequences, and context-dependent classification:

```python
def _detect_phase(action: TrialAction, history: list) -> str:
    """
    Classify action into clinical workflow phase.
    
    Enhanced patterns (Push 3):
    - Context-dependent classification for ambiguous actions
    - Multi-action sequence detection for phase transitions
    - Fallback heuristic based on step count and history
    """
    action_type = action.action_type

    # ── Explicit Phase I actions ──
    if action_type in ("run_dose_escalation", "observe_safety_signal"):
        return "phase_i_design"

    if action_type == "estimate_effect_size":
        return "phase_i_analysis"

    # ── Explicit Phase II design actions ──
    if action_type in ("set_primary_endpoint", "set_sample_size",
                        "set_inclusion_criteria", "set_exclusion_criteria",
                        "set_dosing_schedule", "set_control_arm",
                        "set_randomization_ratio", "set_blinding"):
        return "phase_ii_design"

    # ── Regulatory ──
    if action_type == "submit_to_fda_review":
        return "regulatory"
    if action_type == "request_protocol_amendment":
        # Amendment phase depends on context:
        # - Before FDA approval: still in regulatory
        # - After FDA approval: back to regulatory (re-submission)
        return "regulatory"

    # ── Monitoring / Adaptive ──
    if action_type == "run_interim_analysis":
        return "monitoring"
    if action_type == "modify_sample_size":
        return "monitoring"  # Adaptation happens during monitoring
    if action_type == "add_biomarker_stratification":
        # Context-dependent:
        # - Before Phase II design: part of phase_i_analysis (discovery)
        # - During Phase II: part of phase_ii_design (enrichment strategy)
        # - After enrollment: monitoring (adaptive stratification)
        past_phases = _get_past_phases(history)
        if "monitoring" in past_phases or "enrollment" in past_phases:
            return "monitoring"
        elif "phase_ii_design" in past_phases:
            return "phase_ii_design"
        else:
            return "phase_i_analysis"

    # ── Analysis ──
    if action_type == "run_primary_analysis":
        return "analysis"

    # ── Conclusion ──
    if action_type == "synthesize_conclusion":
        return "conclusion"

    # ── Fallback heuristic ──
    # For any unrecognized action type, infer phase from history position
    if len(history) == 0:
        return "literature_review"
    
    past_phases = _get_past_phases(history)
    if not past_phases:
        return "literature_review"
    
    # Return the most recent phase (agent is likely still in it)
    last_phase = _detect_phase_from_history(history[-1])
    return last_phase if last_phase else "literature_review"


def _get_past_phases(history: list) -> set:
    """Extract set of phases seen in history."""
    phases = set()
    for i, h in enumerate(history):
        p = _detect_phase(h["action"], history[:i])
        phases.add(p)
    return phases


def compute_ordering_reward(
    current_phase: str,
    history: list,
    tier: int,
    steps_in_current_phase: int,
) -> float:
    """
    Compute r_ordering for a single step.
    
    Returns:
        float: bonus (+0.2 to +0.05) or penalty (-0.3 to -0.5 per skip)
    """
    # Tier-specific parameters
    TIER_PARAMS = {
        0: {"bonus": 0.20, "skip_penalty": -0.30, "redundancy": 0.0,  "phase_budget": None},
        1: {"bonus": 0.20, "skip_penalty": -0.30, "redundancy": 0.0,  "phase_budget": None},
        2: {"bonus": 0.15, "skip_penalty": -0.30, "redundancy": 0.0,  "phase_budget": None},
        3: {"bonus": 0.10, "skip_penalty": -0.50, "redundancy": -0.10, "phase_budget": None},
        4: {"bonus": 0.05, "skip_penalty": -0.50, "redundancy": -0.15, "phase_budget": PHASE_STEP_BUDGETS},
    }

    params = TIER_PARAMS[tier]
    reward = 0.0

    # Phase-order check
    if _is_phase_order_correct(current_phase, history):
        reward += params["bonus"]
    else:
        skipped = _get_skipped_phases(current_phase, history)
        reward += params["skip_penalty"] * len(skipped)

    # Redundancy penalty (tier 3+): repeating actions in a completed sub-phase
    if params["redundancy"] != 0.0:
        if _is_redundant_action(current_phase, history):
            reward += params["redundancy"]

    # Phase efficiency penalty (tier 4 only): exceeding step budget per phase
    if params["phase_budget"] is not None:
        max_steps = params["phase_budget"].get(current_phase, 999)
        if steps_in_current_phase > max_steps:
            reward += -0.05 * (steps_in_current_phase - max_steps)

    return reward


# Phase step budgets for Expert tier
PHASE_STEP_BUDGETS = {
    "literature_review": 2,
    "hypothesis": 2,
    "phase_i_design": 8,
    "phase_i_analysis": 3,
    "phase_ii_design": 10,
    "regulatory": 3,
    "monitoring": 5,
    "analysis": 3,
    "conclusion": 2,
}
```

### Warmup Hint System

At Warmup tier (0), when the agent makes a phase-order mistake, the observation includes a corrective hint instead of (or in addition to) a penalty:

```python
PHASE_HINTS = {
    ("phase_i_design", "phase_ii_design"): 
        "Hint: Consider completing Phase I analysis (estimate_effect_size) before designing Phase II.",
    ("phase_i_design", "regulatory"):
        "Hint: You need a Phase II design (set_primary_endpoint, set_sample_size) before FDA submission.",
    ("phase_i_design", "analysis"):
        "Hint: Clinical trials follow Phase I → Phase II → Regulatory → Monitoring → Analysis.",
    ("literature_review", "monitoring"):
        "Hint: You haven't completed Phase I or Phase II design yet. Start with run_dose_escalation.",
    ("phase_ii_design", "analysis"):
        "Hint: Submit your protocol to FDA review first, then run interim analysis before primary analysis.",
}

def get_phase_hint(from_phase: str, to_phase: str) -> str | None:
    """Return a hint message if available for this phase transition at Warmup tier."""
    return PHASE_HINTS.get((from_phase, to_phase))
```

---

## Cross-Reference

- **Curriculum Policy** (`curriculum_policy.md`): tier definitions drive which penalty parameters apply
- **Verification Spec** (`verification_spec.md`): L2 rule engine enforces hard prerequisites (this doc covers soft phase-order scoring)
- **Reward Spec** (`reward_spec.md`): `r_ordering` is computed by `compute_ordering_reward()` defined here
- **Benchmark Protocol** (`benchmark_protocol.md`): scripted baseline follows perfect phase order; random baseline does not
- **Dashboard Metrics** (`dashboard_metrics.md`): phase compliance rate displayed in Panel 5 capability radar