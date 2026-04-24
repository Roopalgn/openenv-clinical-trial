# Chapter 8: The Rule Engine — FDA Compliance

## Why Rules Matter

In real life, you can't just run a clinical trial however you want. The FDA (Food and Drug Administration) has strict rules. If you violate them, your trial gets shut down and your drug never reaches patients.

Our environment encodes these rules as **hard constraints**. When the agent tries an action that violates a rule, it gets punished immediately.

## The Three Layers of Rules

### Layer 1: Transition Table ("What can I do in this phase?")

Each clinical phase only allows certain actions:

```python
# From server/rules/fda_rules.py
TRANSITION_TABLE = {
    "literature_review": {
        "set_primary_endpoint",
        "observe_safety_signal",
        "estimate_effect_size",
    },
    "hypothesis": {
        "set_primary_endpoint",
        "set_sample_size",
        "set_inclusion_criteria",
        "set_exclusion_criteria",
        "estimate_effect_size",
    },
    "design": {
        "set_sample_size",
        "set_inclusion_criteria",
        "set_exclusion_criteria",
        "set_dosing_schedule",
        "set_control_arm",
        "set_randomization_ratio",
        "set_blinding",
        "add_biomarker_stratification",
        "request_protocol_amendment",
        "enroll_patients",
    },
    "enrollment": {
        "enroll_patients",
        "run_dose_escalation",
        "observe_safety_signal",
        "modify_sample_size",
        "add_biomarker_stratification",
        "request_protocol_amendment",
    },
    "monitoring": {
        "run_interim_analysis",
        "observe_safety_signal",
        "modify_sample_size",
        "request_protocol_amendment",
    },
    "analysis": {
        "run_primary_analysis",
        "estimate_effect_size",
        "synthesize_conclusion",
    },
    "submission": {
        "submit_to_fda_review",
        "request_protocol_amendment",
        "synthesize_conclusion",
    },
}
```

**Reading this table:** If you're in the "enrollment" phase, you can enroll patients, run dose escalation, observe safety, modify sample size, add biomarker stratification, or request a protocol amendment. But you CANNOT set blinding or submit to FDA — those belong to other phases.

**Real-world grounding:** This mirrors ICH E9 guidelines, which specify what activities are appropriate at each stage of a clinical trial.

### Layer 2: Hard Rules ("Specific regulatory requirements")

Beyond the transition table, there are specific rules:

```python
# Sample size minimum
if action.action_type == SET_SAMPLE_SIZE:
    if sample_size < 30:
        violations.append("Sample size below regulatory minimum of 30")

# Must complete Phase I before FDA submission
if action.action_type == SUBMIT_TO_FDA_REVIEW:
    if not latent.phase_i_complete:
        violations.append("Cannot submit to FDA: Phase I not completed")
    if not latent.protocol_submitted:
        violations.append("Cannot submit to FDA: protocol not submitted")

# Must do interim analysis before primary analysis
if action.action_type == RUN_PRIMARY_ANALYSIS:
    if not latent.interim_complete:
        violations.append("Cannot run primary analysis: interim not completed")

# Must have patients enrolled for interim analysis
if action.action_type == RUN_INTERIM_ANALYSIS:
    if latent.patients_enrolled <= 0:
        violations.append("Cannot run interim analysis: no patients enrolled")

# Must synthesize only after trial is complete
if action.action_type == SYNTHESIZE_CONCLUSION:
    if not latent.trial_complete:
        violations.append("Cannot synthesize: trial not complete")
```

### Layer 3: Prerequisite Rules ("What must happen first?")

Some actions require other actions to have been performed first:

```python
# From server/rules/prerequisite_rules.py
_HISTORY_PREREQUISITES = {
    SET_DOSING_SCHEDULE:           [SET_PRIMARY_ENDPOINT],         # Must know what you're measuring
    SET_CONTROL_ARM:               [SET_PRIMARY_ENDPOINT],         # Must know what you're comparing
    SET_RANDOMIZATION_RATIO:       [SET_CONTROL_ARM],              # Must have a control arm first
    SET_BLINDING:                  [SET_RANDOMIZATION_RATIO],      # Must randomize before blinding
    RUN_DOSE_ESCALATION:           [SET_DOSING_SCHEDULE],          # Must have a dosing schedule
    ESTIMATE_EFFECT_SIZE:          [RUN_DOSE_ESCALATION],          # Must have dose data first
    ADD_BIOMARKER_STRATIFICATION:  [SET_INCLUSION_CRITERIA],       # Must define who's eligible
}
```

These form a **dependency chain**:

```
SET_PRIMARY_ENDPOINT
├── SET_DOSING_SCHEDULE
│   └── RUN_DOSE_ESCALATION
│       └── ESTIMATE_EFFECT_SIZE
└── SET_CONTROL_ARM
    └── SET_RANDOMIZATION_RATIO
        └── SET_BLINDING

SET_INCLUSION_CRITERIA
└── ADD_BIOMARKER_STRATIFICATION
```

If the agent tries to run dose escalation without first setting a dosing schedule, it gets a violation.

## How Compliance Checking Works

```python
def check_fda_compliance(action, latent) -> ComplianceResult:
    violations = []
    
    # Check 1: Is this action allowed in the current phase?
    permitted = TRANSITION_TABLE[latent.episode_phase]
    if action.action_type not in permitted:
        violations.append(f"'{action.action_type}' not permitted in '{latent.episode_phase}'")
    
    # Check 2: Specific hard rules
    # (sample size, FDA submission prereqs, etc.)
    ...
    
    # Check 3: History prerequisites
    prerequisite_violations = check_prerequisites(action, latent)
    violations.extend(prerequisite_violations)
    
    return ComplianceResult(
        valid=(len(violations) == 0),
        violations=violations
    )
```

The result is a simple object:
- `valid: True/False` — did all checks pass?
- `violations: list[str]` — what went wrong (if anything)?

## What Happens on Violation

When the agent takes an invalid action:

```python
# From episode_manager.py step() method
compliance = check_fda_compliance(action, self._latent)

if not compliance.valid:
    # 1. Create penalty reward (no positive components)
    reward = RewardBreakdown(
        r_validity=-1.0,
        r_ordering=0.0,
        r_info_gain=0.0,
        r_efficiency=0.0,
        r_novelty=0.0,
        r_penalty=-0.5 × len(violations),  # -0.5 per violation
        r_terminal_success=0.0,
        r_terminal_calibration=0.0,
    )
    
    # 2. DON'T update the hidden state (action rejected entirely)
    # 3. Return observation showing the violations
    obs.rule_violations = compliance.violations
    # 4. Episode continues (not terminated — agent gets another chance)
    return obs, reward, done=False, info
```

Key design choices:
1. **State doesn't change** — invalid actions are rejected entirely (no budget spent, no time passes)
2. **Episode continues** — the agent gets another chance (harsh but not fatal)
3. **Violations are visible** — the agent sees what went wrong in the observation

> **Design Decision Box: Why Not Just Block Invalid Actions?**
>
> We COULD prevent the agent from ever seeing invalid actions (by filtering them out).
> But we chose to let the agent TRY invalid actions and get punished because:
>
> 1. **Learning signal:** The agent learns WHY certain actions are invalid in certain states. 
>    This is more informative than just hiding options.
> 2. **Available actions hint:** We DO show available actions in the observation, but the agent
>    must learn to use that information. If it ignores the hint, it gets punished.
> 3. **Realistic:** In real life, researchers CAN submit bad protocols — they just get rejected.

## The Phase Detector

The phase detector classifies each action into one of 7 clinical phases:

```python
PHASE_ORDER = [
    "literature_review",     # Reviewing existing knowledge
    "hypothesis",            # Forming a hypothesis about the drug
    "design",                # Designing the trial protocol
    "enrollment",            # Recruiting patients
    "monitoring",            # Monitoring the trial
    "analysis",              # Analyzing results
    "submission",            # Submitting to FDA
]

_ACTION_TO_PHASE = {
    # hypothesis
    ESTIMATE_EFFECT_SIZE: "hypothesis",
    ADD_BIOMARKER_STRATIFICATION: "hypothesis",
    
    # design
    SET_PRIMARY_ENDPOINT: "design",
    SET_SAMPLE_SIZE: "design",
    SET_INCLUSION_CRITERIA: "design",
    SET_DOSING_SCHEDULE: "design",
    # ... more design actions ...
    
    # enrollment
    ENROLL_PATIENTS: "enrollment",
    
    # monitoring
    RUN_DOSE_ESCALATION: "monitoring",
    OBSERVE_SAFETY_SIGNAL: "monitoring",
    RUN_INTERIM_ANALYSIS: "monitoring",
    
    # analysis
    RUN_PRIMARY_ANALYSIS: "analysis",
    SYNTHESIZE_CONCLUSION: "analysis",
    
    # submission
    SUBMIT_TO_FDA_REVIEW: "submission",
}
```

The phase detector then checks if the transition is valid:

```python
def detect_phase(action, history) -> (phase_name, phase_order_correct):
    phase_name = _ACTION_TO_PHASE[action.action_type]
    
    if not history:
        return phase_name, True  # First action is always correct
    
    last_phase = history[-1]
    last_idx = PHASE_ORDER.index(last_phase)
    current_idx = PHASE_ORDER.index(phase_name)
    
    # Regression (going backward): not correct
    if current_idx < last_idx:
        return phase_name, False
    
    # Skipping phases: not correct
    if current_idx - last_idx > 1:
        return phase_name, False
    
    # Same phase or advancing by exactly one: correct
    return phase_name, True
```

## The Ideal Action Sequence

Based on the rules and phase ordering, here's what a well-designed episode looks like:

```
Step  1: set_primary_endpoint          (design)      → What are we measuring?
Step  2: set_sample_size               (design)      → How many patients?
Step  3: set_inclusion_criteria        (design)      → Who can participate?
Step  4: set_exclusion_criteria        (design)      → Who's excluded?
Step  5: set_dosing_schedule           (design)      → Drug dose and timing
Step  6: set_control_arm               (design)      → Placebo/standard care?
Step  7: set_randomization_ratio       (design)      → 1:1 or 2:1 drug:control?
Step  8: set_blinding                  (design)      → Double-blind?
Step  9: enroll_patients (n=50)        (enrollment)  → First patients
Step 10: run_dose_escalation           (monitoring)  → Phase I: find safe dose
Step 11: observe_safety_signal         (monitoring)  → Check for side effects
Step 12: estimate_effect_size          (hypothesis)  → Phase II: does it work?
Step 13: add_biomarker_stratification  (hypothesis)  → Who responds best?
Step 14: enroll_patients (n=100)       (enrollment)  → Phase III: more patients
Step 15: run_interim_analysis          (monitoring)  → Mid-trial check
Step 16: modify_sample_size            (monitoring)  → Adjust if needed
Step 17: run_primary_analysis          (analysis)    → Final statistical test
Step 18: synthesize_conclusion         (analysis)    → Write up findings
Step 19: submit_to_fda_review          (submission)  → Submit to FDA
```

But the expert agent would do much more — adapting based on what's discovered at each step. If the interim analysis shows the drug isn't working, a smart agent might stop early (saving budget) rather than continuing to Phase III.

---

## Chapter 8 Glossary

| Keyword | Definition |
|---------|-----------|
| **FDA (Food and Drug Administration)** | U.S. agency regulating drugs and clinical trials |
| **ICH E9** | International Council for Harmonisation guideline for clinical trial statistics |
| **Transition Table** | Maps each trial phase to allowed actions |
| **Prerequisite** | An action that must be completed before another action |
| **Compliance Check** | Verifying an action follows all rules |
| **Hard Constraint** | An absolute rule that cannot be violated |
| **Dependency Chain** | A sequence of prerequisites (A requires B requires C) |
| **Phase Ordering** | The correct sequence of clinical trial phases |
| **Regression** | Going backward to an earlier phase (penalized) |
| **Protocol** | The detailed plan for how a clinical trial will be conducted |
| **Double-Blind** | Neither patients nor researchers know who gets drug vs placebo |
| **Randomization** | Randomly assigning patients to drug or control groups |
