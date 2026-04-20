# Multi-Layer Verification Specification

> **Gap G2:** KubeSRE used 3-layer verification: (1) real cluster health check, (2) rule-engine constraint validation, (3) LLM judge for qualitative assessment. Our environment needs the same multi-layer approach with clinical trial–specific checks. This document specifies each layer, when it fires, and how conflicts resolve.

## Overview

| Layer | Name | When It Fires | What It Checks | Verdict Type |
|-------|------|--------------|----------------|-------------|
| L1 | Programmatic Ground-Truth | Terminal (done=True) | Agent claims vs `TrialLatentState` hidden truth | Pass/Fail + numeric score |
| L2 | Rule-Engine Constraints | Per-step + Terminal | FDA rules, prerequisites, protocol validity | Block / Penalty / Pass |
| L3 | LLM Judge (Optional) | Terminal only | Workflow quality, reasoning coherence, clinical plausibility | 0.0–1.0 quality score |

### Design Principle: Programmatic First, LLM Last

> **KubeSRE lesson:** The LLM judge is never the sole arbiter. Programmatic checks (pod Running, health endpoints) are the source of truth. The LLM judge adds qualitative signal on top. Our equivalent: `scipy.stats` power calculations and ground-truth comparison are the source of truth. The LLM judge is optional enrichment.

```
                          ┌─────────────────────┐
                          │   Agent Completes    │
                          │   Trial Design       │
                          └──────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
             ┌──────┴──────┐                   ┌──────┴──────┐
             │  Per-Step    │                   │  Terminal    │
             │  (L2 only)  │                   │  (L1+L2+L3) │
             └──────┬──────┘                   └──────┬──────┘
                    │                                  │
          ┌────────┴────────┐         ┌───────────────┼───────────────┐
          │                  │         │               │               │
     ┌────┴────┐      ┌─────┴─────┐   │         ┌─────┴─────┐   ┌────┴────┐
     │ Hard    │      │ Soft      │   │         │ Rule      │   │ LLM     │
     │ Block   │      │ Penalty   │   │         │ Engine    │   │ Judge   │
     │ (-0.0)  │      │ (-0.15)   │   │         │ (L2)      │   │ (L3)    │
     └─────────┘      └───────────┘   │         └───────────┘   └─────────┘
                                      │
                               ┌──────┴──────┐
                               │ Ground-Truth│
                               │ Check (L1)  │
                               └─────────────┘
```

---

## Layer 1: Programmatic Ground-Truth Verification

### Purpose

Compare the agent's final trial design and conclusions against the hidden `TrialLatentState` values. This is the **primary success metric** — no LLM involved.

### When It Fires

- **Terminal only** — after `synthesize_conclusion` or episode timeout
- Runs automatically before terminal reward computation

### Checks

```python
class GroundTruthVerifier:
    """Layer 1: Programmatic verification against TrialLatentState."""

    def verify(self, agent_result: TrialResult,
               ground_truth: TrialLatentState) -> VerificationResult:
        checks = {}

        # 1. Statistical significance — did the trial detect the true effect?
        trial_power = calculate_power(
            effect_size=ground_truth.true_effect_size,
            n=agent_result.sample_size,
            alpha=agent_result.alpha,
        )
        checks["power_adequate"] = {
            "passed": trial_power >= 0.80,
            "value": trial_power,
            "threshold": 0.80,
            "reward_key": "r_terminal_power",
        }

        # 2. Type I error control
        checks["type_i_controlled"] = {
            "passed": agent_result.alpha <= 0.05,
            "value": agent_result.alpha,
            "threshold": 0.05,
        }

        # 3. Effect size estimate accuracy
        relative_error = abs(
            agent_result.effect_estimate - ground_truth.true_effect_size
        ) / ground_truth.true_effect_size
        checks["effect_estimate_accurate"] = {
            "passed": relative_error <= 0.30,
            "value": relative_error,
            "threshold": 0.30,
            "reward_key": "r_terminal_calibration",
        }

        # 4. Responder population identification
        checks["responder_identified"] = {
            "passed": (agent_result.responder_population
                       == ground_truth.true_responder_population),
            "value": agent_result.responder_population,
            "expected": ground_truth.true_responder_population,
            "reward_key": "r_terminal_calibration",
        }

        # 5. Dose selection (within ±1 level of optimal)
        dose_error = abs(agent_result.selected_dose - ground_truth.true_mtd)
        checks["dose_appropriate"] = {
            "passed": dose_error <= 50,  # Within 50mg of MTD
            "value": agent_result.selected_dose,
            "expected": ground_truth.true_mtd,
        }

        # 6. Overall trial success (p < alpha with true parameters)
        p_value = simulate_trial(
            effect_size=ground_truth.true_effect_size,
            n=agent_result.sample_size,
            alpha=agent_result.alpha,
            subgroup=agent_result.target_population,
            ground_truth=ground_truth,
        )
        checks["trial_significant"] = {
            "passed": p_value < agent_result.alpha,
            "value": p_value,
            "threshold": agent_result.alpha,
            "reward_key": "r_terminal_success",
        }

        # Aggregate
        total_passed = sum(1 for c in checks.values() if c["passed"])
        total_checks = len(checks)

        return VerificationResult(
            layer="L1_ground_truth",
            checks=checks,
            passed=total_passed,
            total=total_checks,
            score=total_passed / total_checks,
            verdict="pass" if total_passed >= 4 else "fail",
        )
```

### Scoring Integration

| Check | Reward Component | Impact |
|-------|-----------------|--------|
| `power_adequate` | `r_terminal_power` | +2.0 (≥0.90), +1.5 (≥0.80), -2.0 (<0.60) |
| `type_i_controlled` | `r_terminal_fda` | Required for FDA pass |
| `effect_estimate_accurate` | `r_terminal_calibration` | +1.0 (≤30% error), +0.5 (≤50%) |
| `responder_identified` | `r_terminal_calibration` | +3.0 if correct population |
| `dose_appropriate` | `r_terminal_calibration` | Contributes to mechanism knowledge |
| `trial_significant` | `r_terminal_success` | +5.0 to +7.0 if significant |

---

## Layer 2: Rule-Engine Constraint Validation

### Purpose

Enforce hard and soft rules representing FDA regulations, clinical protocol requirements, and good clinical practice (GCP). Unlike L1 (terminal only), L2 fires **every step**.

### When It Fires

- **Per-step:** before executing each action (hard constraints block, soft constraints penalize)
- **Terminal:** final protocol validation against all accumulated rules

### Per-Step Rules

```python
class RuleEngine:
    """Layer 2: FDA constraint validation."""

    # Hard constraints — block action execution
    HARD_CONSTRAINTS = {
        "estimate_effect_size": [
            ("dose_escalation_done", lambda s: s.dose_escalation_count >= 1,
             "Cannot estimate effect without Phase I dose escalation data"),
        ],
        "set_sample_size": [
            ("effect_estimated", lambda s: s.effect_estimated,
             "Cannot set sample size without effect size estimate"),
        ],
        "submit_to_fda_review": [
            ("endpoint_set", lambda s: s.primary_endpoint is not None,
             "FDA requires pre-specified primary endpoint"),
            ("sample_size_set", lambda s: s.sample_size is not None,
             "FDA requires planned sample size"),
        ],
        "run_interim_analysis": [
            ("fda_approved", lambda s: s.fda_approved,
             "Cannot run interim before FDA protocol approval"),
        ],
        "run_primary_analysis": [
            ("fda_approved", lambda s: s.fda_approved,
             "Cannot analyze before approved protocol"),
        ],
        "synthesize_conclusion": [
            ("analysis_done", lambda s: s.primary_analysis_complete,
             "Cannot conclude without primary analysis"),
        ],
        "modify_sample_size": [
            ("interim_done", lambda s: s.interim_complete,
             "Sample size modification requires interim analysis data"),
        ],
        "add_biomarker_stratification": [
            ("effect_estimated", lambda s: s.effect_estimated,
             "Need Phase I data to define biomarker stratification"),
        ],
    }

    # Soft constraints — allow action but apply penalty
    SOFT_CONSTRAINTS = {
        "set_sample_size": [
            ("min_n", lambda s, params: params.get("n", 0) >= 30,
             "Sample size < 30 is statistically inadvisable", -0.15),
            ("max_n", lambda s, params: params.get("n", 0) <= 1000,
             "Sample size > 1000 raises ethical and cost concerns", -0.10),
        ],
        "set_randomization_ratio": [
            ("valid_ratio", lambda s, params: params.get("ratio") in ["1:1", "2:1", "3:1"],
             "Unusual randomization ratio", -0.10),
        ],
        "run_dose_escalation": [
            ("max_cohorts", lambda s, params: s.dose_escalation_count < 8,
             "Excessive dose escalation (>8 cohorts)", -0.15),
        ],
    }

    def check_hard(self, action: TrialAction, state: TrialState) -> list:
        """Return list of violated hard constraints. If any, action is blocked."""
        violations = []
        constraints = self.HARD_CONSTRAINTS.get(action.action_type, [])
        for name, check_fn, message in constraints:
            if not check_fn(state):
                violations.append(RuleViolation(
                    rule=name, action=action.action_type,
                    message=message, severity="hard",
                ))
        return violations

    def check_soft(self, action: TrialAction, state: TrialState) -> list:
        """Return list of violated soft constraints with penalties."""
        violations = []
        constraints = self.SOFT_CONSTRAINTS.get(action.action_type, [])
        for name, check_fn, message, penalty in constraints:
            if not check_fn(state, action.parameters):
                violations.append(RuleViolation(
                    rule=name, action=action.action_type,
                    message=message, severity="soft", penalty=penalty,
                ))
        return violations
```

### Terminal Rules

At episode end, the full protocol is validated:

```python
TERMINAL_RULES = [
    ("primary_endpoint_prespecified",
     lambda r: r.primary_endpoint is not None,
     "No primary endpoint specified"),
    ("sample_size_justified",
     lambda r: r.sample_size is not None and r.sample_size >= 30,
     "Sample size not justified or too small"),
    ("randomization_specified",
     lambda r: r.randomization_set,
     "No randomization specified"),
    ("blinding_specified",
     lambda r: r.blinding is not None,
     "Blinding not specified"),
    ("control_arm_specified",
     lambda r: r.control_arm is not None,
     "No control arm specified"),
    ("safety_monitoring_planned",
     lambda r: r.safety_committee_set,
     "No safety monitoring committee specified"),
    ("alpha_controlled",
     lambda r: r.alpha <= 0.05,
     "Type I error not controlled (alpha > 0.05)"),
    ("interim_alpha_spending",
     lambda r: r.alpha_spent <= r.alpha_budget if r.interim_count > 0 else True,
     "Alpha spending exceeds budget after interim analyses"),
]
```

### How Hard vs Soft Interact

| Situation | Result |
|-----------|--------|
| Hard constraint violated | Action is **blocked**. Returns error in observation. Reward = 0 for this step. Step counter still increments. |
| Soft constraint violated | Action **executes** but incurs penalty (-0.10 to -0.15). Penalty logged in `r_penalty` component. |
| Both hard + soft violated | Hard takes precedence — action blocked. Soft penalties not applied (action didn't execute). |
| No violations | Action executes normally. `r_validity` = +0.3. |

---

## Layer 3: LLM Judge (Optional)

### Purpose

Qualitative assessment of the agent's clinical reasoning and workflow coherence. This is **not used for reward computation** during training (to avoid reward hacking via prompt manipulation). It is used only for:

1. **Evaluation/benchmarking** — richer comparison between policies
2. **Dashboard display** — human-readable quality assessment
3. **Pitch material** — shows environment sophistication to judges

### When It Fires

- **Terminal only** — after episode completes
- **Disabled during training** — only activated in `eval_compare.py` and dashboard mode
- **Can be toggled** via `--use-llm-judge` flag

### Judge Prompt Template

```python
LLM_JUDGE_PROMPT = """You are an experienced clinical trial reviewer evaluating an AI agent's 
trial design for {disease_name}.

## Agent's Trial Design
{agent_design_summary}

## Episode Transcript (Key Actions)
{action_summary}

## Evaluate on these 5 dimensions (0.0 to 1.0 each):

1. **Scientific Rigor** — Did the agent follow proper Phase I→II→III logic? 
   Did it gather adequate safety data before efficacy testing?

2. **Statistical Appropriateness** — Is the sample size justified? 
   Is the primary endpoint clinically meaningful? Is the analysis plan sound?

3. **Patient Safety** — Did the agent monitor safety signals? 
   Did it set appropriate stopping rules? Would the DSMB approve this design?

4. **Regulatory Readiness** — Would this protocol pass FDA IND review? 
   Are all required elements specified?

5. **Clinical Innovation** — Did the agent use biomarker stratification, 
   adaptive designs, or other advanced strategies appropriately?

Respond in JSON format:
{{
    "scientific_rigor": 0.0-1.0,
    "statistical_appropriateness": 0.0-1.0,
    "patient_safety": 0.0-1.0,
    "regulatory_readiness": 0.0-1.0,
    "clinical_innovation": 0.0-1.0,
    "overall_quality": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""
```

### Judge Persona Scaling (from KubeSRE)

| Tier | Judge Persona | Evaluation Style |
|------|--------------|------------------|
| Warmup | Junior Reviewer | Lenient — focuses on "did the agent try?" |
| Beginner | Junior→Senior | Standard — expects correct workflow |
| Intermediate | Senior Reviewer | Strict — expects justified decisions |
| Advanced | Senior→Principal | Very strict — expects optimal strategies |
| Expert | Principal Investigator | Harshest — expects near-human-expert reasoning |

The persona is injected into the prompt prefix: `"You are a {persona} clinical trial reviewer..."`

### Why LLM Judge is Optional

1. **Cost** — running an LLM judge per episode during training adds latency and API costs
2. **Reward hacking** — LLM judges can be gamed by producing verbose/confident-sounding but wrong conclusions
3. **Reproducibility** — LLM scores are non-deterministic; programmatic checks are deterministic
4. **KubeSRE lesson** — even KubeSRE used the LLM judge as supplementary, not primary. Pod health checks were the real ground truth.

---

## Conflict Resolution

When layers disagree:

| Scenario | L1 (Ground-Truth) | L2 (Rules) | L3 (LLM Judge) | Resolution |
|----------|-------------------|------------|-----------------|------------|
| Good trial, correct claims | Pass | Pass | High (0.8+) | **Success** — clear win. All rewards positive. |
| Good trial, FDA violation | Pass | Fail (1 soft) | High | **Partial success** — L1 drives terminal reward. L2 penalty reduces total by -0.15 per violation. |
| Good result, wrong reasoning | Pass | Pass | Low (0.3) | **Success with caveat** — reward is based on L1+L2 (programmatic). L3 noted in eval report but does NOT reduce reward. |
| Bad trial, claims success | Fail | Pass | Low | **Failure** — L1 is authoritative. No success reward. L3 confirming low quality is consistent. |
| Wrong design, FDA compliant | Fail | Pass | Medium | **Failure** — FDA compliance alone doesn't mean the trial works. L1 (ground truth) is the source of truth. |
| Right idea, rule violated | Pass (partial) | Fail (hard) | High | **Mixed** — L1 partial credit, L2 hard failure blocks some actions. This is correct: good instincts but poor execution. |

### Resolution Rule

```
final_verdict = L1_verdict  # Ground truth is always authoritative
reward = L1_reward_components + L2_penalty_components
llm_quality_score = L3_score  # Logged but not added to reward
```

**L1 determines success/failure. L2 modifies reward via penalties. L3 is informational only.**

---

## Verification Summary Table

This table is logged per episode for diagnostics:

```json
{
    "episode_id": 142,
    "verification": {
        "L1_ground_truth": {
            "checks_passed": 5,
            "checks_total": 6,
            "score": 0.833,
            "verdict": "pass",
            "details": {
                "power_adequate": true,
                "type_i_controlled": true,
                "effect_estimate_accurate": true,
                "responder_identified": true,
                "dose_appropriate": true,
                "trial_significant": false
            }
        },
        "L2_rule_engine": {
            "hard_violations": 0,
            "soft_violations": 1,
            "total_penalty": -0.15,
            "terminal_rules_passed": 7,
            "terminal_rules_total": 8,
            "verdict": "pass_with_warnings"
        },
        "L3_llm_judge": {
            "enabled": true,
            "persona": "senior",
            "scores": {
                "scientific_rigor": 0.85,
                "statistical_appropriateness": 0.78,
                "patient_safety": 0.90,
                "regulatory_readiness": 0.72,
                "clinical_innovation": 0.65
            },
            "overall_quality": 0.78,
            "reasoning": "Agent demonstrated strong Phase I execution but missed biomarker stratification opportunity."
        }
    }
}
```

---

## Implementation Notes for Suyash

1. `GroundTruthVerifier` (L1) is a pure function: `verify(agent_result, ground_truth) → VerificationResult`. No side effects.
2. `RuleEngine` (L2) lives in `server/rules.py`. Called from `step()` before action execution.
3. `LLMJudge` (L3) is behind a feature flag. Default OFF during training. OFF by default to avoid API cost.
4. All three layers write to the same episode transcript JSONL for post-hoc analysis.
5. The `TrialJudge` class orchestrates all 3 layers at terminal time.
