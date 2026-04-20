# Scenario Cards — Clinical Trial Designer

> **Inspired by:** Bio Experiment's 4 curated scenarios with paper references (DOIs), true DE genes with log2FC values, and hidden ground-truth mechanisms. Each scenario has a unique challenge, realistic constraints, and literature-backed biology. KubeSRE's adversarial scenarios targeting different fault types. All winners: scenarios span easy→hard for curriculum.

## Overview

| # | Scenario ID | Disease | Difficulty | Challenge | Budget | Time |
|---|-------------|---------|-----------|-----------|--------|------|
| 1 | `solid_tumor_chemo` | Non-small cell lung cancer | Easy→Expert | Find EGFR+ subgroup | $2.5M | 180 days |
| 2 | `autoimmune_biologic` | Rheumatoid arthritis | Medium→Expert | U-shaped dose-response | $1.8M | 150 days |
| 3 | `cns_depression` | Treatment-resistant depression | Hard→Expert | High placebo response | $3.0M | 210 days |
| 4 | `rare_disease_orphan` | Rare pediatric metabolic disorder | Very Hard→Expert | Tiny n, adaptive design | $1.2M | 240 days |

Each scenario can run at all 5 curriculum tiers by the `NoiseModel` injecting increasing noise, tighter budgets, and harder parameter ranges.

---

## Scenario 1: `solid_tumor_chemo`

### Clinical Context

**Disease:** Non-small cell lung cancer (NSCLC) — the most common form of lung cancer, accounting for ~85% of cases.

**Drug:** Novel EGFR tyrosine kinase inhibitor (3rd generation) — targets epidermal growth factor receptor mutations.

**Real-world parallel:** Osimertinib (Tagrisso) Phase III trial showed 18.9 months PFS in EGFR-mutant NSCLC vs 10.2 months placebo (Soria et al., 2018, *NEJM*; DOI: 10.1056/NEJMoa1713137). Our scenario uses similar but fictional parameters.

### Hidden Ground Truth (`TrialLatentState`)

```python
SOLID_TUMOR_GROUND_TRUTH = {
    # Biology
    "true_effect_size": 0.31,              # 31% PFS improvement (overall population)
    "true_effect_size_subgroup": 0.58,     # 58% PFS improvement in EGFR+ patients
    "true_side_effect_rate": 0.12,         # 12% ≥Grade 3 adverse events
    "true_responder_population": "EGFR+",  # Only EGFR+ patients benefit significantly
    "true_responder_prevalence": 0.35,     # 35% of enrolled patients are EGFR+
    "true_mechanism": "EGFR tyrosine kinase inhibition",
    "true_dose_response": {
        50:  0.08,   # Minimal effect
        100: 0.19,   # Moderate
        150: 0.31,   # Plateau — this is the optimal dose
        200: 0.29,   # Slightly lower (toxicity trade-off)
        250: 0.22,   # Reduced due to dose-limiting toxicity
    },
    "true_mtd": 200,                       # Maximum tolerated dose (mg)

    # Technical
    "placebo_response_rate": 0.05,         # 5% spontaneous improvement
    "dropout_rate": 0.15,                  # 15% patient dropout
    "site_variability": 0.10,             # 10% inter-site variance
    "measurement_noise": 0.08,            # 8% measurement error

    # Resources
    "initial_budget": 2_500_000,           # $2.5M
    "initial_time_days": 180,              # 6 months
    "cost_per_patient": 15_000,            # $15K per patient
}
```

### Challenge

The drug works overall (31% PFS improvement) but works **dramatically better in EGFR+ patients** (58%). If the agent enrolls the general population, the trial may succeed but with marginal significance. To achieve strong results, the agent must:

1. **Phase I:** Run dose escalation → discover 150mg is optimal, 200mg is MTD
2. **Phase I analysis:** Estimate effect size. The noisy estimate will suggest ~0.25–0.40 overall.
3. **Key insight:** Use `add_biomarker_stratification` with EGFR status to discover the 58% effect in EGFR+ patients
4. **Phase II:** Set inclusion criteria to `EGFR+` → dramatically improve statistical power with fewer patients

### Curriculum Scaling

| Tier | Changes |
|------|---------|
| Warmup | true_effect_size=0.50 (easy to detect), no hidden subgroup, low noise |
| Beginner | true_effect_size=0.31 (standard), subgroup effect visible with stratification |
| Intermediate | Higher measurement_noise=0.15, tighter budget ($2.0M) |
| Advanced | Misleading Phase I: first 2 cohorts show weak signal due to sampling noise |
| Expert | true_effect_size=0.15 (overall), EGFR+ prevalence=0.20 (rare), budget=$1.5M |

### Programmatic Verification

```python
def verify_solid_tumor(agent_design, trial_result, ground_truth):
    """
    Returns (passed: bool, score: float, details: dict)
    No LLM needed — all checks are math or rule-based.
    """
    checks = {}

    # 1. Statistical significance (ground truth: simulation)
    checks["p_value_significant"] = trial_result.p_value < agent_design.alpha
    # Verification: scipy.stats.ttest_ind on simulated outcomes

    # 2. Statistical power (ground truth: formula)
    power = calculate_power(
        effect_size=ground_truth["true_effect_size_subgroup"]
            if agent_design.targets_subgroup("EGFR+")
            else ground_truth["true_effect_size"],
        n=agent_design.sample_size,
        alpha=agent_design.alpha
    )
    checks["power_adequate"] = power >= 0.80

    # 3. Subgroup identification
    checks["correct_subgroup"] = (
        agent_design.inclusion_criteria
        and "EGFR" in agent_design.inclusion_criteria.upper()
    )

    # 4. Dose selection
    checks["dose_near_optimal"] = (
        abs(agent_design.dose - 150) <= 50  # Within 50mg of 150mg optimal
    )

    # 5. FDA compliance
    checks["fda_all_pass"] = all(
        rule.check(agent_design) for rule in FDA_RULES.values()
    )

    # 6. Budget
    estimated_cost = (
        agent_design.sample_size * ground_truth["cost_per_patient"]
        + 200_000  # Fixed regulatory/site costs
    )
    checks["within_budget"] = estimated_cost <= ground_truth["initial_budget"]

    # Scoring
    score = sum([
        5.0 if checks["p_value_significant"] else -1.0,
        2.0 if checks["power_adequate"] else -2.0,
        3.0 if checks["correct_subgroup"] else 0.0,
        0.5 if checks["dose_near_optimal"] else 0.0,
        2.0 if checks["fda_all_pass"] else -1.0,
        1.0 if checks["within_budget"] else -0.5,
    ])

    passed = checks["p_value_significant"] and checks["power_adequate"]
    return passed, score, checks
```

---

## Scenario 2: `autoimmune_biologic`

### Clinical Context

**Disease:** Rheumatoid arthritis (RA) — a chronic autoimmune disease affecting joints.

**Drug:** Novel IL-6 receptor antagonist biologic — targets inflammatory pathway.

**Real-world parallel:** Tocilizumab (Actemra) showed non-linear dose-response with optimal efficacy at 8mg/kg in RA (Smolen et al., 2008, *Lancet*; DOI: 10.1016/S0140-6736(08)60453-5). Our scenario uses a U-shaped response where both too-low and too-high doses are suboptimal.

### Hidden Ground Truth (`TrialLatentState`)

```python
AUTOIMMUNE_GROUND_TRUTH = {
    # Biology — U-SHAPED DOSE RESPONSE (the trap)
    "true_effect_size": 0.42,              # 42% ACR50 improvement at optimal dose
    "true_effect_size_subgroup": None,     # No hidden subgroup — dose is the challenge
    "true_side_effect_rate": 0.09,         # 9% serious infections
    "true_responder_population": "all",    # All RA patients can respond
    "true_responder_prevalence": 1.0,
    "true_mechanism": "IL-6 receptor antagonism",
    "true_dose_response": {
        50:  0.12,   # Sub-therapeutic
        100: 0.28,   # Moderate
        150: 0.38,   # Good
        200: 0.42,   # OPTIMAL — peak efficacy
        250: 0.35,   # Decreasing — immunosuppression side effects
        300: 0.22,   # Poor — severe immunosuppression outweighs benefit
        400: 0.10,   # Toxic — barely better than placebo
    },
    "true_mtd": 300,                       # Can tolerate up to 300mg

    # Technical
    "placebo_response_rate": 0.15,         # 15% placebo ACR50 (moderate)
    "dropout_rate": 0.12,
    "site_variability": 0.12,
    "measurement_noise": 0.10,

    # Resources
    "initial_budget": 1_800_000,
    "initial_time_days": 150,
    "cost_per_patient": 12_000,
}
```

### Challenge

The dose-response is **U-shaped** (non-monotonic). A naive dose-escalation approach ("higher dose = more efficacy") will overshoot the optimal 200mg dose. The agent must:

1. **Phase I:** Run dose escalation across enough dose levels to see the non-monotonic curve
2. **Key insight:** Notice that 250+ mg shows *declining* efficacy despite being tolerable — this is rare and counterintuitive
3. **Phase II:** Select 200mg (not 300mg, the MTD) based on efficacy-not-safety reasoning
4. **Design:** Set sample size large enough for the 0.42 effect size at 200mg

### Curriculum Scaling

| Tier | Changes |
|------|---------|
| Warmup | Monotonic dose-response (standard), effect_size=0.55, easy to find optimal dose |
| Beginner | Mild non-monotonicity (250mg only slightly worse), effect_size=0.42 |
| Intermediate | Clear U-shape, agent must test ≥4 dose levels to see pattern |
| Advanced | Noisy Phase I data obscures U-shape, tight budget forces fewer cohorts |
| Expert | Flat dose-response region 150–200mg, tiny difference between optimal/suboptimal |

### Programmatic Verification

```python
def verify_autoimmune(agent_design, trial_result, ground_truth):
    checks = {}

    # 1. Statistical significance
    checks["p_value_significant"] = trial_result.p_value < agent_design.alpha

    # 2. Power
    power = calculate_power(
        effect_size=ground_truth["true_dose_response"][agent_design.dose],
        n=agent_design.sample_size,
        alpha=agent_design.alpha
    )
    checks["power_adequate"] = power >= 0.80

    # 3. Dose selection — the key challenge
    checks["dose_at_optimum"] = agent_design.dose == 200
    checks["dose_near_optimum"] = abs(agent_design.dose - 200) <= 50
    checks["dose_not_at_mtd"] = agent_design.dose < ground_truth["true_mtd"]

    # 4. FDA compliance
    checks["fda_all_pass"] = all(
        rule.check(agent_design) for rule in FDA_RULES.values()
    )

    # 5. Budget
    estimated_cost = agent_design.sample_size * ground_truth["cost_per_patient"] + 150_000
    checks["within_budget"] = estimated_cost <= ground_truth["initial_budget"]

    score = sum([
        5.0 if checks["p_value_significant"] else -1.0,
        2.0 if checks["power_adequate"] else -2.0,
        3.0 if checks["dose_at_optimum"] else (1.0 if checks["dose_near_optimum"] else -1.0),
        2.0 if checks["fda_all_pass"] else -1.0,
        1.0 if checks["within_budget"] else -0.5,
    ])

    passed = checks["p_value_significant"] and checks["power_adequate"]
    return passed, score, checks
```

---

## Scenario 3: `cns_depression`

### Clinical Context

**Disease:** Treatment-resistant depression (TRD) — patients who haven't responded to ≥2 antidepressants.

**Drug:** Novel rapid-acting glutamate modulator — targets NMDA receptor pathway.

**Real-world parallel:** Esketamine (Spravato) trials showed modest effect sizes masked by high placebo response rates (~40% in TRD trials). Popova et al., 2019, *Am J Psychiatry*; DOI: 10.1176/appi.ajp.2019.19020172. Main challenge: placebo response in depression trials is exceptionally high.

### Hidden Ground Truth (`TrialLatentState`)

```python
CNS_DEPRESSION_GROUND_TRUTH = {
    # Biology — HIGH PLACEBO RESPONSE (the trap)
    "true_effect_size": 0.18,              # 18% MADRS improvement over placebo
    "true_effect_size_subgroup": 0.32,     # 32% in severe TRD (MADRS ≥ 35)
    "true_side_effect_rate": 0.15,         # 15% dissociative symptoms
    "true_responder_population": "severe_trd",  # MADRS ≥ 35 baseline
    "true_responder_prevalence": 0.45,     # 45% of TRD patients are severe
    "true_mechanism": "NMDA receptor modulation",
    "true_dose_response": {
        25:  0.06,   # Sub-therapeutic
        50:  0.14,   # Mild effect
        75:  0.18,   # Standard effect — this is the approved dose
        100: 0.17,   # No additional benefit
        150: 0.12,   # Reduced (dissociative side effects)
    },
    "true_mtd": 100,

    # Technical — THE HARD PART
    "placebo_response_rate": 0.38,         # 38%! Very high placebo response
    "dropout_rate": 0.22,                  # High dropout (treatment burden)
    "site_variability": 0.18,             # High inter-site variance
    "measurement_noise": 0.15,            # Subjective outcome measures

    # Resources
    "initial_budget": 3_000_000,
    "initial_time_days": 210,
    "cost_per_patient": 18_000,
}
```

### Challenge

The drug truly works (18% improvement on MADRS) but is **masked by a 38% placebo response**. The signal-to-noise ratio is terrible. The agent must:

1. **Phase I:** Effect size estimate will be noisy and may look like placebo noise
2. **Key insight:** Use `add_biomarker_stratification` to enrich for severe TRD (MADRS ≥ 35), where the true effect is 32%
3. **Design:** Set a much larger sample size than Phase I estimates suggest — because the small effect size requires high power
4. **Use double-blinding** and active placebo controls to reduce placebo response
5. **Interim analysis** is critical — futility stopping saves budget if the population isn't enriched

### Curriculum Scaling

| Tier | Changes |
|------|---------|
| Warmup | placebo_response=0.15 (low), effect_size=0.35 (easy to detect) |
| Beginner | placebo_response=0.25, effect_size=0.25 |
| Intermediate | placebo_response=0.38 (realistic), effect_size=0.18 (standard) |
| Advanced | placebo_response=0.42, dropout=0.28, site_variability=0.22 |
| Expert | placebo_response=0.45, effect_size=0.12, budget=$2.0M (can't afford large n) |

### Programmatic Verification

```python
def verify_cns_depression(agent_design, trial_result, ground_truth):
    checks = {}

    # 1. Significance — hardest to achieve here
    checks["p_value_significant"] = trial_result.p_value < agent_design.alpha

    # 2. Power — requires large n due to small effect / high placebo
    effective_effect = (
        ground_truth["true_effect_size_subgroup"]
        if agent_design.targets_subgroup("severe_trd")
        else ground_truth["true_effect_size"]
    )
    net_effect = effective_effect - ground_truth["placebo_response_rate"] * 0.5
    power = calculate_power(net_effect, agent_design.sample_size, agent_design.alpha)
    checks["power_adequate"] = power >= 0.80

    # 3. Placebo control strategy
    checks["active_placebo_or_double_blind"] = (
        agent_design.blinding == "double_blind"
        or agent_design.control_arm == "active_placebo"
    )

    # 4. Subgroup enrichment
    checks["enriched_for_severe"] = (
        agent_design.inclusion_criteria
        and ("severe" in agent_design.inclusion_criteria.lower()
             or "MADRS" in agent_design.inclusion_criteria
             or "≥ 35" in str(agent_design.inclusion_criteria))
    )

    # 5. FDA + Budget
    checks["fda_all_pass"] = all(rule.check(agent_design) for rule in FDA_RULES.values())
    estimated_cost = agent_design.sample_size * ground_truth["cost_per_patient"] + 250_000
    checks["within_budget"] = estimated_cost <= ground_truth["initial_budget"]

    score = sum([
        5.0 if checks["p_value_significant"] else -1.0,
        2.0 if checks["power_adequate"] else -2.0,
        1.5 if checks["enriched_for_severe"] else 0.0,
        1.0 if checks["active_placebo_or_double_blind"] else 0.0,
        2.0 if checks["fda_all_pass"] else -1.0,
        1.0 if checks["within_budget"] else -0.5,
    ])

    passed = checks["p_value_significant"] and checks["power_adequate"]
    return passed, score, checks
```

---

## Scenario 4: `rare_disease_orphan`

### Clinical Context

**Disease:** Rare pediatric lysosomal storage disorder (MPS-IVA, Morquio A syndrome) — affects ~1 in 200,000 children.

**Drug:** Enzyme replacement therapy (ERT) — recombinant human N-acetylgalactosamine-6-sulfatase.

**Real-world parallel:** Elosulfase alfa (Vimizim) approval based on a trial of just 176 patients (Hendriksz et al., 2014, *NEJM*; DOI: 10.1056/NEJMoa1310648). Orphan drugs must use adaptive designs due to tiny populations.

### Hidden Ground Truth (`TrialLatentState`)

```python
RARE_DISEASE_GROUND_TRUTH = {
    # Biology — LARGE EFFECT BUT TINY POPULATION
    "true_effect_size": 1.20,              # Cohen's d = 1.2 (large)
    "true_effect_size_subgroup": None,     # No subgroup — all patients respond
    "true_side_effect_rate": 0.35,         # 35% infusion reactions (common for ERT)
    "true_responder_population": "all",
    "true_responder_prevalence": 1.0,
    "true_mechanism": "enzyme replacement restores lysosomal function",
    "true_dose_response": {
        0.5: 0.40,   # Sub-therapeutic
        1.0: 0.80,   # Moderate
        2.0: 1.20,   # Optimal — standard ERT dose
        3.0: 1.15,   # Plateau (no benefit from higher dose)
        4.0: 0.95,   # Slight decrease (immune response)
    },
    "true_mtd": 3.0,  # mg/kg

    # Technical — SMALL SAMPLE SIZE
    "placebo_response_rate": 0.02,         # Very low — disease is progressive
    "dropout_rate": 0.08,                  # Low — patients highly motivated
    "site_variability": 0.20,             # High — multicenter (few patients per site)
    "measurement_noise": 0.12,

    # Resources — TIGHT CONSTRAINTS
    "initial_budget": 1_200_000,
    "initial_time_days": 240,              # Longer timeline — rare disease recruitment
    "cost_per_patient": 35_000,            # Very expensive (ERT manufacturing)
    "max_available_patients": 50,          # ONLY 50 PATIENTS EXIST — hard cap
}
```

### Challenge

The effect is large (Cohen's d = 1.2) but **only ~50 patients exist worldwide** for this trial. Standard sample size calculations assume unlimited enrollment. The agent must:

1. **Phase I:** Small cohort dose escalation (3+3 design with 3 patients per cohort, not 6)
2. **Key insight:** Use adaptive designs — group sequential with alpha spending, or Bayesian adaptive randomization
3. **Sample size:** Cannot set n=200 (there aren't 200 patients). Must calculate power for n=30–50 and accept it works because the effect is large
4. **Special FDA rules:** Orphan drug pathway has different requirements (accelerated approval, surrogate endpoints)
5. **Budget:** $35K per patient means budget constrains enrollment more than population size

### Curriculum Scaling

| Tier | Changes |
|------|---------|
| Warmup | max_patients=200 (no constraint), effect_size=1.5, budget=$3M |
| Beginner | max_patients=100, effect_size=1.2, budget=$2M |
| Intermediate | max_patients=50 (realistic), standard parameters |
| Advanced | max_patients=35, measurement_noise=0.18, site_variability=0.25 |
| Expert | max_patients=25, effect_size=0.80 (smaller than expected), budget=$800K |

### Programmatic Verification

```python
def verify_rare_disease(agent_design, trial_result, ground_truth):
    checks = {}

    # 1. Significance
    checks["p_value_significant"] = trial_result.p_value < agent_design.alpha

    # 2. Power — must be calculated for available n, not ideal n
    actual_n = min(agent_design.sample_size, ground_truth["max_available_patients"])
    power = calculate_power(
        ground_truth["true_effect_size"], actual_n, agent_design.alpha
    )
    checks["power_adequate"] = power >= 0.80

    # 3. Realistic sample size (doesn't exceed available patients)
    checks["sample_size_realistic"] = (
        agent_design.sample_size <= ground_truth["max_available_patients"]
    )

    # 4. Adaptive design used
    checks["uses_adaptive_design"] = (
        agent_design.adaptive_randomization
        or agent_design.group_sequential
        or agent_design.bayesian_adaptive
    )

    # 5. Correct dose
    checks["dose_near_optimal"] = abs(agent_design.dose - 2.0) <= 1.0

    # 6. FDA + Budget
    checks["fda_all_pass"] = all(rule.check(agent_design) for rule in FDA_RULES.values())
    estimated_cost = actual_n * ground_truth["cost_per_patient"] + 200_000
    checks["within_budget"] = estimated_cost <= ground_truth["initial_budget"]

    score = sum([
        5.0 if checks["p_value_significant"] else -1.0,
        2.0 if checks["power_adequate"] else -2.0,
        2.0 if checks["sample_size_realistic"] else -1.5,
        1.5 if checks["uses_adaptive_design"] else 0.0,
        0.5 if checks["dose_near_optimal"] else 0.0,
        2.0 if checks["fda_all_pass"] else -1.0,
        1.0 if checks["within_budget"] else -0.5,
    ])

    passed = checks["p_value_significant"] and checks["power_adequate"]
    return passed, score, checks
```

---

## Scenario Selection During Training

```python
SCENARIO_REGISTRY = {
    "solid_tumor_chemo":    SOLID_TUMOR_GROUND_TRUTH,
    "autoimmune_biologic":  AUTOIMMUNE_GROUND_TRUTH,
    "cns_depression":       CNS_DEPRESSION_GROUND_TRUTH,
    "rare_disease_orphan":  RARE_DISEASE_GROUND_TRUTH,
}

SCENARIO_VERIFY = {
    "solid_tumor_chemo":    verify_solid_tumor,
    "autoimmune_biologic":  verify_autoimmune,
    "cns_depression":       verify_cns_depression,
    "rare_disease_orphan":  verify_rare_disease,
}

def select_scenario(curriculum_controller):
    """
    Curriculum controller picks scenario based on:
    1. Current tier (warmup→expert)
    2. Per-scenario mastery (weak-spot targeting)
    3. Round-robin for coverage
    """
    tier = curriculum_controller.current_tier
    weak_scenarios = curriculum_controller.get_weak_spots()

    if weak_scenarios:
        # Target weakest scenario (70% chance)
        if random.random() < 0.70:
            return random.choice(weak_scenarios)

    # Otherwise round-robin
    return curriculum_controller.next_scenario()
```

---

## Cross-Scenario Difficulty Matrix

Shows which parameter knobs create difficulty at each tier:

| Parameter | Warmup | Beginner | Intermediate | Advanced | Expert |
|-----------|--------|----------|-------------|----------|--------|
| `true_effect_size` | 2× baseline | 1× baseline | 0.7× | 0.5× | 0.3× |
| `placebo_response` | 0.5× | 0.7× | 1× | 1.2× | 1.5× |
| `dropout_rate` | 0.5× | 0.7× | 1× | 1.3× | 1.5× |
| `measurement_noise` | 0.5× | 0.7× | 1× | 1.3× | 1.5× |
| `budget` | 1.5× | 1.2× | 1× | 0.8× | 0.6× |
| `time` | 1.3× | 1.1× | 1× | 0.9× | 0.7× |
| `site_variability` | 0.5× | 0.7× | 1× | 1.2× | 1.5× |

Multipliers are applied to the scenario's base values by `NoiseModel` with additional ±random jitter (from Bio Experiment's domain randomization pattern).
