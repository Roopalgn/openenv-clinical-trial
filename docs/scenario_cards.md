# Scenario Cards — Clinical Trial Designer

## Overview

| # | Scenario ID | Disease | Challenge | Budget | Time |
|---|-------------|---------|-----------|--------|------|
| 1 | `solid_tumor_chemo` | Non-small cell lung cancer | Find EGFR+ subgroup | $2.5M | 180 days |
| 2 | `autoimmune_biologic` | Rheumatoid arthritis | U-shaped dose-response | $1.8M | 150 days |
| 3 | `cns_depression` | Treatment-resistant depression | High placebo masks effect | $3.0M | 210 days |
| 4 | `rare_disease_orphan` | Rare pediatric metabolic disorder | Tiny n, adaptive design | $1.2M | 240 days |

Each scenario runs at all 5 curriculum tiers via `NoiseModel` scaling.

---

## Scenario 1: `solid_tumor_chemo`

**Drug:** Novel 3rd-gen EGFR tyrosine kinase inhibitor for NSCLC.
**Real-world parallel:** Osimertinib (Tagrisso) — 18.9 months PFS in EGFR-mutant vs 10.2 months placebo (Soria et al., 2018, NEJM).

**Hidden ground truth:**
- `true_effect_size`: 0.31 (31% PFS improvement overall)
- `true_effect_size_subgroup`: 0.58 (58% in EGFR+ patients)
- `true_responder_population`: "EGFR+" (35% prevalence)
- `true_dose_response`: 50mg→0.08, 100mg→0.19, **150mg→0.31** (optimal), 200mg→0.29, 250mg→0.22
- `true_mtd`: 200mg
- `placebo_response_rate`: 0.05, `dropout_rate`: 0.15

**Challenge:** Drug works overall (31%) but **dramatically better in EGFR+ patients** (58%). Agent must:
1. Phase I: Dose escalation → discover 150mg optimal, 200mg MTD
2. Use `add_biomarker_stratification` to discover 58% EGFR+ effect
3. Phase II: Set inclusion to EGFR+ → much better power with fewer patients

**Curriculum scaling:** Warmup has effect=0.50, no subgroup. Expert has effect=0.15, EGFR+ prevalence=20%, budget=$1.5M.

---

## Scenario 2: `autoimmune_biologic`

**Drug:** Novel IL-6 receptor antagonist for rheumatoid arthritis.
**Real-world parallel:** Tocilizumab (Actemra) — non-linear dose-response (Smolen et al., 2008, Lancet).

**Hidden ground truth:**
- `true_effect_size`: 0.42 (42% ACR50 improvement at optimal dose)
- `true_responder_population`: "all" (no hidden subgroup — dose is the challenge)
- `true_dose_response`: 50→0.12, 100→0.28, 150→0.38, **200→0.42** (optimal), 250→0.35, 300→0.22, 400→0.10
- `true_mtd`: 300mg
- `placebo_response_rate`: 0.15, `dropout_rate`: 0.12

**Challenge:** U-shaped dose-response. Higher dose ≠ more efficacy. Agent must:
1. Test enough dose levels to see the non-monotonic curve
2. Select 200mg (not MTD 300mg) based on efficacy reasoning
3. This is counterintuitive — most naive strategies overshoot

**Curriculum scaling:** Warmup has monotonic dose-response. Expert has flat region 150–200mg with tiny difference.

---

## Scenario 3: `cns_depression`

**Drug:** Novel rapid-acting glutamate modulator (NMDA pathway) for treatment-resistant depression.
**Real-world parallel:** Esketamine (Spravato) — modest effect masked by ~40% placebo response (Popova et al., 2019, Am J Psychiatry).

**Hidden ground truth:**
- `true_effect_size`: 0.18 (18% MADRS improvement)
- `true_effect_size_subgroup`: 0.32 (32% in severe TRD, MADRS ≥ 35)
- `true_responder_population`: "severe_trd" (45% prevalence)
- `true_dose_response`: 25→0.06, 50→0.14, **75→0.18** (approved dose), 100→0.17, 150→0.12
- `placebo_response_rate`: **0.38** (very high — the trap)
- `dropout_rate`: 0.22, `measurement_noise`: 0.15

**Challenge:** Drug works but 38% placebo response **masks the signal**. Agent must:
1. Recognize that Phase I effect estimates will be noisy and small
2. Enrich for severe TRD (MADRS ≥ 35) where effect is 32%
3. Use double-blinding and large sample size
4. Interim analysis critical — futility stops save budget if not enriched

**Curriculum scaling:** Warmup has placebo=0.15, effect=0.35. Expert has placebo=0.45, effect=0.12, budget=$2.0M.

---

## Scenario 4: `rare_disease_orphan`

**Drug:** Enzyme replacement therapy for Morquio A syndrome (~1 in 200K children).
**Real-world parallel:** Elosulfase alfa (Vimizim) — approved from trial of just 176 patients (Hendriksz et al., 2014, NEJM).

**Hidden ground truth:**
- `true_effect_size`: 1.20 (Cohen's d — large)
- `true_responder_population`: "all" (all patients respond)
- `true_dose_response`: 0.5mg/kg→0.40, 1.0→0.80, **2.0→1.20** (optimal), 3.0→1.15, 4.0→0.95
- `max_available_patients`: **50** (hard cap)
- `cost_per_patient`: $35,000 (very expensive)
- `placebo_response_rate`: 0.02, `dropout_rate`: 0.08

**Challenge:** Large effect but **only ~50 patients exist worldwide**. Agent must:
1. Phase I with tiny cohorts (3 per cohort, not 6)
2. Use adaptive Bayesian design — group sequential with alpha spending
3. Accept n=30–50 (power works because effect is large)
4. Navigate orphan drug FDA pathway (different requirements)

**Curriculum scaling:** Warmup has max_patients=200. Expert has max_patients=30, budget=$800K.

---

## Verification

All scenarios use the same objective verification:
1. **Statistical significance** — `scipy.stats` p-value
2. **Power adequacy** — `calculate_power(effect_size, n, alpha)` ≥ 0.80
3. **Subgroup/dose identification** — match against hidden ground truth
4. **FDA compliance** — rule engine binary pass/fail
5. **Budget** — arithmetic cost check
