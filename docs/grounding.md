# Statistical Grounding and Validation

This document records the external references used to ground the environment's statistical assumptions and curriculum design.

## Why Grounding Exists

The environment uses programmatic verification. To keep those checks trustworthy, the trial-power assumptions are cross-checked against external sources used in clinical trial design and RL curriculum literature.

## Grounding Sources

1. rpact (R package, LGPL-3)
- Scope: group-sequential and adaptive confirmatory trial design, operating characteristics, and simulation support.
- Why used: reference implementation for confirmatory design workflows and boundary calibration.
- Validation signal: the package maintainers report an extensive automated test suite (approximately 39k checks).
- Link: https://cran.r-project.org/package=rpact

2. Berry et al. (2010), Bayesian Adaptive Methods for Clinical Trials
- Why used: practical adaptive-trial design patterns and Bayesian adaptation concepts.
- Link: https://www.taylorfrancis.com/books/mono/10.1201/9781439825480/bayesian-adaptive-methods-clinical-trials-scott-berry-brad-carlin-jack-lee-peter-muller

3. Wassmer and Brannath (2016), Group Sequential and Confirmatory Adaptive Designs in Clinical Trials
- Why used: confirmatory adaptive design, alpha control, and boundary interpretation.
- Link: https://link.springer.com/book/10.1007/978-3-319-32562-0

4. Narvekar et al. (2020), Curriculum Learning for Reinforcement Learning Domains (JMLR)
- Why used: curriculum progression and weak-spot targeting rationale.
- Link: https://www.jmlr.org/papers/v21/20-212.html

## Validation Data Artifact

Precomputed scenario tables are stored at:
- server/grounding/rpact_validation.json

The JSON contains:
- fixed and group-sequential critical boundary values
- expected power grids per scenario across sample sizes
- minimum n estimate for target power (0.80)

## Method Notes

Power tables use the same approximation as server/simulator/power_calculator.py:

- Two-sided alpha: 0.05
- z_critical = Phi^{-1}(1 - alpha/2)
- ncp = d * sqrt(n/4)
- power = Phi(ncp - z_critical) + Phi(-ncp - z_critical)

Where d is Cohen's d and n is total sample size across both arms.

## Scenario-Level Summary

| Scenario | Effect size used for table | Approx n for 80% power | Comment |
|---|---:|---:|---|
| solid_tumor_chemo (overall) | 0.31 | 327 | General population signal is modest |
| solid_tumor_chemo (EGFR+) | 0.58 | 94 | Enrichment sharply improves power |
| autoimmune_biologic | 0.42 | 178 | Detectable with mid-range sample sizes |
| cns_depression (overall) | 0.18 | 969 | High-noise regime, very sample hungry |
| cns_depression (enriched severe TRD) | 0.32 | 307 | Enrichment is operationally important |
| rare_disease_orphan | 1.20 | 22 | Large effect compensates for tiny n |

## Optional rpact Reproduction Snippet (R)

```r
library(rpact)

# Two-look O'Brien-Fleming design example
x <- getDesignGroupSequential(
  kMax = 2,
  alpha = 0.05,
  sided = 2,
  typeOfDesign = "OF"
)
summary(x)

# Fixed-design power grid example
# (for equal-variance normal endpoint)
# Replace effect and n with scenario values.
getPowerMeans(
  groups = 2,
  thetaH1 = 0.31,
  stDev = 1,
  nFixed = c(80, 120, 160, 200, 260),
  alpha = 0.05,
  sided = 2
)
```

## Integration Points

- Referenced by ARCHITECTURE.md for system-level trust claims.
- Referenced by docs/internal/pitch_notes.md for judge-facing evidence.
- Consumed as static validation data via server/grounding/rpact_validation.json.
