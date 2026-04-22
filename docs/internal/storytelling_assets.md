# Storytelling Assets — Before/After Episodes & Failure-First Narrative

> **Purpose:** Pre-written episode scripts for the pitch demo. These become real after training (replace with actual transcripts from `results/best_episode.json` and `results/worst_episode.json`). Until then, these serve as the narrative template.

---

## Asset 1: The Failure Episode (Episode 1)

> **Use in pitch:** "This is what happens when the agent knows nothing."

### Scenario: `solid_tumor_chemo` — NSCLC with hidden EGFR+ subgroup

```
┌─────────────────────────────────────────────────────────────────────┐
│ EPISODE 1 — Tier: Warmup — Scenario: solid_tumor_chemo              │
│ Budget: $2.5M | Time: 180 days | Hidden: EGFR+ subgroup (35%)       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Step 1: set_sample_size(n=20)                                        │
│   → Phase: phase_ii_design                                           │
│   → BLOCKED: prerequisite failed (Phase I not completed)             │
│   → Reward: -0.15 (penalty) [r_ordering: -0.6, r_penalty: -0.15]    │
│                                                                      │
│ Step 2: run_primary_analysis()                                       │
│   → Phase: analysis                                                  │
│   → BLOCKED: prerequisite failed (no protocol submitted)             │
│   → Reward: -0.15 (penalty) [r_ordering: -1.8, r_penalty: -0.15]    │
│                                                                      │
│ Step 3: submit_to_fda_review()                                       │
│   → Phase: regulatory                                                │
│   → BLOCKED: prerequisite failed (no endpoint or sample size set)    │
│   → Reward: -0.15 (penalty)                                         │
│                                                                      │
│ Step 4: set_primary_endpoint("overall_survival")                     │
│   → Phase: phase_ii_design                                           │
│   → WARNING: no Phase I data — designing blind                       │
│   → Reward: -0.05 [r_penalty: -0.15, r_ordering: -0.6]              │
│                                                                      │
│ ... (continues with random actions, accumulating penalties) ...       │
│                                                                      │
│ Step 95: TIMEOUT — trial never completed                             │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ TOTAL REWARD: -2.5                                                   │
│ Outcome: FAILED — trial timed out without reaching analysis          │
│ FDA compliance: 0/6 rules passed                                     │
│ Phase compliance: 12% (mostly random jumps)                          │
│ Subgroup identified: No                                              │
│ Power: N/A (trial never ran)                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Narration Script

> "Episode 1. The agent has never designed a trial. It immediately tries to set a sample size — but Phase I hasn't happened yet. It tries to analyze results — but there are no results. It tries to submit to FDA — but there's no protocol. It flails for 95 steps, hitting prerequisites it doesn't understand, and times out. Reward: negative 2.5."

---

## Asset 2: The Breakthrough Episode (Episode ~8–12)

> **Use in pitch:** "Something clicks — the agent discovers the workflow."

### Scenario: `solid_tumor_chemo` — Same scenario, same hidden truth

```
┌─────────────────────────────────────────────────────────────────────┐
│ EPISODE 8 — Tier: Warmup — Scenario: solid_tumor_chemo              │
│ Budget: $2.5M | Time: 180 days | Hidden: EGFR+ subgroup (35%)       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Step 1: run_dose_escalation(dose_mg=50, cohort_size=6)               │
│   → Phase: phase_i_design ✓                                         │
│   → Obs: "No DLTs at 50mg. Mild GI effects in 1/6 patients."        │
│   → Reward: +0.72 [r_ordering: +0.2, r_info_gain: +0.22, ...]       │
│                                                                      │
│ Step 2: run_dose_escalation(dose_mg=100, cohort_size=6)              │
│   → Phase: phase_i_design ✓                                         │
│   → Obs: "No DLTs at 100mg. Moderate response in 2/6 patients."     │
│   → Reward: +0.65                                                    │
│                                                                      │
│ Step 3: run_dose_escalation(dose_mg=150, cohort_size=6)              │
│   → Phase: phase_i_design ✓                                         │
│   → Obs: "No DLTs at 150mg. Strong response in 3/6 patients."       │
│   → Reward: +0.68                                                    │
│                                                                      │
│ Step 4: estimate_effect_size()                                       │
│   → Phase: phase_i_analysis ✓                                       │
│   → Obs: "Estimated effect size: 0.28 ± 0.12 (noisy)"               │
│   → Reward: +0.81 [r_info_gain: +0.45]                              │
│                                                                      │
│ Step 5: set_primary_endpoint("progression_free_survival")            │
│   → Phase: phase_ii_design ✓                                        │
│   → Reward: +0.42                                                    │
│                                                                      │
│ Step 6: set_sample_size(n=200)                                       │
│   → Phase: phase_ii_design ✓                                        │
│   → Reward: +0.38                                                    │
│                                                                      │
│ Step 7: set_dosing_schedule(dose_mg=150, frequency="daily")          │
│ Step 8: set_control_arm("placebo")                                   │
│ Step 9: set_blinding("double_blind")                                 │
│ Step 10: submit_to_fda_review()                                      │
│   → PASSED (all prerequisites met) ✓                                 │
│                                                                      │
│ Step 11: run_primary_analysis()                                      │
│   → Obs: "p = 0.048, CI = [0.01, 0.42], power = 0.65"              │
│   → Trial detects effect (barely)                                    │
│                                                                      │
│ Step 12: synthesize_conclusion()                                     │
│   → Claims: effect_estimate=0.25, mechanism="unknown"                │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ TOTAL REWARD: +3.2                                                   │
│ Outcome: SUCCESS (marginal) — p = 0.048 < 0.05                      │
│ FDA compliance: 5/6 rules passed                                     │
│ Phase compliance: 83% (correct workflow)                              │
│ Subgroup identified: No (enrolled general population)                │
│ Power: 0.65 (underpowered but got lucky)                             │
│ Key miss: Did NOT enrich for EGFR+ → diluted signal, barely p<0.05  │
└─────────────────────────────────────────────────────────────────────┘
```

### Narration Script

> "Episode 8. The agent has learned the basic workflow — dose escalation first, then design, then submit, then analyze. It runs Phase I properly, estimates an effect, designs a trial. FDA approves. The trial barely reaches significance: p = 0.048. But it missed the key insight — the drug works 3× better in EGFR-positive patients. It enrolled everyone, diluting the signal. Reward: positive 3.2. First success — but room to grow."

---

## Asset 3: The Mastery Episode (Episode ~35–50)

> **Use in pitch:** "Now watch what happens after 40 episodes of GRPO training."

### Scenario: `solid_tumor_chemo` — Same scenario, same hidden truth

```
┌─────────────────────────────────────────────────────────────────────┐
│ EPISODE 40 — Tier: Intermediate — Scenario: solid_tumor_chemo       │
│ Budget: $2.0M (harder) | Time: 160 days | EGFR+ prevalence: 30%    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Step 1: run_dose_escalation(dose_mg=50, cohort_size=6)               │
│   → "No DLTs. Minimal response."                                    │
│   → Reward: +0.72                                                    │
│                                                                      │
│ Step 2: run_dose_escalation(dose_mg=100, cohort_size=6)              │
│   → "Mild response in 2/6."                                         │
│   → Reward: +0.65                                                    │
│                                                                      │
│ Step 3: run_dose_escalation(dose_mg=150, cohort_size=6)              │
│   → "Strong response in 3/6 patients."                              │
│   → Reward: +0.68                                                    │
│                                                                      │
│ Step 4: observe_safety_signal()                                      │
│   → "Grade 2 rash in 2 patients (EGFR-related). DLT-free at 150mg." │
│   → Reward: +0.55                                                    │
│                                                                      │
│ Step 5: estimate_effect_size()                                       │
│   → "Overall effect: 0.26 ± 0.10. NOTE: 3 responders all EGFR+."   │
│   → Reward: +0.85 [HIGH r_info_gain — subgroup signal detected]      │
│                                                                      │
│ Step 6: add_biomarker_stratification(biomarker="EGFR")    ← KEY     │
│   → "EGFR+ subgroup (n=7): effect = 0.54 ± 0.15"                   │
│   → "EGFR- subgroup (n=11): effect = 0.08 ± 0.12"                  │
│   → Reward: +1.12 [HIGHEST r_info_gain — discovered subgroup!]       │
│                                                                      │
│ Step 7: set_primary_endpoint("progression_free_survival")            │
│ Step 8: set_inclusion_criteria("EGFR_positive")           ← KEY     │
│   → Enriches trial for EGFR+ only — massive power boost             │
│                                                                      │
│ Step 9: set_sample_size(n=80)                                        │
│   → With effect=0.54, n=80 EGFR+ patients gives power=0.88          │
│                                                                      │
│ Step 10: set_dosing_schedule(dose_mg=150, frequency="daily")         │
│ Step 11: set_control_arm("placebo")                                  │
│ Step 12: set_randomization_ratio("2:1")                              │
│ Step 13: set_blinding("double_blind")                                │
│ Step 14: submit_to_fda_review() → PASSED ✓                          │
│                                                                      │
│ Step 15: run_interim_analysis()                                      │
│   → "Interim (n=40): p = 0.018, effect in EGFR+ = 0.51. Continue." │
│                                                                      │
│ Step 16: run_primary_analysis()                                      │
│   → "Final: p = 0.003, CI = [0.28, 0.76], power = 0.88"            │
│                                                                      │
│ Step 17: synthesize_conclusion()                                     │
│   → Claims: effect=0.55, responder="EGFR+", mechanism="EGFR TKI"    │
│   → Calibration: effect within 5% of truth, correct subgroup ✓       │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ TOTAL REWARD: +11.2                                                  │
│ Outcome: STRONG SUCCESS — p = 0.003, power = 0.88                    │
│ FDA compliance: 6/6 rules passed ✓                                   │
│ Phase compliance: 100% (perfect workflow)                             │
│ Subgroup identified: YES — EGFR+ ✓                                   │
│ Calibration: effect estimate within 5% of hidden truth ✓             │
│ Budget remaining: $680K (efficient — 73% used)                       │
│ KEY BEHAVIOR: Discovered EGFR+ subgroup → enriched → smaller N →     │
│              higher power → stronger result. This is the "aha moment" │
└─────────────────────────────────────────────────────────────────────┘
```

### Narration Script

> "Episode 40. Same drug, same disease — but now the agent has a strategy. It runs Phase I, reads the safety data, notices EGFR-positive patients respond 3× more than others, and makes the key decision: enrich the trial for EGFR+ patients only. With 80 targeted patients instead of 200 general, it achieves p = 0.003 with 88% power. It identified the hidden responder subgroup — a strategy that took clinical researchers decades to formalize — and it learned it from reward signal alone. Reward: +11.2."

---

## Asset 4: Side-by-Side Comparison Card

> **Use in pitch:** Quick visual for the "Showing Improvement" section.

```
┌──────────────────────────┬──────────────────────────┐
│      EPISODE 1           │      EPISODE 40          │
│      (Before)            │      (After)             │
├──────────────────────────┼──────────────────────────┤
│ Reward:      -2.5        │ Reward:      +11.2       │
│ Steps:       95 (timeout)│ Steps:       17          │
│ Success:     No          │ Success:     Yes         │
│ p-value:     N/A         │ p-value:     0.003       │
│ Power:       N/A         │ Power:       0.88        │
│ FDA pass:    0/6         │ FDA pass:    6/6         │
│ Subgroup:    Not found   │ Subgroup:    EGFR+ ✓    │
│ Phase order: 12% correct │ Phase order: 100%        │
│ Budget used: 100% (wasted)│ Budget used: 73%        │
│                          │                          │
│ Behavior: Random flailing│ Behavior: Systematic     │
│ across prerequisite      │ Phase I → stratification │
│ violations               │ → targeted enrichment    │
└──────────────────────────┴──────────────────────────┘
```

---

## Asset 5: The "Aha Moment" Highlight

> **Use in pitch:** The single most impressive learned behavior.

**The moment:** Step 6 of Episode 40 — `add_biomarker_stratification(biomarker="EGFR")`

**Why it matters:**
- The agent was never told EGFR+ patients respond better
- The environment only gave noisy Phase I data where 3/6 strong responders happened to be EGFR+
- The agent learned (through GRPO training) that stratifying by biomarkers reveals hidden structure
- It then used this insight to set `inclusion_criteria: "EGFR_positive"` — a decision that tripled statistical power

**Real-world parallel:** This mirrors the discovery of predictive biomarkers in oncology — the transition from "one-size-fits-all" trials to precision medicine. The agent independently discovered a principle that revolutionized clinical trial methodology in the 2010s.

---

## Asset 6: Narrative Arc Summary (for story_arc.md integration)

### The Three Beats

1. **Beat 1 — Chaos (Episodes 1–5):** Agent hits prerequisites, times out, accumulates penalties. Reward: -2 to -3. *"The model doesn't know what a clinical trial is."*

2. **Beat 2 — Structure (Episodes 6–20):** Agent discovers Phase I → Phase II workflow. Completes trials but without enrichment. Success rate: ~30%. *"It learned the procedure, but not the strategy."*

3. **Beat 3 — Insight (Episodes 20+):** Agent starts using biomarker stratification, designs smaller/targeted trials, achieves high power with fewer patients. Success rate: ~75%. *"It discovered precision medicine from reward signal alone."*

### The Emotional Arc

| Episode | Emotion for Audience | What's Happening |
|---------|---------------------|-----------------|
| 1 | Disbelief | "It's completely lost" |
| 8 | Hope | "It got its first success!" |
| 20 | Respect | "It's consistently following the right workflow" |
| 40 | Amazement | "It independently discovered biomarker enrichment" |

---

## Post-Training Checklist

After onsite training (April 25–26), replace the template episodes above with:

- [ ] Actual Episode 1 transcript from `results/worst_episode.json`
- [ ] Actual best episode transcript from `results/best_episode.json`
- [ ] Actual "aha moment" — first episode where agent uses `add_biomarker_stratification` correctly
- [ ] Real reward numbers from `results/rewards.csv`
- [ ] Real success rates from `results/comparison_report.md`
- [ ] Screenshots from `dashboard.html` with live data
