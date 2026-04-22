# Demo Story Arc: From Blind to FDA-Ready


## The 3-Minute Pitch Structure

### Hook (15 seconds)

> "Can a 1.5B-parameter language model learn to design a clinical trial — from scratch? No medical training. No textbook examples. Just a drug with an unknown effect, a budget, and a deadline."

### Act 1: The Cold Start (45 seconds)

**Episode 1 — The agent knows nothing.**

The agent receives its first scenario: "Design a Phase II trial for a novel EGFR inhibitor in non-small cell lung cancer."

It has never designed a trial. It doesn't know what statistical power means, what FDA requires, or that the drug only works in EGFR+ patients. It guesses random parameters:

- Sets sample size to 20 (underpowered)
- Skips dose escalation entirely (Phase I violation)
- Picks overall survival as endpoint without justification
- Submits to FDA review — rejected for 4 protocol violations
- Trial simulation runs: p = 0.41 (fails to detect effect)

**Reward: -2.5** — timeout with FDA rejection.

*Show: reward curve starting negative, scatter of failed episodes.*

### Act 2: First Light (60 seconds)

**Episode 8 — Something clicks.**

The agent discovers the dose escalation → safety signal → effect estimate workflow. It runs Phase I properly for the first time, reads the safety data, estimates an effect size.

But it still enrolls the general population. The true effect is concentrated in EGFR+ patients (hidden subgroup). With the diluted signal, the trial barely reaches significance.

- Phase I complete in correct order
- Statistical power = 0.65 (still underpowered)
- FDA compliance: 3/5 rules pass
- Trial simulation: p = 0.048 (barely significant, fragile)

**Reward: +3.2** — first positive episode.

*Show: reward curve crossing zero, the "aha" moment.*

### Act 3: The Environment Fights Back (45 seconds)

**Episode 20+ — Harder scenarios.**

As the agent masters simple scenarios (large-effect, homogeneous population), the curriculum escalates. Now it faces:

- `autoimmune_biologic`: U-shaped dose-response curve — standard increasing doses fail
- `cns_depression`: High placebo response masks a real but small drug effect
- `rare_disease_orphan`: Only 50 eligible patients — must use adaptive design

The agent must learn to:

- Stratify by biomarkers (find the hidden responder population)
- Use adaptive sample size re-estimation
- Apply correct alpha-spending for interim analyses

*Show: curriculum progression chart (warmup → beginner → intermediate).*

### Act 4: Mastery (30 seconds)

**Episode 40 — Systematic design.**

The trained agent receives `solid_tumor_chemo`:

1. Runs dose escalation across 6 cohorts (correct Phase I)
2. Reads safety: identifies EGFR+ patients have 3x response rate
3. Sets inclusion criteria to EGFR+ (finds hidden subgroup!)
4. Calculates sample size for power = 0.85
5. Submits protocol — FDA compliance: 5/5 rules pass
6. Runs interim analysis at 50% enrollment — continues (not futile)
7. Final analysis: p = 0.003, CI = [0.15, 0.47]

**Reward: +11.2** — highest tier.

*Show: before/after episode side-by-side, final reward curve with clear upward trend.*

### Close (15 seconds)

> "The agent learned to go from blind guessing to FDA-compliant trial design. Not from medical textbooks — from reward signal alone. This is what happens when the environment is honest: real math, real rules, real consequences."

## Visual Assets Needed


1. **Reward curve**: per-episode scatter + rolling average + trend line (from `plot_rewards.py`)
   - *show 3 runs with different seeds → proves robustness*
2. **Before/after episode**: side-by-side terminal output of Episode 1 vs Episode 40
   - *Episode 1 blind commands vs Episode 7 systematic debugging*
3. **Curriculum chart**: bar chart showing tier progression over episodes
   - *warmup → beginner → intermediate → advanced → expert progression*
4. **Architecture diagram**: from `ARCHITECTURE.md`
5. **Dashboard screenshot**: live episode replay from `dashboard.html`
   - *dashboard.html + dashboard.py for live demo*
6. **Action diversity heatmap**:
   - Base model: bright rows on `set_sample_size`, `run_primary_analysis` (repetitive, 2–3 actions dominate)
   - Trained model: spread across 12–15 of 19 actions (genuine exploration)
   - *Visual proof of learning beyond reward — the agent explores more intelligently*
7. **Capability radar chart**:
   - 6 axes: trial success, phase compliance, FDA pass, action diversity, budget efficiency, subgroup identification
   - Base model: small polygon (low on all axes)
   - Trained model: expanded polygon (high on most axes)
   - *Shows "not just scoring higher — exploring more intelligently"*
8. **Component reward trends** (NEW — from Bio):
   - 8 subplots showing each reward component over training
   - Shows *which* skills improve: e.g., `r_ordering` improves first (learns phase sequence), then `r_info_gain` (learns to gather information), then `r_validity` (learns FDA rules)
   - *decomposed reward makes it clear which behaviors the agent learned*

## Key Storytelling Principles

- **Failure first**: Start with the agent failing badly — makes improvement dramatic
  - *Episode 1 reward +1.80 (random kubectl), Episode 4: +6.58 (learned discovery)*
- **Show the math**: Power calculations, p-values, FDA rules — prove the verification is objective
  - *Bio: prerequisite rules, ground-truth calibration, budget math*
- **Environment co-evolution**: If we find bugs through training, that's part of the story (this iterative refinement IS the demo)
  - *winning insight: "training exposed bugs in the command parser, judge truncation, and health check race conditions. Fixing them made both the environment and agent better."*
  - *If our simulator's noise model is too easy or too hard, discovering this through training IS the demo*
- **Numbers, not vibes**: "48% success rate vs 12% baseline" beats "the agent got better"
  - *Mean 3.48, Best 6.79, across 7 episodes*
- **Visual proof over text**: Heatmaps, radar charts, and reward curves are more convincing than bullet points
  - *Tool usage heatmaps and capability radar charts are more convincing than bullet points*
- **The "aha moment"**: Identify and highlight the specific episode where the agent does something smart for the first time
  - *Our version: the first time the agent sets `inclusion_criteria: "EGFR+"` — it discovered the hidden subgroup*