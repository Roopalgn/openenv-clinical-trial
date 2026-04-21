# Problem Statement & Judging Alignment

## Problem Statement

Design an OpenEnv RL environment where an LLM agent learns to design clinical trials that detect true drug effects under realistic constraints. The agent operates in a partially observable world: the true effect size, responder subgroup, safety profile, and dose-response curve are hidden. The agent must plan a multi-phase trial (Phase I safety → Phase II efficacy → regulatory submission → statistical analysis) that would succeed in a rigorous simulated trial execution.

This is a **professional world-modeling task** (Theme #3.1) — the agent interacts with real statistical tools, real FDA constraint rules, and a trial simulator that produces ground-truth outcomes from hidden parameters. No LLM judge is needed for the core success criterion: either the trial detects the true effect (p < 0.05 with power ≥ 0.80) or it doesn't.

## Why Clinical Trial Design

> **Domain credibility:** Bio Experiment won 2nd place partly because they grounded scenarios in real papers with DOIs and real biological ground truth. We ground our domain in published FDA statistics, ICH guidelines, and real clinical trial failure data.

- **Real professional task**: Clinical trial design is a high-stakes, multi-step professional workflow where mistakes cost billions of dollars and years of time.
  - Average cost of bringing a drug to market: **$2.6 billion** (DiMasi et al., 2016, *J Health Economics*)
  - Average development timeline: **10–15 years** (PhRMA, 2021)
  - Overall clinical trial failure rate: **~90%** (Wong et al., 2019, *Biostatistics*)
  - Phase II failure rate due to inadequate efficacy: **~57%** (Hwang et al., 2016, *JAMA Internal Medicine*)
- **Objective verification**: Trial success is determined by math (statistical power, p-values, confidence intervals) and codified rules (FDA ICH E9 guidelines) — not by LLM judgment.
  - ICH E9: *Statistical Principles for Clinical Trials* (1998) — the international standard for trial design
  - ICH E9(R1): *Estimands and Sensitivity Analysis in Clinical Trials* (2019) — modern addendum on handling intercurrent events
  - 21 CFR Part 312: *Investigational New Drug Application* — FDA regulatory framework
- **Partially observable**: The true drug effect, responder population, and safety profile are hidden. The agent must design experiments (dose escalation, interim analysis) to discover them.
- **Long-horizon planning**: A full trial spans 55–100 steps across Phase I, Phase II, regulatory, and analysis phases. Early decisions (endpoint choice, sample size) constrain later outcomes.
- **Progressive difficulty**: Easy scenarios have large effect sizes in homogeneous populations. Hard scenarios have hidden responder subgroups, misleading Phase I signals, and tiny effects masked by placebo response.

### No Existing RL Environment Covers This

Unlike K8s debugging (KubeSRE) or bioinformatics (Bio Experiment), there is **no existing OpenEnv environment for clinical trial design**. The closest academic work:
- **TrialGPT** (Jin et al., 2023): uses LLMs for patient-trial matching, but is a classification task, not RL.
- **Clinical trial optimization** (Berry et al., 2011): Bayesian adaptive designs, but formulated as statistical optimization, not agent interaction.
- **Our contribution**: First RL environment where an LLM learns the full trial design workflow through interaction with a hidden-state simulator.

## Alignment to Judging Criteria

> **Strategy insight:** KubeSRE won 40% innovation by being the first RL environment for real K8s debugging. Bio Experiment won 30% storytelling with "hidden biology revealed through systematic experiments." We target both.

### Environment Innovation (40%)

| What Judges Want | What We Deliver | Winner Precedent |
|---|---|---|
| Novel, creative, challenging environment | Clinical trial design is unexplored in RL — no existing OpenEnv environment covers it | KubeSRE: first RL for real K8s debugging |
| Meaningfully tests agent behavior | Agent must reason about hidden biology, statistical power, and regulatory constraints simultaneously | Bio: hidden DE genes + realistic noise |
| Real interaction with tools/APIs | Real scipy.stats power calculations, real FDA rule engine, real trial simulation — not mocked | KubeSRE: real kubectl against live cluster |
| Partially observable world | Hidden ground truth: true effect size, responder subgroup, safety profile, dose-response curve | Bio: POMDP with hidden ground truth |
| Consistent internal state | TrialLatentState maintains hidden truth; TrialState tracks agent's decisions and resource usage | Bio: TransitionEngine + OutputGenerator |
| Self-improving/adaptive | AdversarialDesigner at expert tier targets weak spots; curriculum co-evolves with agent | KubeSRE: Claude designs targeted incidents |

### Storytelling (30%)

| What Judges Want | What We Deliver | Winner Precedent |
|---|---|---|
| Clear problem explanation | "Can a small LLM learn to design a clinical trial from scratch?" | KubeSRE: "From blind to on-call" |
| Engaging demo | Story arc: failure → learning → success with before/after episode comparison | KubeSRE: Episode 1 (-2.0) → Episode 7 (+6.79) |
| Easy to follow | 3-act structure: blind start → discovers Phase I workflow → masters adaptive design | VRAM: pattern repetition vs real exploration |

### Showing Improvement in Rewards (20%)

| What Judges Want | What We Deliver | Winner Precedent |
|---|---|---|
| Observable training progress | Decomposed reward with 8 components — each independently trackable | Bio: 6-component decomposed reward |
| Reward curves and metrics | plot_rewards.py generates per-episode curves with rolling average and trend | KubeSRE: reward curves across 3 training runs |
| Before/after behavior | eval_compare.py runs base vs trained model side-by-side; capability radar chart | VRAM: capability profile expansion radar |
| Multiple metric types | Success rate, action diversity heatmap, phase compliance, FDA pass rate | VRAM: tool diversity, reward, loss dashboard |

### Reward and Training Script/Pipeline Setup (10%)

| What Judges Want | What We Deliver | Winner Precedent |
|---|---|---|
| Coherent reward logic | Decomposed: r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty + r_terminal | Bio: same decomposition pattern |
| Meaningful improvement in inference | GRPO with TRL, report success rate improvement and reward trend | KubeSRE: +1.80 → +6.79 |
| Training pipeline | train.py (GRPO + LoRA + vLLM colocate) + train_colab.ipynb (Colab notebook) | All winners: complete pipeline |
| Reproducibility | Seeded NoiseModel, episode transcripts JSONL, documented hyperparams | Bio: seeded numpy.Generator |

## Objectives

1. Build an OpenEnv-compatible environment (`reset`/`step`/`state`/`schema` API) served via FastAPI and deployed on HF Spaces.
2. Implement a clinical trial simulator with hidden ground truth (TrialLatentState) and programmatic verification (statistical power, FDA rules, trial simulation).
3. Design a decomposed reward function with 8 interpretable components.
4. Implement a 5-tier curriculum from warmup (large effects, easy population) to expert (tiny effects, hidden subgroups, adaptive design required).
5. Ship a complete training pipeline: train.py (GRPO), eval_compare.py, plot_rewards.py, train_colab.ipynb.
6. Demonstrate clear reward improvement over training with before/after episode comparison.
7. Tell a compelling story: "From blind guessing to FDA-ready trial design."

## Success Metrics

> **Inspired by:** KubeSRE's concrete results (Episode 1: +1.80, Episode 7: +6.79, Mean: 3.48). Bio's trajectory datasets. VRAM's shaped reward improvement (5.75 vs 4.74 = 21%). Setting measurable targets with specific verification methods.

| Metric | Target | Verification Method |
|---|---|---|
| Trial success rate (trained) | ≥ 60% on warmup, ≥ 30% on intermediate | eval_compare.py |
| Reward trend | Positive slope over training | plot_rewards.py trend line |
| Phase workflow compliance | Agent follows correct phase order ≥ 70% of the time | Phase-order scoring in reward |
| FDA rule pass rate | ≥ 80% of trials pass all hard constraints | Rule engine pass/fail log |
| Statistical power | Agent designs trials with power ≥ 0.80 in ≥ 50% of episodes | Power calculation in simulator |
| Curriculum advancement | Agent reaches intermediate tier within 30 episodes | Curriculum controller log |
| Reproducibility | Same seed → same episode outcome | NoiseModel seeded generator |
