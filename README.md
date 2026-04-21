# OpenEnv Clinical Trial Designer

### Can a small LLM learn to design a clinical trial — from scratch?

We gave a tiny language model a drug with an unknown effect, a budget, a deadline, and zero medical training. No textbook examples. No few-shot prompts. Just a PagerDuty-style alert: *"Design a Phase II trial for a novel EGFR inhibitor in non-small cell lung cancer."*

Within 40 episodes, it learned to run dose escalation, read safety signals, identify the hidden responder subgroup (EGFR+ patients), calculate statistical power, and submit an FDA-compliant protocol. By episode 8, it was designing trials that barely reached significance. By episode 40, it was hitting p < 0.005 with 85% power.

This is **OpenEnv Clinical Trial Designer** — an RL environment where an agent learns to design clinical trials that detect true drug effects under realistic constraints, using GRPO training against a simulator with hidden ground truth.

Built with [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv/tree/v0.2.1) | Theme #3.1: Professional Tasks

## Motivation

Clinical trial design is a high-stakes professional task where:

- **Mistakes are expensive**: A poorly designed trial wastes $100M+ and years of time
- **Verification is objective**: Either the trial detects the true effect (p < 0.05 with power ≥ 0.80) or it doesn't — no LLM judge needed
- **The world is partially observable**: True effect size, responder subgroup, safety profile, and dose-response curve are all hidden from the agent
- **Planning is long-horizon**: A full trial spans 55-100 steps across Phase I → Phase II → regulatory → analysis
- **Real rules exist**: FDA ICH E9 guidelines are codified as hard constraints — not suggestions

No existing OpenEnv environment covers clinical trial design. This combines real statistical tools (scipy.stats), real regulatory rules (FDA compliance engine), and a trial simulator with hidden ground truth — the same pattern that won the previous hackathon.

## The Environment

### Hidden World (TrialLatentState)

On each `reset()`, the simulator secretly sets:

- `true_effect_size` — the real drug effect (agent must discover this)
- `true_side_effect_rate` — real adverse event rate
- `true_responder_population` — e.g., "EGFR+ only" (agent doesn't know this)
- `placebo_response_rate` — background noise that masks the signal
- `dropout_rate` — patients who leave the trial
- Budget and time constraints (randomized ±30% / ±20% per episode)

The agent never sees these values directly. It must design experiments to discover them.

### Agent Actions (TrialAction)

The agent has 19 discrete action types across 5 categories:

| Phase | Actions | Real-World Analog |
|---|---|---|
| Design | set_primary_endpoint, set_sample_size, set_inclusion_criteria, set_exclusion_criteria, set_dosing_schedule, set_control_arm, set_randomization_ratio, set_blinding | Standard trial protocol components |
| Phase I | run_dose_escalation, observe_safety_signal, estimate_effect_size | 3+3 dose escalation, safety monitoring |
| Phase II | run_interim_analysis, modify_sample_size, add_biomarker_stratification | Adaptive design, enrichment |
| Regulatory | submit_to_fda_review, request_protocol_amendment | FDA compliance check |
| Analysis | run_primary_analysis, synthesize_conclusion | Final statistical test, trial conclusion |

### Verification (No LLM Judge Needed)

1. **Statistical power** — `scipy.stats` calculates power from effect size and sample size (pure math)
2. **FDA compliance** — hard rule engine checks 6+ codified constraints (binary pass/fail)
3. **Trial simulation** — runs the designed trial against hidden ground truth, returns p-value and confidence interval
4. **Budget check** — cost = n_patients × cost_per_patient + site_costs (arithmetic)

### Reward Structure

**Per-step rewards** (decomposed, 8 components):

| Key | What It Measures | Weight | Verification |
|---|---|---|---|
| `r_validity` | FDA rule compliance | 0.8 | Rule engine |
| `r_ordering` | Correct phase workflow | 1.0 | Phase detection heuristic |
| `r_info_gain` | Information gained from experiments | 1.2 | Bayesian update quality |
| `r_efficiency` | Budget and time efficiency | 0.6 | Math |
| `r_novelty` | Exploring new action types | +0.1 | Action history check |
| `r_penalty` | Soft constraint violations | -0.15/each | Rule engine |
| `r_terminal_success` | Trial detects true effect | +5.0 to +8.0 | Simulation math |
| `r_terminal_calibration` | Agent's conclusions match hidden truth | +3.0 to +4.0 | Ground truth comparison |

**Reward variance for GRPO**: Successful trials score +8 to +14, failed trials score -2 to -3.

### Curriculum (5 tiers)

| Tier | Difficulty | What Changes |
|---|---|---|
| Warmup | 0.0–0.25 | Large effect size, homogeneous population |
| Beginner | 0.25–0.40 | Medium effect, some noise |
| Intermediate | 0.40–0.60 | Small effect, enrichment needed |
| Advanced | 0.60–0.80 | Hidden responder subgroup, misleading Phase I signals |
| Expert | 0.80–0.95 | Tiny effect, high dropout, adaptive design required |

### Scenarios (4 initial)

| ID | Disease | Challenge | True Effect |
|---|---|---|---|
| `solid_tumor_chemo` | Non-small cell lung cancer | Find EGFR+ subgroup | 31% PFS improvement in EGFR+ only |
| `autoimmune_biologic` | Rheumatoid arthritis | U-shaped dose-response | 200mg optimal (not max dose) |
| `cns_depression` | Treatment-resistant depression | High placebo response masks effect | 18% improvement over placebo |
| `rare_disease_orphan` | Rare pediatric metabolic disorder | Tiny n, adaptive design required | Cohen's d = 1.2 but n < 50 |

## Tasks

The environment tests the agent on increasingly difficult clinical trial design tasks:

1. **Basic trial design** (Warmup): Design a straightforward Phase II trial with large effect size and homogeneous population. Agent must learn the basic workflow: dose escalation → safety check → effect estimate → protocol design → FDA submission → analysis.

2. **Population enrichment** (Intermediate): Identify a hidden responder subgroup from noisy Phase I data and enrich the Phase II population to boost statistical power.

3. **Adaptive design** (Advanced): Use interim analysis and sample size re-estimation when the initial design is underpowered. Requires alpha-spending for multiple looks.

4. **Rare disease** (Expert): Design a trial with fewer than 50 eligible patients. Must use adaptive Bayesian designs, relaxed endpoints, and creative statistical approaches.

## Setup

```bash
# Clone and install
git clone https://github.com/Roopalgn/openenv-clinical-trial.git
cd openenv-clinical-trial
pip install -e .

# Run environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/ping
```

## Docker

```bash
docker build -t clinical-trial-env .
docker run -p 8000:8000 clinical-trial-env
```

## Training

```bash
# GRPO training (requires GPU)
python train.py --vllm-mode colocate --num-generations 8

# Evaluation
python eval_compare.py

# Plot reward curves
python plot_rewards.py
```

## Documentation

**Design & Specs:**
- [Architecture & System Diagram](ARCHITECTURE.md)
- [Problem Statement & Judging Alignment](docs/problem_statement.md)
- [Reward Decomposition Spec](docs/reward_spec.md)
- [Scenario Cards (4 scenarios with ground truth)](docs/scenario_cards.md)
- [Phase Workflow & Scoring](docs/phase_workflow.md)
- [Curriculum Progression Policy](docs/curriculum_policy.md)
- [Adaptive Difficulty (G4)](docs/adaptive_difficulty_spec.md)
- [Multi-Layer Verification Spec](docs/verification_spec.md)

**Training & Evaluation:**
- [Training Runbook (GRPO config + procedure)](docs/training_runbook.md)
- [Evaluation Report Template](docs/evaluation_report_template.md)
- [Benchmark Protocol (baselines)](docs/benchmark_protocol.md)
- [Dashboard Metrics Format](docs/dashboard_metrics.md)

**Storytelling & Pitch:**
- [Demo Story Arc (3-min pitch)](docs/story_arc.md)
- [Pitch Notes (judge-aligned)](docs/pitch_notes.md)
- [Storytelling Assets (before/after episodes)](docs/storytelling_assets.md)
- [HF Mini-Blog Draft](docs/mini_blog_draft.md)
- [Evaluation Criteria & Metrics](docs/evaluation_criteria.md)

**Reference:**
- [Detailed Roadmap](docs/ROADMAP.md)
- [Winner Comparison & Intelligence](docs/comparison.md)
- [Knowledge Base](docs/KnowledgeBase.md)
- [Milestone Map](docs/milestone_map.md)

## Live Dashboard

Open `dashboard.html` in a browser for a 6-panel demo dashboard with simulated training data:
- Episode replay with phase-colored timeline
- Reward curves with rolling average and tier markers
- Curriculum progression bar
- Per-scenario success breakdown
- Agent capability radar (trained vs baseline)
- Action log

Connects to the environment server's WebSocket for live updates during training.

## Team

- **Roopal Guha Neogi** — Environment design, reward engineering, docs, storytelling
- **Suyash Kumar** — Environment implementation, training pipeline, evaluation scripts

Roadmap and push split: see `docs/ROADMAP.md`.
Merge to `main` only after both teammates approve checklist pass.

## Expected Baseline Scores

> Based on scripted policy evaluation at Warmup tier (50 episodes, seed 42). Actual training results will be added onsite April 25–26.

| Policy | Success Rate | Avg Reward | Avg Steps | Subgroup Found | Power ≥ 0.80 |
|--------|-------------|-----------|-----------|---------------|-------------|
| Random | ~5% | -1.5 ± 0.8 | 95 (timeout) | 2% | 3% |
| Scripted | ~40% | +2.8 ± 3.2 | 22 ± 6 | 0% (no enrichment) | 45% |
| Trained (expected) | ~75% | +8.5 ± 3.0 | 18 ± 5 | 60%+ | 80%+ |

**Key contrasts for the pitch:**
- Random agent times out 95% of episodes — it cannot learn workflow ordering from scratch
- Scripted agent follows correct workflow but never discovers the hidden EGFR+ subgroup
- Trained agent learns to enrich for subgroups, achieving 3× the reward with fewer steps

## On-Par-with-Winners Checklist

- Real consequences and hidden state.
- Multi-layer verification and anti-reward-hacking safeguards.
- Curriculum progression with mastery thresholds.
- Interpretable reward components.
- Clear storytelling with evidence of learning progression.
- Adaptive difficulty with weak-spot targeting (G4).
- Live demo dashboard for 3-minute pitch (G14).
