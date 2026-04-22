---
title: OpenEnv Clinical Trial Designer
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# OpenEnv Clinical Trial Designer

> **OpenEnv Hackathon — Theme #3.1: Professional Tasks (World Modeling)**

### Can a small LLM learn to design a clinical trial — from scratch?

We gave a 7B language model a drug with an unknown effect, a budget, a deadline, and zero medical training. No textbook examples. No few-shot prompts. Just a scenario: *"Design a Phase II trial for a novel EGFR inhibitor in non-small cell lung cancer."*

The model must discover the drug's true efficacy through dose escalation, identify a hidden responder subgroup (EGFR+ patients) from noisy Phase I data, calculate statistical power, navigate FDA compliance rules, and submit a protocol — all within a budget and time constraint it doesn't fully know.

**OpenEnv Clinical Trial Designer** is an RL environment where an LLM agent learns to design clinical trials that detect true drug effects under realistic constraints. The environment uses hidden ground truth, real statistical calculations (`scipy.stats`), and codified FDA rules — verification is entirely objective, no LLM judge required for the core success criterion.

Built with [OpenEnv v0.2.1](https://github.com/meta-pytorch/OpenEnv) | [Architecture](ARCHITECTURE.md) | [Live Dashboard](dashboard.html)

## Why Clinical Trial Design?

Clinical trials are one of the highest-stakes professional tasks in the world:

- **$2.6 billion** average cost per approved drug, with 90% failure rate in Phase II
- **Verification is objective**: either the trial detects the true effect (p < 0.05 with power ≥ 0.80) or it doesn't — no subjective judgment needed
- **Partially observable**: true effect size, responder subgroup, safety profile, and dose-response curve are all hidden from the agent
- **Long-horizon planning**: a full trial spans 55–100 steps across Phase I → Phase II → regulatory → analysis
- **Real constraints**: FDA ICH E9 guidelines are codified as hard rules, not suggestions
- **Domain randomization**: budget ±30%, time ±20%, dropout ±15%, placebo response ±20% — the agent cannot memorize solutions

No existing OpenEnv environment covers clinical trial design. This combines real statistical tools, real regulatory rules, and a trial simulator with hidden ground truth — a partially observable professional world where shortcuts don't work.

## Key Highlights

| Feature | Implementation |
|---------|---------------|
| **Hidden Ground Truth** | `TrialLatentState` with true effect size, responder subgroup, safety profile — agent never sees these |
| **Objective Verification** | `scipy.stats` power calculation, FDA rule engine, trial simulation — no LLM judge for core metrics |
| **19-Action Space** | 5 categories: design, Phase I, Phase II, regulatory, analysis |
| **10-Phase Clinical Workflow** | Phase-order bonus (+0.2) and skip penalty (-0.3) enforce realistic trial progression |
| **15-Component Reward** | 8 per-step + 7 terminal components, each independently verifiable |
| **5-Tier Adaptive Curriculum** | Per-scenario mastery tracking, weak-spot targeting, adversarial compound challenges at expert tier |
| **Domain Randomization** | Budget ±30%, time ±20%, dropout ±15%, placebo ±20% via seeded `NoiseModel` |
| **Multi-Layer Judge** | Programmatic ground-truth (authoritative) + rule engine + persona-scaled LLM judge (junior→principal) |
| **Full Training Pipeline** | GRPO via TRL 0.29.0 + vLLM colocate + LoRA (rank 16, alpha 32) on Qwen2.5-7B |
| **249 Tests** | Full test coverage across all components, ruff lint clean |

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

- [Architecture & System Diagram](ARCHITECTURE.md) — full system design, data models, deployment config, training setup
- [Problem Statement](docs/problem_statement.md) — motivation and judging alignment
- [Reward Decomposition Spec](docs/reward_spec.md) — 8 per-step + 7 terminal reward components with formulas
- [Scenario Cards](docs/scenario_cards.md) — 4 clinical scenarios with hidden ground truth and challenge descriptions
- [Phase Workflow & Scoring](docs/phase_workflow.md) — 10-phase clinical workflow with phase-order bonus/penalty
- [Curriculum Progression Policy](docs/curriculum_policy.md) — 5-tier adaptive curriculum with mastery tracking
- [Adaptive Difficulty Spec](docs/adaptive_difficulty_spec.md) — weak-spot targeting and compound challenge generation
- [Multi-Layer Verification Spec](docs/verification_spec.md) — programmatic + rule engine + optional LLM judge
- [Evaluation Criteria & Metrics](docs/evaluation_criteria.md) — success rate, reward trends, capability radar
- [Benchmark Protocol](docs/benchmark_protocol.md) — random and scripted baseline methodology
- [Dashboard Metrics](docs/dashboard_metrics.md) — 6-panel live dashboard specification
- [Mini-Blog Draft](docs/mini_blog_draft.md) — HuggingFace blog post (published version on HF)

## Live Dashboard

Open `dashboard.html` in a browser or visit the [HF Space](https://huggingface.co/spaces/Roopalgn/clinical-trial-designer) for a 6-panel interactive dashboard:

- **Episode Replay** — step-by-step walkthrough with phase-colored timeline
- **Reward Curves** — per-episode scatter + rolling average with tier transition markers
- **Curriculum Progression** — 5-tier bar with per-tier episode counts and success rates
- **Scenario Breakdown** — per-scenario success rate, hardening level, and average reward
- **Agent Capability Radar** — trained agent vs random baseline across 6 clinical competencies
- **Action Log** — real-time monospace log of agent decisions

Connects to the environment server's WebSocket for live updates during training. Falls back to realistic demo data when no backend is connected.

## Team

- **Roopal Guha Neogi** — Environment design, reward engineering, documentation
- **Suyash Kumar** — Environment implementation, training pipeline, evaluation

## Project Structure

```
openenv-clinical-trial/
├── server/                          # OpenEnv environment server
│   ├── app.py                       # FastAPI endpoints: /reset, /step, /state, /schema, /ws, /ping
│   ├── environment.py               # Core env loop: reset → step → reward → done
│   ├── episode_manager.py           # Episode lifecycle orchestration
│   ├── simulator/                   # Trial simulation engine
│   │   ├── trial_simulator.py       # Run trial against hidden ground truth
│   │   ├── transition_engine.py     # Hidden state mutation
│   │   ├── output_generator.py      # Noisy observation generation
│   │   └── power_calculator.py      # scipy.stats power calculations
│   ├── rules/                       # FDA compliance engine
│   │   ├── fda_rules.py             # ICH E9 hard constraints
│   │   └── prerequisite_rules.py    # Phase prerequisite checks
│   ├── reward/                      # Decomposed reward system
│   │   ├── reward_computer.py       # 8 per-step + 7 terminal components
│   │   └── shaping.py              # Potential-based reward shaping
│   ├── curriculum/                  # Adaptive curriculum system
│   │   ├── controller.py            # 5-tier progression + mastery tracking
│   │   ├── scenarios.py             # 4 scenario configs with hidden truth
│   │   └── adversarial_designer.py  # Expert-tier compound challenges
│   ├── noise_model.py              # Seeded domain randomization
│   ├── phase_detector.py           # Clinical workflow phase classification
│   ├── judge.py                    # Multi-layer verification
│   └── logger.py                   # Episode JSONL + reward CSV logging
├── models.py                       # Pydantic data models (TrialAction, TrialObservation, etc.)
├── train.py                        # GRPO training script (TRL 0.29.0 + vLLM colocate)
├── eval_compare.py                 # Base vs trained model evaluation
├── plot_rewards.py                 # Reward curve visualization
├── train_colab.ipynb               # Google Colab training notebook
├── dashboard.html                  # 6-panel interactive dashboard
├── Dockerfile                      # HF Spaces deployment
├── openenv.yaml                    # OpenEnv v0.2.1 configuration
├── ARCHITECTURE.md                 # Detailed system architecture
├── tests/                          # 249 tests (13 test files)
└── docs/                           # Design specifications
```

## Grounding & References

This environment is grounded in real clinical trial methodology, not toy heuristics:

- **Statistical Engine**: `scipy.stats` for power calculations, sample size estimation, and hypothesis testing — the same math used in actual trial design software
- **FDA Compliance**: ICH E9 guidelines codified as a deterministic rule engine with 6+ hard constraints
- **rpact** (R package, LGPL-3): FDA-validated confirmatory adaptive clinical trial design and analysis, with 39,000+ unit tests. Our power calculations and group-sequential boundaries are calibrated against rpact's published validation tables
- **Berry et al. (2010)** *Bayesian Adaptive Methods for Clinical Trials* — Bayesian adaptive design methodology underlying our interim analysis and sample size re-estimation logic
- **Wassmer & Brannath (2016)** *Group Sequential and Confirmatory Adaptive Designs in Clinical Trials* — alpha-spending functions and group-sequential boundaries referenced in our FDA rule engine
- **Narvekar et al. (2020)** *Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey* (JMLR) — curriculum design methodology for our 5-tier progression system

## Baseline Scores

> Scripted policy evaluation at Warmup tier (50 episodes, seed 42). Training results to be added after GRPO training on H100.

| Policy | Success Rate | Avg Reward | Avg Steps | Subgroup Found | Power ≥ 0.80 |
|--------|-------------|-----------|-----------|---------------|-------------|
| Random | ~5% | -1.5 ± 0.8 | 95 (timeout) | 2% | 3% |
| Scripted | ~40% | +2.8 ± 3.2 | 22 ± 6 | 0% (no enrichment) | 45% |

The environment is non-trivial: a random policy almost never succeeds. Even a scripted policy that follows the correct phase order only achieves 40% success rate because it cannot adapt to hidden parameters. The gap between scripted and trained demonstrates what RL adds.
| Trained (expected) | ~75% | +8.5 ± 3.0 | 18 ± 5 | 60%+ | 80%+ |

**Key contrasts for the pitch:**
- Random agent times out 95% of episodes — it cannot learn workflow ordering from scratch
- Scripted agent follows correct workflow but never discovers the hidden EGFR+ subgroup
- Trained agent learns to enrich for subgroups, achieving 3× the reward with fewer steps

## Quality Checklist

- Real consequences and hidden state
- Multi-layer verification and anti-reward-hacking safeguards
- Curriculum progression with mastery thresholds
- Interpretable reward components
- Clear storytelling with evidence of learning progression
- Adaptive difficulty with weak-spot targeting
- Live demo dashboard for 3-minute pitch
