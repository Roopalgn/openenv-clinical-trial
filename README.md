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

> **Meta PyTorch OpenEnv Hackathon — Theme #3.1: Professional Tasks (World Modeling)**

A drug works — but only in 15% of patients. The FDA needs proof. Can a small LLM learn to design a clinical trial that finds those patients before running out of money?

## Links

| Resource | URL |
|----------|-----|
| **Live Environment** | [HuggingFace Space](https://roopalgn-openenv-clinical-trial.hf.space) |
| **Training Notebook** | [Colab Notebook](train_colab.ipynb) |
| **Mini-Blog / Writeup** | [Draft Shell](docs/mini_blog_draft.md) *(replace with published HF blog URL onsite)* |
| **Code Repository** | [GitHub](https://github.com/Roopalgn/openenv-clinical-trial) |
| **Architecture** | [ARCHITECTURE.md](ARCHITECTURE.md) |

---

## Onsite Fill Checklist

Before final submission, update these items in this README:

- Replace the writeup link above with the published Hugging Face blog or video URL.
- Fill the trained-policy row in the Results table with real metrics from the final run.
- Add the final reward curve image at `results/reward_curve.png`.
- Add a before/after episode comparison using the best transcript pair from `logs/episode_transcripts/`.
- Verify every deliverable link works in a logged-out browser: Space, notebook, writeup, repo.

---

## The Problem

Clinical trials cost **$2.6 billion** per approved drug and take 10–15 years. **90% fail** in Phase II — most due to poor trial design: wrong patients, wrong dose, wrong endpoints.

We built an RL environment where an LLM agent learns to design clinical trials that detect true drug effects under realistic constraints. The agent receives a disease scenario with **hidden ground truth** — the drug's real efficacy, true responder subgroup, and side-effect profile are invisible. It must discover them through experimentation.

**Verification is entirely objective** — `scipy.stats` calculates statistical power, a rule engine checks FDA compliance, and a trial simulation returns the p-value. No LLM judge needed for core success metrics.

---

## Why Clinical Trial Design?

| Property | Why It Fits OpenEnv |
|----------|-------------------|
| **Partially observable** | True effect size, responder subgroup, safety profile, dose-response — all hidden from the agent |
| **Long-horizon** | 9–10 step action sequence across 7 clinical phases with prerequisite dependencies |
| **Objective verification** | `scipy.stats` power calculations, FDA rule engine, trial simulation against hidden ground truth |
| **Real constraints** | FDA ICH E9 rules codified as hard constraints, randomized budgets and timelines |
| **Domain randomization** | Budget ±30%, time ±20%, dropout ±15%, placebo ±20% — no memorization possible |
| **Novel** | First OpenEnv environment for clinical trial design |

Clinical trial design is uniquely suited because it is a **real professional task with math-verified outcomes**. Unlike game or code environments, the agent must make design decisions under genuine partial observability, satisfy hard legal constraints, and produce designs whose quality is measured by `scipy.stats` power calculations against hidden ground truth. The 5-tier adaptive curriculum with domain randomization ensures the agent cannot memorize solutions.

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Hidden Ground Truth** | `TrialLatentState` — true effect size, responder subgroup, safety profile invisible to agent |
| **Objective Verification** | `scipy.stats` power, FDA rule engine, trial simulation — no LLM judge for core metrics |
| **19-Action Space** | Design (8) · Phase I (3) · Phase II (3) · Regulatory (2) · Analysis (2) · Terminal (1) |
| **7-Phase Workflow** | Phase-order bonus (+0.1) and skip penalty (−0.3) enforce realistic trial progression |
| **8-Component Reward** | 6 per-step + 2 terminal, each independently verifiable. Range: −3 to +15 |
| **5-Tier Curriculum** | Warmup → Beginner → Intermediate → Advanced → Expert with per-scenario mastery |
| **Domain Randomization** | Seeded `NoiseModel` with ±30% budget, ±20% time, ±15% dropout, ±20% placebo |
| **Multi-Layer Judge** | Programmatic ground-truth + rule engine + optional LLM judge (junior→principal) |
| **Training Pipeline** | GRPO via TRL 0.29.0 + vLLM colocate + LoRA on Qwen2.5 (1.5B/3B/7B) |
| **Comprehensive Tests** | Full coverage, ruff lint clean |

---

## The Environment

### Hidden World

On each `reset()`, the simulator secretly sets the drug's true efficacy, responder subgroup, dose-response curve, and safety profile. The agent never sees these directly — it must run experiments to discover them.

### Agent Actions (19 total)

| Phase | Actions | Real-World Analog |
|-------|---------|-------------------|
| **Design** | set_primary_endpoint, set_sample_size, set_inclusion_criteria, set_exclusion_criteria, set_dosing_schedule, set_control_arm, set_randomization_ratio, set_blinding | Standard trial protocol |
| **Phase I** | run_dose_escalation, observe_safety_signal, estimate_effect_size | 3+3 dose finding, safety monitoring |
| **Phase II** | run_interim_analysis, modify_sample_size, add_biomarker_stratification | Adaptive design, enrichment |
| **Regulatory** | submit_to_fda_review, request_protocol_amendment | FDA compliance |
| **Analysis** | run_primary_analysis, synthesize_conclusion | Final statistical test |

### Verification (No LLM Judge)

1. **Statistical power** — `scipy.stats` from effect size and sample size (pure math)
2. **FDA compliance** — rule engine checks 6+ codified constraints (binary pass/fail)
3. **Trial simulation** — runs trial against hidden ground truth, returns p-value
4. **Budget** — cost = n_patients × cost_per_patient + site_costs (arithmetic)

### Scenarios (4 × 5 tiers × domain randomization = unlimited unique episodes)

Each scenario encodes a distinct clinical challenge with hidden ground truth the agent must discover. Combined with 5 difficulty tiers and per-episode domain randomization (±30% budget, ±20% time, ±15% dropout, ±20% placebo response), no two episodes are the same — the agent cannot memorize solutions.

| ID | Disease | Challenge | Hidden Insight |
|----|---------|-----------|---------------|
| `solid_tumor_chemo` | Lung cancer (NSCLC) | Find EGFR+ subgroup | 58% effect in EGFR+ vs 31% overall |
| `autoimmune_biologic` | Rheumatoid arthritis | U-shaped dose-response | 200mg optimal, higher doses worse |
| `cns_depression` | Treatment-resistant depression | High placebo masks effect | Enrich for severe TRD subgroup |
| `rare_disease_orphan` | Rare pediatric disorder | Only ~50 patients | Adaptive Bayesian design required |

At Expert tier, the `AdversarialDesigner` generates targeted parameter configurations that exploit the agent's weakest scenario × phase combinations.

### Curriculum

| Tier | Difficulty | What Changes |
|------|-----------|-------------|
| Warmup | Easy | Large effect, homogeneous population |
| Beginner | Medium | Medium effect, some noise |
| Intermediate | Hard | Small effect, enrichment needed |
| Advanced | Very Hard | Hidden subgroup, misleading Phase I |
| Expert | Extreme | Tiny effect, high dropout, adaptive design required |

---

## Results

> Results will be filled after the onsite training run. The training notebook (`train_colab.ipynb`) now trains on full episode action plans: the model outputs the ordered clinical actions, and the reward function executes exactly those actions with no hidden fallback sequence.

| Policy | Avg Reward | Steps | Description |
|--------|-----------|-------|-------------|
| Random action plan | varies | 1–12 | Random ordered action list, often invalid or incomplete |
| **Trained (GRPO)** | **[FILL after run]** | **9–12** | **Model learns valid phase ordering, enrollment, analysis, and design parameters** |

**Run summary (fill after training)**

| Field | Value |
|------|------|
| Model | `Qwen2.5-1.5B-Instruct-bnb-4bit + LoRA` |
| Training episodes | `20` |
| Seed | `42` |
| Best episode reward | `[FILL]` |
| Final avg reward | `[FILL]` |
| Trend slope | `[FILL]` |

### Reward Curve

`Validation run curve saved from Colab. Keep this slot and replace the file in-repo once the HF-credit run is complete.`

![Reward Curve](results/reward_curve.png)

Caption: `In the first successful Colab validation run, rewards stayed stably positive with a best observed value of 18.53, a final average of 17.52, and a shallow but positive trend slope of 0.002 over 20 episodes.`

### Before vs After Episode

Use one early failure transcript and one late success transcript.

| Metric | Before Training | After Training |
|--------|-----------------|----------------|
| Episode ID | `[FILL]` | `[FILL]` |
| Total reward | `[FILL]` | `[FILL]` |
| Steps | `[FILL]` | `[FILL]` |
| FDA pass rate | `[FILL]` | `[FILL]` |
| Key behavior | `[FILL]` | `[FILL]` |
| Outcome | `[FILL]` | `[FILL]` |

Narrative template: `Early in training the agent [FILL failure pattern]. After training it [FILL improved workflow], which led to [FILL measurable outcome].`

---

## Reward Structure (8 components)

### Per-Step Reward (6 components)

```
r_step = r_validity + r_ordering + r_info_gain + r_efficiency + r_novelty + r_penalty
```

| Component | What It Measures | Reward | Verification |
|-----------|-----------------|--------|-------------|
| `r_validity` | FDA rule compliance | +0.05 pass, −2.0 invalid | Rule engine (binary) |
| `r_ordering` | Correct phase workflow | +0.1 correct, −0.3×N skip | Phase detection |
| `r_info_gain` | Information from experiments + milestone bonuses | +0.1 to +1.5 | Power × base + first-time milestone |
| `r_efficiency` | Budget efficiency (terminal only) | 0.0 to +0.3 | Math (remaining / initial budget) |
| `r_novelty` | Trying new action types | +0.1 first use | Action history |
| `r_penalty` | Compliance violations | −0.5 per violation | Rule engine |

### Terminal Reward (2 components)

Fires once when `trial_complete=True` after `run_primary_analysis`.

| Component | Condition | Reward |
|-----------|----------|--------|
| `r_terminal_success` | Trial succeeds (p < α, no failure) | +4.0 |
| | Trial completes but fails | −1.0 |
| `r_terminal_calibration` | CI accuracy vs true effect size | 0.0 to +2.0 |

### Episode Total

$$R_{\text{episode}} = \sum_{t=1}^{T} r_{\text{step}_t} + r_{\text{terminal}}$$

| Outcome | Typical Total |
|---------|-------------|
| Optimal design (high power + efficient + calibrated) | +10 to +15 |
| Good design (trial succeeds, partial calibration) | +5 to +10 |
| Failed trial (p ≥ 0.05 or budget/time exceeded) | −1 to +3 |
| Parse failure / invalid sequence | −3 |

This 18-point spread (+15 to −3) is what GRPO needs — clear separation between good and great designs for stable policy gradients.

---

## Setup

```bash
git clone https://github.com/Roopalgn/openenv-clinical-trial.git
cd openenv-clinical-trial
pip install -e .

# Run environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Health check
curl http://localhost:8000/ping
```

### Docker

```bash
docker build -t clinical-trial-env .
docker run -p 8000:7860 clinical-trial-env
```

### Training

```bash
# Dry-run (no GPU — validates pipeline)
python train.py --dry-run --episodes 2

# GRPO training on GPU
python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 20
python train.py --model-size 3b --model-path Qwen/Qwen2.5-3B-Instruct --episodes 50
python train.py --model-size 7b --model-path Qwen/Qwen2.5-7B-Instruct --episodes 50

# Evaluation
python eval_compare.py --base-only --episodes 10
python eval_compare.py --model-path outputs/grpo/checkpoint-final --episodes 20

# Reward curve
python plot_rewards.py
```

Notebooks: `train_colab.ipynb`, `train_kaggle.ipynb`

---

## Documentation

| Document | What It Covers |
|----------|---------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data models, deployment, training setup |
| [Reward Spec](docs/reward_spec.md) | 8-component reward with formulas |
| [Scenario Cards](docs/scenario_cards.md) | 4 scenarios with hidden ground truth |
| [Phase Workflow](docs/phase_workflow.md) | 7-phase clinical workflow with scoring |
| [Curriculum Policy](docs/curriculum_policy.md) | 5-tier adaptive curriculum |
| [Verification Spec](docs/verification_spec.md) | Multi-layer verification (math + rules + optional LLM) |
| [Statistical Grounding](docs/grounding.md) | rpact validation, Berry 2010, Wassmer & Brannath 2016 |
| [Adaptive Difficulty](docs/adaptive_difficulty_spec.md) | Weak-spot targeting and parameter hardening |
| [Onsite Roadmap](docs/onsite_roadmap.md) | Step-by-step H100 training execution plan |

---

## Dashboard

Open `dashboard.html` or visit the [HF Space](https://roopalgn-openenv-clinical-trial.hf.space) for a 6-panel interactive dashboard:

- **Episode Replay** — step-by-step walkthrough with phase-colored timeline
- **Reward Curves** — per-episode scatter + rolling average
- **Curriculum Progression** — tier advancement with episode counts
- **Scenario Breakdown** — per-scenario success rates
- **Agent Capability Radar** — trained vs random baseline
- **Action Log** — real-time agent decisions

---

## Grounding

This environment is grounded in real clinical trial methodology:

- **scipy.stats** for power calculations and hypothesis testing — same math as actual trial software
- **FDA ICH E9** guidelines codified as a deterministic rule engine
- **rpact** (R, LGPL-3) — FDA-validated adaptive trial design with 39K+ unit tests. Our power calculations calibrated against rpact's validation tables
- **Berry et al. (2010)** Bayesian Adaptive Methods for Clinical Trials
- **Wassmer & Brannath (2016)** Group Sequential and Confirmatory Adaptive Designs
- **Narvekar et al. (2020)** Curriculum Learning for RL Domains (JMLR)

---

## Project Structure

```
openenv-clinical-trial/
├── server/                     # OpenEnv environment server
│   ├── app.py                  # FastAPI: /reset, /step, /state, /schema, /ws, /ping
│   ├── environment.py          # Core env loop
│   ├── episode_manager.py      # Episode lifecycle
│   ├── simulator/              # Trial simulation engine
│   ├── rules/                  # FDA compliance engine
│   ├── reward/                 # 8-component decomposed reward
│   ├── curriculum/             # 5-tier adaptive curriculum
│   ├── judge.py                # Multi-layer verification
│   └── logger.py               # Episode JSONL + reward CSV
├── models.py                   # Pydantic data models
├── train.py                    # GRPO training (TRL 0.29.0 + vLLM + LoRA)
├── eval_compare.py             # Base vs trained evaluation
├── plot_rewards.py             # Reward curve visualization
├── train_colab.ipynb           # Colab training notebook
├── dashboard.html              # 6-panel interactive dashboard
├── Dockerfile                  # HF Spaces deployment
├── openenv.yaml                # OpenEnv v0.2.3 config
├── ARCHITECTURE.md             # System architecture
├── tests/                      # 249 tests
└── docs/                       # Design specs
```

---

## Team

- **Roopal Guha Neogi** — Environment design, reward engineering, documentation
- **Suyash Kumar** — Environment implementation, training pipeline, deployment
