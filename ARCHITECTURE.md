# Clinical Trial Designer — Architecture

An RL agent learns to design clinical trials via GRPO, interacting with a simulator that holds hidden ground truth.

```
H100 (all-in-one)                              Trial Simulator
┌────────────────────────────────────┐          ┌──────────────────────┐
│                                    │          │ TrialLatentState     │
│  OpenEnv Server :8000              │          │  true_effect_size    │
│  ├─ Environment (reset/step)       │ ────────►│  true_responders     │
│  ├─ Trial Simulator                │          │  true_safety_profile │
│  ├─ FDA Rule Engine                │          │  placebo_response    │
│  ├─ Reward Computer                │          │  dropout_rate        │
│  ├─ Curriculum Controller          │          └──────────────────────┘
│  ├─ Phase Detector                 │
│  ├─ TransitionEngine               │          Verification Layers
│  ├─ OutputGenerator                │          ┌──────────────────────┐
│  ├─ NoiseModel (domain random.)    │          │ 1. Power ≥ 0.80?    │
│  ├─ AdversarialDesigner (expert+)  │          │ 2. FDA rules pass?   │
│  └─ TrialJudge (multi-layer)      │ ────────►│ 3. Trial sim p<0.05? │
│       ├─ Programmatic (power,      │          │ 4. Budget under?     │
│       │   p-value, FDA rules)      │          │ 5. Calibration match?│
│       └─ Persona-scaled LLM judge  │          └──────────────────────┘
│           ├─ Junior  (diff < 0.4)  │
│           ├─ Senior  (0.4–0.7)    │          Episode Logging
│           └─ Principal (> 0.7)     │          ┌──────────────────────┐
│                                    │          │ episode_transcripts/ │
│  GRPO Trainer (TRL 0.29.0)         │ ────────►│  *.jsonl (per-step)  │
│  ├─ Agent model + LoRA (BF16)      │          │  reward_log.csv      │
│  ├─ vLLM colocate (inference)      │          │  curriculum_log.csv  │
│  └─ 8 rollouts × grad_accum=8     │          └──────────────────────┘
│                                    │
└────────────────────────────────────┘
```

> **Inspired by:** System diagram pattern from KubeSRE's ARCHITECTURE.md. Adversarial designer from KubeSRE (Claude designs incidents targeting weak spots). TransitionEngine + OutputGenerator + NoiseModel from Bio Experiment (separates hidden state updates from noisy output generation). Judge persona scaling from KubeSRE (junior/senior/principal). JSONL episode transcripts from KubeSRE (agent_transcripts.jsonl).

## One Episode

```
1. train.py calls env.reset()
   → Curriculum picks scenario + difficulty
   → NoiseModel randomizes hidden parameters (±30% budget, ±20% time, etc.)
   → TrialLatentState created with hidden ground truth
   → Agent sees: scenario description + initial observation (no hidden values)

2. Agent generates actions (55–100 steps):
   Phase I:  run_dose_escalation → observe_safety_signal → estimate_effect_size
   Phase II: set_primary_endpoint → set_sample_size → set_inclusion_criteria → ...
   Regulatory: submit_to_fda_review
   Analysis: run_primary_analysis → synthesize_conclusion

3. Each action (inspired by Bio Experiment's TransitionEngine → OutputGenerator pipeline):
   → RuleEngine checks FDA constraints + prerequisites (hard block if violated)
   → TransitionEngine updates TrialLatentState (hidden: enroll patients, spend budget, etc.)
   → OutputGenerator conditions on new hidden state, injects realistic noise via NoiseModel
   → Phase detector classifies into workflow phase (like KubeSRE: triage→investigate→fix→verify)
   → Reward computer calculates 8 decomposed components
   → Phase-order bonus/penalty applied (+0.2 correct / -0.3×skipped)
   → Step logged to episode_transcripts/*.jsonl (action, observation, reward, phase, hidden_state)

4. Terminal (done=True):
   → Trial simulation runs with hidden ground truth (like KubeSRE's pod health check)
   → Programmatic verification: power, p-value, FDA compliance, budget
   → Terminal calibration: agent's conclusions vs hidden truth (like Bio's calibration score)
   → Terminal reward: +5 to +14 (success) or -2 to -3 (failure)
   → Overconfidence penalty: -0.5 per high-confidence wrong claim (from Bio Experiment)
   → Curriculum records outcome, adjusts difficulty for next episode
   → If at expert tier: AdversarialDesigner notes weak spots for next scenario

5. GRPO computes advantages across 8 parallel rollouts, updates weights
   → Episode transcript saved: full action-observation-reward sequence for offline analysis
```

## Component Responsibilities

### Environment Server (`server/app.py`)
- FastAPI app with OpenEnv endpoints: `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/ping`
- Serves the environment via HTTP and WebSocket
- Deployed on HF Spaces as Docker container using `openenv-base` image

### Trial Simulator (`server/simulator/`)
- Holds `TrialLatentState` (hidden ground truth) — inspired by Bio Experiment's hidden DE genes / true cell populations
- Executes agent's trial design against hidden parameters
- Returns p-values, confidence intervals, adverse event rates
- Uses real scipy.stats for power calculations and statistical tests — like KubeSRE's real kubectl against live cluster

### TransitionEngine (`server/simulator/transition_engine.py`)
> *Pattern from Bio Experiment: separates hidden state mutation from observation generation*
- Receives validated action from RuleEngine
- Mutates `TrialLatentState`: enrolls patients, spends budget, advances time, records adverse events
- Degrades data quality on soft violations (like Bio's quality degradation on rule violations)
- Never produces observations directly — hands updated state to OutputGenerator

### OutputGenerator (`server/simulator/output_generator.py`)
> *Pattern from Bio Experiment: conditions on hidden state, then injects realistic noise*
- Takes updated `TrialLatentState` from TransitionEngine
- Generates observation conditioned on hidden state (e.g., noisy effect size estimate from true value)
- Calls NoiseModel to inject measurement noise, site variability, dropout artifacts
- Agent only sees this noisy output — never the clean hidden state

### FDA Rule Engine (`server/rules/`)
> *Pattern from Bio Experiment's prerequisite rules: hard constraints that block invalid actions*
- Encodes ICH E9 guidelines as hard constraints
- Checks: phase prerequisites, sample size floors, alpha-spending, randomization, blinding
- Returns binary pass/fail per rule — no LLM judgment
- Blocked actions return error observation (not silent failure)

### Reward Computer (`server/reward/`)
> *Pattern from all 3 winners: decomposed, interpretable, independently verifiable components*
- 8 decomposed components: validity, ordering, info_gain, efficiency, novelty, penalty, terminal_success, terminal_calibration
- Each component independently verifiable and debuggable
- Potential-based shaping: γ·(φ(s') − φ(s)) where φ = milestone_completion × budget_efficiency
- Terminal calibration: agent's conclusions vs hidden truth — like Bio's ground-truth calibration score (4.0 weight)
- Overconfidence penalty: -0.5 per high-confidence wrong claim — directly from Bio Experiment

### Curriculum Controller (`server/curriculum/`)
> *Pattern from KubeSRE: 5 tiers with per-fault mastery tracking + fast-track advancement*
- 5 tiers: warmup → beginner → intermediate → advanced → expert
- Per-scenario-type mastery tracking (70% success over window = graduated) — same threshold as KubeSRE
- Fast-track: 90%+ success after 3 episodes → skip min_episodes requirement — from KubeSRE
- Weak-spot targeting: harder parameters for mastered scenario types
- Judge strictness scales with difficulty (junior → senior → principal)

### AdversarialDesigner (`server/curriculum/adversarial_designer.py`)
> *Pattern from KubeSRE: Claude designs incidents targeting agent's tracked weak spots*
- Activates at expert tier (difficulty > 0.80)
- Analyzes agent's failure patterns across episodes (which scenarios fail, which phases trip up)
- Generates targeted scenario parameters: e.g., if agent struggles with subgroup identification, inject more hidden subgroups with misleading Phase I signals
- Compound challenges: multiple hidden factors simultaneously (small effect + hidden subgroup + high dropout)
- All generated scenarios validated as solvable within step budget — from KubeSRE's inject/fix pair validation

### Phase Detector (`server/phase_detector.py`)
> *Pattern from KubeSRE: triage → investigate → fix → verify workflow with phase-order bonus*
- Classifies each action into clinical workflow phase
- Phases: literature_review → hypothesis → design → enrollment → monitoring → analysis → submission
- Bonus (+0.2) for correct phase ordering, penalty (-0.3) per skipped phase

### NoiseModel (`server/noise_model.py`)
> *Pattern from Bio Experiment: centralized stochasticity with seeded numpy.Generator*
- Centralizes all stochasticity with seeded `numpy.Generator`
- Domain randomization: budget ±30%, time ±20%, dropout ±15%, placebo ±20%
- Same seed → same episode (reproducibility guarantee)
- Prevents overfitting to specific parameter values — Bio's key defense against memorization

### TrialJudge (`server/judge.py`)
> *Pattern from KubeSRE: programmatic health check + LLM judge + persona scaling*
- Layer 1 (programmatic): power check, p-value check, FDA rule pass, budget check — authoritative
- Layer 2 (persona-scaled LLM): workflow quality assessment, justification quality scoring
  - Junior persona (difficulty < 0.4): lenient, gives hints in observation — like KubeSRE's junior judge
  - Senior persona (0.4–0.7): standard clinical expectations, no hints
  - Principal persona (> 0.7): strict, penalizes inefficiency, flags redundant tests
- Programmatic layer is authoritative — LLM layer never overrides it (same as KubeSRE)

### Episode Logger (`server/logger.py`)
> *Pattern from KubeSRE: episode_transcripts.jsonl + agent_transcripts.jsonl*
- Saves full episode transcript to JSONL: one line per step with action, observation, reward breakdown, phase, hidden state snapshot
- Saves reward CSV: episode_id, step, each reward component, total, cumulative
- Saves curriculum log: episode_id, tier, scenario_id, difficulty, outcome, advancement
- Enables offline debugging, reward curve plotting, and before/after comparison

## Data Models

```
TrialAction
├── action_type: ActionType (enum, 19 values)
├── parameters: dict
├── justification: str
└── confidence: float (0.0–1.0)

TrialObservation (agent sees this — noisy, generated by OutputGenerator)
├── scenario_description: str
├── phase_data: dict (noisy results from experiments — NOT raw hidden values)
├── resource_status: dict (budget_remaining, time_remaining, patients_enrolled)
├── rule_violations: list[str]
├── available_actions: list[str] (valid actions given current prerequisites)
├── steps_taken: int
├── max_steps: int
├── hint: str (only at junior difficulty — judge persona feature)
├── done: bool
└── reward: float

TrialState (episode metadata — visible to training loop)
├── episode_id: str
├── step_count: int
├── difficulty: float
├── scenario_id: str
├── curriculum_tier: str
├── curriculum_stats: dict
├── action_diversity: float (unique actions / total actions — from VRAM)
├── phase_compliance_rate: float
└── is_resolved: bool

TrialLatentState (HIDDEN — never sent to agent, like Bio's hidden DE genes)
├── true_effect_size: float
├── true_side_effect_rate: float
├── true_responder_population: str
├── true_responder_criteria: list[str] (e.g., ["BRCA1+", "age < 65"])
├── true_dose_response: dict
├── true_mechanism: str
├── placebo_response_rate: float
├── dropout_rate: float
├── site_variability: float
├── measurement_noise: float
├── budget_remaining: float
├── time_remaining_days: int
├── patients_enrolled: int
├── phase_i_complete: bool
├── mtd_identified: bool
├── effect_estimated: bool
├── protocol_submitted: bool
├── interim_complete: bool
└── trial_complete: bool

EpisodeTranscript (JSONL — one line per step, like KubeSRE)
├── episode_id: str
├── step: int
├── action: TrialAction
├── observation: TrialObservation
├── reward_breakdown: dict (each of 8 components)
├── total_reward: float
├── phase_detected: str
├── phase_order_correct: bool
├── hidden_state_snapshot: TrialLatentState (for offline debugging only)
└── timestamp: str
```

## File Structure (Target)

```
openenv-clinical-trial/
├── server/
│   ├── app.py                    # FastAPI + OpenEnv endpoints
│   ├── environment.py            # Core env: reset → step → reward → done
│   ├── simulator/
│   │   ├── trial_simulator.py    # Run trial against hidden truth
│   │   ├── transition_engine.py  # Hidden state mutation (Bio pattern)
│   │   ├── output_generator.py   # Noisy observation from hidden state (Bio pattern)
│   │   └── power_calculator.py   # scipy.stats power calculations
│   ├── rules/
│   │   ├── fda_rules.py          # FDA ICH E9 constraint engine
│   │   └── prerequisite_rules.py # Phase prerequisite checks
│   ├── reward/
│   │   ├── reward_computer.py    # 8-component decomposed reward
│   │   └── shaping.py            # Potential-based reward shaping
│   ├── curriculum/
│   │   ├── controller.py         # 5-tier curriculum + mastery tracking
│   │   ├── scenarios.py          # 4 scenario configs with hidden truth
│   │   └── adversarial_designer.py # Expert-tier scenario generation (KubeSRE pattern)
│   ├── noise_model.py            # Seeded domain randomization (Bio pattern)
│   ├── phase_detector.py         # Clinical workflow phase classification
│   ├── judge.py                  # Multi-layer verification (programmatic + LLM)
│   └── logger.py                 # Episode transcript JSONL + reward CSV (KubeSRE pattern)
├── models.py                     # TrialAction, TrialObservation, TrialState, etc.
├── client.py                     # Sync client for training
├── train.py                      # GRPO training (TRL + vLLM colocate)
├── eval_compare.py               # Base vs trained model comparison
├── plot_rewards.py               # Reward curve visualization
├── train_colab.ipynb             # Google Colab training notebook
├── Dockerfile                    # HF Spaces deployment (openenv-base image)
├── openenv.yaml                  # OpenEnv v0.2.1 Space config
├── ARCHITECTURE.md               # This file
├── README.md                     # Project overview
└── docs/
    ├── ROADMAP.md
    ├── problem_statement.md
    ├── evaluation_criteria.md
    ├── story_arc.md
    ├── phase_workflow.md
    ├── comparison.md
    ├── hack_info.md
    └── KnowledgeBase.md
```

## Deployment

```dockerfile
# Dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# openenv.yaml
spec_version: 1
name: clinical_trial_designer
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

## Training Setup (H100)

### Hardware Requirements

| Component | Spec | Purpose |
|-----------|------|---------|
| GPU | 1× NVIDIA H100 80GB | Model + vLLM colocate |
| RAM | ≥ 64GB | Episode buffers, curriculum state |
| SSD | ≥ 100GB | Checkpoints (every 50 steps × ~15GB) |
| Network | Low-latency to env server | Docker on same machine preferred |

### Environment + Training on Same Machine

```bash
# Terminal 1: Environment server (CPU-only, low resource)
docker compose up -d
curl http://localhost:8000/ping  # Verify: {"status": "ok"}

# Terminal 2: GRPO training (GPU — uses ~70% VRAM for vLLM, 30% for training)
python train.py \
    --model unsloth/Qwen2.5-7B-bnb-4bit \
    --env-url http://localhost:8000 \
    --lora-r 16 --lora-alpha 32 \
    --num-generations 8 \
    --max-steps 500 \
    --lr 5e-6 \
    --output-dir checkpoints/grpo_clinical_trial \
    --reward-csv results/rewards.csv \
    --seed 42
```

### H100 Memory Budget

```
Total VRAM: 80 GB
├── vLLM inference engine (colocate): ~50 GB (7B model BF16 + KV cache for 8 sequences)
├── LoRA training parameters:         ~2 GB (rank 16, all linear layers)
├── Optimizer states (AdamW):          ~4 GB
├── Gradient accumulation buffers:     ~4 GB (grad_accum=4)
└── Headroom:                         ~20 GB
```

If OOM occurs: reduce `vllm_gpu_utilization` to 0.5 or use 4-bit quantized inference model.

### Checkpoint → HuggingFace Hub

```bash
# After training completes
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='checkpoints/grpo_clinical_trial',
    repo_id='Roopalgn/clinical-trial-designer-grpo',
    repo_type='model',
)
"
```

---

## Complete System Diagram (All Implemented Components)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP (H100)                                  │
│                                                                              │
│  ┌─────────────────────┐        ┌──────────────────────────────────────┐    │
│  │  GRPOTrainer (TRL)  │        │  OpenEnv Server :8000 (Docker)       │    │
│  │  ├── Qwen2.5-7B     │◄──────►│  ├── /reset  → CurriculumController│    │
│  │  │   + LoRA (r=16)  │  HTTP  │  │             → NoiseModel          │    │
│  │  ├── vLLM colocate  │        │  │             → TrialLatentState     │    │
│  │  ├── 8 rollouts     │        │  ├── /step   → RuleEngine            │    │
│  │  ├── AdamW (5e-6)   │        │  │             → TransitionEngine     │    │
│  │  └── reward_fn()────│────────│──│             → OutputGenerator      │    │
│  └─────────────────────┘        │  │             → RewardComputer       │    │
│                                  │  │             → PhaseDetector        │    │
│  ┌─────────────────────┐        │  │             → EpisodeLogger        │    │
│  │  Results / Logs      │        │  ├── /state  → TrialState            │    │
│  │  ├── rewards.csv     │        │  ├── /schema → ActionSpace           │    │
│  │  ├── curriculum.jsonl│        │  ├── /ws     → Live step updates     │    │
│  │  ├── transcripts/    │        │  ├── /ping   → Health check          │    │
│  │  │   └── *.jsonl     │        │  └── /dashboard → dashboard.html     │    │
│  │  └── checkpoints/    │        └──────────────────────────────────────┘    │
│  │      └── step_NNN/   │                                                    │
│  └─────────────────────┘                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      ENVIRONMENT INTERNALS                                    │
│                                                                              │
│  reset() ─────────────────────────────────────────────────────►              │
│  │                                                                           │
│  │  1. CurriculumController.get_next_scenario()                             │
│  │     ├── Per-scenario mastery tracking (sliding window)                    │
│  │     ├── Weak-spot targeting (70/30 split)                                │
│  │     ├── Fast-track advancement (≥90% in 5 → skip min_episodes)          │
│  │     └── Tier: warmup → beginner → intermediate → advanced → expert       │
│  │                                                                           │
│  │  2. AdversarialDesigner (expert tier only)                               │
│  │     ├── FailureAnalyzer.get_weak_spots()                                 │
│  │     ├── Parameter hardening (4 steps per scenario)                        │
│  │     ├── Compound challenges (needle_in_haystack, budget_crunch, ...)     │
│  │     └── Solvability guarantee (power check + budget check)               │
│  │                                                                           │
│  │  3. NoiseModel.apply_domain_randomization()                              │
│  │     ├── Budget ±30%, time ±20%, dropout ±15%                             │
│  │     ├── Seeded numpy.Generator (reproducible)                            │
│  │     └── Noise scaling by tier (±10% warmup → ±50% expert)               │
│  │                                                                           │
│  │  4. TrialLatentState created (hidden from agent)                          │
│  │     ├── true_effect_size, true_responder_population                      │
│  │     ├── true_dose_response, true_mechanism                               │
│  │     └── placebo_response_rate, dropout_rate, site_variability            │
│  │                                                                           │
│  step(action) ────────────────────────────────────────────────►              │
│  │                                                                           │
│  │  1. RuleEngine.check(action, state)                                       │
│  │     ├── Hard prerequisites (block action if failed)                       │
│  │     └── FDA ICH E9 compliance (6 codified rules)                         │
│  │                                                                           │
│  │  2. TransitionEngine.apply(action, latent_state)                          │
│  │     ├── Enrolls patients, spends budget, advances time                   │
│  │     ├── Runs statistical tests (scipy.stats)                             │
│  │     └── Updates internal Phase I/II state                                │
│  │                                                                           │
│  │  3. OutputGenerator.generate(latent_state)                                │
│  │     ├── Conditions observation on hidden state                           │
│  │     ├── Injects measurement noise via NoiseModel                         │
│  │     └── Agent sees ONLY this noisy output                                │
│  │                                                                           │
│  │  4. PhaseDetector.detect(action, history)                                 │
│  │     └── Classifies into 10 workflow phases                               │
│  │                                                                           │
│  │  5. RewardComputer.compute(action, state, latent_state)                   │
│  │     ├── 8 per-step components (validity, ordering, info_gain, ...)       │
│  │     ├── Potential-based shaping: γ·(φ(s') − φ(s))                       │
│  │     ├── Judge persona scaling (junior → principal)                       │
│  │     └── If done: 7 terminal components (success, calibration, power, ...)│
│  │                                                                           │
│  │  6. EpisodeLogger.log_step(...)                                           │
│  │     ├── JSONL: action, observation, reward breakdown, phase, latent       │
│  │     ├── CSV: per-episode reward totals                                   │
│  │     └── WebSocket: push step to dashboard.html                           │
│  │                                                                           │
│  │  7. TrialJudge (multi-layer verification)                                 │
│  │     ├── L1: Programmatic ground-truth (authoritative)                    │
│  │     ├── L2: Rule engine soft constraints                                  │
│  │     └── L3: Optional LLM judge (informational only)                      │
│  │                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      EVALUATION & PITCH ASSETS                                │
│                                                                              │
│  eval_compare.py ──► Random vs Scripted vs Trained comparison table          │
│  plot_rewards.py ──► Reward scatter + rolling avg + tier markers PNG         │
│  dashboard.html  ──► 6-panel live/demo dashboard (embeddable in HF Space)    │
│  train_colab.ipynb ─► Minimal Colab notebook (judging minimum requirement)   │
│  mini_blog_draft.md ► HuggingFace blog post (judging minimum requirement)    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Post-Training Results

> **Section to be filled onsite April 25–26 after GRPO training completes.**

### Training Configuration (Actual)

| Parameter | Value |
|-----------|-------|
| Model | `unsloth/Qwen2.5-7B-bnb-4bit` |
| LoRA | rank 16, alpha 32, target all linear |
| GRPO generations | 8 |
| Max training steps | 500 |
| Learning rate | 5e-6 |
| vLLM GPU utilization | 0.7 |
| Environment | Docker on same H100 machine |
| Seed | 42 |

### Results Summary

| Metric | Value |
|--------|-------|
| Total episodes completed | [fill] |
| Final curriculum tier reached | [fill] |
| Best single episode reward | [fill] |
| Final rolling avg reward (window=20) | [fill] |
| Training wall-clock time | [fill] |
| Checkpoints saved | [fill] |

### Key Observations

- [Fill after training: what behaviors emerged, what the agent learned, any bugs discovered]
- [Fill: curriculum progression timeline — when did each tier advance?]
- [Fill: which reward components improved fastest?]
