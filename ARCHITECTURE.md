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
│  ├─ NoiseModel (domain random.)    │          Verification Layers
│  └─ TrialJudge (multi-layer)      │          ┌──────────────────────┐
│       ├─ Programmatic (power,      │          │ 1. Power ≥ 0.80?    │
│       │   p-value, FDA rules)      │ ────────►│ 2. FDA rules pass?   │
│       └─ Optional LLM (workflow)   │          │ 3. Trial sim p<0.05? │
│                                    │          │ 4. Budget under?     │
│  GRPO Trainer (TRL 0.29.0)         │          └──────────────────────┘
│  ├─ Agent model + LoRA (BF16)      │
│  ├─ vLLM colocate (inference)      │
│  └─ 8 rollouts × grad_accum=8     │
│                                    │
└────────────────────────────────────┘
```

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

3. Each action:
   → Phase detector classifies into workflow phase
   → Rule engine checks FDA constraints + prerequisites
   → Simulator updates hidden state (enroll patients, spend budget, etc.)
   → Reward computer calculates 8 decomposed components
   → Phase-order bonus/penalty applied

4. Terminal (done=True):
   → Trial simulation runs with hidden ground truth
   → Programmatic verification: power, p-value, FDA compliance, budget
   → Terminal reward: +5 to +14 (success) or -2 to -3 (failure)
   → Curriculum records outcome, adjusts difficulty for next episode

5. GRPO computes advantages across 8 parallel rollouts, updates weights
```

## Component Responsibilities

### Environment Server (`server/app.py`)
- FastAPI app with OpenEnv endpoints: `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/ping`
- Serves the environment via HTTP and WebSocket
- Deployed on HF Spaces as Docker container using `openenv-base` image

### Trial Simulator (`server/simulator/`)
- Holds `TrialLatentState` (hidden ground truth)
- Executes agent's trial design against hidden parameters
- Returns p-values, confidence intervals, adverse event rates
- Uses real scipy.stats for power calculations and statistical tests

### FDA Rule Engine (`server/rules/`)
- Encodes ICH E9 guidelines as hard constraints
- Checks: phase prerequisites, sample size floors, alpha-spending, randomization, blinding
- Returns binary pass/fail per rule — no LLM judgment

### Reward Computer (`server/reward/`)
- 8 decomposed components: validity, ordering, info_gain, efficiency, novelty, penalty, terminal_success, terminal_calibration
- Each component independently verifiable and debuggable
- Potential-based shaping: γ·(φ(s') − φ(s)) where φ = milestone_completion × budget_efficiency

### Curriculum Controller (`server/curriculum/`)
- 5 tiers: warmup → beginner → intermediate → advanced → expert
- Per-scenario-type mastery tracking (70% success over window = graduated)
- Weak-spot targeting: harder parameters for mastered scenario types
- Judge strictness scales with difficulty (future Push 3)

### Phase Detector (`server/phase_detector.py`)
- Classifies each action into clinical workflow phase
- Phases: literature_review → hypothesis → design → enrollment → monitoring → analysis → submission
- Bonus (+0.2) for correct phase ordering, penalty (-0.3) for phase skipping

### NoiseModel (`server/noise_model.py`)
- Centralizes all stochasticity with seeded `numpy.Generator`
- Domain randomization: budget ±30%, time ±20%, dropout ±15%, placebo ±20%
- Same seed → same episode (reproducibility guarantee)

### TrialJudge (`server/judge.py`)
- Layer 1 (programmatic): power check, p-value check, FDA rule pass, budget check
- Layer 2 (optional LLM): workflow quality assessment for debugging/demo
- Programmatic layer is authoritative — LLM layer never overrides it

## Data Models

```
TrialAction
├── action_type: ActionType (enum, 19 values)
├── parameters: dict
├── justification: str
└── confidence: float (0.0–1.0)

TrialObservation
├── scenario_description: str
├── phase_data: dict (noisy results from experiments)
├── resource_status: dict (budget_remaining, time_remaining, patients_enrolled)
├── rule_violations: list[str]
├── steps_taken: int
├── max_steps: int
├── hint: str (only at junior difficulty)
├── done: bool
└── reward: float

TrialState
├── episode_id: str
├── step_count: int
├── difficulty: float
├── scenario_id: str
├── curriculum_stats: dict
└── is_resolved: bool

TrialLatentState (HIDDEN — never sent to agent)
├── true_effect_size: float
├── true_side_effect_rate: float
├── true_responder_population: str
├── true_dose_response: dict
├── placebo_response_rate: float
├── dropout_rate: float
├── budget_remaining: float
├── time_remaining_days: int
└── patients_enrolled: int
```

## File Structure (Target)

```
openenv-clinical-trial/
├── server/
│   ├── app.py                    # FastAPI + OpenEnv endpoints
│   ├── environment.py            # Core env: reset → step → reward → done
│   ├── simulator/
│   │   ├── trial_simulator.py    # Run trial against hidden truth
│   │   └── power_calculator.py   # scipy.stats power calculations
│   ├── rules/
│   │   ├── fda_rules.py          # FDA ICH E9 constraint engine
│   │   └── prerequisite_rules.py # Phase prerequisite checks
│   ├── reward/
│   │   ├── reward_computer.py    # 8-component decomposed reward
│   │   └── shaping.py            # Potential-based reward shaping
│   ├── curriculum/
│   │   ├── controller.py         # 5-tier curriculum + mastery tracking
│   │   └── scenarios.py          # 4 scenario configs with hidden truth
│   ├── noise_model.py            # Seeded domain randomization
│   ├── phase_detector.py         # Clinical workflow phase classification
│   └── judge.py                  # Multi-layer verification (programmatic + LLM)
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

```bash
# Terminal 1: Environment server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: GRPO training
python train.py --vllm-mode colocate --num-generations 8 --max-steps 50
```
