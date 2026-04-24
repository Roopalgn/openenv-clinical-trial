# Chapter 5: The Environment — Where Our Agent Lives

## Architecture Overview

Our environment is where the magic happens. It's the "world" that the agent interacts with. Let's look at its architecture:

```
┌─────────────────────────────────────────────────┐
│              EpisodeManager                      │
│  (Central orchestrator — manages everything)     │
│                                                  │
│  reset() → starts new episode                    │
│  step()  → processes one action                  │
│                                                  │
│  Uses:                                           │
│  ├── CurriculumController (picks scenario)       │
│  ├── NoiseModel (randomizes parameters)          │
│  ├── TransitionEngine (updates hidden state)     │
│  ├── OutputGenerator (creates noisy observations)│
│  ├── RewardComputer (calculates rewards)         │
│  ├── PhaseDetector (classifies clinical phase)   │
│  ├── FDARuleEngine (checks compliance)           │
│  ├── TrialSimulator (simulates trial outcomes)   │
│  ├── TrialJudge (multi-layer verification)       │
│  └── EpisodeLogger (saves transcripts)           │
└─────────────────────────────────────────────────┘
```

Think of EpisodeManager as a **conductor** of an orchestra. It doesn't play any instrument itself — it coordinates all the other components to work together.

## The Two Ways to Use the Environment

### Way 1: Direct Python (for Training)

During training, we call the environment directly from Python — no network, no HTTP:

```python
# From server/environment.py
class Environment(BaseEnvironment):
    def __init__(self):
        self._manager = EpisodeManager()
    
    def reset(self, seed=None):
        obs = self._manager.reset(seed=seed)
        return obs
    
    def step(self, action):
        obs, reward, done, info = self._manager.step(action)
        return obs
    
    def step_full(self, action):
        """Extended version that returns reward + done flag too"""
        obs, reward, done, info = self._manager.step(action)
        return obs, reward.model_dump(), done, info
```

> **Design Decision Box: Why two methods?**
>
> `step()` returns only the observation (required by OpenEnv interface).
> `step_full()` returns everything: observation + reward breakdown + done flag + info.
> Training scripts need the full information, so they use `step_full()`.
> The OpenEnv interface only requires `step()`, so we keep both for compatibility.

### Way 2: HTTP API (for Deployment / Testing)

For deployment on HuggingFace Spaces, we wrap the environment in a FastAPI web server:

```python
# From server/app.py
@app.post("/reset")
def reset(body: ResetRequest):
    return _manager.reset(seed=body.seed)

@app.post("/step")
def step(action: TrialAction):
    obs, reward, done, info = _manager.step(action)
    return StepResponse(observation=obs, reward=reward.model_dump(), done=done, info=info)

@app.get("/ping")
def ping():
    return {"status": "ok"}
```

Same EpisodeManager, two different interfaces:
- **Direct Python:** Fast, no overhead, used during training
- **FastAPI HTTP:** Accessible from browsers and external agents, used for demos

## The Reset/Step Lifecycle

Every episode follows the same lifecycle. Let's trace through it step by step.

### Reset (Starting a New Episode)

When `reset(seed=42)` is called, here's what happens inside EpisodeManager:

```
Step 1: SELECT SCENARIO (CurriculumController)
   ├── Check current curriculum tier (e.g., tier 1 = beginner)
   ├── Pick the appropriate scenario (e.g., solid_tumor_chemo)
   └── At expert tier (>0.80), AdversarialDesigner generates a targeted scenario

Step 2: RANDOMIZE PARAMETERS (NoiseModel)
   ├── Create a seeded random number generator (RNG)
   ├── Budget: $10M × random factor (±30%) → maybe $7.8M or $12.3M
   ├── Time: 540 days × random factor (±20%) → maybe 450 or 640 days
   ├── Dropout rate: randomized ±15%
   └── Placebo response: randomized ±20%

Step 3: SAMPLE HIDDEN GROUND TRUTH
   ├── true_effect_size: random value in [0.25, 0.55] → e.g., 0.38
   ├── true_side_effect_rate: random in [0.15, 0.35] → e.g., 0.22
   ├── true_placebo_response: random in [0.05, 0.15] → e.g., 0.09
   └── true_dropout_rate: random in [0.05, 0.15] → e.g., 0.11

Step 4: BUILD HIDDEN STATE (TrialLatentState)
   ├── All hidden ground truth values
   ├── budget_remaining = randomized budget
   ├── time_remaining_days = randomized time
   ├── patients_enrolled = 0
   ├── All milestone flags = False
   ├── episode_phase = "literature_review"
   └── action_history = []

Step 5: CLEAR CACHES, INITIALIZE LOGGER

Step 6: GENERATE INITIAL OBSERVATION (OutputGenerator)
   └── Returns what the agent sees (scenario description, empty phase data)
```

Here's the actual code (simplified from `server/episode_manager.py`):

```python
def reset(self, seed=None):
    # Step 1: Pick scenario
    scenario = select_scenario(self._curriculum_tier, scenario_rng)
    
    # Step 2: Randomize with noise model
    noise_model = NoiseModel(seed=resolved_seed)
    randomized = noise_model.randomize(scenario)
    
    # Step 3: Sample hidden truth
    true_effect = float(rng.uniform(effect_lo, effect_hi))
    true_side = float(rng.uniform(side_lo, side_hi))
    
    # Step 4: Build hidden state
    self._latent = TrialLatentState(
        true_effect_size=true_effect,
        budget_remaining=randomized.budget_usd,
        patients_enrolled=0,
        phase_i_complete=False,
        # ... all other fields ...
    )
    
    # Step 5-6: Initialize logger, generate observation
    self._logger = EpisodeLogger(episode_id=self._episode_id)
    return output_gen.generate(latent=self._latent, ...)
```

### Step (Processing One Action)

When `step(action)` is called:

```
Step 1: CHECK FDA COMPLIANCE
   ├── Is this action allowed in the current phase?
   ├── Are all prerequisites met?
   ├── Does it violate any hard rules?
   └── If invalid → return negative reward, don't mutate state

Step 2: UPDATE HIDDEN STATE (TransitionEngine)
   ├── Subtract cost from budget (e.g., dose escalation costs $50,000)
   ├── Subtract time (e.g., dose escalation takes 90 days)
   ├── If action is ENROLL_PATIENTS → add patients, cost per patient
   ├── Set milestone flags (e.g., RUN_DOSE_ESCALATION → phase_i_complete = True)
   ├── Record adverse events (stochastic, based on true side effect rate)
   └── Degrade data quality on soft violations (low confidence → more noise)

Step 3: DETECT CLINICAL PHASE (PhaseDetector)
   ├── Classify action into phase (e.g., "monitoring", "analysis")
   └── Check if phase order is correct (no skipping, no regression)

Step 4: SIMULATE TRIAL RESULT (TrialSimulator)
   ├── Calculate statistical power based on current enrollment + true effect
   ├── Compute p-value with noise
   └── Generate confidence intervals

Step 5: COMPUTE REWARD (RewardComputer)
   ├── 8 decomposed components
   ├── + shaping bonus: γ·(φ(s') − φ(s))
   └── Total reward for this step

Step 6: JUDGE VERIFICATION (TrialJudge)
   ├── Layer 1: Programmatic checks (power, p-value, FDA)
   ├── Layer 2: Persona-scaled feedback
   └── Overconfidence penalty if applicable

Step 7: CHECK IF DONE
   ├── trial_complete = True? → done!
   ├── budget_remaining ≤ 0? → done! (failure)
   ├── time_remaining ≤ 0? → done! (failure)
   └── steps ≥ max_steps? → done! (timeout)

Step 8: LOG TRANSCRIPT
   └── Save action + observation + reward + hidden state to JSONL

Step 9: RETURN (observation, reward_breakdown, done, info)
```

## Data Models: What Flows Through the System

Our project uses **Pydantic models** — Python classes that enforce data types. If you've never used Pydantic, think of them as Python dataclasses with automatic validation.

### TrialAction (What the agent sends)

```python
class TrialAction(BaseModel):
    action_type: ActionType       # Which of 19 actions? (enum)
    parameters: dict[str, Any]    # Details like {"sample_size": 200}
    justification: str            # Why did you choose this? (text)
    confidence: float             # How sure are you? (0.0 to 1.0)
```

**Example action:**
```json
{
    "action_type": "set_sample_size",
    "parameters": {"sample_size": 200},
    "justification": "Based on effect size ~0.35, need N=200 for 80% power",
    "confidence": 0.75
}
```

### TrialObservation (What the agent receives)

```python
class TrialObservation(BaseModel):
    scenario_description: str      # "EGFR+ solid tumour chemotherapy..."
    phase_data: dict               # Noisy experimental results
    resource_status: dict          # {budget_remaining, time_remaining, patients_enrolled}
    rule_violations: list[str]     # Any violations from last action
    available_actions: list[str]   # What actions are valid right now
    steps_taken: int               # Current step number
    max_steps: int                 # Maximum steps (100)
    hint: str                      # Only populated at junior difficulty
    done: bool                     # Is the episode over?
    reward: float                  # Reward from last action
```

### TrialLatentState (Hidden from the agent)

```python
class TrialLatentState(BaseModel):
    # Hidden drug properties — the "truth" the agent must discover
    true_effect_size: float            # Does the drug work? By how much?
    true_side_effect_rate: float       # How dangerous is it?
    true_responder_population: str     # Who does it help? (e.g., "EGFR+")
    true_dose_response: dict           # How does effect change with dose?
    placebo_response_rate: float       # How much does placebo help?
    dropout_rate: float                # How many patients quit?
    
    # Resources
    budget_remaining: float
    time_remaining_days: int
    patients_enrolled: int
    
    # Progress milestones
    phase_i_complete: bool
    mtd_identified: bool
    effect_estimated: bool
    protocol_submitted: bool
    interim_complete: bool
    trial_complete: bool
    
    # Tracking
    adverse_events: int
    episode_phase: str
    action_history: list[str]
    seed: int
```

**Key insight:** The agent NEVER sees TrialLatentState directly. It only sees TrialObservation, which is a **noisy, filtered version** of the latent state. This is partial observability in action.

### The Configuration Chain

```python
# From server/config.py — runtime settings from environment variables
class Settings(BaseSettings):
    log_path: Path = Path("./logs")          # Where to save logs
    default_seed: int | None = None          # Reproducibility seed
    host: str = "0.0.0.0"                    # Server host
    port: int = 8000                         # Server port
    curriculum_start_tier: int = 0           # Starting difficulty
    judge_llm_model: str | None = None       # Optional LLM judge model
    judge_llm_api_key: str | None = None     # LLM judge API key
```

Configuration flows through environment variables, which means you can change behavior without touching code:

```bash
# Start easy with hints
export CURRICULUM_START_TIER=0

# Start at expert level, no hints
export CURRICULUM_START_TIER=4

# Enable LLM-based judge (optional, costs money)
export JUDGE_LLM_MODEL=gpt-4o-mini
export JUDGE_LLM_API_KEY=sk-...
```

## File Structure

Here's how the code is organized:

```
server/
├── __init__.py                   # Package marker
├── app.py                        # FastAPI web server (HTTP interface)
├── config.py                     # Runtime settings (env vars)
├── environment.py                # OpenEnv wrapper (direct Python interface)
├── episode_manager.py            # ★ Central orchestrator
├── judge.py                      # Trial judge (verification)
├── logger.py                     # Episode logging (JSONL + CSV)
├── noise_model.py                # Domain randomization
├── phase_detector.py             # Clinical phase classification
├── dashboard.py                  # Web dashboard for monitoring
│
├── simulator/
│   ├── trial_simulator.py        # Simulates trial outcomes
│   ├── transition_engine.py      # Updates hidden state
│   ├── output_generator.py       # Generates noisy observations
│   └── power_calculator.py       # Statistical power math
│
├── curriculum/
│   ├── controller.py             # 5-tier curriculum logic
│   ├── scenarios.py              # Scenario definitions
│   └── adversarial_designer.py   # Expert-level scenario generation
│
├── reward/
│   ├── reward_computer.py        # 8-component reward calculation
│   └── shaping.py                # Potential-based reward shaping
│
└── rules/
    ├── fda_rules.py              # FDA compliance checking
    └── prerequisite_rules.py     # Action prerequisite checking
```

> **Design Decision Box: Why So Many Small Files?**
>
> Each file has ONE responsibility. This is called the **Single Responsibility Principle**. Benefits:
> 1. **Testable:** Each component can be tested in isolation (see `tests/` folder)
> 2. **Debuggable:** When reward seems wrong, you know to look in `reward_computer.py`
> 3. **Replaceable:** Want to change how phases are detected? Only edit `phase_detector.py`
>
> This is the opposite of putting everything in one big file. With 249 tests across 13 test files, every component is independently verified.

---

## Chapter 5 Glossary

| Keyword | Definition |
|---------|-----------|
| **EpisodeManager** | The central orchestrator that coordinates all components |
| **FastAPI** | A Python web framework for building HTTP APIs |
| **HTTP/REST API** | A way for programs to communicate over the network |
| **WebSocket** | A protocol for real-time, two-way communication |
| **Pydantic** | A Python library for data validation using type hints |
| **BaseModel** | Pydantic's base class for validated data models |
| **Lifecycle** | The sequence of operations from start to finish (reset → step → done) |
| **Single Responsibility Principle** | Each module/class should do only one thing |
| **Environment Variable** | A setting passed to a program from outside (not hard-coded) |
| **Seeded RNG** | A random number generator initialized with a seed for reproducibility |
