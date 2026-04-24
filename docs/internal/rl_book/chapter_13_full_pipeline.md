# Chapter 13: The Full Pipeline — From Reset to Weight Update

## Putting It All Together

In previous chapters, we learned each piece individually. Now let's trace through **one complete training iteration** — from environment reset to model weight update. This is the "assembly line" of our project.

## One Complete Training Episode 

### Phase 1: Environment Reset

```python
# 1. train.py starts a new episode
ep_seed = 42 + episode_number  # Deterministic seed for reproducibility
obs = env.reset(seed=ep_seed)
```

**Inside env.reset():**
```
CurriculumController → selects scenario (e.g., solid_tumor_chemo at Tier 1)
NoiseModel(seed=42) → randomizes parameters:
  budget: $10M × 1.13 = $11.3M
  time: 540 × 0.88 = 475 days
  dropout: 0.12
  placebo: 0.09

Hidden ground truth sampled:
  true_effect_size = 0.38
  true_side_effect_rate = 0.22
  true_responder_population = "all"

TrialLatentState created (all hidden)
TrialState created (metadata for training loop)
OutputGenerator produces initial TrialObservation
```

**Agent receives:**
```json
{
    "scenario_description": "EGFR+ solid tumour chemotherapy...",
    "phase_data": {"current_phase": "literature_review"},
    "resource_status": {"budget_remaining": 11300000, "time_remaining_days": 475},
    "available_actions": ["set_primary_endpoint", "observe_safety_signal", "estimate_effect_size"],
    "steps_taken": 0,
    "max_steps": 100,
    "hint": "Start by defining your primary endpoint and study design.",
    "done": false,
    "reward": 0.0
}
```

### Phase 2: Model Generates Responses

```python
# 2. Build prompt from observation
prompt = f"""You are designing a clinical trial.

Scenario: {obs.scenario_description}
Phase data: {json.dumps(obs.phase_data)}
Resources: {json.dumps(obs.resource_status)}
Available actions: {obs.available_actions}
Steps taken: {obs.steps_taken}/{obs.max_steps}
Hint: {obs.hint}

Respond with a JSON object: 
{{"action_type": "...", "parameters": {{}}, "justification": "...", "confidence": 0.8}}"""

# 3. Tokenize
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
# Tokens: [2048, 539, 1711, 306, ...] (roughly 400 tokens)

# 4. Generate 8 responses
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    num_return_sequences=8,  # ← 8 parallel generations
    do_sample=True,
    temperature=0.7,
)
```

**The model produces 8 different responses:**
```
Gen 1: '{"action_type": "set_primary_endpoint", "parameters": {"endpoint": "PFS"}, 
         "justification": "PFS is standard for oncology", "confidence": 0.85}'

Gen 2: '{"action_type": "set_primary_endpoint", "parameters": {"endpoint": "OS"}, 
         "justification": "Overall survival is gold standard", "confidence": 0.70}'

Gen 3: '{"action_type": "estimate_effect_size", "parameters": {}, 
         "justification": "Need to know drug efficacy first", "confidence": 0.60}'

Gen 4: 'I would recommend starting with...'  ← Invalid JSON!

Gen 5-8: (various valid/invalid responses)
```

### Phase 3: Parse Actions and Step Through Environment

```python
# 5. For the primary generation (Gen 1), parse and step
response_text = decode(outputs[0])
action = _build_action_from_text(response_text, step_idx=0)

# Successfully parsed:
# TrialAction(action_type="set_primary_endpoint", parameters={"endpoint": "PFS"},
#             justification="PFS is standard for oncology", confidence=0.85)

# 6. Environment processes the action
next_obs, reward_dict, done, info = env.step_full(action)
```

**Inside env.step_full():**
```
FDA Check → "set_primary_endpoint" allowed in "literature_review"? ✓

TransitionEngine:
  budget: $11.3M → $11.295M (-$5,000)
  time: 475 → 468 days (-7 days)
  action_history: ["set_primary_endpoint"]

PhaseDetector → "design" phase (first step, order correct ✓)

TrialSimulator → calculates power=0.0 (no patients yet)

RewardComputer:
  r_validity: +1.0 (valid action)
  r_ordering: +0.2 (correct phase order)
  r_info_gain: 0.0 (set_ actions don't generate data)
  r_efficiency: +2.0 × ($11.295M / $11.3M) = +1.999
  r_novelty: +0.2 (first time using this action)
  r_penalty: 0.0
  r_terminal: 0.0 (not done yet)
  r_shaping: 0.0 (no milestone completed)
  
  TOTAL: +3.399

OutputGenerator → noisy observation with updated state
```

### Phase 4: Continue Episode (55-100 Steps)

The loop continues, with the agent taking actions and the environment responding:

```
Step  0: set_primary_endpoint     → r=+3.4   budget=$11.295M
Step  1: set_sample_size(200)     → r=+3.3   budget=$11.293M
Step  2: set_inclusion_criteria   → r=+3.2   budget=$11.290M
Step  3: set_dosing_schedule      → r=+3.1   budget=$11.280M
Step  4: set_control_arm          → r=+3.0   budget=$11.275M
Step  5: set_randomization_ratio  → r=+2.9   budget=$11.273M
Step  6: set_blinding             → r=+2.8   budget=$11.269M
Step  7: enroll_patients(50)      → r=+2.5   budget=$10.769M  (50 × $10K)
Step  8: run_dose_escalation      → r=+2.7   budget=$10.719M  (Phase I ✓)
Step  9: observe_safety_signal    → r=+2.3   budget=$10.704M
Step 10: estimate_effect_size     → r=+2.6   budget=$10.684M  (effect estimated ✓)
Step 11: add_biomarker_strat      → r=+2.4   budget=$10.659M
Step 12: enroll_patients(100)     → r=+1.8   budget=$9.659M   (Phase III patients)
Step 13: run_interim_analysis     → r=+2.9   budget=$9.629M   (interim ✓)
Step 14: run_primary_analysis     → r=+14.2  budget=$9.579M   (trial complete! ✓)
         ↑ Includes terminal rewards:
           r_terminal_success: +10.0 (p < 0.05, trial succeeded!)
           r_terminal_calibration: +3.8 (CI close to true effect)
Step 15: synthesize_conclusion    → r=+1.5
Step 16: submit_to_fda_review     → r=+1.2   budget=$9.479M   (done!)

EPISODE COMPLETE: 17 steps, total_reward = +51.7, outcome = "success"
```

### Phase 5: Log Everything

```python
# 7. Each step was logged to JSONL
# logs/episode_transcripts/{episode_id}.jsonl contains:
{
    "episode_id": "0041cb86-0430-41f1-8ab7-cbed00d84de4",
    "step": 0,
    "action": {"action_type": "set_primary_endpoint", ...},
    "observation": {"scenario_description": "...", ...},
    "reward_breakdown": {"r_validity": 1.0, "r_ordering": 0.2, ...},
    "total_reward": 3.399,
    "phase_detected": "design",
    "phase_order_correct": true,
    "hidden_state_snapshot": {"true_effect_size": 0.38, ...},
    "timestamp": "2026-04-25T10:30:00Z"
}

# 8. Episode summary logged to CSV
# logs/reward_log.csv:
# episode,seed,total_reward,steps,terminal_outcome,timestamp
# 0,42,51.7,17,success,2026-04-25T10:30:17Z

# 9. Curriculum logged
# logs/curriculum_log.csv:
# episode_id,tier,scenario_id,difficulty,outcome,new_tier
```

### Phase 6: GRPO Weight Update

```python
# 10. After collecting experiences from all 8 generations:
experiences = [
    {"prompt": prompt, "response": gen_1_text, "reward": 51.7},
    {"prompt": prompt, "response": gen_2_text, "reward": 38.2},
    {"prompt": prompt, "response": gen_3_text, "reward": 12.5},
    {"prompt": prompt, "response": gen_4_text, "reward": -2.0},  # invalid JSON
    ...
]

# 11. Compute advantages (relative to group mean)
mean_reward = mean([51.7, 38.2, 12.5, -2.0, ...]) = 18.3
advantages = [51.7-18.3, 38.2-18.3, 12.5-18.3, -2.0-18.3, ...]
           = [+33.4,    +19.9,     -5.8,      -20.3,     ...]

# 12. Update LoRA weights
# Gen 1 had advantage +33.4 → STRONGLY increase probability of this response
# Gen 4 had advantage -20.3 → STRONGLY decrease probability of this response
# The model's LoRA weights are adjusted accordingly
```

### Phase 7: Curriculum Advancement Check

```python
# 13. Record this episode's outcome for curriculum tracking
self._episode_history.append(True)  # success=True

# 14. Check if agent should advance
metrics = EpisodeMetrics(success=True, episode_history=self._episode_history)
self._curriculum_tier = advance_curriculum(self._curriculum_tier, metrics)
# If 7 of last 10 episodes succeeded (70%) → tier advances from 1 to 2!
```

## The Logging Trail

After training, you have a complete audit trail:

```
logs/
├── reward_log.csv              ← One row per episode: reward, steps, outcome
├── curriculum_log.csv          ← One row per episode: tier, scenario, advancement
└── episode_transcripts/
    ├── 0041cb86-...-.jsonl     ← Full step-by-step transcript
    ├── 022f4b8d-...-.jsonl     ← Every action, observation, reward, hidden state
    └── ... (one file per episode)

outputs/
├── grpo/
│   ├── checkpoint-10/          ← Model weights saved every 10 episodes
│   ├── checkpoint-20/
│   └── ...
├── training_summary.json       ← Final statistics
└── reward_curve.png            ← Reward over time plot
```

This complete trail enables:
1. **Debugging:** "Episode 42 failed. Let me replay the transcript and see where things went wrong."
2. **Evaluation:** "Compare reward curves between different training runs."
3. **Analysis:** "At which curriculum tier does the agent start identifying biomarkers?"

## The Complete Data Flow Diagram

```
User runs: python train.py --model-path Qwen/Qwen2.5-7B-Instruct --episodes 300

For each episode:
  ┌─────────────┐     seed     ┌──────────────────┐
  │  train.py   │────────────→ │  Environment     │
  │             │              │  ├─ Curriculum    │ → selects scenario
  │  (GRPO +   │              │  ├─ NoiseModel    │ → randomizes params
  │   LoRA +   │              │  ├─ EpisodeManager│ → creates hidden state
  │   vLLM)    │              │  │                │
  │             │  observation │  │                │
  │             │←────────────│  │                │
  │             │              │  │                │
  │  Model     │    action    │  │                │
  │  generates │────────────→ │  ├─ FDA Rules     │ → validates action
  │  8 actions │              │  ├─ Transition    │ → updates hidden state
  │             │              │  ├─ OutputGen    │ → creates noisy obs
  │             │ obs+reward  │  ├─ Simulator    │ → calculates stats
  │             │←────────────│  ├─ RewardComp   │ → 8 reward components
  │             │              │  ├─ PhaseDetect  │ → classifies phase
  │             │              │  ├─ Judge        │ → verifies + hints
  │  Compare 8 │              │  └─ Logger       │ → saves transcript
  │  rewards,  │              └──────────────────┘
  │  compute   │
  │  advantages│
  │  update    │
  │  LoRA      │
  │  weights   │
  └─────────────┘
       │
       ▼
  Next episode (with improved weights)
```

---

## Chapter 13 Glossary

| Keyword | Definition |
|---------|-----------|
| **Training Iteration** | One complete episode: reset → steps → terminal → weight update |
| **Rollout** | Running the agent through an episode to collect experiences |
| **Experience** | A (state, action, reward) tuple collected during rollout |
| **Weight Update** | Adjusting model parameters based on collected experiences |
| **Checkpoint** | A saved copy of model weights at a point during training |
| **Training Summary** | JSON file with final statistics (mean reward, episodes, etc.) |
| **Episode Transcript** | JSONL file with step-by-step trace of an entire episode |
| **Reward Curve** | Plot showing how reward changes over training episodes |
| **Audit Trail** | Complete record of all decisions and outcomes for debugging |
