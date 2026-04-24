# Chapter 2: Reinforcement Learning — Teaching by Doing

## The Core Idea

Remember the dog analogy from Chapter 1? Let's formalize it into the language that RL researchers use. Don't worry — every term gets a plain English explanation.

### The RL Loop

Every RL system follows this loop, over and over:

```
        ┌──────────────┐
        │              │
        │  ENVIRONMENT │ ← The "world" (clinical trial simulator)
        │              │
        └──────┬───────┘
               │
          Observation + Reward
          ("Here's what happened
           and how well you did")
               │
               ▼
        ┌──────────────┐
        │              │
        │    AGENT     │ ← The "learner" (language model)
        │              │
        └──────┬───────┘
               │
            Action
           ("I want to
            do this next")
               │
               ▼
        ┌──────────────┐
        │              │
        │  ENVIRONMENT │ ← Takes the action, updates, sends new observation
        │              │
        └──────────────┘
```

This repeats for many "steps" until the episode ends (the trial is complete, budget runs out, or time expires).

### Real-World Analogy: Learning to Cook

Imagine you're learning to cook without a recipe book:

1. **Observation:** You see ingredients on the counter (flour, eggs, butter, sugar)
2. **Action:** You decide to mix flour + eggs
3. **Reward:** The result tastes terrible (reward: -1)
4. **New Observation:** You see the mixture plus remaining ingredients
5. **Action:** You add sugar and butter
6. **Reward:** Better! (reward: +0.5)
7. ... many attempts later ...
8. You've learned to make a cake

You never read a recipe. You just tried things and learned from feedback. That's RL.

## The Five Core Concepts

### 1. State (s) — "What the world looks like right now"

The state is a complete description of the current situation. In our project, the state has two parts:

**What the agent sees** (Observation):
```python
# From our project's models.py
class TrialObservation:
    scenario_description: str        # "EGFR+ solid tumour chemotherapy..."
    phase_data: dict                 # Noisy experiment results
    resource_status: dict            # Budget remaining, time remaining
    rule_violations: list[str]       # What went wrong
    available_actions: list[str]     # What you CAN do right now
    steps_taken: int                 # How many actions so far
    max_steps: int                   # Maximum allowed
    hint: str                        # Helper hint (only at easy difficulty)
    done: bool                       # Is the episode over?
    reward: float                    # How good was your last action?
```

**What the agent CANNOT see** (Hidden State):
```python
# The agent never sees this — it's the "ground truth"
class TrialLatentState:
    true_effect_size: float          # Does the drug ACTUALLY work? By how much?
    true_side_effect_rate: float     # How dangerous is it REALLY?
    true_responder_population: str   # Who does it ACTUALLY help? (e.g., "EGFR+")
    budget_remaining: float          # Real remaining budget
    # ... more hidden values ...
```

> **Why is some state hidden?** In real clinical trials, you don't know the true effect of a drug — that's what the trial is trying to determine! Your measurements are noisy and uncertain. Our simulator mimics this realistically.

### 2. Action (a) — "What the agent decides to do"

An action is a decision the agent makes. In our project, the agent can take 19 different actions:

```python
class ActionType(str, Enum):
    SET_PRIMARY_ENDPOINT = "set_primary_endpoint"       # What are we measuring?
    SET_SAMPLE_SIZE = "set_sample_size"                  # How many patients?
    SET_INCLUSION_CRITERIA = "set_inclusion_criteria"    # Who can join the trial?
    SET_EXCLUSION_CRITERIA = "set_exclusion_criteria"    # Who's excluded?
    SET_DOSING_SCHEDULE = "set_dosing_schedule"          # How much drug, how often?
    SET_CONTROL_ARM = "set_control_arm"                  # What does the comparison group get?
    SET_RANDOMIZATION_RATIO = "set_randomization_ratio"  # How to split patients?
    SET_BLINDING = "set_blinding"                        # Who knows who gets what?
    RUN_DOSE_ESCALATION = "run_dose_escalation"          # Test increasing drug doses
    OBSERVE_SAFETY_SIGNAL = "observe_safety_signal"      # Check for side effects
    ESTIMATE_EFFECT_SIZE = "estimate_effect_size"        # Estimate how well drug works
    RUN_INTERIM_ANALYSIS = "run_interim_analysis"        # Mid-trial check
    MODIFY_SAMPLE_SIZE = "modify_sample_size"            # Change patient count
    ADD_BIOMARKER_STRATIFICATION = "add_biomarker_stratification"  # Split by genetics
    SUBMIT_TO_FDA_REVIEW = "submit_to_fda_review"        # Regulatory submission
    REQUEST_PROTOCOL_AMENDMENT = "request_protocol_amendment"  # Change the plan
    RUN_PRIMARY_ANALYSIS = "run_primary_analysis"        # Final statistical test
    SYNTHESIZE_CONCLUSION = "synthesize_conclusion"      # Write up findings
    ENROLL_PATIENTS = "enroll_patients"                   # Recruit patients
```

Each action also comes with:
- **Parameters:** Details (e.g., `{"sample_size": 200}`)
- **Justification:** Why the agent chose this (a text explanation)
- **Confidence:** How sure the agent is (0.0 to 1.0)

### 3. Reward (r) — "How good was that decision?"

After each action, the environment gives the agent a number (the reward) indicating how good or bad that action was.

```
Good action:  +1.0  ("You correctly ordered a safety check before enrolling patients")
Bad action:   -1.0  ("You tried to submit to FDA without completing Phase I — violation!")
Great outcome: +10.0 ("The trial succeeded — drug works, patients benefited!")
Terrible:     -2.0  ("You ran out of budget before finishing the trial")
```

The agent's goal is to **maximize the total reward across all steps** in an episode. We'll explore our reward system in extreme detail in Chapter 7.

### 4. Policy (π) — "The agent's strategy"

The policy is the agent's decision-making strategy. It's a function that takes an observation and outputs an action:

```
π(observation) → action
```

In our project, the policy is a **language model** (Qwen 2.5-7B). Given a text description of the current trial state, it generates a JSON action:

```
Input (observation as text):
"You are designing a clinical trial.
 Scenario: EGFR+ solid tumour chemotherapy...
 Phase data: {"current_phase": "design", "patients_enrolled": 0}
 Resources: {"budget_remaining": 8000000, "time_remaining_days": 365}
 Available actions: ["set_primary_endpoint", "set_sample_size", ...]
 Steps taken: 3/100"

Output (action as JSON):
{"action_type": "set_sample_size", "parameters": {"sample_size": 200},
 "justification": "Based on estimated effect size of 0.31, need N=200 for 80% power",
 "confidence": 0.75}
```

**Training RL = improving the policy.** We start with a language model that has no idea how to design clinical trials. Through many episodes of trial-and-error, the model learns which actions lead to high rewards.

### 5. Episode — "One complete trial from start to finish"

An episode is one complete run-through of the environment. In our project:

```
Episode starts: Agent sees a new drug scenario
  ↓
Step 1:  Agent sets the primary endpoint
Step 2:  Agent sets the sample size
Step 3:  Agent sets inclusion criteria
...
Step 40: Agent runs the primary analysis
Step 41: Agent synthesizes conclusions
  ↓
Episode ends: Trial succeeded (reward: +12) or failed (reward: -2)
```

Each episode is 55–100 steps. After each episode, the agent's weights are updated to make it slightly better at the task.

## The Big Questions

### How does the agent actually improve?

Think of it this way. The agent plays many episodes:

- **Episode 1:** Random actions, terrible score (-2.0)
- **Episode 2:** Slightly less random, still bad (-1.5)
- ...
- **Episode 50:** Starting to learn the workflow (+3.0)
- ...
- **Episode 200:** Reliably designing good trials (+10.0)

After each batch of episodes, we compare the outcomes:
- "When I did action X in situation Y, I got a high reward. I should do more of that."
- "When I did action Z in situation W, I got punished. I should avoid that."

The specific algorithm we use for this improvement is called **GRPO** (explained in detail in Chapter 11).

### What's the difference between RL and just... programming the agent?

Great question! You might think: "Why not just write a program that follows the correct clinical trial workflow?"

```python
# Why not just do this?
def design_trial(scenario):
    set_primary_endpoint("progression_free_survival")
    set_sample_size(200)
    set_inclusion_criteria(...)
    # ... hard-coded steps ...
```

Three reasons RL is better:

1. **Context matters.** The right sample size depends on the drug's effect size, which you only discover during the trial. A fixed program can't adapt.

2. **Partially observable.** You never see the true drug effect — only noisy measurements. The agent must reason under uncertainty.

3. **Combinatorial explosion.** 19 action types, each with parameters, over 55–100 steps = trillions of possible action sequences. No human can write rules for all cases.

RL discovers strategies that **no human would hard-code** — like recognizing that in a depression trial with high placebo response, you should focus on enriching for severe patients rather than increasing sample size.

## Types of RL Algorithms

There are many RL algorithms. Here are the main families:

### Value-Based Methods
"Learn how good each state is, then pick the action that leads to the best state."

Example: **Q-Learning** (used in simple games like Atari)

```
Q-table says: In state "2 patients enrolled, budget=$1M", 
  action "enroll_more" has value 3.5
  action "run_analysis" has value 1.2
→ Pick "enroll_more" because 3.5 > 1.2
```

**Why we DON'T use this:** Our state space is text (infinite possibilities). You can't make a table for every possible trial description.

### Policy Gradient Methods ← What We Use
"Directly adjust the policy (strategy) to produce better actions."

Instead of learning a value table, we directly adjust the language model's weights so it generates better trial design decisions.

```
Before training:  model generates random/bad actions
After training:   model generates expert-level trial designs
```

Our specific algorithm is **GRPO (Group Relative Policy Optimization)** — a policy gradient method designed specifically for language models. Chapter 11 covers this in depth.

### Actor-Critic Methods
"Combine both: learn values AND adjust the policy."

Popular algorithms: PPO (Proximal Policy Optimization), A2C, SAC.

**Why we chose GRPO over PPO:**
- PPO needs a separate "critic" network that estimates value. This is hard to set up for text-based environments.
- GRPO compares multiple generations directly ("which of these 8 attempts was best?"). This works naturally with language models, which can easily generate multiple different responses.
- GRPO is from TRL (Transformer Reinforcement Learning library by HuggingFace), the standard toolkit for RL with language models.

> **Design Decision Box: Why GRPO over PPO?**
> 
> PPO (Proximal Policy Optimization) is the "classic" RL algorithm for language models (used in ChatGPT's RLHF). But PPO requires training a separate value function (critic), which adds complexity and compute cost. GRPO instead generates 8 parallel responses, scores all of them with our reward function, and uses relative rankings to compute advantages. This "group relative" approach eliminates the critic entirely, reducing GPU memory by ~30% and simplifying the training code significantly.

## The Exploration-Exploitation Dilemma

One of the deepest problems in RL:

- **Exploitation:** Do what you already know works (safe, but you might miss better strategies)
- **Exploration:** Try new things (risky, but might discover superior strategies)

**Cooking analogy:** You know how to make a decent pasta. Do you:
- (A) Keep making pasta every night? (exploitation)
- (B) Try a risky new Thai recipe? (exploration)

If you always exploit, you never discover Thai food. If you always explore, you eat terrible experiments every night.

In our project, exploration is handled by:
1. **Temperature sampling** during generation (randomness in the model's choices)
2. **Novelty reward** (+0.2 for trying an action type you haven't used yet)
3. **Curriculum learning** (gradually increasing difficulty keeps the challenge fresh)

---

## A Simple RL Example in Python

Here's the simplest possible RL system, so you can see the loop in code:

```python
import random

# Super simple environment: guess a number between 1-10
class NumberGuessingEnv:
    def __init__(self):
        self.secret = random.randint(1, 10)
    
    def reset(self):
        self.secret = random.randint(1, 10)
        return "Guess a number 1-10"  # observation
    
    def step(self, action):
        if action == self.secret:
            return "Correct!", +10.0, True   # obs, reward, done
        elif abs(action - self.secret) <= 2:
            return "Close!", +1.0, False
        else:
            return "Far away", -1.0, False

# Super simple agent: remembers what worked
class SimpleAgent:
    def __init__(self):
        # For each number, track total reward
        self.scores = {i: 0.0 for i in range(1, 11)}
        self.counts = {i: 0 for i in range(1, 11)}
    
    def choose_action(self):
        # 20% of the time, explore (random)
        if random.random() < 0.2:
            return random.randint(1, 10)
        # 80% of the time, exploit (pick best known)
        best = max(self.scores, key=lambda x: self.scores[x] / max(self.counts[x], 1))
        return best
    
    def learn(self, action, reward):
        self.scores[action] += reward
        self.counts[action] += 1

# Training loop
env = NumberGuessingEnv()
agent = SimpleAgent()

for episode in range(1000):
    obs = env.reset()
    total_reward = 0
    for step in range(5):  # max 5 guesses per episode
        action = agent.choose_action()
        obs, reward, done = env.step(action)
        agent.learn(action, reward)
        total_reward += reward
        if done:
            break
```

Our project works exactly like this loop — but the environment is a clinical trial simulator, and the agent is a 7-billion-parameter language model.

---

## Chapter 2 Glossary

| Keyword | Definition |
|---------|-----------|
| **State (s)** | Complete description of the current situation |
| **Observation** | What the agent actually sees (may be partial/noisy) |
| **Action (a)** | A decision the agent makes |
| **Reward (r)** | A number indicating how good the action was |
| **Policy (π)** | The agent's strategy: maps observations to actions |
| **Episode** | One complete run from start to finish |
| **Step** | One action within an episode |
| **Exploration** | Trying new, uncertain actions to discover better strategies |
| **Exploitation** | Using known-good actions to maximize reward |
| **Policy Gradient** | A family of RL algorithms that directly optimize the policy |
| **PPO** | Proximal Policy Optimization — a popular RL algorithm |
| **GRPO** | Group Relative Policy Optimization — the algorithm we use |
| **Partial Observability** | When the agent can't see the full state |
| **Q-Learning** | A value-based RL algorithm (we don't use this) |
