# The OpenEnv Clinical Trial Textbook
### From Zero RL Knowledge to Building a Hackathon-Winning Environment

> **Author's note:** You are an intermediate Python programmer. You know functions, classes, loops, dicts, and have used libraries like NumPy. This textbook will take you from "what is RL?" to "I built and trained an RL environment for clinical trial design." Every concept uses real-life analogies first, then code. Viva questions are included for each chapter — these are the questions a hackathon judge or interviewer would ask you.

> **Living document:** This textbook is updated with every push. New chapters and viva questions are added as the project grows.

---

## Table of Contents

1. [What is Reinforcement Learning?](#chapter-1--what-is-reinforcement-learning)
2. [The RL Loop — Agent, Environment, Reward](#chapter-2--the-rl-loop--agent-environment-reward)
3. [Partially Observable Environments (POMDPs)](#chapter-3--partially-observable-environments-pomdps)
4. [Reward Engineering — The Art of Teaching Through Scores](#chapter-4--reward-engineering--the-art-of-teaching-through-scores)
5. [Curriculum Learning — Don't Throw a Kid Into Calculus](#chapter-5--curriculum-learning--dont-throw-a-kid-into-calculus)
6. [GRPO — The Training Algorithm](#chapter-6--grpo--the-training-algorithm)
7. [OpenEnv — The Framework We're Building On](#chapter-7--openenv--the-framework-were-building-on)
8. [Our Project — Clinical Trial Designer](#chapter-8--our-project--clinical-trial-designer)
9. [How Winners Built Their Environments](#chapter-9--how-winners-built-their-environments)
10. [Rules, Pitfalls, and Battle-Tested Lessons](#chapter-10--rules-pitfalls-and-battle-tested-lessons)
11. [Tech Stack and Deployment](#chapter-11--tech-stack-and-deployment)
12. [Pitch Strategy](#chapter-12--pitch-strategy)

---

## Chapter 1 — What is Reinforcement Learning?

### Real-life analogy

Imagine you're learning to cook without a recipe. You try adding salt — the dish tastes better (positive reward). You try adding sugar to soup — it tastes weird (negative reward). Over many attempts, you learn which actions lead to good food. Nobody gave you rules. You learned from the *consequences* of your actions.

That's reinforcement learning (RL).

### The formal version

RL is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent takes **actions**, the environment responds with **observations** and **rewards**, and the agent learns to maximize cumulative reward over time.

```
Agent: "I'll add 200mg dose to the trial"
Environment: "Here's what happened — 3 patients had side effects, effect size estimated at 0.15"
Reward: +0.3 (valid action, some information gained)
```

### How is RL different from other ML?

| Type | Learns from | Example |
|------|-------------|---------|
| Supervised Learning | Labeled examples (input → correct output) | "This image is a cat" |
| Unsupervised Learning | Patterns in data (no labels) | "These customers cluster together" |
| Reinforcement Learning | Trial and error + reward signal | "This action sequence led to a good outcome" |

Key difference: In RL, there's no "correct answer" dataset. The agent must *discover* good strategies by exploring.

### Why RL for LLMs?

Standard LLM training (next-token prediction) teaches the model to *sound right*. RL training teaches it to *be right*. An LLM trained with RL on clinical trial design doesn't just write plausible protocols — it writes protocols that actually pass statistical tests.

### Viva Questions — Chapter 1

1. **Q:** What's the difference between supervised learning and reinforcement learning?
   **A:** In supervised learning, you have labeled data telling you the right answer. In RL, there's no right answer given — the agent discovers it through trial, error, and reward signals.

2. **Q:** Why can't we just use supervised learning to train an LLM for clinical trials?
   **A:** We don't have a dataset of "perfect trial designs." Clinical trial design involves sequential decisions under uncertainty. RL lets the agent learn from the *outcomes* of its designs (did the trial detect the drug effect?) rather than memorizing past designs.

3. **Q:** Give a real-life example of reinforcement learning that isn't a game.
   **A:** A thermostat learning the best schedule: it tries different temperature settings (actions), measures comfort and energy cost (rewards), and learns the optimal heating/cooling pattern over time.

---

## Chapter 2 — The RL Loop — Agent, Environment, Reward

### Real-life analogy

Think of a driving test:
- **Agent:** You (the student driver)
- **Environment:** The road, traffic, signals
- **Action:** Turn left, brake, accelerate
- **Observation:** What you see — red light, car ahead, pedestrian crossing
- **Reward:** Examiner's score — +1 for smooth stop, -1 for running a red light
- **Episode:** One complete driving test (start to finish)
- **Step:** Each decision you make during the test

### The formal loop

```
1. Environment.reset()         → Starting observation (new episode begins)
2. Agent sees observation      → Decides an action
3. Environment.step(action)    → Returns (new_observation, reward, done)
4. Repeat 2-3 until done=True → Episode ends
5. Go to 1 for next episode   → Agent (hopefully) does better
```

### In Python (pseudocode)

```python
env = TrialEnvironment()

for episode in range(num_episodes):
    obs = env.reset()                      # new clinical scenario
    total_reward = 0
    done = False
    
    while not done:
        action = agent.decide(obs)         # LLM generates next action
        obs, reward, done = env.step(action)
        total_reward += reward
    
    agent.learn(total_reward)              # update weights based on outcome
```

### Key terms to remember

| Term | Meaning | In our project |
|------|---------|----------------|
| **State** | Full truth about the world (may be hidden) | True drug effect, true responder population |
| **Observation** | What the agent actually sees | Trial progress, safety signals, budget remaining |
| **Action** | What the agent does | `set_sample_size(n=200)`, `run_dose_escalation()` |
| **Reward** | Numeric feedback signal | +5.0 if trial detects true effect, -2.0 for underpowered design |
| **Episode** | One complete interaction (reset to done) | One full clinical trial design attempt |
| **Step** | One action-observation-reward cycle | One design decision |
| **Policy** | Agent's strategy (observation → action mapping) | The LLM's learned behavior |

### How many episodes?

An episode is one full trial design attempt (55–100 steps). During training with GRPO:
- Each training step generates 8 parallel rollouts (episodes)
- A typical training run: 50–100 training steps = **400–800 episodes total**
- On H100 with vLLM colocate, expect **30–80 usable episodes per hour** depending on model size
- With 2 days of onsite compute (April 25–26), realistically **200–600 total training episodes**
- Curriculum milestones are checked at episodes 15, 30, 40, and 80

### Viva Questions — Chapter 2

1. **Q:** What's the difference between state and observation?
   **A:** State is the complete truth about the world. Observation is what the agent can see. In our project, the true drug effect size is part of the state but hidden from the agent's observation — the agent must infer it from experimental data.

2. **Q:** Why do we need multiple episodes instead of just one really long one?
   **A:** Each episode is a fresh trial scenario. The agent needs many different scenarios (different drugs, different patient populations) to learn general strategies, not just memorize one specific solution.

3. **Q:** What happens when `done=True`?
   **A:** The episode ends. Terminal reward fires (did the trial succeed?), the episode transcript is saved, and the agent uses the outcome to update its policy.

4. **Q:** How many episodes do we expect during training?
   **A:** With GRPO generating 8 parallel rollouts per step and ~50–100 training steps, we get 400–800 episodes. On H100 onsite compute (April 25–26), realistically 200–600 total usable training episodes across 2 days.

---

## Chapter 3 — Partially Observable Environments (POMDPs)

### Real-life analogy

You're a doctor treating a patient. You can see their symptoms (fever, cough), run tests (blood work, X-ray), but you can never *directly see* the disease process inside their body. You must infer the hidden truth from partial information and decide treatment based on incomplete knowledge.

That's a **Partially Observable Markov Decision Process (POMDP)**.

### Why "partially observable"?

In many RL problems, the agent sees everything (like chess — all pieces are visible). But real professional tasks have hidden information:

| Domain | Hidden (Agent can't see) | Observable (Agent can see) |
|--------|-------------------------|---------------------------|
| **Clinical trial** | True drug effect, true responder population, true dose-response curve | Phase I safety data, interim results, budget spent |
| **SRE debugging** | Root cause of outage | Pod logs, error messages, metrics |
| **Biology experiment** | True gene expression levels | Experimental results after each step |

### In our project

When `reset()` is called, the simulator secretly rolls hidden ground truth:

```python
# HIDDEN — agent never sees this directly
class TrialLatentState:
    true_effect_size: float           # e.g., 0.23 (23% tumor reduction)
    true_side_effect_rate: float      # e.g., 0.08 (8% serious adverse events)
    true_responder_population: str    # e.g., "BRCA1+ only"
    true_mechanism: str               # e.g., "inhibits VEGF pathway"
    true_dose_response: dict          # dose → effect curve
```

The agent must *design experiments* (Phase I dose escalation, Phase II enrollment criteria) that help it *infer* these hidden values. The better the agent's experimental design, the more accurate its inference, and the higher its reward.

### The information-gathering challenge

This is what makes our environment hard and interesting:
1. **Phase I:** Agent runs dose escalation → gets noisy safety/efficacy signals → must estimate true effect size
2. **Phase II:** Agent uses its Phase I estimate to design the main trial → but if the estimate is wrong, the trial fails
3. **The trap:** A drug might look promising overall but only works in a subgroup (e.g., BRCA1+ patients). If the agent doesn't set smart inclusion criteria, it dilutes the effect and the trial fails statistically.

### Viva Questions — Chapter 3

1. **Q:** What does POMDP stand for, and why is it relevant to clinical trials?
   **A:** Partially Observable Markov Decision Process. It's relevant because the true drug effect, true responder population, and true dose-response curve are hidden. The agent must make trial design decisions based on incomplete information gleaned from experiments.

2. **Q:** How does the agent "discover" hidden information?
   **A:** Through its actions. Running dose escalation in Phase I gives noisy signal about the true effect size. Adding biomarker stratification helps identify the true responder subgroup. Each action provides information, but the agent must balance information gathering vs. resource spending.

3. **Q:** Why not just tell the agent the true effect size?
   **A:** That would defeat the purpose. Real clinical researchers don't know the true effect — they must design experiments to estimate it. Our environment trains the LLM to do the same. If we gave away the answer, the agent would learn nothing about experimental design.

---

## Chapter 4 — Reward Engineering — The Art of Teaching Through Scores

### Real-life analogy

You're grading a student's exam. You could:
- **Option A:** Give one final grade at the end (pass/fail). The student has no idea which questions they got wrong.
- **Option B:** Grade each question individually. The student knows exactly what to improve.

Option B is **decomposed reward**. It's dramatically better for learning.

### Why reward design matters

The reward function is the single most important design decision in RL. It defines *what the agent learns to optimize*. A bad reward function creates agents that find loopholes. A good one creates agents that genuinely solve the problem.

```
Bad reward:   +1 if trial succeeds, 0 otherwise
              → Agent gets almost no learning signal (too sparse)

Good reward:  +0.3 per FDA rule followed
              +0.2 per valid action sequence  
              +0.4 per information gained
              +5.0 if trial detects true effect
              -2.0 for underpowered design
              → Agent learns incrementally which actions are valuable
```

### Our reward structure

#### Per-step rewards (after every action):

| Component | What it rewards | How it's verified | Weight |
|-----------|----------------|-------------------|--------|
| FDA rule compliance | Following regulatory rules | Hard rule engine (binary check) | +0.3 per rule |
| Valid action sequence | Correct phase ordering | Prerequisite check | +0.2 |
| Information gain | Learning about hidden state from Phase I | Bayesian update quality | +0.4 |
| Budget efficiency | Not wasting resources | Math (cost / budget) | +0.1 |
| Soft violation | Doing something wrong but not fatal | Rule engine | -0.15 each |

#### Terminal rewards (when trial simulation runs at episode end):

| Component | What it rewards | How it's verified | Reward |
|-----------|----------------|-------------------|--------|
| Trial detects true effect | Core success — p < 0.05 | Simulation + stats math | +5.0 |
| Statistical power ≥ 0.80 | Adequate sample size | `scipy.stats` formula | +2.0 |
| All FDA rules pass | Regulatory compliance | Rule engine | +2.0 |
| Correct responder population | Found the right patients | Hidden state match | +3.0 |
| Budget under limit | Financial discipline | Cost math | +1.0 |
| Early futility detection | Stopped a doomed trial early | Simulation | +1.0 bonus |
| Underpowered design | Too few patients | Formula | -2.0 |
| Wrong primary endpoint | Measured the wrong thing | Domain rules | -1.5 |
| Overconfident wrong claims | Said "I'm 95% sure" and was wrong | Calibration check | -0.5 each |

#### Reward variance (what GRPO needs)

| Outcome | Total reward |
|---------|-------------|
| Successful trial (right population, right design) | +8 to +14 |
| Failed trial (wrong population, wasted budget) | -2 to 0 |
| Timeout / FDA rejection | -3 |

GRPO needs clear separation between good and bad episodes to compute advantages. A range of -3 to +14 gives plenty of signal.

### Potential-based reward shaping

A more advanced technique: $R_{\text{shaped}} = R_{\text{original}} + \gamma \cdot (\varphi(s') - \varphi(s))$

Where $\varphi(s)$ is a potential function (e.g., milestone completion fraction × budget efficiency). This gives the agent a "compass" toward good states without changing the optimal policy. Think of it as gravity pulling the agent toward progress.

### The golden rule of verification

> **If you need an LLM to judge whether the agent succeeded, your environment is weak.**

Our verification is all math and rules:
1. **Statistical power** → `scipy.stats.norm` — pure math
2. **FDA compliance** → hard-coded rule engine — binary pass/fail
3. **Trial outcome** → simulation with hidden ground truth — p-value is a number
4. **Budget** → arithmetic

No LLM judge needed for core success criteria.

### Viva Questions — Chapter 4

1. **Q:** Why decompose rewards instead of giving one big reward at the end?
   **A:** Decomposed rewards give learning signal at every step. With a single terminal reward, the agent doesn't know which of its 80+ actions contributed to success or failure. Decomposed rewards let GRPO pinpoint what's working.

2. **Q:** What is reward hacking? Give an example from our project.
   **A:** When the agent finds a shortcut to maximize reward without actually solving the problem. Example: if we rewarded "number of actions taken" without quality checks, the agent would spam meaningless actions. Our multi-layer verification (rules + math + simulation) prevents this.

3. **Q:** What is potential-based reward shaping and why is it theoretically safe?
   **A:** It adds $\gamma \cdot (\varphi(s') - \varphi(s))$ to the reward. Because the shaping terms telescope (cancel out over a full episode), the optimal policy doesn't change — only the learning speed improves. It's like giving hints that don't change what the best strategy is.

4. **Q:** Why is "LLM as judge" problematic for the core success criterion?
   **A:** LLM judges are slow (API call per step), expensive (token costs), noisy (different runs give different scores), and exploitable (agent can learn to generate text that fools the judge). Math and rule-based verification is instant, free, deterministic, and unexploitable.

---

## Chapter 5 — Curriculum Learning — Don't Throw a Kid Into Calculus

### Real-life analogy

You don't teach a kid multiplication by starting with triple integrals. You start with 2×3, then 12×8, then fractions, then algebra, then calculus. Each level builds on the last. If a level is too hard, there's zero learning signal — the kid just guesses randomly.

Same for RL agents.

### Why curriculum is mandatory

If every episode has a tiny drug effect, high dropout, and a hidden responder subgroup — the agent will fail every episode and learn nothing. GRPO needs *some* successes to compute what "better than average" looks like.

### Our curriculum (5 tiers)

| Tier | Difficulty | What changes | Real-life analog |
|------|-----------|-------------|-----------------|
| **Warmup** (0.0–0.25) | Easy | Large effect size, homogeneous population | 2×3 = 6 |
| **Beginner** (0.25–0.40) | Medium | Medium effect, some noise | 12×8 = 96 |
| **Intermediate** (0.40–0.60) | Hard | Small effect, need correct population enrichment | Quadratic equations |
| **Advanced** (0.60–0.80) | Very Hard | Hidden responder subgroup, misleading Phase I signal | Calculus |
| **Expert** (0.80–0.95) | Extreme | Tiny effect, high dropout, adaptive design required | Research-level proofs |

### Advancement rules

```python
# Agent advances when it masters current tier
if success_rate > 0.90 and episodes_at_tier >= min_episodes:
    advance_to_next_tier()

# Fast-track: if agent gets 90%+ quickly, skip the minimum episode requirement
if success_rate > 0.95:
    advance_immediately()
```

### 4 starting scenarios

| Scenario | Disease | Challenge | True Effect |
|----------|---------|-----------|-------------|
| `solid_tumor_chemo` | Non-small cell lung cancer | Find EGFR+ subgroup | 31% PFS improvement in EGFR+ only |
| `autoimmune_biologic` | Rheumatoid arthritis | Find optimal dose on U-shaped curve | U-shaped response, 200mg optimal |
| `cns_depression` | Treatment-resistant depression | High placebo response masks drug effect | 18% improvement over placebo |
| `rare_disease_orphan` | Rare pediatric metabolic disorder | Tiny sample size, adaptive design needed | Large effect (Cohen's d = 1.2) but n < 50 |

Each scenario can be run at all 5 difficulty tiers by varying noise, dropout, and budget constraints.

### Viva Questions — Chapter 5

1. **Q:** What happens if you skip curriculum and train directly on expert-level scenarios?
   **A:** The agent fails every episode, sees no positive reward signal, and GRPO computes zero useful gradients. The model doesn't learn. It's like teaching calculus to someone who doesn't know multiplication.

2. **Q:** How does the curriculum prevent the agent from just memorizing specific scenarios?
   **A:** Two ways: (1) domain randomization — even within a tier, noise, dropout rates, and budget vary randomly each episode, and (2) multiple scenarios with different diseases and challenges, so the agent must learn general trial design strategies.

3. **Q:** What is the "fast-track" rule and why is it important?
   **A:** If the agent achieves >90% success rate, it advances to the next tier even before completing the minimum number of episodes. Without this, a strong agent wastes training compute on easy problems it's already mastered. With limited onsite compute time (April 25–26), efficiency matters.

---

## Chapter 6 — GRPO — The Training Algorithm

### Real-life analogy

Imagine 8 students take the same exam (same questions, same difficulty). Their scores range from 40% to 95%. You tell all students: "Be more like the students who scored above average, and less like those below average." That's GRPO.

### What is GRPO?

**Group Relative Policy Optimization.** It's an RL algorithm designed for training LLMs. Instead of maintaining a separate "critic" network (like PPO), it generates a *group* of responses and uses relative ranking within the group as the learning signal.

### How it works (simplified)

```python
for each training step:
    # 1. Generate a GROUP of responses to the same scenario
    rollouts = [agent.generate(scenario) for _ in range(8)]  # 8 parallel episodes
    
    # 2. Score each rollout using environment rewards
    rewards = [environment.score(rollout) for rollout in rollouts]
    # e.g., rewards = [-2.1, +3.4, +8.7, -1.0, +5.2, +11.3, +0.8, +6.1]
    
    # 3. Compute advantages (how much better/worse than the group mean)
    mean_reward = average(rewards)       # = +4.05
    advantages = [r - mean_reward for r in rewards]
    # advantages = [-6.15, -0.65, +4.65, -5.05, +1.15, +7.25, -3.25, +2.05]
    
    # 4. Update model: increase probability of high-advantage actions,
    #    decrease probability of low-advantage actions
    agent.update(rollouts, advantages)
```

### Why GRPO over PPO?

| Feature | PPO | GRPO |
|---------|-----|------|
| Needs a critic/value network? | Yes (expensive, hard to train) | No (uses group comparison) |
| Works with sparse rewards? | Poorly (value function struggles) | Well (just needs variance within group) |
| Memory usage | High (two networks) | Lower (one model + rollouts) |
| Good for LLMs? | Okay | Designed for it (TRL integration) |

### What GRPO needs from the environment

1. **Reward variance** — if all 8 rollouts score the same, there's no gradient (nothing to learn). Our range of -3 to +14 ensures variety.
2. **Reproducible episodes** — same scenario should give comparable difficulty each time. Domain randomization adds variation, but the core challenge is stable.
3. **Fast step() execution** — 8 rollouts × 80 steps × cost-per-step adds up. Our math-based verification is instant (vs. LLM judge which would be 640 API calls per training step).

### The training command

```bash
# Terminal 1: Environment server (or use the HF Space URL)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Dry-run first (random policy, no model loading — validates pipeline)
python train.py --dry-run --episodes 2 --model-size 1.5b

# Terminal 2: Full GRPO training with vLLM acceleration
python train.py \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \   # or 3B/7B
    --model-size 1.5b \             # sets LoRA rank, batch, seq_len, grad_accum
    --episodes 100 \                # training episodes
    --vllm-mode colocate \          # vLLM shares GPU with training
    --num-generations 8 \            # 8 parallel rollouts per GRPO step
    --max-steps 50 \                 # steps per episode
    --seed 42 \                      # reproducibility
    --output-dir ./outputs/grpo      # checkpoints + reward CSV
```

### Model size presets (--model-size flag)

| Preset | LoRA rank | Batch size | Seq length | Grad accum | Target model |
|--------|-----------|------------|------------|------------|--------------|
| `1.5b` | 8 | 1 | 2048 | 4 | Qwen2.5-1.5B-Instruct |
| `3b` | 16 | 1 | 3072 | 4 | Qwen2.5-3B-Instruct |
| `7b` | 16 | 1 | 4096 | 8 | Qwen2.5-7B-Instruct |

### Viva Questions — Chapter 6

1. **Q:** Explain GRPO in one sentence.
   **A:** Generate multiple responses to the same prompt, rank them by reward, and nudge the model toward the better responses and away from the worse ones.

2. **Q:** Why does GRPO need reward variance?
   **A:** If all 8 rollouts get the same reward, the advantages are all zero, and the gradient is zero — the model doesn't update. The environment must produce a spread of outcomes (some good, some bad) for learning to happen.

3. **Q:** What is "vLLM colocate" mode?
   **A:** vLLM (a fast LLM inference engine) runs on the same GPU as training. It generates the 8 rollouts quickly, then the GPU switches to computing gradients. This avoids the cost of a separate inference server.

4. **Q:** How many training episodes do we expect with 2 days of H100 compute?
   **A:** With 8 rollouts per step and 50–100 training steps per run, each run produces 400–800 episodes. Across 2 days with tuning, restarts, and different curriculum configs, expect 200–600 *effective* training episodes that contribute to learning.

---

## Chapter 7 — OpenEnv — The Framework We're Building On

### What it is

OpenEnv is a framework that turns RL environments into **FastAPI web servers**. Instead of the agent and environment sharing memory (like in Gym), they communicate over HTTP. This means:
- Environment can run in Docker on any cloud
- Agent (LLM) can be trained separately
- Multiple agents can train against the same environment
- Easy to deploy on HuggingFace Spaces

### The contract

```python
from openenv import Environment, create_app

class ClinicalTrialEnv(Environment):
    def reset() -> TrialObservation       # start new episode, inject scenario
    def step(action) -> TrialObservation   # apply action, return obs + reward + done
    
    @property
    def state(self) -> TrialState          # current episode state

# Our actual implementation: FastAPI app in server/app.py
app = FastAPI(title="Clinical Trial Designer Environment")
```

This gives you:
```
POST /reset     → start new episode (optional seed parameter)
POST /step      → take action, get StepResponse (observation + reward + done + info)
GET  /state     → current trial state
GET  /schema    → TrialAction + TrialObservation JSON schemas
GET  /transcripts → NDJSON episode transcripts for demo replay
WS   /ws        → WebSocket for streaming actions
GET  /ping      → health check ({"status": "ok"})
```

### Deployment

```yaml
# openenv.yaml — tells the platform how to run your environment
spec_version: 1
name: clinical_trial_designer
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

```dockerfile
# Dockerfile — container that runs on HuggingFace Spaces
# NOTE: Using python:3.11-slim until ghcr.io/meta-pytorch/openenv-base:0.2.1 is publicly available
FROM python:3.11-slim
ENV PORT=7860              # HF Spaces default; local dev: docker run -p 8000:7860
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir fastapi uvicorn pydantic scipy numpy matplotlib pandas
COPY server/ ./server/
COPY models.py entrypoint.sh ./
RUN chmod +x entrypoint.sh && useradd -m appuser && chown -R appuser /app
USER appuser
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/ping')"
ENTRYPOINT ["./entrypoint.sh"]
```

### Viva Questions — Chapter 7

1. **Q:** Why serve the environment as a web API instead of a Python library?
   **A:** Decouples agent from environment. The agent (LLM + training loop) can run on a GPU machine while the environment runs in Docker anywhere. This also matches real-world deployment where the environment persists as a service.

2. **Q:** What OpenEnv version are we using and why?
   **A:** openenv-core==0.2.3. It provides the `Environment` base class and the standard endpoint contract (reset/step/state/schema/ping). Our FastAPI app in `server/app.py` implements these endpoints with additional features like `/transcripts` for demo replay.

3. **Q:** What happens if `/ping` fails during judging?
   **A:** Your submission is dead on arrival. The judges' automated checker hits `/ping` first. No response = disqualified before they even look at your code.

---

## Chapter 8 — Our Project — Clinical Trial Designer

### The elevator pitch

> A drug works. But only in 15% of patients. The FDA needs proof. Can an LLM learn to design a clinical trial that finds those patients before running out of money?

### What makes this a good RL problem

1. **Real-world grounding** — FDA rules, statistical power, clinical protocols are all real and codified
2. **Hidden state** — true drug effect, true responder population are never revealed to the agent
3. **Long horizon** — 55–100 steps per episode (Phase I → Phase II → Analysis)
4. **Math-based verification** — `scipy.stats` computes power, simulation determines p-value, rules check FDA compliance
5. **Information flow** — Phase I informs Phase II design (long-horizon planning challenge)
6. **Curriculum** — easy drugs → hard drugs → hidden subgroups → adaptive designs

### The world (hidden ground truth)

```python
class TrialLatentState:
    # Biology (AGENT CANNOT SEE)
    true_effect_size: float                    # e.g., 0.23
    true_responder_criteria: List[str]         # e.g., ["BRCA1+", "age < 65"]
    true_dose_response: Dict[float, float]     # dose → effect curve
    true_mechanism: str                        # e.g., "inhibits VEGF pathway"
    
    # Technical noise (AGENT CANNOT SEE)
    placebo_response_rate: float
    dropout_rate: float
    site_variability: float
    measurement_noise: float
    
    # Progress flags (agent CAN see)
    phase_i_complete: bool
    mtd_identified: bool
    effect_estimated: bool
    protocol_submitted: bool
    interim_complete: bool
    trial_complete: bool
    
    # Resources (agent CAN see)
    budget_remaining: float
    time_remaining_days: int
    patients_enrolled: int
```

### Agent actions (19 total)

| Phase | Action | What it does |
|-------|--------|-------------|
| Design | `set_primary_endpoint` | Choose what to measure (Overall Survival, Progression-Free Survival, etc.) |
| Design | `set_sample_size` | Power calculation → how many patients |
| Design | `set_inclusion_criteria` | Who can enroll (this is where the agent finds the responder population) |
| Design | `set_exclusion_criteria` | Who is excluded |
| Design | `set_dosing_schedule` | Dose amount, frequency, cycle length |
| Design | `set_control_arm` | Placebo vs. standard of care |
| Design | `set_randomization_ratio` | 1:1, 2:1, etc. |
| Design | `set_blinding` | Open-label, single-blind, double-blind |
| Phase I | `run_dose_escalation` | 3+3 design, find maximum tolerated dose |
| Phase I | `observe_safety_signal` | Read adverse event data (noisy view of hidden truth) |
| Phase I | `estimate_effect_size` | Bayesian update from Phase I data |
| Phase II | `run_interim_analysis` | Check futility/efficacy at 50% enrollment |
| Phase II | `modify_sample_size` | Adaptive design adjustment |
| Phase II | `add_biomarker_stratification` | Enrich for suspected responder subgroup |
| Regulatory | `submit_to_fda_review` | Automated compliance check |
| Regulatory | `request_protocol_amendment` | Change design mid-trial |
| Analysis | `run_primary_analysis` | Final statistical test (p-value computed) |
| Analysis | `synthesize_conclusion` | Write trial conclusion |

### Verification — all math, no LLM judge

```python
# 1. Statistical Power — pure math
from scipy.stats import norm
from math import sqrt

def calculate_power(effect_size, n, alpha=0.05):
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = effect_size * sqrt(n / 2) - z_alpha
    return norm.cdf(z_beta)
# If power < 0.80 → underpowered → reward penalty

# 2. FDA Compliance — hard rules
FDA_RULES = {
    "phase_ii_min_n": 100,
    "primary_endpoint_must_be_prespecified": True,
    "interim_analysis_requires_alpha_spending": True,
    "randomization_required_for_phase_iii": True,
    "safety_monitoring_committee_required": True,
    "informed_consent_required": True,
}

# 3. Trial Simulation — run it with hidden truth
def simulate_trial(design, ground_truth):
    # Sample patients, apply true effects, add noise, run stats test
    p_value = run_statistical_test(treatment_outcomes, control_outcomes)
    success = p_value < design.alpha
    return TrialResult(p_value, success, adverse_events)

# 4. Budget — arithmetic
cost = n_patients * cost_per_patient + site_costs + regulatory_fees
over_budget = cost > trial_budget
```

### Episode flow

```
Phase I (20–30 steps):
  → dose_escalation × 6 cohorts
  → observe_safety_signal × 3
  → estimate_effect_size (Bayesian update)
  → decide: go/no-go to Phase II

Phase II (30–40 steps):
  → set_primary_endpoint
  → set_sample_size (power calculation)
  → set_inclusion_criteria (find the responder population!)
  → set_dosing_schedule
  → submit_to_fda_review
  → run_interim_analysis (at 50% enrollment)
  → modify_sample_size if needed
  → run_primary_analysis

Conclusion (5–10 steps):
  → synthesize_conclusion
  → Terminal reward fires

Total: 55–100 steps per episode
```

### Viva Questions — Chapter 8

1. **Q:** Why clinical trials specifically? What makes it a good fit for OpenEnv?
   **A:** Clinical trials have objective success criteria (p-value < 0.05, power ≥ 0.80), real regulatory rules (FDA), hidden ground truth (true drug effect), and require long-horizon sequential decisions. All verification is math-based — no LLM judge needed.

2. **Q:** Walk me through what the agent does in one episode.
   **A:** The agent starts with a scenario (e.g., lung cancer drug). In Phase I, it runs dose escalation to find a safe dose and estimates the effect size. In Phase II, it uses that estimate to set sample size, choose the primary endpoint, set inclusion criteria to target the right patients, and runs the trial. The simulator computes the p-value using the hidden ground truth. If p < 0.05 and power ≥ 0.80, the trial succeeds.

3. **Q:** What's the hardest thing for the agent to learn?
   **A:** Identifying the hidden responder population. A drug might show a weak overall effect but a strong effect in BRCA1+ patients. The agent must learn to set inclusion criteria like `"BRCA1+"` in `add_biomarker_stratification` — this requires using Phase I data to form a hypothesis, then testing it in Phase II. Most failed episodes happen because the agent enrolls too broadly.

4. **Q:** How does your environment avoid reward hacking?
   **A:** Four verification layers, all objective: (1) `scipy.stats` computes real power, (2) FDA rules are binary pass/fail, (3) trial simulation uses hidden ground truth, (4) budget is arithmetic. The agent can't game any of these. An optional LLM judge layer adds qualitative assessment but doesn't control the core reward.

---

## Chapter 9 — How Winners Built Their Environments

### The non-negotiable pattern (all 3 winners had these)

1. **Real world state** — not just text. Real cluster health, real gene data, real product catalog.
2. **Actions that change state** — real commands, real math, real API calls.
3. **Verification WITHOUT an LLM judge** — system state, math, test pass/fail.
4. **Curriculum** — easy → hard, progressive difficulty.
5. **Long episodes** — 15–100+ steps.
6. **Clear reward variance** — GRPO needs +high vs. -low separation.

### 1st Place — Kube SRE Gym

- **Domain:** Kubernetes Site Reliability Engineering
- **Task:** Agent gets a PagerDuty alert, diagnoses and fixes a broken K8s cluster using real `kubectl` commands
- **Verification:** Pod status is ground truth — Running or not. No LLM needed.
- **Reward:** Per-step LLM judge (-1 to +1) + repeat penalty (-0.15) + resolution bonus (+1 to +5, efficiency-scaled) + timeout (-2.0) + phase-order bonus (+0.2)
- **Curriculum:** Warmup (single fault) → Beginner → Intermediate → Advanced (multi-fault) → Expert (adversarial LLM-designed incidents)
- **Key insight:** Environment co-evolved with the agent. Training exposed bugs in the command parser. Fixing them made both better.

### 2nd Place — Bio Experiment

- **Domain:** Biological Research / Single-Cell Genomics
- **Task:** Agent plans an experiment pipeline with hidden ground truth (true DE genes, true effect sizes)
- **Verification:** Prerequisite rules + budget math + hidden state match
- **Reward:** $R_t = r_{\text{validity}}(0.3) + r_{\text{ordering}}(0.2) + r_{\text{info\_gain}}(0.4) + r_{\text{efficiency}}(0.3) + r_{\text{novelty}}(0.1) + r_{\text{penalty}}(-0.15)$
- **Key insight:** Decomposed reward makes debugging trivial. Each component is independently verifiable.

### 3rd Place — EcomRLVE

- **Domain:** E-commerce Shopping Assistant
- **Task:** Help simulated customers with 2M real Amazon products (FAISS index)
- **Verification:** Product relevance ranking + constraint satisfaction + return policy math
- **Key insight:** Real product catalog (not mocked) forces genuine search and reasoning

### Finalist — Voyager-VRAM

- **Domain:** Workplace Project Management
- **Task:** Manage a 6-week project across 31 tools with hidden stale data and changing deadlines
- **Key innovation:** Skill Library (reusable tool sequences) + Working Memory + Episodic Memory across episodes
- **Result:** 21% improvement just from architecture, before any training

### Viva Questions — Chapter 9

1. **Q:** What's the single most important thing all winners had in common?
   **A:** Objective verification without LLM judges. Pod status, p-values, product relevance scores — all measurable, deterministic, and unexploitable.

2. **Q:** How is our project inspired by the winners without copying?
   **A:** We borrowed architectural patterns (decomposed reward from Bio Experiment, curriculum tiers from KubeSRE, hidden state from all three) but applied them to a completely different domain (clinical trials). Our verification uses `scipy.stats` power calculations and FDA rule engines — neither exists in any winner's codebase.

3. **Q:** What was KubeSRE's most important lesson for us?
   **A:** Environment co-evolution. Training will expose bugs in our simulator, rule engine, and reward logic. We should expect to iterate on the environment during training (April 25–26), so the codebase must be modular enough to hot-fix without breaking other components.

---

## Chapter 10 — Rules, Pitfalls, and Battle-Tested Lessons

### Environment Design Rules

| Rule | Why | Winner evidence |
|------|-----|----------------|
| One clear success criterion | Agent needs a north star | KubeSRE: pod running. Us: p < 0.05 |
| Real tools/APIs, not mocked | Mocked responses teach exploitation, not skill | KubeSRE: real kubectl |
| Prerequisite chains | Enforces realistic workflow | Bio: can't run DE before normalization |
| Reward variance | GRPO needs +high vs -low separation | All winners: -3 to +14 range |
| Multi-layer verification | Prevents reward hacking | KubeSRE: health check + LLM judge |
| Environment must fight back | Too-easy rewards cause plateaus | KubeSRE lesson |
| Repeat penalty | Prevents spamming same action | -0.15 per repeated action |

### Training Rules

| Rule | Why |
|------|-----|
| GRPO over PPO | Better for sparse delayed rewards, no value function needed |
| 8 parallel rollouts | Gives GRPO enough variance to compute advantages |
| Curriculum is mandatory | Cold start on hard problems = no learning signal |
| Fast-track advancement | 90%+ success rate → skip min_episodes requirement |
| Save episode transcripts | JSONL format for debugging and offline analysis |

### Pitfalls to Avoid

| Pitfall | What goes wrong | Prevention |
|---------|----------------|------------|
| LLM-only verification | Slow, expensive, noisy, exploitable | Use math + rules for core checks |
| Too-generous rewards | Agent finds plateau, stops improving | Scale rewards so failure is clearly negative |
| Static scenarios | Agent memorizes, doesn't generalize | Domain randomization + multiple scenarios |
| Single-fault only | Too easy, no curriculum progression | 5 tiers × 4 scenarios = 20 combinations |
| Mocked tool responses | Agent exploits mock patterns | Use real scipy, real rule engine |
| Truncated observations | Missing data corrupts decisions | KubeSRE bug: judge cut off pods alphabetically |

### Viva Questions — Chapter 10

1. **Q:** Name three pitfalls in RL environment design and how you avoid each.
   **A:** (1) LLM-only verification → we use scipy.stats and hard rule engine. (2) Static scenarios → we use domain randomization and 4 different disease scenarios. (3) Too-generous rewards → our failed trials score -2 to 0 while successes score +8 to +14, giving clear separation.

2. **Q:** Why 8 parallel rollouts specifically?
   **A:** It's the sweet spot for GRPO. Fewer rollouts = not enough variance to compute meaningful advantages. More rollouts = more compute per step without proportional benefit. 8 gives reliable advantage estimates while fitting in GPU memory with vLLM colocate.

---

## Chapter 11 — Tech Stack and Deployment

```
Environment:     openenv-core==0.2.3 (FastAPI-based RL framework)
Server:          FastAPI 0.111.0 + uvicorn 0.29+
Training:        HF TRL 0.29.0 (GRPOTrainer) + vLLM colocate + LoRA (peft ≥0.11)
Models:          Qwen2.5-1.5B / 3B / 7B-Instruct + LoRA (BF16 on H100)
Deployment:      Docker (python:3.11-slim) → HuggingFace Spaces (PORT 7860)
Compute:         H100 80GB (HF credits, onsite April 25-26)
Stats:           scipy 1.13.0 (power calculations, p-values)
Data:            numpy 1.26.4, pandas 2.2.2, matplotlib 3.8.4
Linting:         ruff 0.4.4, pytest 8.2.0
```

### Compute timeline

- **Now → April 24:** Build everything, validate pipeline with `--dry-run`. No GPU training. Environment, rewards, rules, curriculum, docs, dashboard, notebook validation.
- **April 25–26 onsite:** ALL GRPO training happens here with HuggingFace H100 credits. Run training, generate reward curves, capture before/after episodes, polish deliverables.

> **Rule (confirmed by organisers):** NO training is allowed before April 25. Only dry-run pipeline validation and notebook testing on Kaggle/Colab.

### Viva Questions — Chapter 11

1. **Q:** What model are you using and why?
   **A:** Qwen2.5-Instruct family with LoRA. We have three size presets: 1.5B (fast iteration, ~3 GB VRAM), 3B (middle ground, ~6 GB), 7B (highest quality, ~14 GB BF16). On H100 (80 GB), even 7B is comfortable. We start with 1.5B for fast signal checks, then scale up. LoRA keeps trainable parameters small (~1% of total) so training converges faster.

2. **Q:** Why Docker + HuggingFace Spaces?
   **A:** Hackathon requirement. The environment must be deployable as a Docker container on HF Spaces. Judges will run it against their evaluation harness. Our Space is live at `https://roopalgn-openenv-clinical-trial.hf.space`.

3. **Q:** What endpoints does your server expose?
   **A:** `POST /reset` (start new episode), `POST /step` (take action), `GET /state` (current state), `GET /schema` (action/observation JSON schemas), `GET /ping` (health check), `GET /transcripts` (episode replay data), `WS /ws` (WebSocket streaming).

---

## Chapter 12 — Pitch Strategy

### The 3-minute structure (judging: 40% innovation, 30% storytelling, 20% rewards, 10% pipeline)

**Minute 1 — The Story (targets 30% storytelling score)**

> "A drug works. But only in 15% of patients. The FDA needs proof. How do you design a trial that finds those patients before you run out of money?"

Show the cold start: agent's first attempt scores -2.5. Random inclusion criteria, wrong dose, underpowered study.

**Minute 2 — The Environment (targets 40% innovation score)**

Show the architecture: hidden ground truth → simulator → FDA rule engine → statistical verification → decomposed reward. Demonstrate Phase I → Phase II information flow. Show the curriculum progression.

**Minute 3 — Reward Curves + Demo (targets 20% reward improvement + 10% pipeline)**

Show reward curve improving across training. Show the agent learning to enrich for the responder population. Before/after: random inclusion criteria → learned BRCA1+ enrichment. Show the exact numbers: episode 1 reward -2.5 → episode 80 reward +11.2.

### Viva Questions — Chapter 12

1. **Q:** Why should we care about clinical trial design as an RL problem?
   **A:** Clinical trials cost $2B+ and take 10+ years. 90% fail. Better trial design — choosing the right patients, right dose, right endpoints — could save billions and get treatments to patients faster. An LLM that learns to design trials through RL could democratize this expertise.

2. **Q:** What makes your environment different from just fine-tuning on clinical trial papers?
   **A:** Fine-tuning teaches the model to *sound like* a clinical researcher. Our RL environment teaches it to *be* one. The agent's designs are tested against simulated ground truth — it learns from outcomes, not examples.

---

## Appendix — Glossary

| Term | Definition |
|------|-----------|
| **GRPO** | Group Relative Policy Optimization — RL algorithm that ranks a group of responses |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning that trains ~1% of model parameters |
| **vLLM** | Fast LLM inference engine used for generating rollouts during training |
| **POMDP** | Partially Observable MDP — agent can't see full state |
| **OpenEnv** | Meta's framework for RL environments served as FastAPI apps |
| **TRL** | Transformer Reinforcement Learning — HuggingFace library for RL training of LLMs |
| **ICH E9** | International Council for Harmonisation guideline for statistical principles in clinical trials |
| **MTD** | Maximum Tolerated Dose — highest dose with acceptable side effects |
| **PFS** | Progression-Free Survival — time until disease worsens |
| **ORR** | Overall Response Rate — percentage of patients who respond to treatment |
| **Cohen's d** | Standardized effect size (difference in means / pooled standard deviation) |
| **Alpha spending** | Statistical technique to control false positive rate across interim analyses |

---

> **Last updated:** Push 7 (2026-04-22) — Updated tech stack (Qwen2.5 family, openenv-core 0.2.3), training commands (--dry-run, --model-size presets), deployment details (Dockerfile, HF Space live), compute timeline (no training before Apr 25), and viva Q&A throughout.
