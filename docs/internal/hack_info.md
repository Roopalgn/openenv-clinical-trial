# Meta PyTorch OpenEnv Hackathon — Reference Document

> **Event:** Meta PyTorch OpenEnv Hackathon × Scaler School of Technology — Grand Finale
> **Date:** 25–26 April 2026 | **Venue:** Scaler School of Technology, Electronic City, Bangalore
> **Our Theme:** #3.1 Professional Tasks (World Modeling)

---

## 1. Hackathon Themes

### Theme #1 — Multi-Agent Interactions

- Environments involving cooperation, competition, negotiation, and coalition formation
- Enables agents to model beliefs/incentives of others in partially observable settings
- Drives theory-of-mind reasoning and emergent strategic behavior
- **Expected Outcome:** An environment to train multi-agent task handling in an LLM
- **Examples:** Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games

### Theme #2 — (Super) Long-Horizon Planning & Instruction Following

- Environments requiring deep, multi-step reasoning with sparse or delayed rewards
- Goal: agents decompose goals, track state over extended trajectories, recover from early mistakes
- Push beyond shallow next-token reasoning toward structured planning and durable internal representations
- **Expected Outcome:** An environment capturing challenging long-horizon tasks beyond context memory limits
- **Examples:** Research-planning simulators, large-scale codebase refactoring, strategic resource management, long-horizon logistics, 300+ scattered instructions

### Theme #3 — World Modeling

#### 3.1 Professional Tasks ← **OUR THEME**

- Environments requiring real interaction with tools, APIs, or dynamic systems
- Model does real hard work instead of exploiting shortcuts
- Enables agents to maintain consistent internal state, update beliefs based on outcomes, orchestrate multi-step workflows
- Strengthens causal reasoning and persistent world models
- **Expected Outcome:** An environment capturing nuances of a defined partially observable world
- **Examples:** Dynamic browser/API ecosystems, enterprise applications, scientific workflow loops (papers → code → experiments), economic simulations with feedback, tool-discovery benchmarks

#### 3.2 Personalized Tasks

- Environments offering real personalized task handling (personal messages, dinner conflicts, tough emails)
- **Expected Outcome:** Realistic simulation of handling personal tasks, conflicts, and delegations
- **Examples:** Executive assistant meeting planner, dinner/drive planning, email/message replying, shopping

### Theme #4 — Self-Improvement

- Environments where agents generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula
- Goal: recursive skill amplification — agents learn to drive their own capability growth
- **Expected Outcome:** An environment for improving self-play of an LLM over defined tasks
- **Examples:** Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula

### Theme #5 — Wild Card: Impress Us!

- Open-ended — will reward out-of-the-box ideas
- Must meaningfully add value to LLM training on a certain task

---

## 2. Guidelines for Problem Statement

- It is **NOT mandatory** to choose the same problem statement as Round 1
- Only keep it if it aligns with the above themes
- You can start working on your problem statement once finalized
- Post-training happens **onsite on 25th & 26th** when you receive HuggingFace compute credits
- Before the onsite: work on building the environment, agent behaviors, reward model
- Evaluate whether your work aligns with the judging criteria below

---

## 3. Judging Criteria

### 3.1 Minimum Requirements (Non-Negotiable)

> **NOTE:** Submissions missing any of these are at a **serious disadvantage**.

- [ ] **OpenEnv (latest release)** — build on top of the framework; don't reinvent the wheel
- [ ] **Working training script** using Unsloth or HF TRL, ideally as a Colab notebook so judges can re-run it
- [ ] **Evidence of actual training** — at minimum, loss and reward plots from a real run
- [ ] **Short writeup** — a mini-blog on HuggingFace OR a < 2 minute video on YouTube (or short slide deck)
  - All materials must be **linked from your README** so judges can access them easily
- [ ] **HuggingFace Space** — environment must be hosted and runnable
- [ ] **README** that motivates the problem, explains how the env works, and shows results
  - Must link to: HF Space, blog/video, all additional references
- [ ] **No big video files** in HF Hub submission — use URL links to external materials

### 3.2 Judging Overview (Scoring Weights)

| Criterion | Weight | What It Means |
|-----------|--------|---------------|
| **Environment Innovation** | **40%** | Is the environment novel, creative, or genuinely challenging? Does it meaningfully test agent behavior in a way that hasn't been done before? |
| **Storytelling & Presentation** | **30%** | Can you clearly explain the problem, the environment, and what the agent learned? Is the demo engaging and easy to follow for a non-technical audience? |
| **Showing Improvement in Rewards** | **20%** | Is there observable evidence of training progress? Reward curves, before/after behavior, comparison against a baseline — anything that proves the agent learned something. |
| **Reward & Training Pipeline** | **10%** | Is the reward logic coherent? Does the pipeline produce meaningful improvement in the trained agent's behavior? |

### 3.3 What Judges Look For (Detailed Guide)

> *Read this before you start building, and again before you submit.*

**NOTE:** Only one submission per team. The URL link you submit is what judges pull — **changes/commits after deadline will not be considered.**

#### TL;DR

> Build an environment that an LLM could actually be trained on to get measurably better at something interesting. Then show that training. Then tell the story.
>
> A messy but ambitious environment with real training evidence beats a polished but boring one.
> Pick a problem that excites you — that energy comes through in the pitch.

---

## 4. What Makes a Submission Stand Out

### 4.1 Pick an Ambitious, Original Problem

The themes are deliberately open — use them as launching pads, not boxes. Judges have seen chess, snake, tic-tac-toe, and grid-world clones. To score well on innovation, you need a genuinely fresh angle.

**Ask yourself:**
- Does this environment exist to teach an LLM something it currently can't do well?
- Is the domain underexplored in RL/LLM training?
- Could a researcher write a paper about training on this?

### 4.2 Design a Reward Signal That Actually Teaches

A great environment has a reward function that:
- Provides a **rich, informative signal** (not just 0/1 at the end)
- Captures something **hard to measure** in a clever way
- Uses OpenEnv's **Rubric system** thoughtfully (composable rubrics > monolithic scoring)
- Is **hard to game** — an agent that exploits the reward without solving the task should not score high

### 4.3 Show Real Training, End to End

> The bar isn't "training script exists." The bar is "training script runs against the environment, the agent learns, and you can show it."

Concretely:
- Training loop **connects to your environment** (not a static dataset)
- Train long enough that the **curves mean something**
- Compare a **trained agent vs. random/untrained baseline** — quantitative and/or qualitative
- Include the **plots and numbers** in your README and writeup

### 4.4 Make Your Plots Readable

Reviewers spend **seconds, not minutes**, on each plot:
- **Label both axes** (e.g., "training step" / "episode" on x, "reward" / "loss" on y) with units
- **Save plots as .png or .jpg** and commit them to the repo (not just in a Colab cell or deleted Wandb run)
  - If you used Wandb, include the link to that specific run
- **Embed key plots in your README** with a one-line caption
- For multiple runs (baseline vs. trained, ablations), **put them on the same axes** for obvious comparison

### 4.5 Tell a Story, Not an API Doc

Your README, blog, and pitch should answer:
1. **Problem:** What capability gap or interesting domain are you targeting?
2. **Environment:** What does the agent see, do, and get rewarded for?
3. **Results:** What changed after training? Show it.
4. **Why it matters:** Who would care, and why?

> A reviewer should be able to read your README in 3–5 minutes and want to try your environment.

**NOTE:** If you have a video, HF post, or anything else interesting, link it from your README.

### 4.6 Engineer It Cleanly (Table Stakes)

Engineering quality matters less than ambition, but sloppy work hurts:
- Use OpenEnv's `Environment` / `MCPEnvironment` base classes properly
- Respect the **client/server separation** (clients should never import server internals)
- Follow the **standard Gym-style API** (`reset`, `step`, `state`)
- Have a **valid `openenv.yaml`** manifest
- **Don't use reserved tool names** (`reset`, `step`, `state`, `close`) for MCP tools

### 4.7 Final Note from Organizers

> Judges are looking for environments that push the frontier of what we can train LLMs to do. Be ambitious. Pick a problem you find genuinely interesting — that almost always produces better work than chasing what you think judges want. Good luck.

---

## 5. Team Confirmation

> This email serves as your official team ticket to the finale.

### Event Details

- **Date:** 25–26 April 2026
- **Venue:** Scaler School of Technology, Electronic City, Bangalore
- **Category:** Team of 2

### Team Members

| Role | Name | Email |
|------|------|-------|
| **Team Leader** | Roopal Guha Neogi | roopal.guhaneogi@gmail.com |
| **Team Member 2** | Suyash Kumar | suyashk102@gmail.com |

### Pre-Event Checklist

- [ ] Join the **private Discord** (MANDATORY) — all updates shared there first
- [ ] Check the **travel guide** (venue details, directions, nearby stay options)
- [ ] **Present this email at entry** — no entry without it
- [ ] Carry a **valid government-issued ID**
- [ ] Carry your **college/company ID** used during registration

### Entry Policy

- Entry will NOT be permitted if details don't match registration
- All team members must be individually registered in the system
- New/unregistered members added to travel details will NOT be allowed on campus
- Organisers reserve the right to deny entry if verification criteria are not met

---

## 6. Submission Design Expectations

- Choose one or more themes and design your own problem statement
- Simulate realistic scenarios, enable meaningful agent interaction, support measurable outcomes

### Required Submission Components

1. The **problem statement**
2. The **environment** in which the agent(s) operate
3. The **capabilities** of the agent(s)
4. The **tasks** to be performed
5. The **reward model / evaluation logic**
6. The **post-training or self-improvement strategy**

### Recommendations for High Scores

- Define clear, structured tasks and environments
- Incorporate robust evaluation and reward mechanisms
- Reflect real-world complexity aligned with OpenEnv principles

---

## 7. Competitor Repos (Round 2, India)

| # | Repo | Domain |
|---|------|--------|
| 1 | [veriRL](https://github.com/SupreethRao99/veriRL) | Verilog RTL verification |
| 2 | [cyber_range](https://github.com/softsideof/cyber_range) | SOC / Cybersecurity |
| 3 | [Parlay](https://github.com/sh4shv4t/Parlay) | Multi-party negotiation |
| 4 | [smartcity-traffic](https://github.com/thevivekkelkar/smartcity-traffic) | Traffic management |
| 5 | [Sovereign-SRE-Gym](https://github.com/sharad0x/Sovereign-SRE-Gym) | SRE / Zero-trust |
| 6 | [multi-agent-trading-env](https://github.com/ARKAISW/multi-agent-trading-env) | Trading desk |
| 7 | [MedFlow-OpenEnv](https://github.com/shriom17/MedFlow-OpenEnv) | Hospital triage |

---

## 8. Codex Gap Analysis — Where We Are Materially Behind

### 8.1 P0 Gaps (Must Fix)

#### Gap 1: Phase system is architected but not actually driving the environment

> **This is the largest code-level gap in the repo.**

**Evidence:**
- `server/episode_manager.py` initializes `latent.episode_phase = "literature_review"` on reset
- `server/phase_detector.py` classifies phases from actions
- `server/rules/fda_rules.py` checks action validity against `latent.episode_phase`
- `server/simulator/output_generator.py` builds `available_actions` from `latent.episode_phase`
- **BUT** `server/simulator/transition_engine.py` **never updates** `latent.episode_phase`
- `tests/test_integration.py` even documents and asserts that episode phase stays `literature_review` throughout

**Consequence:**
- Phase abstraction exists, but phase progression is not authoritative
- `available_actions` and FDA transition logic are anchored to the initial phase
- A large part of the designed action/state space is dead or underused
- Strong tests accidentally preserve this if they encode the simplification

#### Gap 2: Scenario descriptions promise richer biology than the latent state contains

**Evidence:**
- `server/curriculum/scenarios.py` describes biomarker enrichment, dose-response structure, placebo effects, rare-disease constraints
- BUT `server/episode_manager.py` reset() builds latent state with:
  - `true_responder_population="all"`
  - `true_responder_criteria=[]`
  - `true_dose_response={}`
  - `true_mechanism="unknown"`
- `server/simulator/output_generator.py` has code paths for dose-response/responder hints, but fields are mostly empty

**Consequence:**
- Code advertises a richer clinical-trial environment than it actually simulates
- Biomarker and dose-related actions have much less real semantic payoff than code structure suggests
- Top same-domain competitor (OpenENV-Hackathon) is much deeper here

#### Gap 3: Adversarial curriculum is underfed and weaker than it looks

**Evidence:**
- `server/curriculum/adversarial_designer.py` expects `true_effect_size`, `dropout_rate`, biomarker usage
- `server/episode_manager.py` feeds `analyze_failures()` with `{success, true_effect_size=None, dropout_rate=None}`
- No biomarker-use signal is passed

**Consequence:**
- Adversarial designer cannot actually learn the weak spots it claims to target
- `small_effect` and `high_dropout` counters never get meaningful signal
- Expert-tier adversarial logic is much weaker than module names suggest

#### Gap 4: Training runner is not a real competitive training pipeline yet

**Evidence:**
- `train.py` instantiates `GRPOTrainer` but **never calls `trainer.train()`**
- Comment in `train.py` explicitly says the manual rollout loop drives logging and `trainer.train()` "can be called instead"
- `_grpo_reward_fn()` is defined but not actually integrated into a real trainer loop
- The manual loop in `rollout_func()` is closer to evaluation logging than trainer-backed optimization

**Consequence:**
- We have training scaffolding, not a training system on par with EcomRLVE-Gym, OpenENV-Hackathon, or veriRL
- **Major competitive weakness** — those repos wire reward functions, datasets, and trainer flows credibly

#### Gap 5: Judge is stage-insensitive, distorting learning signal

**Evidence:**
- `server/judge.py` checks budget, power ≥ 0.80, and p-value < 0.05 on **every step** using hidden latent truth
- Early design actions are judged against end-state statistical criteria before the agent has even finished the workflow

**Consequence:**
- Feedback loop is partly hindsight-based instead of phase-appropriate
- Weaker than environments where reward/judge logic is tied to stage-specific progress

#### Gap 6: Single session only

**Evidence:**
- `server/environment.py` sets `SUPPORTS_CONCURRENT_SESSIONS = False`
- `server/app.py` uses one global `_manager = EpisodeManager()`

**Consequence:**
- Weaker than OpenENV-Hackathon, veriRL, MedFlow-OpenEnv (which have session isolation or concurrent workflows)
- Not critical for hackathon demo

### 8.2 P1 Gaps (Nice to Fix)

#### Gap 7: Resource physics and observation noise are much simpler than top environments

**Evidence:**
- `server/noise_model.py` randomizes only a few scalar ranges
- `server/simulator/transition_engine.py` applies mostly fixed cost/time constants
- `server/simulator/trial_simulator.py` uses lightweight statistical proxies

**Compared with:**
- OpenENV-Hackathon: richer latent technical/biological state, hard/soft rule propagation, modality-specific output generation
- cyber_range: explicit network topology, alerts, forensics, attacker progression
- veriRL: real external evaluators (not proxies)

#### Gap 8: Breadth of module names without matching breadth of realized behavior

The strongest same-domain comparison is OpenENV-Hackathon, which has:
- Task generation + procedural scenario generation
- Latent biology state + technical state
- Hard and soft rules
- Detailed transition engine + output generator
- Decomposed reward with shaping and terminal calibration

Our repo has analogous module names, but many paths are thinner in semantics. **The architecture is there. The content density is not.**

#### Gap 9: Tests validate simplified behavior that should eventually change

The clearest example is phase progression. The test suite is a strength, but it needs to evolve with the simulator rather than just defend current simplified behavior.

---

## 9. Competitor-by-Competitor Comparison

### OpenENV-Hackathon (Bio Experiment) — **Most important direct comparator**

Solves a closely related hidden-state scientific planning problem with a deeper implementation.

| Dimension | Them | Us |
|-----------|------|-----|
| Latent state | Richer latent/observed loop in `hackathon_environment.py` | Simpler, fields often empty |
| Scenario generation | `generator.py` + `procedural_generator.py` | Less scenario richness |
| Rules engine | Prerequisites, resource constraints, redundancy, causal validity, tool compatibility | FDA rules + basic constraints |
| Simulator | More detailed `transition.py` + `output_generator.py` | Thinner semantic paths |
| Rewards | Substantially richer and more stage-aware | 8-component but stage-insensitive |
| Training | Much closer to a real training system | `trainer.train()` not wired |

**Where we're stronger:** Easier to reason about, stronger automated verification, simpler API/dashboard/logging.
**Bottom line:** They're ahead on environment depth. We're ahead on code trustworthiness.

### EcomRLVE-Gym (E-commerce)

| Dimension | Them | Us |
|-----------|------|-----|
| Scope | Multi-environment platform | Single environment |
| Orchestration | Strong env selection, adaptive difficulty, tool execution, user simulator | Simpler curriculum |
| Verification | Deep environment-specific deterministic verification | Ground-truth + rule engine |
| Training | More credible GRPO integration | `trainer.train()` not wired |

**Where we're stronger:** Far better test coverage, lower complexity, cleaner maintainability.
**Bottom line:** They're the broadest platform. We're more disciplined but materially less complete.

### cyber_range (SOC/Security)

| Dimension | Them | Us |
|-----------|------|-----|
| Interaction surface | True multi-tool surface via `cyber_environment.py` | Clinical actions via API |
| World state | Living network topology + attack + forensics | Scalar latent variables |
| Curriculum + Judge | Deeper `attack_designer.py` + `cyber_judge.py` | Rule-based templates |
| Reward shaping | Tuned for actual RL signal variance | Standard potential-based |

**Where we're stronger:** Better typing, cleaner modular boundaries, better test posture.
**Bottom line:** They're ahead on realism. We're ahead on reliability engineering.

### veriRL (Verilog RTL)

| Dimension | Them | Us |
|-----------|------|-----|
| Evaluation | Real tool-backed (iverilog, yosys, SymbiYosys) | Synthetic simulator |
| Concurrency | Multi-file environment, concurrent sessions | Single session |
| Task packaging | Strong real-world feedback loop | Clinical scenario cards |
| Tests | Very good | Also very good |

**Where we're stronger:** Easier API/demo shell, simpler state reasoning.
**Bottom line:** Their evaluator is much more grounded. They're ahead overall because their grading surface is real.

### kube-sre-gym (1st Place Winner)

| Dimension | Them | Us |
|-----------|------|-----|
| Backend | Real GKE Kubernetes cluster | Synthetic simulator |
| Adversarial | Richer, operationally grounded incident design (Claude) | Rule-based templates |
| Judge/Scenarios | Workflow-aware incident response | Phase-based clinical workflow |

**Where we're stronger:** Determinism, reproducibility, testability, lower operational dependency.
**Bottom line:** They trade determinism for realism — and for demos/storytelling, that realism is a competitive asset.

### Parlay (Negotiation)

| Dimension | Them | Us |
|-----------|------|-----|
| Reward design | Game-theoretic (ZOPA, Nash, Shapley) — sharper | 8-component decomposed |
| Session flow | Coherent WebSocket + MCP | Standard REST + WebSocket |
| Pipeline | SFT→GRPO 2-stage | GRPO single-stage |

**Where we're stronger:** Better test depth, better API/product shell, more explicit modularization.
**Bottom line:** Same maturity band. They're stronger in domain-specific reward math. We're stronger in verification.

### MedFlow-OpenEnv (Hospital Triage)

- **Stronger:** Dynamic arrivals, doctor scheduling, beds, queue-state simulation
- **Weaker:** Simple reward (+0.15/−0.10), no GRPO training, "future scope: replace dummy RL"
- **We are ahead overall**

### smartcity-traffic (Traffic Management)

- **Stronger:** Explicit multi-agent / federated baseline story
- **Weaker:** Q-Learning (not LLM), simple 2×2 grid
- **We are clearly ahead overall**

### multi-agent-trading-env (Trading Desk)

- **Stronger:** Frontend polish and dashboard surface
- **Weaker:** Backend incomplete (`env.trading_env.TradingEnv` module missing from repo tree)
- **Our codebase is more complete**

### Sovereign-SRE-Gym (SRE)

- **Stronger:** Interesting multi-agent delegation concept with adversarial NPCs
- **Weaker:** No training pipeline, early stage
- **We are clearly ahead overall**

---

## 10. The Most Important Strategic Insight

> **Our repo is not losing because the architecture is wrong.**

It is losing because:
1. The phase system is **not authoritative**
2. The latent biology is **under-instantiated**
3. The adversarial curriculum is **underfed**
4. The training loop is **not fully wired**

**That is good news.**

It means the fastest path forward is **not a rewrite**. It is **semantic completion** of the existing architecture.
