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
- [ ] **All URLs and links must be in your README** — HF Space, Colab notebook, blog/video, repo. If validation can't reach a deliverable from the README, it counts as missing.

### 3.1a Automated Validation Gate (Pre-Judge Filter)

> **CRITICAL:** If any item below is missing or broken at deadline, the submission **will not reach a human judge** regardless of quality.

- **Public, cloneable HF Space** at the submitted URL. Test from a **logged-out browser**. Private spaces, dead links, or 404s = automatic disqualification.
- **Valid OpenEnv structure:** proper `Environment` / `MCPEnvironment` base class, Gym-style `reset` / `step` / `state`, and a parseable `openenv.yaml`.
- **Training evidence committed as image files** (`.png` / `.jpg`): at minimum a loss curve and a reward curve. **Wandb-only links and plots that live only in a Colab cell don't count** — they may not be reachable when validation runs.
- **Runnable training script** (Unsloth, HF TRL, or other frameworks), preferably linked as a Colab notebook so it can be re-executed end to end. Python script is also acceptable.
- **README links every deliverable:** HF Space, training notebook, and writeup (blog/video/slides), with **key plots embedded inline**. If validation can't reach a deliverable from the README, it counts as missing.

### 3.1b Tips on Deliverables

- **HF Blog:** Write a markdown article and place it in your repo. HuggingFace supports markdown blog posts directly.
- **Training Notebook:** Ensure it is runnable and doesn't contain errors. Share the codebase link to the training script's `.ipynb` or add the publicly accessible Google Colab notebook link to your README.

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

### Submission Logistics

- **Deadline:** Submissions close at **5:00 PM on April 26**. No submissions accepted after this. **No extensions under any circumstances.**
- **How:** A Google Form will be shared on campus. You must submit:
  1. **HuggingFace Space URL**
  2. **Colab Notebook link**
  3. **Code repository link**
  4. **YouTube video URL** or **HuggingFace blog post URL**
- **Only one submission per team.** The URL you submit is what judges pull — **changes/commits after deadline will not be considered.**

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