# OpenEnv Clinical Trial Designer — Knowledge Base

> A complete guide to understanding our project, the decisions behind it, and how to explain it to judges.
> Written for hackathon participants who know Python but not RL or clinical trials.

---

## Table of Contents

1. [What We Built](#1-what-we-built)
2. [Why Clinical Trials?](#2-why-clinical-trials)
3. [RL Basics — What You Need to Know](#3-rl-basics)
4. [Our Environment — How It Works](#4-our-environment)
5. [The Reward System — How the Agent Learns](#5-the-reward-system)
6. [The Curriculum — Teaching in Stages](#6-the-curriculum)
7. [GRPO — The Training Algorithm](#7-grpo)
8. [OpenEnv — The Framework](#8-openenv)
9. [Tech Stack](#9-tech-stack)
10. [Pitch Strategy](#10-pitch-strategy)
11. [Viva Questions — Full List](#11-viva-questions)

---

## 1. What We Built

**One-liner:** An RL environment where an LLM learns to design clinical trials that detect hidden drug effects — verified by math, not by another LLM.

**The scenario:** A drug works, but only in 15% of patients. The FDA needs proof. The agent has a budget, a deadline, and zero medical training. It must discover which patients benefit, design a trial to prove it, and pass FDA compliance — all through trial and error.

**What makes it hard:**
- The true drug effect, true responder subgroup, and safety profile are **hidden** (partially observable)
- The agent must run experiments (dose escalation, safety monitoring) to **discover** this hidden information
- Verification is **objective** — `scipy.stats` calculates power, a rule engine checks FDA compliance, simulation returns the p-value
- No LLM judge needed for core success metrics

**What we implemented:**
- 19 actions across 5 clinical phases (design, Phase I, Phase II, regulatory, analysis)
- 4 disease scenarios with different hidden challenges (subgroup discovery, dose-response, placebo masking, rare disease)
- 5-tier adaptive curriculum that gets harder as the agent improves
- 15-component decomposed reward (8 per-step + 7 terminal)
- Full GRPO training pipeline with TRL 0.29.0 + LoRA
- 249 tests, deployed on HuggingFace Spaces

---

## 2. Why Clinical Trials?

### Why this is a great RL problem

Clinical trials cost **$2.6 billion per approved drug** and take 10–15 years. **90% fail** in Phase II. Poor trial design — wrong patients, wrong dose, wrong endpoints — accounts for over half of failures.

We chose clinical trials because they fit every criterion judges care about:

| Criterion | Why Clinical Trials Fit |
|-----------|------------------------|
| **Partially observable** | True drug effect, responder subgroup, and safety profile are hidden — agent must run experiments to discover them |
| **Long-horizon planning** | 55–100 steps across Phase I → Phase II → regulatory → analysis |
| **Objective verification** | `scipy.stats` calculates power, FDA rules are codified, trial simulation returns p-values — no LLM judge needed |
| **Real-world grounding** | FDA ICH E9 guidelines, rpact-validated power calculations, real statistical methodology |
| **Domain randomization** | Budget ±30%, time ±20%, dropout ±15%, placebo ±20% — agent can't memorize solutions |
| **Novel domain** | No existing OpenEnv environment covers clinical trial design |

### Why not just fine-tune on clinical trial papers?

Fine-tuning teaches a model to **sound like** a clinical researcher. Our RL environment teaches it to **be** one. The agent's designs are tested against simulated ground truth — it learns from outcomes, not examples. If its trial fails (p ≥ 0.05), it gets penalized. If it finds the hidden subgroup and designs a powered trial, it gets rewarded.

### Viva: Why clinical trials?

**Q:** Why should judges care about clinical trial design as an RL problem?
**A:** It's a $2.6B problem with 90% failure rate. Better trial design saves billions and gets treatments to patients faster. An LLM that learns through RL could democratize this expertise.

**Q:** Why not just use supervised learning?
**A:** We don't have a dataset of "perfect trial designs." Trial design involves sequential decisions under uncertainty. RL lets the agent learn from outcomes (did the trial detect the drug effect?) rather than memorizing past designs.

---

## 3. RL Basics

### The loop

```
1. env.reset()        → Agent gets a new clinical scenario
2. Agent picks action  → e.g., "run dose escalation at 100mg"
3. env.step(action)   → Returns observation + reward + done
4. Repeat until done   → Episode ends (trial concluded or timeout)
5. Go to 1            → New scenario, hopefully agent does better
```

### Key terms

| Term | In Our Project |
|------|---------------|
| **Agent** | The LLM (Qwen2.5 + LoRA) |
| **Environment** | Our clinical trial simulator (FastAPI server) |
| **State** | Full truth including hidden biology (agent can't see all of it) |
| **Observation** | What agent sees: phase data, safety signals, budget, available actions |
| **Action** | One of 19 clinical trial actions (e.g., `set_primary_endpoint`, `run_dose_escalation`) |
| **Reward** | Per-step (8 components) + terminal (7 components) |
| **Episode** | One complete trial design attempt (55–100 steps) |
| **Policy** | The LLM's learned behavior (observation → action mapping) |

### What is partial observability?

The agent can't see the full truth. The true drug effect, responder subgroup, and dose-response curve are **hidden**. The agent must design experiments to infer them — just like a real clinical researcher.

| Hidden (agent can't see) | Observable (agent can see) |
|--------------------------|---------------------------|
| True drug effect size | Phase I safety data (noisy) |
| True responder subgroup (e.g., EGFR+) | Interim analysis results |
| True dose-response curve | Budget remaining |
| True safety profile | Available actions |

### Viva: RL basics

**Q:** What's the difference between state and observation?
**A:** State is the full truth. Observation is what the agent sees. The true drug effect is part of the state but hidden from observation — the agent must infer it from experiments.

**Q:** Why multiple episodes?
**A:** Each episode is a different clinical scenario (different drug, disease, patient population). The agent needs many scenarios to learn general trial design strategies, not just memorize one solution.

---

## 4. Our Environment

### Hidden ground truth

On each `reset()`, the environment secretly sets:

```python
class TrialLatentState:
    true_effect_size: float              # e.g., 0.31 (31% improvement)
    true_responder_population: str       # e.g., "EGFR+" (only these patients benefit)
    true_responder_criteria: list        # e.g., ["EGFR_mutation"]
    true_dose_response: dict             # dose → effect curve
    true_side_effect_rate: float         # e.g., 0.12 (12% adverse events)
    placebo_response_rate: float         # background noise masking the signal
    dropout_rate: float                  # patients who leave the trial
    budget_remaining: float              # money left
    time_remaining_days: int             # deadline
```

The agent **never sees** the biology fields directly. It must run experiments to discover them.

### 19 actions

| Category | Actions | What They Do |
|----------|---------|-------------|
| **Design** (8) | `set_primary_endpoint`, `set_sample_size`, `set_inclusion_criteria`, `set_exclusion_criteria`, `set_dosing_schedule`, `set_control_arm`, `set_randomization_ratio`, `set_blinding` | Standard trial protocol components |
| **Phase I** (3) | `run_dose_escalation`, `observe_safety_signal`, `estimate_effect_size` | 3+3 dose finding, safety monitoring, Bayesian effect estimation |
| **Phase II** (3) | `run_interim_analysis`, `modify_sample_size`, `add_biomarker_stratification` | Adaptive design, population enrichment |
| **Regulatory** (2) | `submit_to_fda_review`, `request_protocol_amendment` | FDA compliance check, protocol changes |
| **Analysis** (2) | `run_primary_analysis`, `synthesize_conclusion` | Final statistical test, trial conclusion |
| **Terminal** (1) | Episode ends when `synthesize_conclusion` is called or timeout | Terminal reward fires |

### 4 scenarios

| Scenario | Disease | Hidden Challenge | Key Insight Agent Must Discover |
|----------|---------|-----------------|-------------------------------|
| `solid_tumor_chemo` | Lung cancer (NSCLC) | Drug works in EGFR+ patients only | Enrich for EGFR+ subgroup → 58% effect vs 31% overall |
| `autoimmune_biologic` | Rheumatoid arthritis | U-shaped dose-response | 200mg is optimal — higher doses are worse |
| `cns_depression` | Treatment-resistant depression | High placebo response masks drug | Must enrich for severe TRD subgroup |
| `rare_disease_orphan` | Rare pediatric disorder | Only ~50 patients exist | Must use adaptive Bayesian design with relaxed endpoints |

### 10-phase clinical workflow

The agent should follow this order (learned through reward signal, not hard-coded):

```
literature_review → hypothesis → phase_i_design → phase_i_analysis →
phase_ii_design → regulatory → enrollment → monitoring → analysis → conclusion
```

Following this order: **+0.2 bonus per step**. Skipping phases: **-0.3 × N penalty**.

### Verification — all math, no LLM judge

This is our strongest innovation argument:

1. **Statistical power** → `scipy.stats.norm` calculates power from effect size and sample size (pure math)
2. **FDA compliance** → hard-coded rule engine with 6+ constraints (binary pass/fail)
3. **Trial simulation** → runs the trial against hidden ground truth, returns p-value (math)
4. **Budget** → cost = n_patients × cost_per_patient + site_costs (arithmetic)

> **Golden rule:** If you need an LLM to judge whether the agent succeeded, your environment is weak. Ours uses math.

### Viva: Environment design

**Q:** Walk me through one episode.
**A:** Agent gets a scenario (e.g., lung cancer drug). Phase I: dose escalation → safety check → effect estimate. Phase II: set sample size, choose endpoint, set inclusion criteria to target the right patients, run the trial. Simulator computes p-value using hidden truth. If p < 0.05 and power ≥ 0.80, the trial succeeds.

**Q:** What's the hardest thing for the agent to learn?
**A:** Identifying the hidden responder population. The drug might show weak overall effect but strong effect in EGFR+ patients. The agent must use `add_biomarker_stratification` to discover this — requires forming a hypothesis from Phase I data and testing it in Phase II.

**Q:** How do you prevent reward hacking?
**A:** Four verification layers, all objective: scipy.stats computes real power, FDA rules are binary, trial simulation uses hidden ground truth, budget is arithmetic. The agent can't game any of these.

**Q:** Why not use real clinical trial data?
**A:** Grounded in rpact/scipy.stats power calculations and FDA ICH E9 rules — the same math used in actual trial software. Not toy heuristics. Real trial databases have privacy constraints and don't provide hidden ground truth for verification.

---

## 5. The Reward System

### Why decomposed rewards?

Giving one big reward at the end (pass/fail) gives almost no learning signal — the agent doesn't know which of its 80+ actions contributed to success or failure. Decomposed rewards let GRPO pinpoint what's working.

### Per-step rewards (8 components)

| Component | What It Rewards | Weight | How It's Verified |
|-----------|----------------|--------|-------------------|
| `r_validity` | FDA rule compliance | 1.0 | Rule engine (binary) |
| `r_ordering` | Correct phase workflow | 1.0 | Phase detection heuristic |
| `r_info_gain` | Information gained from experiments | 1.0 | Bayesian update quality |
| `r_efficiency` | Budget/time efficiency | 1.0 | Math (cost / budget) |
| `r_novelty` | Trying new action types | 1.0 | Action history check |
| `r_penalty` | Soft violations | 1.0 | Rule engine |
| `r_shaping` | Progress toward milestones | — | Potential-based: γ·(φ(s') − φ(s)) |

### Terminal rewards (7 components, fire when episode ends)

| Component | What It Rewards | Reward |
|-----------|----------------|--------|
| `r_terminal_success` | Trial detects true effect (p < 0.05) | +5.0 to +7.0 |
| `r_terminal_calibration` | Agent's claims match hidden truth | +3.0 to +4.0 |
| `r_terminal_power` | Statistical power ≥ 0.80 | +2.0 |
| `r_terminal_fda` | All FDA rules pass | +2.0 |
| `r_terminal_budget` | Under budget | +1.0 |
| `r_terminal_futility` | Stopped a doomed trial early | +1.0 bonus |
| `r_terminal_overconf` | Overconfident wrong claims | -0.5 each |

### Total reward range

| Outcome | Total Reward |
|---------|-------------|
| Successful trial (right population, right design) | +8 to +14 |
| Failed trial (wrong population, wasted budget) | -2 to 0 |
| Timeout / FDA rejection | -3 |

This range (-3 to +14) gives GRPO the variance it needs to compute meaningful advantages.

### Potential-based reward shaping

$R_{\text{shaped}} = R_{\text{original}} + \gamma \cdot (\varphi(s') - \varphi(s))$

Where $\varphi(s)$ = milestone completion fraction × budget efficiency. This gives the agent a "compass" toward progress without changing the optimal policy. The shaping terms telescope (cancel over a full episode), so what's optimal doesn't change — only learning speed improves.

### Viva: Rewards

**Q:** Why decompose rewards instead of one big reward at the end?
**A:** With a single terminal reward, the agent doesn't know which of its 80+ actions contributed to success. Decomposed rewards give signal at every step, letting GRPO pinpoint what's working.

**Q:** What is reward hacking? Example from our project?
**A:** Agent finds a shortcut to maximize reward without solving the problem. Example: if we rewarded "number of actions taken" the agent would spam meaningless actions. Our multi-layer verification (rules + math + simulation) prevents this.

**Q:** What is potential-based reward shaping and why is it safe?
**A:** It adds γ·(φ(s') − φ(s)) to the reward. Because shaping terms telescope over a full episode, the optimal policy doesn't change — only learning speed improves. Safe by construction.

---

## 6. The Curriculum

### Why curriculum matters

If every episode has a tiny drug effect, high dropout, and a hidden subgroup — the agent fails every time and learns nothing. GRPO needs some successes to compute what "better than average" looks like.

### 5 tiers

| Tier | Difficulty | What Changes | Real Analogy |
|------|-----------|-------------|-------------|
| **Warmup** | 0.0–0.25 | Large effect size, homogeneous population | 2 × 3 = 6 |
| **Beginner** | 0.25–0.40 | Medium effect, some noise | 12 × 8 = 96 |
| **Intermediate** | 0.40–0.60 | Small effect, enrichment needed | Quadratic equations |
| **Advanced** | 0.60–0.80 | Hidden subgroup, misleading Phase I | Calculus |
| **Expert** | 0.80–0.95 | Tiny effect, high dropout, adaptive design | Research-level proofs |

### Advancement rules

- **Per-scenario tracking** — mastery is tracked per scenario × tier, not globally
- **Sliding window** — success rate computed over last W episodes (prevents history dilution)
- **Fast-track** — ≥90% success in 5 episodes → advance immediately (don't waste compute on easy problems)
- **No demotion** — once advanced, the agent stays (GRPO updates prevent catastrophic regression)

### Adaptive difficulty

After mastering a scenario, parameters **harden within the same tier**: effect size shrinks, budget tightens, noise increases, subgroups become rarer. The environment also **targets weak spots** — preferentially selects scenarios the agent struggles with.

### Viva: Curriculum

**Q:** What happens if you skip curriculum and train on expert directly?
**A:** Agent fails every episode, zero positive reward signal, GRPO computes zero useful gradients. No learning at all.

**Q:** How do you prevent memorization?
**A:** Domain randomization (noise, dropout, budget vary per episode) + 4 different disease scenarios + parameter hardening after mastery.

**Q:** What's the fast-track rule?
**A:** ≥90% success rate → advance immediately. Prevents wasting limited H100 compute on already-mastered problems.

---

## 7. GRPO

### What it is

**Group Relative Policy Optimization.** Generate 8 responses to the same scenario. Rank them by reward. Nudge the model toward the better responses, away from the worse ones.

### How it works (simplified)

```python
for each training step:
    # Generate 8 parallel rollouts (episodes) for the same scenario
    rollouts = [run_episode(agent, env) for _ in range(8)]
    rewards = [r.total_reward for r in rollouts]
    
    # Compute advantages (how much better/worse than group average)
    mean_reward = mean(rewards)
    advantages = [r - mean_reward for r in rewards]
    
    # Update model weights: increase probability of high-advantage actions
    agent.update(rollouts, advantages)
```

### Why GRPO over PPO?

| Feature | PPO | GRPO |
|---------|-----|------|
| Needs separate critic network? | Yes (expensive) | No (uses group comparison) |
| Works with sparse rewards? | Poorly | Well (just needs variance within group) |
| Memory usage | High (two networks) | Lower (one model + rollouts) |
| Designed for LLMs? | Not specifically | Yes (TRL integration) |

### What GRPO needs from us

1. **Reward variance** — if all 8 rollouts score the same, no gradient. Our range -3 to +14 ensures variety.
2. **Fast step()** — 8 rollouts × 80 steps adds up. Our math-based verification is instant (vs. LLM judge which would be 640 API calls per step).
3. **Reproducible episodes** — same scenario should give comparable difficulty each time.

### Viva: GRPO

**Q:** Explain GRPO in one sentence.
**A:** Generate multiple responses, rank by reward, nudge model toward the better ones and away from the worse ones.

**Q:** Why does GRPO need reward variance?
**A:** If all 8 rollouts get the same reward, advantages are all zero, gradient is zero — model doesn't update. The environment must produce a spread of outcomes.

**Q:** What is vLLM colocate?
**A:** vLLM (fast LLM inference) runs on the same GPU as training. It generates rollouts quickly, then the GPU switches to gradients. Avoids needing a separate inference server.

**Q:** How many episodes with 2 days of H100?
**A:** 8 rollouts/step × 50–100 steps = 400–800 per run. Realistically 200–600 effective episodes across tuning, restarts, and debugging.

---

## 8. OpenEnv

### What it is

A framework by Meta that turns RL environments into **FastAPI web servers**. Agent and environment communicate over HTTP, which means:
- Environment runs in Docker on any cloud (we use HuggingFace Spaces)
- Agent (LLM) can train separately on a GPU machine
- Easy to deploy and test

### Our endpoints

```
POST /reset         → Start new episode, get initial observation
POST /step          → Take action, get observation + reward + done
GET  /state         → Current trial state
GET  /schema        → Action/observation JSON schemas
GET  /ping          → Health check → {"status": "ok"}
GET  /transcripts   → Episode replay data (NDJSON)
WS   /ws            → WebSocket for live streaming
```

### Why web API?

Decouples agent from environment. Judges can hit `/ping` to verify your Space is live. The automated validation gate checks this first — if `/ping` fails, you're disqualified before a human ever looks at your code.

### Viva: OpenEnv

**Q:** Why serve as a web API instead of a Python library?
**A:** Decouples agent from environment. Environment runs in Docker anywhere, agent trains on GPU. Also enables the judges' automated checker to verify the Space is live.

**Q:** What happens if `/ping` fails during judging?
**A:** Automatic disqualification. The validation gate hits `/ping` first.

---

## 9. Tech Stack

```
Environment:     openenv-core==0.2.3 (FastAPI-based RL framework)
Server:          FastAPI 0.111.0 + uvicorn
Training:        HF TRL 0.29.0 (GRPOTrainer) + vLLM colocate + LoRA (peft ≥0.11)
Models:          Qwen2.5-1.5B / 3B / 7B-Instruct (BF16 on H100)
Deployment:      Docker (python:3.11-slim) → HuggingFace Spaces (PORT 7860)
Compute:         H100 80GB (HF credits, onsite Apr 25–26)
Stats:           scipy 1.13.0 (power calculations, p-values)
Tests:           pytest, 249 passing. ruff for linting.
```

### Model presets

| Preset | LoRA Rank | Batch | Seq Len | Target |
|--------|-----------|-------|---------|--------|
| `1.5b` | 8 | 1 | 2048 | Qwen2.5-1.5B-Instruct |
| `3b` | 16 | 1 | 3072 | Qwen2.5-3B-Instruct |
| `7b` | 32 | 1 | 4096 | Qwen2.5-7B-Instruct |

### Viva: Tech stack

**Q:** What model and why?
**A:** Qwen2.5-Instruct with LoRA. Three size presets for fast iteration (1.5B ~3 GB) to highest quality (7B ~14 GB). LoRA trains ~1% of parameters so training converges faster. All fit comfortably on H100 (80 GB).

**Q:** Why Docker + HuggingFace Spaces?
**A:** Hackathon requirement. Judges run the environment against their evaluation harness. Our Space is live at `roopalgn-openenv-clinical-trial.hf.space`.

---

## 10. Pitch Strategy

### Judging weights

| Criterion | Weight | What Judges Want |
|-----------|--------|-----------------|
| **Environment Innovation** | 40% | Novel domain, real verification, hidden state, curriculum |
| **Storytelling** | 30% | Compelling narrative, before/after episodes, co-evolution bugs |
| **Showing Improvement** | 20% | Reward curves, baseline comparison, trained agent's learned behaviors |
| **Reward & Pipeline** | 10% | Decomposed reward, training pipeline, TRL/GRPO integration |

### 3-minute structure

**Minute 1 — The Story (30% storytelling):**
> "A drug works. But only in 15% of patients. The FDA needs proof. Can an LLM learn to design a clinical trial that finds those patients?"

Show the cold start: agent's first attempt scores -2.5. Skips Phase I, wrong dose, underpowered.

**Minute 2 — The Environment (40% innovation):**
Architecture: hidden ground truth → simulator → FDA rule engine → statistical verification → decomposed reward. Show Phase I → Phase II information flow. Show curriculum progression.

**Minute 3 — Results + Co-evolution (20% improvement + 10% pipeline):**
Reward curve improving. Before/after: random inclusion criteria → learned EGFR+ enrichment. Bugs found during training → environment improved.

### What winners did that we must match

1. **Multiple training runs with overlay comparison** — not just one lucky run
2. **Co-evolution story** — bugs found during training that improved the environment (document in `training_log.md`)
3. **Real `trainer.train()` calls** — not manual rollout loops
4. **Reward curves committed as .png** — Wandb-only doesn't count

---

## 11. Viva Questions — Full List

### Environment Design

1. **Q:** Why clinical trials specifically?
   **A:** Objective verification (p-value, power, FDA rules), real-world grounding ($2.6B/drug, 90% fail), partial observability, long-horizon planning, novel domain.

2. **Q:** Walk me through one episode.
   **A:** Scenario → Phase I (dose escalation, safety, effect estimate) → Phase II (endpoint, sample size, inclusion criteria, enrichment) → FDA review → trial simulation → p-value. If p < 0.05 and power ≥ 0.80 → success.

3. **Q:** What's the hardest thing for the agent?
   **A:** Finding the hidden responder population. Drug shows weak overall effect but strong EGFR+ effect. Agent must discover this from noisy Phase I data and enrich the trial.

4. **Q:** How is this different from a chatbot?
   **A:** Hidden ground truth the agent can't see, objective math-based verification, no shortcut exploits. A chatbot just generates plausible text.

### Reward & Verification

5. **Q:** Why decompose rewards?
   **A:** Single terminal reward gives no signal about which of 80+ actions helped. Decomposed rewards let GRPO pinpoint what's working at every step.

6. **Q:** Why is "LLM as judge" problematic?
   **A:** Slow (API call per step), expensive (tokens), noisy (different runs = different scores), exploitable (agent learns to fool the judge). Math verification is instant, free, deterministic, and unexploitable.

7. **Q:** How do you prevent reward hacking?
   **A:** Four objective layers: scipy.stats power, FDA rule engine, trial simulation, budget arithmetic. None use an LLM.

### Training

8. **Q:** Explain GRPO in one sentence.
   **A:** Generate 8 rollouts, rank by reward, push model toward better ones and away from worse ones.

9. **Q:** Why GRPO over PPO?
   **A:** No critic network needed, better for sparse rewards, lower memory, designed for LLMs with TRL integration.

10. **Q:** What if training shows no signal?
    **A:** Debug reward weights, switch to 1.5B for fast iteration, document the failure analysis as Statement 4 evidence. Even flat curves with analysis beat zero.

### Architecture

11. **Q:** Why serve as a web API?
    **A:** Decouples agent from environment. Judges can verify Space is live. Automated validation gate checks `/ping` first.

12. **Q:** Where did you train?
    **A:** All training onsite on HF H100 credits. Pipeline validated pre-event with dry-runs on Colab and Kaggle.

13. **Q:** Why not real clinical trial data?
    **A:** Grounded in rpact/scipy.stats (same math as FDA-validated software) + ICH E9 rules. Not toy heuristics. Real databases don't provide hidden ground truth for verification.

14. **Q:** How does difficulty adapt?
    **A:** Per-scenario mastery tracking with sliding windows. After mastery: parameters harden (effect shrinks, budget tightens, noise increases). Weak-spot targeting selects scenarios the agent struggles with.

---

## Glossary

| Term | Definition |
|------|-----------|
| **GRPO** | Group Relative Policy Optimization — ranks a group of responses by reward |
| **LoRA** | Low-Rank Adaptation — trains ~1% of model parameters for efficiency |
| **vLLM** | Fast LLM inference engine for generating rollouts |
| **POMDP** | Partially Observable MDP — agent can't see full state |
| **OpenEnv** | Meta's framework for RL environments as FastAPI apps |
| **TRL** | Transformer Reinforcement Learning — HuggingFace's RL training library |
| **ICH E9** | International guideline for statistical principles in clinical trials |
| **MTD** | Maximum Tolerated Dose — highest dose with acceptable side effects |
| **PFS** | Progression-Free Survival — time until disease worsens |
| **Cohen's d** | Standardized effect size (difference in means / pooled SD) |
