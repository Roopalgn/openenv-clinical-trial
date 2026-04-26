# Competitive Analysis — OpenEnv Hackathon Round 2

**Date**: April 26, 2026  
**Scope**: ~250 repos evaluated, top 10 shortlisted, deep comparison against 3 previous winners

---

## Judging Criteria

| Criterion | Weight | What It Means |
|-----------|--------|---------------|
| **Environment Innovation** | **40%** | Is the environment novel, creative, or genuinely challenging? |
| **Storytelling & Presentation** | **30%** | Clear problem explanation, engaging demo, accessible to non-technical audience? |
| **Showing Improvement in Rewards** | **20%** | Observable training progress — reward curves, before/after behavior, baselines? |
| **Reward & Training Pipeline** | **10%** | Coherent reward logic, meaningful improvement in agent behavior? |

---

## Previous Winners (Round 1)

| Place | Repo | Score | Key Strength |
|-------|------|:---:|---|
| 🥇 1st | **sid-rp/kube-sre-gym** | 9.5 | Real K8s cluster, 4-Act narrative, 3 training runs with lessons from each |
| 🥈 2nd | **mhtruong1031/OpenENV-Hackathon** (Bio Experiment) | 9.0 | Bio experiment POMDP, 4 scenarios with paper DOIs, deep architecture docs |
| 🥉 3rd | **owlgebra-ai/ShopRLVE-Gym** | — | Minimal README at time of evaluation |

### What Made Kube SRE Gym Win (1st Place)

1. **Narrative hook**: Opens with *"Can a 0.6B model learn to be an on-call SRE — from scratch?"*
2. **4-Act story structure**: Cold Start → First Light → Environment Fights Back → Environment Improves Itself
3. **Real infrastructure**: kubectl against a live GKE cluster, not a simulator
4. **3 separate training runs** with honest analysis of each (Run 1: too noisy → Run 2: too generous → Run 3: adversarial, working)
5. **Before/after behavioral evidence**: Showed what the agent literally typed at episode 1 vs episode 7
6. **Co-evolution narrative**: "The agent's failures taught us to fix the environment" — bugs discovered through training
7. **Reward plot images committed**: Visual proof of training progress

### What Made Bio Experiment Win (2nd Place)

1. **Deep technical architecture**: POMDP with hidden biological state, 21 action types, 9 sub-agent roles
2. **4 curated scenarios** grounded in real biology with paper DOIs
3. **Comprehensive interfaces**: In-process, HTTP, WebSocket, local agent, GRPO training, rollout collection, benchmark
4. **Literature benchmark**: Paper-aligned action sequences compared against expected findings
5. **Exhaustive documentation**: Every component described with code examples

---

## Top 10 Competitors (Round 2)

| # | Repo | Weighted | Domain | Standout Feature |
|---|------|:---:|---|---|
| 1 | **sreenathmmenon/asha-sahayak** | 9.3 | Healthcare (ASHA workers) | 44 medical cases, 0.31→0.947 reward (+142%), multi-agent, Indian govt IMNCI protocols |
| 2 | **sri11223/openEnv** (SENTINEL) | 9.3 | AI Safety (oversight) | AI-supervising-AI, 9.7× score improvement, constitutional principles, zero-shot generalization |
| 3 | **shreyas-garg/OpenEnv** (LeniencyBench) | 9.3 | AI Safety (policy drift) | Discovered real LLM failure: obeys loosening, ignores tightening (0% accuracy), SFT fix to 91.3% |
| 4 | **prithidevghosh/mindflayer** | 9.1 | Social deduction | 0.5B model learns deceptive Theory-of-Mind vs 3 GPT-4o-mini investigators |
| 5 | **abhinavgautam01/GPU_Budget_Negotiation_Arena** | 9.0 | Resource negotiation | 5-agent negotiation, 11 reward components, SFT→GRPO pipeline, 55 tests |
| 6 | **parthdagia05/among-us-deception-gym** | 8.9 | Social deduction | Vote accuracy 32.7%→96.7%, 4 mechanical rewards, 90% OOD accuracy |
| 7 | **SupreethRao99/veriRL** | 8.9 | Hardware design | Verilog synthesis graded by real EDA tools (iverilog, yosys, SymbiYosys) |
| 8 | **Chirag0096/ShiftLog-Gym** | 8.8 | SRE (causal memory) | Causal recall 18.4%→91.2%, MTTR 18.3→3.5 steps, 3-stage curriculum |
| 9 | **dino65-dev/incident-response-env** | 8.8 | Cybersecurity (SOC) | 6 escalating incidents, POET-inspired evolution + Elo ratings, 11 grading dimensions |
| 10 | **Dinesh052/crisis-negotiator-openenv** | 8.7 | Crisis negotiation | FBI BCSM model, 6-agent system, 4 adversarial self-play rounds, ToM prediction |

### Honorable Mentions

| Repo | Score | Notable |
|------|:---:|---|
| **ehsaaniqbal/circuit_detective** | 8.7 | Mechanistic interpretability via RL — genuinely unprecedented domain |
| **1919-14/intellicredit-openenv** | 8.7 | Multi-agent MSME credit appraisal, 10× reward on hardest task |
| **Rushhaabhhh/HONEST-RL-Calibrator** | 8.5 | Brier-score reward for calibrated confidence — novel RL objective |
| **Deltasthicc/F1_Simulator_OpenENV** | 8.8 | Weather_roulette 0.344→0.935, postmortem self-improvement loop |
| **maddycruzz/openenv-negotiation** | 8.6 | 1B model matches 70B on structured negotiation via GRPO |
| **Bhavneet1492/openenv-methanol-apc** | 8.3 | Production-grade digital twin, 5 kinetic models, zero emergency shutdowns |

---

## Our Project: openenv-clinical-trial

### Current Estimated Score

| Criterion | Weight | Our Score | Notes |
|-----------|--------|:---:|---|
| Environment Innovation | 40% | **8/10** | POMDP, 8-component reward, math-verified, power-gating — strong |
| Storytelling & Presentation | 30% | **5/10** | Technical but no narrative arc, no compelling hook, blog buried |
| Showing Improvement in Rewards | 20% | **5/10** | Reward table exists but no plot image, no before/after transcripts |
| Reward & Training Pipeline | 10% | **8/10** | Coherent decomposed reward, GRPO pipeline working |
| **Weighted Total** | | **~6.5** | |

### What We Do Well

1. **8-component decomposed reward** — on par with or better than most competitors
2. **Math-verified outcomes** (scipy.stats, not LLM judge) — only ~5 repos do this; most rely on LLM judges
3. **Latent state / POMDP design** — hidden true effect size forces genuine reasoning under uncertainty
4. **Power-gated terminal bonus** — elegant anti-gaming that prevents small-n lucky p-values
5. **Architecture documentation** — ARCHITECTURE.md is thorough with ASCII diagrams
6. **267 passing tests** — one of the highest test counts in the competition
7. **Debugging narrative in blog** — 3 problems + 3 fixes shows honest iteration

### Gaps vs Top Competitors

#### Gap 1: Storytelling / Narrative Hook (30% weight — BIGGEST impact)

**What the winners do:**
- Kube SRE: *"Can a 0.6B model learn to be an on-call SRE — from scratch?"* + 4 dramatic Acts
- Mindflayer: *"Can a 0.5B model learn to lie?"*
- ASHA: Opens with India's 1.07M frontline health workers

**What we do:**
- Opens with a technical badge and table of links. No question. No story. No drama.

**Recommended fix:** Add a 3-sentence narrative hook ("Can a 1.5B model learn to design a statistically rigorous clinical trial from scratch? We gave it a budget, a novel compound with unknown efficacy, and zero knowledge of ICH guidelines. Within 30 episodes, it was designing Phase I→II trials with adequate statistical power."). Add 2-3 "Act"-style sections.

#### Gap 2: Reward Plot Image Missing (20% weight)

**What the winners do:**
- Kube SRE: 3 separate reward curve PNGs with analysis of each
- Among Us: Clear accuracy progression plots
- ASHA: Before/after metric tables with held-out eval

**What we do:**
- README references `![Reward Plot](docs/reward_plot.png)` but **no .png file exists in the repo**. This is a broken image for judges.

**Recommended fix:** Generate and commit the reward plot. Show at least 2 views: raw reward per step + rolling average.

#### Gap 3: Before/After Agent Behavior (20% weight)

**What the winners do:**
- Kube SRE: Shows the agent's actual kubectl commands at episode 1 vs episode 7
- Among Us: Vote accuracy 32.7% → 96.7% with specific examples
- ShiftLog: MTTR 18.3 → 3.5 steps

**What we do:**
- Reward table (+7.26 → +8.11 rolling avg) but no episode transcripts shown. We have 1000+ transcripts in `logs/episode_transcripts/` but none are surfaced.

**Recommended fix:** Pick 1 early transcript (agent skipping phases, failing) and 1 late transcript (agent completing full workflow). Show the actual action sequences side-by-side.

#### Gap 4: Baseline / Ablation Comparison

**What the winners do:**
- Kube SRE: 3 training runs (too noisy / too generous / working) with lessons from each
- LeniencyBench: Cross-model baselines (GPT-4o, Claude, etc.)
- SENTINEL: Random vs trained with percentages

**What we do:**
- Single 30-step run. No random baseline, no ablation, no failed run analysis.

**Recommended fix:** Add at minimum: "Random policy baseline: +2.1 avg | Trained (30 steps): +7.58 avg | Improvement: +261%". Show the "Before Fixes" collapsed run alongside the working run.

#### Gap 5: Held-Out / Generalization Evidence

**What the winners do:**
- Among Us: 90% OOD accuracy
- SENTINEL: Zero-shot generalization to unseen misbehavior types
- Snitch-env: Held-out variant-3 accuracy (statistically significant, p=0.0017)

**What we do:**
- All metrics from training run itself. No held-out evaluation.

**Recommended fix:** Run 5-10 episodes with unseen seeds and report the reward separately.

#### Gap 6: Anti-Gaming / Limitations Section

**What the winners do:**
- LeniencyBench: Honest limitations section naming every weakness
- Among Us: 4 anti-hacking safeguards documented
- Oversight Arena: Ablation proving guardrails are load-bearing

**What we do:**
- Power-gating and violation penalties exist but aren't framed as anti-gaming. No limitations section.

**Recommended fix:** Add "Anti-Gaming Safeguards" subsection. Add "Known Limitations" (single model size, 30 steps, no held-out eval).

---

## Priority-Ranked Action Items

| Priority | Action | Effort | Score Impact |
|----------|--------|--------|:---:|
| **P0** | Commit reward plot image(s) to repo | 5 min | +2-3 pts on Reward (20%) |
| **P0** | Rewrite README opening with narrative hook + story arc | 30 min | +2-3 pts on Storytelling (30%) |
| **P1** | Add before/after episode transcript excerpts | 20 min | +1-2 pts on Storytelling + Reward |
| **P1** | Add baseline comparison (random vs trained numbers) | 15 min | +1-2 pts on Reward |
| **P2** | Add anti-gaming + known limitations section | 10 min | +1 pt on Innovation |
| **P2** | Show multiple training runs or ablation | 30 min | +1 pt on Reward |
| **P3** | Run and report held-out evaluation | 15 min | +1 pt on Pipeline |

### Projected Score After P0+P1 Fixes

| Criterion | Weight | Current | After Fixes |
|-----------|--------|:---:|:---:|
| Environment Innovation | 40% | 8 | 8.5 |
| Storytelling & Presentation | 30% | 5 | 7.5 |
| Showing Improvement | 20% | 5 | 7.5 |
| Pipeline Quality | 10% | 8 | 8.5 |
| **Weighted Total** | | **~6.5** | **~7.9** |

---

## Key Insight

Our **environment engineering is competitive** (top-quartile innovation and pipeline). The gap is almost entirely in **presentation**: storytelling (30%) and visual evidence of improvement (20%). These are the cheapest fixes with the highest ROI — they require no changes to the actual codebase, just better surfacing of what already exists.

The 1st place winner (Kube SRE) didn't win on environment complexity — they won on **the story of how their agent learned**. We have that story (3 debugging problems → 3 fixes → working training) but it's buried in `docs/blog.md` instead of front-and-center in the README.
