# Intelligence Report: OpenEnv Hackathon - Multi-Agent RL Environment Design
## Executive Summary
This document synthesizes key intelligence from past OpenEnv Hackathon winners and finalists, providing actionable insights for building competitive multi-agent RL environments aligned with the hackathon themes.

## Hackathon Context & Themes
### Theme #1 - Multi-Agent Interactions
- **Goal:** Environments involving cooperation, competition, negotiation, and coalition formation for theory-of-mind reasoning and emergent strategic behavior.

- **Sub-themes:**

- Fleet AI: Scalable oversight - train oversight agents to monitor other AI agents
- Halluminate: Multi-actor environments where agents manage multiple actors to discover and achieve tasks
### Theme #2 - (Super) Long-Horizon Planning & Instruction Following
- **Goal:** Deep, multi-step reasoning with sparse/delayed rewards, enabling agents to decompose goals, track state over extended trajectories, and recover from mistakes.

- **Sub-themes:**

- Scale AI: Long-horizon workflows for Sales, Project Management, or HR & IT
- Mercor: Capped/uncapped rewards where frontier model rewards scale with token output
### Theme #3 - World Modeling
#### 3.1 Professional Tasks
- Real interaction with tools, APIs, or dynamic systems requiring consistent internal state and multi-step workflows.

- **Sub-themes:**

- Scaler AI Labs: Multi-App RL for enterprise workflows demonstrating complex business rules
#### 3.2 Personalized Tasks
- Personal assistant tasks like replying to messages, handling conflicts, email management.

- **Sub-themes:**

- Patronus AI: Consumer workflows with schema drift where APIs, contracts, and policies change
### Theme #4 - Self-Improvement
- **Goal:** Agents generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula for recursive skill amplification.

- **Sub-themes:**

- Snorkel AI: Simulated experts-in-the-loop with changing requirements/preferences
### Theme #5 - Wild Card
- Open-ended creativity that meaningfully adds value to LLM training.

## Winner Analysis: What Made Them Win
### 🥇 1st Place: Kube SRE Gym ($15K Prize)
- **Core Innovation:** Self-improving Kubernetes SRE training environment with adversarial self-play

#### Key Success Factors
Real Infrastructure, Not Simulation

Live GKE cluster with actual kubectl commands
Real OOMKills, CrashLoopBackOffs, ImagePullBackOffs
No shortcuts - every action has real consequences
Adversarial Self-Play Architecture

Adversarial Designer (Claude) → Real GKE Cluster → Agent (Qwen 1.7B) → LLM Judge → Curriculum Controller
Claude designs targeted incidents based on agent's weak spots
Curriculum automatically escalates difficulty
Multi-layer verification (programmatic + LLM judge)
Progressive Curriculum System

5 difficulty tiers: warmup → beginner → intermediate → advanced → expert
Per-fault-type mastery tracking (70% success = graduated)
Fast-track advancement (90%+ success after 3 episodes)
Judge persona scales: junior (lenient) → senior → principal (strict)
Decomposed Reward Function

R_t = validity + ordering + info_gain + efficiency + novelty + penalty + shaping
Per-step LLM judge score (-1.0 to +1.0)
Repeat penalty (-0.15 per repeated command)
Resolution bonus (+1.0 to +5.0, efficiency-scaled)
Phase-order bonus (+0.2 for correct SRE workflow)
Clear separation: successful episodes +3 to +8, failed -2.0
Phase-Aware SRE Workflow

Triage → Investigation → Mitigation → Fix → Verification
Rewards following correct diagnostic order
Penalizes skipping phases (jumping to fix without investigation)
Environment Co-Evolution

Training exposed bugs in the environment itself
Command parser was too strict, judge truncated snapshots
The agent's failures taught them to fix the environment
#### Technical Stack
Model: Qwen3-1.7B + LoRA (BF16)
Training: GRPO with TRL 0.29.0, vLLM colocate
Infrastructure: H100 GPU (80GB), 8 rollouts × grad_accum=8
Judge: Claude (external API) or Qwen3-14B (self-hosted)
#### Results
Episode 1: +1.80 reward (blind start)
Episode 4: +6.58 reward (learned kubectl discovery)
Episode 7: +6.79 reward (systematic debugging)
Mean: 3.48 | Best: 6.79 with adversarial difficulty
#### Winning Storytelling
Act 1: Agent knows nothing, tries random commands, fails (-2.0)
Act 2: Discovers kubectl get pods -A, connects symptoms to fixes (+3.95)
Act 3: Environment fights back with compound incidents
Act 4: Environment improves itself through agent failures
### 🥈 2nd Place: Bio Experiment Environment
- **Core Innovation:** POMDP-based biological experiment planning with hidden ground truth

#### Key Success Factors
Partially Observable World

Hidden biological state (true DE genes, pathways, regulatory networks)
Hidden technical noise (dropout, doublets, ambient RNA)
Agent only sees noisy simulated outputs
Must discover truth through structured experiments
Rich Action Vocabulary

21 discrete experiment steps across 3 categories:
Wet Lab (8): collect_sample, sequence_cells, validate_marker, etc.
Computational (10): run_qc, differential_expression, trajectory_analysis, etc.
Meta (3): synthesize_conclusion, design_followup, request_subagent_review
Realistic Scientific Constraints

Prerequisites (HARD): Can't run DE without normalized data
Resource constraints: Budget ($80K-$120K) and time (120-180 days)
Redundancy (SOFT): Penalizes repeating completed steps
Causal validity (SOFT): Can't make claims without evidence
Decomposed Reward System

R_t = validity(0.3) + ordering(0.2) + info_gain(0.4) + efficiency(0.3) + novelty(+0.1) + penalty(-0.15/violation) + shaping(γ=0.99)
Terminal reward adds: completeness (3.0) + calibration (4.0) + efficiency (1.0)
Overconfidence penalty (-0.5/claim for wrong high-confidence claims)
Four Curated Scenarios

Cardiac disease DE (easy, $80K, 120 days)
Hematopoiesis trajectory (medium, $100K, 150 days)
Perturbation immune (hard, $120K, 180 days)
Biomarker validation lung (medium, $90K, 150 days)
Each with paper references (DOIs), true DE genes, ground-truth mechanisms
Domain Randomization

Budget ±30%, time ±20%
Technical noise, batch effects, cell proportions, effect sizes
Prevents overfitting to specific scenarios
#### Technical Stack
Framework: OpenEnv 0.2.0, Pydantic 2.0 for schemas
Training: TRL GRPO + Unsloth 4-bit quantization
Models: Qwen2.5-7B-Instruct, Qwen3.5-0.8B/4B
Tools: Scanpy, Seurat, DESeq2, SCENIC (70+ bioinformatics tools registered)
Architecture Highlights:
NoiseModel centralizes all stochasticity (seeded numpy.Generator)
OutputGenerator conditions on hidden state, then injects realistic noise
RuleEngine enforces scientific constraints before action execution
TransitionEngine updates hidden state, spends resources, degrades quality on violations
### 🏆 Finalist: Voyager-VRAM
- **Core Innovation:** Workplace simulator with Voyager-inspired memory architecture

#### Key Success Factors
31 Tools Across 7 Channels

Email (5 tools), Slack (5), Drive (4), Sheets (5), Calendar (4), Notes (4), Meta-search (4)
Mirrors real project manager information navigation
Hidden state: stale spreadsheets, chat-only constraints, changed deadlines
Voyager Architecture Components

a) Skill Library:

Extracts reusable tool-call sequences from successful episodes
Learned: mail listing skill, mail-search-then-open workflow
Compounds capability over time
b) Working Memory (within episode):

Current Goal, Active Plan, Discovered Facts, Pending Subgoals, Recent Errors
Structured state persists across steps
c) Episodic Memory (cross-episode):

What worked, what failed, which tools were effective
Guides future exploration
Real Exploration vs Pattern Repetition

Basic LLM: hammers mail.list_inbox repeatedly
Voyager agent: diverse tool usage (searches, opens threads, reads sheets, checks chat)
21% improvement in shaped reward (5.75 vs 4.74)
Expert Iteration Training

Generate 4 trajectories per prompt
Keep the best (by shaped reward)
Fine-tune with SFT
Repeat × 3 rounds, 32 prompts each
SFT loss: 8.78 → 8.60 → 8.50
Capability Profile Expansion

Post-training: higher tool diversity, more facts discovered, better shaped reward
Not just scoring higher - exploring more intelligently
#### Technical Stack
Model: Qwen2.5-1.5B-Instruct (4-bit via Unsloth)
LoRA: Rank 16 (18.5M trainable / 1.56B total = 1.18%)
Environment: 25-step episodes, client_brief scenario
Future Direction:
VRAM-PM: Shrinking memory budgets forcing consolidation
Explicit memory probes forcing agents to learn what to forget
Aligns with cutting-edge research: MEM1, Memory-R1, Mem-alpha
## Critical Success Patterns Across Winners
1. Real Consequences Over Simulation
Kube SRE: Live K8s cluster, not mocked responses
Bio Experiment: Realistic noise injection, not deterministic outputs
VRAM: Hidden state requiring active discovery
Lesson: Shortcuts are detectable. Agents learn to exploit simulators, not solve problems.

2. Adversarial/Adaptive Difficulty
Kube SRE: Claude designs incidents targeting weak spots
Bio Experiment: Domain randomization prevents overfitting
VRAM: Voyager skill library compounds over time
Lesson: Static environments plateau. Self-improving environments co-evolve with agents.

3. Decomposed, Interpretable Rewards
All winners broke rewards into components:

Validity, ordering, info gain, efficiency, novelty, penalties
Terminal bonuses for completeness and calibration
Clear separation between success/failure for GRPO variance
Lesson: Opaque rewards are undebuggable. Decomposition enables reward engineering.

4. Progressive Curriculum
Kube SRE: 5 tiers with mastery tracking and fast-track advancement
Bio Experiment: 4 scenarios from easy to hard
VRAM: Expert iteration with best-of-N selection
Lesson: Cold starts fail. Warm-up → gradual escalation → expert challenges.

5. Multi-Layer Verification
Kube SRE: Programmatic health checks + LLM judge
Bio Experiment: Rule engine + reward decomposition + terminal calibration
VRAM: Working memory + episodic memory + skill library
Lesson: Single-point verification is fragile. Redundant checks catch edge cases.

6. Compelling Narrative
All winners told a story:

Kube SRE: "From blind to on-call" - agent learns from scratch
Bio Experiment: "Hidden biology" - agent discovers truth through experiments
VRAM: "Pattern repetition vs real exploration" - visual proof of learning
Lesson: Judges are human. Storytelling + demo > technical specs.

## Technical Implementation Insights
### Architecture Patterns
1. Environment Server Pattern
FastAPI/OpenEnv Server :8000
├─ Environment (reset/step)
├─ Backend (K8s/Simulator)
├─ Judge (LLM scoring)
├─ Curriculum Controller
└─ Adversarial Designer (optional)
2. Training Loop Pattern
H100 GPU
├─ OpenEnv Server :8000 (environment)
├─ vLLM :8001 (judge model, optional)
└─ GRPO Trainer (TRL + vLLM colocate)
   ├─ Agent model (Qwen 1.7B-8B + LoRA)
   ├─ 4-8 rollouts per prompt
   └─ Gradient accumulation
3. Reward Computation Pattern
# Per-step reward
r_step = (
    w_validity * validity_score +
    w_ordering * ordering_score +
    w_info_gain * (quality * (1 - uncertainty)) +
    w_efficiency * (1 - resource_fraction) +
    w_novelty * novelty_bonus +
    w_penalty * sum(violations) +
    gamma * (phi(s_next) - phi(s_current))  # potential shaping
)

# Terminal reward
r_terminal = (
    w_completeness * milestone_fraction +
    w_calibration * ground_truth_match +
    w_efficiency * resource_remaining +
    w_penalty * overconfidence_penalty
)
### Model Selection
Winners used small models (0.8B-8B):

Qwen3-1.7B (1st place)
Qwen2.5-7B, Qwen3.5-0.8B/4B (2nd place)
Qwen2.5-1.5B (finalist)
Why small models won:

Faster iteration during development
Lower compute cost for rollouts
Easier to debug (less emergent behavior)
LoRA fine-tuning is tractable
Proves the environment is the innovation, not model scale
### Training Techniques
GRPO (Group Relative Policy Optimization):

All winners used GRPO over PPO
Compares multiple rollouts of same prompt
Stable advantages without value function
Better for sparse, delayed rewards
Unsloth 4-bit Quantization:

2nd place and finalist used Unsloth
4-bit loading + LoRA for H100 efficiency
LoRA rank 16-32 typical (1-2% trainable params)
Expert Iteration:

VRAM used best-of-4 rejection sampling
3 rounds of SFT on best trajectories
Simple but effective for small models
### Infrastructure
H100 Setup (typical):

# Terminal 1: Judge model (optional)
trl vllm-serve --model Qwen/Qwen3-14B --port 8001

# Terminal 2: Environment server
uv run server

# Terminal 3: GRPO training
python train.py --vllm-mode colocate --num-generations 8
Cost Optimization:

External judge (Claude API) frees 28GB GPU memory
Colocated vLLM for agent inference
8 rollouts typical, 4 minimum for GRPO variance
## Judging Criteria Breakdown
Environment Innovation (40%)
What judges look for:

Novel problem space or unique twist on existing domain
Genuine difficulty (not trivially solvable)
Meaningful test of agent behavior
Real-world relevance
Winner strategies:

Kube SRE: First RL environment for real K8s debugging
Bio Experiment: POMDP with hidden ground truth + realistic noise
VRAM: Workplace memory management with Voyager architecture
Anti-patterns to avoid:

Toy problems with obvious solutions
Environments that can be gamed with simple heuristics
Overly complex environments that obscure the core challenge
Storytelling (30%)
What judges look for:

Clear problem explanation
Engaging demo showing agent behavior
Before/after comparison
Emotional arc (struggle → learning → success)
Winner strategies:

Kube SRE: "Episode 1: blind, Episode 4: discovery, Episode 7: mastery"
Bio Experiment: "Hidden biology revealed through systematic experiments"
VRAM: "Pattern repetition vs real exploration" with visual heatmaps
Demo best practices:

Show failure first (establishes difficulty)
Show learning curve (reward plots, skill progression)
Show final success (agent solving hard case)
Use visuals (charts, heatmaps, trajectory plots)
Showing Improvement in Rewards (20%)
What judges look for:

Observable training progress
Reward curves showing upward trend
Before/after behavior comparison
Metrics beyond just reward (steps-to-solve, success rate, tool diversity)
Winner strategies:

Kube SRE: Reward curves across 3 training runs, showing progression
Bio Experiment: Trajectory datasets with success rate, mean reward, episode length
VRAM: Training dashboard with loss, reward, and capability radar chart
Key metrics to track:

Mean/median reward per episode
Success rate (% episodes reaching goal)
Episode length (steps to completion)
Skill diversity (unique actions used)
Curriculum progression (difficulty tier reached)
Reward and Training Pipeline (10%)
What judges look for:

Coherent reward logic
Meaningful improvement in agent inference
Clean training pipeline (reproducible)
Evidence of actual learning (not just memorization)
Winner strategies:

Kube SRE: Decomposed reward with 6 components, phase-aware scoring
Bio Experiment: Decomposed reward + terminal calibration against ground truth
VRAM: Expert iteration with best-of-N selection
Pipeline best practices:

Provide Colab notebook for easy reproduction
Document hyperparameters and training time
Show training logs (loss, reward, KL divergence)
Include evaluation script comparing base vs trained model
## Actionable Recommendations
For Theme #1 (Multi-Agent Interactions)
Winning approach:

Build a negotiation or resource allocation environment
Multiple agents with conflicting goals
Emergent cooperation/competition dynamics
Oversight agent monitors and explains behavior
Technical implementation:

Multi-agent OpenEnv server with separate agent IDs
Shared state visible to all agents
Private state visible only to individual agents
Reward based on coalition formation, negotiation success, or oversight accuracy
Example: Compute cluster allocation where agents negotiate for GPU time, form coalitions, and an oversight agent detects unfair behavior.

For Theme #2 (Long-Horizon Planning)
Winning approach:

50+ step episodes with sparse terminal reward
Requires maintaining state across context limits
Decompose goals into subgoals
Recover from early mistakes
Technical implementation:

Episode memory system (working memory + episodic memory)
Subgoal tracking and completion rewards
Mistake recovery bonus (fixing earlier errors)
Curriculum starting with short horizons, scaling to 100+ steps
Example: Research paper implementation - read papers, understand requirements, write code, run experiments, debug failures, write report.

For Theme #3.1 (Professional Tasks)
Winning approach:

Real tool/API interaction (not mocked)
Multi-step workflows with dependencies
Partial observability (hidden state)
Realistic failure modes
Technical implementation:

Integrate real APIs (GitHub, Jira, Slack, etc.)
Hidden state requiring discovery
Tool costs (time, money, rate limits)
Verification against ground truth
Example: DevOps incident response (like Kube SRE but for AWS/Azure), or enterprise workflow automation (like VRAM but for sales/HR).

For Theme #3.2 (Personalized Tasks)
Winning approach:

Personal assistant scenarios
Conflicting constraints (dinner vs work meeting)
Tone/style matching for email replies
Delegation and prioritization
Technical implementation:

Calendar, email, message simulators
Personality profiles for contacts
Conflict detection and resolution
Reward based on user satisfaction (simulated)
Example: Executive assistant managing calendar conflicts, replying to emails in user's style, delegating tasks to team.

For Theme #4 (Self-Improvement)
Winning approach (proven by 1st place):

Adversarial designer generates challenges
Curriculum tracks mastery per skill
Difficulty escalates automatically
Environment improves alongside agent
Technical implementation:

LLM designer (Claude/GPT-4) creates scenarios
Curriculum controller tracks per-skill success rate
Mastery threshold (70%+) graduates skills
Adversarial mode unlocks at difficulty 0.6+
Example: Kube SRE Gym is the template. Adapt to other domains (security incident response, code review, system design).

## Minimum Viable Submission Checklist
### Must-Have (Minimum Requirements)
- Uses OpenEnv (latest release)
- Minimal training script using Unsloth or HF TRL in Colab
- Mini-blog on HuggingFace or mini-video on YouTube (<2 minutes)
- Environment innovation (novel problem or unique twist)
- Observable training progress (reward curves or before/after)
### Should-Have (Competitive)
- Decomposed reward function (3+ components)
- Progressive curriculum (3+ difficulty levels)
- Multi-layer verification (programmatic + LLM judge)
- Real consequences (not easily gamed)
- Compelling narrative (struggle → learning → success)
- Visual demo (charts, heatmaps, trajectory plots)
### Nice-to-Have (Winning Edge)
- Adversarial/adaptive difficulty
- Self-improving environment
- Domain randomization
- Multiple scenarios/tasks
- Skill library or memory system
- Expert iteration or best-of-N training
- Evaluation suite (multiple metrics)
- Paper references or ground truth validation
## Common Pitfalls to Avoid
1. Toy Problems
Environment solvable with simple heuristics
No genuine difficulty or exploration required
Agents memorize instead of learn
Fix: Add stochasticity, hidden state, or adversarial elements.

2. Opaque Rewards
Single reward number with no decomposition
Impossible to debug why agent succeeds/fails
No clear signal for GRPO variance
Fix: Break into 4-6 components with clear weights.

3. Static Difficulty
All episodes equally hard
Agent plateaus after initial learning
No progression or mastery tracking
Fix: Implement curriculum with 3-5 tiers.

4. Weak Verification
Single check that can be gamed
No ground truth comparison
False positives common
Fix: Multi-layer verification (programmatic + LLM + ground truth).

5. Poor Storytelling
Technical dump without narrative
No demo of agent behavior
No before/after comparison
Fix: Structure as story with emotional arc. Show failure → learning → success.

6. Unreproducible Training
No Colab notebook
Missing hyperparameters
Unclear training time or compute requirements
Fix: Provide complete Colab with documented hyperparameters.

## Winning Formula Template
1. Pick a domain with real-world relevance
   ↓
2. Design environment with:
   - Real consequences (not mocked)
   - Hidden state (partial observability)
   - Multi-step workflows (3-10 steps minimum)
   - Realistic failure modes
   ↓
3. Implement decomposed reward:
   - Validity (action succeeded?)
   - Ordering (correct sequence?)
   - Info gain (learned something?)
   - Efficiency (resource usage)
   - Novelty (exploration bonus)
   - Penalties (violations)
   ↓
4. Build progressive curriculum:
   - Tier 1: Warmup (easy, single-step)
   - Tier 2: Beginner (medium, 2-3 steps)
   - Tier 3: Intermediate (hard, 4-6 steps)
   - Tier 4: Advanced (very hard, 7-10 steps)
   - Tier 5: Expert (adversarial, 10+ steps)
   ↓
5. Add adversarial/adaptive elements:
   - LLM designer targets weak spots
   - Domain randomization prevents overfitting
   - Difficulty escalates with mastery
   ↓
6. Train with GRPO:
   - Small model (0.8B-8B)
   - LoRA fine-tuning (rank 16-32)
   - 4-8 rollouts per prompt
   - 10-30 episodes for demo
   ↓
7. Create compelling demo:
   - Show failure (episode 1)
   - Show learning (reward curve)
   - Show success (episode N)
   - Use visuals (charts, heatmaps)
   ↓
8. Package for judges:
   - Colab notebook (training)
   - HF blog or YouTube video (<2 min)
   - GitHub repo with README
   - Live HF Space (optional but impressive)
## Final Insights
What Separates Winners from Participants
1st place did 3 things exceptionally:

Real infrastructure - Live K8s cluster, not simulation
Self-improvement - Environment evolved alongside agent
Storytelling - "From blind to on-call" narrative
2nd place did 3 things exceptionally:

Hidden ground truth - POMDP with realistic noise
Scientific rigor - 70+ tools, 4 scenarios, paper references
Decomposed rewards - 6 components + terminal calibration
Finalist did 3 things exceptionally:

Memory architecture - Voyager-inspired skill library
Visual proof - Heatmaps showing exploration vs repetition
Research alignment - Connected to MEM1, Memory-R1, Mem-alpha
The Meta-Lesson
The environment is the innovation, not the model.

All winners used small models (0.8B-8B). They won because their environments were:

Novel and challenging
Self-improving or adaptive
Verifiable and interpretable
Compelling to watch
Build an environment that teaches agents something they can't learn elsewhere.

## Quick Start: 48-Hour Hackathon Plan
Day 1 (Environment)
Hour 0-4: Pick domain, design core mechanics
Hour 4-8: Implement basic environment (reset/step)
Hour 8-12: Add reward decomposition (3-4 components)
Hour 12-16: Implement curriculum (3 tiers minimum)
Hour 16-20: Add verification and ground truth
Hour 20-24: Test with random policy, fix bugs
Day 2 (Training & Demo)
Hour 24-28: Set up GRPO training pipeline
Hour 28-32: Run training (10-20 episodes)
Hour 32-36: Create reward curves and metrics
Hour 36-40: Build demo (Colab + visuals)
Hour 40-44: Write blog/record video
Hour 44-48: Polish and submit
## Critical Path Items
Must finish by Hour 24: Working environment with decomposed rewards
Must finish by Hour 36: Training run with observable improvement
Must finish by Hour 44: Demo materials (blog/video)
## Resources & References
### Code Repositories
1st Place: https://huggingface.co/spaces/openenv-community/kube-sre-gym
OpenEnv: https://github.com/meta-pytorch/OpenEnv
TRL: https://github.com/huggingface/trl
Unsloth: https://github.com/unslothai/unsloth
### Key Papers
GRPO: Group Relative Policy Optimization
Voyager: Lifelong Learning with LLMs
Curriculum Learning: Bengio et al.
Adversarial Self-Play: OpenAI Five, AlphaStar
### Tools
OpenEnv: Environment framework
TRL: GRPO training
Unsloth: 4-bit quantization + LoRA
vLLM: Fast inference
HuggingFace Spaces: Deployment
## Conclusion
The winning formula is clear:

Real consequences over simulation
Adversarial/adaptive difficulty
Decomposed, interpretable rewards
Progressive curriculum with mastery tracking
Multi-layer verification against ground truth
Compelling narrative with visual proof
The meta-insight:

"The agent's failures taught us to fix the environment. This is the self-improvement loop we didn't expect — not just the model getting better, but the training infrastructure co-evolving with it."

— Kube SRE Gym (1st Place)

Build an environment that improves itself. That's how you win.

Document compiled from analysis of OpenEnv Hackathon winners and finalists. All code examples, metrics, and insights are derived from public repositories and documentation.


## Repos of Top 3
Repos of top 3 -
- 1st: https://github.com/sid-rp/kube-sre-gym
#### Project Description
Kube SRE Gym is a self-improving RL environment where a small language model (Qwen3-1.7B) learns to diagnose and fix real Kubernetes production incidents from scratch. The agent interacts with a live GKE cluster via kubectl commands — OOMKills, CrashLoopBackOffs, and ImagePullBackOffs are real Kubernetes events, not simulations. An adversarial designer (Claude) creates targeted incidents based on the agent's tracked weaknesses, while a curriculum controller escalates difficulty as mastery improves. Training uses GRPO (TRL 0.29.0 + vLLM) with an LLM judge that scores SRE workflow quality using three expert personas (Junior/Senior/Principal). Within 8 episodes, the agent learned to discover cluster topology, identify fault types from pod status, and apply correct fixes — all from reward signal alone, with zero hardcoded knowledge of the cluster.

- 2nd: https://github.com/mhtruong1031/OpenENV-Hackathon/
#### Project Description
RL environment for autonomous biologist agents. Simulate an entire biological worldstate based on scientifically-accurate single-cell standards. Naive agents try to solve a frontier problem, such as identifying key pathways and markers for cancer growth, and have access to 40+ tool calls for fully implemented bioinformatics procedures. Agents probe and modify the world state and try to recover the generated "truth" of the given world using intermediate experimental results to guide future thinking. Problems range in difficulty and complexity, requiring more and more complex thought and workflows. Rewards are based on feasible flow of experiments and the accuracy of the final study conclusion, based on how close the recovered world state is to the hidden "true" world state.

- 3rd: https://github.com/owlgebra-ai/ShopRLVE-Gym
