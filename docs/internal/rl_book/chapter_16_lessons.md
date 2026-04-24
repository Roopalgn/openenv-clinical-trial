# Chapter 16: Lessons, Pitfalls, and Next Steps

## Design Decisions: Why We Chose What We Chose

Throughout this book, we've highlighted individual design decisions. Here's the consolidated list:

### 1. GRPO over PPO
| | GRPO | PPO |
|---|---|---|
| Critic network needed? | No | Yes (+30% memory) |
| Works with reward function? | Yes | Yes |
| Natural for LLMs? | Yes (multiple generations) | Needs adaptation |
| GPU memory on single H100? | Fits | Tight |
**Winner: GRPO** — No critic means we fit on one GPU.

### 2. Objective Verification over LLM Judge
| | Math/Rules | LLM Judge |
|---|---|---|
| Reproducible? | Yes (same input → same output) | No (stochastic) |
| Cost? | Free (scipy.stats) | $0.01-0.10 per judgment |
| Gameable? | No | Yes (persuasive but wrong answers) |
| Speed? | Microseconds | Seconds |
**Winner: Math** — Reproducibility and zero cost.

### 3. Decomposed Rewards over Single Score
| | 8 Components | One Number |
|---|---|---|
| Debuggable? | Yes (see exactly what went wrong) | No (black box) |
| Tunable? | Yes (adjust individual weights) | Hard |
| Testable? | Yes (each component independently) | Hard |
**Winner: Decomposed** — Debuggability alone justifies the complexity.

### 4. LoRA over Full Fine-Tuning
| | LoRA | Full Fine-Tune |
|---|---|---|
| GPU memory? | ~14 GB | ~56 GB |
| Training speed? | Slightly slower per step | Faster per step |
| Quality? | ~95% of full fine-tuning | 100% |
| Fits on H100? | Yes, with room for vLLM | Barely |
**Winner: LoRA** — Quality sacrifice is minimal, memory savings enormous.

### 5. Curriculum Learning over Uniform Difficulty
| | Curriculum | Uniform |
|---|---|---|
| Early training? | Agent succeeds, learns from success | Agent fails constantly, learns nothing |
| Compute efficiency? | Focused on agent's actual skill level | Wastes time on too-easy or too-hard |
| Final performance? | Higher | Lower (less efficient learning) |
**Winner: Curriculum** — Like teaching a child: ABCs before Shakespeare.

### 6. Domain Randomization over Fixed Parameters
| | Randomized | Fixed |
|---|---|---|
| Generalization? | Good (works on unseen parameters) | Poor (memorizes specific values) |
| Reproducibility? | Yes (seeded RNG) | Yes (naturally) |
| Robustness? | High | Low |
**Winner: Randomization** — Prevents overfitting to specific scenarios.

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Reward Hacking
**What happens:** The agent finds a loophole in the reward function that gives high rewards without actually solving the problem.

**Example:** "The agent discovered that running `estimate_effect_size` 50 times in a row gives high info_gain rewards without ever completing the trial."

**How we prevent it:**
- Novelty reward encourages unique actions (repeating gives 0)
- Efficiency reward penalizes wasteful budget spending
- Timeout punishment (-2.0) for not completing the trial
- FDA rules block many exploitable action sequences

### Pitfall 2: Catastrophic Forgetting
**What happens:** The agent learns a new skill but forgets a previously mastered one.

**Example:** "After learning advanced depression trials, the agent forgot how to handle basic oncology trials."

**How we prevent it:**
- LoRA (only modifies a tiny fraction of weights — base knowledge preserved)
- Curriculum never goes backward (no demotion)
- Domain randomization mixes scenarios within each tier

### Pitfall 3: Mode Collapse
**What happens:** The model always generates the exact same response regardless of input.

**Example:** "For every observation, the agent outputs: `{action_type: set_primary_endpoint}`"

**How we prevent it:**
- Temperature sampling (0.7) during generation
- 8 parallel generations provide diversity
- Novelty reward bonus for trying different actions
- Phase ordering rewards encourage the full workflow

### Pitfall 4: The Credit Assignment Problem
**What happens:** Agent can't figure out which action led to success or failure.

**Example:** "The trial failed at step 40. Was it because of the dose chosen at step 8? The sample size at step 3? The missing biomarker at step 12?"

**How we prevent it:**
- Per-step rewards (not just terminal) — every action gets immediate feedback
- Potential-based reward shaping — breadcrumb rewards for partial progress
- High-quality decomposed rewards — the agent can learn from each component

### Pitfall 5: Overfitting to Training Seeds
**What happens:** Agent memorizes the specific scenarios from training seeds.

**Example:** "The agent always enrolls exactly 327 patients because that worked on seed 42."

**How we prevent it:**
- Domain randomization (±30% budget, ±20% time, etc.)
- Different seeds each episode
- Multiple scenarios with different challenges
- Adversarial designer at expert tier generates novel scenarios

## How to Build Your Own RL Project

If you want to build a project like this, here's the roadmap:

### Step 1: Define Your Environment
- What problem are you solving?
- What are the observations (what the agent sees)?
- What are the actions (what the agent can do)?
- What are the rewards (how you measure success)?

### Step 2: Build the Simulator
- Implement the environment with reset() and step()
- Make everything deterministic given a seed
- Write tests for every component

### Step 3: Design the Reward
- Start with a simple reward (success=+1, failure=-1)
- Decompose into interpretable components
- Add reward shaping if episodes are long
- Test that optimal behavior actually gets the highest reward

### Step 4: Add Curriculum
- Start with the easiest version of your problem
- Define clear advancement criteria
- Scale difficulty gradually

### Step 5: Set Up Training
- Choose your base model (Qwen, Llama, etc.)
- Configure LoRA (start with small rank, increase if needed)
- Configure GRPO (8 generations, learning rate 1e-5)
- Run a dry-run first (no GPU, verify pipeline)

### Step 6: Train and Iterate
- Monitor reward curves
- Check for reward hacking
- Adjust reward components if needed
- Save checkpoints frequently

### Step 7: Evaluate and Deploy
- Compare trained vs random agent
- Test on held-out seeds
- Deploy via Docker + FastAPI

## The Technology Stack

Here's every library we use and why:

| Library | Version | Purpose | Why This One? |
|---|---|---|---|
| **Python** | 3.11 | Language | Industry standard for AI |
| **PyTorch** | 2.x | Neural network framework | Most popular for research |
| **TRL** | 0.29.0 | RL for language models | HuggingFace, includes GRPO |
| **PEFT** | latest | LoRA implementation | Standard with TRL |
| **vLLM** | latest | Fast inference | Best throughput for LLMs |
| **Transformers** | latest | Model loading | HuggingFace ecosystem |
| **FastAPI** | latest | Web framework | Modern, fast, auto-docs |
| **Pydantic** | v2 | Data validation | Type safety, serialization |
| **scipy** | latest | Statistics (power calc) | Gold standard for stats |
| **numpy** | latest | Numerical computing | Foundation library |
| **uvicorn** | latest | ASGI server | Runs FastAPI |
| **matplotlib** | latest | Plotting | Standard plotting library |

## Glossary of All Key Terms

| Term | Definition |
|---|---|
| **AI** | Artificial Intelligence — systems that perform intelligent tasks |
| **ML** | Machine Learning — learning patterns from data |
| **RL** | Reinforcement Learning — learning from rewards |
| **LLM** | Large Language Model — neural network trained on text |
| **GRPO** | Group Relative Policy Optimization — our training algorithm |
| **PPO** | Proximal Policy Optimization — alternative RL algorithm |
| **DPO** | Direct Preference Optimization — learning from preferences |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning technique |
| **PEFT** | Parameter-Efficient Fine-Tuning — family including LoRA |
| **BF16** | Brain Float 16 — memory-efficient number format |
| **vLLM** | Fast inference engine for language models |
| **TRL** | Transformer Reinforcement Learning library |
| **Transformer** | Neural network architecture with self-attention |
| **Self-Attention** | Mechanism for tokens to focus on relevant other tokens |
| **Token** | A piece of text that the model processes |
| **Policy (π)** | The agent's decision-making strategy |
| **Reward** | A number indicating how good an action was |
| **Advantage** | How much better an action was vs. average |
| **Gradient** | Direction of steepest change (used to update weights) |
| **Learning Rate** | How big each weight adjustment step is |
| **Episode** | One complete run from start to finish |
| **Step** | One action within an episode |
| **Environment** | The world the agent interacts with |
| **Observation** | What the agent sees (may be noisy/partial) |
| **Latent State** | The true hidden state (agent can't see this) |
| **Partial Observability** | Agent can't see the full state |
| **Domain Randomization** | Adding random variation to prevent memorization |
| **Curriculum Learning** | Starting easy and gradually increasing difficulty |
| **Reward Shaping** | Adding extra signals to help the agent learn faster |
| **Potential-Based Shaping** | Shaping that preserves optimal policy |
| **Credit Assignment** | Figuring out which actions caused success/failure |
| **Reward Hacking** | Agent exploiting loopholes in the reward |
| **Overfitting** | Memorizing training data instead of learning general rules |
| **Catastrophic Forgetting** | Losing old skills when learning new ones |
| **Mode Collapse** | Model always generating the same output |
| **Statistical Power** | Probability of detecting a real effect (≥80% required) |
| **p-value** | Probability of results if drug doesn't work (<0.05 = significant) |
| **Effect Size** | How much the drug works (Cohen's d) |
| **Clinical Trial** | Controlled experiment testing a drug |
| **FDA** | U.S. Food and Drug Administration |
| **Phase I/II/III** | Stages of clinical testing |
| **Biomarker** | Biological indicator (e.g., EGFR+ gene) |
| **Placebo** | Inactive treatment for control group |
| **Docker** | Platform for packaging applications in containers |
| **FastAPI** | Python web framework for building APIs |
| **OpenEnv** | HuggingFace framework for RL environments |

---

## What's Next?

If you've read this entire book, you now understand:
- How AI, ML, and RL work from first principles
- How language models generate text and learn from rewards
- How to build a complex RL environment with rules, rewards, and curriculum
- How GRPO training works with LoRA and vLLM
- How to evaluate, deploy, and debug your RL system

**Your next project ideas:**
1. **Different domain:** Apply the same architecture to drug interaction prediction, surgical planning, or financial portfolio management
2. **Better rewards:** Add more reward components (e.g., patient diversity, ethical compliance)
3. **Multi-agent:** Have two agents collaborate — one designs the trial, another reviews it
4. **Bigger model:** Scale from 7B to 70B parameters (needs more GPUs)
5. **Real data integration:** Connect to ClinicalTrials.gov API for real-world validation

The most important lesson: **Start simple, test everything, scale gradually.** Don't try to build the expert-level system on day one. Build the warmup tier first, verify it works, then add complexity.

Good luck, and welcome to the world of reinforcement learning! 🧪
