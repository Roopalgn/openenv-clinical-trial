# Chapter 11: GRPO — How the Agent Actually Learns

## The Million-Dollar Question

We've covered the environment, rewards, rules, and curriculum. But how does the agent actually get BETTER? How do the model's weights change so that it goes from making random decisions to designing expert-level clinical trials?

The answer is **GRPO: Group Relative Policy Optimization**.

## Building Up to GRPO

### Step 1: What is a Gradient?

Remember from Chapter 3 that a neural network has weights (numbers). The gradient tells us: "If I nudge this weight up slightly, does the output get better or worse?"

**Analogy:** You're blindfolded on a hill, trying to find the lowest point (the valley). You can't see, but you can feel the slope under your feet. The gradient is the slope — it tells you which direction is downhill.

```
If gradient is positive: moving right goes uphill (bad)
If gradient is negative: moving right goes downhill (good)
We always step OPPOSITE to the gradient (downhill)
```

In neural networks:
- We calculate the gradient of the **loss** with respect to each weight
- We adjust weights to **decrease** the loss (or **increase** the reward)
- This is called **gradient descent**

### Step 2: What is Policy Gradient?

In RL, we don't have a fixed dataset with "correct answers." Instead, we have a policy (the model's behavior) and rewards. Policy gradient methods adjust the model to produce **more of what got rewarded and less of what got punished**.

The core idea:

```
For each action the model took:
  If the reward was HIGH → increase the probability of taking this action
  If the reward was LOW  → decrease the probability of taking this action
```

Mathematically: 

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A_t\right]$$

Don't panic! Let me translate:
- $\nabla_\theta$ = "adjust the weights in the direction that..."
- $\log \pi_\theta(a_t | s_t)$ = "...increases the log-probability of the action we took..."
- $A_t$ = "...proportional to how good that action was (the advantage)"

If the advantage $A_t$ is positive (good action), we increase the action's probability.
If it's negative (bad action), we decrease it.

### Step 3: What is Advantage?

The **advantage** tells us: "Was this action better or worse than AVERAGE?"

```
Example:
  Average reward from this state: +3.0
  Reward we actually got:         +7.0
  Advantage: +7.0 - 3.0 = +4.0   (much better than average → reinforce!)
  
  Average reward from this state: +3.0
  Reward we actually got:         +1.0
  Advantage: +1.0 - 3.0 = -2.0   (worse than average → discourage!)
```

The advantage is more useful than the raw reward because it's **relative**. Even in a hard scenario where all rewards are negative, the advantage distinguishes "less bad" from "more bad."

## GRPO: The Algorithm We Use

GRPO stands for **Group Relative Policy Optimization**. It's a policy gradient method specifically designed for language models. Let's break it down.

### The Key Insight: Compare Multiple Outputs

For each situation, GRPO generates **multiple** responses (typically 8) and compares them:

```
Situation: "You're designing a lung cancer trial. 120 patients enrolled. 
            Budget: $4.2M. Phase I complete."

Generation 1: {"action_type": "run_interim_analysis", ...}    → reward: +3.2
Generation 2: {"action_type": "enroll_patients", ...}          → reward: +1.8  
Generation 3: {"action_type": "set_sample_size", ...}          → reward: -1.0
Generation 4: {"action_type": "run_interim_analysis", ...}    → reward: +2.9
Generation 5: {"action_type": "observe_safety_signal", ...}    → reward: +0.5
Generation 6: {"action_type": "estimate_effect_size", ...}     → reward: +2.1
Generation 7: {"action_type": "modify_sample_size", ...}       → reward: -0.5
Generation 8: {"action_type": "run_interim_analysis", ...}    → reward: +3.5

Mean reward: +1.56

Advantages:
Gen 1: +3.2 - 1.56 = +1.64  (good!)
Gen 2: +1.8 - 1.56 = +0.24  (slightly above average)
Gen 3: -1.0 - 1.56 = -2.56  (bad!)
Gen 8: +3.5 - 1.56 = +1.94  (best!)
```

GRPO then updates the model to:
- **INCREASE** probability of generations 1, 8 (high advantage)
- **DECREASE** probability of generation 3 (low advantage)
- **Slightly adjust** others proportionally

### Why 8 Generations?

```
1 generation:  No comparison possible. Can't compute advantage.
2 generations: Very noisy comparison. Plus or minus, that's it.
4 generations: Better, but still noisy.
8 generations: Good balance of comparison quality vs. GPU memory.
16 generations: Slightly better but uses too much GPU memory.
```

8 is the sweet spot for our setup (H100 80GB GPU).

### No Critic Network Needed!

In PPO (the traditional RL algorithm for LLMs), you need a separate "critic" network that estimates the average value of each state. This critic:
- Uses extra GPU memory (~30% overhead)
- Must be trained alongside the main model
- Can be wrong, leading to bad value estimates

GRPO eliminates the critic by computing advantages **from the group directly**:

```
Average reward of the 8 generations = estimated value
Advantage = individual reward - average

No separate network needed!
```

> **Design Decision Box: GRPO vs PPO vs DPO**
>
> | Algorithm | How It Works | Pros | Cons |
> |---|---|---|---|
> | **PPO** | Critic estimates value, policy is updated | Well-proven, stable | Needs critic (+30% memory) |
> | **DPO** | Pairs of "good" and "bad" responses, no RL needed | Simple, stable | Needs a preference dataset (we don't have one) |
> | **GRPO** ★ | Group of responses, relative ranking for advantage | No critic, works with reward function | Needs N parallel generations per step |
>
> We chose GRPO because:
> 1. No critic = less GPU memory (critical for 7B model on single GPU)
> 2. Works with any reward function (we have a detailed 8-component one)
> 3. Natural fit for language models (generate multiple, rank, learn)
> 4. Part of TRL library (well-maintained by HuggingFace)

### The GRPO Update Step

After generating 8 responses and computing advantages:

```python
# Simplified GRPO update (conceptual)
for i, (response, advantage) in enumerate(zip(responses, advantages)):
    # Calculate how likely the model currently considers this response
    log_prob = model.log_probability(response, given=prompt)
    
    # Calculate how likely the OLD model considered this response
    log_prob_old = old_model.log_probability(response, given=prompt)
    
    # Ratio: how much has the model changed?
    ratio = exp(log_prob - log_prob_old)
    
    # Clipped objective (prevents too-large updates)
    clipped_ratio = clip(ratio, 1 - epsilon, 1 + epsilon)
    
    # Loss = minimum of clipped and unclipped (conservative)
    loss = -min(ratio * advantage, clipped_ratio * advantage)
    
    # Update model weights
    loss.backward()
    optimizer.step()
```

The "clipping" (typically ε=0.2) prevents the model from changing too much in one step. Without clipping, the model could make a huge update that breaks everything. This is the "Proximal" part — keeping updates close to the current policy.

## Our Training Code

Here's how training actually works in our project:

```python
# From train.py (simplified)

def train(args):
    # 1. Load the language model
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,  # BF16 for memory efficiency
        device_map="auto",           # Automatically use GPU
    )
    
    # 2. Configure LoRA (see Chapter 12)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,                    # LoRA rank
        lora_alpha=64,           # Scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # Which layers to train
    )
    
    # 3. Configure GRPO
    grpo_config = GRPOConfig(
        output_dir="outputs/grpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=8,            # Generate 8 responses per step
        max_completion_length=4096,   # Max tokens per response
        learning_rate=1e-5,           # How big each weight update is
        bf16=True,                    # Use BF16 math
        seed=42,                      # Reproducibility
    )
    
    # 4. Create the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )
    
    # 5. Run episodes
    for episode in range(args.episodes):
        # Reset environment
        obs = env.reset(seed=args.seed + episode)
        
        for step_idx in range(args.max_steps):
            # Build prompt from observation
            prompt = build_prompt(obs)
            
            # Model generates 8 responses
            responses = model.generate(prompt, num_return_sequences=8)
            
            # Score each response through the environment
            rewards = []
            for response in responses:
                action = parse_action(response)
                obs, reward, done, info = env.step_full(action)
                rewards.append(reward)
            
            # GRPO computes advantages and updates weights
            advantages = [r - mean(rewards) for r in rewards]
            # ... weight update happens inside trainer ...
            
            if done:
                break
```

### The Reward Function for GRPO

GRPO needs a reward function that takes a model completion and returns a number:

```python
def _grpo_reward_fn(completions, env, seed, max_steps):
    """For each of the 8 completions, run a mini-episode and return rewards."""
    rewards = []
    for i, completion in enumerate(completions):
        env.reset(seed=seed + i)
        total = 0.0
        for step_idx in range(max_steps):
            action = _build_action_from_text(completion, step_idx)
            _, reward_dict, done, _ = env.step_full(action)
            total += sum(reward_dict.values())
            if done:
                break
        rewards.append(total)
    return rewards
```

Each of the 8 completions gets scored by actually running it through the environment. This is ground-truth scoring — no approximation.

## The Learning Rate: How Fast to Learn

The **learning rate** (1e-5 = 0.00001) controls how big each weight update is:

```
Too high (1e-3): Model changes too fast → unstable, forgets pre-training
Too low (1e-7):  Model barely changes → learning takes forever
Just right (1e-5): Gradual, stable improvement
```

**Analogy:** The learning rate is like how far you step when walking blindfolded toward the valley (gradient descent). Big steps get you there faster but you might overshoot. Tiny steps are precise but slow. 1e-5 is a well-tested default for fine-tuning 7B language models.

## Training Dynamics: What Actually Happens

```
Early training (episodes 1-30):
  - Model generates mostly garbage JSON
  - Fallback parser kicks in frequently
  - Rewards are mostly negative
  - Agent learns: "Some actions give penalties, others don't"

Mid training (episodes 30-100):
  - Model generates valid JSON ~70% of the time
  - Agent learns the basic workflow
  - Rewards start getting positive (+3 to +5)
  - Agent learns: "Running Phase I before Phase II is important"

Late training (episodes 100-300):
  - Model generates valid JSON ~95% of the time
  - Agent navigates scenarios strategically
  - Rewards range from +6 to +12
  - Agent learns: "For depression trials, enrich for severe patients"
  - Agent learns: "Don't always pick the highest dose"
```

---

## Chapter 11 Glossary

| Keyword | Definition |
|---------|-----------|
| **GRPO (Group Relative Policy Optimization)** | RL algorithm that compares multiple outputs to compute advantages |
| **Policy Gradient** | Family of RL algorithms that directly optimize the policy |
| **Gradient** | Direction and magnitude of steepest change in a function |
| **Gradient Descent** | Optimization by stepping in the direction opposite to the gradient |
| **Advantage** | How much better an action was compared to average |
| **Critic** | A separate network that estimates state values (GRPO doesn't need one) |
| **PPO (Proximal Policy Optimization)** | Classic RL algorithm with a critic network |
| **DPO (Direct Preference Optimization)** | Learning from preference pairs (good vs bad) |
| **Clipping (ε-clipping)** | Limiting how much the policy can change in one update |
| **Learning Rate** | How big each weight adjustment step is |
| **num_generations** | Number of parallel responses GRPO generates per step (8) |
| **TRL (Transformer Reinforcement Learning)** | HuggingFace library for RL with language models |
| **Advantage Function** | A(s,a) = Q(s,a) - V(s) or simply reward - mean(rewards) in GRPO |
| **Log Probability** | How likely the model considers a particular text sequence |
