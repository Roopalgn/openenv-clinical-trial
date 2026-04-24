# Chapter 1: What is AI, Really?

## The Big Picture: A Family Tree

Imagine you're organizing a library. Here's how AI, ML, and Deep Learning relate to each other:

```
Artificial Intelligence (AI)
├── Rule-Based Systems ("if temperature > 100, alert doctor")
├── Machine Learning (ML)
│   ├── Supervised Learning ("here are 1000 X-rays labeled cancer/no-cancer, learn the pattern")
│   ├── Unsupervised Learning ("here are 1000 customers, find natural groups")
│   └── Reinforcement Learning ("try designing trials, I'll tell you how good each attempt was")
│       └── Deep RL ("use a neural network brain to make decisions")
│           └── LLM-based RL ← THIS IS WHAT WE'RE BUILDING
└── Expert Systems, Search Algorithms, etc.
```

### AI = Any Machine That Acts Smart

AI is the broadest category. Your email spam filter? AI. A chess computer? AI. Siri? AI. It's any system that does something that would require intelligence if a human did it.

### Machine Learning = Learning from Data (Not Just Rules)

Traditional programming: you write rules → computer follows them.

```python
# Traditional programming
def diagnose(temperature):
    if temperature > 100.4:
        return "fever"
    else:
        return "normal"
```

Machine Learning: you give the computer examples → it discovers the rules itself.

```python
# Machine learning (conceptual)
examples = [
    {"temperature": 101.2, "label": "fever"},
    {"temperature": 98.6, "label": "normal"},
    {"temperature": 103.0, "label": "fever"},
    # ... thousands more ...
]
model = learn_pattern(examples)  # The computer figures out the rules
model.predict({"temperature": 99.8})  # → "normal"
```

**The key difference:** In traditional programming, YOU figure out the rules. In ML, the COMPUTER figures out the rules from examples.

### Deep Learning = ML with Neural Networks

A neural network is a particular way of organizing the "rule-finding" process. It's inspired (loosely) by how brain neurons connect to each other. 

Think of it like this: instead of one simple rule, you have layers of tiny decision-makers, each one looking at a small piece of the problem, passing their opinion to the next layer. After many layers (that's the "deep" part), you get a sophisticated answer.

```
Input: [temperature=101.2, heart_rate=95, age=45, ...]
  ↓
Layer 1: 128 tiny calculators each look at the input
  ↓
Layer 2: 64 tiny calculators each combine Layer 1's outputs
  ↓
Layer 3: 32 tiny calculators refine further
  ↓
Output: "fever" (probability: 0.92)
```

Each "tiny calculator" (neuron) does this simple math:
```python
output = activation(weight_1 * input_1 + weight_2 * input_2 + ... + bias)
```

The **weights** and **bias** are numbers the computer learns during training. That's it. The magic is that millions of these simple operations, stacked in layers, can learn incredibly complex patterns.

### Reinforcement Learning = Learning by Trial and Error

This is the type of ML we use in our project. Instead of learning from labeled examples ("this X-ray shows cancer"), the agent learns by **trying things and getting feedback**.

**Real-world analogy:** Imagine teaching a dog to sit. You don't show the dog 10,000 pictures of dogs sitting. Instead:
1. Dog does something → you say "good dog!" (positive reward) or "no!" (negative reward)
2. Over time, the dog learns which behaviors get treats
3. Eventually, the dog sits on command

That's reinforcement learning. The dog is the **agent**, the room is the **environment**, "sit!" is the **observation**, sitting down is the **action**, and the treat is the **reward**.

In our project:
- **Agent** = A language model (AI that can read and write text)
- **Environment** = A simulated clinical trial
- **Observation** = "Here's a cancer drug study. You've enrolled 50 patients. Budget: $1.2M remaining."
- **Action** = "Run an interim analysis" or "Enroll 100 more patients"
- **Reward** = +5 if the trial succeeds, -2 if it fails

### Why Not Just Use Supervised Learning?

You might ask: "Why not just show the AI a bunch of well-designed clinical trials and say 'learn to do this'?"

Two problems:

1. **There aren't enough examples.** There are only a few thousand well-documented clinical trials in history. Deep learning needs millions of examples.

2. **There's no single right answer.** Designing a clinical trial involves dozens of sequential decisions. Should you enroll 200 patients or 500? Test dose escalation first or jump to Phase II? These aren't yes/no questions — the right answer depends on what you learned in previous steps. RL handles this naturally because it learns from sequences of decisions, not isolated examples.

---

## Key Terminology You'll See Everywhere

| Term | Plain English | Example from Our Project |
|------|--------------|------------------------|
| **Model** | The thing that makes predictions (the "brain") | Qwen 2.5-7B language model |
| **Training** | The process of adjusting the model's weights so it gets better | Running GRPO for 300 episodes |
| **Inference** | Using the trained model to make predictions (no learning) | Asking the trained agent to design a trial |
| **Parameters** | The numbers inside the model that get adjusted during training | 7 billion weights in Qwen 2.5-7B |
| **Loss/Reward** | How we measure if the model is doing well | Reward score from -2 to +14 |
| **Epoch** | One complete pass through all training data | One full curriculum cycle |
| **Batch** | A small group of examples processed together | 8 trial episodes at once |

---

## What Makes Our Project Special?

Our project sits at the intersection of several AI disciplines:

```
Language Model (reads/writes English)
    +
Reinforcement Learning (learns from trial and error)
    +
Domain Simulation (realistic clinical trial environment)
    +
Curriculum Learning (starts easy, gets harder)
    =
An AI that learns to design clinical trials
```

This is **not** a chatbot. The AI doesn't just answer questions about clinical trials. It  actually **designs** them — choosing sample sizes, identifying patient populations, managing budgets, navigating FDA regulations, and deciding when to stop a trial that isn't working.

---

## Chapter 1 Glossary

| Keyword | Definition |
|---------|-----------|
| **AI (Artificial Intelligence)** | Any system that performs tasks requiring human-like intelligence |
| **ML (Machine Learning)** | Systems that learn patterns from data rather than following explicit rules |
| **Deep Learning** | ML using neural networks with many layers |
| **Neural Network** | A computation model with layers of interconnected nodes (neurons) |
| **Supervised Learning** | Learning from labeled examples (input → correct output pairs) |
| **Unsupervised Learning** | Finding patterns in unlabeled data |
| **Reinforcement Learning (RL)** | Learning by taking actions and receiving rewards |
| **Agent** | The decision-maker in RL (our language model) |
| **Environment** | The world the agent interacts with (our trial simulator) |
| **Weight** | A number inside the model that gets adjusted during training |
| **Parameter** | Same as weight (in the context of neural networks) |
