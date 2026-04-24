# Chapter 3: Language Models — The Brain of Our Agent

## What is a Language Model?

A language model is a program that has been trained on massive amounts of text to predict "what comes next." That's it. At its core, it's a next-word predictor.

```
Input:  "The cat sat on the ___"
Output: "mat" (probability: 0.35), "floor" (0.20), "chair" (0.15), ...
```

But here's the amazing thing: when you scale this simple idea to **billions** of parameters and **trillions** of words of training data, the model doesn't just predict words — it learns to reason, follow instructions, write code, and yes, design clinical trials.

## Tokens: How Computers Read Text

Computers don't understand letters. They understand numbers. So the first step is converting text to numbers, using a process called **tokenization**.

```
Text:   "Run dose escalation with 100mg"
Tokens: ["Run", " dose", " escal", "ation", " with", " 100", "mg"]
IDs:    [6955,  17216,  42719,   367,   449,   1041,  8838]
```

Notice:
- Common words like "Run" get one token
- Less common words like "escalation" get split into pieces ("escal" + "ation")
- Numbers and units get their own tokens

**Why tokens matter:** The model processes tokens, not characters. A model with "4096 max tokens" can handle roughly 3,000 words of text. This limits how much context (trial history, observations) we can feed the agent.

In our project:
```python
# From train.py — we tokenize the observation before feeding it to the model
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
```

## The Transformer: The Architecture Behind Modern AI

Almost every modern language model (GPT, Claude, Llama, Qwen — the one we use) is built on an architecture called the **Transformer**, invented in 2017.

### The Key Idea: Attention

Imagine reading this sentence: "The **patient** who was enrolled in the **trial** last **month** showed significant improvement."

As a human, when you read "improvement," your brain connects it back to "patient", "trial", and "month." You don't process each word in isolation — you pay **attention** to relevant earlier words.

Transformers do exactly this. For each token, they compute **attention scores** to every other token, determining which ones are relevant:

```
Processing the word "improvement":
  → "patient": attention = 0.35 (very relevant — who improved?)
  → "trial":   attention = 0.25 (relevant — context of improvement)
  → "month":   attention = 0.15 (somewhat relevant — time frame)
  → "the":     attention = 0.01 (not relevant)
  → "who":     attention = 0.02 (not relevant)
```

This is called **self-attention**, and it's computed for every token, at every layer, simultaneously. A single layer of a Transformer computes:

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

Don't worry about the math — just understand that:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What do I contain?"
- **V (Value)**: "What information do I provide?"

Each token asks (Q), "Who has relevant information for me?", looks at everyone's keys (K) to find matches, and combines the matching values (V).

### Why Transformers Changed Everything

Before Transformers, we had RNNs (Recurrent Neural Networks) that processed text one word at a time, left to right:

```
RNN:    word₁ → word₂ → word₃ → ... → word₁₀₀₀  (slow, forgets early words)
Transformer: [word₁, word₂, word₃, ..., word₁₀₀₀]  (all at once, remembers everything)
```

Transformers process **all tokens simultaneously** (in parallel), which means:
1. They can run on GPUs (which are good at parallel computing)
2. They don't forget early tokens (attention reaches everything)
3. They scale to billions of parameters

## Pre-Training and Fine-Tuning

Language models are created in two stages:

### Stage 1: Pre-Training (Done by Big Companies)

A company (like Alibaba for Qwen, MetaAI for Llama) trains a model on the entire internet:

```
Training data (trillions of tokens):
- Wikipedia, books, scientific papers
- Code repositories (GitHub)
- News articles, forums, Q&A sites
- Clinical trial reports, FDA documents
```

This takes **thousands of GPUs running for weeks** and costs millions of dollars. The result is a "foundation model" — like Qwen2.5-7B — that understands language generally but isn't specialized at anything.

### Stage 2: Fine-Tuning (What We Do)

We take the pre-trained model and teach it our specific task through reinforcement learning:

```
Pre-trained Qwen 2.5-7B:
  "I can answer general questions, write code, summarize text..."
  "But I have no idea how to design a clinical trial."

After RL fine-tuning:
  "I know that for EGFR+ tumors, biomarker stratification 
   dramatically improves statistical power."
  "I should run dose escalation before estimating effect size."
  "A depression trial needs extra patients because placebo 
   response is high."
```

> **Design Decision Box: Why Qwen 2.5-7B?**
>
> We need a model that:
> 1. **Understands English** well enough to read complex scenario descriptions
> 2. **Generates structured JSON** reliably (our actions are JSON objects)
> 3. **Fits on one GPU** (7B parameters × 2 bytes in BF16 = 14GB)
> 4. **Is open-source** (we need to modify the weights via RL training)
>
> Qwen 2.5-7B-Instruct checks all boxes. It's from Alibaba Cloud, fully open-source,
> instruction-tuned (follows directions well), and generates JSON reliably.
> Larger models (13B, 70B) perform better but don't fit on a single H100 80GB GPU
> with our training setup (model + optimizer + gradients + vLLM inference engine).

## How Generation Works

When the model generates text, it does so **one token at a time**:

```
Step 1: Input: "Design a trial for"
        Model predicts next token probabilities:
        "lung" (0.15), "cancer" (0.12), "EGFR" (0.08), ...
        Sample: "lung"

Step 2: Input: "Design a trial for lung"
        Model predicts: "cancer" (0.45), "disease" (0.15), ...
        Sample: "cancer"

Step 3: Input: "Design a trial for lung cancer"
        Model predicts: "." (0.10), "with" (0.25), "patients" (0.15), ...
        Sample: "with"

... and so on until the model produces a stop token or hits max length.
```

### Temperature: Controlling Randomness

**Temperature** controls how random the model's choices are:

```
Temperature = 0.0: Always pick the most likely token (deterministic, boring)
Temperature = 0.7: Mostly pick likely tokens, sometimes surprise (balanced) ← We use this
Temperature = 1.5: Very random, creative, but often nonsensical
```

In our project (from train.py):
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,        # Generate up to 256 tokens
    num_return_sequences=8,     # Generate 8 different attempts
    do_sample=True,             # Use random sampling (not greedy)
    temperature=0.7,            # Balanced randomness
    pad_token_id=tokenizer.eos_token_id,
)
```

We generate **8 different sequences** (num_return_sequences=8) for each step. This is crucial for our training algorithm (GRPO) because it compares multiple attempts to determine which approach is best.

## Parameters and Model Size

When we say "7B parameters," we mean the model contains 7 billion numbers (weights). These are organized in layers:

```
Qwen 2.5-7B architecture (simplified):
  └── Embedding layer: converts tokens → vectors (4096 dimensions)
  └── 32 Transformer blocks, each containing:
      ├── Self-Attention (Q, K, V projections)
      │   ├── q_proj: 4096 × 4096 = 16.7M parameters
      │   ├── k_proj: 4096 × 1024 = 4.2M parameters
      │   └── v_proj: 4096 × 1024 = 4.2M parameters
      └── Feed-Forward Network (two large linear layers)
          ├── up_proj:   4096 × 11008 = 45M parameters
          └── down_proj: 11008 × 4096 = 45M parameters
  └── Output head: vectors → token probabilities
```

Total: ~7,000,000,000 parameters

**Memory required:**
- Each parameter is stored as a **BF16** (Brain Float 16) number = 2 bytes
- 7B × 2 bytes = **14 GB** just for the model weights
- Training needs 3-4× more (gradients + optimizer state) = **42-56 GB**
- This is why we need an H100 GPU (80 GB VRAM)

## BF16: Why Not Regular Numbers?

Regular Python floats are 64-bit (8 bytes). Full-precision neural network floats are 32-bit (4 bytes). BF16 is only 16-bit (2 bytes).

```
FP32:  seeeeeee emmmmmmmmmmmmmmmmmmmmmmm  (32 bits, 4 bytes)
BF16:  seeeeeee emmmmmm                    (16 bits, 2 bytes)
```

BF16 keeps the same range as FP32 (same number of exponent bits) but with less precision. For neural networks, this precision loss is almost unnoticeable, but the memory savings are enormous:

- FP32: 7B params × 4 bytes = 28 GB
- BF16: 7B params × 2 bytes = 14 GB ← Half the memory!

> **Design Decision Box: Why BF16 over FP16?**
>
> FP16 (regular half-precision) has fewer exponent bits, meaning very large or very small numbers can cause "overflow" or "underflow" errors. BF16 avoids this by keeping the same exponent range as FP32. Modern GPUs (A100, H100) have dedicated BF16 hardware.

## Instruction Tuning: Following Orders

The "Instruct" in "Qwen2.5-7B-Instruct" means the model has been fine-tuned to follow instructions:

```
Base model (Qwen2.5-7B):
  Input:  "Set the sample size to 200 for the cancer trial"
  Output: "for the cancer trial set the sample size to 200 patients in the..." 
  (just continues the text — doesn't follow the instruction)

Instruct model (Qwen2.5-7B-Instruct):
  Input:  "Set the sample size to 200 for the cancer trial"
  Output: '{"action_type": "set_sample_size", "parameters": {"sample_size": 200}, ...}'
  (follows the instruction and produces structured output)
```

We use the Instruct version because our agent needs to **follow instructions** (the prompt describes what format to use) rather than just continuing text.

---

## How Our Agent "Thinks"

Here's what actually happens when our agent takes one step:

1. **Build the prompt** — Combine the current observation into text:
```python
# From train.py
prompt = (
    f"You are designing a clinical trial.\n\n"
    f"Scenario: {obs.scenario_description}\n"
    f"Phase data: {json.dumps(obs.phase_data)}\n"
    f"Resources: {json.dumps(obs.resource_status)}\n"
    f"Available actions: {obs.available_actions}\n"
    f"Steps taken: {obs.steps_taken}/{obs.max_steps}\n"
    f"Hint: {obs.hint}\n\n"
    "Respond with a JSON object: "
    '{"action_type": "...", "parameters": {}, "justification": "...", "confidence": 0.8}'
)
```

2. **Tokenize** — Convert text to numbers the model understands

3. **Generate** — Model produces 8 different responses

4. **Parse** — Extract the JSON action from the generated text

5. **Fallback** — If the text isn't valid JSON, use a safe default action:
```python
# From train.py
def _build_action_from_text(text, step):
    try:
        data = json.loads(text)
        return TrialAction(...)
    except Exception:
        # Fallback: cycle through action types
        action_type = ACTION_CYCLE[step % len(ACTION_CYCLE)]
        return TrialAction(
            action_type=action_type,
            parameters={},
            justification="fallback: could not parse model output",
            confidence=0.5,
        )
```

This fallback is important early in training when the model often generates garbage. The cycle ensures the environment still gets valid actions, so training can continue.

---

## Chapter 3 Glossary

| Keyword | Definition |
|---------|-----------|
| **Language Model (LM)** | A neural network trained to predict/generate text |
| **LLM (Large Language Model)** | A language model with billions of parameters |
| **Token** | A piece of text (word, subword, or character) that the model processes |
| **Tokenizer** | The tool that converts text → tokens and back |
| **Transformer** | The neural network architecture used by all modern LLMs |
| **Self-Attention** | Mechanism where each token learns to focus on relevant other tokens |
| **Pre-Training** | Initial training on massive text data (done by model creators) |
| **Fine-Tuning** | Adapting a pre-trained model for a specific task (what we do) |
| **Instruct Tuning** | Fine-tuning a model to follow instructions |
| **BF16 (Brain Float 16)** | A 16-bit number format that halves memory usage |
| **Temperature** | Parameter controlling randomness in text generation (0=deterministic, 1=random) |
| **Generation** | The process of a model producing text token by token |
| **Qwen 2.5-7B** | The specific model we use (7 billion parameters, by Alibaba) |
| **Foundation Model** | A large pre-trained model before any task-specific fine-tuning |
| **Embedding** | Converting a token into a numerical vector (list of numbers) |
