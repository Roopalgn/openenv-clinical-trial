# Chapter 12: Training Infrastructure — LoRA, vLLM, and GPUs

## The Problem: Models Are Huge

Qwen 2.5-7B has 7 billion parameters. Training all of them requires:
- **Model weights:** 14 GB (7B × 2 bytes in BF16)
- **Gradients:** 14 GB (one gradient per parameter)
- **Optimizer state:** 28 GB (Adam optimizer stores 2 values per parameter)
- **Activations:** 10-20 GB (intermediate computation values)
- **Total:** 66-76 GB

An H100 GPU has 80 GB. With full training, we'd barely fit, with no room for generating text during training. That's where **LoRA** comes in.

## LoRA: Training 0.1% of the Model

### The Core Idea

LoRA (**Low-Rank Adaptation**) is a technique that says: "Instead of training ALL 7 billion parameters, let's add a tiny number of new parameters and only train those."

**Analogy:** Imagine you have a 100-page essay (the pre-trained model). Instead of rewriting the entire essay (full fine-tuning), you add sticky notes to a few key pages (LoRA adapters). The original essay stays unchanged — the sticky notes add extra information.

### How LoRA Works (Simplified)

In a Transformer, the key layers are large matrices (tables of numbers). For example, the attention layer has a "query projection" matrix `W` of size 4096 × 4096 = 16.7 million parameters.

Full fine-tuning: update all 16.7M parameters in `W`.

LoRA: instead of modifying `W`, add two small matrices `A` and `B`:

```
Original: output = W × input                     (16.7M parameters)
LoRA:     output = W × input + B × A × input     (+2 × 4096 × 32 = 262K parameters)
                   ↑ frozen    ↑ trainable
```

Where:
- `W` is frozen (never changes) — 16.7M parameters, but no gradients needed
- `A` is a 32 × 4096 matrix (131K trainable parameters)
- `B` is a 4096 × 32 matrix (131K trainable parameters)
- Total trainable: 262K (vs 16.7M) = 1.6% of the original layer

The number **32** is the "rank" (the `r` parameter). Lower rank = fewer trainable parameters but less expressiveness.

### Our LoRA Configuration

```python
# From train.py
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Causal language model task
    r=32,                           # Rank = 32 (for 7B model)
    lora_alpha=64,                  # Scaling factor = rank × 2
    lora_dropout=0.05,              # 5% dropout for regularization
    bias="none",                    # Don't train bias parameters
    target_modules=["q_proj", "v_proj"],  # Only adapt attention Q and V projections
)
```

Let me explain each parameter:

**`r=32` (rank):** How many new parameters per layer. Higher rank = more capacity but more memory.

```python
# From train.py — model size presets
MODEL_SIZE_PRESETS = {
    "1.5b": {"lora_r": 8,  ...},   # Smaller model → smaller LoRA
    "3b":   {"lora_r": 16, ...},   # Medium model → medium LoRA  
    "7b":   {"lora_r": 32, ...},   # Large model → larger LoRA
}
```

**`lora_alpha=64` (scaling):** Controls how much the LoRA adapters influence the output. Rule of thumb: `alpha = 2 × r`.

**`target_modules=["q_proj", "v_proj"]`:** We only adapt the Query and Value projections in the attention mechanism. Why not all layers?

> **Design Decision Box: Why Only q_proj and v_proj?**
>
> Research (Hu et al., 2021, the LoRA paper) found that adapting Q and V projections captures most of the fine-tuning benefit while using the least parameters. Adding K projections and feed-forward layers helps marginally but doubles memory usage.
>
> For our use case (learning trial design behavior, not new language), Q+V is sufficient and keeps memory under control on a single H100.

**`lora_dropout=0.05`:** Randomly zeroes out 5% of LoRA activations during training. This prevents overfitting (memorizing specific scenarios instead of learning general strategies).

### Memory Savings with LoRA

```
Full fine-tuning:
  Model weights:    14 GB
  Gradients:        14 GB
  Optimizer state:  28 GB
  Total: ~56 GB

LoRA fine-tuning:
  Model weights:    14 GB (frozen, no gradients)
  LoRA weights:     ~50 MB (tiny!)
  LoRA gradients:   ~50 MB
  LoRA optimizer:   ~100 MB
  Total: ~14.2 GB  ← 4× less memory!
```

The saved memory lets us run vLLM inference AND training on the same GPU.

## vLLM: Fast Text Generation

### Why We Need a Separate Inference Engine

During GRPO training, for each step we need to generate **8 different responses**. Using the raw model for generation is slow because:
- It generates one token at a time
- Each token requires a full model forward pass
- 8 responses × 256 tokens × full forward pass = slow

**vLLM** is a specialized inference engine that makes generation **5-10×** faster through:

1. **Continuous Batching:** Processes multiple generation requests simultaneously
2. **PagedAttention:** Efficiently manages GPU memory for attention caches
3. **Prefix Caching:** If multiple prompts share the same prefix, compute it once

### Colocate Mode

Our training uses vLLM in **colocate mode** — meaning the training model and the inference engine share the same GPU:

```python
# From train.py
grpo_config = GRPOConfig(
    vllm_mode="colocate",  # Share GPU between training and inference
    ...
)
```

```
Without colocate: Two GPUs needed
  GPU 0: Training (weights + gradients)
  GPU 1: vLLM inference (separate copy of weights)

With colocate: One GPU
  GPU 0: Training + vLLM inference (shared weights)
  Saves: 14 GB of GPU memory (one fewer model copy)
```

> **Design Decision Box: Why vLLM Colocate?**
>
> We only have ONE H100 GPU. Without colocate mode, we'd need two GPUs — one for training, one for inference. Colocate mode shares the model weights between training and inference, fitting everything on a single H100 80GB.
>
> The tradeoff: training and inference can't run simultaneously. They take turns. But since our episodes are sequential anyway (step by step), this doesn't slow us down.

## GPU Basics: What's Actually Happening

### What is a GPU?

A **GPU (Graphics Processing Unit)** was originally designed for drawing video game graphics. But researchers discovered that the same hardware (thousands of parallel processors) is perfect for neural network math.

```
CPU: 8-16 powerful cores     → Good at sequential tasks (one after another)
GPU: 10,000+ simple cores    → Good at parallel tasks (many at once)

Neural network: multiply millions of numbers simultaneously
→ Perfect for GPU!
```

### The NVIDIA H100

Our training runs on an **NVIDIA H100 80GB SXM5** — currently one of the most powerful GPUs available:

```
H100 Specs:
  GPU Memory: 80 GB HBM3          ← Where model + data fits
  Memory Bandwidth: 3.35 TB/s     ← How fast data moves to/from memory
  BF16 Compute: 989 TFLOPS        ← Trillions of BF16 operations per second
  TDP: 700W                       ← Power consumption (a LOT)
  Cost: ~$30,000 per GPU          ← Why cloud rental is popular
```

### Memory Layout During Training

Here's what fits in the H100's 80 GB during our training:

```
┌──────────────────────────────────────────────┐
│              H100 80GB VRAM                   │
│                                              │
│  Model weights (BF16, frozen):     14.0 GB   │
│  LoRA adapter weights:              0.1 GB   │
│  LoRA gradients:                    0.1 GB   │
│  LoRA optimizer state:              0.2 GB   │
│  vLLM KV cache (inference):       10.0 GB   │
│  Activations (training):          15.0 GB   │
│  CUDA overhead + fragmentation:     5.0 GB   │
│                                              │
│  Total used: ~44.4 GB                        │
│  Free: ~35.6 GB (headroom for spikes)        │
└──────────────────────────────────────────────┘
```

## Gradient Accumulation: Simulating Bigger Batches

We can only fit **1 sample** in GPU memory at a time (per_device_train_batch_size=1). But training with a batch of 1 is noisy — the gradient from one example might point in the wrong direction.

**Solution:** Gradient accumulation. Process 8 samples one at a time, accumulate the gradients, then do one weight update:

```
Step 1: Process sample 1, compute gradient, SAVE (don't update yet)
Step 2: Process sample 2, compute gradient, ADD to saved gradient
Step 3: Process sample 3, compute gradient, ADD to saved gradient
...
Step 8: Process sample 8, compute gradient, ADD to saved gradient
→ Now update weights using the averaged gradient of all 8 samples
```

```python
grpo_config = GRPOConfig(
    per_device_train_batch_size=1,       # 1 sample fits in memory
    gradient_accumulation_steps=8,        # Accumulate 8 before updating
    # Effective batch size = 1 × 8 = 8
)
```

This gives the same result as processing 8 samples at once (batch_size=8) but with only the memory cost of 1 sample.

## The Model Size Presets

Our training code supports different model sizes with appropriate settings:

```python
MODEL_SIZE_PRESETS = {
    "1.5b": {
        "lora_r": 8,        # Smaller model → smaller LoRA
        "batch": 1,
        "seq_len": 2048,    # Shorter context window
        "grad_accum": 4,    # Less accumulation needed
    },
    "3b": {
        "lora_r": 16,
        "batch": 1,
        "seq_len": 3072,
        "grad_accum": 4,
    },
    "7b": {
        "lora_r": 32,       # Larger model → larger LoRA
        "batch": 1,
        "seq_len": 4096,    # Full context window
        "grad_accum": 8,    # More accumulation for stability
    },
}
```

## The Dry-Run: Testing Without a GPU

Before spending expensive GPU hours on training, we have a **dry-run** mode:

```python
def _dry_run(args):
    """Run 2 episodes with a random policy — no GPU needed."""
    env = Environment()
    
    for ep in range(2):
        obs = env.reset(seed=args.seed + ep)
        for step_idx in range(args.max_steps):
            # Use fallback (cycles through actions, no model needed)
            action = _build_action_from_text("", step_idx)
            next_obs, reward_dict, done, _ = env.step_full(action)
            if done:
                break
```

This verifies:
1. The environment works (reset/step)
2. Rewards are computed correctly
3. Logging works
4. The CSV and plot pipeline works

All without loading a multi-GB model or needing a GPU. This saves hours of debugging.

---

## Chapter 12 Glossary

| Keyword | Definition |
|---------|-----------|
| **LoRA (Low-Rank Adaptation)** | Technique to fine-tune only a small number of added parameters |
| **Rank (r)** | Number of dimensions in LoRA's low-rank matrices |
| **lora_alpha** | Scaling factor for LoRA's contribution to the output |
| **lora_dropout** | Random zeroing of LoRA activations (prevents overfitting) |
| **target_modules** | Which model layers get LoRA adapters (q_proj, v_proj) |
| **Frozen weights** | Model parameters that don't change during training |
| **vLLM** | High-performance inference engine for language models |
| **Colocate mode** | Sharing GPU between training and inference (saves memory) |
| **GPU (Graphics Processing Unit)** | Parallel processor used for neural network computation |
| **VRAM** | GPU memory (Video RAM) |
| **H100** | NVIDIA's high-end data center GPU (80 GB) |
| **BF16 (Brain Float 16)** | 16-bit number format that halves memory usage |
| **Gradient Accumulation** | Processing multiple samples before updating weights |
| **Effective Batch Size** | batch_size × gradient_accumulation_steps |
| **KV Cache** | Cached key/value tensors for faster text generation |
| **Dry Run** | Testing the full pipeline without training (no GPU needed) |
| **PEFT (Parameter-Efficient Fine-Tuning)** | Family of techniques including LoRA |
