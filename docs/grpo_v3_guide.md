# GRPO V3 Training Guide

## What Was Wrong (The Bug)

### How GRPO Learning Works

GRPO (Group Relative Policy Optimization) trains a language model by:

1. Giving it a **prompt** (e.g., "Here's a clinical trial state, pick an action")
2. Generating **8 different completions** (because `temperature=0.7` adds randomness)
3. **Scoring each completion** with a reward function (our HF Space environment)
4. Computing the **advantage**: `(reward - mean) / std`
5. Updating the model to make **high-reward completions more likely**

The critical insight: **step 4 only works if `std > 0`**. If all 8 completions get the same reward, std=0, advantage=0, gradient=0, and the model learns **nothing**. The reward curve stays flat.

### Why All 8 Completions Got the Same Reward

Three bugs combined to make every completion score identically:

#### Bug 1: Different Environment Seed Per Completion

The old code gave each completion a **different environment seed**:

```python
# OLD (V2) — BROKEN
seed = args.seed + 10000 + seed_counter[0]  # increments per completion
seed_counter[0] += 1
```

This means completion #1 sees env-state-A, completion #2 sees env-state-B, etc. Since the untrained model can't produce valid JSON yet, all completions fall back to the same default action. Different env states + same action = roughly similar reward anyway, but GRPO can't meaningfully compare them because the environments are different.

**What we need**: All 8 completions should see the **same environment state** so the only variable is what the model outputs.

#### Bug 2: Hardcoded Confidence Value

The old code sent a fixed confidence to the environment:

```python
# OLD — always 0.5 (or 0.8 in earlier versions)
env_step(action_type, parameters, confidence=0.5)
```

The judge applies an **overconfidence penalty** (-0.5 per violation) when `confidence >= 0.8` and the check fails. With a fixed confidence, this penalty is identical for all completions → zero variance.

#### Bug 3: Model's Confidence Output Was Ignored

Even when the model learned to output different confidence values like `{"confidence": 0.4}` vs `{"confidence": 0.9}`, the old code never parsed it. It always used the hardcoded value.

### Net Result

All 8 completions → identical reward → `std = 0` → `advantage = 0` → **flat line on the graph**.

---

## What V3 Fixes

### Fix 1: Same Seed for All Completions in a Group

```python
# NEW (V3) — all 8 completions get the SAME env state
prompt_idx = _call_counter[0] // NUM_GENERATIONS
group_seed = prompt_seeds[prompt_idx % len(prompt_seeds)]
# Every completion in this batch uses group_seed
```

Now: completion A picks "enroll_patients" → reward X, completion B picks "observe_safety_signal" → reward Y. GRPO sees real variance and can compute meaningful advantages.

### Fix 2: Low Default Confidence (0.3)

```python
# NEW — default confidence below the 0.8 overconfidence threshold
def env_step(action_type, parameters=None, justification="", confidence=0.3):
```

When the model can't produce valid JSON, the fallback confidence is 0.3 — safely below the 0.8 threshold. No constant -1.0 penalty flooding all completions.

### Fix 3: Parse Confidence from Model Output

```python
# NEW — extract confidence from model's JSON
conf = float(parsed.get("confidence", 0.3))
conf = min(max(conf, 0.0), 1.0)
# Then pass it to env_step:
result = env_step(action["action_type"], ..., confidence=action["confidence"])
```

As the model learns, different completions will output different confidence values → different overconfidence penalties → natural reward variance → stronger GRPO signal.

---

## How to Run (Step by Step)

### Prerequisites

- Google Colab account (free tier with T4 GPU works)
- The HF Space must be running at `https://roopalgn-openenv-clinical-trial.hf.space`

### Steps

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload the notebook**: File → Upload notebook → select `train_colab_v3.ipynb` from the repo

3. **Select GPU runtime**: Runtime → Change runtime type → T4 GPU → Save

4. **Run Cell 1 (Install)**:
   - This clears stale Unsloth cache and installs dependencies
   - **After this cell finishes, restart the runtime**: Runtime → Restart runtime
   - This is mandatory — the packages need a fresh Python process

5. **Run Cell 2 (Config)**: Sets up constants. No output besides a config confirmation.

6. **Run Cell 3 (Helpers)**: Defines `env_reset()`, `env_step()`, `parse_action()`, etc.

7. **Run Cell 4 (Connection Test + Dry Run)**:
   - This is the **key validation cell**
   - It tests 5 different actions on the SAME seed and prints rewards
   - **What to look for**: different actions should give different rewards
   - ✓ Good: `set_primary_endpoint → 1.5`, `enroll_patients → 0.8`, `run_dose_escalation → -1.2`
   - ✗ Bad: all actions give the same reward (means server-side bug)
   - It also tests confidence variance (same action, different confidence values)

8. **Run Cell 5 (Generate Prompts)**: Creates 40 diverse prompts by resetting the env with different seeds. Should print "Generated 40 prompts from 40 unique seeds".

9. **Run Cell 6 (Load Model)**: Downloads and loads the Qwen2.5-1.5B model with LoRA. Takes ~2 min on Colab.

10. **Run Cell 7 (Train)**: The actual GRPO training. This is the long cell (~30-60 min on T4 for 40 steps).
    - **Watch the logs**: you should see `reward` and `reward_std` values
    - **Key signal**: `reward_std > 0` means the fix is working
    - If you see `reward_std = 0.000` for most steps, something is still broken

11. **Run Cell 8 (Plot + Save)**: Generates the reward curve plot and saves artifacts.
    - **What you want to see**: an upward-sloping trend line (slope > 0)
    - The plot shows reward ± std on the left, and reward_std over time on the right
    - Files saved to `outputs/grpo_v3/`

12. **Run Cell 9 (Eval)**: Runs the trained model on 5 fresh seeds (greedy decoding). Shows what actions the model picks and the rewards it gets.

13. **Run Cell 10 (Save Checkpoint)**: Saves the LoRA weights.

---

## How to Verify the Fix Worked

### Signal 1: Dry Run (Cell 4)

The dry run should show reward variance across actions:

```
set_primary_endpoint    → reward = 1.5000
enroll_patients         → reward = 0.8000
run_dose_escalation     → reward = -1.2000
```

If all rewards are identical here, the server-side reward function may need fixes.

### Signal 2: reward_std > 0 During Training (Cell 7)

In the training logs, look for:

```
Step 1: reward=0.45, reward_std=0.82   ← GOOD (std > 0)
Step 2: reward=0.51, reward_std=0.71   ← GOOD
```

vs the old broken behavior:

```
Step 1: reward=0.20, reward_std=0.00   ← BAD (std = 0, no learning)
```

### Signal 3: Positive Slope on Reward Curve (Cell 8)

The plot title shows `slope=X.XXXX`. A positive slope means the model is improving. Even a small positive slope (e.g., 0.01) is a huge improvement over the previous flat 0.002.

### Signal 4: Eval Quality (Cell 9)

The eval cell shows what actions the trained model picks. After training, you'd hope to see:
- Actions that match the current phase (e.g., `set_primary_endpoint` early, not `submit_to_fda_review`)
- Valid JSON output (not gibberish that needs fallback)
- Reasonable confidence values (not always 0.8 or 0.3)

---

## What's Left (Next Steps)

### Immediate (Before Spending HF Credits)

| # | Task | Why |
|---|------|-----|
| 1 | **Run V3 on Colab** | Verify reward_std > 0 and positive slope on free T4 |
| 2 | **Check the reward curve** | Must see upward trend before paying for GPU time |
| 3 | **Sync to HF Space** | Need HF token: `$env:HF_TOKEN = "hf_..."` then `git push hf main` |

### If Colab Shows Positive Slope → Scale Up

| # | Task | Details |
|---|------|---------|
| 4 | Change `MODEL_SIZE = "3b"` or `"7b"` | Bigger models learn faster, but need more VRAM |
| 5 | Increase `NUM_PROMPTS = 100+` | More training steps = more learning |
| 6 | Run on HF Jobs with paid GPU | Use the A10G/A100 credits for serious training |
| 7 | Save and push the model checkpoint to HF Hub | For the hackathon judges to evaluate |

### If Colab Still Shows Flat Reward

| # | Task | Details |
|---|------|---------|
| A | Check Cell 4 dry run output | If rewards identical across actions → server-side issue |
| B | Check if model produces valid JSON at all | If 100% fallback → try lowering `temperature` to 0.6 |
| C | Add more action types to the prompt | Model might need clearer instruction |

### HF Space Sync (Currently Blocked)

The push to `hf` remote failed because `HF_TOKEN` isn't set. To fix:

```powershell
# Option 1: Set token in environment
$env:HF_TOKEN = "hf_your_token_here"
git remote set-url hf "https://Roopalgn:$env:HF_TOKEN@huggingface.co/spaces/Roopalgn/openenv-clinical-trial"
git push hf main

# Option 2: Use huggingface-cli
pip install huggingface_hub
huggingface-cli login
git push hf main
```

Get your token from: https://huggingface.co/settings/tokens (needs `write` permission).

---

## File Structure Reference

```
train_colab_v3.ipynb  ← NEW: the fixed training notebook (run this on Colab)
train_colab_v2.py     ← OLD: Python script version (has the seed bug)
train_colab.ipynb     ← OLD: V1 notebook (has multiple bugs)
train.py              ← Server-side training script (for HF Jobs, not Colab)
server/
  reward/
    reward_computer.py  ← Reward function (already fixed: V2 magnitudes)
  phase_detector.py     ← Phase ordering reward (already fixed: V2 bonuses)
  judge.py              ← Overconfidence penalty logic (0.8 threshold)
  episode_manager.py    ← Environment orchestrator
outputs/
  grpo_v3/              ← Created by training (reward_log.csv, plot, summary)
```
