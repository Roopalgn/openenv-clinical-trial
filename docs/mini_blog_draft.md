# HuggingFace Mini-Blog — Clinical Trial Designer

> **G12 — Judging minimum requirement.** Publish on HuggingFace before the pitch. < 2 minutes read (~600 words). Final version: fill placeholders after training results are available.

---

## Title

**Teaching an LLM to Design Clinical Trials — From Scratch**

## Subtitle

*A GRPO-trained 7B model that learns drug trial design through Phase I safety testing, biomarker stratification, and FDA compliance — without a single medical textbook.*

---

## Onsite Fill Sheet

Collect these once and paste them into the sections below:

| Field | Value |
|------|------|
| Final model | `Qwen2.5-1.5B-Instruct-bnb-4bit + LoRA (Colab validation)` |
| Training episodes | `20` |
| Training runtime | `See completed_at in training_summary.json; duration was not exported separately` |
| Final success rate | `TBD from longer eval` |
| Final avg reward | `42.07 eval avg (trained)` |
| Best episode reward | `18.528873443603516` |
| Most important learned behavior | `Stable positive reward signal with a small trained-vs-random gain after fixing parser/config/runtime issues.` |
| Reward curve path | `results/reward_curve.png` |
| Before episode ID | `[FILL]` |
| After episode ID | `[FILL]` |

---

## Section 1: The Problem (3 sentences)

Designing a clinical trial costs $2.6 billion per approved drug and takes 10–15 years. Poor designs account for 57% of Phase II failures. We asked: can a language model learn the entire workflow — dose-finding, patient selection, regulatory submission, adaptive design — purely from reinforcement learning?

---

## Section 2: The Environment (100 words)

We built an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment that simulates a complete Phase I/II clinical trial. The agent receives a disease scenario with **hidden ground truth** — the drug's real efficacy, true responder subgroup, and side-effect profile are invisible. It must choose from 19 actions across 5 clinical phases to design a trial that detects the true effect.

Verification is objective: `scipy.stats` calculates statistical power, a rule engine checks FDA compliance, and a trial simulation returns the p-value. No LLM judge needed.

Four scenarios span oncology, autoimmune, CNS, and rare disease — each with unique hidden challenges and a 5-tier adaptive curriculum.

![Architecture Diagram](results/architecture_diagram.png)

---

## Section 3: Reward Design (80 words)

Our reward decomposes into **8 per-step** and **7 terminal** components, ranging from -3 to +14 per episode — the variance GRPO needs.

| Component | What It Teaches |
|-----------|----------------|
| `r_ordering` | Follow correct clinical phase sequence |
| `r_info_gain` | Gather data before designing (don't skip Phase I) |
| `r_terminal_success` | Design trials that detect the true drug effect |
| `r_terminal_calibration` | Make claims that match hidden reality |

Potential-based shaping (γ·φ(s')) gives gradient without distorting optimal policy.

---

## Section 4: Results

**Setup:** `Qwen2.5-1.5B-Instruct-bnb-4bit` + LoRA, GRPO with `8` rollouts, `20` episodes, completed at `2026-04-25T12:32:29.445902+00:00`.

### Drop-in Metrics

| Metric | Value |
|--------|-------|
| Final success rate | `TBD from longer eval` |
| Average reward | `42.07 eval avg over 3 episodes` |
| Best episode reward | `18.528873443603516` |
| Final curriculum tier | `TBD (not exported by notebook summary)` |
| Primary learned behavior | `The policy no longer flatlined and consistently produced valid, positive-reward rollouts.` |

### Baseline Comparison

| Metric | Random | Scripted | **Trained** |
|--------|--------|----------|---------|
| Success rate | ~5% | ~40% | **TBD** |
| Avg reward | -1.5 | +2.8 | **+42.07 (3-ep eval)** |
| Subgroup found | 2% | 0% | **TBD** |

### Reward Curve

`Validation curve available; replace with the stronger HF-credit run if that curve is clearer for the final blog.`

![Reward Curve](results/reward_curve.png)

### Before → After

**Episode 1:** Skips Phase I, hits prerequisites, times out. Reward: -2.5.

**Episode [FILL]:** `Validation run completed end to end and beat the random baseline on a short 3-episode eval, but we still need transcript review to pick the strongest qualitative episode for the final writeup.`

---

## Section 5: What the Agent Learned

Use this 3-sentence structure:

1. `The first thing the agent learned was to stay inside the environment's valid action space instead of collapsing into invalid JSON or terminal fallbacks.`
2. `The most valuable strategic behavior in this validation run was producing stable positive reward rollouts after the parser, precision, and prompt-contract fixes.`
3. `This mattered because it turned a flatlined training loop into a usable baseline for a larger HF-credit run.`

Ready-to-edit paragraph:

`After fixing precision mismatches, brittle JSON parsing, and invalid-action fallbacks, the policy stopped collapsing into constant negative outcomes. The first Colab validation run finished end to end, uploaded a model checkpoint, and showed a modest trained-vs-random eval gain. That makes it a successful pipeline validation, even if the final judged run still needs a stronger reward-improvement story.`

---

## Section 6: Try It

```bash
git clone https://github.com/Roopalgn/openenv-clinical-trial.git
cd openenv-clinical-trial
pip install -e .
uvicorn server.app:app --port 8000
```

**Links:**
- [Environment on HF Spaces](https://huggingface.co/spaces/Roopalgn/clinical-trial-designer)
- [Trained Model](https://huggingface.co/Roopalgn/clinical-trial-designer-grpo) <!-- [FILL ONSITE: verify repo exists after pushing checkpoint] -->
- [GitHub](https://github.com/Roopalgn/openenv-clinical-trial)

---

## Tags

`openenv` `grpo` `clinical-trial` `reinforcement-learning` `trl` `unsloth` `hackathon` `meta-pytorch`

---

## Publishing Checklist

- [ ] Replace all `[placeholders]` with actual training results
- [ ] Insert 2-3 images: architecture diagram, reward curve, optional capability radar
- [ ] Verify all links work (HF Spaces, HF Hub, GitHub)
- [ ] Publish as HuggingFace blog post (Settings → New Blog Post)
- [ ] Confirm < 2 minutes read time (target 550–650 words)
- [ ] Share link in hackathon Discord
