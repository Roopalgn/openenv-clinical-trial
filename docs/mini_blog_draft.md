# HuggingFace Mini-Blog — Clinical Trial Designer

> **G12 — Judging minimum requirement.** Publish on HuggingFace before the pitch. < 2 minutes read (~600 words). Final version: fill placeholders after training results are available.

---

## Title

**Teaching an LLM to Design Clinical Trials — From Scratch**

## Subtitle

*A GRPO-trained 7B model that learns drug trial design through Phase I safety testing, biomarker stratification, and FDA compliance — without a single medical textbook.*

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

## Section 4: Results ([N] episodes on H100)

**Setup:** Qwen2.5-7B + LoRA (rank 16), GRPO with 8 rollouts, [N] training steps.

### Baseline Comparison

| Metric | Random | Scripted | **Trained** |
|--------|--------|----------|---------|
| Success rate | ~5% | ~40% | **[__]%** |
| Avg reward | -1.5 | +2.8 | **+[__]** |
| Subgroup found | 2% | 0% | **[__]%** |

### Reward Curve

![Reward Curve](results/reward_curve.png)

### Before → After

**Episode 1:** Skips Phase I, hits prerequisites, times out. Reward: -2.5.

**Episode [N]:** Runs dose escalation → discovers EGFR+ subgroup → enriches trial → p = 0.003, power 0.88. Reward: +11.2.

---

## Section 5: What the Agent Learned (`[FILL ONSITE]`)

`[FILL ONSITE: 50 words describing what the agent actually learned. Example below — replace with real observations:]`

The agent independently discovered **biomarker stratification** — enriching the trial for responsive patients rather than powering for everyone. This strategy, which revolutionized oncology in the 2010s, emerged from reward signal alone. It also learned FDA workflow ordering, adaptive sample sizing, and futility stopping.

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
- [ ] Insert 3 images: architecture diagram, reward curve, capability radar
- [ ] Verify all links work (HF Spaces, HF Hub, GitHub)
- [ ] Publish as HuggingFace blog post (Settings → New Blog Post)
- [ ] Confirm < 2 minutes read time (target 550–650 words)
- [ ] Share link in hackathon Discord
