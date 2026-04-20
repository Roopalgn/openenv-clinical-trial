# HuggingFace Mini-Blog Draft — Clinical Trial Designer

> **G12 — Judging minimum requirement.** Must be published on HuggingFace before the pitch. < 2 minutes read. This is the draft outline; final version updated after training results are available.

---

## Title

**Teaching an LLM to Design Clinical Trials — From Scratch**

## Subtitle

*A GRPO-trained agent that learns drug trial design through Phase I safety testing, FDA compliance, and biomarker stratification — without a single medical textbook.*

---

## Section 1: The Problem (15 seconds read)

> Designing a clinical trial takes teams of experts months of work. Poor trial design wastes $2.6B per approved drug (DiMasi et al., 2016) and leads to a 90% failure rate across all phases (Wong et al., 2019).

**Question:** Can a language model learn the nuances of clinical trial design — dose-finding, patient selection, regulatory compliance, adaptive strategies — purely through reinforcement learning?

We built an OpenEnv environment to find out.

---

## Section 2: The Environment (30 seconds read)

Our environment simulates a complete Phase I/II clinical trial:

**The agent receives:** A disease scenario (e.g., NSCLC with a novel EGFR inhibitor) with a hidden ground truth — the drug's true efficacy, true responder subgroup, and true side-effect profile. The agent never sees these directly.

**The agent acts:** 19 actions across 5 clinical phases — dose escalation, safety monitoring, trial design (endpoint, sample size, inclusion criteria), FDA submission, interim analysis, and final synthesis.

**The agent is judged:** Programmatically against ground truth using `scipy.stats` power calculations, not LLM judges. Did the trial detect the true effect? Did the agent identify the right patient population? Was the sample size adequate? Did the protocol pass FDA constraints?

**4 scenarios** span easy (large-effect oncology) to very hard (rare pediatric disease with 50 patients). A 5-tier curriculum ramps difficulty automatically.

![Architecture Diagram]  
*← Insert `results/architecture_diagram.png` here*

---

## Section 3: Reward Design (30 seconds read)

Our reward has **8 per-step components** and **7 terminal components**, totaling a range of **-3 to +14** per episode — the variance GRPO needs.

| Component | What It Teaches |
|-----------|----------------|
| `r_validity` | Follow FDA rules |
| `r_ordering` | Follow correct clinical phase order |
| `r_info_gain` | Run experiments to gather data before designing |
| `r_efficiency` | Don't waste budget or time |
| `r_novelty` | Use diverse actions, don't repeat |
| `r_terminal_success` | Design a trial that detects the true drug effect |
| `r_terminal_calibration` | Make claims that match hidden reality |

**Potential-based shaping** (γ·(φ(s')−φ(s))) gives gradient toward progress without distorting the optimal policy.

---

## Section 4: Training Results (30 seconds read)

> *Fill in after onsite training April 25–26.*

**Setup:** Qwen2.5-7B with LoRA (rank 16), GRPO with 8 rollouts, trained for [N] episodes on HuggingFace H100.

### Reward Curve

![Reward Curve]  
*← Insert `results/reward_curve.png` here*

### Baseline Comparison

| Metric | Random | Scripted | Trained |
|--------|--------|----------|---------|
| Warmup success | __% | __% | **__%** |
| Intermediate success | __% | __% | **__%** |
| Expert success | __% | __% | **__%** |
| Avg reward | __ | __ | **__** |

### Capability Radar

![Capability Radar]  
*← Insert `results/capability_radar.png` here*

---

## Section 5: What the Agent Learned (15 seconds read)

> *Fill in after training with specific behavioral observations.*

**Before training (Episode 1):**
- Skips dose escalation, guesses sample size, FDA rejects protocol
- Reward: -2.5

**After training (Episode [N]):**
- [Describe specific learned behavior: runs Phase I systematically, discovers subgroup, adapts sample size]
- Reward: +[X.X]

**Key insight:** The agent independently discovered [biomarker stratification / dose optimization / futility stopping / adaptive design] — strategies that took clinical researchers decades to formalize.

---

## Section 6: Try It Yourself

```bash
# Run our environment
docker pull ghcr.io/roopalgn/clinical-trial-designer:latest
docker run -p 8000:8000 ghcr.io/roopalgn/clinical-trial-designer:latest

# Train your own agent
pip install trl unsloth
python train.py --max-steps 100
```

**Links:**
- [Environment on HF Spaces](https://huggingface.co/spaces/Roopalgn/clinical-trial-designer)
- [Trained Model on HF Hub](https://huggingface.co/Roopalgn/clinical-trial-designer-grpo)
- [GitHub Repository](https://github.com/Roopalgn/openenv-clinical-trial)

---

## Tags

`openenv` `grpo` `clinical-trial` `reinforcement-learning` `trl` `hackathon` `meta-pytorch`

---

## Publishing Checklist

- [ ] Replace all `[placeholders]` with actual training results
- [ ] Insert 3 images: architecture diagram, reward curve, capability radar
- [ ] Verify all links work (HF Spaces, HF Hub, GitHub)
- [ ] Publish as HuggingFace blog post (not just README)
- [ ] Confirm < 2 minutes read time (aim for 500–700 words final)
- [ ] Share link in hackathon Discord
