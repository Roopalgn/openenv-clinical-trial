# Pre-Onsite Roadmap — OpenEnv Clinical Trial Designer

> **Status:** All pre-onsite work complete as of Apr 24. Ready for onsite training Apr 25–26.

---

## What We Built (Push 1–7, all complete)

### Environment
- OpenEnv v0.2.3 FastAPI environment for clinical trial design (Theme #3.1: Professional Tasks)
- 19-action space across 5 clinical phases (design, Phase I, Phase II, regulatory, analysis)
- `TrialLatentState` with hidden ground truth (true effect size, responder population, dose-response, safety profile)
- 4 clinical scenarios: solid tumor (EGFR+ subgroup), autoimmune (U-shaped dose), CNS depression (high placebo), rare disease (tiny n)
- 5-tier adaptive curriculum with per-scenario mastery tracking and weak-spot targeting
- 15-component decomposed reward (8 per-step + 7 terminal), all math/rule-verified
- Multi-layer judge: programmatic ground-truth + FDA rule engine + optional LLM judge
- Domain randomization via seeded `NoiseModel` (budget ±30%, time ±20%, dropout ±15%, placebo ±20%)
- Phase-order scoring: +0.2 for correct workflow, -0.3 × N for skipping phases
- Potential-based reward shaping: γ·(φ(s') − φ(s))
- 249/249 tests pass, ruff lint clean

### Training Pipeline
- `train.py` with GRPO via TRL 0.29.0 + vLLM colocate + LoRA
- `--model-size` presets: 1.5b (rank 8), 3b (rank 16), 7b (rank 32)
- `--dry-run` mode for pipeline validation without GPU
- `eval_compare.py` for base vs trained comparison
- `plot_rewards.py` for reward curve visualization
- Random policy baseline established: −61.6 reward

### Deployment
- HF Space live: `https://roopalgn-openenv-clinical-trial.hf.space`
- Docker image builds and runs (PORT 7860)
- `/ping`, `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/transcripts` all responding
- Colab + Kaggle notebooks validated with dry-run

### Documentation
- ARCHITECTURE.md with system diagram and component docs
- Statistical grounding: rpact validation tables, Berry 2010, Wassmer & Brannath 2016, Narvekar 2020
- Mini-blog draft with `[FILL ONSITE]` placeholders
- Pitch notes and training log templates ready
- Onsite roadmap with step-by-step execution plan

---

## Known Code Gaps (to fix onsite, Day 1 Hours 0–2)

| # | Gap | Owner | Effort |
|---|-----|-------|--------|
| G23 | Phase system stays `"literature_review"` forever — `transition_engine.py` never updates `latent.episode_phase` | Suyash | ~1hr |
| G24 | Latent biology hardcoded (`true_responder_criteria=[]`, `true_dose_response={}`) instead of populated from scenarios | Roopal | ~1hr |
| G25 | Adversarial designer receives `true_effect_size=None` — counters never increment | Roopal | ~30min |
| G26 | `trainer.train()` never called — `GRPOTrainer` is constructed but unused (`noqa: F841`) | Suyash | ~1hr |
| G27 | Judge checks power ≥ 0.80 on every step even before patients enrolled | — | ~1hr |
| G28 | Single session only (`SUPPORTS_CONCURRENT_SESSIONS=False`) | — | P2, skip |

**Gate:** All P0 fixes (G23, G24, G25, G26) must pass `pytest tests/ -q` before training starts.

---

## Submission Checklist

- [x] GitHub repo (public, clean)
- [x] HF Space URL (deployed, `/ping` responds)
- [ ] Mini-blog on HuggingFace — **ONSITE** (draft ready)
- [x] Training script using HF TRL in Colab (runnable)
- [ ] Reward/loss plots committed as `.png/.jpg` — **ONSITE**
- [ ] README links all materials with plots inline — **ONSITE**
- [x] Valid `openenv.yaml`
- [x] Client/server separation
- [x] Gym-style API (reset, step, state)

> **Deadline:** 5:00 PM April 26. Google Form on campus.
> **Form fields:** HF Space URL, Colab Notebook link, Code repo link, YouTube/HF blog URL.

---

## Team

- **Roopal Guha Neogi** — Environment design, reward engineering, documentation, deliverables
- **Suyash Kumar** — Environment implementation, training pipeline, evaluation, deployment

- [x] GitHub repo (public, clean)
- [x] HF Space URL (deployed, `/ping` responds)
- [ ] Mini-blog on HuggingFace — **ONSITE** (draft ready)
- [x] Training script using HF TRL in Colab (runnable)
- [ ] Reward/loss plots committed as `.png/.jpg` — **ONSITE**
- [ ] README links all materials with plots inline — **ONSITE**
- [x] Valid `openenv.yaml`
- [x] Client/server separation
- [x] Gym-style API (reset, step, state)

> **Deadline:** 5:00 PM April 26. Google Form on campus.
> **Form fields:** HF Space URL, Colab Notebook link, Code repo link, YouTube/HF blog URL.

---

## Team

- **Roopal Guha Neogi** — Environment design, reward engineering, documentation, deliverables
