# Pre-Onsite Roadmap — OpenEnv Clinical Trial Designer

> **Status:** Historical pre-onsite snapshot (Apr 24). Items below are kept for context; implementation has since moved to the current 7-phase / 8-component reward stack.

---

## What We Built (Push 1–7, all complete)

### Environment
- OpenEnv v0.2.3 FastAPI environment for clinical trial design (Theme #3.1: Professional Tasks)
- 19-action space across 7 workflow phases (`literature_review → hypothesis → design → enrollment → monitoring → analysis → submission`)
- `TrialLatentState` with hidden ground truth (true effect size, responder population, dose-response, safety profile)
- 4 clinical scenarios: solid tumor (EGFR+ subgroup), autoimmune (U-shaped dose), CNS depression (high placebo), rare disease (tiny n)
- 5-tier adaptive curriculum with per-scenario mastery tracking and weak-spot targeting
- 8-component decomposed reward (6 per-step + 2 terminal), all math/rule-verified
- Multi-layer judge: programmatic ground-truth + FDA rule engine + optional LLM judge
- Domain randomization via seeded `NoiseModel` (budget ±30%, time ±20%, dropout ±15%, placebo ±20%)
- Phase-order scoring: +0.1 for correct workflow, -0.3 × N for skipping phases
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

## Known Code Gaps (Historical Onsite Plan)

| # | Gap | Owner | Effort |
|---|-----|-------|--------|
| G23 | Phase system stayed `"literature_review"` forever — `transition_engine.py` did not update `latent.episode_phase` | Suyash | ~1hr |
| G24 | Latent biology was hardcoded (`true_responder_criteria=[]`, `true_dose_response={}`) instead of populated from scenarios | Roopal | ~1hr |
| G25 | Adversarial designer received `true_effect_size=None` — counters never incremented | Roopal | ~30min |
| G26 | `trainer.train()` was never called — `GRPOTrainer` was constructed but unused (`noqa: F841`) | Suyash | ~1hr |
| G27 | Judge checked power ≥ 0.80 on every step even before patients were enrolled | — | ~1hr |
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
