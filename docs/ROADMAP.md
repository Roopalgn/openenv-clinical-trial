# OpenEnv Clinical Trial Designer - Detailed Roadmap

## 0. Gap Analysis vs Winners (Kube SRE Gym, Bio Experiment, Voyager-VRAM)

| # | Gap | Winner Evidence | Severity |
|---|-----|----------------|----------|
| G1 | No ground-truth verification system (hidden latent state checked programmatically) | KubeSRE: real cluster health checks; BioExp: hidden DE genes + NoiseModel | CRITICAL |
| G2 | No multi-layer verification (programmatic + LLM judge) | KubeSRE: health check + LLM judge + phase-aware scoring | CRITICAL |
| G3 | No domain randomization / noise injection | BioExp: budget ±30%, time ±20%, technical noise, batch effects | HIGH |
| G4 | No adversarial / adaptive difficulty designer | KubeSRE: Claude designs incidents targeting weak spots | HIGH |
| G5 | No full GRPO train.py with LoRA + vLLM colocate + reward CSV logging | All 3 winners shipped complete training scripts | HIGH |
| G6 | No Colab training notebook | All 3 winners; judging minimum requirement | HIGH |
| G7 | No openenv.yaml or OpenEnv base Docker image usage | All 3 winners used openenv.yaml + `ghcr.io/meta-pytorch/openenv-base` | HIGH |
| G8 | No reward visualization / plot script | KubeSRE: plot_rewards.py; BioExp: dashboard.html | MEDIUM |
| G9 | No eval_compare.py (base vs trained side-by-side) | KubeSRE: eval.py; BioExp: eval_compare.py | MEDIUM |
| G10 | No ARCHITECTURE.md or design doc | KubeSRE: ARCHITECTURE.md, DESIGN_ADVERSARIAL_JUDGE.md | MEDIUM |
| G11 | No potential-based reward shaping γ·(φ(s')−φ(s)) | All winners used it per comparison.md reward pattern | MEDIUM |
| G12 | No mini-blog on HF or mini-video on YouTube | Judging minimum requirement | HIGH |
| G13 | No phase-aware clinical workflow scoring (like SRE triage→investigate→fix→verify) | KubeSRE: _detect_phase + phase-order bonus/penalty | MEDIUM |
| G14 | No working demo dashboard for live pitch | BioExp: dashboard.html + dashboard.py; VRAM: visual proof | MEDIUM |
| G15 | No episode transcript JSONL with full conversation history | KubeSRE: episode_transcripts.jsonl + agent_transcripts.jsonl | MEDIUM |

### G1–G15 Status: ALL CLOSED (Push 1–6 fully merged, 249/249 tests pass)

### Post-Merge Gaps (G16–G22) — Identified Apr 22 via winner deep-dive

| # | Gap | Winner Evidence | Severity | Rubric Weight |
|---|-----|----------------|----------|---------------|
| G16 | No actual training run — zero reward curves, zero before/after episodes | KubeSRE: 3 training runs with reward curves + failure analysis; BioExp: trajectory datasets; EcomRLVE: trajectory generation | CRITICAL | 20% (Showing Improvement) + 30% (Storytelling) |
| G17 | No grounding in validated clinical trial simulator (no "WOFOST" analog) | KubeSRE: real GKE cluster; BioExp: Scanpy/DESeq2 references; EcomRLVE: 2M real Amazon products | MEDIUM | Weakens 40% (Environment Innovation) defensibility |
| G18 | No HF Space deployed — submission requires GitHub repo + HF Space URL | All 3 winners deployed to HF Spaces | HIGH | Submission requirement |
| G19 | Mini-blog not published on HuggingFace (draft exists in docs/mini_blog_draft.md) | All winners had published deliverables | HIGH | Minimum requirement |
| G20 | Colab notebook untested on free tier — organizers provide HF H100 credits, NOT Colab credits | KubeSRE: kube_sre_gym_colab.ipynb tested; BioExp: multiple notebooks | MEDIUM | Minimum requirement |
| G21 | No environment co-evolution story (cannot exist without actual training runs) | KubeSRE: 3 bugs discovered during training became Statement 4 narrative | CRITICAL | Statement 4 self-improvement |
| G22 | AdversarialDesigner is rule-based templates, not LLM-generated like KubeSRE's Claude designer | KubeSRE: Claude generates targeted multi-fault incidents (483 LOC adversarial_designer.py) | LOW | Nice-to-have; rule-based is defensible |

### Submission Checklist (from hack_info.md — ALL mandatory)

- [ ] GitHub repo (public, clean, with training results)
- [ ] HF Space URL (deployed, `/ping` responds)
- [ ] Mini-blog on HuggingFace OR mini-video on YouTube (< 2 minutes)
- [ ] Training script using Unsloth or HF TRL in Colab
- [ ] Observable training progress (reward curves, before/after behavior)
- [ ] 3-minute pitch + 2-minute Q&A prepared
- [ ] Valid government-issued ID + college/company ID for venue entry

> **IMPORTANT:** Organizers provide HuggingFace H100 credits onsite (Apr 25–26), NOT Google Colab credits. The Colab notebook must work on the free tier (T4 GPU) or connect to the HF Space remotely.

## 1. Mission and Success Bar

- Build a production-quality OpenEnv environment for clinical trial design (Theme #3.1: Professional Tasks).
- Match winner patterns from `comparison.md`:
  - Real world state and objective verification
  - Decomposed reward shaping
  - Progressive curriculum and long-horizon trajectories
  - Reproducible training + clear storytelling
  - Containerized HF Space deployability

## 2. Collaboration Model (No Interference Workflow)

- Branch strategy:
  - Roopal: one active long-lived branch, e.g. `feature/roopal-core`
  - Suyash: one active long-lived branch, e.g. `feature/suyash-core`
  - Keep push milestones inside PRs (labels/titles), not by creating a new branch per push.
- Merge strategy:
  - Both open PRs to `develop` first.
  - One integration PR from `develop` to `main` only after full checklist pass.
  - For each milestone (`Push 1` to `Push 6`), open a focused PR from each active branch into `develop`.
  - Use PR titles prefixed with milestone tags, e.g. `[Push 1] ...`, `[Push 2] ...`.
- File ownership by default:
  - Roopal owns docs/storyline/reward policy/training narrative files.
  - Suyash owns environment engine/simulator/rule-engine/container wiring.
- Shared files (require both approvals):
  - `README.md`
  - `docs/ROADMAP.md`
  - API schema files
  - reward constants and action schema

## 3. Push-by-Push Plan

## Push 1

### Roopal: tasks

- Finalize problem statement, objectives, and judging alignment in docs.
- Define evaluation criteria and acceptance metrics (success rate, reward trend, curriculum progress).
- Draft story arc for demo: failure -> learning -> success.
- Write baseline README sections: motivation, environment, tasks.
- **(G10)** Write initial `ARCHITECTURE.md` with system diagram (env server, simulator, rule engine, curriculum, judge). Model after KubeSRE's architecture doc.
- **(G13)** Define clinical trial phase-aware workflow and phase ordering: literature review → hypothesis → design → enrollment → monitoring → analysis → submission. Document bonus/penalty for phase skipping.

### Suyash: tasks

- Scaffold environment package and OpenEnv/FastAPI app entrypoints.
- Create reset/step/state skeleton with typed action/observation/state models.
- Add Dockerfile + health endpoint (`/ping`) and local run command.
- Add basic lint/test tooling.
- **(G7)** Create `openenv.yaml` (spec_version: 1, runtime: fastapi, app: server.app:app, port: 8000). Use `ghcr.io/meta-pytorch/openenv-base:latest` as Dockerfile base image.
- **(G1)** Implement `TrialLatentState` with hidden ground-truth fields (`true_effect_size`, `true_side_effect_rate`, `true_responder_population`) as the source-of-truth for programmatic verification. Agent must never see these directly.

### Joint gate before push

- Agree on action and observation schema names.
- Agree on API response envelope fields.
- Confirm container starts and ping works.
- `openenv.yaml` validated by OpenEnv CLI.
- `TrialLatentState` fields agreed and hidden from observation.
- **LOCK shared files after Push 1:** Section 5 naming contract (model names, function names, reward keys, state keys, API endpoints, scenario IDs) and `openenv.yaml` are frozen. Any change after this point requires an issue + approval from both members. This enables both branches to diverge safely through Push 6 and merge cleanly at the end.

### PR name: `[Push 1] Roopal: scaffold docs, architecture, phase workflow` / `[Push 1] Suyash: env skeleton, openenv.yaml, latent state`

## Push 2

### Roopal: tasks

- Define reward decomposition spec and write formulas/weights doc.
- Draft scenario cards (4 initial clinical scenarios with difficulty labels).
- Define milestone map for long-horizon episode phases.
- **(G11)** Define potential-based shaping function φ(s) for clinical trial state (e.g., φ = milestone_completion_fraction × budget_efficiency). Document the γ·(φ(s')−φ(s)) formula and why it preserves optimal policy.
- **(G1)** For each scenario card, specify hidden ground-truth values (true_effect_size, true_side_effect_rate, etc.) and the programmatic verification function that checks the agent's final trial design against these values. No LLM-only judging.

### Suyash: tasks

- Implement rule-engine checks (FDA constraints, prerequisite checks).
- Implement reward components as separate functions.
- Add structured episode logging format.
- **(G3)** Implement `NoiseModel` class that centralizes all stochasticity with a seeded `numpy.Generator`. Add domain randomization: budget ±30%, time ±20%, dropout rate ±15%, placebo response ±20%. Prevent overfitting to fixed scenario parameters.
- **(G15)** Implement JSONL episode transcript logger that saves full conversation history per episode (scenario, actions, outputs, rewards, done, latent state seed). Model after KubeSRE's `episode_transcripts.jsonl`.

### Joint gate before push

- Validate reward keys are consistent across docs/code.
- Validate one full episode trace writes expected JSONL schema.
- NoiseModel with same seed reproduces same randomized scenario.
- Ground-truth verification returns correct pass/fail for a hand-crafted episode.

### PR name: `[Push 2] Roopal: reward spec, scenarios, shaping function` / `[Push 2] Suyash: rule engine, noise model, episode logging`

## Push 3

### Roopal: tasks

- Design curriculum progression policy and mastery thresholds.
- Write benchmark protocol (random policy baseline + scripted baseline).
- Define dashboard metrics table format.
- **(G2)** Design the multi-layer verification spec: (1) programmatic ground-truth check via `TrialLatentState`, (2) rule-engine constraint pass, (3) optional LLM judge for qualitative assessment. Document when each layer fires and how conflicts resolve.
- **(G13)** Define phase-order bonus/penalty table: +0.2 for correct clinical workflow order, -0.3 for skipping phases (e.g., jumping to enrollment without hypothesis). Write the `_detect_phase()` heuristic patterns for clinical trial actions.

### Suyash: tasks

- Implement curriculum controller and scenario randomization.
- Implement hidden-state generator and seeded reproducibility.
- Add safety checks for invalid action transitions.
- **(G2)** Implement `TrialJudge` with programmatic verification layer (compare agent's trial design against `TrialLatentState` ground truth: power >= 0.8, type-I error <= 0.05, FDA compliance pass). Add optional LLM judge layer for workflow quality scoring.
- **(G13)** Implement `_detect_phase(action, history)` function that classifies each agent action into clinical workflow phase. Add phase-order bonus (+0.2) and skip penalty (-0.3) to per-step reward.

### Joint gate before push

- Same seed reproduces same episode.
- Curriculum tier advancement logic tested.
- Multi-layer verification tested: programmatic check catches a wrong trial design that "looks reasonable."
- Phase-order scoring tested on a hand-crafted good-order vs bad-order episode.

### PR name: `[Push 3] Roopal: curriculum policy, verification spec, phase scoring` / `[Push 3] Suyash: curriculum controller, judge, phase detection`

## Push 4

### Roopal: tasks

- Prepare model-training runbook (GRPO settings, expected outputs, logs).
- Create evaluation report template and charts checklist.
- Draft mini-blog/2-minute demo structure.
- **(G6)** Create `train_colab.ipynb` — Colab notebook that connects to HF Space, configures GRPO with TRL, runs training episodes, saves checkpoints to HF Hub. Include markdown cells explaining each step. This is a **judging minimum requirement**.
- **(G12)** Draft the HuggingFace mini-blog outline (problem → environment → reward design → training results → what agent learned). This is a **judging minimum requirement**.

### Suyash: tasks

- Build minimal train runner integration with environment.
- Add evaluation script (base vs trained comparison).
- Add runtime config via env vars.
- **(G5)** Implement full `train.py` with: GRPO via TRL + vLLM colocate, LoRA config (rank 16, alpha 32), reward CSV logger per episode, system prompt for clinical trial agent, `rollout_func` that calls env.reset/step. Model after KubeSRE's train.py structure.
- **(G9)** Implement `eval_compare.py` that runs both base model and trained checkpoint through N random scenarios and reports: success rate, avg reward, avg steps, scenario-type breakdown. Model after KubeSRE's eval.py.
- **(G8)** Implement `plot_rewards.py` that reads reward CSV and generates: per-episode reward scatter + rolling average, trend line with slope annotation, best/mean/final stats. Save as PNG.

### Joint gate before push

- End-to-end local dry run completes.
- Baseline metrics are generated and saved.
- `train.py --max-steps 2` runs without error (smoke test).
- `eval_compare.py` produces comparison table.
- `plot_rewards.py` generates a valid PNG from reward CSV.
- Colab notebook cells run sequentially without error.

### PR name: `[Push 4] Roopal: training runbook, Colab notebook, mini-blog draft` / `[Push 4] Suyash: train.py, eval_compare.py, plot_rewards.py`

## Push 5

### Roopal: tasks

- Improve prompt/action instructions for better policy behavior.
- Tune reward weights based on diagnostics.
- Update README with observed baseline scores.
- **(G4)** Design adaptive difficulty spec: after agent masters a scenario type (>70% success over window), the environment should randomize harder parameter ranges (tighter budgets, rarer diseases, compound endpoints). Document how weak-spot targeting works for curriculum.
- **(G14)** Build `dashboard.html` — a single-page live demo dashboard showing: current episode replay, reward curve chart, scenario difficulty progression, agent action log. Embeddable in HF Space. This is critical for the 3-minute pitch.

### Suyash: tasks

- Optimize simulator performance and caching.
- Harden error handling and edge-case validations.
- Add deterministic integration tests.
- **(G4)** Implement adaptive difficulty in curriculum controller: when agent masters a scenario type, shift domain randomization to harder parameter ranges (lower budgets, higher dropout, smaller effect sizes). Add `get_weak_spots()` method that returns scenario types below mastery threshold.
- **(G14)** Implement `dashboard.py` backend that serves dashboard.html via the FastAPI app, streams live episode data via SSE/WebSocket, and exposes `/dashboard` endpoint.

### Joint gate before push

- No schema drift from agreed contract names.
- Performance and stability pass threshold.
- Adaptive difficulty tested: mastered scenario types get harder parameters.
- Dashboard renders in browser and shows a replayed episode.

### PR name: `[Push 5] Roopal: reward tuning, adaptive difficulty spec, dashboard UI` / `[Push 5] Suyash: adaptive curriculum, dashboard backend, hardening`

## Push 6

### Roopal: tasks

- Finalize storytelling assets (before/after episodes, failure-first narrative).
- Prepare judge-facing pitch notes aligned to scoring weights.
- Final proofreading of all docs for consistency.
- **(G12)** Publish HuggingFace mini-blog (< 2 min read) with: problem statement, environment screenshot/diagram, reward curve showing improvement, before/after episode comparison. **Judging minimum requirement.**
- **(G10)** Finalize `ARCHITECTURE.md` with actual training results, H100 setup instructions, and the complete system diagram reflecting all implemented components.

### Suyash: tasks

- Final HF Space container hardening.
- Validate Docker build/run and startup health.
- Add final CI checks for tests/lint/format.
- **(G7)** Validate full OpenEnv compatibility: `openenv.yaml` passes CLI validation, Docker image builds from `openenv-base`, `/reset`, `/step`, `/state`, `/schema`, `/ws`, `/ping` all respond correctly.
- **(G15)** Add episode transcript export endpoint (`/transcripts`) that returns JSONL of all logged episodes for demo replay.

### Joint gate before push

- HF Space pass checklist complete.
- `main` merge readiness approved by both.
- Mini-blog published on HuggingFace.
- Dashboard works on deployed HF Space.
- At least 3 complete episode transcripts saved for demo replay.

### PR name: `[Push 6] Roopal: storytelling, mini-blog, final docs` / `[Push 6] Suyash: HF Space hardening, OpenEnv validation, transcript export`

## Push 7 — Pre-Onsite Readiness (Apr 22–24, before travel)

**Goal:** Everything deployment-ready and smoke-tested so onsite time is 100% training + deliverable creation. No code-level surprises on Apr 25.

### Roopal: tasks

- **(G18)** Deploy to HuggingFace Spaces: create HF Space repo (Docker SDK), push Docker image, verify `/ping` responds publicly. Document the public Space URL in `README.md`.
- **(G20)** Test `train_colab.ipynb` on Colab **free tier** (T4 GPU, 15 GB VRAM). If Qwen2.5-7B does not fit on T4, add a fallback cell using `Qwen/Qwen2.5-1.5B-Instruct` with Unsloth 4-bit quantization. Add a "Connect to HF Space" mode where Colab sends requests to the deployed HF Space environment instead of running locally.
- **(G17)** Add grounding citations and validation data:
  - Create `docs/grounding.md` citing: **rpact** (R, LGPL-3, FDA-validated, 39K unit tests — confirmatory adaptive trial design + simulation), **Berry et al. (2010)** _Bayesian Adaptive Methods for Clinical Trials_, **Wassmer & Brannath (2016)** _Group Sequential and Confirmatory Adaptive Designs_, **Narvekar et al. (2020)** _Curriculum Learning for RL Domains_ (JMLR, 950 citations).
  - Precompute rpact-equivalent validation tables for each scenario (expected power at various sample sizes, critical boundary values). Store as `server/grounding/rpact_validation.json`. Reference in `ARCHITECTURE.md` and `docs/pitch_notes.md`.
- Prepare onsite training checklist document: exact terminal commands, env vars to set, model size fallback ladder (7B → 3B → 1.5B), expected runtime per model size, disk/VRAM budget for H100.
- Update `docs/pitch_notes.md` with placeholders for `[INSERT REWARD CURVE]`, `[INSERT BEFORE/AFTER EPISODE]`, `[INSERT CO-EVOLUTION BUG]` — to be filled onsite.

### Suyash: tasks

- **(G18)** Validate Docker build works end-to-end on a clean machine: `docker build -t ct-env . && docker run -p 8000:8000 ct-env` → `/ping` returns 200. Fix any missing dependencies or startup errors.
- **(G20)** Add `--dry-run` mode to `train.py` that runs 2 episodes with a random policy to verify the full pipeline: model load → rollout → reward computation → CSV log → plot generation. This catches integration bugs **before** onsite.
- Add `--model-size` flag to `train.py` that auto-selects LoRA rank + batch size + sequence length for different model sizes:
  - `1.5b`: LoRA rank 8, batch 1, seq 2048, grad_accum 4
  - `3b`: LoRA rank 16, batch 1, seq 3072, grad_accum 4
  - `7b`: LoRA rank 16, batch 1, seq 4096, grad_accum 8
- **(G22, optional)** Add LLM adversarial mode to `adversarial_designer.py`: when env var `ADVERSARIAL_LLM_BACKEND` is set (e.g., `anthropic` or `openai`), call the external LLM to generate scenario parameters instead of rule-based compound templates. Graceful fallback to rule-based if API call fails or env var is unset.
- Verify `pyproject.toml` pins: `trl==0.29.0`, `peft>=0.11`, `unsloth`. Ensure `pip install -e ".[train]"` works clean.

### Joint gate before push

- HF Space `/ping` responds publicly with HTTP 200.
- `docker build` + `docker run` works on a fresh machine.
- `python train.py --dry-run --episodes 2` completes without error, writes reward CSV, generates plot PNG.
- Colab notebook runs on free tier (T4) with 1.5B model fallback without OOM.
- `docs/grounding.md` written and referenced from ARCHITECTURE.md.
- Onsite checklist document reviewed by both.

### PR name: `[Push 7] Roopal: HF deploy, Colab T4, grounding, onsite checklist` / `[Push 7] Suyash: Docker validation, dry-run mode, model-size flag`

---

## Push 8 — Onsite Training + Deliverables (Apr 25–26, Scaler campus)

**Goal:** Generate the 50% rubric evidence — training runs with real reward curves (20% Showing Improvement) + compelling before/after storytelling (30% Storytelling). Document every environment bug as Statement 4 co-evolution.

> **Compute:** HuggingFace H100 credits provided onsite. NOT Colab credits.

### Phase 1: First Training Run (Day 1, Hours 0–4)

**Suyash leads execution, Roopal monitors + documents.**

- **(G16)** Launch GRPO training on HF H100. **Start with the smallest viable model** for fast iteration:
  - `python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 20 --seed 42`
  - Target: ~20 episodes in ~1 hour. This gives reward data FAST.
- Log every episode to CSV (`logs/rewards.csv`) + JSONL (`logs/transcripts.jsonl`).
- After Run 1 completes, run `python plot_rewards.py` to check signal quality.
- **(G21)** **Document EVERY bug or surprise discovered during training.** Examples:
  - Reward too generous / too sparse → adjust weights
  - Agent stuck on same action → add repeat penalty or novelty bonus
  - Phase ordering penalties too harsh → relax warmup tier
  - Output parsing failures → fix `_build_action_from_text`
  - These bugs **ARE the Statement 4 co-evolution story.** Log them in `docs/training_log.md`.

### Phase 2: Iterate + Scale (Day 1, Hours 4–10)

- If Run 1 shows learning signal (positive reward slope): Run 2 with more episodes (50–100) and optionally a larger model (3B).
- If Run 1 shows NO signal (flat or negative): debug reward weights, check shaping function, try different model size. **Document the iteration** — this is also part of the story.
- Run `python eval_compare.py` to get base (random) vs trained comparison tables.
- Extract 3 key episode transcripts from JSONL: (1) episode 1 failure, (2) mid-training breakthrough, (3) late-episode success.

### Phase 3: Deliverables (Day 2, Hours 0–6)

**Roopal leads deliverables, Suyash runs final eval + checkpoint upload.**

#### Roopal: tasks

- **(G19)** Publish mini-blog on HuggingFace: fill all `[fill]` placeholders in `docs/mini_blog_draft.md` with real data, then publish. Must include: problem statement, architecture diagram, reward curve image, before/after episode comparison, what the agent learned, co-evolution bugs.
- Update `README.md` § Results with actual training metrics (replace placeholder tables).
- Update `ARCHITECTURE.md` § Post-Training Results with: config table, summary stats, key observations.
- Fill `docs/pitch_notes.md` placeholders with real reward numbers and episode excerpts.
- Update `dashboard.html` demo data with real episode transcripts.

#### Suyash: tasks

- Push trained LoRA checkpoint to HuggingFace Hub (e.g., `roopal-gn/clinical-trial-agent-lora`).
- Run final `eval_compare.py` with trained checkpoint. Save output as `results/eval_report.json`.
- Generate final `results/reward_curve.png` via `plot_rewards.py`.
- Commit all training artifacts to repo: `logs/rewards.csv`, `logs/transcripts.jsonl`, `results/eval_report.json`, `results/reward_curve.png`.
- Verify HF Space still responds correctly after any code changes from bug fixes.

### Phase 4: Pitch Prep (Day 2, Hours 6–8)

**Joint — both rehearse together.**

- Rehearse 3-minute pitch with real data:
  - 0:00–0:30 — Problem: clinical trials cost $2.6B avg, 90% failure rate, high-stakes professional task.
  - 0:30–1:30 — Environment demo: show dashboard, hidden state, 19-action space, 10-phase workflow, 5-tier curriculum.
  - 1:30–2:30 — Training results: reward curve, before/after episode, curriculum tier progression.
  - 2:30–3:00 — Statement 4: bugs found during training → environment improved → self-improvement loop.
- Prepare Q&A answers for likely judge questions:
  - "Why not real clinical trial data?" → Grounded in rpact/scipy.stats power calcs + FDA ICH E9 rules, not toy heuristics.
  - "How does difficulty adapt?" → Per-scenario mastery tracking, weak-spot targeting, adversarial compound challenges at expert tier.
  - "What did the agent learn?" → Show episode 1 vs episode N side by side.
  - "How is this different from a chatbot?" → Hidden ground truth, objective verification, no shortcut exploits.

### Joint gate before push

- At least 1 complete training run with reward CSV + reward curve plot.
- At least 3 episode transcripts extracted for demo (failure / breakthrough / success).
- Mini-blog published on HuggingFace with real data.
- `README.md` updated with actual results.
- Trained LoRA checkpoint on HuggingFace Hub.
- `docs/training_log.md` documents bugs found and fixes applied (Statement 4 evidence).
- Pitch rehearsed with real data at least once.

### PR name: `[Push 8] Roopal: mini-blog, results docs, pitch` / `[Push 8] Suyash: training runs, checkpoint, eval, artifacts`

---

### Risk Matrix for Apr 25–26

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 7B model OOMs on H100 with env server | Medium | High | Start with 1.5B (KubeSRE won with 1.7B). `--model-size` flag makes switching instant. |
| GRPO shows no learning signal | Medium | Critical | Debug reward weights live. Even flat curves with annotated analysis are better than zero. KubeSRE's Run 2 had negative slope — they documented it as a lesson. |
| HF H100 credits delayed or limited | Low | Critical | Have Colab T4 fallback with 1.5B + 4-bit quant. |
| Docker image fails on HF Spaces | Medium | High | Validated in Push 7. Have local demo backup for pitch. |
| TRL version incompatibility with vLLM | Medium | Medium | Pinned in pyproject.toml. Tested in `--dry-run`. |
| No time for large training run | High | Medium | Even 10–20 episodes with annotations beats zero. KubeSRE's first real run was 12 episodes. |

## 4. Pre-Merge Checklist

### Push 1–6 (Code Complete) ✅ ALL DONE

- [x] OpenEnv compatibility verified (`openenv.yaml` + base image + all endpoints).
- [x] Docker image builds cleanly.
- [x] Container runs and responds on ping endpoint.
- [x] Action/observation/state schemas locked and documented.
- [x] At least 3+ task families with graders/verification.
- [x] Decomposed reward with documented components and weights.
- [x] Potential-based reward shaping (γ·(φ(s')−φ(s))) implemented.
- [x] Curriculum tiers and mastery progression implemented.
- [x] Domain randomization via NoiseModel with seeded reproducibility.
- [x] Multi-layer verification: programmatic ground-truth + rule engine + optional LLM judge.
- [x] Phase-aware clinical workflow scoring with bonus/penalty.
- [x] `train.py` with GRPO + LoRA + vLLM colocate working.
- [x] `train_colab.ipynb` created with all cells.
- [x] `plot_rewards.py` generates reward curves from CSV.
- [x] Episode transcripts saved as JSONL with full history.
- [x] README includes motivation, setup, usage, task descriptions.
- [x] `ARCHITECTURE.md` with system diagram and training setup.
- [x] `dashboard.html` renders live episode replay and reward curves.
- [x] Adaptive difficulty / weak-spot targeting tested.
- [x] 249/249 tests pass, ruff lint clean.

### Push 7 (Pre-Onsite) — Before Apr 25

- [ ] HF Space deployed and `/ping` responds publicly.
- [ ] Docker build + run verified on clean machine.
- [ ] `train.py --dry-run --episodes 2` completes without error.
- [ ] Colab notebook tested on free tier (T4) with 1.5B fallback.
- [ ] `docs/grounding.md` written with rpact + paper citations.
- [ ] `--model-size` flag in train.py (1.5b / 3b / 7b presets).
- [ ] Onsite checklist document prepared.

### Push 8 (Onsite) — Apr 25–26

- [ ] At least 1 GRPO training run completed with reward CSV.
- [ ] Reward curve plot generated (`results/reward_curve.png`).
- [ ] `eval_compare.py` run: base vs trained comparison saved.
- [ ] 3+ episode transcripts extracted for demo.
- [ ] Training bugs documented in `docs/training_log.md` (Statement 4).
- [ ] Mini-blog published on HuggingFace (< 2 min read).
- [ ] README updated with actual training results.
- [ ] ARCHITECTURE.md post-training section filled.
- [ ] Trained LoRA checkpoint pushed to HF Hub.
- [ ] Pitch rehearsed with real data.

## 5. Naming Contract (Do Not Rename Without Joint Approval)

## 5.1 Core Models

- `TrialAction`
- `TrialObservation`
- `TrialState`
- `TrialLatentState`
- `TrialResult`
- `ScenarioConfig`
- `RewardBreakdown`

## 5.2 Core Functions

- `reset`
- `step`
- `simulate_trial`
- `calculate_power`
- `check_fda_compliance`
- `compute_reward`
- `advance_curriculum`

## 5.3 Reward Keys

- `r_validity`
- `r_ordering`
- `r_info_gain`
- `r_efficiency`
- `r_novelty`
- `r_penalty`
- `r_terminal_success`
- `r_terminal_calibration`

## 5.4 State and Config Keys

- `true_effect_size`
- `true_side_effect_rate`
- `true_responder_population`
- `placebo_response_rate`
- `dropout_rate`
- `budget_remaining`
- `time_remaining_days`
- `patients_enrolled`

## 5.5 API and Endpoint Names

- `/reset`
- `/step`
- `/state`
- `/schema`
- `/ws`
- `/ping`

## 5.6 Scenario IDs (Initial)

- `solid_tumor_chemo`
- `autoimmune_biologic`
- `cns_depression`
- `rare_disease_orphan`

## 6. Coordination Cadence

- Daily 15-minute sync:
  - Yesterday completed
  - Today planned
  - Blockers and schema changes
- Schema change rule:
  - Any rename requires an issue + approval from both members.
- Conflict prevention:
  - Rebase before opening PR.
  - Never force-push shared branches.

## 7. Definition of Done (Winner-Level)

- Environment behavior is objectively verifiable (not LLM-judged only).
- Reward learning signal is interpretable and improves over training.
- Demo shows clear capability progression.
- Long-horizon tasks are non-trivial and failure modes are realistic.
- Repo is reproducible and containerized for HF Space.
- **At least 1 training run with documented reward curves** (20% rubric).
- **Before/after episode comparison with storytelling arc** (30% rubric).
- **HF Space deployed and publicly accessible** (submission requirement).
- **Mini-blog published on HuggingFace** (minimum requirement).
- **Colab notebook runs on free tier** (minimum requirement).
- **Environment co-evolution documented** — bugs found during training, fixes applied, lessons learned (Statement 4).
- **Grounding citations** — validated simulator references (rpact, FDA ICH E9) in docs.
