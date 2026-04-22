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
| G16 | No actual training run — zero reward curves, zero before/after episodes | KubeSRE: 3 training runs with reward curves + failure analysis; BioExp: trajectory datasets; EcomRLVE: trajectory generation | CRITICAL | 20% (Showing Improvement) + 30% (Storytelling) — **all training onsite Apr 25–26 on HF H100** |
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

> **IMPORTANT (UPDATED Apr 22):** Organisers confirmed **NO training is allowed before Apr 25.** All GRPO training happens onsite using HuggingFace H100 credits. Kaggle/Colab notebooks are used for pipeline validation (dry-run) and as judge-facing deliverables only. Push 7 Phase B is preparation/validation only. Push 8 onsite is where ALL training happens.

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

## Push 7 — Deploy + Maximum Preparation (Apr 22–24, before travel)

**Goal:** Merge all Phase A work, battle-test the full training pipeline with dry-runs, and prepare every artifact so Apr 25 onsite training is **zero-friction execution** from minute one. No actual model training before onsite — only pipeline validation and deliverable preparation.

> **RULE (confirmed by organisers):** NO training before Apr 25. All GRPO runs happen onsite on HF H100 credits. Kaggle/Colab are for dry-run pipeline validation and notebook testing only.
>
> **Resources:** `docs/internal/resources.md` contains full hackathon reference material — OpenEnv docs, TRL/GRPO guides, reward engineering papers, common pitfalls, and Q&A from organiser sessions. Study before onsite.

### Free GPU Platform Reference (for onsite fallback + notebook testing)

| Platform | GPU | VRAM | Weekly Quota | Session Limit | Stability | Use Case |
|----------|-----|------|-------------|---------------|-----------|----------|
| **Kaggle** | T4 ×2 or P100 | 16 GB | **30 hrs/week** (sometimes higher) | 12 hrs/session | High — no random disconnects | **Dry-run validation** + notebook testing; onsite fallback if H100 unavailable |
| **Colab Free** | T4 | 15 GB | Dynamic / unpublished | 12 hrs max | Medium — sessions can terminate randomly | **Judge deliverable** — notebook must run here; dry-run only |
| **HF H100 (onsite)** | H100 | 80 GB | Provided by organisers | Event duration | N/A | **PRIMARY — ALL real training happens here** |

**VRAM Budget Reference (for onsite model selection):**

| Model | BF16 (H100) | 4-bit (T4 fallback) | Fits H100? | Fits T4? |
|-------|-------------|---------------------|------------|----------|
| Qwen2.5-1.5B | ~3 GB | ~3.7 GB | ✅ Trivial | ✅ Comfortably |
| Qwen2.5-3B | ~6 GB | ~6.2 GB | ✅ Easy | ✅ Yes |
| Qwen2.5-7B | ~14 GB | ~10.4 GB | ✅ Comfortable | ⚠️ Tight |

### Phase A: Deploy + Smoke-Test ✅ COMPLETE

> **Status:** Both Roopal and Suyash Phase A tasks are done on their respective branches as of Apr 22.

#### Roopal: tasks — ✅ ALL DONE

- ✅ **(G18)** HF Space deployed at `https://roopalgn-openenv-clinical-trial.hf.space/ping` — responds publicly.
- ✅ **(G17)** `docs/grounding.md` created with rpact, Berry 2010, Wassmer & Brannath 2016, Narvekar 2020 citations.
- ✅ `server/grounding/rpact_validation.json` created with precomputed validation tables.
- ✅ ARCHITECTURE.md and pitch_notes.md updated with grounding references.
- ✅ `train_kaggle.ipynb` scaffold created (Kaggle GPU setup, HF Space HTTP smoke-check, training template).
- ✅ README.md has Space placeholder section and Kaggle notebook reference.

#### Suyash: tasks — ✅ ALL DONE

- ✅ **(G18)** Docker build validated: PORT 8000 → 7860 aligned with HF Spaces, CI port mapping fixed.
- ✅ `--dry-run` mode added to `train.py`: 2 episodes with random policy, writes reward CSV + generates plot PNG. Random policy baseline: −61.6 reward.
- ✅ `--model-size` flag added (1.5b / 3b / 7b presets with auto-configured LoRA rank, batch, seq, grad_accum).
- ✅ Dynamic `save_steps` formula + `save_total_limit=3` for disk safety.
- ✅ `eval_compare.py` got `--base-only` flag for baseline establishment before any checkpoint exists.
- ✅ `pyproject.toml` pins verified: `trl==0.29.0`, `peft>=0.11`, unsloth isolated to `train-unsloth` extra.
- ✅ CI all green (lint, format, pytest 249/249, Docker smoke test).

#### Phase A Gate — ✅ PASSED

- ✅ HF Space `/ping` responds publicly with HTTP 200.
- ✅ `docker build` + `docker run` works.
- ✅ `python train.py --dry-run --episodes 2` completes, writes CSV + PNG.
- ✅ `python train.py --model-size 1.5b --dry-run` logs correct preset.
- ✅ `python eval_compare.py --base-only --episodes 3` completes.
- ✅ `docs/grounding.md` written and referenced from ARCHITECTURE.md.
- ✅ 249/249 tests pass, CI all green.

### Phase B: Maximum Preparation (Apr 23–24)

> **NO training before Apr 25.** Use this time to validate notebooks with dry-run, prepare deliverable templates, and ensure onsite execution is frictionless.

#### Available Artifacts (on main after merge)

- `outputs/grpo/reward_log.csv` — random policy baseline from dry-run (2 episodes, mean reward −61.6)
- `outputs/grpo/training_summary.json` — dry-run summary (model: `dry-run/random-policy`, seed 42, tier 0)
- `docs/internal/resources.md` — full hackathon reference guide (OpenEnv, TRL, GRPO, reward engineering, pitfalls)

#### Roopal: tasks (validate + prepare)

- **(G20)** Test `train_colab.ipynb` on Colab **free tier** (T4) with `--dry-run` mode. Verify it connects to HF Space, runs 2 dry-run episodes, produces valid CSV. If Qwen2.5-7B setup cell fails on T4, add a 1.5B + Unsloth 4-bit fallback cell.
- **(G20)** Test `train_kaggle.ipynb` on Kaggle (T4/P100) with `--dry-run` mode. Same validation — both notebooks become judge deliverables.
- Prepare `docs/onsite_checklist.md` — step-by-step terminal commands for onsite:
  - Exact `git clone` + `pip install` sequence for H100 environment
  - H100 environment variable setup (`CUDA_VISIBLE_DEVICES`, `HF_TOKEN`, etc.)
  - Model size fallback ladder: try 7B first → 3B fallback → 1.5B safety net
  - Expected runtime per model size on H100 (estimated: 7B ~15 min/episode, 3B ~8 min/episode, 1.5B ~3 min/episode)
  - Checkpoint frequency and disk budget
- Prepare deliverable templates with `[FILL ONSITE]` placeholders:
  - `docs/mini_blog_draft.md`: all sections written, reward curve/episode slots marked
  - `docs/internal/pitch_notes.md`: pitch script complete except data placeholders
  - `docs/training_log.md`: template with columns ready for Statement 4 bugs
- Study `docs/internal/resources.md` for reward engineering best practices and common pitfalls before onsite.

#### Suyash: tasks (validate + harden)

- Run full integration test on main: `python train.py --dry-run --episodes 5 --model-size 1.5b` → verify CSV, PNG, JSONL all correct.
- Run `python eval_compare.py --base-only --episodes 5` → verify baseline JSON report.
- **(G22, optional)** Add LLM adversarial mode to `adversarial_designer.py` if time permits.
- Stress-test `train.py` with edge cases: `--episodes 1`, `--episodes 100` (dry-run), verify checkpoint logic, verify resume-from-checkpoint works.
- Verify HF Space is still live and `/ping` + `/reset` + `/step` all respond after merges.

#### Branch Merge Status ✅ COMPLETE (Apr 22)

Both branches merged to main:
- **Suyash branch** (6 commits): `train.py` --dry-run + --model-size + dynamic save_steps, `eval_compare.py` --base-only, pyproject.toml unsloth isolation, Dockerfile PORT 7860, CI port fix. Conflict in train.py resolved (kept Suyash's _dry_run + direct plot import, merged back _apply_model_size_preset + terminal_outcome + _write_summary from main).
- **Roopal branch**: dry-run output artifacts (`outputs/grpo/reward_log.csv`, `outputs/grpo/training_summary.json`), train.py plot arg fix (--csv/--out).
- **Random policy baseline established:** mean reward −61.6, 2 episodes, tier 0.

#### Phase B Gate

- Both branches merged to main cleanly. ✅ DONE
- `python train.py --dry-run --episodes 5` works on merged main.
- Colab notebook tested on free tier with dry-run.
- Kaggle notebook tested with dry-run.
- `docs/onsite_checklist.md` created and reviewed by both.
- All deliverable templates have `[FILL ONSITE]` placeholders.
- HF Space live and responding.

### PR name: `[Push 7] Merge Phase A + onsite preparation`

---

## Push 8 — Onsite Training + Deliverables (Apr 25–26, Scaler campus)

**Goal:** Execute ALL GRPO training on H100, generate the 50% rubric evidence (reward curves + storytelling), and ship all deliverables. Pipeline is battle-tested from Push 7 dry-runs — execution should be smooth.

> **Compute:** HuggingFace H100 credits provided onsite (80 GB VRAM). This is where ALL training happens.
>
> **Advantage from Push 7:** `--dry-run` validated, `--model-size` presets tested, notebooks verified, onsite checklist ready. No debugging — just execute.

### What We Arrive With (from Push 7 Preparation)

- ✅ HF Space live at `https://roopalgn-openenv-clinical-trial.hf.space`
- ✅ `train.py` with `--dry-run` + `--model-size` flags tested and working
- ✅ Random policy baseline: −61.6 reward (from dry-run)
- ✅ `eval_compare.py --base-only` baseline established
- ✅ Colab + Kaggle notebooks validated with dry-run
- ✅ `docs/onsite_checklist.md` with exact step-by-step commands
- ✅ All deliverable templates with `[FILL ONSITE]` placeholders
- ✅ `docs/training_log.md` template ready for Statement 4 bugs
- ✅ `docs/internal/resources.md` with full hackathon reference material
- ✅ `docs/grounding.md` + `rpact_validation.json` ready for judge questions

### Phase 1: First Training Run (Day 1, Hours 0–4)

**Suyash leads execution, Roopal monitors + documents.**

- **(G16)** Launch GRPO training on H100. **Strategy: start small, scale fast.**
  - **Run 1 (fast signal check):** `python train.py --model-size 1.5b --model-path Qwen/Qwen2.5-1.5B-Instruct --episodes 20 --seed 42`
    - Target: ~20 episodes in ~1 hour. Uses only ~3 GB of 80 GB VRAM. This gives reward data FAST.
    - BF16 natively on H100, no quantization needed.
  - **Run 2 (scale up):** If Run 1 shows signal → immediately: `python train.py --model-size 3b --model-path Qwen/Qwen2.5-3B-Instruct --episodes 50 --seed 42`
    - Or if ambitious: `--model-size 7b` with 30 episodes — H100 handles full 7B in BF16 easily.
  - **If Run 1 shows NO signal:** Debug reward weights, check shaping function, try different curriculum tier. **Document the iteration** — this is ALSO part of the Statement 4 co-evolution story.
- Log every episode to CSV (`logs/rewards.csv`) + JSONL (`logs/transcripts.jsonl`).
- After each run, immediately run `python plot_rewards.py` to check signal quality.
- **(G21)** **Document EVERY bug or surprise discovered during training** in `docs/training_log.md`:
  - Reward too generous / too sparse → adjust weights
  - Agent stuck on same action → add repeat penalty or novelty bonus
  - Phase ordering penalties too harsh → relax warmup tier
  - Output parsing failures → fix `_build_action_from_text`
  - These bugs **ARE the Statement 4 co-evolution story.**

### Phase 2: Iterate + Final Eval (Day 1, Hours 4–10)

- If Run 1/2 show learning signal (positive reward slope): Run 3 with more episodes (50–100) and/or larger model.
- Run `python eval_compare.py` to get base (random, −61.6) vs trained comparison tables.
  - Save output as `results/eval_report.json`.
- Extract 3 key episode transcripts from JSONL: (1) episode 1 failure, (2) mid-training breakthrough, (3) late-episode success.
- Generate overlaid reward curve plot showing all runs if multiple were completed.
- Focus on curriculum progression: did the agent reach higher tiers on longer runs?

### Phase 3: Deliverables (Day 2, Hours 0–6)

**Roopal leads deliverables, Suyash runs final checkpoint upload.**

#### Roopal: tasks

- **(G19)** Publish mini-blog on HuggingFace: fill all `[FILL ONSITE]` placeholders in `docs/mini_blog_draft.md` with real training data. Must include: problem statement, architecture diagram, reward curve image, before/after episode comparison, what the agent learned, co-evolution bugs.
- Update `README.md` § Results with actual training metrics from best run (replace placeholder tables).
- Update `ARCHITECTURE.md` § Post-Training Results with: config table, summary stats, key observations.
- Fill `docs/internal/pitch_notes.md` with real reward numbers and episode excerpts.
- Update `dashboard.html` demo data with best episode transcripts.

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
  - 1:30–2:30 — Training results: reward curve showing improvement, before/after episode, curriculum tier progression.
  - 2:30–3:00 — Statement 4: bugs found during training → environment improved → self-improvement loop.
- Prepare Q&A answers for likely judge questions:
  - "Why not real clinical trial data?" → Grounded in rpact/scipy.stats power calcs + FDA ICH E9 rules, not toy heuristics.
  - "How does difficulty adapt?" → Per-scenario mastery tracking, weak-spot targeting, adversarial compound challenges at expert tier.
  - "What did the agent learn?" → Show episode 1 vs episode N side by side.
  - "How is this different from a chatbot?" → Hidden ground truth, objective verification, no shortcut exploits.
  - "Where did you train?" → All training onsite on HF H100 credits. Pipeline was validated pre-event with dry-runs.

### Joint gate before push

- At least 1 complete training run with reward CSV + reward curve plot.
- At least 3 episode transcripts extracted for demo (failure / breakthrough / success).
- Mini-blog published on HuggingFace with real data.
- `README.md` updated with actual results.
- Trained LoRA checkpoint on HuggingFace Hub.
- `docs/training_log.md` documents bugs found and fixes applied (Statement 4 evidence).
- Pitch rehearsed with real data at least once.

### PR name: `[Push 8] Roopal: mini-blog, results docs, pitch` / `[Push 8] Suyash: H100 training, checkpoint, eval, artifacts`

---

### Risk Matrix

| Risk | Phase | Likelihood | Impact | Mitigation |
|------|-------|-----------|--------|------------|
| 7B model OOMs on H100 with env server | Push 8 | Low | Medium | H100 has 80 GB VRAM — 7B BF16 uses ~14 GB. `--model-size` flag makes switching instant. Start with 1.5B. |
| GRPO shows no learning signal | Push 8 | Medium | Critical | Debug reward weights live. Start with 1.5B for fast iteration. Even flat curves with annotated analysis are better than zero. |
| HF H100 credits delayed or limited | Push 8 | Low | Critical | Have Kaggle T4 fallback with 1.5B + 4-bit quant. Notebooks pre-validated with dry-run. |
| Docker image fails on HF Spaces | Push 7 | Low | High | Already validated. Space is live. Have local demo backup. |
| TRL version incompatibility with vLLM | Push 7 | Low | Medium | Pinned in pyproject.toml. Tested in `--dry-run`. |
| Merge conflicts between roopal + Suyash | Push 7 | ~~Low~~ | ~~Medium~~ | ✅ RESOLVED — merged Apr 22, train.py conflict resolved manually. |
| Pipeline bug during first real training | Push 8 | Medium | High | Mitigated by extensive dry-run testing. `--dry-run` validated on both branches. Onsite checklist has debug commands. |
| No time for large training run | Push 8 | Medium | Medium | Even 10–20 episodes with annotations beats zero. Start with 1.5B (fastest). |

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

### Push 7 (Deploy + Preparation) — Before Apr 25

#### Phase A: Deploy ✅ COMPLETE
- [x] HF Space deployed and `/ping` responds publicly.
- [x] Docker build + run verified (PORT 7860 aligned).
- [x] `train.py --dry-run --episodes 2` completes without error.
- [x] `train.py --model-size` flag working (1.5b / 3b / 7b presets).
- [x] `eval_compare.py --base-only` baseline established (−61.6 reward).
- [x] `docs/grounding.md` written with rpact + paper citations.
- [x] `server/grounding/rpact_validation.json` created.
- [x] `train_kaggle.ipynb` scaffold created.
- [x] CI all green (lint, format, pytest 249/249, Docker).

#### Phase B: Preparation (Apr 23–24)
- [x] Both branches merged to main cleanly (Apr 22 — Suyash + roopal merged, train.py conflict resolved).
- [ ] `python train.py --dry-run --episodes 5` works on merged main.
- [ ] Colab notebook tested on free tier (T4) with dry-run.
- [ ] Kaggle notebook tested with dry-run.
- [ ] `docs/onsite_checklist.md` created and reviewed.
- [ ] All deliverable templates have `[FILL ONSITE]` placeholders ready.
- [ ] HF Space live and responding after merges.

### Push 8 (Onsite Training + Deliverables) — Apr 25–26

- [ ] At least 1 GRPO training run completed (1.5B or larger, 20+ episodes) with reward CSV.
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
