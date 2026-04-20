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

## 4. Pre-Merge Checklist (Must Finish Before Main)

- OpenEnv compatibility verified (`openenv.yaml` + base image + all endpoints).
- Docker image builds cleanly.
- Container runs and responds on ping endpoint.
- Action/observation/state schemas locked and documented.
- At least 3+ task families with graders/verification.
- Decomposed reward with documented components and weights.
- Potential-based reward shaping (γ·(φ(s')−φ(s))) implemented.
- Curriculum tiers and mastery progression implemented.
- Domain randomization via NoiseModel with seeded reproducibility.
- Multi-layer verification: programmatic ground-truth + rule engine + optional LLM judge.
- Phase-aware clinical workflow scoring with bonus/penalty.
- Baseline and trained metrics generated via `eval_compare.py`.
- `train.py` with GRPO + LoRA + vLLM colocate working.
- `train_colab.ipynb` runs end-to-end in Colab.
- `plot_rewards.py` generates reward curves from CSV.
- Episode transcripts saved as JSONL with full history.
- README includes motivation, setup, usage, task descriptions, baseline scores.
- `ARCHITECTURE.md` with system diagram and training setup.
- Demo narrative prepared with evidence of learning (before/after episodes).
- `dashboard.html` renders live episode replay and reward curves.
- Mini-blog published on HuggingFace (< 2 min read).
- Adaptive difficulty / weak-spot targeting tested.

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
