# OpenEnv Clinical Trial Designer - Detailed Roadmap

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
  - Roopal: `feature/roopal-<push-number>-<topic>`
  - Suyash: `feature/suyash-<push-number>-<topic>`
- Merge strategy:
  - Both open PRs to `develop` first.
  - One integration PR from `develop` to `main` only after full checklist pass.
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

### Suyash: tasks

- Scaffold environment package and OpenEnv/FastAPI app entrypoints.
- Create reset/step/state skeleton with typed action/observation/state models.
- Add Dockerfile + health endpoint (`/ping`) and local run command.
- Add basic lint/test tooling.

### Joint gate before push

- Agree on action and observation schema names.
- Agree on API response envelope fields.
- Confirm container starts and ping works.

## Push 2

### Roopal: tasks

- Define reward decomposition spec and write formulas/weights doc.
- Draft scenario cards (4 initial clinical scenarios with difficulty labels).
- Define milestone map for long-horizon episode phases.

### Suyash: tasks

- Implement rule-engine checks (FDA constraints, prerequisite checks).
- Implement reward components as separate functions.
- Add structured episode logging format.

### Joint gate before push

- Validate reward keys are consistent across docs/code.
- Validate one full episode trace writes expected JSONL schema.

## Push 3

### Roopal: tasks

- Design curriculum progression policy and mastery thresholds.
- Write benchmark protocol (random policy baseline + scripted baseline).
- Define dashboard metrics table format.

### Suyash: tasks

- Implement curriculum controller and scenario randomization.
- Implement hidden-state generator and seeded reproducibility.
- Add safety checks for invalid action transitions.

### Joint gate before push

- Same seed reproduces same episode.
- Curriculum tier advancement logic tested.

## Push 4

### Roopal: tasks

- Prepare model-training runbook (GRPO settings, expected outputs, logs).
- Create evaluation report template and charts checklist.
- Draft mini-blog/2-minute demo structure.

### Suyash: tasks

- Build minimal train runner integration with environment.
- Add evaluation script (base vs trained comparison).
- Add runtime config via env vars.

### Joint gate before push

- End-to-end local dry run completes.
- Baseline metrics are generated and saved.

## Push 5

### Roopal: tasks

- Improve prompt/action instructions for better policy behavior.
- Tune reward weights based on diagnostics.
- Update README with observed baseline scores.

### Suyash: tasks

- Optimize simulator performance and caching.
- Harden error handling and edge-case validations.
- Add deterministic integration tests.

### Joint gate before push

- No schema drift from agreed contract names.
- Performance and stability pass threshold.

## Push 6

### Roopal: tasks

- Finalize storytelling assets (before/after episodes, failure-first narrative).
- Prepare judge-facing pitch notes aligned to scoring weights.
- Final proofreading of all docs for consistency.

### Suyash: tasks

- Final HF Space container hardening.
- Validate Docker build/run and startup health.
- Add final CI checks for tests/lint/format.

### Joint gate before push

- HF Space pass checklist complete.
- `main` merge readiness approved by both.

## 4. Pre-Merge Checklist (Must Finish Before Main)

- OpenEnv compatibility verified.
- Docker image builds cleanly.
- Container runs and responds on ping endpoint.
- Action/observation/state schemas locked and documented.
- At least 3+ task families with graders/verification.
- Decomposed reward with documented components and weights.
- Curriculum tiers and mastery progression implemented.
- Baseline and trained metrics generated.
- README includes motivation, setup, usage, task descriptions, baseline scores.
- Demo narrative prepared with evidence of learning.

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
