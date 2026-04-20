# Project Status

> **Convention:** Every change to this project is logged here with date, branch, and description. Update this file in every commit.

---

## Change Log

### 2026-04-20

| Time | Branch | Change | Files |
|------|--------|--------|-------|
| — | `main` | Created team roadmap | `docs/ROADMAP.md` |
| — | `main` | Created root project guide | `README.md` |
| — | `main` | Formatting improvements | `docs/comparison.md`, `docs/hack_info.md`, `docs/KnowledgeBase.md` |
| — | `main` | Added Section 0 gap analysis (15 gaps G1–G15), gap fix tasks per push, PR names, pre-merge checklist (21 items) | `docs/ROADMAP.md` |
| — | `main` | Added LOCK statement to Push 1 joint gate | `docs/ROADMAP.md` |
| — | `Roopal` | **[Push 1] Roopal: scaffold docs, architecture, phase workflow** | See below |
| — | `Roopal` | Created problem statement with judging alignment | `docs/problem_statement.md` |
| — | `Roopal` | Created evaluation criteria and acceptance metrics | `docs/evaluation_criteria.md` |
| — | `Roopal` | Created demo story arc (3-min pitch, 4 acts) | `docs/story_arc.md` |
| — | `Roopal` | Created phase-aware workflow with bonus/penalty rules | `docs/phase_workflow.md` |
| — | `Roopal` | Created ARCHITECTURE.md with system diagram and data models | `ARCHITECTURE.md` |
| — | `Roopal` | Rewrote README with motivation, environment, rewards, curriculum, setup | `README.md` |
| — | `Roopal` | Rewrote KnowledgeBase.md as progressive textbook (basics → project) | `docs/KnowledgeBase.md` |
| — | `Roopal` | Updated project status to log all changes | `docs/project status.md` |
| — | `Roopal` | Enhanced all Push 1 docs with explicit winner-inspired patterns | See below |
| — | `Roopal` | ARCHITECTURE.md: added TransitionEngine, OutputGenerator, AdversarialDesigner, judge personas, episode logger, JSONL transcripts (from KubeSRE + Bio) | `ARCHITECTURE.md` |
| — | `Roopal` | evaluation_criteria.md: added action diversity metric (VRAM), steps-to-completion (KubeSRE), component trends (Bio), capability radar chart (VRAM) | `docs/evaluation_criteria.md` |
| — | `Roopal` | phase_workflow.md: added KubeSRE→clinical mapping table, judge persona scaling per tier, recovery bonus for protocol amendments | `docs/phase_workflow.md` |
| — | `Roopal` | problem_statement.md: added domain references (ICH E9, DiMasi 2016, Wong 2019), winner precedent column, existing work comparison | `docs/problem_statement.md` |
| — | `Roopal` | story_arc.md: added action heatmap (VRAM), capability radar (VRAM), component trends (Bio), environment co-evolution narrative (KubeSRE) | `docs/story_arc.md` |
| — | `Roopal` | **[Push 2] Roopal: reward spec, scenarios, shaping function** | See below |
| — | `Roopal` | Created reward decomposition spec: 8 per-step components + 7 terminal components + potential-based shaping φ(s) with γ·(φ(s')−φ(s)) | `docs/reward_spec.md` |
| — | `Roopal` | Created 4 scenario cards with hidden ground-truth values, curriculum scaling tables, programmatic verification functions per scenario | `docs/scenario_cards.md` |
| — | `Roopal` | Created milestone map: 18 milestones across 3 macro-phases, step budgets per tier, velocity metrics, φ(s) integration | `docs/milestone_map.md` |
| — | `main` | Cherry-picked 5 spec/contract files to main for Suyash | `ARCHITECTURE.md`, `docs/reward_spec.md`, `docs/scenario_cards.md`, `docs/milestone_map.md`, `docs/phase_workflow.md` |
| — | `Roopal` | Rebased onto updated main, force-pushed | All files |
| — | `Roopal` | **[Push 3] Roopal: curriculum policy, verification spec, phase scoring** | See below |
| — | `Roopal` | Created curriculum progression policy: 5 tiers, per-scenario mastery tracking, weak-spot targeting (70/30), fast-track advancement, noise scaling table | `docs/curriculum_policy.md` |
| — | `Roopal` | Created benchmark protocol: random + scripted baselines, execution commands, output JSON schema, pitch integration table | `docs/benchmark_protocol.md` |
| — | `Roopal` | Created dashboard metrics format: 5 panels (replay, reward curves, curriculum, scenarios, radar), CSV/JSONL schemas, data flow diagram | `docs/dashboard_metrics.md` |
| — | `Roopal` | Created multi-layer verification spec (G2): L1 programmatic ground-truth, L2 rule-engine constraints, L3 optional LLM judge, conflict resolution matrix | `docs/verification_spec.md` |
| — | `Roopal` | Enhanced phase_workflow.md (G13): added 10×10 transition matrix, tier penalty multipliers, phase step budgets, expanded _detect_phase() with context-dependent patterns, compute_ordering_reward(), warmup hint system | `docs/phase_workflow.md` |

---

## Current Branch Status

| Branch | Last Push | Status |
|--------|-----------|--------|
| `main` | Spec files for Suyash (ARCHITECTURE, reward, scenarios, milestones, phase workflow) | Up to date |
| `Roopal` | Push 3 Roopal tasks complete | All 5 tasks done |
| `Suyash` | Not started | Pending: curriculum controller, judge, phase detection |

---

## Compute Timeline

- **Now → April 24:** Build environment, reward logic, curriculum, rule engine, simulator, OpenEnv server, docs, dashboard. No GPU needed.
- **April 25–26 onsite:** Post-training with HuggingFace compute credits. Run GRPO training, generate reward curves, before/after episodes, baseline scores for pitch.

---

## Next Milestones Before Merge to Main

- Finalize environment schemas (action, observation, state) and lock naming contract.
- Implement objective graders and decomposed reward components.
- Add curriculum tiers and benchmark/evaluation scripts.
- Validate container build/run and ping health endpoint.
- Produce baseline metrics and training-improvement evidence.

