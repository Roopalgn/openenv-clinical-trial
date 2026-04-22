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
| — | `Roopal` | **[Push 4] Roopal: training runbook, Colab notebook, mini-blog draft** | See below |
| — | `Roopal` | Created training runbook: GRPO hyperparams, system prompt, step-by-step onsite procedure, expected timeline, troubleshooting table, checkpoint strategy | `docs/training_runbook.md` |
| — | `Roopal` | Created evaluation report template: baseline comparison table, per-scenario breakdown, component trends, curriculum timeline, charts checklist with specs | `docs/evaluation_report_template.md` |
| — | `Roopal` | Created HF mini-blog draft (G12): 6-section structure (problem → env → reward → results → what learned → try it), publishing checklist | `docs/mini_blog_draft.md` |
| — | `Roopal` | Created Colab training notebook (G6): 9 cells — install deps, env connection, Unsloth model load, reward function, dataset, GRPO config, training, evaluation, HF Hub upload, reward curve plot | `train_colab.ipynb` |
| — | `Roopal` | **[Push 5] Roopal: reward tuning, adaptive difficulty spec, dashboard UI** | See below |
| — | `Roopal` | Improved system prompt with per-action guidance, decision tips, and workflow explanation. Added reward weight tuning guide with diagnostic procedure + decision matrix | `docs/training_runbook.md` |
| — | `Roopal` | Updated README with expected baseline scores table (random vs scripted vs trained), key pitch contrasts, and updated checklist (G4 + G14) | `README.md` |
| — | `Roopal` | Created adaptive difficulty spec (G4): mastery detection, 4-step hardening schedule per scenario × axis, weak-spot targeting with FailureAnalyzer, 5 compound challenges at Expert tier, solvability guarantee | `docs/adaptive_difficulty_spec.md` |
| — | `Roopal` | Built dashboard.html (G14): 6-panel single-page dashboard (episode replay, reward curves, curriculum progression, scenario breakdown, capability radar, action log). Canvas-drawn charts, WebSocket live updates, demo data generator for offline pitch | `dashboard.html` |
| — | `Roopal` | **[Push 6] Roopal: storytelling, mini-blog, final docs** | See below |
| — | `Roopal` | Created judge-facing pitch notes: timing breakdown aligned to 40/30/20/10 scoring weights, talking points per axis, top-10 Q&A prep, demo flow, pitch don'ts | `docs/pitch_notes.md` |
| — | `Roopal` | Created storytelling assets: 3 scripted episodes (failure/breakthrough/mastery), side-by-side comparison card, "aha moment" highlight, narrative arc summary | `docs/storytelling_assets.md` |
| — | `Roopal` | Finalized mini-blog (G12): tightened to ~600 words, publish-ready format with clear placeholders for post-training data, streamlined sections | `docs/mini_blog_draft.md` |
| — | `Roopal` | Finalized ARCHITECTURE.md (G10): H100 memory budget, hardware requirements table, complete system diagram (all components from Push 1-5 reflected), checkpoint upload instructions, post-training results template | `ARCHITECTURE.md` |
| — | `Roopal` | Proofread README: updated documentation links (all 18 docs listed), fixed reward weights to match tuned values, added Live Dashboard section, updated Team section | `README.md` |

### 2026-04-21

| Time | Branch | Change | Files |
|------|--------|--------|-------|
| — | `main` | Merged Roopal branch (Push 1–6 docs/specs) into main | 20 files |
| — | `Suyash` | Suyash PR #8: Complete environment engine — Push 1-6 (models, FastAPI, rule engine, reward, noise model, curriculum, judge, train.py, eval, plot, CI, Docker, tests) | 50 files, 8203 lines |
| — | `main` | PR review: found 3 issues (shaping not wired, curriculum not advancing, adversarial not wired) | — |
| — | `Suyash` | Suyash fix commit: wired shaping_bonus, advance_curriculum, AdversarialDesigner into episode_manager; fixed _efficiency_reward hardcoded budget | `server/episode_manager.py`, `server/reward/reward_computer.py`, `Dockerfile` |
| — | `main` | Merged Suyash PR #8 into main after fix verification | 50 files |

### 2026-04-22

| Time | Branch | Change | Files |
|------|--------|--------|-------|
| — | `main` | Integration smoke test on merged codebase: 249/249 tests pass, ruff lint clean, all 6 endpoints verified live (/ping, /reset, /step, /state, /schema, /transcripts) | — |
| — | `main` | Updated project status with full merge history | `docs/project status.md` |

---

## Current Branch Status

| Branch | Status |
|--------|--------|
| `main` | **FULLY MERGED** — all Roopal docs (Push 1–6) + all Suyash code (Push 1–6) |
| `Roopal` | Archived — merged to main |
| `Suyash` | Archived — merged to main via PR #8 |

---

## Integration Test Results (2026-04-22)

| Check | Result |
|-------|--------|
| `pytest tests/` | 249/249 passed (11.45s) |
| `ruff check .` | All checks passed |
| `/ping` | 200 `{"status": "ok"}` |
| `/reset` (seed=42) | 200, warmup scenario, actions available |
| `/step` (run_dose_escalation) | 200, reward=-2.0 (correctly blocked by FDA rules) |
| `/state` | 200, tier=0, scenario=solid_tumor_chemo_warmup |
| `/schema` | 200, TrialAction + TrialObservation schemas |
| `/transcripts` | 200, JSONL transcript data |

---

## Compute Timeline

- **Now → April 24:** Final polish, pitch rehearsal, HF Space prep, blog finalization. No GPU needed.
- **April 25–26 onsite:** GRPO training with HuggingFace H100 credits. Generate reward curves, before/after episodes, baseline scores for pitch.

---

## Remaining Before Hackathon (Apr 25)

- [ ] Create HuggingFace Space repo and push placeholder
- [ ] Docker build test locally (if Docker available)
- [ ] Rehearse 3-min pitch with dashboard demo
- [ ] Publish mini-blog draft to HF
- [ ] Fill evaluation_report_template.md after onsite training

