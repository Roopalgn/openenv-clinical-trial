# OpenEnv Clinical Trial Designer

## Project Overview

- This project builds a clinical-trial-design RL environment using OpenEnv.
- Agent goal: design trial protocols that detect true treatment effects under realistic constraints.
- Core challenge: hidden world state (effect size, responder subgroup, safety profile, dropout, placebo effects).

## Why This Project

- It directly targets professional world-modeling tasks (OpenEnv Theme #3.1).
- It aligns with winner patterns from `docs/comparison.md`:
  - Objective verification
  - Reward decomposition
  - Progressive curriculum
  - Long-horizon episodes
  - Reproducible training and containerized deployment

## Target Outcomes

- Build an environment where agents learn valid clinical workflows.
- Improve reward and success metrics over training.
- Demonstrate robust behavior under harder scenarios and noisy observations.

## Planned Repository Structure

- `docs/`
  - `comparison.md`
  - `hack_info.md`
  - `KnowledgeBase.md`
  - `ROADMAP.md`
- `src/` (to be added)
  - `environment/`
  - `rules/`
  - `reward/`
  - `simulation/`
  - `curriculum/`
- `tests/` (to be added)
- `Dockerfile` (to be added)

## Must-Have Quality Gates Before Main Merge

- OpenEnv API compatibility (`reset`, `step`, `state`, `schema`).
- Objective graders and rule checks (not just LLM judgment).
- Decomposed rewards with documented weights.
- 3+ scenario tasks with increasing difficulty.
- Baseline + trained metrics captured.
- Docker build/run works and health ping responds.
- Documentation complete and coherent.

## Team Workflow

- Roadmap and push split: see `docs/ROADMAP.md`.
- Naming contract and non-rename list: see `docs/ROADMAP.md`.
- Merge to `main` only after both teammates approve checklist pass.

## On-Par-with-Winners Checklist

- Real consequences and hidden state.
- Multi-layer verification and anti-reward-hacking safeguards.
- Curriculum progression with mastery thresholds.
- Interpretable reward components.
- Clear storytelling with evidence of learning progression.
