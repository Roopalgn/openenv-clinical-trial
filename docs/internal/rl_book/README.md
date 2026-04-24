# Reinforcement Learning from Scratch: Building an AI Clinical Trial Designer

## A Complete Guide for Python Programmers with Zero AI/ML Experience

**Author:** OpenEnv Clinical Trial Designer Team  
**Target Audience:** Python programmers who have never touched AI, ML, or RL  
**Project Reference:** OpenEnv Clinical Trial Designer  

---

## Table of Contents

| Chapter | Title | What You'll Learn |
|---------|-------|-------------------|
| 1 | [What is AI, Really?](chapter_01_what_is_ai.md) | AI, ML, Deep Learning — the family tree |
| 2 | [Reinforcement Learning: Teaching by Doing](chapter_02_reinforcement_learning.md) | The RL paradigm, agents, environments, rewards |
| 3 | [Language Models: The Brain of Our Agent](chapter_03_language_models.md) | What LLMs are, tokens, transformers, fine-tuning |
| 4 | [Our Mission: Designing Clinical Trials with AI](chapter_04_the_mission.md) | Why clinical trials, the problem we're solving |
| 5 | [The Environment: Where Our Agent Lives](chapter_05_the_environment.md) | OpenEnv, environments, observations, actions |
| 6 | [Hidden State: The Secret Truth](chapter_06_hidden_state.md) | Partial observability, ground truth, noisy observations |
| 7 | [The Reward System: Teaching Right from Wrong](chapter_07_rewards.md) | Reward decomposition, shaping, terminal rewards |
| 8 | [The Rule Engine: FDA Compliance](chapter_08_rules.md) | Hard constraints, prerequisites, compliance checking |
| 9 | [The Simulator: Running Virtual Trials](chapter_09_simulator.md) | Trial simulation, statistical power, p-values |
| 10 | [Curriculum Learning: Baby Steps to Expert](chapter_10_curriculum.md) | 5-tier curriculum, scenarios, difficulty scaling |
| 11 | [GRPO: How the Agent Actually Learns](chapter_11_grpo.md) | GRPO algorithm, policy gradients, advantages |
| 12 | [Training Infrastructure: LoRA, vLLM, and GPUs](chapter_12_training_infra.md) | LoRA, quantization, vLLM, GPU training |
| 13 | [The Full Pipeline: From Reset to Weight Update](chapter_13_full_pipeline.md) | End-to-end walkthrough of one training episode |
| 14 | [Evaluation and Comparison](chapter_14_evaluation.md) | Measuring progress, comparing agents |
| 15 | [Deployment: Sharing Your Work](chapter_15_deployment.md) | Docker, FastAPI, HuggingFace Spaces |
| 16 | [Lessons, Pitfalls, and Next Steps](chapter_16_lessons.md) | What we learned, common mistakes, where to go |

---

## How to Read This Book

1. **Chapters 1–3** build your AI/ML foundation from zero. If you don't know what a "model" is, start here.
2. **Chapters 4–6** introduce the clinical trial problem and how we model it as an RL environment.
3. **Chapters 7–10** deep-dive into each component: rewards, rules, simulation, curriculum.
4. **Chapters 11–13** cover the actual training: the algorithm, infrastructure, and full pipeline.
5. **Chapters 14–16** cover evaluation, deployment, and practical wisdom.

Every chapter includes:
- Real-world analogies (no jargon without explanation)
- Code from the actual project (not toy examples)
- "Why this decision?" boxes explaining design choices
- "Keyword Glossary" at the end of each chapter

---

## Prerequisites

- Basic Python (variables, functions, classes, loops, dicts)
- Willingness to read code
- No AI/ML/RL knowledge required — we build everything from scratch
