# Hackathon Themes and Event Information

## Theme #1 - Multi-Agent Interactions

- Environments for this theme involve cooperation, competition, negotiation, and coalition formation.
- Learning from these environments enables agents to model beliefs and incentives of others in partially observable settings.
- This drives theory-of-mind reasoning and emergent strategic behavior.
- **Expected Outcome:** an environment that can be used to train multi-agent task handling in an LLM.
- **Example environments:** market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.
- **Sub-themes with bonus prizes:**
  - **Fleet AI. Scalable Oversight:** environments that train oversight agents to monitor, analyze, and explain behavior of other AI agents in complex multi-agent settings.
  - **Halluminate. Multi-Actor Environments:** realistic environments where an agent interacts with and manages multiple actors (agents) to discover and achieve a task.

## Theme #2 - (Super) Long-Horizon Planning and Instruction Following

- Build environments that require deep, multi-step reasoning with sparse or delayed rewards.
- Goal is to enable agents to decompose goals, track state over extended trajectories, and recover from early mistakes.
- Aim is to move beyond shallow next-token reasoning toward structured planning and durable internal representations.
- **Expected Outcome:** an environment that captures and improves LLM behaviour on challenging long-horizon tasks needing long-running sessions beyond context memory limits.
- **Example environments:** research-planning simulators, large-scale codebase refactoring tasks, strategic resource management worlds, long-horizon logistics optimization, extremely complicated long-horizon instruction following (e.g., 300 instructions scattered around).
- **Sub-themes with bonus prizes:**
  - **Scale AI:** long-horizon workflows for non-code business use cases in Sales, Project management, or HR and IT.
  - **Mercor:** an environment with capped/uncapped rewards where frontier model rewards scale with token output.

## Theme #3 - World Modeling

### 3.1 Professional Tasks

- Develop environments requiring real interaction with tools, APIs, or dynamic systems where models do real hard work instead of exploiting shortcuts.
- Learning from these environments should enable agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows.
- Goal is to strengthen causal reasoning and persistent world models.
- **Expected Outcome:** an environment capturing nuances of a defined partially observable world and improving LLM interaction with it.
- **Example environments:** dynamic browser/API ecosystems, enterprise applications, scientific workflow loops (papers -> code -> experiments), economic simulations with feedback, tool-discovery benchmarks.
- **Sub-themes with bonus prizes:**
  - **Scaler AI Labs. Multi-App RL Environment for Enterprise Workflows:** create RL environments to demonstrate complex workflows and business-rule nuances in large enterprises.

### 3.2 Personalized Tasks

- Develop environments for real personalized task handling.
- Example use cases include replying to personal messages, handling dinner/work conflicts, replying to tough emails, and other personal assistant tasks.
- **Expected Outcome:** an environment that gives the model a realistic simulation of handling personal tasks, conflicts, and delegations.
- **Example environments:** executive assistant meeting planner, dinner and drive planning, email/message replying, shopping, etc.
- **Sub-themes with bonus prizes:**
  - **Patronus AI. Consumer Workflows with Schema Drift:** multi-step consumer workflow environments where schemas, API contracts, and policies/rules change.

## Theme #4 - Self-Improvement

- Focus is to create environments where agents learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula.
- Instead of optimizing fixed tasks, agents should learn to drive their own capability growth.
- Objective is recursive skill amplification.
- **Expected Outcome:** an environment for improving self-play of an LLM over a defined set of tasks.
- **Example environments:** self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.
- **Sub-themes with bonus prizes:**
  - **Snorkel AI. Simulated Experts-in-the-Loop:** environment that simulates interactions with subject-matter experts with changing requirements/preferences.

## Theme #5: Wild Card - Impress Us!

- If ideas do not fit the boxes above, out-of-the-box tasks are welcome.
- Submissions should still meaningfully add value to LLM training on a specific task.

## Guidelines for Problem Statement

- It is **not mandatory** to choose the same problem statement as Round 1.
- Choose the same problem statement only if it aligns with the provided hackathon themes.
- You can start working on your problem statement once finalized.
- Post-training can be done onsite on 25th and 26th when compute credits are provided for HuggingFace.
- Before onsite, focus on building the environment, agent behaviours, reward model, and evaluating alignment with judging criteria.

## Judging Criteria

### Minimum requirements

- Usage of OpenEnv (latest release).
- Show a minimal training script using Unsloth or HF TRL in Colab.
- Write a mini-blog on HuggingFace or mini-video on YouTube talking about your submission (< 2 minutes).

### First Round Judging Overview

- **Pitch Format:** each team has 3 minutes to pitch and 2 minutes for Q&A (5 minutes total).
- **Evaluation criteria:**
  - **Environment Innovation (40%):** Is the environment novel, creative, or challenging? Does it meaningfully test agent behavior?
  - **Storytelling (30%):** Does the team clearly explain the problem, environment, and agent behavior? Is the demo engaging and easy to follow?
  - **Showing Improvement in Rewards (20%):** Does the demo show observable training progress (reward curves, metrics, before/after behavior)?
  - **Reward and Training Script/Pipeline Setup (10%):** Is reward logic coherent, and does the pipeline produce meaningful improvement in agent inference?
- Each evaluator judges about 10-15 teams and submits scores individually.
- Cerebral Valley aggregates all judges' scores to determine the top 15 finalist projects.

## Team Confirmation Email

- Hi Roopal Guha Neogi,
- Your solo/team spot at the Meta PyTorch OpenEnv Hackathon x Scaler School of Technology - Grand Finale is officially confirmed.
- This email serves as your official team ticket to the finale.

### Event details

- **Date:** 25-26 April 2026
- **Venue:** Scaler School of Technology, Electronic City, Bangalore

### Participation category

- Team of 2

### Team members

- **Team Member 1 (Team Leader):**
  - Name: Roopal Guha Neogi
  - Email: roopal.guhaneogi@gmail.com
- **Team Member 2:**
  - Name: Suyash Kumar
  - Email: suyashk102@gmail.com

### What to do right now

- Join the private Discord (MANDATORY): Join here.
- All major updates and announcements will be shared there first.
- Check the travel guide: Read Here.
- Travel guide includes venue details, directions, and nearby stay options.

### Important - Entry to Campus

- You must present this email at entry.
- Teams/participants without this email will not be allowed on campus.
- Going forward, all communication will be shared only with the team leader.

### Please carry for verification

- A valid government-issued ID.
- Your college/company ID used during registration.

### Entry policy notes

- Entry will not be permitted if details do not match registration.
- All team members must be individually registered in the system.
- New/unregistered members added to travel details will not be allowed on campus.
- Organisers reserve the right to deny entry if verification criteria are not met.

## Round 2 Theme Reveal Summary

- Multi-Agent Interactions
- Long-Horizon Planning and Instruction Following
- World Modeling across professional and personal tasks
- Self-Improving agent systems

These themes reflect real-world AI environment design and agent behavior that the hackathon evaluates.

## Submission Design Expectations

- Choose one or more themes and design your own problem statement.
- Simulate realistic scenarios, enable meaningful agent interaction, and support measurable outcomes.

As part of submission, clearly define:

- The **problem statement**
- The **environment** in which the agent(s) operate
- The **capabilities** of the agent(s)
- The **tasks** to be performed
- The **reward model/evaluation logic**
- The **post-training or self-improvement strategy**

## Recommendation for High Scores

- Define clear, structured tasks and environments.
- Incorporate robust evaluation and reward mechanisms.
- Reflect real-world complexity aligned with OpenEnv principles.

## Immediate Next Step

- Begin refining design and evaluation right away.
- Training and implementation happen onsite with provided compute credits.
