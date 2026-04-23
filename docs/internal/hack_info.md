Theme #1 - Multi-Agent Interactions
Environments for this theme involve cooperation, competition, negotiation, and coalition formation. Learning from these environments will enable agents to model the beliefs and incentives of others in partially observable settings. This drives theory-of-mind reasoning and emergent strategic behavior.
Expected Outcome: an environment that can be used to train multi-agent task handling in a LLM
Example environments: Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.
Theme #2 - (Super) Long-Horizon Planning & Instruction Following
You will build environments that require deep, multi-step reasoning with sparse or delayed rewards. After using these environments, the goal is to enable agents to decompose goals, track state over extended trajectories, and recover from early mistakes. The aim is to push beyond shallow next-token reasoning toward structured planning and durable internal representations. 
Expected Outcome: an environment that can capture and improve LLM behaviour on challenging long horizon tasks that need long running sessions beyond context memory limits. 
Example environments: (Think of OpenClaw workflows with Multi-turn tasks). Research-planning simulators, large-scale codebase refactoring tasks, strategic resource management worlds, long-horizon logistics optimization, extremely complicated long-horizon instruction following (e.g., 300 instructions scattered around).
Theme #3 - World Modeling
#3.1 Professional Tasks
Here you will develop environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts to arrive at the desired outcome. Learning from these environments will enable agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows. The goal is to strengthen causal reasoning and persistent world models.
Expected Outcome: an environment capturing nuances of a defined partially observable world and improve LLM interaction with it
Example environments: Dynamic browser/API ecosystems, enterprise applications, scientific workflow loops (papers → code → experiments), economic simulations with feedback, tool-discovery benchmarks.

#3.2 Personalized Tasks
Here we will develop an environment that offers real personalized task handling, imagine replying to personal messages or handling dinner conflicts due to work conflicts, replying to tough emails. Think any personal assistant tasks


Expected Outcome: An environment that gives the model a realistic simulation of handling personal tasks, conflicts and managing them as delegations

Example environments: Executive Assistant Meeting Planner, Dinner and drive planning, email and message replying, shopping, etc

Theme #4 - Self-Improvement
The focus here is to create environments where agents can learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula. Rather than optimizing fixed tasks, the goal is for agents to learn to drive their own capability growth. The objective is recursive skill amplification.
Expected Outcome: an environment for improving self-play of a LLM over a defined set of tasks
Example environments: Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

Theme #5: Wild Card - Impress Us!
We do not want to limit your focus if your idea doesn’t fit the boxes above, we want and WILL reward out of box tasks, please be creative but remember to add submissions that meaningfully add value to LLM training on a certain task. 

Guidelines for Problem Statement
It is NOT mandatory to choose the same problem statement as Round 1. Only choose the same problem statement if it aligns with the above provided Hackathon themes.
You can start working on your problem statement once you have finalized it. Post-training can be done onsite on 25th & 26th when you receive compute credits for HuggingFace.
Before the onsite, we suggest you work on building the environment, agent behaviours, reward model and evaluate if your work aligns with the judging criteria given below.


Judging Criteria
Minimum requirements: 
Usage of OpenEnv (latest release)
Show a minimal training script for your environment using Unsloth or HF TRL in Colab
Write a mini-blog on HuggingFace or mini-video on YouTube talking about your submission, <2 minutes
Your OpenEnv compliant environment should be hosted on Hugging Face Spaces.

Judging Overview
Evaluation: Teams will be scored based on the following criteria:
Environment Innovation (40%): Is the environment novel, creative, or challenging? Does it meaningfully test the agent’s behavior?
Storytelling (30%): Does the team clearly explain the problem, environment, and agent behavior? Is the demo engaging and easy to follow?
Showing Improvement in Rewards (20%): Does the demo provide observable evidence of training progress (reward curves, metrics, or before/after behavior)?
Reward and Training Script/Pipeline Setup (10%): Is the reward logic coherent, and does the pipeline produce meaningful improvement in the agent’s inference (how it acts in the environment)?

OpenEnv Hackathon - What Judges Look For

This guide tells you what makes a strong submission for the OpenEnv Hackathon (India 2026).
Read it before you start building, and again before you submit.

For the list of themes and example problems, refer to the top sections.

NOTE: Please remember only one submission per team. If you have multiple ideas, pick the best one and go for it. Please make sure that the URL link of your environment is submitted as judges will pull the environment from the URL to evaluate it. Changes or commits after the submission deadline will not be considered.

TL;DR

Build an environment that an LLM could actually be trained on to get measurably better at
something interesting. Then show that training. Then tell the story.

A messy but ambitious environment with real training evidence beats a polished but boring one.
Pick a problem that excites you (that energy comes through in the pitch).

Judging Criteria

Criterion: Environment Innovation
Weight: 40%
What it means:
Is the environment novel, creative, or genuinely challenging?
Does it meaningfully test agent behavior in a way that hasn't been done before?


Criterion: Storytelling & Presentation
Weight: 30%
What it means:
Can you clearly explain the problem, the environment, and what the agent learned?
Is the demo engaging and easy to follow for a non-technical audience?


Criterion: Showing Improvement in Rewards
Weight: 20%
What it means:
Is there observable evidence of training progress? Reward curves, before/after behavior,
comparison against a baseline -- anything that proves the agent learned something.


Criterion: Reward & Training Pipeline
Weight: 10%
What it means:
Is the reward logic coherent? Does the pipeline produce meaningful improvement in the trained
agent's behavior?


Minimum Submission Requirements

NOTE: These are non-negotiable. Submissions missing any of these are at a serious disadvantage.
Use OpenEnv (latest release). Build on top of the framework; don’t reinvent the wheel.
A working training script using Unsloth or Hugging Face TRL, ideally as a Colab notebook so judges can re-run it.
Evidence that you actually trained; at minimum, loss and reward plots from a real run.
A short writeup: a mini-blog on Hugging Face or a < 2 minute video on YouTube explaining what your environment does and what you trained, or a short slide deck of presentation. Please make sure that all materials are linked from your README file so that judges can access them easily.
Push your environment to a Hugging Face Space so it’s discoverable and runnable.
A README that motivates the problem, explains how the env works, and shows results.
README should have a link to the environment in the Hugging Face Space. It should also have all additional references to other materials (e.g. videos, blog posts, slides, presentations, etc.) that you want to include.
Please do not include big video files in your Env submission on HF Hub as we would like to have a small size for each env (Please use url as reference link to additional materials).

What Makes a Submission Stand Out

Pick an ambitious, original problem
The themes (problems) are deliberately open. Use them as launching pads, not boxes. Judges have seen a lot of chess, snake, tic-tac-toe, and grid-world clones. To score well on innovation,
you need a genuinely fresh angle. Some questions to ask yourself:
Does this environment exist to teach an LLM something it currently can’t do well?
Is the domain underexplored in RL/LLM training?
Could a researcher write a paper about training on this?

Design a reward signal that actually teaches
A great environment has a reward function that:
Provides a rich, informative signal (not just 0/1 at the end)
Captures something hard to measure in a clever way
Uses OpenEnv’s Rubric system thoughtfully (composable rubrics > monolithic scoring)
Is hard to game; an agent that exploits the reward without solving the task should not get high scores

Show real training, end to end
The bar isn’t “training script exists.” The bar is “training script runs against the environment, the
agent learns, and you can show it.” Concretely:
Your training loop should connect to your environment (not a static dataset)
Train long enough that the curves mean something
Compare a trained agent vs. a random/untrained baseline; quantitative and/or qualitative
Include the plots and numbers in your README and writeup

Make your plots readable
Reviewers spend seconds, not minutes, on each plot. Help them out:
Label both axes (e.g. “training step” / “episode” on x, “reward” / “loss” on y) and include units where they apply
Save plots as .png or .jpg and commit them to the repo (don’t leave them only in a Colab cell or a deleted Wandb run) (if you ran via Wandb, please include the link to that specific run of your plots)
Embed the key plots in your README with a one-line caption explaining what each one shows If you have multiple runs (baseline vs. trained, ablations, etc.), put them on the same axes so the comparison is obvious

Tell a story, not an API doc
Your README, blog, and pitch should answer:
Problem) what capability gap or interesting domain are you targeting?
Environment) what does the agent see, do, and get rewarded for?
Results) what changed after training? Show it.
Why does it matter) who would care, and why?

A reviewer should be able to read your README in 3~5 minutes and want to try your
environment.

NOTE: If you have a video, HF post, or anything else interesting, please make sure that it’s linked
  from your README as a link.

Engineer it cleanly (table stakes)
Engineering quality matters less than ambition, but sloppy work hurts. Make sure you:
Use OpenEnv’s Environment / MCPEnvironment base classes properly
Respect the client / server separation (clients should never import server internals)
Follow the standard Gym-style API (reset, step, state)
Have a valid openenv.yaml manifest
Don’t use reserved tool names (reset, step, state, close) for MCP tools

Final Note

Judges are looking for environments that push the frontier of what we can train LLMs to do. Be
ambitious. Pick a problem you find genuinely interesting; that almost always produces better
work than chasing what you think judges want. Good luck.


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



github.com/SupreethRao99/veriRL
github.com/softsideof/cyber_range
github.com/sh4shv4t/Parlay
github.com/thevivekkelkar/smartcity-traffic
github.com/sharad0x/Sovereign-SRE-Gym
github.com/ARKAISW/multi-agent-trading-env
github.com/shriom17/MedFlow-OpenEnv

Where We Are Materially Behind
P0 Gaps
1. Our phase system is architected but not actually driving the environment
This is the largest code-level gap in the repo.

Evidence:

server/episode_manager.py initializes latent.episode_phase = "literature_review" on reset.
server/phase_detector.py classifies phases from actions.
server/rules/fda_rules.py checks action validity against latent.episode_phase.
server/simulator/output_generator.py builds available_actions from latent.episode_phase.
But server/simulator/transition_engine.py never updates latent.episode_phase.
tests/test_integration.py even documents and asserts that the episode phase stays literature_review throughout the episode.
Consequence:

The environment has a phase abstraction, but phase progression is not authoritative.
available_actions and FDA transition logic are effectively anchored to the initial phase.
A large part of the designed action/state space is dead or underused.
This is the kind of issue that strong tests can accidentally preserve if the tests encode the simplification.

2. The scenario descriptions promise richer biology than the latent state actually contains
Evidence:

server/curriculum/scenarios.py describes biomarker enrichment, dose-response structure, placebo effects, and rare-disease constraints.
But in server/episode_manager.py, reset builds latent state with:
true_responder_population="all"
true_responder_criteria=[]
true_dose_response={}
true_mechanism="unknown"
server/simulator/output_generator.py has code paths to reveal dose-response and responder hints, but those fields are mostly empty or trivial.
Consequence:

Our code advertises a richer clinical-trial environment than it actually simulates.
Biomarker and dose-related actions have much less real semantic payoff than the code structure suggests.
The top same-domain competitor, OpenENV-Hackathon, is much deeper here because its latent/task generation actually fills those concepts in.
3. Our adversarial curriculum is underfed and therefore weaker than it looks
Evidence:

server/curriculum/adversarial_designer.py expects episode history with things like true_effect_size, dropout_rate, and biomarker usage.
In server/episode_manager.py, when we feed failure history into analyze_failures, we pass only {success, true_effect_size=None, dropout_rate=None}.
No biomarker-use signal is passed.
Consequence:

The adversarial designer cannot actually learn the weak spots it claims to target.
small_effect and high_dropout counters do not get meaningful signal.
In practice the expert-tier adversarial logic is much weaker than the module names suggest.
4. Our training runner is not a real competitive training pipeline yet
Evidence:

train.py instantiates GRPOTrainer.
But it never calls trainer.train().
The comment in train.py explicitly says the manual rollout loop is what is driving logging and that trainer.train() "can be called instead".
_grpo_reward_fn() is defined but not actually integrated into a real trainer loop.
The manual loop in rollout_func() uses model generation and environment stepping, but it is much closer to rollout/evaluation logging than to actual trainer-backed optimization.
Consequence:

We have training scaffolding, not a training system that is on par with EcomRLVE-Gym, OpenENV-Hackathon, or veriRL.
This is a major competitive weakness because those repos actually wire reward functions, datasets, and trainer flows more credibly.
5. Our judge is stage-insensitive in a way that can distort learning
Evidence:

server/judge.py checks budget, power, and p-value on every step using hidden latent truth.
Early design actions can therefore be judged against end-state statistical success criteria before the agent has even finished the trial workflow.
Consequence:

The feedback loop is partly hindsight-based instead of phase-appropriate.
This is weaker than environments where the reward/judge logic is tied more tightly to stage-specific progress or realistic task completion.
6. We only support one active session through the main runtime
Evidence:

server/environment.py sets SUPPORTS_CONCURRENT_SESSIONS = False.
server/app.py uses one global _manager = EpisodeManager().
Consequence:

We are weaker than OpenENV-Hackathon, veriRL, MedFlow-OpenEnv, and others that are coded for better session isolation or concurrent workflows.
P1 Gaps
7. Our resource physics and observation noise are much simpler than the top environments
Evidence:

server/noise_model.py randomizes only a few scalar ranges.
server/simulator/transition_engine.py mostly applies fixed cost/time constants plus a few scalar degradations.
server/simulator/trial_simulator.py uses lightweight statistical proxies.
Compared with:

OpenENV-Hackathon has richer latent technical/biological state, hard and soft rule propagation, and modality-specific output generation.
cyber_range has explicit network topology, alerts, forensics, and attacker progression.
veriRL uses real external evaluators rather than lightweight proxies.
8. We have breadth of module names, but not always breadth of realized behavior
The strongest same-domain comparison is OpenENV-Hackathon.

That repo has:

task generation
procedural scenario generation
latent biology state
technical state
hard and soft rules
detailed transition engine
detailed output generator
decomposed reward with shaping and terminal calibration
Our repo has analogous module names, but many of those paths are thinner in semantics. The architecture is there. The content density is not.

9. Some of our tests currently validate simplified behavior that should eventually change
The clearest example is phase progression.

That means we should treat the test suite as a strength, but also as something that needs to evolve with the simulator rather than just defend the current behavior.

Competitor-by-Competitor Comparison
OpenENV-Hackathon
This is the most important direct comparator because it solves a closely related hidden-state scientific planning problem using a deeper implementation.

Where they are stronger:

server/hackathon_environment.py has a richer latent/observed loop.
server/tasks/generator.py plus server/tasks/procedural_generator.py give them more scenario richness.
server/rules/engine.py models prerequisites, resource constraints, redundancy, causal validity, and tool compatibility.
server/simulator/transition.py and server/simulator/output_generator.py realize the pipeline in more detail.
server/rewards/reward.py is substantially richer and more stage-aware.
training_script.py is much closer to a real training system than our train.py.
Where we are stronger:

We are easier to reason about.
We have stronger automated verification discipline.
Our API/dashboard/logging shell is simpler and easier to maintain.
Bottom line:

They are ahead on environment depth.
We are ahead on code trustworthiness.
EcomRLVE-Gym
Where they are stronger:

Multi-environment platform, not just one environment.
Strong server/openenv.py orchestration: env selection, adaptive difficulty, tool execution, seen-set tracking, user simulator, terminal reward composition.
rewards/verifiers.py contains deep environment-specific deterministic verification logic.
difficulty/adaptive.py is a more fully realized adaptive system than ours.
training/grpo.py is a more credible GRPO integration.
Where we are stronger:

Far better test coverage.
Lower complexity and lower onboarding cost.
Cleaner small-team maintainability.
Bottom line:

They are the broadest environment/training platform here.
We are much more disciplined but materially less complete.
cyber_range
Where they are stronger:

cyber_environment.py exposes a true multi-tool interaction surface.
network_simulator.py and attack_engine.py encode a living world, not just scalar latent variables.
attack_designer.py and cyber_judge.py make curriculum and judge layers deeper than ours.
Reward shaping is tuned for actual RL signal variance.
Where we are stronger:

Better typing and cleaner modular boundaries.
Better test posture.
Less dependency on external judge behavior for core correctness.
Bottom line:

They are ahead on realism and agent interaction design.
We are ahead on reliability engineering.
veriRL
Where they are stronger:

Real tool-backed evaluation in server/evaluator.py.
Multi-file environment and concurrent sessions in server/verirl_env_environment.py.
Strong task packaging and real-world feedback loop.
Very good tests.
Where we are stronger:

Easier API/demo shell.
Simpler local reasoning for environment state.
Bottom line:

Their evaluator is much more grounded than our simulator.
They are ahead overall because their grading surface is real, not mostly synthetic.
kube-sre-gym
Where they are stronger:

Real backend interaction through Kubernetes-oriented modules.
Adversarial incident design is richer and more operationally grounded.
Judge and scenario generation encode workflow-aware incident response.
Where we are stronger:

Determinism, reproducibility, and testability.
Lower operational dependency footprint.
Bottom line:

They trade determinism for realism.
For demos and benchmark storytelling, that realism is a competitive asset we do not yet match.
Parlay
Where they are stronger:

Game-theoretic reward design is sharper than ours.
Negotiation scoring is more mathematically closed-form.
WebSocket session flow is coherent.
Where we are stronger:

Better test depth.
Better API/product shell.
More explicit simulator/rules/reward modularization.
Bottom line:

Same maturity band overall.
They are stronger in domain-specific reward math.
We are stronger in verification and infrastructure hygiene.
MedFlow-OpenEnv
Where they are stronger:

Dynamic arrivals, doctor scheduling, beds, and queue-state simulation are concretely implemented.
Where we are stronger:

Far stronger tests.
Better code organization.
Stronger logging, curriculum, and rules decomposition.
Bottom line:

We are ahead overall.
smartcity-traffic
Where they are stronger:

Explicit multi-agent / federated baseline story.
Where we are stronger:

Everything around verification, modularity, API, and engineering discipline.
Bottom line:

We are clearly ahead overall.
multi-agent-trading-env
Where they are stronger:

Frontend polish and dashboard surface.
Where we are weaker or stronger:

Their backend is currently incomplete from a codebase perspective: multiple files import env.trading_env.TradingEnv, but that module is absent from the repo tree I inspected.
That makes our repo substantially stronger in backend completeness and trustworthiness.
Bottom line:

They may demo better visually, but our codebase is more complete.
Sovereign-SRE-Gym
Where they are stronger:

Interesting multi-agent delegation concept.
Where we are stronger:

Simulator maturity
rules/reward depth
tests
API maturity
maintainability
Bottom line:

We are clearly ahead overall.
The Most Important Strategic Insight
Our repo is not losing because the architecture is wrong.

It is losing because:

the phase system is not authoritative,
the latent biology is under-instantiated,
the adversarial curriculum is underfed,
and the training loop is not fully wired.
That is good news.

It means the fastest path forward is not a rewrite. It is semantic completion of the existing architecture.

