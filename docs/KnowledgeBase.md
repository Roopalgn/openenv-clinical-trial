# Knowledge Base — OpenEnv Hackathon
## Clinical Trial Designer Project

## 1. What OpenEnv Actually Is
OpenEnv is a framework for building RL training environments served as FastAPI apps.

Agent (LLM) ──sends action──► Environment.step() ──returns──► observation + reward
                                      ↑
                         YOUR world lives here
The contract:

class YourEnvironment(Environment):
    def reset() -> YourObservation      # start new episode, inject scenario
    def step(action) -> YourObservation # apply action, return obs + reward + done
    def state -> State                  # current episode state
Served via:

app = create_app(YourEnvironment, YourAction, YourObservation, env_name="your_env")
# → FastAPI with POST /reset, POST /step, GET /state, GET /schema, WS /ws
Deployed as: Docker-based HuggingFace Space using openenv.yaml

Trained via: HF TRL GRPO — agent generates rollouts against the live environment, gets reward signal, updates weights.

## 2. What Every Winner Had in Common
### The Non-Negotiable Pattern
1. Real world state (not just text)
2. Actions that change that state (real commands / real math)
3. Verification WITHOUT an LLM judge (system state, math, test pass/fail)
4. Curriculum (easy → hard, progressive difficulty)
5. Long episodes (15–100+ steps)
6. Clear reward variance (GRPO needs +high vs -low separation)
### The Single Most Important Rule
Ground truth must be objective. Either the pod is running or it isn't. Either the p-value is < 0.05 or it isn't. Either the books balance or they don't.

If you need an LLM to judge whether the agent succeeded, your environment is weak.

## 3. Past Winners — What They Built & Why They Won
### 🥇 1st Place — Kube SRE (kube-sre-gym)
Domain: Kubernetes Site Reliability Engineering

What it is: Agent receives a PagerDuty alert about a broken K8s cluster. Must diagnose and fix using real kubectl commands against a live GKE cluster.

Real-world grounding:

Live GKE cluster (not simulated)
Real kubectl commands execute against real pods
Real failure modes: OOMKill, CrashLoopBackOff, ImagePullBackOff, scale-to-zero
Real SRE workflow: triage → investigate → fix → verify
Verification (no LLM needed for core check):

Pod status is ground truth: Running or not
Restart counts, OOM flags are real K8s events
LLM judge used only as secondary confirmation layer
Reward structure:

Per-step: LLM judge score (-1.0 to +1.0) for SRE workflow quality
Repeat penalty: -0.15 per repeated command
Resolution bonus: +1.0 to +5.0 (efficiency-scaled, faster = higher)
Timeout: net -2.0 for failed episodes
Phase-order bonus: +0.2 for correct triage→investigate→fix→verify sequence
Curriculum:

Warmup (0.0–0.25): single easy faults (OOM, crashloop, image pull)
Beginner (0.25–0.40): medium faults (bad config, scale zero)
Intermediate (0.40–0.60): harder investigation required
Advanced (0.60–0.80): compound multi-fault scenarios
Expert (0.80–0.95): adversarial LLM-designed incidents across all 3 namespaces
Adversarial Designer:

Claude designs incidents targeting agent's tracked weak spots
Multi-fault scenarios spread across namespaces with red herrings
Scenarios must be solvable within step budget (inject/fix pairs validated)
Judge personas (scale with difficulty):

Junior (< 0.4): lenient, gives hints
Senior (0.4–0.7): standard SRE expectations
Principal (> 0.7): strict, penalizes inefficiency
Key insight that won it: Environment co-evolved with the agent. Training exposed bugs in the command parser, judge truncation, and health check race conditions. Fixing them made both the environment and agent better.

Episode length: 15–25 steps (scales with difficulty)

Model: Qwen3-1.7B + LoRA, GRPO with 8 parallel rollouts

### 🥈 2nd Place — Bio Experiment Environment
Domain: Biological Research / Single-Cell Genomics

What it is: Agent plans a biological experiment pipeline step-by-step. Hidden ground truth (true DE genes, true effect sizes, true cell populations) is never revealed. Agent must design experiments that would discover it.

Real-world grounding:

Real bioinformatics tools: Scanpy, Seurat, DESeq2, Monocle3, SCENIC (all real)
Real scientific workflow: collect → QC → normalize → cluster → DE → conclude
Real lab constraints: budget ($80K–$120K), time (120–180 days), action costs
Literature-backed scenarios with real DOIs and true DE genes with log2FC values
4 real biological scenarios: cardiac disease, hematopoiesis, perturbation, biomarker validation
Verification:

Prerequisite rules are hard constraints (can't run DE before normalization — real science)
Budget/time math is ground truth
Terminal reward: conclusions compared against hidden ground truth markers/mechanisms
Calibration score: how well agent's claims match true biology
Reward structure (decomposed):

R_t = r_validity(0.3) + r_ordering(0.2) + r_info_gain(0.4) + r_efficiency(0.3) 
      + r_novelty(+0.1) + r_penalty(-0.15/violation) + shaping(γ=0.99)
Terminal reward adds: pipeline completeness (3.0), calibration (4.0), efficiency (1.0), overconfidence penalty (-0.5/wrong high-confidence claim)

POMDP structure:

Hidden: true cell populations, true DE genes, technical noise, failure conditions
Visible: task spec, pipeline history, resource usage, intermediate outputs, discovered markers
Episode length: Up to 30 steps

Key insight: Decomposed reward makes it easy to debug and train against. Each component is independently verifiable.

### 🥉 3rd Place — EcomRLVE
Domain: E-commerce Shopping Assistant

What it is: Agent helps a simulated customer (LLM-driven persona) find products, manage cart, handle returns. Uses real 2M product catalog (Amazon dataset) indexed with FAISS.

Real-world grounding:

2M real Amazon products with FAISS HNSW index (3.4GB, ~10ms search)
Real e-commerce tools: catalog.search, cart.add, order.list, return.initiate
Real return policies with eligibility windows
Persona-driven customer simulator with hidden preferences
Reward:

r_total = w_task × r_task + w_eff × r_eff + w_hall × r_hall
r_task = clip(0.55 × r_rank + 0.35 × r_constraints + 0.10 × r_oos, -1, 1)
8 environment types: Product Discovery, Substitution, Cart Building, Return+Replacement, Order Tracking, Policy QA, Bundle Planning, Multi-Intent Journey

Episode length: Up to 14 turns

### Finalist — VRAM (Voyager-VRAM)
Domain: Workplace Project Management / Memory

What it is: Agent manages a 6-week software project across 31 tools (Email, Slack, Calendar, Drive, Sheets, Notes, Meta-Search). Hidden state includes stale spreadsheets, chat-only constraints, changed deadlines.

Key innovation: Voyager architecture — Skill Library (reusable tool sequences), Working Memory (structured within-episode state), Episodic Memory (cross-episode learning).

Training: Expert Iteration — Best-of-4 rejection sampling + SFT × 3 rounds

Result: 21% improvement (5.75 vs 4.74 shaped reward) before any training, just from architecture.

## 4. OpenEnv Technical Requirements (Minimum to Submit)
- Use OpenEnv v0.2.1 (openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.1)
- Minimal training script using Unsloth or HF TRL in Colab
- Mini-blog on HuggingFace or mini-video on YouTube (< 2 minutes)
- Deployed on HF Spaces as Docker app
- Judging weights:

- Environment Innovation (40%)
- Storytelling (30%)
- Showing Improvement in Rewards (20%)
- Reward + Training Script Setup (10%)
## 5. Our Project — Clinical Trial Designer
### Core Concept
Agent designs a clinical trial to detect a drug effect. The simulator holds hidden ground truth (true effect size, true side effect rate, true responder population). Agent must design a trial that would statistically detect it.

- **Theme:** #3.1 — World Modeling / Professional Tasks

Real-world grounding:

FDA trial design rules are real and codified (Phase I/II/III requirements)
Statistical power calculations are pure math (no LLM needed)
Trial simulation runs with hidden true parameters → p-value is ground truth
Clinical trial protocols follow established procedures (ICH E9, FDA guidance)
### The World (Hidden State)
When reset() is called, the simulator secretly sets:

class TrialGroundTruth:
    true_effect_size: float          # e.g. 0.23 (23% tumor reduction)
    true_side_effect_rate: float     # e.g. 0.08 (8% serious adverse events)
    true_responder_population: str   # e.g. "BRCA1+ only" (agent doesn't know this)
    true_mechanism: str              # e.g. "inhibits VEGF pathway"
    true_dose_response: dict         # dose → effect curve (hidden)
    placebo_response_rate: float     # background noise
    dropout_rate: float              # patients who leave trial
Agent never sees this. It must design a trial that would detect it.

### Agent Actions (What the Agent Does)
class TrialAction:
    action_type: ActionType  # one of the actions below
    parameters: dict
    justification: str
    confidence: float  # 0.0–1.0
Action vocabulary:

Phase	Action	Real-world analog
Design	set_primary_endpoint	Choose what to measure (OS, PFS, ORR)
Design	set_sample_size	Power calculation → n patients
Design	set_inclusion_criteria	Who can enroll
Design	set_exclusion_criteria	Who is excluded
Design	set_dosing_schedule	Dose, frequency, cycle length
Design	set_control_arm	Placebo vs standard of care
Design	set_randomization_ratio	1:1, 2:1, etc.
Design	set_blinding	Open-label, single-blind, double-blind
Phase I	run_dose_escalation	3+3 design, find MTD
Phase I	observe_safety_signal	Read adverse event data
Phase I	estimate_effect_size	Estimate from Phase I data
Phase II	run_interim_analysis	Check futility/efficacy at 50% enrollment
Phase II	modify_sample_size	Adaptive design adjustment
Phase II	add_biomarker_stratification	Enrich for responders
Regulatory	submit_to_fda_review	Check protocol compliance
Regulatory	request_protocol_amendment	Change design mid-trial
Analysis	run_primary_analysis	Final statistical test
Analysis	synthesize_conclusion	Write trial conclusion
### Verification (No LLM Judge Needed)
1. Statistical Power — pure math

from scipy.stats import norm

def calculate_power(effect_size, n, alpha=0.05):
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = effect_size * sqrt(n/2) - z_alpha
    return norm.cdf(z_beta)

# If power < 0.80 → underpowered → reward penalty
# Agent must estimate effect_size from Phase I data (hidden true value)
2. FDA Rule Compliance — hard rules (binary pass/fail)

FDA_RULES = {
    "phase_ii_min_n": 100,
    "primary_endpoint_must_be_prespecified": True,
    "interim_analysis_requires_alpha_spending": True,
    "randomization_required_for_phase_iii": True,
    "safety_monitoring_committee_required": True,
    "informed_consent_required": True,
}
# Each rule is a hard check — no LLM needed
3. Trial Simulation — run it with hidden truth

def simulate_trial(design, ground_truth):
    # Sample patients from true population
    # Apply true effect to treatment arm
    # Apply placebo response to control arm
    # Add dropout, noise, adverse events
    # Run pre-specified statistical test
    # Return: p_value, confidence_interval, adverse_event_rate
    
    p_value = run_statistical_test(treatment_outcomes, control_outcomes)
    success = p_value < design.alpha  # ground truth: did it work?
    return TrialResult(p_value, success, adverse_events)
4. Budget — math

cost = n_patients * cost_per_patient + site_costs + regulatory_fees
over_budget = cost > trial_budget  # binary
### Reward Structure
Per-step rewards:

Component	Verification	Weight
FDA rule compliance	Hard rule engine	+0.3 per rule passed
Valid action sequence	Prerequisite check	+0.2
Information gain from Phase I	Bayesian update quality	+0.4
Budget efficiency	Math	+0.1
Soft violation penalty	Rule engine	-0.15 each
Terminal rewards (when trial simulation runs):

Component	Verification	Weight
Trial detects true effect (p < 0.05)	Simulation math	+5.0
Statistical power ≥ 0.80	Formula	+2.0
All FDA rules pass	Rule engine	+2.0
Correct responder population identified	Hidden state match	+3.0
Budget under limit	Math	+1.0
Interim analysis catches futility early	Simulation	+1.0 bonus
Underpowered design	Formula	-2.0
Wrong primary endpoint	Domain rules	-1.5
Overconfident wrong claims	Calibration check	-0.5 each
Reward variance for GRPO:

Successful trial: +8 to +14
Failed trial (wrong population): -2 to 0
Timeout / FDA rejection: -3
### Episode Structure (Long Horizon)
Phase I (20–30 steps):
  → dose_escalation × 6 cohorts
  → observe_safety_signal × 3
  → estimate_effect_size (Bayesian update)
  → decide: go/no-go to Phase II

Phase II (30–40 steps):
  → set_primary_endpoint
  → set_sample_size (power calculation)
  → set_inclusion_criteria (try to find responder population)
  → set_dosing_schedule
  → submit_to_fda_review
  → run_interim_analysis (at 50% enrollment)
  → modify_sample_size if needed
  → run_primary_analysis

Conclusion (5–10 steps):
  → synthesize_conclusion
  → Terminal reward fires
Total: 80–100 steps per episode

### Curriculum
Tier	Difficulty	What changes
Warmup	0.0–0.25	Large effect size (easy to detect), homogeneous population
Beginner	0.25–0.40	Medium effect, some noise
Intermediate	0.40–0.60	Small effect, need correct population enrichment
Advanced	0.60–0.80	Hidden responder subgroup, misleading Phase I signal
Expert	0.80–0.95	Tiny effect, high dropout, adaptive design required
Scenarios (4 to start, like Bio project)
Name	Disease	Challenge	True Effect
solid_tumor_chemo	Non-small cell lung cancer	Find EGFR+ subgroup	31% PFS improvement in EGFR+ only
autoimmune_biologic	Rheumatoid arthritis	Dose-response curve, find optimal dose	U-shaped response, 200mg optimal
cns_depression	Treatment-resistant depression	High placebo response masks drug effect	18% improvement over placebo
rare_disease_orphan	Rare pediatric metabolic disorder	Tiny n, adaptive design required	Large effect (Cohen's d = 1.2) but n < 50
### Hidden State Structure
class TrialLatentState:
    # Biology
    true_effect_size: float
    true_responder_criteria: List[str]   # e.g. ["BRCA1+", "age < 65"]
    true_dose_response: Dict[float, float]
    true_mechanism: str
    
    # Technical
    placebo_response_rate: float
    dropout_rate: float
    site_variability: float
    measurement_noise: float
    
    # Progress flags (18 milestones like Bio project)
    phase_i_complete: bool
    mtd_identified: bool
    effect_estimated: bool
    protocol_submitted: bool
    interim_complete: bool
    trial_complete: bool
    
    # Resources
    budget_remaining: float
    time_remaining_days: int
    patients_enrolled: int
### Key Design Decisions
Real statistical math — scipy.stats does the power calculations. No LLM.
FDA rules as hard constraints — ICH E9 guidelines encoded as rule engine (like Bio project's prerequisite rules).
Simulation is ground truth — trial either detects effect or doesn't. Same as KubeSRE's pod status.
Phase I → Phase II information flow — agent must use Phase I observations to update its Phase II design. This is the long-horizon planning challenge.
Hidden responder population — the hardest part. Agent must figure out that the drug only works in BRCA1+ patients by designing smart inclusion criteria. This is where the curriculum earns its keep.
Decomposed reward — like Bio project, each component is independently verifiable and debuggable.
## 6. Rules Learned from Winners
### Environment Design Rules
One clear success criterion — pod running, p < 0.05, books balance
Real tools/APIs — not mocked. Real kubectl, real scipy, real SQL
Prerequisite chains — can't run Phase II without Phase I (like Bio project's rule engine)
Reward variance — GRPO needs clear separation between good and bad episodes
No reward hacking — multi-layer verification (programmatic + optional LLM)
Environment must fight back — too-easy rewards cause plateaus (KubeSRE lesson)
Repeat penalty — prevents agent from spamming same action
### Training Rules
GRPO over PPO — better for sparse delayed rewards, no value function needed
8 parallel rollouts — gives GRPO enough variance to compute advantages
Curriculum is mandatory — cold start on hard problems = no learning signal
Fast-track advancement — 90%+ success rate → skip min_episodes requirement
Episode transcripts — save to JSONL for debugging and offline analysis
### Reward Rules
Timeout = net negative — wipe accumulated rewards, set to -2.0 total
Efficiency scaling — faster fixes get higher bonuses (prevents lazy solutions)
Phase-order bonus — reward correct workflow sequence
Overconfidence penalty — high-confidence wrong claims get penalized (Bio project)
Decompose rewards — makes debugging and training easier
### Pitfalls to Avoid
LLM-only verification — too slow, too expensive, too noisy
Too-generous rewards — agent finds plateau and stops improving
Static scenarios — agent memorizes, doesn't generalize
Single-fault only — too easy, no curriculum progression
Mocked tool responses — agent learns to exploit mock, not real behavior
Truncated observations — KubeSRE bug: judge was cutting off pods alphabetically
## 7. Tech Stack (Based on Winners)
Environment:     openenv-core[core] @ v0.2.1
Server:          FastAPI + uvicorn
Training:        HF TRL (GRPOTrainer) + vLLM colocate
Model:           Qwen3-1.7B or Qwen2.5-7B + LoRA (BF16)
Deployment:      Docker → HuggingFace Spaces
Compute:         H100 80GB (training) + GKE/cloud (environment)
Stats:           scipy.stats (power calculations)
### Training command pattern

# Terminal 1: Environment server
uv run server

# Terminal 2: GRPO training
python train.py --vllm-mode colocate --num-generations 8 --max-steps 100
## 8. Pitch Strategy (3 min)
Based on judging criteria (40% innovation, 30% storytelling, 20% reward improvement, 10% pipeline):

Minute 1 — Story (30% of score)

"A drug works. But only in 15% of patients. The FDA needs proof. How do you design a trial that finds those patients before you run out of money?"

Minute 2 — Environment Innovation (40% of score)

Show: hidden ground truth, statistical verification, FDA rule engine, Phase I → Phase II information flow

Minute 3 — Reward Curves + Demo (30% of score)

Show reward curve improving. Show agent learning to enrich for responder population. Show before/after: random inclusion criteria vs. learned BRCA1+ enrichment.
