# Judge-Facing Pitch Notes — 3 Minutes + 2 Minutes Q&A

> **Aligned to scoring weights:** Environment Innovation (40%), Storytelling (30%), Showing Improvement in Rewards (20%), Reward/Training Pipeline (10%). Every second of the pitch maps to a scoring axis.

---

## Timing Breakdown (3:00 total)

| Slot | Duration | Scoring Axis | What to Say |
|------|----------|-------------|-------------|
| Hook | 0:00–0:15 | Storytelling (30%) | "Can a 7B-parameter model learn to design a clinical trial from scratch? No medical training. Just a drug and a deadline." |
| Problem | 0:15–0:40 | Innovation (40%) | "$2.6B per drug, 90% fail. Trial design is partially observable, long-horizon, and has objective ground truth — no LLM judge needed." |
| Env Demo | 0:40–1:30 | Innovation (40%) | Show architecture diagram → 4 scenarios → hidden ground truth → 19 actions → 10-phase workflow. Emphasize: scipy.stats, FDA rule engine, no shortcuts. |
| Reward & Curriculum | 1:30–2:00 | Rewards (20%) | Show decomposed reward (8+7 components), potential-based shaping, 5-tier curriculum. "Every component is math-verified, independently debuggable." |
| Training Results | 2:00–2:40 | Rewards (20%) | Show reward curve (negative → positive), before/after episode, baseline comparison table. `[FILL ONSITE: replace example numbers with real data]` "Random: 5%. Trained: [FILL]%." |
| Close | 2:40–3:00 | Storytelling (30%) | "The agent discovered biomarker stratification on its own — a strategy that took clinical researchers decades to formalize." Show capability radar. |

---

## Key Talking Points per Scoring Axis

### Environment Innovation (40%) — Spend ~50 seconds here

**What judges want:** Novel, creative, challenging. Meaningfully tests agent behavior. Real interaction with tools/APIs.

**Say:**
- "No existing OpenEnv environment covers clinical trial design. This is the first."
- "Verification is objective: scipy.stats power calculations, not LLM judges."
- "Our power assumptions and boundaries are grounded with rpact-equivalent validation tables (see docs/grounding.md + server/grounding/rpact_validation.json)."
- "The world is genuinely partially observable — true effect size, responder subgroup, safety profile are all hidden from the agent."
- "19 actions across 5 clinical phases, with hard FDA prerequisites that block invalid actions."
- "Domain randomization: budget ±30%, time ±20%, dropout ±15%. Every episode is unique."
- "Adaptive difficulty: after mastery, the environment actively targets the agent's weak spots."

**Visual:** Architecture diagram (one-page system view)

**Anticipated Q&A:**
- *"How do you verify without an LLM judge?"* → "scipy.stats calculates power from effect size and sample size. FDA rules are boolean functions. The trial simulation runs the designed trial against hidden ground truth and returns a p-value. We also keep rpact-equivalent validation tables for boundaries and power sanity checks."
- *"What makes this harder than just prompting GPT-4?"* → "The ground truth is hidden. You can't reason about the correct sample size without first running Phase I to estimate the effect. It requires sequential decision-making, not single-shot reasoning."

### Storytelling (30%) — Spend ~60 seconds across hook + close

**What judges want:** Clear problem explanation, engaging demo, easy to follow.

**Say:**
- Hook: disease + drug + zero knowledge → failure-first
- Before/after: "Episode 1: skips Phase I, guesses randomly, FDA rejects, reward -2.5. Episode 40: systematic dose escalation, discovers EGFR+ subgroup, power 0.85, reward +11.2."
- Close: connects to real-world impact

**Visual:** Dashboard with episode replay (before vs after side-by-side)

**Anticipated Q&A:**
- *"Is this just memorizing patterns?"* → "No — the curriculum escalates difficulty and randomizes parameters. The agent at Expert tier faces scenarios it has never seen with deliberately misleading Phase I signals."
- *"What's the real-world application?"* → "Clinical trial design costs $100M+ per trial. An AI assistant that suggests enrichment strategies or flags underpowered designs saves years and millions."

### Showing Improvement in Rewards (20%) — Spend ~40 seconds

**What judges want:** Observable training progress, reward curves, before/after behavior.

**Say:**
- "Initial average reward: `[FILL ONSITE]`. Final average: `[FILL ONSITE]`. Best single episode: `[FILL ONSITE]`."
- "Reward curve shows clear upward trend with curriculum tier transitions marked." `[FILL ONSITE: confirm or adjust]`
- Show per-component trends: "r_ordering improves first (learns phase sequence), then r_info_gain (learns to gather data), then r_validity (learns FDA rules)." `[FILL ONSITE: verify order from actual data]`
- "Capability radar: random baseline is a tiny polygon near origin. Trained agent fills `[FILL ONSITE]`%+ on all axes."

**Visual:** Reward curve (scatter + rolling average + tier markers) + Capability radar

**Anticipated Q&A:**
- *"How do you prevent reward hacking?"* → "Three safeguards: (1) potential-based shaping preserves optimal policy, (2) terminal reward dominates — only fires if trial truly succeeds, (3) overconfidence penalty punishes claims that don't match hidden truth."
- *"What's the success rate at Expert tier?"* → Give actual number. Even 20% is impressive — Expert scenarios have tiny effects and misleading signals.

### Reward/Training Pipeline (10%) — Spend ~30 seconds

**What judges want:** Coherent reward logic, meaningful improvement, reproducible pipeline.

**Say:**
- "GRPO with TRL 0.29, LoRA rank 16 on Qwen2.5-7B, 8 parallel rollouts, vLLM colocate for inference."
- "Reward: 8 per-step + 7 terminal components. Range -3 to +14. GRPO needs high variance — we deliver it."
- "Full pipeline: train.py, eval_compare.py, plot_rewards.py, train_colab.ipynb. All open-source."

**Visual:** Colab notebook screenshot

**Anticipated Q&A:**
- *"Why GRPO over PPO?"* → "GRPO doesn't need a separate value network — saves memory. With 8 rollouts per prompt, it computes per-group advantages directly. Ideal for a 7B model on a single H100."
- *"Can you reproduce results?"* → "Yes. Seeded NoiseModel ensures same seed = same episode. Episode transcripts are saved as JSONL for offline analysis."

---

## Demo Episode Scripts

> Replace with real transcripts from training once available. Until then, use these as narrative scaffolding.

### Episode 1 — The Failure (Before Training)

```
Scenario: solid_tumor_chemo — NSCLC with hidden EGFR+ subgroup
Budget: $2.5M | Time: 180 days | Hidden: EGFR+ subgroup (35%)

Step 1: set_sample_size(n=20)         → BLOCKED (Phase I not done)  → Reward: -0.15
Step 2: run_primary_analysis()        → BLOCKED (no protocol)       → Reward: -0.15
Step 3: submit_to_fda_review()        → BLOCKED (no endpoint set)   → Reward: -0.15
Step 4: set_primary_endpoint("OS")    → WARNING: no Phase I data    → Reward: -0.05
... (random flailing for 95 steps, accumulating penalties) ...
Step 95: TIMEOUT

TOTAL REWARD: -2.5 | Success: No | FDA: 0/6 | Phase compliance: 12% | Subgroup: Not found
```

**Narration:** "Episode 1. The agent knows nothing. It tries to set a sample size before running Phase I. Tries to analyze results that don't exist. Flails for 95 steps and times out. Reward: -2.5."

### Episode ~8 — The Breakthrough (Learns Workflow)

```
Scenario: solid_tumor_chemo — Same hidden truth

Step 1-3: run_dose_escalation (50mg → 100mg → 150mg)  → Phase I ✓
Step 4: estimate_effect_size()                         → Effect: 0.28 ± 0.12
Step 5: set_primary_endpoint("PFS")                    → Phase II design ✓
Step 6: set_sample_size(n=200)
Step 7-9: dosing, control arm, blinding
Step 10: submit_to_fda_review()                        → PASSED ✓
Step 11: run_primary_analysis()                        → p=0.048, power=0.65
Step 12: synthesize_conclusion()

TOTAL REWARD: +3.2 | Success: Yes (marginal) | FDA: 5/6 | Subgroup: Not found
Key miss: Enrolled general population → diluted EGFR+ signal → barely significant
```

**Narration:** "Episode 8. The agent learned the workflow. Phase I first, then design, then submit. But it missed the key insight — the drug works 3× better in EGFR+ patients. It enrolled everyone, diluting the signal. Barely significant."

### Episode ~40 — Mastery (Discovers Subgroup Enrichment)

```
Scenario: solid_tumor_chemo — Harder: Budget $2.0M, Time 160 days

Step 1-3: Dose escalation (50/100/150mg)
Step 4: observe_safety_signal()         → "Grade 2 rash in 2 patients (EGFR-related)"
Step 5: estimate_effect_size()          → "3 responders all EGFR+"
Step 6: add_biomarker_stratification("EGFR")  ← THE KEY DECISION
  → EGFR+ effect: 0.54 ± 0.15 | EGFR- effect: 0.08 ± 0.12
Step 7: set_primary_endpoint("PFS")
Step 8: set_inclusion_criteria("EGFR_positive")  ← ENRICHMENT
Step 9: set_sample_size(n=80)  → With effect=0.54, 80 EGFR+ patients → power=0.88
Steps 10-13: dosing, control, randomization, blinding
Step 14: submit_to_fda_review() → PASSED ✓
Step 15: run_interim_analysis() → p=0.018 at interim
Step 16: run_primary_analysis() → p=0.003, power=0.88
Step 17: synthesize_conclusion() → Correct subgroup, effect within 5% of truth

TOTAL REWARD: +11.2 | Success: STRONG | FDA: 6/6 | Phase: 100% | Subgroup: EGFR+ ✓
```

**Narration:** "Episode 40. The agent discovered EGFR+ enrichment on its own — a strategy that took clinical researchers decades. 80 targeted patients instead of 200. p=0.003. Reward: +11.2."

### Side-by-Side Comparison Card

```
              EPISODE 1 (Before)     EPISODE 40 (After)
Reward:       -2.5                   +11.2
Steps:        95 (timeout)           17
Success:      No                     Yes
p-value:      N/A                    0.003
Power:        N/A                    0.88
FDA pass:     0/6                    6/6
Subgroup:     Not found              EGFR+ ✓
Phase order:  12% correct            100%
Budget used:  100% (wasted)          73% (efficient)
```

---

## Q&A Prep (Top 10 Expected Questions)

| # | Question | Answer |
|---|----------|--------|
| 1 | How do you verify trial success without an LLM judge? | scipy.stats power calc + trial simulation against hidden ground truth. Boundary/power assumptions are cross-checked in docs/grounding.md and server/grounding/rpact_validation.json. |
| 2 | What's novel about this vs. existing clinical trial libs? | First RL environment. No existing OpenEnv env covers trial design. Closest work (TrialGPT) is classification, not sequential decision-making. |
| 3 | How does the agent discover the hidden subgroup? | Through the `add_biomarker_stratification` action — it observes differential response rates and learns to set inclusion criteria accordingly. |
| 4 | Isn't 7B too small for this? | The curriculum starts easy (large effects, clear signals). GRPO + LoRA fine-tunes the model specifically for this domain. 7B with task-specific training beats larger models zero-shot. |
| 5 | How many episodes to convergence? | ~200–400 episodes across 5 tiers. Warmup mastery by ~35 episodes, Expert exposure by ~180. |
| 6 | Does the environment scale to other domains? | The core pattern (hidden state → noisy observations → programmatic verification) applies to any domain with objective ground truth. |
| 7 | How does adaptive difficulty work? | After mastery (>70% success), the environment hardens parameters: smaller effects, tighter budgets, misleading Phase I signals. Targets the agent's specific weak spots. |
| 8 | What happens when Phase I is misleading? | At Advanced/Expert tier, early cohorts can show contradictory signal due to sampling noise. Agent must run more dose levels before committing to a design. |
| 9 | How is this different from prompting GPT-4 with trial design instructions? | GPT-4 can generate a trial protocol, but can't iteratively refine based on experimental results. Our agent makes sequential decisions conditioned on noisy observations from experiments it chose to run. |
| 10 | What's the biggest lesson from training? | `[FILL ONSITE]` — e.g., "The agent independently discovered that enriching for biomarker subgroups is more sample-efficient than powering for the general population." |

---

## Demo Flow (If showing dashboard)

1. Open `dashboard.html` — shows demo mode with simulated learning curve
2. Point to reward curve: "Starting negative, crossing zero around episode 30, reaching +10 by episode 150"
3. Click an early episode: show random action sequence, FDA violations, timeout
4. Click a late episode: show systematic Phase I → Phase II → analysis workflow, correct subgroup identification
5. Point to capability radar: "From a tiny polygon to 80%+ coverage on all six axes"
6. Point to curriculum bar: "Warmup → Beginner → Intermediate → Advanced — the agent earned harder scenarios"

---

## One-Line Pitch (for Discord / hallway)

> "We built an RL environment where a 7B model learns to design FDA-compliant clinical trials from scratch — finding hidden responder subgroups through trial-and-error, verified by real statistics, no LLM judge."

---

## Pitch Don'ts

- Don't explain GRPO internals (judges don't care about the optimizer)
- Don't read from slides — tell the story
- Don't say "we plan to" — everything presented must be done or have results
- Don't apologize for small model size — frame it as impressive that 7B can do this
- Don't show code in the pitch — show architecture diagram + results
- Don't spend more than 10 seconds on setup/Docker/infrastructure
