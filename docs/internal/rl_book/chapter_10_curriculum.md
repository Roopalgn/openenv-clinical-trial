# Chapter 10: Curriculum Learning — Baby Steps to Expert

## Why Not Start with the Hardest Problem?

Imagine learning to play chess by immediately competing against Magnus Carlsen. You'd lose every game. You'd learn nothing because the gap between your skill and the challenge is too wide.

Instead, good teachers start simple and gradually increase difficulty:
1. Learn how pieces move
2. Play against an easy bot
3. Play against a medium bot
4. Play against a hard bot
5. Study openings and endgames
6. Compete against strong players

This is **curriculum learning** — a well-studied technique in RL (referenced in Narvekar et al., 2020, JMLR).

## Our 5-Tier Curriculum

```
Tier 0: WARMUP
  "Here's a drug that obviously works. Just follow the basic steps."
  - Effect size: 0.55–0.85 (very large, easy to detect)
  - Budget: $8M (very generous)
  - Hints provided by junior judge
  - Agent must learn: basic workflow sequence

Tier 1: BEGINNER  
  "Here's a cancer drug. Can you design a proper trial?"
  - EGFR+ lung cancer (solid_tumor_chemo)
  - Effect size: 0.25–0.55 (moderate)
  - Budget: $10M (generous)
  - Agent must learn: proper sample sizing and effect estimation

Tier 2: INTERMEDIATE
  "This drug has a weird dose-response curve."
  - Autoimmune biologic (rheumatoid arthritis)
  - Effect size: 0.20–0.45 (medium)
  - U-shaped dose-response (higher dose ≠ better!)
  - Agent must learn: biomarker stratification and efficient design

Tier 3: ADVANCED
  "The placebo effect is masking the drug's real effect."
  - CNS depression trial
  - Effect size: 0.15–0.35 (small, hard to detect)
  - Placebo response: 35–55% (very high — the trap!)
  - Agent must learn: handling ambiguity, adaptive designs

Tier 4: EXPERT
  "Adversarial scenarios. Everything is hard simultaneously."
  - Rare disease orphan / adversarial designer
  - Tiny patient population (n ≤ 50)
  - OR: compound challenges (small effect + hidden subgroup + high dropout)
  - Agent must learn: near-optimal everything
```

## Scenario Definitions (The Code)

Each tier has a scenario defined in code:

```python
# From server/curriculum/scenarios.py

# Tier 0 — Training wheels
WARMUP = ScenarioConfig(
    scenario_id="solid_tumor_chemo_warmup",
    curriculum_tier=0,
    disease_area="oncology",
    effect_size_range=(0.55, 0.85),       # Very large effect → easy to detect
    side_effect_rate_range=(0.10, 0.25),
    placebo_response_range=(0.05, 0.15),   # Low placebo → clean signal
    dropout_rate_range=(0.05, 0.10),       # Few dropouts → reliable data
    budget_usd=8_000_000.0,               # Generous budget
    time_budget_days=365,
    min_sample_size=60,
    description="Warmup: EGFR+ solid tumour with inflated effect size..."
)

# Tier 4 — The hardest
RARE_DISEASE_ORPHAN = ScenarioConfig(
    scenario_id="rare_disease_orphan",
    curriculum_tier=4,
    disease_area="rare_disease",
    effect_size_range=(0.40, 0.80),        # Large effect needed
    side_effect_rate_range=(0.05, 0.20),
    placebo_response_range=(0.05, 0.15),
    dropout_rate_range=(0.05, 0.15),
    budget_usd=5_000_000.0,               # Tight budget
    time_budget_days=1080,                 # Lots of time (rare disease)
    min_sample_size=10,                    # Tiny n — orphan disease!
    description="Rare disease orphan drug with very small patient population..."
)
```

## Advancement Logic

How does the agent "graduate" from one tier to the next?

```python
# From server/curriculum/controller.py

MASTERY_THRESHOLD = 0.70       # 70% success rate to graduate
FAST_TRACK_THRESHOLD = 0.90    # 90% → skip a tier
FAST_TRACK_MIN_EPISODES = 3    # Need at least 3 episodes for fast-track
ROLLING_WINDOW = 10            # Look at last 10 episodes

def advance_curriculum(tier, metrics):
    """Should the agent advance to the next tier?"""
    
    if tier >= MAX_TIER:  # Already at Expert
        return MAX_TIER
    
    history = metrics.episode_history
    if len(history) == 0:
        return tier  # No data yet
    
    # Check success rate over last 10 episodes
    window = history[-ROLLING_WINDOW:]
    rolling_rate = sum(window) / len(window)
    
    # Fast-track: 90%+ success after 3+ episodes → skip a tier
    if len(history) >= 3 and rolling_rate >= 0.90:
        return min(tier + 2, MAX_TIER)
    
    # Normal: 70%+ success → advance one tier
    if rolling_rate >= 0.70:
        return min(tier + 1, MAX_TIER)
    
    # Not ready yet
    return tier
```

**Example progression:**
```
Episodes 1-10:  Tier 0 (warmup), wins 8/10 = 80% → Advances to Tier 1!
Episodes 11-20: Tier 1 (beginner), wins 5/10 = 50% → Stays at Tier 1
Episodes 21-30: Tier 1 (beginner), wins 7/10 = 70% → Advances to Tier 2!
Episodes 31-33: Tier 2 (intermediate), wins 3/3 = 100% → Fast-track to Tier 4!
```

### Why These Specific Thresholds?

```
Tier 0 → 1: 70% success = consistently designs basic trials
Tier 1 → 2: 70% success = handles realistic oncology trials
Tier 2 → 3: 70% success = navigates tricky dose-response curves  
Tier 3 → 4: 70% success = manages high-noise environments
```

The 70% threshold is lower than you might expect. This is intentional:
- At higher tiers, scenarios are **genuinely harder**
- 70% success at Tier 3 (advanced) represents excellent ability
- Requiring 90%+ would waste training compute on already-mastered tiers

### No Demotion

Once the agent advances, it **never goes back**:

```
Tier 0 → 1 → 2 → 3 → 4  (can only go forward)
```

This prevents the agent from oscillating between tiers, which wastes training time.

## The Adversarial Designer (Expert Tier)

At the expert tier (difficulty > 0.80), something special activates: the **AdversarialDesigner**.

```python
# From server/curriculum/adversarial_designer.py

class AdversarialDesigner:
    """Generates scenarios targeting the agent's weak spots."""
    
    def analyze_failures(self, episode_history):
        """Look at past failures to find patterns."""
        for ep in episode_history:
            if not ep["success"]:
                if ep.get("true_effect_size", 1) <= 0.25:
                    self._weak_spots["small_effect_failures"] += 1
                if ep.get("dropout_rate", 0) >= 0.20:
                    self._weak_spots["high_dropout_failures"] += 1
                if not ep.get("used_biomarker_stratification", False):
                    self._weak_spots["hidden_subgroup_failures"] += 1
    
    def generate_scenario(self, weak_spots):
        """Create a scenario that targets the biggest weakness."""
        dominant = self._dominant_weak_spot(weak_spots)
        
        # Always generate compound challenges:
        return ScenarioConfig(
            scenario_id="adversarial_expert",
            effect_size_range=(0.10, 0.20),     # Small effect (hard)
            dropout_rate_range=(0.25, 0.35),     # High dropout (hard)
            budget_usd=25_000_000.0,
            min_sample_size=200,
            description=f"Adversarial scenario targeting '{dominant}'..."
        )
```

**How it works:**
1. After each expert-level failure, the designer records what went wrong
2. It identifies the agent's **dominant weakness** (e.g., "fails when effect is small")
3. It generates a scenario that specifically targets that weakness
4. The scenario always has **compound challenges** (not just one hard thing)

**Real-world parallel:** This is like a driving instructor who notices you struggle with parallel parking and night driving, then gives you a parallel parking test at night, in the rain.

> **Design Decision Box: Why Adversarial Design?**
>
> Without adversarial design, the expert tier would just be the rare disease scenario repeated. The agent would eventually memorize it. The adversarial designer ensures the agent faces **novel** challenges based on its actual weaknesses. This prevents overfitting to any single expert scenario.

## How Difficulty Scales Details

As the tier increases, the environment changes in several ways:

| Parameter | Warmup | Beginner | Intermediate | Advanced | Expert |
|---|---|---|---|---|---|
| Effect size multiplier | 1.5× | 1.0× | 0.80× | 0.60× | 0.40× |
| Budget multiplier | 1.2× | 1.0× | 0.85× | 0.70× | 0.55× |
| Dropout multiplier | 0.5× | 1.0× | 1.3× | 1.6× | 2.0× |
| Noise level | 0.05 | 0.08 | 0.12 | 0.18 | 0.25 |
| Placebo boost | +0% | +0% | +5% | +10% | +15% |
| Judge persona | Junior | Junior→Senior | Senior | Senior→Principal | Principal |

At warmup:
- The drug effect is 1.5× larger (easy to detect)
- Budget is 1.2× bigger (more room for mistakes)
- Dropout rate is halved (cleaner data)
- Noise is minimal (clear measurements)
- The judge gives hints!

At expert:
- The drug effect is 0.4× smaller (hard to detect)
- Budget is 0.55× smaller (tight)
- Dropout rate is doubled (messy data)
- Noise is 5× higher (unclear measurements)
- The judge is strict and gives NO hints

## The Judge Personas

The TrialJudge changes personality based on difficulty:

### Junior (difficulty < 0.4)
```
"That's a good start! For next time, try running dose escalation 
 before estimating effect size — you'll get more accurate results."
```
- Lenient scoring
- Provides concrete hints
- Doesn't penalize inefficiency

### Senior (difficulty 0.4–0.7)
```
"Your interim analysis timing was appropriate given the enrollment level."
```
- Standard clinical expectations
- No hints
- Balanced feedback

### Principal (difficulty > 0.7)
```
"Unnecessary safety observation after Phase I completion. 
 Inefficient use of $15,000 and 30 days."
```
- Strict
- Penalizes redundant actions
- Penalizes inefficiency
- No hints whatsoever

The judge also applies **overconfidence penalties**:
```python
if action.confidence >= 0.8 and not result_correct:
    penalty = -0.5  # "You said you were 80%+ sure, but you were wrong"
```

## Putting It Together: A Training Run

Here's what a full training run looks like across the curriculum:

```
Episodes   1-15:  Tier 0 (warmup)
  - Agent learns: reset → design → enroll → analyze workflow
  - Average reward: -2 → +4 (improving)
  - Hits 70% success → ADVANCE

Episodes  16-40:  Tier 1 (beginner, solid tumor)
  - Agent learns: proper sample sizing, dose escalation
  - Average reward: +2 → +7
  - Hits 70% success → ADVANCE

Episodes  41-80:  Tier 2 (intermediate, autoimmune)
  - Agent discovers: U-shaped dose-response, picks optimal dose
  - Average reward: +3 → +8
  - Hits 70% success → ADVANCE

Episodes  81-140: Tier 3 (advanced, CNS depression)
  - Agent struggles: high placebo masks signal
  - Agent learns: enrich for severe patients
  - Average reward: +1 → +6
  - Hits 70% success → ADVANCE

Episodes 141-300: Tier 4 (expert, adversarial)
  - Agent faces compound challenges
  - Adversarial designer targets weaknesses
  - Average reward: +3 → +11
  - Terminal tier — keeps improving
```

---

## Chapter 10 Glossary

| Keyword | Definition |
|---------|-----------|
| **Curriculum Learning** | Training strategy that gradually increases difficulty |
| **Tier** | A difficulty level in the curriculum (0=easiest, 4=hardest) |
| **Advancement/Graduation** | Moving to the next difficulty tier |
| **Rolling Window** | Looking at only the most recent N episodes for success rate |
| **Fast-Track** | Skipping a tier due to exceptional performance (90%+) |
| **Mastery Threshold** | Success rate required to advance (70%) |
| **Adversarial Designer** | Component that generates scenarios targeting agent weaknesses |
| **Compound Challenge** | Scenario with multiple hard elements simultaneously |
| **Weak Spot** | An area where the agent consistently fails |
| **Domain Randomization** | Adding random variation to prevent memorization |
| **Overfitting** | Memorizing specific scenarios instead of learning general strategies |
| **Judge Persona** | The judge's personality (junior/senior/principal) |
| **Narvekar et al. (2020)** | Academic reference for curriculum RL (JMLR) |
