"""
Scenario registry for the curriculum controller.

Defines ScenarioConfig instances for all four scenario IDs plus a tier-0 warmup
variant of solid_tumor_chemo with an inflated effect size.
"""

from models import ScenarioConfig

# Tier 0 — warmup (solid_tumor_chemo with inflated effect size, easier)
WARMUP = ScenarioConfig(
    scenario_id="solid_tumor_chemo_warmup",
    curriculum_tier=0,
    disease_area="oncology",
    effect_size_range=(0.55, 0.85),  # inflated vs tier-1 (0.25–0.55)
    side_effect_rate_range=(0.10, 0.25),
    placebo_response_range=(0.05, 0.15),
    dropout_rate_range=(0.05, 0.10),
    budget_usd=8_000_000.0,
    time_budget_days=365,
    min_sample_size=60,
    description=(
        "Warmup scenario: EGFR+ solid-tumour chemotherapy with an inflated "
        "effect size to help the agent learn basic trial-design mechanics."
    ),
)

# Tier 1 — EGFR+ subgroup enrichment
SOLID_TUMOR_CHEMO = ScenarioConfig(
    scenario_id="solid_tumor_chemo",
    curriculum_tier=1,
    disease_area="oncology",
    effect_size_range=(0.25, 0.55),
    side_effect_rate_range=(0.15, 0.35),
    placebo_response_range=(0.05, 0.15),
    dropout_rate_range=(0.05, 0.15),
    budget_usd=10_000_000.0,
    time_budget_days=540,
    min_sample_size=80,
    description=(
        "EGFR+ solid-tumour chemotherapy. Agent must identify the EGFR+ "
        "biomarker subgroup to unlock the true effect size."
    ),
)

# Tier 2 — U-shaped dose-response
AUTOIMMUNE_BIOLOGIC = ScenarioConfig(
    scenario_id="autoimmune_biologic",
    curriculum_tier=2,
    disease_area="immunology",
    effect_size_range=(0.20, 0.45),
    side_effect_rate_range=(0.10, 0.30),
    placebo_response_range=(0.15, 0.30),
    dropout_rate_range=(0.08, 0.18),
    budget_usd=15_000_000.0,
    time_budget_days=720,
    min_sample_size=120,
    description=(
        "Autoimmune biologic with a U-shaped dose-response curve. "
        "Agent must run dose-escalation to find the optimal dose window."
    ),
)

# Tier 3 — high placebo response
CNS_DEPRESSION = ScenarioConfig(
    scenario_id="cns_depression",
    curriculum_tier=3,
    disease_area="psychiatry",
    effect_size_range=(0.15, 0.35),
    side_effect_rate_range=(0.10, 0.25),
    placebo_response_range=(0.35, 0.55),  # high placebo response
    dropout_rate_range=(0.10, 0.25),
    budget_usd=20_000_000.0,
    time_budget_days=900,
    min_sample_size=200,
    description=(
        "CNS depression trial with a high placebo-response rate. "
        "Agent must power the study to detect a small drug-placebo delta."
    ),
)

# Tier 4 — rare disease / tiny n
RARE_DISEASE_ORPHAN = ScenarioConfig(
    scenario_id="rare_disease_orphan",
    curriculum_tier=4,
    disease_area="rare_disease",
    effect_size_range=(0.40, 0.80),  # larger effect needed to compensate tiny n
    side_effect_rate_range=(0.05, 0.20),
    placebo_response_range=(0.05, 0.15),
    dropout_rate_range=(0.05, 0.15),
    budget_usd=5_000_000.0,
    time_budget_days=1080,
    min_sample_size=10,  # tiny n — orphan disease
    description=(
        "Rare-disease orphan drug trial with a very small patient population. "
        "Agent must justify statistical validity under FDA orphan-drug rules."
    ),
)

# Registry — keyed by scenario_id for O(1) lookup
SCENARIOS: dict[str, ScenarioConfig] = {
    WARMUP.scenario_id: WARMUP,
    SOLID_TUMOR_CHEMO.scenario_id: SOLID_TUMOR_CHEMO,
    AUTOIMMUNE_BIOLOGIC.scenario_id: AUTOIMMUNE_BIOLOGIC,
    CNS_DEPRESSION.scenario_id: CNS_DEPRESSION,
    RARE_DISEASE_ORPHAN.scenario_id: RARE_DISEASE_ORPHAN,
}

# Convenience list ordered by tier
SCENARIO_LIST: list[ScenarioConfig] = [
    WARMUP,
    SOLID_TUMOR_CHEMO,
    AUTOIMMUNE_BIOLOGIC,
    CNS_DEPRESSION,
    RARE_DISEASE_ORPHAN,
]
