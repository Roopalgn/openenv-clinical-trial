"""
curriculum — Curriculum controller and scenario registry.

Provides advance_curriculum, select_scenario, EpisodeMetrics, AdversarialDesigner,
and the four initial ScenarioConfig instances (solid_tumor_chemo, autoimmune_biologic,
cns_depression, rare_disease_orphan).
"""

from server.curriculum.adversarial_designer import AdversarialDesigner
from server.curriculum.controller import (
    EpisodeMetrics,
    advance_curriculum,
    select_scenario,
)
from server.curriculum.scenarios import (
    AUTOIMMUNE_BIOLOGIC,
    CNS_DEPRESSION,
    RARE_DISEASE_ORPHAN,
    SCENARIO_LIST,
    SCENARIOS,
    SOLID_TUMOR_CHEMO,
    WARMUP,
)

__all__ = [
    "AdversarialDesigner",
    "EpisodeMetrics",
    "advance_curriculum",
    "select_scenario",
    "WARMUP",
    "SOLID_TUMOR_CHEMO",
    "AUTOIMMUNE_BIOLOGIC",
    "CNS_DEPRESSION",
    "RARE_DISEASE_ORPHAN",
    "SCENARIOS",
    "SCENARIO_LIST",
]
