"""
rule_engine — FDA constraint enforcement and action prerequisite checks.

Provides check_fda_compliance, ComplianceResult, and the phase-to-action
transition table (TRANSITION_TABLE).
"""

from server.rules.fda_rules import (
    TRANSITION_TABLE,
    ComplianceResult,
    check_fda_compliance,
)

__all__ = ["check_fda_compliance", "ComplianceResult", "TRANSITION_TABLE"]
