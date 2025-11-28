"""
Temperature analysis sub-package.

Tools for reverse-engineering Kalshi/NWS settlement temperatures from
Visual Crossing 5-minute observation series.

Key use case:
- Test which rounding rule best maps VC float temps → NWS integer highs
- Identify edge cases where simple rules fail
- Validate temperature-to-bracket mapping before backtesting

Public API:
- DaySeries, RuleStats (datastructures)
- All rule functions (rules)
- evaluate_rules (evaluator)
"""

from analysis.temperature.datastructures import DaySeries, RuleStats
from analysis.temperature.evaluator import evaluate_rules
from analysis.temperature.rules import (
    rule_max_round,
    rule_max_of_rounded,
    rule_ceil_max,
    rule_floor_max,
    rule_plateau_20min,
    rule_ignore_singletons,
    ALL_RULES,
)

__all__ = [
    "DaySeries",
    "RuleStats",
    "evaluate_rules",
    "rule_max_round",
    "rule_max_of_rounded",
    "rule_ceil_max",
    "rule_floor_max",
    "rule_plateau_20min",
    "rule_ignore_singletons",
    "ALL_RULES",
]
