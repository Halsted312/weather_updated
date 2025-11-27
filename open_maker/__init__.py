"""
Open-Maker Strategy Package

A self-contained backtest strategy for posting maker limit orders
at market open and holding to settlement.

Strategy overview:
- At market open (10am ET on event_date - 1), get forecast for event_date
- Find bracket containing forecasted high temp (with optional bias adjustment)
- Post maker limit order at fixed price P (e.g., 40c or 50c)
- Assume fill, hold to settlement
- Maker fees = $0 for weather markets
"""

from .core import (
    run_backtest,
    OpenMakerTrade,
    OpenMakerResult,
    OpenMakerParams,
    print_results,
)

__all__ = [
    "run_backtest",
    "OpenMakerTrade",
    "OpenMakerResult",
    "OpenMakerParams",
    "print_results",
]
