"""
UI module for manual trading terminal.

Provides Rich-based terminal UI components for interactive trading.
"""

from live_trading.ui.display import (
    format_opportunities_table,
    format_opportunity_details,
    format_execution_plan,
    format_positions_table,
)
from live_trading.ui.manual_cli import ManualTradingCLI

__all__ = [
    'ManualTradingCLI',
    'format_opportunities_table',
    'format_opportunity_details',
    'format_execution_plan',
    'format_positions_table',
]
