"""
Base strategy classes and the default open_maker_base strategy.

The base strategy is the original single-bin buy-and-hold logic:
1. At market open, use forecast to pick a bracket
2. Post maker limit order at fixed price P
3. Hold to settlement
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class StrategyParamsBase:
    """Base parameters shared by all strategies."""

    entry_price_cents: float = 50.0
    temp_bias_deg: float = 0.0
    basis_offset_days: int = 1
    bet_amount_usd: float = 200.0


@dataclass
class OpenMakerParams(StrategyParamsBase):
    """Parameters for the base open_maker strategy (buy and hold)."""
    pass


@dataclass
class TradeContext:
    """
    Context passed to strategy for making decisions.

    Contains all data needed to make entry and exit decisions.
    """

    city: str
    event_date: date
    forecast_basis_date: date
    temp_fcst_open: float  # Raw forecast temperature
    temp_adjusted: float  # After bias adjustment

    # Bracket info
    ticker: str
    floor_strike: Optional[float]
    cap_strike: Optional[float]
    bin_index: int  # Position in sorted brackets (0 = lowest)
    total_bins: int  # Total number of brackets

    # All brackets for this event (sorted by strike)
    all_brackets: pd.DataFrame  # columns: ticker, floor_strike, cap_strike

    # Market metadata
    markets_df: pd.DataFrame  # Full market data for the event_date

    # Entry info
    entry_price_cents: float
    num_contracts: int
    amount_usd: float

    # Settlement (for P&L calculation)
    tmax_final: Optional[float] = None


@dataclass
class TradeDecision:
    """Decision made by a strategy."""

    action: str  # "hold" or "exit"
    exit_price_cents: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None


class StrategyBase(ABC):
    """Abstract base class for all trading strategies."""

    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for this strategy."""
        pass

    @abstractmethod
    def decide(self, context: TradeContext, candles_df: Optional[pd.DataFrame] = None) -> TradeDecision:
        """
        Make a trading decision given the context.

        Args:
            context: TradeContext with all relevant data
            candles_df: Optional minute candles for exit strategies

        Returns:
            TradeDecision indicating whether to hold or exit
        """
        pass


class BaseStrategy(StrategyBase):
    """
    The original open_maker_base strategy.

    Simple buy-and-hold: enter at open, hold to settlement.
    No exit logic - all trades settle at maturity.
    """

    def __init__(self, params: Optional[OpenMakerParams] = None):
        """Initialize with optional params (not used by base strategy)."""
        self.params = params

    @property
    def strategy_id(self) -> str:
        return "open_maker_base"

    def decide(self, context: TradeContext, candles_df: Optional[pd.DataFrame] = None) -> TradeDecision:
        """
        Base strategy always holds to settlement.

        No exit logic - just hold.
        """
        return TradeDecision(action="hold")


# Register the base strategy
from . import register_strategy
register_strategy("open_maker_base", BaseStrategy, OpenMakerParams)
