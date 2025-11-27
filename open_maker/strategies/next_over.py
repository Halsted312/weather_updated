"""
NextOver exit strategy.

At decision time (based on predicted high hour), check if the neighbor bracket
is trading rich enough to signal we should exit early.

Exit rule:
- If neighbor bin (i+1) price >= neighbor_price_min_c AND
- Our bin price <= our_price_max_c
- Then exit at our current bid price
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import logging

import pandas as pd
from zoneinfo import ZoneInfo

from .base import StrategyParamsBase, StrategyBase, TradeContext, TradeDecision
from ..utils import get_candle_price_for_exit, get_candle_price_for_signal

logger = logging.getLogger(__name__)


@dataclass
class NextOverParams(StrategyParamsBase):
    """Parameters for the next_over exit strategy."""

    # Entry params (inherited)
    entry_price_cents: float = 45.0
    temp_bias_deg: float = 0.0
    basis_offset_days: int = 1
    bet_amount_usd: float = 200.0

    # Exit timing: minutes before predicted high (negative = before)
    # e.g., -180 = 3 hours before predicted high
    decision_offset_min: int = -180

    # Exit thresholds
    neighbor_price_min_c: int = 50  # Neighbor must be >= this to trigger exit
    our_price_max_c: int = 30  # Our bin must be <= this to trigger exit


class NextOverStrategy(StrategyBase):
    """
    Exit strategy based on neighboring bracket price.

    At decision_time (relative to predicted high hour):
    - Check if neighbor bin (one above) is trading >= neighbor_price_min_c
    - Check if our bin is trading <= our_price_max_c
    - If both conditions met, exit at our current bid price

    Otherwise, hold to settlement.
    """

    def __init__(self, params: NextOverParams):
        self.params = params

    @property
    def strategy_id(self) -> str:
        return "open_maker_next_over"

    def decide(self, context: TradeContext, candles_df: Optional[pd.DataFrame] = None) -> TradeDecision:
        """
        Decide whether to exit based on neighbor bracket price.

        Args:
            context: Trade context with bracket info
            candles_df: Minute candles around decision_time (required for exit logic)

        Returns:
            TradeDecision with action="exit" or action="hold"
        """
        # If no candles provided, can't evaluate exit - hold to settlement
        if candles_df is None or candles_df.empty:
            logger.debug(f"No candles for {context.city}/{context.event_date}, holding to settlement")
            return TradeDecision(action="hold")

        # Check if we're at the edge (no neighbor above)
        if context.bin_index >= context.total_bins - 1:
            logger.debug(f"At top bracket (index {context.bin_index}), no neighbor above - holding")
            return TradeDecision(action="hold")

        # Get neighbor ticker (one bracket above)
        sorted_brackets = context.all_brackets.sort_values("floor_strike", na_position="first")
        neighbor_idx = context.bin_index + 1

        if neighbor_idx >= len(sorted_brackets):
            return TradeDecision(action="hold")

        neighbor_ticker = sorted_brackets.iloc[neighbor_idx]["ticker"]

        # Get decision time from candles (the runner passes a window around decision_time)
        decision_time = candles_df["bucket_start"].max()

        # Get prices at decision time using utils helpers with robust fallback chains
        # For our exit price: yes_bid -> mid -> close -> None
        our_price = get_candle_price_for_exit(
            candles_df, context.ticker, decision_time, window_minutes=10
        )
        # For neighbor signal: yes_bid -> close -> None
        neighbor_price = get_candle_price_for_signal(
            candles_df, neighbor_ticker, decision_time, window_minutes=10
        )

        if our_price is None or neighbor_price is None:
            logger.debug(
                f"Missing prices at decision time: our={our_price}, neighbor={neighbor_price} - holding"
            )
            return TradeDecision(action="hold")

        # Check exit conditions
        if neighbor_price >= self.params.neighbor_price_min_c and our_price <= self.params.our_price_max_c:
            logger.info(
                f"EXIT triggered: {context.city}/{context.event_date} "
                f"our_price={our_price}c <= {self.params.our_price_max_c}c, "
                f"neighbor_price={neighbor_price}c >= {self.params.neighbor_price_min_c}c"
            )

            return TradeDecision(
                action="exit",
                exit_price_cents=our_price,
                exit_time=decision_time,
                exit_reason=f"neighbor_price={neighbor_price}c >= {self.params.neighbor_price_min_c}c",
            )

        return TradeDecision(action="hold")


# Register the strategy
from . import register_strategy
register_strategy("open_maker_next_over", NextOverStrategy, NextOverParams)
