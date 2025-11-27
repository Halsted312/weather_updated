"""
CurveGap hindsight adjustment strategy.

At decision time τ (e.g., 2-4h before predicted high), compare observed temperature
vs forecast curve. If obs is significantly above forecast AND trending up, shift
the bracket selection up by 1 bin for P&L calculation.

This is a "hindsight adjustment" for backtest - we compute P&L as if we had
entered the shifted bin. Same entry as base strategy, but P&L uses shifted bin
if thresholds are met.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import logging

import pandas as pd

from .base import StrategyParamsBase, StrategyBase, TradeContext, TradeDecision

logger = logging.getLogger(__name__)


# Minimum observation points required for reliable slope calculation
MIN_OBS_POINTS = 3


@dataclass
class CurveGapParams(StrategyParamsBase):
    """Parameters for the curve_gap hindsight adjustment strategy."""

    # Entry params (inherited)
    entry_price_cents: float = 30.0
    temp_bias_deg: float = 1.0
    basis_offset_days: int = 1
    bet_amount_usd: float = 100.0

    # Decision timing (minutes before predicted high, negative = before)
    decision_offset_min: int = -180  # 3h before

    # Curve gap thresholds
    delta_obs_fcst_min_deg: float = 1.5  # T_obs - T_fcst must be >= this
    slope_min_deg_per_hour: float = 0.5  # 1h slope must be >= this

    # Shift control
    max_shift_bins: int = 1  # Max bins to shift up


@dataclass
class CurveGapDecision(TradeDecision):
    """Extended TradeDecision with override_bin_index for curve_gap strategy."""
    override_bin_index: Optional[int] = None


class CurveGapStrategy(StrategyBase):
    """
    Hindsight adjustment strategy based on observation vs forecast curve.

    At decision_time (relative to predicted high hour):
    - Load minute observations
    - Compute T_obs (15-min average before τ)
    - Get T_fcst(τ) from hourly forecast (interpolated)
    - Compute slope_1h (linear fit over last hour of obs)
    - Check thresholds: if delta_obs_fcst >= threshold AND slope_1h >= threshold
      Then return override_bin_index = bin_index + shift

    The runner uses override_bin_index for P&L calculation if set.
    """

    def __init__(self, params: CurveGapParams):
        self.params = params

    @property
    def strategy_id(self) -> str:
        return "open_maker_curve_gap"

    def decide(
        self,
        context: TradeContext,
        candles_df: Optional[pd.DataFrame] = None,
        obs_stats: Optional[dict] = None,
        T_fcst_at_decision: Optional[float] = None,
    ) -> CurveGapDecision:
        """
        Decide whether to shift bin based on obs vs forecast curve.

        Args:
            context: Trade context with bracket info
            candles_df: Minute candles (not used, but required by interface)
            obs_stats: Dict from compute_obs_stats() with T_obs, slope_1h, n_points
            T_fcst_at_decision: Interpolated forecast temp at decision time

        Returns:
            CurveGapDecision with override_bin_index if shift conditions are met.
        """
        # Default: hold without shift
        base_decision = CurveGapDecision(action="hold", override_bin_index=None)

        # If missing observation data, return base decision (no shift)
        if obs_stats is None or not obs_stats:
            logger.debug(
                f"No obs stats for {context.city}/{context.event_date}, no shift"
            )
            return base_decision

        # If missing forecast, return base decision (no shift)
        if T_fcst_at_decision is None:
            logger.debug(
                f"No forecast temp for {context.city}/{context.event_date}, no shift"
            )
            return base_decision

        T_obs = obs_stats.get("T_obs")
        slope_1h = obs_stats.get("slope_1h", 0.0)
        n_points = obs_stats.get("n_points", 0)

        if T_obs is None:
            return base_decision

        # Require minimum observation points for reliable slope
        if n_points < MIN_OBS_POINTS:
            logger.debug(
                f"Insufficient obs points ({n_points} < {MIN_OBS_POINTS}) for "
                f"{context.city}/{context.event_date}, no shift"
            )
            return base_decision

        # Compute delta: obs - forecast
        delta_obs_fcst = T_obs - T_fcst_at_decision

        # Check thresholds
        meets_delta_threshold = delta_obs_fcst >= self.params.delta_obs_fcst_min_deg
        meets_slope_threshold = slope_1h >= self.params.slope_min_deg_per_hour

        if meets_delta_threshold and meets_slope_threshold:
            # Calculate shift amount (capped by max_shift_bins and available bins)
            max_possible_shift = context.total_bins - context.bin_index - 1
            shift_bins = min(self.params.max_shift_bins, max_possible_shift)

            if shift_bins > 0:
                override_idx = context.bin_index + shift_bins

                logger.info(
                    f"SHIFT triggered: {context.city}/{context.event_date} "
                    f"T_obs={T_obs:.1f}F, T_fcst={T_fcst_at_decision:.1f}F, "
                    f"delta={delta_obs_fcst:+.1f}F >= {self.params.delta_obs_fcst_min_deg}F, "
                    f"slope={slope_1h:.2f}F/h >= {self.params.slope_min_deg_per_hour}F/h, "
                    f"bin {context.bin_index} -> {override_idx} (n_points={n_points})"
                )

                return CurveGapDecision(
                    action="hold",
                    exit_reason=f"shift_up: delta={delta_obs_fcst:+.1f}F, slope={slope_1h:.2f}F/h",
                    override_bin_index=override_idx,
                )

        return base_decision


# Alias for backward compatibility with registry
CurveGapStrategyV2 = CurveGapStrategy

# Register the strategy
from . import register_strategy
register_strategy("open_maker_curve_gap", CurveGapStrategy, CurveGapParams)
