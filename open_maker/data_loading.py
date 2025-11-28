"""
Data loading functions for open-maker backtests.

This module handles all database queries and file loading for:
- Forecast data (WxForecastSnapshot)
- Settlement data (WxSettlement)
- Market data (KalshiMarket)
- Candle data (KalshiCandle1m)
- Tuned parameters from JSON files
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from sqlalchemy import select

from src.db.models import (
    WxForecastSnapshot,
    WxSettlement,
    KalshiMarket,
    KalshiCandle1m,
)

# Import strategy param classes
from .strategies.base import StrategyParamsBase, OpenMakerParams
from .strategies.next_over import NextOverParams
from .strategies.curve_gap import CurveGapParams

logger = logging.getLogger(__name__)


def load_tuned_params(strategy_id: str, bet_amount_usd: float = 200.0) -> Optional[StrategyParamsBase]:
    """
    Load tuned parameters from JSON file created by optuna_tuner.

    Args:
        strategy_id: Strategy identifier (e.g., "open_maker_base")
        bet_amount_usd: Bet amount to use (not saved in tuned params)

    Returns:
        Params object for the strategy, or None if not found
    """
    config_dir = Path(__file__).parent.parent / "config"
    param_path = config_dir / f"{strategy_id}_best_params.json"

    if not param_path.exists():
        logger.debug(f"No tuned params found at {param_path}")
        return None

    try:
        with param_path.open("r") as f:
            data = json.load(f)

        params = data.get("params", {})

        if strategy_id == "open_maker_base":
            return OpenMakerParams(
                entry_price_cents=params.get("entry_price_cents", 50.0),
                temp_bias_deg=params.get("temp_bias_deg", 0.0),
                basis_offset_days=params.get("basis_offset_days", 1),
                bet_amount_usd=bet_amount_usd,
            )
        elif strategy_id == "open_maker_next_over":
            return NextOverParams(
                entry_price_cents=params.get("entry_price_cents", 40.0),
                temp_bias_deg=params.get("temp_bias_deg", 0.0),
                basis_offset_days=params.get("basis_offset_days", 1),
                bet_amount_usd=bet_amount_usd,
                decision_offset_min=params.get("decision_offset_min", -180),
                neighbor_price_min_c=params.get("neighbor_price_min_c", 50),
                our_price_max_c=params.get("our_price_max_c", 30),
            )
        elif strategy_id == "open_maker_curve_gap":
            return CurveGapParams(
                entry_price_cents=params.get("entry_price_cents", 30.0),
                temp_bias_deg=params.get("temp_bias_deg", 0.0),
                basis_offset_days=params.get("basis_offset_days", 1),
                bet_amount_usd=bet_amount_usd,
                decision_offset_min=params.get("decision_offset_min", -180),
                delta_obs_fcst_min_deg=params.get("delta_obs_fcst_min_deg", 1.5),
                slope_min_deg_per_hour=params.get("slope_min_deg_per_hour", 0.5),
                max_shift_bins=params.get("max_shift_bins", 1),
            )
        else:
            logger.warning(f"Unknown strategy for tuned params: {strategy_id}")
            return None

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load tuned params from {param_path}: {e}")
        return None


def load_forecast_data(
    session,
    city: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Load forecast data for a city and date range.

    For the open-maker strategy, we need the forecast available at market open,
    which is typically the previous day's forecast (lead_days=1).

    Returns DataFrame with columns:
        target_date, basis_date, tempmax_fcst_f, lead_days
    """
    query = select(
        WxForecastSnapshot.target_date,
        WxForecastSnapshot.basis_date,
        WxForecastSnapshot.tempmax_fcst_f,
        WxForecastSnapshot.lead_days,
    ).where(
        WxForecastSnapshot.city == city,
        WxForecastSnapshot.target_date.between(start_date, end_date),
        WxForecastSnapshot.lead_days.in_([0, 1, 2]),  # Get recent forecasts
    ).order_by(WxForecastSnapshot.target_date, WxForecastSnapshot.lead_days)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=[
        "target_date", "basis_date", "tempmax_fcst_f", "lead_days"
    ])
    return df


def load_settlement_data(
    session,
    city: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load settlement data (actual high temps)."""
    query = select(
        WxSettlement.date_local.label("event_date"),
        WxSettlement.tmax_final,
    ).where(
        WxSettlement.city == city,
        WxSettlement.date_local.between(start_date, end_date),
    ).order_by(WxSettlement.date_local)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=["event_date", "tmax_final"])
    return df


def load_market_data(
    session,
    city: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Load market metadata (brackets) including listed_at timestamp."""
    query = select(
        KalshiMarket.ticker,
        KalshiMarket.event_date,
        KalshiMarket.strike_type,
        KalshiMarket.floor_strike,
        KalshiMarket.cap_strike,
        KalshiMarket.result,
        KalshiMarket.listed_at,
    ).where(
        KalshiMarket.city == city,
        KalshiMarket.event_date.between(start_date, end_date),
    ).order_by(KalshiMarket.event_date, KalshiMarket.floor_strike)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=[
        "ticker", "event_date", "strike_type", "floor_strike", "cap_strike", "result", "listed_at"
    ])
    return df


def load_candle_data(
    session,
    tickers: List[str],
    start_time: datetime,
    end_time: datetime,
) -> pd.DataFrame:
    """
    Load minute candle data for a set of tickers in a time range.

    Args:
        session: SQLAlchemy session
        tickers: List of market tickers
        start_time: Start time (UTC)
        end_time: End time (UTC)

    Returns:
        DataFrame with candle data
    """
    if not tickers:
        return pd.DataFrame()

    query = select(
        KalshiCandle1m.ticker,
        KalshiCandle1m.bucket_start,
        KalshiCandle1m.close_c,
        KalshiCandle1m.yes_bid_c,
        KalshiCandle1m.yes_ask_c,
    ).where(
        KalshiCandle1m.ticker.in_(tickers),
        KalshiCandle1m.bucket_start.between(start_time, end_time),
    ).order_by(KalshiCandle1m.bucket_start)

    result = session.execute(query)
    df = pd.DataFrame(result.fetchall(), columns=[
        "ticker", "bucket_start", "close_c", "yes_bid_c", "yes_ask_c"
    ])
    return df


def get_forecast_at_open(
    forecast_df: pd.DataFrame,
    event_date: date,
    basis_offset_days: int = 1,
) -> Optional[Tuple[date, float]]:
    """
    Get the forecast for event_date using the specified basis offset.

    Args:
        forecast_df: DataFrame with forecast data
        event_date: The event date (target_date)
        basis_offset_days: How many days before event_date to get forecast from
                          1 = previous day's forecast (lead_days=1)
                          0 = same day forecast (lead_days=0)

    Returns:
        (basis_date, tempmax_fcst_f) or None if not found
    """
    # Filter for this event_date
    day_forecasts = forecast_df[forecast_df["target_date"] == event_date]

    if day_forecasts.empty:
        return None

    # Look for forecast with the right lead_days
    target_lead_days = basis_offset_days
    matching = day_forecasts[day_forecasts["lead_days"] == target_lead_days]

    if not matching.empty:
        row = matching.iloc[0]
        return row["basis_date"], float(row["tempmax_fcst_f"])

    # Fallback: use whatever forecast is available
    row = day_forecasts.iloc[0]
    return row["basis_date"], float(row["tempmax_fcst_f"])
