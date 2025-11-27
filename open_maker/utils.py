"""
Self-contained utilities for the open-maker strategy.

These are copied/adapted from backtest.midnight_heuristic to keep
the open_maker package independent and avoid cross-strategy coupling.
"""

import logging
import math
from datetime import date, datetime, timedelta
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# City Timezone Mapping
# =============================================================================

CITY_TIMEZONES = {
    "chicago": ZoneInfo("America/Chicago"),
    "austin": ZoneInfo("America/Chicago"),
    "denver": ZoneInfo("America/Denver"),
    "los_angeles": ZoneInfo("America/Los_Angeles"),
    "miami": ZoneInfo("America/New_York"),
    "philadelphia": ZoneInfo("America/New_York"),
}


def get_city_timezone(city: str) -> ZoneInfo:
    """Get timezone for a city."""
    if city not in CITY_TIMEZONES:
        raise ValueError(f"Unknown city: {city}. Available: {list(CITY_TIMEZONES.keys())}")
    return CITY_TIMEZONES[city]


# =============================================================================
# Kalshi Fee Functions
# =============================================================================

def kalshi_taker_fee(price_cents: float, num_contracts: int = 1) -> float:
    """
    Calculate Kalshi taker fee.

    Fee per contract = 0.07 * price * (100 - price) / 100
    Max fee = $1.74 per contract (when price = 50)

    Args:
        price_cents: Price in cents (0-100)
        num_contracts: Number of contracts

    Returns:
        Total fee in dollars
    """
    if price_cents <= 0 or price_cents >= 100:
        return 0.0
    fee_per_contract = 0.07 * price_cents * (100 - price_cents) / 100 / 100
    return fee_per_contract * num_contracts


def kalshi_maker_fee(price_cents: float, num_contracts: int = 1) -> float:
    """
    Calculate Kalshi maker fee.

    For weather markets, maker fee is $0.
    For markets with maker fees: ceil(0.0175 * C * P * (1-P))

    Args:
        price_cents: Price in cents (0-100)
        num_contracts: Number of contracts

    Returns:
        Total fee in dollars (0.0 for weather markets)
    """
    # Weather markets have no maker fees
    return 0.0


# =============================================================================
# Bracket Selection
# =============================================================================

def find_bracket_for_temp(
    markets_df: pd.DataFrame,
    event_date: date,
    temp: float,
) -> Optional[Tuple[str, Optional[float], Optional[float]]]:
    """
    Find the bracket containing a temperature.

    Priority: between > less/less_or_equal > greater/greater_or_equal
    This ensures we match specific brackets before tail brackets.

    Args:
        markets_df: DataFrame with columns [ticker, event_date, strike_type, floor_strike, cap_strike]
        event_date: The event date to filter markets
        temp: Temperature to find bracket for

    Returns:
        (ticker, floor_strike, cap_strike) or None if no matching bracket
    """
    day_markets = markets_df[markets_df["event_date"] == event_date]

    if day_markets.empty:
        return None

    # First pass: check "between" brackets (the specific temperature ranges)
    for _, row in day_markets.iterrows():
        if row["strike_type"] == "between":
            floor_s = row["floor_strike"]
            cap_s = row["cap_strike"]
            if floor_s is not None and cap_s is not None and floor_s <= temp < cap_s:
                return row["ticker"], floor_s, cap_s

    # Second pass: check tail brackets only if no between bracket matched
    # Low tail: "less" or "less_or_equal" (YES if below some threshold)
    for _, row in day_markets.iterrows():
        strike_type = row["strike_type"]
        cap_s = row["cap_strike"]

        if strike_type == "less":
            if cap_s is not None and temp < cap_s:
                return row["ticker"], None, cap_s
        elif strike_type == "less_or_equal":
            if cap_s is not None and temp <= cap_s:
                return row["ticker"], None, cap_s

    # High tail: "greater" or "greater_or_equal" (YES if above some threshold)
    for _, row in day_markets.iterrows():
        strike_type = row["strike_type"]
        floor_s = row["floor_strike"]

        if strike_type == "greater":
            # Only match if cap_strike is NULL (true tail)
            if floor_s is not None and row["cap_strike"] is None and temp >= floor_s:
                return row["ticker"], floor_s, None
        elif strike_type == "greater_or_equal":
            if floor_s is not None and temp > floor_s:
                return row["ticker"], floor_s, None

    return None


def determine_winning_bracket(
    markets_df: pd.DataFrame,
    event_date: date,
    tmax_final: float,
) -> Optional[str]:
    """
    Determine which bracket won based on actual settlement temperature.

    Args:
        markets_df: DataFrame with market data
        event_date: The event date
        tmax_final: Actual high temperature

    Returns:
        Ticker of the winning bracket, or None
    """
    result = find_bracket_for_temp(markets_df, event_date, tmax_final)
    return result[0] if result else None


# =============================================================================
# Position Sizing
# =============================================================================

def calculate_position_size(
    entry_price_cents: float,
    bet_amount_usd: float,
) -> Tuple[int, float]:
    """
    Calculate number of contracts and actual cost.

    Args:
        entry_price_cents: Entry price in cents (0-100)
        bet_amount_usd: Target bet amount in USD

    Returns:
        (num_contracts, actual_cost_usd)
    """
    cost_per_contract = entry_price_cents / 100  # Convert cents to dollars
    if cost_per_contract <= 0:
        return 0, 0.0

    num_contracts = int(bet_amount_usd / cost_per_contract)
    if num_contracts < 1:
        num_contracts = 1

    actual_cost = num_contracts * cost_per_contract
    return num_contracts, actual_cost


def calculate_pnl(
    entry_price_cents: float,
    num_contracts: int,
    bin_won: bool,
    fee_usd: float,
) -> float:
    """
    Calculate P&L for a trade.

    Args:
        entry_price_cents: Entry price in cents (0-100)
        num_contracts: Number of contracts
        bin_won: Whether the bracket resolved YES
        fee_usd: Fee paid in USD

    Returns:
        Net P&L in USD
    """
    entry_price = entry_price_cents / 100  # Convert to dollars

    if bin_won:
        # Win: receive $1 per contract, paid entry_price per contract
        pnl_gross = num_contracts * (1.0 - entry_price)
    else:
        # Lose: contract expires worthless, lose entry cost
        pnl_gross = -num_contracts * entry_price

    return pnl_gross - fee_usd


def calculate_exit_pnl(
    entry_price_cents: float,
    exit_price_cents: float,
    num_contracts: int,
    exit_fee_usd: float,
) -> float:
    """
    Calculate P&L for an early exit.

    Args:
        entry_price_cents: Entry price in cents (0-100)
        exit_price_cents: Exit price in cents (0-100)
        num_contracts: Number of contracts
        exit_fee_usd: Taker fee for exiting

    Returns:
        Net P&L in USD
    """
    entry_price = entry_price_cents / 100
    exit_price = exit_price_cents / 100
    pnl_gross = num_contracts * (exit_price - entry_price)
    return pnl_gross - exit_fee_usd


# =============================================================================
# Predicted High Hour Lookup
# =============================================================================

def get_predicted_high_hour(
    session,
    city: str,
    event_date: date,
) -> Optional[float]:
    """
    Get predicted high hour from yesterday's forecast for today.

    Looks up wx.forecast_snapshot_hourly for basis_date = event_date - 1
    and finds the hour with max temperature for the target_date = event_date.

    Args:
        session: SQLAlchemy session
        city: City ID
        event_date: The target event date

    Returns:
        Hour of day (0.0 - 23.0) when max temp is predicted, or None if not found
    """
    from sqlalchemy import select, func
    from src.db.models import WxForecastSnapshotHourly

    basis_date = event_date - timedelta(days=1)

    # Get hourly forecasts for the event_date from yesterday's basis
    query = select(
        WxForecastSnapshotHourly.target_hour_local,
        WxForecastSnapshotHourly.temp_fcst_f,
    ).where(
        WxForecastSnapshotHourly.city == city,
        WxForecastSnapshotHourly.basis_date == basis_date,
        func.date(WxForecastSnapshotHourly.target_hour_local) == event_date,
    ).order_by(WxForecastSnapshotHourly.target_hour_local)

    result = session.execute(query)
    rows = result.fetchall()

    if not rows:
        logger.debug(f"No hourly forecast for {city}/{event_date} from basis {basis_date}")
        return None

    # Find hour with max temp
    max_temp = None
    max_hour = None
    for target_hour_local, temp_f in rows:
        if temp_f is not None and (max_temp is None or temp_f > max_temp):
            max_temp = temp_f
            max_hour = target_hour_local

    if max_hour is None:
        return None

    # Extract hour of day as float
    # target_hour_local is a datetime - extract hour + minute/60
    hour_of_day = max_hour.hour + max_hour.minute / 60.0
    return hour_of_day


def compute_decision_time_utc(
    city: str,
    event_date: date,
    predicted_high_hour: float,
    offset_minutes: int,
) -> datetime:
    """
    Compute decision time in UTC.

    Args:
        city: City ID (for timezone)
        event_date: The event date
        predicted_high_hour: Hour of day (e.g., 15.0 = 3pm)
        offset_minutes: Minutes relative to predicted high (negative = before)

    Returns:
        Decision time as UTC datetime
    """
    tz = get_city_timezone(city)

    # Create local datetime for predicted high
    hour = int(predicted_high_hour)
    minute = int((predicted_high_hour - hour) * 60)
    local_high = datetime(
        event_date.year, event_date.month, event_date.day,
        hour, minute, tzinfo=tz
    )

    # Apply offset
    decision_local = local_high + timedelta(minutes=offset_minutes)

    # Convert to UTC
    decision_utc = decision_local.astimezone(ZoneInfo("UTC"))
    return decision_utc


# =============================================================================
# Candle Price Lookup with Fallbacks
# =============================================================================

def get_candle_price_for_exit(
    candles_df: pd.DataFrame,
    ticker: str,
    decision_time: datetime,
    window_minutes: int = 5,
) -> Optional[float]:
    """
    Get price for exit/fill from candle data.

    Uses fallback chain:
    1. yes_bid_c (best bid - what we hit to exit)
    2. mid = (yes_bid_c + yes_ask_c) / 2 if both available
    3. close_c (last trade)
    4. None if all missing

    Args:
        candles_df: DataFrame with candle data
        ticker: Market ticker
        decision_time: Target time (UTC)
        window_minutes: Look-back window for finding candles

    Returns:
        Price in cents, or None if not found
    """
    if candles_df.empty:
        return None

    # Filter to this ticker
    ticker_candles = candles_df[candles_df["ticker"] == ticker].copy()
    if ticker_candles.empty:
        return None

    # Ensure bucket_start is timezone-aware
    if ticker_candles["bucket_start"].dt.tz is None:
        ticker_candles["bucket_start"] = ticker_candles["bucket_start"].dt.tz_localize("UTC")

    # Filter to window around decision time
    window_start = decision_time - timedelta(minutes=window_minutes)
    mask = (ticker_candles["bucket_start"] >= window_start) & (ticker_candles["bucket_start"] <= decision_time)
    window_candles = ticker_candles[mask]

    if window_candles.empty:
        return None

    # Get most recent candle in window
    latest = window_candles.sort_values("bucket_start").iloc[-1]

    # Fallback chain
    # 1. Try yes_bid_c first
    if pd.notna(latest.get("yes_bid_c")):
        return float(latest["yes_bid_c"])

    # 2. Try mid if both bid and ask available
    bid = latest.get("yes_bid_c")
    ask = latest.get("yes_ask_c")
    if pd.notna(bid) and pd.notna(ask):
        return float((bid + ask) / 2)

    # 3. Fall back to close_c
    if pd.notna(latest.get("close_c")):
        return float(latest["close_c"])

    return None


def get_candle_price_for_signal(
    candles_df: pd.DataFrame,
    ticker: str,
    decision_time: datetime,
    window_minutes: int = 5,
) -> Optional[float]:
    """
    Get price for signal (neighbor bracket comparison).

    Simpler fallback chain:
    1. yes_bid_c
    2. close_c

    Args:
        candles_df: DataFrame with candle data
        ticker: Market ticker
        decision_time: Target time (UTC)
        window_minutes: Look-back window for finding candles

    Returns:
        Price in cents, or None if not found
    """
    if candles_df.empty:
        return None

    # Filter to this ticker
    ticker_candles = candles_df[candles_df["ticker"] == ticker].copy()
    if ticker_candles.empty:
        return None

    # Ensure bucket_start is timezone-aware
    if ticker_candles["bucket_start"].dt.tz is None:
        ticker_candles["bucket_start"] = ticker_candles["bucket_start"].dt.tz_localize("UTC")

    # Filter to window around decision time
    window_start = decision_time - timedelta(minutes=window_minutes)
    mask = (ticker_candles["bucket_start"] >= window_start) & (ticker_candles["bucket_start"] <= decision_time)
    window_candles = ticker_candles[mask]

    if window_candles.empty:
        return None

    # Get most recent candle in window
    latest = window_candles.sort_values("bucket_start").iloc[-1]

    # Fallback chain
    # 1. Try yes_bid_c first
    if pd.notna(latest.get("yes_bid_c")):
        return float(latest["yes_bid_c"])

    # 2. Fall back to close_c
    if pd.notna(latest.get("close_c")):
        return float(latest["close_c"])

    return None


# =============================================================================
# Bracket Index Helpers
# =============================================================================

def get_bracket_index(
    markets_df: pd.DataFrame,
    event_date: date,
    ticker: str,
) -> Tuple[int, int, pd.DataFrame]:
    """
    Get the index of a bracket in sorted order.

    Brackets are sorted by floor_strike (NULL first for low tail).

    Args:
        markets_df: DataFrame with market data
        event_date: Event date to filter
        ticker: Ticker to find

    Returns:
        (index, total_count, sorted_brackets_df)
    """
    day_markets = markets_df[markets_df["event_date"] == event_date].copy()

    # Sort by floor_strike (NULL first for low tail bracket)
    sorted_markets = day_markets.sort_values(
        "floor_strike", na_position="first"
    ).reset_index(drop=True)

    total = len(sorted_markets)

    # Find index of our ticker
    idx_mask = sorted_markets["ticker"] == ticker
    if not idx_mask.any():
        return -1, total, sorted_markets

    idx = sorted_markets[idx_mask].index[0]
    return idx, total, sorted_markets


def get_neighbor_ticker(
    sorted_brackets: pd.DataFrame,
    current_index: int,
    direction: str = "up",
) -> Optional[str]:
    """
    Get the ticker of a neighboring bracket.

    Args:
        sorted_brackets: Brackets sorted by strike
        current_index: Current bracket index
        direction: "up" (higher strike) or "down" (lower strike)

    Returns:
        Neighbor ticker or None if at edge
    """
    if direction == "up":
        neighbor_idx = current_index + 1
    else:
        neighbor_idx = current_index - 1

    if neighbor_idx < 0 or neighbor_idx >= len(sorted_brackets):
        return None

    return sorted_brackets.iloc[neighbor_idx]["ticker"]
