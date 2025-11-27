"""
Self-contained utilities for the open-maker strategy.

These are copied/adapted from backtest.midnight_heuristic to keep
the open_maker package independent and avoid cross-strategy coupling.
"""

import math
from datetime import date
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


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
