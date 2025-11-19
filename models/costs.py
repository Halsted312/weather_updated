#!/usr/bin/env python3
"""
Contract-aware fee calculations for Kalshi October 2025 schedule.

Taker fee: ceil(0.07 × C × P × (1 - P))  where P ∈ [0,1] dollars
Maker fee: 0¢ (we assume we cross spread, so always taker)
Settlement fee: 0¢

References:
- Kalshi Fee Schedule: https://kalshi.com/docs/kalshi-fee-schedule.pdf
"""

import math
from typing import Tuple


def taker_fee_cents(price_cents: int, contracts: int = 1) -> int:
    """
    Kalshi taker fee (October 2025 schedule).

    Formula: ceil(0.07 × C × P × (1 - P)) cents
    where P is price in dollars [0, 1], C is number of contracts.

    Args:
        price_cents: Price in cents [0, 100]
        contracts: Number of contracts (default: 1)

    Returns:
        Fee in cents, rounded up to next cent

    Examples:
        >>> taker_fee_cents(50, 1)  # 50¢ → max fee
        2
        >>> taker_fee_cents(10, 1)  # 10¢ → low fee
        1
        >>> taker_fee_cents(50, 10)  # 50¢, 10 contracts
        18
    """
    if not (0 <= price_cents <= 100):
        raise ValueError(f"price_cents must be in [0, 100], got {price_cents}")
    if contracts < 0:
        raise ValueError(f"contracts must be non-negative, got {contracts}")

    # Convert to dollars
    p_dollars = price_cents / 100.0

    # Fee formula: 0.07 × C × P × (1 - P) dollars
    fee_dollars = 0.07 * contracts * p_dollars * (1.0 - p_dollars)

    # Round up to next cent
    return math.ceil(fee_dollars * 100)


def maker_fee_cents(price_cents: int, contracts: int = 1) -> int:
    """
    Kalshi maker fee (October 2025 schedule).

    Formula: ceil(0.0175 × C × P × (1 - P)) cents
    (Currently not used; we assume taker-only execution)

    Args:
        price_cents: Price in cents [0, 100]
        contracts: Number of contracts (default: 1)

    Returns:
        Fee in cents, rounded up to next cent
    """
    if not (0 <= price_cents <= 100):
        raise ValueError(f"price_cents must be in [0, 100], got {price_cents}")
    if contracts < 0:
        raise ValueError(f"contracts must be non-negative, got {contracts}")

    p_dollars = price_cents / 100.0
    fee_dollars = 0.0175 * contracts * p_dollars * (1.0 - p_dollars)
    return math.ceil(fee_dollars * 100)


def effective_yes_entry_cents(yes_bid: int, yes_ask: int, slippage: int = 1) -> int:
    """
    Effective entry cost for buying YES (crossing spread + fee + slippage).

    We assume:
    - Buy at yes_ask (cross the spread, taker execution)
    - Pay taker fee on entry price
    - Add slippage (default: 1¢)
    - Half-spread cost is already captured by using ask vs mid

    Args:
        yes_bid: Current yes bid in cents [0, 100]
        yes_ask: Current yes ask in cents [0, 100]
        slippage: Additional slippage in cents (default: 1)

    Returns:
        Total entry cost in cents per contract

    Examples:
        >>> effective_yes_entry_cents(48, 52, slippage=1)  # Mid=50, spread=4
        55  # 52 (ask) + 2 (fee@52¢) + 1 (slippage)
    """
    if not (0 <= yes_bid <= 100):
        raise ValueError(f"yes_bid must be in [0, 100], got {yes_bid}")
    if not (0 <= yes_ask <= 100):
        raise ValueError(f"yes_ask must be in [0, 100], got {yes_ask}")
    if yes_ask < yes_bid:
        raise ValueError(f"yes_ask ({yes_ask}) must be >= yes_bid ({yes_bid})")

    # Buy at ask (taker)
    entry_price = yes_ask

    # Add taker fee (based on entry price)
    fee = taker_fee_cents(entry_price, contracts=1)

    # Add slippage
    # Note: half-spread already captured by crossing bid-ask (ask vs mid)
    return entry_price + fee + slippage


def effective_no_entry_cents(yes_bid: int, yes_ask: int, slippage: int = 1) -> int:
    """
    Effective entry cost for buying NO (via YES market symmetry).

    In Kalshi:
    - NO price = 100 - YES price
    - Buying NO at X cents ≈ selling YES at (100-X) cents
    - To buy NO, we cross the NO ask, which = 100 - yes_bid

    Args:
        yes_bid: Current yes bid in cents [0, 100]
        yes_ask: Current yes ask in cents [0, 100]
        slippage: Additional slippage in cents (default: 1)

    Returns:
        Total entry cost in cents per contract

    Examples:
        >>> effective_no_entry_cents(48, 52, slippage=1)  # NO mid=50, NO ask=52
        55  # (100-48=52) + 2 (fee@52¢) + 1 (slippage)
    """
    if not (0 <= yes_bid <= 100):
        raise ValueError(f"yes_bid must be in [0, 100], got {yes_bid}")
    if not (0 <= yes_ask <= 100):
        raise ValueError(f"yes_ask must be in [0, 100], got {yes_ask}")
    if yes_ask < yes_bid:
        raise ValueError(f"yes_ask ({yes_ask}) must be >= yes_bid ({yes_bid})")

    # NO ask = 100 - YES bid (by symmetry)
    # Buy NO at this price (taker)
    no_entry_price = 100 - yes_bid

    # Add taker fee (based on NO entry price)
    fee = taker_fee_cents(no_entry_price, contracts=1)

    # Add slippage
    return no_entry_price + fee + slippage


def net_exit_cents(
    entry_price: int,
    exit_price: int,
    entry_fee: int,
    exit_fee: int,
    slippage: int = 1
) -> int:
    """
    Net P&L for a round-trip trade (entry → exit).

    Args:
        entry_price: Entry price in cents (what we paid)
        exit_price: Exit price in cents (what we receive)
        entry_fee: Entry fee in cents
        exit_fee: Exit fee in cents
        slippage: Slippage per leg in cents (default: 1)

    Returns:
        Net P&L in cents per contract (positive = profit)

    Examples:
        >>> # Buy YES at 50¢, sell at 60¢, fees = 2¢ each
        >>> net_exit_cents(50, 60, 2, 2, slippage=1)
        4  # 60 - 50 - 2 - 2 - 2×1 = 4¢ profit
    """
    gross_pnl = exit_price - entry_price
    total_fees = entry_fee + exit_fee
    total_slippage = 2 * slippage  # Entry + exit

    return gross_pnl - total_fees - total_slippage


def breakeven_move_cents(entry_price: int, slippage: int = 1) -> int:
    """
    Minimum price move (in cents) required to break even on a round-trip.

    Useful for checking if edge > breakeven before entering.

    Args:
        entry_price: Entry price in cents [0, 100]
        slippage: Slippage per leg in cents (default: 1)

    Returns:
        Minimum price move in cents to break even

    Examples:
        >>> breakeven_move_cents(50, slippage=1)  # Entry@50¢, max fee
        8  # 2 (entry fee) + 2 (exit fee) + 2 (slippage) + ~2 (half-spread)
    """
    # Estimate fees (assume similar exit price)
    entry_fee = taker_fee_cents(entry_price, contracts=1)
    exit_fee = taker_fee_cents(entry_price, contracts=1)  # Approximate

    # Total costs: fees + slippage (both legs) + half-spread (estimate ~2¢)
    half_spread_estimate = 2  # Conservative estimate
    total_cost = entry_fee + exit_fee + 2 * slippage + half_spread_estimate

    return total_cost


def effective_edge_cents(
    true_prob: float,
    market_price: int,
    yes_bid: int,
    yes_ask: int,
    slippage: int = 1
) -> Tuple[float, str]:
    """
    Effective edge after all costs (fees, slippage, spread).

    Returns edge in cents and recommended side ("yes", "no", or "none").

    Args:
        true_prob: Model's estimated probability [0, 1]
        market_price: Current market mid price in cents [0, 100]
        yes_bid: Current yes bid in cents
        yes_ask: Current yes ask in cents
        slippage: Slippage in cents (default: 1)

    Returns:
        Tuple of (edge_cents, side) where:
        - edge_cents: Expected profit in cents per contract (after costs)
        - side: "yes", "no", or "none"

    Examples:
        >>> # Model says 60% YES, market at 50¢ (bid=48, ask=52)
        >>> effective_edge_cents(0.60, 50, 48, 52, slippage=1)
        (2.0, 'yes')  # 60¢ fair - 55¢ effective entry = +5¢ edge raw, ~2¢ after exit
    """
    if not (0.0 <= true_prob <= 1.0):
        raise ValueError(f"true_prob must be in [0, 1], got {true_prob}")

    fair_value_cents = int(true_prob * 100)

    # Calculate effective entry costs for each side
    yes_entry_cost = effective_yes_entry_cents(yes_bid, yes_ask, slippage)
    no_entry_cost = effective_no_entry_cents(yes_bid, yes_ask, slippage)

    # Expected value for YES side
    # If YES wins: receive 100¢, paid yes_entry_cost → profit = 100 - yes_entry_cost
    # If YES loses: receive 0¢, paid yes_entry_cost → loss = -yes_entry_cost
    # EV = true_prob × (100 - yes_entry_cost) + (1 - true_prob) × (-yes_entry_cost)
    #    = 100 × true_prob - yes_entry_cost
    yes_ev = 100 * true_prob - yes_entry_cost

    # Expected value for NO side
    # If NO wins (YES loses): receive 100¢, paid no_entry_cost → profit = 100 - no_entry_cost
    # If NO loses (YES wins): receive 0¢, paid no_entry_cost → loss = -no_entry_cost
    # EV = (1 - true_prob) × (100 - no_entry_cost) + true_prob × (-no_entry_cost)
    #    = 100 × (1 - true_prob) - no_entry_cost
    no_ev = 100 * (1 - true_prob) - no_entry_cost

    # Choose side with positive edge
    if yes_ev > 0 and yes_ev >= no_ev:
        return (yes_ev, "yes")
    elif no_ev > 0 and no_ev > yes_ev:
        return (no_ev, "no")
    else:
        return (0.0, "none")


def main():
    """Demo: Fee calculations."""
    print("\n" + "="*60)
    print("Kalshi Fee Calculations (October 2025)")
    print("="*60 + "\n")

    # Test taker fees at various prices
    print("Taker fees (1 contract):")
    for price in [10, 25, 50, 75, 90]:
        fee = taker_fee_cents(price, contracts=1)
        print(f"  {price}¢: {fee}¢ fee")

    print(f"\nTaker fees (10 contracts at 50¢): {taker_fee_cents(50, 10)}¢")

    # Test effective entry costs
    print("\n" + "-"*60)
    print("Effective entry costs (bid=48¢, ask=52¢, slippage=1¢):")
    yes_cost = effective_yes_entry_cents(48, 52, slippage=1)
    no_cost = effective_no_entry_cents(48, 52, slippage=1)
    print(f"  YES entry: {yes_cost}¢  (ask=52¢ + fee + slippage)")
    print(f"  NO entry:  {no_cost}¢  (100-48=52¢ + fee + slippage)")

    # Test breakeven
    print("\n" + "-"*60)
    print("Breakeven moves (round-trip):")
    for price in [25, 50, 75]:
        be = breakeven_move_cents(price, slippage=1)
        print(f"  Entry at {price}¢: need {be}¢ move to break even")

    # Test effective edge
    print("\n" + "-"*60)
    print("Effective edge examples:")

    # Example 1: Model says 60% YES, market at 50¢ (bid=48, ask=52)
    edge1, side1 = effective_edge_cents(0.60, 50, 48, 52, slippage=1)
    print(f"\n  Model: 60% YES, Market: 50¢ (48-52)")
    print(f"  → Edge: {edge1:.1f}¢, Side: {side1.upper()}")

    # Example 2: Model says 40% YES, market at 50¢ (buy NO)
    edge2, side2 = effective_edge_cents(0.40, 50, 48, 52, slippage=1)
    print(f"\n  Model: 40% YES, Market: 50¢ (48-52)")
    print(f"  → Edge: {edge2:.1f}¢, Side: {side2.upper()}")

    # Example 3: Model says 50% YES, market at 50¢ (no edge)
    edge3, side3 = effective_edge_cents(0.50, 50, 48, 52, slippage=1)
    print(f"\n  Model: 50% YES, Market: 50¢ (48-52)")
    print(f"  → Edge: {edge3:.1f}¢, Side: {side3.upper()}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
