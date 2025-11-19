#!/usr/bin/env python3
"""
Kalshi fee calculator (October 2025 fee schedule).

Implements exact formulas from https://kalshi.com/docs/kalshi-fee-schedule.pdf

Fees:
- Taker (immediate match): 7.0% of C × P × (1-P), rounded up to next cent
- Maker (resting order): 1.75% of C × P × (1-P), rounded up to next cent
- Settlement: $0.00 (no settlement fee)

Where:
- C = number of contracts
- P = price (as fraction, e.g., 0.50 for 50¢)

Rounding: Always round UP to the next cent (math.ceil after converting to cents).

Examples from fee schedule:
- P=0.50, C=1 → Taker: $0.02, Maker: $0.01
- P=0.50, C=100 → Taker: $1.75, Maker: $0.44
- P=0.01 or P=0.99, C=1 → Taker: $0.01 (minimum)
"""

import math
from typing import Literal


# Fee rate constants (as of October 2025)
TAKER_RATE = 0.07  # 7.0%
MAKER_RATE = 0.0175  # 1.75%
SETTLEMENT_FEE_CENTS = 0  # No settlement fee


def taker_fee_cents(contracts: int, price_cents: int) -> int:
    """
    Calculate taker fee in cents (rounded up).

    Args:
        contracts: Number of contracts
        price_cents: Price in cents (0-100)

    Returns:
        Fee in cents (integer, rounded up)

    Examples:
        >>> taker_fee_cents(1, 50)
        2
        >>> taker_fee_cents(100, 50)
        175
        >>> taker_fee_cents(1, 1)
        1
        >>> taker_fee_cents(1, 99)
        1
    """
    if contracts <= 0:
        return 0

    # Convert price to fraction [0, 1]
    p = price_cents / 100.0

    # Calculate fee in dollars: 0.07 × C × P × (1-P)
    fee_dollars = TAKER_RATE * contracts * p * (1 - p)

    # Convert to cents and round UP (handle floating point precision)
    fee_cents_raw = fee_dollars * 100
    # Round to 10 decimal places to avoid float precision errors, then ceil
    fee_cents = math.ceil(round(fee_cents_raw, 10))

    return fee_cents


def maker_fee_cents(contracts: int, price_cents: int) -> int:
    """
    Calculate maker fee in cents (rounded up).

    Args:
        contracts: Number of contracts
        price_cents: Price in cents (0-100)

    Returns:
        Fee in cents (integer, rounded up)

    Examples:
        >>> maker_fee_cents(1, 50)
        1
        >>> maker_fee_cents(100, 50)
        44
        >>> maker_fee_cents(1, 1)
        1
        >>> maker_fee_cents(1, 99)
        1
    """
    if contracts <= 0:
        return 0

    # Convert price to fraction [0, 1]
    p = price_cents / 100.0

    # Calculate fee in dollars: 0.0175 × C × P × (1-P)
    fee_dollars = MAKER_RATE * contracts * p * (1 - p)

    # Convert to cents and round UP (handle floating point precision)
    fee_cents_raw = fee_dollars * 100
    # Round to 10 decimal places to avoid float precision errors, then ceil
    fee_cents = math.ceil(round(fee_cents_raw, 10))

    return fee_cents


def taker_fee_dollars(contracts: int, price_cents: int) -> float:
    """
    Calculate taker fee in dollars.

    Args:
        contracts: Number of contracts
        price_cents: Price in cents (0-100)

    Returns:
        Fee in dollars (float)

    Examples:
        >>> taker_fee_dollars(1, 50)
        0.02
        >>> taker_fee_dollars(100, 50)
        1.75
    """
    return taker_fee_cents(contracts, price_cents) / 100.0


def maker_fee_dollars(contracts: int, price_cents: int) -> float:
    """
    Calculate maker fee in dollars.

    Args:
        contracts: Number of contracts
        price_cents: Price in cents (0-100)

    Returns:
        Fee in dollars (float)

    Examples:
        >>> maker_fee_dollars(1, 50)
        0.01
        >>> maker_fee_dollars(100, 50)
        0.44
    """
    return maker_fee_cents(contracts, price_cents) / 100.0


def total_trade_cost_cents(
    contracts: int,
    price_cents: int,
    side: Literal["buy", "sell"],
    fee_type: Literal["taker", "maker"] = "taker",
) -> int:
    """
    Calculate total cost (price + fee) for a trade in cents.

    Args:
        contracts: Number of contracts
        price_cents: Price in cents (0-100)
        side: "buy" or "sell"
        fee_type: "taker" or "maker" (default: taker)

    Returns:
        Total cost in cents (contract cost + fee)

    Examples:
        >>> # Buy 1 contract at 50¢ with taker fee
        >>> total_trade_cost_cents(1, 50, "buy", "taker")
        52
        >>> # Sell 1 contract at 50¢ with taker fee (collect price, pay fee)
        >>> total_trade_cost_cents(1, 50, "sell", "taker")
        48
        >>> # Buy 100 contracts at 50¢ with maker fee
        >>> total_trade_cost_cents(100, 50, "buy", "maker")
        5044
    """
    # Calculate fee
    if fee_type == "taker":
        fee = taker_fee_cents(contracts, price_cents)
    else:
        fee = maker_fee_cents(contracts, price_cents)

    # Contract cost
    contract_cost = contracts * price_cents

    # Buy: pay price + fee
    # Sell: collect price - fee
    if side == "buy":
        return contract_cost + fee
    else:
        return contract_cost - fee


def net_payout_cents(
    contracts: int,
    entry_price_cents: int,
    exit_price_cents: int,
    entry_fee_type: Literal["taker", "maker"] = "taker",
    exit_fee_type: Literal["taker", "maker"] = "taker",
) -> int:
    """
    Calculate net P&L for a round-trip trade (buy then sell, or sell then buy).

    Args:
        contracts: Number of contracts
        entry_price_cents: Entry price in cents
        exit_price_cents: Exit price in cents
        entry_fee_type: Fee type for entry ("taker" or "maker")
        exit_fee_type: Fee type for exit ("taker" or "maker")

    Returns:
        Net P&L in cents (positive = profit, negative = loss)

    Examples:
        >>> # Buy at 40¢, sell at 60¢ (taker both sides: 2¢ fee each side)
        >>> net_payout_cents(1, 40, 60, "taker", "taker")
        16
        >>> # Buy at 50¢, sell at 50¢ (breakeven before fees, loss after)
        >>> net_payout_cents(1, 50, 50, "taker", "taker")
        -4
        >>> # Buy at 40¢, sell at 60¢ (maker both sides, 100 contracts)
        >>> net_payout_cents(100, 40, 60, "maker", "maker")
        1916
    """
    # Entry cost (negative for buy, positive for sell)
    entry_cost = -total_trade_cost_cents(contracts, entry_price_cents, "buy", entry_fee_type)

    # Exit proceeds (positive for sell, negative for buy cover)
    exit_proceeds = total_trade_cost_cents(contracts, exit_price_cents, "sell", exit_fee_type)

    # Net P&L
    return entry_cost + exit_proceeds


def settlement_payout_cents(contracts: int, result: Literal["YES", "NO"]) -> int:
    """
    Calculate settlement payout (no settlement fee).

    Args:
        contracts: Number of contracts held
        result: Market result ("YES" or "NO")

    Returns:
        Settlement payout in cents

    Examples:
        >>> settlement_payout_cents(100, "YES")
        10000
        >>> settlement_payout_cents(100, "NO")
        0
    """
    # YES contracts pay 100¢ each, NO contracts pay 0¢
    if result == "YES":
        return contracts * 100
    else:
        return 0


def breakeven_price_cents(
    entry_price_cents: int,
    entry_fee_type: Literal["taker", "maker"] = "taker",
    exit_fee_type: Literal["taker", "maker"] = "taker",
) -> int:
    """
    Calculate breakeven exit price needed to net zero P&L (1 contract).

    Args:
        entry_price_cents: Entry price in cents
        entry_fee_type: Fee type for entry
        exit_fee_type: Fee type for exit

    Returns:
        Breakeven exit price in cents

    Examples:
        >>> # Buy at 50¢ (taker), need to sell higher to break even (+4¢)
        >>> breakeven_price_cents(50, "taker", "taker")
        54
        >>> # Buy at 50¢ (maker), lower breakeven (+2¢)
        >>> breakeven_price_cents(50, "maker", "maker")
        52
    """
    # Binary search for breakeven price
    for exit_price in range(0, 101):
        pnl = net_payout_cents(1, entry_price_cents, exit_price, entry_fee_type, exit_fee_type)
        if pnl >= 0:
            return exit_price

    # Should never reach here (max exit price is 100¢)
    return 100


def main():
    """Demo: Calculate fees for common scenarios."""
    print("\n" + "="*60)
    print("Kalshi Fee Calculator (October 2025 Schedule)")
    print("="*60 + "\n")

    # Test cases from PDF
    test_cases = [
        (1, 50, "taker"),
        (100, 50, "taker"),
        (100, 50, "maker"),
        (1, 1, "taker"),
        (1, 99, "taker"),
        (50, 25, "taker"),
        (50, 75, "taker"),
    ]

    print("Fee Calculations:")
    print("-" * 60)
    print(f"{'Contracts':>10} {'Price':>8} {'Type':>8} {'Fee ($)':>10} {'Fee (¢)':>10}")
    print("-" * 60)

    for contracts, price_cents, fee_type in test_cases:
        if fee_type == "taker":
            fee_cents = taker_fee_cents(contracts, price_cents)
            fee_dollars = taker_fee_dollars(contracts, price_cents)
        else:
            fee_cents = maker_fee_cents(contracts, price_cents)
            fee_dollars = maker_fee_dollars(contracts, price_cents)

        print(f"{contracts:>10} {price_cents:>7}¢ {fee_type:>8} ${fee_dollars:>9.2f} {fee_cents:>9}¢")

    # Round-trip P&L examples
    print("\n" + "="*60)
    print("Round-Trip P&L Examples (1 contract, taker both sides)")
    print("="*60)
    print(f"{'Entry':>8} {'Exit':>8} {'Gross P&L':>12} {'Fees':>8} {'Net P&L':>10}")
    print("-" * 60)

    round_trips = [
        (40, 60),  # Profit
        (50, 50),  # Breakeven (loss after fees)
        (60, 40),  # Loss
        (25, 75),  # Large profit
    ]

    for entry, exit in round_trips:
        gross_pnl = exit - entry
        entry_fee = taker_fee_cents(1, entry)
        exit_fee = taker_fee_cents(1, exit)
        total_fees = entry_fee + exit_fee
        net_pnl = net_payout_cents(1, entry, exit, "taker", "taker")

        print(f"{entry:>7}¢ {exit:>7}¢ {gross_pnl:>11}¢ {total_fees:>7}¢ {net_pnl:>9}¢")

    # Breakeven analysis
    print("\n" + "="*60)
    print("Breakeven Analysis (1 contract)")
    print("="*60)
    print(f"{'Entry':>8} {'Fee Type':>12} {'Breakeven':>12} {'Move Needed':>15}")
    print("-" * 60)

    breakeven_cases = [
        (50, "taker", "taker"),
        (50, "maker", "maker"),
        (25, "taker", "taker"),
        (75, "taker", "taker"),
    ]

    for entry, entry_fee, exit_fee in breakeven_cases:
        breakeven = breakeven_price_cents(entry, entry_fee, exit_fee)
        move = breakeven - entry

        print(f"{entry:>7}¢ {entry_fee:>12} {breakeven:>11}¢ {move:>14}¢")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
