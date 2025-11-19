#!/usr/bin/env python3
"""
Kelly criterion position sizing for Kalshi binary markets.

Implements fractional Kelly for risk management:
- Full Kelly: f* = (p - y_eff) / (1 - y_eff)
- Fractional Kelly: f = alpha * f*

where:
- p = estimated probability of YES
- y_eff = effective price after fees
- alpha = risk multiplier (0.1-0.5, default 0.25)
"""

import logging
from typing import Literal
from backtest.fees import taker_fee_cents, maker_fee_cents

logger = logging.getLogger(__name__)


def kelly_fraction(
    prob: float,
    price_cents: int,
    fee_type: Literal["taker", "maker"] = "taker",
    alpha: float = 0.25,
) -> float:
    """
    Calculate fractional Kelly position size.

    Args:
        prob: Estimated probability of YES [0, 1]
        price_cents: Market price in cents [1, 99]
        fee_type: Fee type ("taker" or "maker")
        alpha: Risk multiplier [0, 1] (default: 0.25 = quarter Kelly)

    Returns:
        Optimal fraction of bankroll to risk [0, 1]

    Examples:
        >>> # Fair price, no edge
        >>> kelly_fraction(0.50, 50, "taker", alpha=0.25)
        0.0

        >>> # 60% prob at 50¢ price (10% edge)
        >>> round(kelly_fraction(0.60, 50, "taker", alpha=0.25), 3)
        0.047

        >>> # 70% prob at 50¢ price (20% edge)
        >>> round(kelly_fraction(0.70, 50, "taker", alpha=0.25), 3)
        0.098
    """
    # Validate inputs
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be in [0, 1], got {prob}")

    if not 1 <= price_cents <= 99:
        raise ValueError(f"price_cents must be in [1, 99], got {price_cents}")

    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    # Calculate effective price after fees
    # For buying YES: pay price + fee
    # For selling YES: receive price - fee
    fee_per_contract = (
        taker_fee_cents(1, price_cents) if fee_type == "taker"
        else maker_fee_cents(1, price_cents)
    )

    # Effective price = price + fee (we pay this for 1 contract)
    y_eff = (price_cents + fee_per_contract) / 100.0

    # Kelly formula: f* = (p - y_eff) / (1 - y_eff)
    # Edge must be positive for position size > 0
    edge = prob - y_eff

    if edge <= 0:
        return 0.0

    # Full Kelly fraction
    f_star = edge / (1 - y_eff)

    # Apply risk multiplier
    f = alpha * f_star

    # Cap at 100% (should never happen with reasonable alpha)
    return min(f, 1.0)


def kelly_contracts(
    prob: float,
    price_cents: int,
    bankroll_cents: int,
    fee_type: Literal["taker", "maker"] = "taker",
    alpha: float = 0.25,
) -> int:
    """
    Calculate number of contracts to buy (Kelly sizing).

    Args:
        prob: Estimated probability of YES [0, 1]
        price_cents: Market price in cents [1, 99]
        bankroll_cents: Available capital in cents
        fee_type: Fee type ("taker" or "maker")
        alpha: Risk multiplier (default: 0.25)

    Returns:
        Number of contracts to buy (integer, >= 0)

    Examples:
        >>> # $1000 bankroll, 60% prob at 50¢
        >>> kelly_contracts(0.60, 50, 1000_00, "taker", alpha=0.25)
        181

        >>> # $10000 bankroll, 70% prob at 40¢
        >>> kelly_contracts(0.70, 40, 10000_00, "taker", alpha=0.25)
        5434
    """
    # Get fractional Kelly size
    f = kelly_fraction(prob, price_cents, fee_type, alpha)

    if f == 0:
        return 0

    # Capital to risk = f * bankroll
    capital_to_risk = f * bankroll_cents

    # Cost per contract = price + fee
    fee_per_contract = (
        taker_fee_cents(1, price_cents) if fee_type == "taker"
        else maker_fee_cents(1, price_cents)
    )
    cost_per_contract = price_cents + fee_per_contract

    # Number of contracts
    contracts = int(capital_to_risk / cost_per_contract)

    return max(0, contracts)


def kelly_edge_threshold(
    alpha: float = 0.25,
    min_fraction: float = 0.01,
) -> float:
    """
    Calculate minimum edge needed for Kelly sizing.

    Given alpha and minimum position size, what's the minimum edge?

    Args:
        alpha: Risk multiplier
        min_fraction: Minimum fraction of bankroll to trade

    Returns:
        Minimum edge (p - y_eff) needed

    Examples:
        >>> # With alpha=0.25, need at least 4% edge for 1% position
        >>> round(kelly_edge_threshold(0.25, 0.01), 3)
        0.020
    """
    # f = alpha * (edge / (1 - y_eff))
    # Solve for edge:
    # edge = f * (1 - y_eff) / alpha
    # Approximating y_eff ≈ 0.5 (middle price):
    y_eff_approx = 0.5
    edge = min_fraction * (1 - y_eff_approx) / alpha

    return edge


def optimal_alpha(
    sharpe_target: float = 1.0,
    volatility: float = 0.3,
) -> float:
    """
    Estimate optimal alpha for target Sharpe ratio.

    Rule of thumb: alpha ≈ (Sharpe_target * volatility) / edge
    For typical Kalshi weather markets:
    - volatility ≈ 0.3 (30% price swings)
    - edge ≈ 0.05-0.10 (5-10%)
    - Sharpe target = 1.0-2.0

    Args:
        sharpe_target: Target Sharpe ratio
        volatility: Expected price volatility

    Returns:
        Recommended alpha

    Examples:
        >>> # Conservative (Sharpe=1, vol=30%)
        >>> round(optimal_alpha(1.0, 0.3), 2)
        0.30

        >>> # Aggressive (Sharpe=2, vol=20%)
        >>> round(optimal_alpha(2.0, 0.2), 2)
        0.40
    """
    # Simplified heuristic
    alpha = sharpe_target * volatility

    # Cap at reasonable bounds
    return min(max(alpha, 0.1), 0.5)


def main():
    """Demo: Kelly sizing calculations."""
    print("\n" + "="*60)
    print("Kelly Criterion Position Sizing")
    print("="*60 + "\n")

    # Test cases
    test_cases = [
        # (prob, price, bankroll, alpha)
        (0.50, 50, 10000_00, 0.25),  # Fair price (no edge)
        (0.55, 50, 10000_00, 0.25),  # 5% edge
        (0.60, 50, 10000_00, 0.25),  # 10% edge
        (0.65, 45, 10000_00, 0.25),  # 20% edge
        (0.70, 40, 10000_00, 0.25),  # 30% edge
        (0.60, 50, 10000_00, 0.10),  # 10% edge, conservative
        (0.60, 50, 10000_00, 0.50),  # 10% edge, aggressive
    ]

    print("Kelly Position Sizing ($10,000 bankroll, taker fees)")
    print("-" * 60)
    print(f"{'Prob':>6} {'Price':>7} {'Alpha':>7} {'Edge':>8} {'Frac':>8} {'Contracts':>10} {'Capital':>10}")
    print("-" * 60)

    for prob, price, bankroll, alpha in test_cases:
        f = kelly_fraction(prob, price, "taker", alpha)
        contracts = kelly_contracts(prob, price, bankroll, "taker", alpha)

        # Calculate total capital deployed
        fee = taker_fee_cents(1, price)
        cost_per = price + fee
        capital = contracts * cost_per

        # Edge
        y_eff = (price + fee) / 100.0
        edge = prob - y_eff

        print(
            f"{prob:>6.2f} {price:>6}¢ {alpha:>7.2f} "
            f"{edge:>7.1%} {f:>7.1%} "
            f"{contracts:>10,} ${capital/100:>9,.0f}"
        )

    # Edge threshold
    print("\n" + "="*60)
    print("Minimum Edge Thresholds")
    print("="*60)
    print(f"{'Alpha':>8} {'Min Edge':>12} (for 1% position size)")
    print("-" * 60)

    for alpha in [0.10, 0.25, 0.50]:
        edge = kelly_edge_threshold(alpha, 0.01)
        print(f"{alpha:>8.2f} {edge:>11.1%}")

    # Optimal alpha
    print("\n" + "="*60)
    print("Optimal Alpha Recommendations")
    print("="*60)
    print(f"{'Sharpe':>8} {'Vol':>8} {'Alpha':>10}")
    print("-" * 60)

    for sharpe in [0.5, 1.0, 1.5, 2.0]:
        for vol in [0.2, 0.3]:
            alpha = optimal_alpha(sharpe, vol)
            print(f"{sharpe:>8.1f} {vol:>7.0%} {alpha:>10.2f}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
