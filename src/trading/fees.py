"""
Kalshi Fee Model

Implements the actual Kalshi fee formula, not the simplified version.

Official formula:
    taker_fee = ceil(0.07 × P × N × (1 - P) × 100) / 100

Where:
    P = price in dollars [0, 1]
    N = number of contracts

Key insights:
- Fee is ~1.75¢ per contract at 50¢ price
- Fee is much smaller at edges (0¢ near 1¢, 10¢ near 99¢)
- Per-contract rounding means small trades get hit harder
- Maker fees are typically 0% for weather markets (but can vary)
"""

import math
from typing import Tuple


def taker_fee_per_contract(price_cents: int) -> int:
    """
    Calculate Kalshi taker fee for a single contract.

    Args:
        price_cents: Price in cents [1, 99]

    Returns:
        Fee in cents per contract (integer)
    """
    P = price_cents / 100.0  # Convert to dollars
    # Official formula gives fee in dollars: 0.07 * P * (1-P)
    # Convert to cents and round up
    fee_dollars_raw = 0.07 * P * (1 - P) * 100
    fee_cents = math.ceil(fee_dollars_raw)
    return int(fee_cents)


def taker_fee_total(price_cents: int, num_contracts: int) -> int:
    """
    Calculate total Kalshi taker fee with proper rounding.

    Official formula: ceil(0.07 × P × N × (1 - P) × 100) / 100
    We compute in cents directly to avoid float rounding issues.

    Args:
        price_cents: Price in cents [1, 99]
        num_contracts: Number of contracts

    Returns:
        Total fee in cents (integer, with ceiling rounding)
    """
    P = price_cents / 100.0  # Convert to dollars

    # Apply Kalshi's formula: fee in cents with ceiling
    fee_cents_raw = 0.07 * P * num_contracts * (1 - P) * 100
    fee_cents = math.ceil(fee_cents_raw)

    return int(fee_cents)


def maker_fee_total(price_cents: int, num_contracts: int) -> int:
    """
    Calculate Kalshi maker fee.

    For most weather markets, maker fee = 0%.
    Some special markets may have maker fees - this can be extended.

    Args:
        price_cents: Price in cents [1, 99]
        num_contracts: Number of contracts

    Returns:
        Total fee in cents (typically 0, integer)
    """
    # Weather markets: 0% maker fee
    return 0


def classify_liquidity_role(
    side: str,
    action: str,
    price_cents: int,
    best_bid: int,
    best_ask: int
) -> str:
    """
    Determine if an order will be maker or taker.

    Maker: Order rests in book, provides liquidity
    Taker: Order crosses spread, removes liquidity (pays 7% fee)

    Rules:
    - For buy YES: price >= best_ask → taker, otherwise maker
    - For sell YES: price <= best_bid → taker, otherwise maker
    - For buy NO: (similar logic on NO side)
    - For sell NO: (similar logic on NO side)

    Args:
        side: "yes" or "no"
        action: "buy" or "sell"
        price_cents: Order limit price in cents
        best_bid: Current best bid in cents
        best_ask: Current best ask in cents

    Returns:
        "maker" or "taker"
    """
    if side == "yes" and action == "buy":
        # Buying YES: crossing the ask makes you a taker
        return "taker" if price_cents >= best_ask else "maker"

    elif side == "yes" and action == "sell":
        # Selling YES: hitting the bid makes you a taker
        return "taker" if price_cents <= best_bid else "maker"

    elif side == "no" and action == "buy":
        # Buying NO is equivalent to selling YES
        # NO ask = 100 - YES bid
        no_ask = 100 - best_bid
        return "taker" if price_cents >= no_ask else "maker"

    elif side == "no" and action == "sell":
        # Selling NO is equivalent to buying YES
        # NO bid = 100 - YES ask
        no_bid = 100 - best_ask
        return "taker" if price_cents <= no_bid else "maker"

    else:
        raise ValueError(f"Invalid side={side}, action={action}")


def compute_ev_per_contract(
    side: str,
    action: str,
    price_cents: int,
    model_prob: float,
    role: str
) -> float:
    """
    Compute expected value per contract in cents.

    EV = E[payout] - cost - fee

    For YES contract:
        - Payout: $1 if event happens (prob = model_prob), $0 otherwise
        - Cost: price_cents / 100 dollars
        - Fee: depends on maker vs taker

    For NO contract:
        - Payout: $1 if event DOESN'T happen (prob = 1 - model_prob), $0 otherwise
        - Cost: price_cents / 100 dollars
        - Fee: depends on maker vs taker

    Args:
        side: "yes" or "no"
        action: "buy" or "sell"
        price_cents: Trade price in cents
        model_prob: Model's probability that event happens [0, 1]
        role: "maker" or "taker"

    Returns:
        Expected value per contract in cents
    """
    P_dollars = price_cents / 100.0

    # Calculate fee
    if role == "maker":
        fee_cents = maker_fee_total(price_cents, 1)
    else:  # taker
        fee_cents = taker_fee_total(price_cents, 1)

    # Calculate EV based on position
    if side == "yes" and action == "buy":
        # Long YES: pay P, get $1 if event happens
        # EV = model_prob * (1 - P) + (1 - model_prob) * (-P) - fee
        # Simplify: model_prob - P - fee
        ev_cents = model_prob * 100 - price_cents - fee_cents

    elif side == "yes" and action == "sell":
        # Short YES: receive P, pay $1 if event happens
        # EV = model_prob * (-1 + P) + (1 - model_prob) * P - fee
        # Simplify: P - model_prob - fee
        ev_cents = price_cents - model_prob * 100 - fee_cents

    elif side == "no" and action == "buy":
        # Long NO: pay P, get $1 if event DOESN'T happen
        # EV = (1 - model_prob) * (1 - P) + model_prob * (-P) - fee
        # Simplify: (1 - model_prob) - P - fee
        ev_cents = (1 - model_prob) * 100 - price_cents - fee_cents

    elif side == "no" and action == "sell":
        # Short NO: receive P, pay $1 if event DOESN'T happen
        # EV = (1 - model_prob) * (-1 + P) + model_prob * P - fee
        # Simplify: P - (1 - model_prob) - fee
        ev_cents = price_cents - (1 - model_prob) * 100 - fee_cents

    else:
        raise ValueError(f"Invalid side={side}, action={action}")

    return ev_cents


def find_best_trade(
    model_prob: float,
    yes_bid: int,
    yes_ask: int,
    min_ev_cents: float = 3.0,
    maker_fill_prob: float = 0.4
) -> Tuple[str, str, int, float, str]:
    """
    Find the best trade (if any) across all sides.

    Evaluates:
    1. Buy YES at ask (taker) - guaranteed fill
    2. Buy YES with limit between bid and ask (maker) - partial fill prob
    3. Sell YES at bid (taker) - guaranteed fill
    4. Sell YES with limit between bid and ask (maker) - partial fill prob

    Maker orders have better EV (no fee) but lower fill probability.
    We adjust maker EV by estimated fill probability.

    Returns best EXPECTED EV trade (accounting for fill prob), or None if insufficient.

    Args:
        model_prob: Model probability event happens [0, 1]
        yes_bid: Best YES bid in cents
        yes_ask: Best YES ask in cents
        min_ev_cents: Minimum EV required to trade
        maker_fill_prob: Probability maker order fills (default 0.4 = 40%)

    Returns:
        (side, action, price, ev_cents, role) or (None, None, None, 0, None)
    """
    if yes_bid <= 0 or yes_ask >= 100 or yes_ask <= yes_bid:
        # Invalid or no liquidity
        return (None, None, None, 0.0, None)

    candidates = []

    # 1. Buy YES at ask (taker)
    role = classify_liquidity_role("yes", "buy", yes_ask, yes_bid, yes_ask)
    ev = compute_ev_per_contract("yes", "buy", yes_ask, model_prob, role)
    if ev >= min_ev_cents:
        candidates.append(("yes", "buy", yes_ask, ev, role))

    # 2. Buy YES with maker limit (post between bid and ask)
    if yes_ask - yes_bid > 1:  # Only if spread > 1 cent
        make_price = yes_bid + 1  # Improve bid by 1 cent
        role = classify_liquidity_role("yes", "buy", make_price, yes_bid, yes_ask)
        if role == "maker":  # Ensure we're actually making
            ev = compute_ev_per_contract("yes", "buy", make_price, model_prob, role)
            # Adjust for fill probability (maker orders don't always fill)
            ev_expected = ev * maker_fill_prob
            if ev_expected >= min_ev_cents:
                candidates.append(("yes", "buy", make_price, ev_expected, role))

    # 3. Sell YES at bid (taker) - for overpriced brackets
    role = classify_liquidity_role("yes", "sell", yes_bid, yes_bid, yes_ask)
    ev = compute_ev_per_contract("yes", "sell", yes_bid, model_prob, role)
    if ev >= min_ev_cents:
        candidates.append(("yes", "sell", yes_bid, ev, role))

    # 4. Sell YES with maker limit (post between bid and ask)
    if yes_ask - yes_bid > 1:
        make_price = yes_ask - 1  # Improve ask by 1 cent
        role = classify_liquidity_role("yes", "sell", make_price, yes_bid, yes_ask)
        if role == "maker":  # Ensure we're actually making
            ev = compute_ev_per_contract("yes", "sell", make_price, model_prob, role)
            # Adjust for fill probability
            ev_expected = ev * maker_fill_prob
            if ev_expected >= min_ev_cents:
                candidates.append(("yes", "sell", make_price, ev_expected, role))

    # Return best EV trade
    if candidates:
        best = max(candidates, key=lambda x: x[3])  # Sort by EV
        return best
    else:
        return (None, None, None, 0.0, None)


# ===== EXAMPLES & TESTS =====

if __name__ == "__main__":
    # Example: Verify fee calculation
    print("Kalshi Fee Examples:")
    print()

    for price in [10, 25, 50, 75, 90]:
        fee_per = taker_fee_per_contract(price)
        fee_10 = taker_fee_total(price, 10)
        fee_100 = taker_fee_total(price, 100)
        print(f"Price {price}¢: {fee_per:.3f}¢ per contract, "
              f"10 contracts = {fee_10:.2f}¢, 100 contracts = {fee_100:.2f}¢")

    print()
    print("Maker/Taker Classification:")
    print()

    # Spread: bid=45, ask=48
    test_cases = [
        ("yes", "buy", 45, "maker"),  # Posting at bid
        ("yes", "buy", 46, "maker"),  # Inside spread
        ("yes", "buy", 48, "taker"),  # Crossing ask
        ("yes", "buy", 50, "taker"),  # Well above ask
        ("yes", "sell", 48, "maker"),  # Posting at ask
        ("yes", "sell", 47, "maker"),  # Inside spread
        ("yes", "sell", 45, "taker"),  # Hitting bid
        ("yes", "sell", 40, "taker"),  # Well below bid
    ]

    for side, action, price, expected in test_cases:
        role = classify_liquidity_role(side, action, price, 45, 48)
        status = "✓" if role == expected else "✗"
        print(f"{status} {action.upper()} {side.upper()} @ {price}¢: {role} (expected {expected})")

    print()
    print("EV Calculation:")
    print()

    # Model thinks 60% chance event happens
    # Market: bid=45, ask=48 (implies ~46.5%)
    model_prob = 0.60

    print(f"Model probability: {model_prob:.1%}")
    print(f"Market: bid=45¢, ask=48¢ (implied {((45+48)/2):.1f}%)")
    print()

    # Buy YES at ask (taker)
    role = classify_liquidity_role("yes", "buy", 48, 45, 48)
    ev = compute_ev_per_contract("yes", "buy", 48, model_prob, role)
    print(f"Buy YES @ 48¢ ({role}): EV = {ev:.2f}¢")

    # Buy YES at 46 (maker)
    role = classify_liquidity_role("yes", "buy", 46, 45, 48)
    ev = compute_ev_per_contract("yes", "buy", 46, model_prob, role)
    print(f"Buy YES @ 46¢ ({role}): EV = {ev:.2f}¢")

    # Find best trade
    print()
    print("Best trade finder:")
    side, action, price, ev, role = find_best_trade(model_prob, 45, 48, min_ev_cents=3.0)
    if side:
        print(f"✓ TRADE: {action.upper()} {side.upper()} @ {price}¢ ({role}), EV = {ev:.2f}¢")
    else:
        print("✗ NO TRADE (insufficient edge)")
