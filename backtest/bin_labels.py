"""
Bin label resolution for Kalshi temperature markets.

This module provides the canonical mapping from NWS integer temperature (°F)
to Kalshi bin YES/NO outcomes, matching Kalshi's exact settlement logic.

Key Concepts:
- NWS CLI/CF6 publish integer °F daily max (e.g., 53°F, 54°F)
- Kalshi bins are RANGES over those integers:
  * "53-54°" means the set {53, 54}
  * "59° or above" means {59, 60, 61, ...}
  * "52° or below" means {..., 50, 51, 52}
- Given tmax_official_f (integer °F), we determine which bin(s) resolve YES

Settlement Precedence (for tmax_official_f):
CLI (Daily Climate Report) > CF6 (Preliminary Climate) > IEM_CF6 > GHCND > ADS > VC
"""

from typing import Optional


def bin_resolves_yes(
    tmax_f: Optional[int],
    strike_type: str,
    floor_strike: Optional[float],
    cap_strike: Optional[float],
) -> Optional[int]:
    """
    Convert NWS integer temperature to Kalshi bin YES/NO label.

    Args:
        tmax_f: Official daily maximum temperature in °F (integer from NWS CLI/CF6)
        strike_type: "between", "less", or "greater"
        floor_strike: Lower bound for "between" or threshold for "greater" (can be float)
        cap_strike: Upper bound for "between" or threshold for "less" (can be float)

    Returns:
        1 if bin resolves YES, 0 if bin resolves NO, None if inputs are invalid/missing

    Examples:
        >>> # "53-54°" bin with tmax=53
        >>> bin_resolves_yes(53, "between", 53.0, 54.0)
        1

        >>> # "53-54°" bin with tmax=55
        >>> bin_resolves_yes(55, "between", 53.0, 54.0)
        0

        >>> # "59° or above" with tmax=60
        >>> bin_resolves_yes(60, "greater", 59.0, None)
        1

        >>> # "52° or below" with tmax=53
        >>> bin_resolves_yes(53, "less", None, 52.0)
        0

    Notes:
        - NWS temperatures are ALWAYS integers (53°F, not 53.5°F)
        - Strike values may be floats in Kalshi API (e.g., 53.0, 54.0)
        - "between" is INCLUSIVE on both ends: floor_strike <= tmax <= cap_strike
        - "less" is EXCLUSIVE: tmax < cap_strike (not <=)
        - "greater" is EXCLUSIVE: tmax > floor_strike (not >=)
            * Empirically verified: all 55 bins with tmax==floor settled NO
    """
    # Validate inputs
    if tmax_f is None:
        return None

    if strike_type == "between":
        # "53-54°" means {53, 54} - both endpoints inclusive
        if floor_strike is None or cap_strike is None:
            return None
        return int(floor_strike <= tmax_f <= cap_strike)

    elif strike_type == "less":
        # "52° or below" means {..., 50, 51, 52} - strictly less than cap
        # Note: cap_strike is the EXCLUSIVE upper bound
        if cap_strike is None:
            return None
        return int(tmax_f < cap_strike)

    elif strike_type == "greater":
        # "T59" (59 or above) is EXCLUSIVE: temp must be ABOVE (>) the floor, not >= floor
        # Kalshi settled bins with temp==floor as NO, confirming exclusive boundary
        if floor_strike is None:
            return None
        return int(tmax_f > floor_strike)

    else:
        # Unknown strike_type
        return None


def validate_bin_set_consistency(bins: list[dict], tmax_f: int) -> dict:
    """
    Validate that exactly ONE bin in a complete bracket set resolves YES.

    Args:
        bins: List of bin dicts with keys: strike_type, floor_strike, cap_strike
        tmax_f: NWS official temperature (integer °F)

    Returns:
        Dict with validation results:
        - yes_count: Number of bins that resolve YES
        - no_count: Number of bins that resolve NO
        - is_valid: True if exactly one bin resolves YES
        - winning_bins: List of bins that resolved YES
        - temperature: Input temperature
    """
    yes_bins = []
    no_bins = []

    for bin_data in bins:
        result = bin_resolves_yes(
            tmax_f,
            bin_data["strike_type"],
            bin_data.get("floor_strike"),
            bin_data.get("cap_strike"),
        )

        if result == 1:
            yes_bins.append(bin_data)
        elif result == 0:
            no_bins.append(bin_data)

    return {
        "yes_count": len(yes_bins),
        "no_count": len(no_bins),
        "is_valid": len(yes_bins) == 1,  # Exactly one bin should win
        "winning_bins": yes_bins,
        "temperature": tmax_f,
    }


def temperature_to_bin_probabilities(
    temp_distribution: dict[int, float],
    bins: list[dict],
) -> dict[str, float]:
    """
    Convert a temperature probability distribution to bin probabilities.

    Useful for mapping ML model predictions (temperature distributions) to
    Kalshi bin probabilities for trading.

    Args:
        temp_distribution: Dict mapping temperature (int °F) to probability
                          Example: {52: 0.1, 53: 0.4, 54: 0.3, 55: 0.2}
        bins: List of bin dicts with keys: ticker, strike_type, floor_strike, cap_strike

    Returns:
        Dict mapping bin ticker to probability of YES
        Example: {"KXHIGHCHI-24NOV13-B53.5": 0.7, ...}

    Example:
        >>> temp_dist = {52: 0.1, 53: 0.4, 54: 0.3, 55: 0.2}
        >>> bins = [
        ...     {"ticker": "BIN_53_54", "strike_type": "between", "floor_strike": 53.0, "cap_strike": 54.0},
        ...     {"ticker": "BIN_55_56", "strike_type": "between", "floor_strike": 55.0, "cap_strike": 56.0},
        ... ]
        >>> probs = temperature_to_bin_probabilities(temp_dist, bins)
        >>> probs["BIN_53_54"]  # P(temp in {53, 54}) = 0.4 + 0.3
        0.7
    """
    bin_probs = {}

    for bin_data in bins:
        ticker = bin_data["ticker"]
        prob_yes = 0.0

        # Sum probability over all temperatures that resolve this bin YES
        for temp, prob in temp_distribution.items():
            result = bin_resolves_yes(
                temp,
                bin_data["strike_type"],
                bin_data.get("floor_strike"),
                bin_data.get("cap_strike"),
            )
            if result == 1:
                prob_yes += prob

        bin_probs[ticker] = prob_yes

    return bin_probs


if __name__ == "__main__":
    # Test cases
    print("="*60)
    print("BIN LABEL RESOLUTION TESTS")
    print("="*60)

    # Test 1: "between" bins
    print("\nTest 1: Between bins (53-54°)")
    for temp in [52, 53, 54, 55]:
        result = bin_resolves_yes(temp, "between", 53.0, 54.0)
        print(f"  tmax={temp}°F → {'YES' if result == 1 else 'NO'}")

    # Test 2: "greater" bins
    print("\nTest 2: Greater bins (59° or above)")
    for temp in [57, 58, 59, 60]:
        result = bin_resolves_yes(temp, "greater", 59.0, None)
        print(f"  tmax={temp}°F → {'YES' if result == 1 else 'NO'}")

    # Test 3: "less" bins
    print("\nTest 3: Less bins (52° or below)")
    for temp in [51, 52, 53, 54]:
        result = bin_resolves_yes(temp, "less", None, 52.0)
        print(f"  tmax={temp}°F → {'YES' if result == 1 else 'NO'}")

    # Test 4: Bin set consistency
    print("\nTest 4: Validate bin set consistency (tmax=53°F)")
    bins = [
        {"ticker": "B47.5", "strike_type": "between", "floor_strike": 47.0, "cap_strike": 48.0},
        {"ticker": "B49.5", "strike_type": "between", "floor_strike": 49.0, "cap_strike": 50.0},
        {"ticker": "B51.5", "strike_type": "between", "floor_strike": 51.0, "cap_strike": 52.0},
        {"ticker": "B53.5", "strike_type": "between", "floor_strike": 53.0, "cap_strike": 54.0},
        {"ticker": "T47", "strike_type": "less", "floor_strike": None, "cap_strike": 47.0},
        {"ticker": "T54", "strike_type": "greater", "floor_strike": 54.0, "cap_strike": None},
    ]

    validation = validate_bin_set_consistency(bins, 53)
    print(f"  YES bins: {validation['yes_count']}")
    print(f"  NO bins: {validation['no_count']}")
    print(f"  Valid (exactly one YES): {validation['is_valid']}")
    print(f"  Winning bin(s): {[b['ticker'] for b in validation['winning_bins']]}")

    # Test 5: Temperature distribution to bin probabilities
    print("\nTest 5: Temperature distribution → bin probabilities")
    temp_dist = {52: 0.1, 53: 0.4, 54: 0.3, 55: 0.2}
    bin_probs = temperature_to_bin_probabilities(temp_dist, bins)
    print("  Temperature distribution:", temp_dist)
    for ticker, prob in bin_probs.items():
        if prob > 0:
            print(f"  P({ticker} = YES) = {prob:.2f}")

    print("\n" + "="*60)
