"""
Bin label resolution for Kalshi temperature markets.

This module provides the canonical mapping from NWS integer temperature (°F)
to Kalshi bin YES/NO outcomes, matching Kalshi's exact settlement logic.
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
    """
    if tmax_f is None:
        return None

    if strike_type == "between":
        if floor_strike is None or cap_strike is None:
            return None
        return int(floor_strike <= tmax_f <= cap_strike)

    if strike_type == "less":
        if cap_strike is None:
            return None
        return int(tmax_f < cap_strike)

    if strike_type == "greater":
        if floor_strike is None:
            return None
        return int(tmax_f > floor_strike)

    return None


def validate_bin_set_consistency(bins: list[dict], tmax_f: int) -> dict:
    """
    Validate that exactly ONE bin in a complete bracket set resolves YES.

    Args:
        bins: List of bin dicts with keys: strike_type, floor_strike, cap_strike
        tmax_f: NWS official temperature (integer °F)

    Returns:
        Dict summarizing YES/NO counts and which bin(s) resolved YES.
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
        "is_valid": len(yes_bins) == 1,
        "winning_bins": yes_bins,
        "temperature": tmax_f,
    }
