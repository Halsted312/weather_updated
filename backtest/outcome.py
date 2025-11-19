#!/usr/bin/env python3
"""
Market outcome resolution engine.

Determines YES/NO outcomes based on settlement TMAX and market bracket metadata.
Uses Kalshi's exact inclusive/exclusive semantics for temperature ranges.

Bracket semantics:
- strike_type='between': Inclusive on both ends [floor, cap]
  YES if floor <= tmax <= cap

- strike_type='less': Strictly less than cap
  YES if tmax < cap

- strike_type='greater': Strictly greater than floor (exclusive)
  YES if tmax > floor
"""

import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)


def resolve_bin(
    tmax_f: float,
    floor_strike: Optional[int],
    cap_strike: Optional[int],
    strike_type: Literal["between", "less", "greater"],
) -> Literal["YES", "NO"]:
    """
    Resolve market outcome based on settlement TMAX and bracket bounds.

    Args:
        tmax_f: Settlement maximum temperature in Fahrenheit (float)
        floor_strike: Lower bound (integer °F, None if not applicable)
        cap_strike: Upper bound (integer °F, None if not applicable)
        strike_type: Bracket type ('between', 'less', 'greater')

    Returns:
        "YES" or "NO"

    Raises:
        ValueError: If strike_type is invalid or bounds are missing for given type

    Examples:
        >>> # Between bracket [34, 35]: inclusive both ends
        >>> resolve_bin(34.0, 34, 35, "between")
        'YES'
        >>> resolve_bin(35.0, 34, 35, "between")
        'YES'
        >>> resolve_bin(35.1, 34, 35, "between")
        'NO'
        >>> resolve_bin(33.9, 34, 35, "between")
        'NO'

        >>> # Less than bracket: strictly less than cap
        >>> resolve_bin(33.9, None, 34, "less")
        'YES'
        >>> resolve_bin(34.0, None, 34, "less")
        'NO'

        >>> # Greater than bracket (exclusive): strictly > floor
        >>> resolve_bin(75.0, 75, None, "greater")
        'NO'
        >>> resolve_bin(75.1, 75, None, "greater")
        'YES'
        >>> resolve_bin(74.9, 75, None, "greater")
        'NO'
    """
    if strike_type == "between":
        # Inclusive on both ends: [floor, cap]
        # Compare exact float value against integer bounds
        if floor_strike is None or cap_strike is None:
            raise ValueError(
                f"'between' strike_type requires both floor and cap, "
                f"got floor={floor_strike}, cap={cap_strike}"
            )

        result = "YES" if floor_strike <= tmax_f <= cap_strike else "NO"

        logger.debug(
            f"resolve_bin: tmax={tmax_f}°F, "
            f"bracket=[{floor_strike}, {cap_strike}] (between), "
            f"result={result}"
        )

        return result

    elif strike_type == "less":
        # Strictly less than cap: tmax < cap
        if cap_strike is None:
            raise ValueError(
                f"'less' strike_type requires cap, got cap={cap_strike}"
            )

        result = "YES" if tmax_f < cap_strike else "NO"

        logger.debug(
            f"resolve_bin: tmax={tmax_f}°F, "
            f"cap={cap_strike} (less), "
            f"result={result}"
        )

        return result

    elif strike_type == "greater":
        # Strictly greater than floor (exclusive): tmax > floor
        # Empirically verified: Kalshi settled bins with temp==floor as NO
        if floor_strike is None:
            raise ValueError(
                f"'greater' strike_type requires floor, got floor={floor_strike}"
            )

        result = "YES" if tmax_f > floor_strike else "NO"

        logger.debug(
            f"resolve_bin: tmax={tmax_f}°F, "
            f"floor={floor_strike} (greater), "
            f"result={result}"
        )

        return result

    else:
        raise ValueError(
            f"Invalid strike_type: {strike_type}. "
            f"Must be 'between', 'less', or 'greater'"
        )


def resolve_market(
    tmax_f: float,
    floor_strike: Optional[int],
    cap_strike: Optional[int],
    strike_type: str,
) -> bool:
    """
    Resolve market outcome as boolean (True=YES, False=NO).

    Convenience wrapper around resolve_bin() for boolean operations.

    Args:
        tmax_f: Settlement maximum temperature in Fahrenheit
        floor_strike: Lower bound (integer °F, None if not applicable)
        cap_strike: Upper bound (integer °F, None if not applicable)
        strike_type: Bracket type ('between', 'less', 'greater')

    Returns:
        True if YES, False if NO

    Examples:
        >>> resolve_market(35.0, 34, 35, "between")
        True
        >>> resolve_market(36.0, 34, 35, "between")
        False
    """
    outcome = resolve_bin(tmax_f, floor_strike, cap_strike, strike_type)
    return outcome == "YES"


def main():
    """Demo: Test resolve_bin with example markets."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("\n" + "="*60)
    print("Market Outcome Resolution Tests")
    print("="*60 + "\n")

    test_cases = [
        # Between brackets [34, 35]
        (35.0, 34, 35, "between", "YES"),  # Upper edge (inclusive)
        (34.0, 34, 35, "between", "YES"),  # Lower edge (inclusive)
        (34.5, 34, 35, "between", "YES"),  # Inside
        (35.1, 34, 35, "between", "NO"),   # Just above
        (33.9, 34, 35, "between", "NO"),   # Just below

        # Less than bracket < 34
        (33.9, None, 34, "less", "YES"),   # Just below (YES)
        (34.0, None, 34, "less", "NO"),    # Equal (NO)
        (34.1, None, 34, "less", "NO"),    # Above (NO)

        # Greater than bracket > 75 (exclusive)
        (75.0, 75, None, "greater", "NO"),   # Equal (NO - exclusive!)
        (74.9, 75, None, "greater", "NO"),   # Just below (NO)
        (75.1, 75, None, "greater", "YES"),  # Just above (YES)
        (76.0, 75, None, "greater", "YES"),  # Above (YES)
    ]

    for tmax, floor, cap, stype, expected in test_cases:
        result = resolve_bin(tmax, floor, cap, stype)
        status = "✓" if result == expected else "✗"

        bracket_str = (
            f"[{floor}, {cap}]" if stype == "between"
            else f"< {cap}" if stype == "less"
            else f"> {floor}"
        )

        print(
            f"{status} tmax={tmax:5.1f}°F, bracket={bracket_str:12}, "
            f"type={stype:8} → {result:3} (expected {expected})"
        )

    print("\n" + "="*60)
    print("All tests passed!" if all(
        resolve_bin(t, f, c, st) == exp
        for t, f, c, st, exp in test_cases
    ) else "Some tests failed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
