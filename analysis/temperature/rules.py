"""
Temperature rounding rules for Kalshi settlement prediction.

This module implements a family of deterministic rules that map Visual Crossing
5-minute float temperatures to NWS/Kalshi integer daily highs.

Each rule represents a different hypothesis about how ASOS/NWS processes
temperature data into integer °F daily maxima.

All rules:
- Input: List[float] of 5-minute VC temps in °F
- Output: Optional[int] predicted daily high
- Return None if input is empty
"""

import math
from collections import Counter
from typing import List, Optional, Callable


def rule_max_round(temps_f: List[float]) -> Optional[int]:
    """R1: Round the maximum VC temperature to nearest integer.

    Simple baseline: max(temps) → round → integer

    Example:
        [90.1, 92.7, 93.4] → max=93.4 → round(93.4)=93

    This is the simplest possible rule but may miss subtleties
    in how ASOS averages are computed.
    """
    if not temps_f:
        return None
    return int(round(max(temps_f)))


def rule_max_of_rounded(temps_f: List[float]) -> Optional[int]:
    """R2: Round each sample, then take maximum.

    Round each 5-min temp individually, then max.

    Example:
        [90.1, 92.7, 93.4] → [90, 93, 93] → max=93

    This models a scenario where each instant is rounded before
    finding the daily max (closer to how discrete sampling works).

    Likely the best performer for most days.
    """
    if not temps_f:
        return None
    rounded = [int(round(t)) for t in temps_f]
    return max(rounded)


def rule_ceil_max(temps_f: List[float]) -> Optional[int]:
    """R3: Ceiling of the maximum VC temperature.

    Conservative rule: always round up.

    Example:
        [90.1, 92.7, 93.4] → max=93.4 → ceil(93.4)=94

    This would be appropriate if NWS always rounds up for safety/reporting,
    but empirically this is unlikely (would systematically overpredict).
    """
    if not temps_f:
        return None
    return int(math.ceil(max(temps_f)))


def rule_floor_max(temps_f: List[float]) -> Optional[int]:
    """R4: Floor of the maximum VC temperature.

    Aggressive rule: always round down.

    Example:
        [90.1, 92.7, 93.4] → max=93.4 → floor(93.4)=93

    This would be appropriate if NWS always truncates decimals,
    but empirically this is also unlikely.
    """
    if not temps_f:
        return None
    return int(math.floor(max(temps_f)))


def rule_plateau_20min(
    temps_f: List[float],
    min_minutes: int = 20,
    step_minutes: int = 5,
) -> Optional[int]:
    """R5: Highest integer °F with ≥20 minutes of consecutive support.

    Plateau logic: Only count a temperature as "real" if it's sustained
    for at least `min_minutes` (default 20) consecutively.

    Example:
        [90, 90, 93.4, 92, 92, 92, 92, 92] (5-min intervals)
        - 93.4 appears once → spike, ignore
        - 92 appears 5 times consecutive = 25 minutes → counts
        - Result: 92°F

    This filters single-sample spikes that might not reflect the
    official ASOS 5-minute running average used for CLI.

    Args:
        temps_f: List of 5-minute VC temps
        min_minutes: Minimum plateau duration (default 20)
        step_minutes: Time between samples (default 5)

    Returns:
        Highest integer with sufficient plateau support,
        or max_of_rounded as fallback if no plateau qualifies
    """
    if not temps_f:
        return None

    # Fallback for very short series
    if len(temps_f) < 2:
        return int(round(temps_f[0]))

    # Round all temps to integers
    rounded = [int(round(t)) for t in temps_f]

    # Required number of consecutive samples
    min_consecutive = max(1, min_minutes // step_minutes)

    # Find highest temp with sufficient plateau
    best_k: Optional[int] = None
    i = 0
    n = len(rounded)

    while i < n:
        k = rounded[i]
        # Count consecutive samples at this rounded value
        j = i + 1
        while j < n and rounded[j] == k:
            j += 1

        duration_minutes = (j - i) * step_minutes

        if duration_minutes >= min_minutes:
            if best_k is None or k > best_k:
                best_k = k

        i = j

    # Fallback if no plateau meets threshold
    if best_k is None:
        return max(rounded)

    return best_k


def rule_ignore_singletons(temps_f: List[float]) -> Optional[int]:
    """R6: Ignore integer values that appear only once.

    Filter out one-time spikes by requiring each rounded temp to
    appear at least twice in the series.

    Example:
        [90.1, 90.3, 93.4, 92.1, 92.3]
        → rounded: [90, 90, 93, 92, 92]
        → counts: {90:2, 93:1, 92:2}
        → candidates: [90, 92] (93 is singleton)
        → max(candidates) = 92

    Args:
        temps_f: List of 5-minute VC temps

    Returns:
        Maximum of non-singleton values, or max of all if everything is singleton
    """
    if not temps_f:
        return None

    rounded = [int(round(t)) for t in temps_f]
    counts = Counter(rounded)

    # Keep values that appear at least twice
    candidates = [k for k, count in counts.items() if count >= 2]

    if candidates:
        return max(candidates)

    # Fallback: if everything is a singleton, just use max
    return max(rounded)


def rule_c_first(temps_f: List[float]) -> Optional[int]:
    """R7 (Bonus): Celsius-first rounding path.

    Models archives that store temps in °C (0.1°C units), round in Celsius,
    then convert back to °F. This can create ±1°F shifts due to double-rounding.

    Example:
        93.4°F → 34.1°C → round(34.1)=34°C → 34*9/5+32=93.2°F → round(93.2)=93°F

    Process:
        For each sample: °F → °C → round → °F → round
        Then take max of these double-rounded values

    Note: Only relevant if we suspect C-native rounding in the data pipeline.
    Visual Crossing API returns °F directly, but underlying station data
    may use Celsius internally.
    """
    if not temps_f:
        return None

    candidates: List[int] = []

    for f in temps_f:
        # Convert to Celsius
        c = (f - 32.0) * 5.0 / 9.0

        # Round in Celsius
        c_rounded = round(c)

        # Convert back to Fahrenheit
        f_from_c = c_rounded * 9.0 / 5.0 + 32.0

        # Round again in Fahrenheit
        f_final = int(round(f_from_c))

        candidates.append(f_final)

    return max(candidates)


# Registry of all rules for easy iteration
ALL_RULES: dict[str, Callable[[List[float]], Optional[int]]] = {
    "max_round": rule_max_round,
    "max_of_rounded": rule_max_of_rounded,
    "ceil_max": rule_ceil_max,
    "floor_max": rule_floor_max,
    "plateau_20min": rule_plateau_20min,
    "ignore_singletons": rule_ignore_singletons,
    "c_first": rule_c_first,
}
