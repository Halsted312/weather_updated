"""Utilities for inferring Kalshi strike metadata when the API omits it.

The discovery scripts and database loaders expect `strike_type`, `floor_strike`,
and `cap_strike` to be present on every market row.  In practice the public API
occasionally omits these fields for archived markets which in turn corrupts the
backtester (missing strike_type defaults to 0.5 odds inside
`ModelKellyBacktestStrategy`).

This module centralises a set of heuristics so every ingest path can recover
strike metadata from the textual description (subtitle/title/rules) and fall
back to sane defaults.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_BETWEEN_PATTERNS = [
    re.compile(
        r"between\s+(-?\d+(?:\.\d+)?)\s*(?:°|deg|degrees)?\s*(?:f|c)?\s+and\s+(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)"),
]
_GREATER_PATTERNS = [
    re.compile(
        r"(?:greater(?:\s+than)?|above|over|at\s+least|≥|>)\s+(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"g(\d+(?:\.\d+)?)", re.IGNORECASE),
]
_LESS_PATTERNS = [
    re.compile(
        r"(?:less(?:\s+than)?|below|under|at\s+most|≤|<)\s+(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(r"l(\d+(?:\.\d+)?)", re.IGNORECASE),
]


def _clean_value(raw: str) -> Optional[float]:
    if raw is None:
        return None
    clean = raw.replace("°", "").replace("F", "").replace("C", "").strip()
    clean = clean.rstrip("fFcC")
    try:
        return float(clean)
    except (TypeError, ValueError):
        return None


def _infer_from_text(text: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not text:
        return None, None, None

    for pattern in _BETWEEN_PATTERNS:
        match = pattern.search(text)
        if match:
            floor = _clean_value(match.group(1))
            cap = _clean_value(match.group(2))
            if floor is not None and cap is not None:
                if floor > cap:
                    floor, cap = cap, floor
                return "between", floor, cap

    for pattern in _GREATER_PATTERNS:
        match = pattern.search(text)
        if match:
            floor = _clean_value(match.group(1))
            if floor is not None:
                return "greater", floor, None

    for pattern in _LESS_PATTERNS:
        match = pattern.search(text)
        if match:
            cap = _clean_value(match.group(1))
            if cap is not None:
                return "less", None, cap

    return None, None, None


def _normalise_type(
    strike_type: Optional[str],
    floor: Optional[float],
    cap: Optional[float],
) -> Optional[str]:
    stype = (strike_type or "").lower() or None
    if stype in {"between", "less", "greater"}:
        return stype
    if floor is not None and cap is not None:
        return "between"
    if floor is not None:
        return "greater"
    if cap is not None:
        return "less"
    return None


def ensure_strike_metadata(market: Dict[str, Any]) -> Dict[str, Any]:
    """Return market dict with strike metadata populated whenever possible."""

    strike_type = market.get("strike_type")
    floor = market.get("floor_strike")
    cap = market.get("cap_strike")

    if strike_type and strike_type.lower() in {"between", "less", "greater"}:
        market["strike_type"] = strike_type.lower()
        return market

    texts = [
        market.get("subtitle"),
        market.get("title"),
        market.get("rules_primary"),
        market.get("rules_secondary"),
        market.get("description"),
    ]

    inferred: Tuple[Optional[str], Optional[float], Optional[float]] = (None, None, None)
    for text in texts:
        inferred = _infer_from_text(text or "")
        if inferred[0]:
            break

    inferred_type, inferred_floor, inferred_cap = inferred

    floor = floor if floor is not None else inferred_floor
    cap = cap if cap is not None else inferred_cap

    strike_type = _normalise_type(strike_type, floor, cap)
    if not strike_type:
        strike_type = inferred_type

    if strike_type not in {"between", "less", "greater"}:
        if market.get("event_ticker", "").lower().endswith("b") and floor is not None and cap is not None:
            strike_type = "between"

    if strike_type is None:
        logger.debug("Unable to infer strike metadata for %s", market.get("ticker"))
    else:
        market["strike_type"] = strike_type
        market["floor_strike"] = floor
        market["cap_strike"] = cap

    return market


__all__ = ["ensure_strike_metadata"]
