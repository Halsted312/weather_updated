#!/usr/bin/env python3
"""
Model-driven Kelly strategy with fee-aware edge calculation.

Strategy:
- Uses Ridge model probabilities to estimate true probability
- Calculates fee-aware edge (YES vs NO) using effective entry costs
- Gates entry: edge ≥ τ_open (default: 3¢)
- Sizes position: fractional Kelly (α=0.25)
- Filters: max spread ≤ 3¢, slippage=1¢

Fee model:
- Uses taker fees: ceil(0.07 × C × P × (1-P)) cents
- Effective entry = ask + fee + slippage (for YES)
- Effective entry = (100 - bid) + fee + slippage (for NO)
- Breakeven prob = effective_entry / 100

Kelly sizing:
- f* = (p_model - p_breakeven) / (1 - p_breakeven)
- Position size = α × f* × bankroll (α=0.25)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from models.costs import (
    effective_yes_entry_cents,
    effective_no_entry_cents,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecParams:
    """Execution parameters for model strategy."""
    max_spread_cents: int = 3       # Filter: skip if spread > 3¢
    slippage_cents: int = 1          # Slippage per leg
    tau_open_cents: int = 5          # Entry gate: edge ≥ 5¢ after costs
    tau_close_cents: float = 0.5     # Exit gate: edge ≤ 0.5¢ (not used in current baseline)
    alpha_kelly: float = 0.25        # Fractional Kelly multiplier
    max_bankroll_pct_city_day_side: float = 0.10  # Max 10% per city-day-side
    max_trade_notional_pct: float = 0.02  # Max % of bankroll allocated per individual trade
    allowed_time_windows: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    time_of_day_overrides: List[Dict[str, Any]] = field(default_factory=list)
    bracket_spread_overrides: Dict[str, int] = field(default_factory=dict)
    sigma_gate: Optional[float] = None
    humidity_gate: Optional[float] = None


def breakeven_prob_yes(yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    """
    Breakeven probability for YES side.

    Args:
        yes_bid: YES bid in cents [0, 100]
        yes_ask: YES ask in cents [0, 100]
        slippage: Slippage in cents (default: 1)

    Returns:
        Breakeven probability (effective_entry / 100)
    """
    y_eff = effective_yes_entry_cents(yes_bid, yes_ask, slippage=slippage)
    return y_eff / 100.0


def breakeven_prob_no(yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    """
    Breakeven probability for NO side.

    NO probability is (1 - YES probability), so:
    - Effective NO entry = (100 - yes_bid) + fee + slippage
    - Breakeven for NO = effective_no_entry / 100

    Args:
        yes_bid: YES bid in cents
        yes_ask: YES ask in cents
        slippage: Slippage in cents

    Returns:
        Breakeven probability for NO side
    """
    n_eff = effective_no_entry_cents(yes_bid, yes_ask, slippage=slippage)
    return n_eff / 100.0


def kelly_fraction_yes(p: float, p_be: float) -> float:
    """
    Kelly fraction for YES side (binary payoff 0/100¢).

    Formula: f* = (p - p_be) / (1 - p_be)

    Where:
    - p: Model's estimated probability of YES
    - p_be: Breakeven probability (effective_entry / 100)

    Args:
        p: Model probability (0 to 1)
        p_be: Breakeven probability (0 to 1)

    Returns:
        Kelly fraction (0 to 1, clipped)
    """
    if p <= p_be:
        return 0.0

    # Avoid division by zero
    denom = max(1e-9, 1.0 - p_be)
    f_star = (p - p_be) / denom

    # Clip to [0, 1]
    return max(0.0, min(1.0, f_star))


def edge_cents_yes(p: float, yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    """
    Expected edge (in cents) for YES side after all costs.

    EV = 100 × p - effective_yes_entry

    Args:
        p: Model probability
        yes_bid: YES bid in cents
        yes_ask: YES ask in cents
        slippage: Slippage in cents

    Returns:
        Edge in cents (can be negative)
    """
    y_eff = effective_yes_entry_cents(yes_bid, yes_ask, slippage=slippage)
    return 100.0 * p - y_eff


def edge_cents_no(p: float, yes_bid: int, yes_ask: int, slippage: int = 1) -> float:
    """
    Expected edge (in cents) for NO side after all costs.

    EV_NO = 100 × (1 - p) - effective_no_entry

    Args:
        p: Model probability (of YES)
        yes_bid: YES bid in cents
        yes_ask: YES ask in cents
        slippage: Slippage in cents

    Returns:
        Edge in cents for NO side (can be negative)
    """
    n_eff = effective_no_entry_cents(yes_bid, yes_ask, slippage=slippage)
    return 100.0 * (1.0 - p) - n_eff


class ModelKellyStrategy:
    """
    Model-driven Kelly strategy.

    Uses Ridge model probabilities to compute fee-aware edge and size positions
    with fractional Kelly.

    Execution filters:
    - Max spread ≤ 3¢
    - Edge ≥ 3¢ after costs
    - Fractional Kelly (α=0.25)
    """

    def __init__(self, exec_params: Optional[ExecParams] = None):
        """
        Initialize strategy with execution parameters.

        Args:
            exec_params: ExecParams instance (uses defaults if None)
        """
        self.params = exec_params or ExecParams()
        logger.info(f"ModelKellyStrategy initialized with params: {self.params}")

    def signal_for_row(self, row: pd.Series) -> Optional[Dict]:
        """
        Generate signal for a single market observation.

        Args:
            row: pandas Series with columns:
                - yes_bid_close: YES bid in cents
                - yes_ask_close: YES ask in cents
                - p_model: Model probability (0 to 1)
                - market_ticker: Market ticker
                - city: City name (optional, for logging)
                - event_date: Event date (optional, for logging)

        Returns:
            Dict with signal fields if trade is viable, else None:
            - action: "BUY_YES", "BUY_NO", or "NONE"
            - kelly_frac: Kelly fraction (before α multiplier)
            - edge_cents: Expected edge in cents after costs
            - spread_cents: Bid-ask spread
            - contracts: Number of contracts (placeholder, sized by caller)

            Returns None if:
            - Spread > max_spread
            - Edge < tau_open for both sides
            - Kelly fraction ≤ 0
        """
        # Extract fields
        try:
            bid = int(row["yes_bid_close"])
            ask = int(row["yes_ask_close"])
            p = float(row["p_model"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Missing or invalid fields in row: {e}")
            return None

        strike_type = row.get("strike_type")
        minute_local = row.get("minute_of_day_local")
        sigma_est = row.get("sigma_est")
        humidity_val = row.get("humidity_var", row.get("humidity_pct"))

        if self.params.allowed_time_windows and minute_local is not None:
            windows = self.params.allowed_time_windows.get(strike_type) or self.params.allowed_time_windows.get("all")
            if windows and not any(start <= minute_local <= end for start, end in windows):
                return None

        max_spread_cents = self.params.max_spread_cents
        if strike_type and strike_type in self.params.bracket_spread_overrides:
            max_spread_cents = self.params.bracket_spread_overrides[strike_type]
        tau_open_cents = self.params.tau_open_cents

        if self.params.time_of_day_overrides and minute_local is not None:
            for override in self.params.time_of_day_overrides:
                window = override.get("window")
                if not window:
                    continue
                start, end = window
                target = override.get("strike_type")
                if target not in (None, "all", strike_type):
                    continue
                if start <= minute_local <= end:
                    if "max_spread_cents" in override:
                        max_spread_cents = min(max_spread_cents, override["max_spread_cents"])
                    if "tau_open_cents" in override:
                        tau_open_cents = override["tau_open_cents"]

        if self.params.sigma_gate is not None and sigma_est is not None:
            try:
                if float(sigma_est) > self.params.sigma_gate:
                    return None
            except (TypeError, ValueError):
                pass

        if self.params.humidity_gate is not None and humidity_val is not None:
            try:
                if float(humidity_val) > self.params.humidity_gate:
                    return None
            except (TypeError, ValueError):
                pass

        spread = max(0, ask - bid)
        if spread > max_spread_cents:
            return None

        # B6 PATCH: Calculate edge for both sides (YES and NO)
        e_yes = edge_cents_yes(p, bid, ask, slippage=self.params.slippage_cents)
        e_no = edge_cents_no(p, bid, ask, slippage=self.params.slippage_cents)

        # Choose side with max edge
        if e_yes >= e_no and e_yes >= tau_open_cents:
            # Trade YES
            p_be = breakeven_prob_yes(bid, ask, slippage=self.params.slippage_cents)
            f_star = kelly_fraction_yes(p, p_be)

            if f_star <= 0.0:
                return None

            return {
                "action": "BUY_YES",
                "kelly_frac": float(f_star),
                "edge_cents": float(e_yes),
                "spread_cents": int(spread),
                "p_model": float(p),
                "p_breakeven": float(p_be),
            }

        elif e_no > e_yes and e_no >= tau_open_cents:
            # Trade NO
            # For NO, use (1-p) as the probability that NO wins
            p_no = 1.0 - p
            p_be_no = breakeven_prob_no(bid, ask, slippage=self.params.slippage_cents)
            f_star = kelly_fraction_yes(p_no, p_be_no)  # Same Kelly formula

            if f_star <= 0.0:
                return None

            return {
                "action": "BUY_NO",
                "kelly_frac": float(f_star),
                "edge_cents": float(e_no),
                "spread_cents": int(spread),
                "p_model": float(p),
                "p_breakeven": float(p_be_no),
            }

        else:
            # No trade (edge too small or negative)
            return None


def main():
    """Demo: Test ModelKellyStrategy with sample data."""
    print("\n" + "="*60)
    print("ModelKellyStrategy Demo")
    print("="*60 + "\n")

    # Create strategy
    strategy = ModelKellyStrategy()

    # Sample market observations
    samples = [
        {"yes_bid_close": 48, "yes_ask_close": 52, "p_model": 0.60, "market_ticker": "TEST-1"},
        {"yes_bid_close": 48, "yes_ask_close": 52, "p_model": 0.40, "market_ticker": "TEST-2"},
        {"yes_bid_close": 48, "yes_ask_close": 52, "p_model": 0.50, "market_ticker": "TEST-3"},
        {"yes_bid_close": 45, "yes_ask_close": 55, "p_model": 0.60, "market_ticker": "TEST-4"},  # Wide spread
    ]

    for sample in samples:
        row = pd.Series(sample)
        signal = strategy.signal_for_row(row)

        print(f"Market: {sample['market_ticker']}")
        print(f"  Bid/Ask: {sample['yes_bid_close']}/{sample['yes_ask_close']}")
        print(f"  Model prob: {sample['p_model']:.2f}")

        if signal:
            print(f"  → Signal: {signal['action']}")
            print(f"    Edge: {signal['edge_cents']:.1f}¢")
            print(f"    Kelly fraction: {signal['kelly_frac']:.3f}")
            print(f"    Breakeven prob: {signal['p_breakeven']:.3f}")
        else:
            print(f"  → No signal (spread/edge filter)")

        print()

    print("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
