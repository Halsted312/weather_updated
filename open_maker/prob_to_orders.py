"""
Probability-to-Orders Bridge Module

Converts model P[delta] distributions to trading recommendations by comparing
model-implied probabilities with market prices for each bracket.

The model predicts delta = settlement - t_base (where t_base = round(vc_max_f_sofar)).
This module converts these delta probabilities to:
- P[settlement >= strike] for each bracket
- Trading signals based on edge vs market

Usage:
    from open_maker.prob_to_orders import DeltaProbToOrders

    bridge = DeltaProbToOrders(
        delta_classes=list(range(-10, 11)),  # [-10, ..., 0, ..., +10] (21 classes)
        min_edge_pct=10.0,  # 10% edge required
    )

    # Get trading recommendations for all brackets
    recs = bridge.get_recommendations(
        delta_proba=model_proba,  # shape (n_classes,)
        t_base=72.0,
        brackets_df=brackets,  # columns: ticker, floor_strike, cap_strike, yes_bid, yes_ask
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HorizonRiskConfig:
    """
    Configuration for horizon-aware position sizing and edge thresholds.

    Professor's Point (B): Use variance to derive position sizing multipliers.

    The idea: uncertainty (std of delta) increases with hours_to_event_close.
    - Near close (0-2h): std ≈ 1.0, variance ≈ 1.0 → full size, low edge threshold
    - Early D-1 (24h+): std ≈ 3.7, variance ≈ 13.7 → 1/14 size, high edge threshold

    Attributes:
        bucket_edges: Hours-to-close boundaries defining buckets.
                      Default: [2, 6, 12, 18] creates buckets [0,2), [2,6), [6,12), [12,18), [18,+inf)
        size_multipliers: Position size multiplier for each bucket (5 values for 4 edges).
                          Default: [1.0, 0.5, 0.25, 0.15, 0.08] from variance analysis.
        edge_multipliers: Edge threshold multiplier for each bucket.
                          Default: [1.0, 1.2, 1.5, 2.0, 2.5] (stricter early).
    """
    bucket_edges: list[float] = field(default_factory=lambda: [2.0, 6.0, 12.0, 18.0])
    size_multipliers: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.15, 0.08])
    edge_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.2, 1.5, 2.0, 2.5])

    def get_bucket_index(self, hours_to_close: float) -> int:
        """Get the bucket index for a given hours_to_close value."""
        for i, edge in enumerate(self.bucket_edges):
            if hours_to_close < edge:
                return i
        return len(self.bucket_edges)  # Last bucket (beyond all edges)

    def get_size_multiplier(self, hours_to_close: float) -> float:
        """Get position size multiplier for given hours_to_close."""
        idx = self.get_bucket_index(hours_to_close)
        return self.size_multipliers[idx]

    def get_edge_multiplier(self, hours_to_close: float) -> float:
        """Get edge threshold multiplier for given hours_to_close."""
        idx = self.get_bucket_index(hours_to_close)
        return self.edge_multipliers[idx]


# Default horizon config based on empirical variance analysis
DEFAULT_HORIZON_CONFIG = HorizonRiskConfig()


@dataclass
class BracketRecommendation:
    """Trading recommendation for a single bracket."""

    ticker: str
    floor_strike: Optional[float]
    cap_strike: Optional[float]

    # Model probabilities
    model_prob_yes: float  # P[bracket wins]
    model_prob_no: float   # P[bracket loses]

    # Market prices (in cents, 0-100)
    market_yes_bid: float
    market_yes_ask: float
    market_no_bid: float   # = 100 - yes_ask
    market_no_ask: float   # = 100 - yes_bid

    # Recommendation
    action: str  # "buy_yes", "buy_no", "hold", "skip"
    side: str    # "yes", "no", "none"
    entry_price: float  # Price to enter at (cents)
    edge_pct: float  # (model_prob - market_price) / market_price * 100
    expected_value: float  # EV in cents per contract

    # Horizon-aware fields (Point B)
    size_multiplier: float = 1.0  # Position size multiplier from horizon config
    hours_to_close: Optional[float] = None  # Time to event close
    effective_edge_threshold: float = 5.0  # Adjusted edge threshold

    # Rationale
    reason: str = ""


class DeltaProbToOrders:
    """
    Convert model delta probabilities to bracket trading recommendations.

    The model predicts P[delta=k] for each delta class k.
    Given t_base (current max so far, rounded), we convert to:
    - P[settlement = T] = P[delta = T - t_base]
    - P[settlement >= strike] = sum over k where (t_base + k) >= strike
    """

    def __init__(
        self,
        delta_classes: list[int],
        min_edge_pct: float = 5.0,
        min_prob: float = 0.10,
        use_maker_pricing: bool = True,
        horizon_config: Optional[HorizonRiskConfig] = None,
    ):
        """
        Initialize the probability-to-orders bridge.

        Args:
            delta_classes: List of delta values the model can predict [-10, ..., 0, ..., +10]
            min_edge_pct: Minimum edge required to recommend a trade (default 5%)
            min_prob: Minimum model probability to consider (avoid low-confidence trades)
            use_maker_pricing: If True, use bid for sells and ask for buys (maker pricing)
            horizon_config: Optional HorizonRiskConfig for time-aware position sizing.
                           If None, no horizon adjustment is applied.
        """
        self.delta_classes = np.array(sorted(delta_classes))
        self.min_edge_pct = min_edge_pct
        self.min_prob = min_prob
        self.use_maker_pricing = use_maker_pricing
        self.horizon_config = horizon_config

        # Create delta-to-index mapping
        self._delta_to_idx = {d: i for i, d in enumerate(self.delta_classes)}

    def _prob_settlement_at(self, delta_proba: np.ndarray, t_base: float, temp: int) -> float:
        """
        Compute P[settlement = temp] from delta distribution.

        settlement = t_base + delta -> delta = settlement - t_base
        """
        delta_needed = temp - int(round(t_base))
        if delta_needed in self._delta_to_idx:
            return float(delta_proba[self._delta_to_idx[delta_needed]])
        return 0.0

    def _prob_settlement_ge(self, delta_proba: np.ndarray, t_base: float, strike: float) -> float:
        """
        Compute P[settlement >= strike] from delta distribution.

        P[settlement >= strike] = sum over delta where (t_base + delta) >= strike
        """
        t_base_int = int(round(t_base))
        min_delta_needed = int(np.ceil(strike - t_base_int))

        # Sum probabilities for all delta >= min_delta_needed
        total_prob = 0.0
        for delta, idx in self._delta_to_idx.items():
            if delta >= min_delta_needed:
                total_prob += delta_proba[idx]

        return float(np.clip(total_prob, 0.0, 1.0))

    def _prob_settlement_le(self, delta_proba: np.ndarray, t_base: float, strike: float) -> float:
        """
        Compute P[settlement <= strike] from delta distribution.
        """
        return 1.0 - self._prob_settlement_ge(delta_proba, t_base, strike + 1)

    def _prob_settlement_in_range(
        self,
        delta_proba: np.ndarray,
        t_base: float,
        floor_strike: Optional[float],
        cap_strike: Optional[float]
    ) -> float:
        """
        Compute P[floor <= settlement <= cap] from delta distribution.

        Handles:
        - "between" brackets: floor <= T <= cap
        - "greater" brackets: T >= floor (cap is None)
        - "less" brackets: T <= cap (floor is None)
        """
        if floor_strike is None and cap_strike is None:
            return 1.0

        if floor_strike is None:
            # Low tail: P[settlement <= cap]
            return self._prob_settlement_le(delta_proba, t_base, cap_strike)

        if cap_strike is None:
            # High tail: P[settlement >= floor]
            return self._prob_settlement_ge(delta_proba, t_base, floor_strike)

        # Between bracket: P[floor <= settlement <= cap]
        # = P[settlement >= floor] - P[settlement > cap]
        # = P[settlement >= floor] - P[settlement >= cap + 1]
        p_ge_floor = self._prob_settlement_ge(delta_proba, t_base, floor_strike)
        p_ge_cap_plus1 = self._prob_settlement_ge(delta_proba, t_base, cap_strike + 1)
        return float(np.clip(p_ge_floor - p_ge_cap_plus1, 0.0, 1.0))

    def get_recommendation(
        self,
        delta_proba: np.ndarray,
        t_base: float,
        ticker: str,
        floor_strike: Optional[float],
        cap_strike: Optional[float],
        yes_bid: float,
        yes_ask: float,
        hours_to_event_close: Optional[float] = None,
    ) -> BracketRecommendation:
        """
        Get trading recommendation for a single bracket.

        Args:
            delta_proba: Model P[delta=k] array, shape (n_classes,)
            t_base: Current max so far, rounded (basis for delta)
            ticker: Bracket ticker
            floor_strike: Lower bound of bracket (None for low tail)
            cap_strike: Upper bound of bracket (None for high tail)
            yes_bid: Best bid for YES side (cents)
            yes_ask: Best ask for YES side (cents)
            hours_to_event_close: Hours until market close (for horizon-aware sizing)

        Returns:
            BracketRecommendation with action, edge, and rationale
        """
        # Compute horizon-aware adjustments (Point B)
        size_mult = 1.0
        edge_mult = 1.0
        if self.horizon_config is not None and hours_to_event_close is not None:
            size_mult = self.horizon_config.get_size_multiplier(hours_to_event_close)
            edge_mult = self.horizon_config.get_edge_multiplier(hours_to_event_close)

        # Effective edge threshold (stricter when far from close)
        effective_edge_threshold = self.min_edge_pct * edge_mult

        # Model probability that this bracket wins
        model_prob_yes = self._prob_settlement_in_range(
            delta_proba, t_base, floor_strike, cap_strike
        )
        model_prob_no = 1.0 - model_prob_yes

        # Market prices (convert to probability scale 0-1)
        # For buying YES: we pay yes_ask
        # For buying NO: we pay (100 - yes_bid) which is no_ask
        no_bid = 100 - yes_ask
        no_ask = 100 - yes_bid

        # Evaluate buying YES
        buy_yes_price = yes_ask
        buy_yes_edge_pct = 0.0
        buy_yes_ev = 0.0
        if buy_yes_price > 0:
            implied_prob_yes = buy_yes_price / 100.0
            buy_yes_edge_pct = (model_prob_yes - implied_prob_yes) / implied_prob_yes * 100
            # EV = P(win) * (100 - price) - P(lose) * price
            buy_yes_ev = model_prob_yes * (100 - buy_yes_price) - model_prob_no * buy_yes_price

        # Evaluate buying NO
        buy_no_price = no_ask
        buy_no_edge_pct = 0.0
        buy_no_ev = 0.0
        if buy_no_price > 0:
            implied_prob_no = buy_no_price / 100.0
            buy_no_edge_pct = (model_prob_no - implied_prob_no) / implied_prob_no * 100
            # EV = P(win) * (100 - price) - P(lose) * price
            buy_no_ev = model_prob_no * (100 - buy_no_price) - model_prob_yes * buy_no_price

        # Decide action based on edge and probability thresholds
        # Note: edge threshold is now horizon-adjusted (stricter when far from close)
        action = "hold"
        side = "none"
        entry_price = 0.0
        edge_pct = 0.0
        ev = 0.0
        reason = "No sufficient edge"

        if model_prob_yes >= self.min_prob and buy_yes_edge_pct >= effective_edge_threshold and buy_yes_ev > 0:
            if model_prob_no >= self.min_prob and buy_no_edge_pct >= effective_edge_threshold and buy_no_ev > 0:
                # Both sides have edge - pick the one with higher EV
                if buy_yes_ev >= buy_no_ev:
                    action = "buy_yes"
                    side = "yes"
                    entry_price = buy_yes_price
                    edge_pct = buy_yes_edge_pct
                    ev = buy_yes_ev
                    reason = f"YES has higher EV ({buy_yes_ev:.2f} vs {buy_no_ev:.2f})"
                else:
                    action = "buy_no"
                    side = "no"
                    entry_price = buy_no_price
                    edge_pct = buy_no_edge_pct
                    ev = buy_no_ev
                    reason = f"NO has higher EV ({buy_no_ev:.2f} vs {buy_yes_ev:.2f})"
            else:
                action = "buy_yes"
                side = "yes"
                entry_price = buy_yes_price
                edge_pct = buy_yes_edge_pct
                ev = buy_yes_ev
                reason = f"Model: {model_prob_yes*100:.1f}%, Market: {buy_yes_price:.0f}c, Edge: {buy_yes_edge_pct:.1f}%"
        elif model_prob_no >= self.min_prob and buy_no_edge_pct >= effective_edge_threshold and buy_no_ev > 0:
            action = "buy_no"
            side = "no"
            entry_price = buy_no_price
            edge_pct = buy_no_edge_pct
            ev = buy_no_ev
            reason = f"Model: {model_prob_no*100:.1f}%, Market: {buy_no_price:.0f}c, Edge: {buy_no_edge_pct:.1f}%"
        elif model_prob_yes < self.min_prob and model_prob_no < self.min_prob:
            reason = "Low confidence - both probabilities below threshold"
        elif buy_yes_edge_pct < effective_edge_threshold and buy_no_edge_pct < effective_edge_threshold:
            reason = f"Insufficient edge (YES: {buy_yes_edge_pct:.1f}%, NO: {buy_no_edge_pct:.1f}%, threshold: {effective_edge_threshold:.1f}%)"

        return BracketRecommendation(
            ticker=ticker,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            model_prob_yes=model_prob_yes,
            model_prob_no=model_prob_no,
            market_yes_bid=yes_bid,
            market_yes_ask=yes_ask,
            market_no_bid=no_bid,
            market_no_ask=no_ask,
            action=action,
            side=side,
            entry_price=entry_price,
            edge_pct=edge_pct,
            expected_value=ev,
            size_multiplier=size_mult,
            hours_to_close=hours_to_event_close,
            effective_edge_threshold=effective_edge_threshold,
            reason=reason,
        )

    def get_recommendations(
        self,
        delta_proba: np.ndarray,
        t_base: float,
        brackets_df: pd.DataFrame,
        hours_to_event_close: Optional[float] = None,
    ) -> list[BracketRecommendation]:
        """
        Get trading recommendations for all brackets.

        Args:
            delta_proba: Model P[delta=k] array, shape (n_classes,)
            t_base: Current max so far, rounded (basis for delta)
            brackets_df: DataFrame with columns:
                - ticker: Bracket ticker
                - floor_strike: Lower bound (None/NaN for low tail)
                - cap_strike: Upper bound (None/NaN for high tail)
                - yes_bid: Best bid for YES
                - yes_ask: Best ask for YES
            hours_to_event_close: Hours until market close (for horizon-aware sizing)

        Returns:
            List of BracketRecommendation for each bracket
        """
        recommendations = []

        for _, row in brackets_df.iterrows():
            rec = self.get_recommendation(
                delta_proba=delta_proba,
                t_base=t_base,
                ticker=row["ticker"],
                floor_strike=row.get("floor_strike") if pd.notna(row.get("floor_strike")) else None,
                cap_strike=row.get("cap_strike") if pd.notna(row.get("cap_strike")) else None,
                yes_bid=row.get("yes_bid", 0.0),
                yes_ask=row.get("yes_ask", 100.0),
                hours_to_event_close=hours_to_event_close,
            )
            recommendations.append(rec)

        return recommendations

    def find_best_bracket(
        self,
        delta_proba: np.ndarray,
        t_base: float,
        brackets_df: pd.DataFrame,
        hours_to_event_close: Optional[float] = None,
    ) -> Optional[BracketRecommendation]:
        """
        Find the single best bracket to trade based on EV.

        Returns:
            BracketRecommendation with highest positive EV, or None if no good trades
        """
        recs = self.get_recommendations(delta_proba, t_base, brackets_df, hours_to_event_close)

        # Filter to actionable recommendations
        actionable = [r for r in recs if r.action in ("buy_yes", "buy_no")]

        if not actionable:
            return None

        # Return highest EV
        return max(actionable, key=lambda r: r.expected_value)

    def summary_table(
        self,
        recommendations: list[BracketRecommendation],
    ) -> pd.DataFrame:
        """
        Create a summary table of recommendations.

        Returns:
            DataFrame with columns: ticker, strikes, model_p_yes, mkt_yes, edge, ev, action
        """
        rows = []
        for r in recommendations:
            strike_str = ""
            if r.floor_strike is None:
                strike_str = f"<={r.cap_strike:.0f}"
            elif r.cap_strike is None:
                strike_str = f">={r.floor_strike:.0f}"
            else:
                strike_str = f"{r.floor_strike:.0f}-{r.cap_strike:.0f}"

            rows.append({
                "ticker": r.ticker,
                "strikes": strike_str,
                "model_p_yes": f"{r.model_prob_yes*100:.1f}%",
                "mkt_yes_ask": f"{r.market_yes_ask:.0f}c",
                "edge_pct": f"{r.edge_pct:+.1f}%" if r.edge_pct != 0 else "-",
                "ev": f"{r.expected_value:+.2f}c" if r.expected_value != 0 else "-",
                "action": r.action.upper() if r.action != "hold" else "-",
                "reason": r.reason[:40],
            })

        return pd.DataFrame(rows)


def compute_model_expected_settle(delta_proba: np.ndarray, delta_classes: list[int], t_base: float) -> float:
    """
    Compute E[settlement] from model delta distribution.

    E[settlement] = t_base + E[delta] = t_base + sum(delta_k * P[delta=k])
    """
    expected_delta = sum(d * p for d, p in zip(delta_classes, delta_proba))
    return t_base + expected_delta


def compute_model_settle_std(delta_proba: np.ndarray, delta_classes: list[int]) -> float:
    """
    Compute Std[settlement] from model delta distribution.

    Since settlement = t_base + delta, Std[settlement] = Std[delta]
    """
    expected_delta = sum(d * p for d, p in zip(delta_classes, delta_proba))
    variance = sum(((d - expected_delta) ** 2) * p for d, p in zip(delta_classes, delta_proba))
    return float(np.sqrt(variance))
