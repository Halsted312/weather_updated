#!/usr/bin/env python3
"""
Diagnostic tools for backtest analysis.

Provides per-trade logging, edge analysis, calibration metrics (Brier, log-loss),
and visualization utilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TradeDiagnostics:
    """
    Collects and analyzes diagnostic data from backtest trades.

    Tracks model edge, execution quality, and calibration metrics per trade.
    """

    def __init__(self):
        """Initialize empty diagnostics collector."""
        self.trades: List[Dict] = []

    def add_trade(
        self,
        timestamp: datetime,
        market_ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        fee_type: str,
        fee_cents: int,
        p_model: Optional[float] = None,
        p_market: Optional[float] = None,
        edge_cents: Optional[float] = None,
        spread_cents: Optional[int] = None,
        time_to_close_minutes: Optional[float] = None,
        outcome: Optional[str] = None,  # "YES" or "NO"
        pnl_cents: Optional[int] = None,
    ):
        """
        Record a trade with diagnostic information.

        Args:
            timestamp: Trade execution time
            market_ticker: Market identifier
            side: "buy" or "sell"
            contracts: Number of contracts traded
            price_cents: Execution price in cents
            fee_type: "taker" or "maker"
            fee_cents: Fees paid
            p_model: Model probability estimate [0, 1]
            p_market: Market probability from mid-price
            edge_cents: Expected edge in cents
            spread_cents: Bid-ask spread at trade time
            time_to_close_minutes: Minutes until market close
            outcome: Settlement outcome ("YES" or "NO")
            pnl_cents: Realized P&L in cents
        """
        self.trades.append({
            "timestamp": timestamp,
            "market_ticker": market_ticker,
            "side": side,
            "contracts": contracts,
            "price_cents": price_cents,
            "fee_type": fee_type,
            "fee_cents": fee_cents,
            "p_model": p_model,
            "p_market": p_market,
            "edge_cents": edge_cents,
            "spread_cents": spread_cents,
            "time_to_close_minutes": time_to_close_minutes,
            "outcome": outcome,
            "pnl_cents": pnl_cents,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()

        df = pd.DataFrame(self.trades)

        # Add derived columns
        if "outcome" in df.columns and "side" in df.columns:
            # Win = YES outcome and bought, or NO outcome and sold
            df["win"] = (
                ((df["outcome"] == "YES") & (df["side"] == "buy")) |
                ((df["outcome"] == "NO") & (df["side"] == "sell"))
            )

        # Edge realization = actual outcome probability vs model probability
        if "outcome" in df.columns and "p_model" in df.columns:
            df["outcome_binary"] = (df["outcome"] == "YES").astype(float)
            df["edge_realized_cents"] = 100 * (df["outcome_binary"] - df["p_market"])

        return df

    def save(self, filepath: str):
        """Save diagnostics to CSV file."""
        df = self.to_dataframe()
        if df.empty:
            logger.warning("No trades to save")
            return

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} trade diagnostics to {filepath}")

    def compute_metrics(self) -> Dict:
        """
        Compute aggregate diagnostic metrics.

        Returns:
            Dictionary with:
            - brier_score: Calibration metric
            - log_loss: Calibration metric
            - edge_accuracy: Correlation between predicted edge and realized edge
            - avg_spread: Average spread paid
            - fee_ratio: Fees / gross P&L
        """
        df = self.to_dataframe()

        if df.empty or "outcome" not in df.columns or "p_model" not in df.columns:
            return {}

        # Filter rows with required data
        valid = df.dropna(subset=["outcome", "p_model"])

        if valid.empty:
            return {}

        # Brier score: avg((p - y)^2)
        y_true = (valid["outcome"] == "YES").astype(float)
        brier = ((valid["p_model"] - y_true) ** 2).mean()

        # Log loss: -avg(y*log(p) + (1-y)*log(1-p))
        # Clip probabilities to avoid log(0)
        p_clipped = valid["p_model"].clip(1e-15, 1 - 1e-15)
        log_loss = -(
            y_true * np.log(p_clipped) + (1 - y_true) * np.log(1 - p_clipped)
        ).mean()

        metrics = {
            "brier_score": float(brier),
            "log_loss": float(log_loss),
            "num_trades": len(valid),
        }

        # Edge correlation (if available)
        if "edge_cents" in valid.columns and "edge_realized_cents" in valid.columns:
            edge_valid = valid.dropna(subset=["edge_cents", "edge_realized_cents"])
            if len(edge_valid) > 1:
                corr = edge_valid["edge_cents"].corr(edge_valid["edge_realized_cents"])
                metrics["edge_correlation"] = float(corr)

        # Spread and fee stats
        if "spread_cents" in valid.columns:
            metrics["avg_spread_cents"] = float(valid["spread_cents"].mean())

        if "fee_cents" in valid.columns and "pnl_cents" in valid.columns:
            total_fees = valid["fee_cents"].sum()
            gross_pnl = (valid["pnl_cents"] + valid["fee_cents"]).sum()
            if gross_pnl != 0:
                metrics["fee_ratio"] = float(total_fees / gross_pnl)

        return metrics

    def print_summary(self):
        """Print diagnostic summary to console."""
        metrics = self.compute_metrics()

        if not metrics:
            logger.info("No diagnostic metrics available")
            return

        logger.info("\n" + "=" * 60)
        logger.info("DIAGNOSTIC METRICS")
        logger.info("=" * 60)

        logger.info(f"Number of trades:        {metrics.get('num_trades', 0)}")
        logger.info(f"Brier score:             {metrics.get('brier_score', 0):.4f}")
        logger.info(f"Log loss:                {metrics.get('log_loss', 0):.4f}")

        if "edge_correlation" in metrics:
            logger.info(f"Edge correlation:        {metrics['edge_correlation']:.3f}")

        if "avg_spread_cents" in metrics:
            logger.info(f"Avg spread:              {metrics['avg_spread_cents']:.1f}¢")

        if "fee_ratio" in metrics:
            logger.info(f"Fee ratio (fees/gross):  {metrics['fee_ratio']:.1%}")

        logger.info("=" * 60)


def analyze_edge_by_bucket(
    df: pd.DataFrame,
    edge_col: str = "edge_cents",
    outcome_col: str = "win",
    n_buckets: int = 5,
) -> pd.DataFrame:
    """
    Analyze edge quality by bucketing predicted edge.

    Args:
        df: DataFrame with trade diagnostics
        edge_col: Column name for predicted edge
        outcome_col: Column name for binary outcome
        n_buckets: Number of buckets for edge quantiles

    Returns:
        DataFrame with bucket statistics
    """
    if df.empty or edge_col not in df.columns or outcome_col not in df.columns:
        return pd.DataFrame()

    valid = df.dropna(subset=[edge_col, outcome_col])

    if valid.empty:
        return pd.DataFrame()

    # Create buckets
    valid["edge_bucket"] = pd.qcut(
        valid[edge_col],
        q=n_buckets,
        labels=[f"Q{i+1}" for i in range(n_buckets)],
        duplicates="drop"
    )

    # Aggregate by bucket
    bucket_stats = valid.groupby("edge_bucket", observed=True).agg({
        edge_col: ["mean", "std", "count"],
        outcome_col: "mean",  # Win rate
        "pnl_cents": "sum" if "pnl_cents" in valid.columns else "count",
    }).round(2)

    return bucket_stats


def compute_per_window_metrics(
    predictions_dir: str,
    settlements_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Brier score and log-loss per walk-forward window.

    Args:
        predictions_dir: Directory containing ridge_preds_*.csv files
        settlements_df: DataFrame with columns [market_ticker, outcome]

    Returns:
        DataFrame with columns [window, brier_score, log_loss, num_predictions]
    """
    pred_dir = Path(predictions_dir)

    if not pred_dir.exists():
        logger.warning(f"Predictions directory not found: {predictions_dir}")
        return pd.DataFrame()

    # Find all prediction files
    pred_files = sorted(pred_dir.glob("*/ridge_preds_*.csv"))

    if not pred_files:
        logger.warning(f"No prediction files found in {predictions_dir}")
        return pd.DataFrame()

    results = []

    for pred_file in pred_files:
        # Parse window name from path
        window_name = pred_file.parent.name

        # Load predictions
        preds = pd.read_csv(pred_file)

        # Merge with settlements
        merged = preds.merge(
            settlements_df[["market_ticker", "outcome"]],
            on="market_ticker",
            how="inner"
        )

        if merged.empty:
            logger.warning(f"No matching settlements for {pred_file.name}")
            continue

        # Compute metrics
        y_true = (merged["outcome"] == "YES").astype(float)
        p_model = merged["p_model"].clip(1e-15, 1 - 1e-15)

        brier = ((p_model - y_true) ** 2).mean()
        log_loss = -(
            y_true * np.log(p_model) + (1 - y_true) * np.log(1 - p_model)
        ).mean()

        results.append({
            "window": window_name,
            "brier_score": float(brier),
            "log_loss": float(log_loss),
            "num_predictions": len(merged),
        })

    return pd.DataFrame(results)
