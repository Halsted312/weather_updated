#!/usr/bin/env python3
"""
Edge diagnostic analysis: compare Ridge model probabilities to market mid-prices.

For each prediction row:
- p_model = calibrated Ridge probability
- p_mkt = yes_mid / 100 (where yes_mid = 0.5 * (yes_bid + yes_ask))
- edge_raw_cents = 100 * (p_model - p_mkt) (ignoring fees)

Output:
- Fraction of rows with |edge_raw_cents| > 1, 2, 3 cents
- Histogram of edge_raw_cents
- Summary stats (mean, median, std, quantiles)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from datetime import date
import argparse
import glob
import pandas as pd
import numpy as np
from sqlalchemy import text

from db.connection import get_session
from ml.dataset import CITY_CONFIG

logger = logging.getLogger(__name__)


def load_predictions(city: str, bracket: str, models_dir: str = "models/trained") -> pd.DataFrame:
    """Load all walk-forward predictions for a city/bracket."""
    pattern = os.path.join(
        models_dir,
        city,
        bracket,
        "win_*",
        "ridge_preds_*.csv"
    )

    pred_files = sorted(glob.glob(pattern))

    if not pred_files:
        logger.error(f"No prediction files found: {pattern}")
        return pd.DataFrame()

    logger.info(f"Loading predictions from {len(pred_files)} windows...")

    dfs = [pd.read_csv(f) for f in pred_files]
    predictions = pd.concat(dfs, ignore_index=True)

    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    predictions["event_date"] = pd.to_datetime(predictions["event_date"]).dt.date

    logger.info(
        f"Loaded {len(predictions)} predictions for {predictions['market_ticker'].nunique()} markets"
    )

    return predictions


def load_candles(city: str) -> pd.DataFrame:
    """Load minute-level candles for a city."""
    series_code = CITY_CONFIG[city]["series_code"]
    series_ticker = f"KXHIGH{series_code}"

    logger.info(f"Loading 1-minute candles for {city}...")

    with get_session() as session:
        query = text("""
            SELECT
                market_ticker,
                timestamp,
                close
            FROM candles
            WHERE market_ticker LIKE :series_pattern
              AND period_minutes = 1
            ORDER BY market_ticker, timestamp
        """)

        result = session.execute(query, {"series_pattern": f"{series_ticker}%"})
        rows = result.fetchall()

        if not rows:
            logger.error(f"No 1-minute candles found for {city}")
            return pd.DataFrame()

    candles = pd.DataFrame(
        rows,
        columns=["market_ticker", "timestamp", "close"]
    )
    candles["timestamp"] = pd.to_datetime(candles["timestamp"])

    # Estimate bid/ask from close price (assume 2¢ spread)
    # close is the YES price in cents, assume bid = close - 1, ask = close + 1
    candles["yes_bid_close"] = (candles["close"] - 1).clip(lower=1)
    candles["yes_ask_close"] = (candles["close"] + 1).clip(upper=99)

    logger.info(
        f"Loaded {len(candles)} candles for {candles['market_ticker'].nunique()} markets"
    )

    return candles


def analyze_edge(predictions: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """
    Join predictions to candles and compute edge diagnostics.

    Returns DataFrame with:
    - p_model: Model probability
    - yes_mid: Market mid price in cents
    - p_mkt: Market implied probability (yes_mid / 100)
    - edge_raw_cents: 100 * (p_model - p_mkt)
    """
    logger.info("Joining predictions to candles...")

    # Join on market_ticker and timestamp
    merged = predictions.merge(
        candles,
        on=["market_ticker", "timestamp"],
        how="inner",
    )

    logger.info(
        f"Joined {len(merged)} rows (from {len(predictions)} predictions and {len(candles)} candles)"
    )

    # Calculate market mid and implied probability
    merged["yes_mid"] = 0.5 * (merged["yes_bid_close"] + merged["yes_ask_close"])
    merged["p_mkt"] = merged["yes_mid"] / 100.0

    # Calculate raw edge (before fees)
    merged["edge_raw_cents"] = 100.0 * (merged["p_model"] - merged["p_mkt"])

    return merged


def print_edge_summary(df: pd.DataFrame):
    """Print summary statistics of edge distribution."""
    edge = df["edge_raw_cents"]

    print("\n" + "="*60)
    print("EDGE DIAGNOSTIC SUMMARY")
    print("="*60 + "\n")

    print(f"Total rows: {len(df):,}\n")

    # Summary stats
    print("Edge distribution (cents):")
    print(f"  Mean:   {edge.mean():>8.2f}¢")
    print(f"  Median: {edge.median():>8.2f}¢")
    print(f"  Std:    {edge.std():>8.2f}¢")
    print(f"  Min:    {edge.min():>8.2f}¢")
    print(f"  Max:    {edge.max():>8.2f}¢")

    # Quantiles
    print("\n  Quantiles:")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"    {int(q*100):>2}%: {edge.quantile(q):>8.2f}¢")

    # Fraction with |edge| > thresholds
    print("\n  Fraction with |edge| > threshold:")
    for thresh in [0.5, 1.0, 2.0, 3.0, 5.0]:
        frac = (edge.abs() > thresh).mean()
        count = (edge.abs() > thresh).sum()
        print(f"    >{thresh:.1f}¢: {frac:>6.1%} ({count:>8,} rows)")

    # Directional breakdown
    print("\n  Directional breakdown:")
    positive = (edge > 0).sum()
    negative = (edge < 0).sum()
    zero = (edge == 0).sum()
    print(f"    Positive edge: {positive:>8,} ({positive/len(df):>6.1%})")
    print(f"    Negative edge: {negative:>8,} ({negative/len(df):>6.1%})")
    print(f"    Zero edge:     {zero:>8,} ({zero/len(df):>6.1%})")

    # Histogram (text-based)
    print("\n  Histogram (¢):")
    bins = np.arange(-10, 11, 1)
    hist, _ = np.histogram(edge, bins=bins)
    max_count = hist.max()

    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        bar_width = int(50 * hist[i] / max_count) if max_count > 0 else 0
        bar = "█" * bar_width
        print(f"    [{left:>4.0f}, {right:>4.0f}): {hist[i]:>8,} {bar}")

    print("\n" + "="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze model edge vs market prices")
    parser.add_argument("--city", default="chicago", help="City name")
    parser.add_argument("--bracket", choices=["between", "greater", "less"],
                        default="between", help="Bracket type")
    parser.add_argument("--models-dir", default="models/trained", help="Models directory")
    parser.add_argument("--output", help="Save detailed results to CSV")

    args = parser.parse_args()

    # Load data
    predictions = load_predictions(args.city, args.bracket, args.models_dir)
    if predictions.empty:
        logger.error("No predictions loaded, exiting")
        return 1

    candles = load_candles(args.city)
    if candles.empty:
        logger.error("No candles loaded, exiting")
        return 1

    # Analyze edge
    edge_df = analyze_edge(predictions, candles)

    # Print summary
    print_edge_summary(edge_df)

    # Save to CSV if requested
    if args.output:
        edge_df.to_csv(args.output, index=False)
        logger.info(f"Saved detailed results to {args.output}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    sys.exit(main())
