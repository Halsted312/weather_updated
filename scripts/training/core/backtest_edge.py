#!/usr/bin/env python3
"""Backtest edge detection strategy.

This script tests the edge detection model by comparing forecast-implied
temperatures vs market-implied temperatures on historical data.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/backtest_edge.py
    PYTHONPATH=. .venv/bin/python scripts/backtest_edge.py --city chicago --days 30
    PYTHONPATH=. .venv/bin/python scripts/backtest_edge.py --city austin --threshold 1.0
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

# Add project root to path (scripts/training/core/ -> 3 levels up)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.db import get_db_session
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.edge.implied_temp import (
    compute_market_implied_temp,
    compute_forecast_implied_temp,
)
from models.edge.detector import detect_edge, EdgeSignal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# City configurations
CITY_CONFIG = {
    "chicago": {"ticker_prefix": "KXHIGHCHI", "tz": "America/Chicago"},
    "austin": {"ticker_prefix": "KXHIGHAUS", "tz": "America/Chicago"},
    "denver": {"ticker_prefix": "KXHIGHDEN", "tz": "America/Denver"},
    "los_angeles": {"ticker_prefix": "KXHIGHLA", "tz": "America/Los_Angeles"},
    "miami": {"ticker_prefix": "KXHIGHMIA", "tz": "America/New_York"},
    "philadelphia": {"ticker_prefix": "KXHIGHPHI", "tz": "America/New_York"},
}


@dataclass
class EdgeSnapshot:
    """Result of edge detection at a single snapshot."""
    day: date
    snapshot_time: datetime
    forecast_temp: float
    market_temp: float
    edge: float
    signal: str
    confidence: float
    base_temp: float
    predicted_delta: int
    settlement_temp: Optional[float] = None
    pnl: Optional[float] = None


def load_bracket_candles_for_event(
    session,
    city: str,
    event_date: date,
    snapshot_time: datetime,
) -> dict[str, pd.DataFrame]:
    """Load bracket candles for all brackets of an event up to snapshot time.

    Returns dict mapping bracket label to candle DataFrame.
    """
    config = CITY_CONFIG.get(city)
    if not config:
        logger.warning(f"Unknown city: {city}")
        return {}

    prefix = config["ticker_prefix"]

    # Event ticker pattern: KXHIGHCHI-25NOV30 for Chicago Nov 30, 2025
    event_suffix = event_date.strftime("%y%b%d").upper()
    ticker_pattern = f"{prefix}-{event_suffix}%"

    # Query all bracket candles for this event up to snapshot time
    query = text("""
        SELECT
            ticker,
            bucket_start,
            yes_bid_close,
            yes_ask_close,
            volume,
            open_interest
        FROM kalshi.candles_1m_dense
        WHERE ticker LIKE :ticker_pattern
          AND bucket_start <= :snapshot_time
        ORDER BY ticker, bucket_start
    """)

    result = session.execute(query, {
        "ticker_pattern": ticker_pattern,
        "snapshot_time": snapshot_time,
    })
    rows = result.fetchall()

    if not rows:
        return {}

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=[
        "ticker", "bucket_start", "yes_bid_close", "yes_ask_close",
        "volume", "open_interest"
    ])

    # Parse bracket from ticker: KXHIGHCHI-25NOV30-T82 -> "82"
    # Or KXHIGHCHI-25NOV30-B82 for "82-83" bracket
    bracket_candles = {}

    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()

        # Extract bracket info from ticker
        parts = ticker.split("-")
        if len(parts) >= 3:
            bracket_part = parts[-1]  # e.g., "T82", "B82", "T83LO", etc.

            # Parse bracket strikes
            # T82 = threshold at 82 (could be 82-83 or >82)
            # B82 = bracket 82-83
            if bracket_part.startswith("T"):
                strike = bracket_part[1:].replace("LO", "").replace("HI", "")
                try:
                    strike_val = int(strike)
                    if "LO" in bracket_part:
                        label = f"<{strike_val}"
                    elif "HI" in bracket_part:
                        label = f">{strike_val}"
                    else:
                        label = f"{strike_val}-{strike_val + 1}"
                except ValueError:
                    label = bracket_part
            elif bracket_part.startswith("B"):
                strike = bracket_part[1:]
                try:
                    strike_val = int(strike)
                    label = f"{strike_val}-{strike_val + 1}"
                except ValueError:
                    label = bracket_part
            else:
                label = bracket_part

            bracket_candles[label] = ticker_df

    return bracket_candles


def get_settlement_temp(session, city: str, event_date: date) -> Optional[float]:
    """Get settlement temperature for an event."""
    query = text("""
        SELECT tmax_final
        FROM wx.settlement
        WHERE city = :city AND date_local = :event_date
    """)
    result = session.execute(query, {"city": city, "event_date": event_date})
    row = result.fetchone()
    return float(row[0]) if row else None


def backtest_edge_strategy(
    city: str,
    start_date: date,
    end_date: date,
    model: OrdinalDeltaTrainer,
    edge_threshold: float = 1.5,
    snapshot_interval_min: int = 60,
) -> pd.DataFrame:
    """Backtest edge detection strategy on historical data.

    Args:
        city: City to backtest
        start_date: Start date
        end_date: End date
        model: Trained ordinal model
        edge_threshold: Minimum edge to generate signal
        snapshot_interval_min: Minutes between snapshots to test

    Returns:
        DataFrame with edge detection results for each snapshot.
    """
    logger.info(f"Backtesting {city} from {start_date} to {end_date}")

    # Load test data (we need features for model predictions)
    test_parquet = Path(f"models/saved/{city}/test_data_full.parquet")
    if test_parquet.exists():
        df_test = pd.read_parquet(test_parquet)
        logger.info(f"Loaded {len(df_test):,} test samples from {test_parquet}")
    else:
        logger.error(f"Test data not found: {test_parquet}")
        return pd.DataFrame()

    results = []

    with get_db_session() as session:
        # Get unique days in test data
        test_days = df_test["day"].unique()
        test_days = [d for d in test_days if start_date <= d <= end_date]
        logger.info(f"Testing {len(test_days)} days")

        for day in sorted(test_days):
            # Get settlement temp for this day
            settlement = get_settlement_temp(session, city, day)
            if settlement is None:
                logger.debug(f"No settlement for {city} {day}")
                continue

            # Get snapshots for this day
            day_df = df_test[df_test["day"] == day].copy()
            if day_df.empty:
                continue

            # Sample snapshots at interval
            unique_times = day_df["cutoff_time"].unique()
            sampled_times = unique_times[::max(1, snapshot_interval_min // 5)]

            for snapshot_time in sampled_times:
                snapshot_df = day_df[day_df["cutoff_time"] == snapshot_time]
                if snapshot_df.empty:
                    continue

                # Get base temp from forecast
                base_temp = snapshot_df["t_forecast_base"].iloc[0]
                if pd.isna(base_temp):
                    base_temp = snapshot_df["fcst_prev_max_f"].iloc[0]
                if pd.isna(base_temp):
                    continue

                # Get model predictions
                try:
                    delta_probs = model.predict_proba(snapshot_df)
                    forecast_result = compute_forecast_implied_temp(
                        delta_probs=delta_probs[0],
                        base_temp=base_temp,
                    )
                except Exception as e:
                    logger.debug(f"Prediction error: {e}")
                    continue

                # Load bracket candles for market-implied temp
                bracket_candles = load_bracket_candles_for_event(
                    session, city, day, snapshot_time
                )

                if not bracket_candles:
                    # No market data, skip
                    continue

                # Compute market-implied temp
                market_result = compute_market_implied_temp(
                    bracket_candles=bracket_candles,
                    snapshot_time=snapshot_time,
                )

                if not market_result.valid:
                    continue

                # Detect edge
                edge_result = detect_edge(
                    forecast_implied=forecast_result.implied_temp,
                    market_implied=market_result.implied_temp,
                    forecast_uncertainty=forecast_result.uncertainty,
                    market_uncertainty=market_result.uncertainty,
                    threshold=edge_threshold,
                )

                # Compute P&L based on settlement
                # Simple model: if signal was correct direction, we won
                pnl = None
                if edge_result.signal != EdgeSignal.NO_TRADE:
                    if edge_result.signal == EdgeSignal.BUY_HIGH:
                        # We thought market was too low (temp will be higher)
                        pnl = 1.0 if settlement > market_result.implied_temp else -1.0
                    else:  # BUY_LOW
                        # We thought market was too high (temp will be lower)
                        pnl = 1.0 if settlement < market_result.implied_temp else -1.0

                results.append(EdgeSnapshot(
                    day=day,
                    snapshot_time=snapshot_time,
                    forecast_temp=forecast_result.implied_temp,
                    market_temp=market_result.implied_temp,
                    edge=edge_result.edge,
                    signal=edge_result.signal.value,
                    confidence=edge_result.confidence,
                    base_temp=base_temp,
                    predicted_delta=forecast_result.predicted_delta,
                    settlement_temp=settlement,
                    pnl=pnl,
                ))

    # Convert to DataFrame
    if not results:
        logger.warning("No results generated")
        return pd.DataFrame()

    df_results = pd.DataFrame([
        {
            "day": r.day,
            "snapshot_time": r.snapshot_time,
            "forecast_temp": r.forecast_temp,
            "market_temp": r.market_temp,
            "edge": r.edge,
            "signal": r.signal,
            "confidence": r.confidence,
            "base_temp": r.base_temp,
            "predicted_delta": r.predicted_delta,
            "settlement_temp": r.settlement_temp,
            "pnl": r.pnl,
        }
        for r in results
    ])

    return df_results


def analyze_results(df: pd.DataFrame) -> dict:
    """Analyze backtest results."""
    if df.empty:
        return {}

    # Overall stats
    n_snapshots = len(df)
    n_signals = len(df[df["signal"] != "no_trade"])
    signal_rate = n_signals / n_snapshots if n_snapshots > 0 else 0

    # Trade stats
    trades = df[df["signal"] != "no_trade"]
    if not trades.empty:
        n_wins = len(trades[trades["pnl"] > 0])
        n_losses = len(trades[trades["pnl"] < 0])
        win_rate = n_wins / len(trades) if len(trades) > 0 else 0
        total_pnl = trades["pnl"].sum()
        avg_edge = trades["edge"].abs().mean()
        avg_confidence = trades["confidence"].mean()
    else:
        n_wins = n_losses = 0
        win_rate = total_pnl = avg_edge = avg_confidence = 0

    # Buy high vs buy low breakdown
    buy_high = df[df["signal"] == "buy_high"]
    buy_low = df[df["signal"] == "buy_low"]

    buy_high_wins = len(buy_high[buy_high["pnl"] > 0]) if not buy_high.empty else 0
    buy_low_wins = len(buy_low[buy_low["pnl"] > 0]) if not buy_low.empty else 0

    return {
        "n_snapshots": n_snapshots,
        "n_signals": n_signals,
        "signal_rate": signal_rate,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_edge": avg_edge,
        "avg_confidence": avg_confidence,
        "buy_high_signals": len(buy_high),
        "buy_high_wins": buy_high_wins,
        "buy_high_win_rate": buy_high_wins / len(buy_high) if not buy_high.empty else 0,
        "buy_low_signals": len(buy_low),
        "buy_low_wins": buy_low_wins,
        "buy_low_win_rate": buy_low_wins / len(buy_low) if not buy_low.empty else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest edge detection strategy")
    parser.add_argument("--city", type=str, default="chicago",
                       choices=list(CITY_CONFIG.keys()),
                       help="City to backtest")
    parser.add_argument("--days", type=int, default=60,
                       help="Number of days to backtest")
    parser.add_argument("--threshold", type=float, default=1.5,
                       help="Edge threshold in degrees F")
    parser.add_argument("--interval", type=int, default=60,
                       help="Snapshot interval in minutes")
    args = parser.parse_args()

    print("=" * 60)
    print("EDGE DETECTION BACKTEST")
    print("=" * 60)
    print(f"City: {args.city}")
    print(f"Edge threshold: {args.threshold}°F")
    print(f"Snapshot interval: {args.interval} min")
    print()

    # Load model
    model_path = Path(f"models/saved/{args.city}/ordinal_catboost_optuna.pkl")
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    logger.info(f"Loading model from {model_path}")
    model = OrdinalDeltaTrainer()
    model.load(model_path)

    # Determine date range
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=args.days)

    print(f"Date range: {start_date} to {end_date}")
    print()

    # Run backtest
    df_results = backtest_edge_strategy(
        city=args.city,
        start_date=start_date,
        end_date=end_date,
        model=model,
        edge_threshold=args.threshold,
        snapshot_interval_min=args.interval,
    )

    if df_results.empty:
        print("No results - check if test data exists")
        return 1

    # Analyze results
    stats = analyze_results(df_results)

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Snapshots analyzed: {stats['n_snapshots']:,}")
    print(f"Signals generated: {stats['n_signals']:,} ({stats['signal_rate']:.1%})")
    print()
    print(f"Trades: {stats['n_signals']}")
    print(f"  Wins: {stats['n_wins']}")
    print(f"  Losses: {stats['n_losses']}")
    print(f"  Win Rate: {stats['win_rate']:.1%}")
    print(f"  Total P&L: {stats['total_pnl']:.1f} units")
    print()
    print(f"BUY_HIGH signals: {stats['buy_high_signals']}")
    print(f"  Win rate: {stats['buy_high_win_rate']:.1%}")
    print(f"BUY_LOW signals: {stats['buy_low_signals']}")
    print(f"  Win rate: {stats['buy_low_win_rate']:.1%}")
    print()
    print(f"Avg edge magnitude: {stats['avg_edge']:.2f}°F")
    print(f"Avg confidence: {stats['avg_confidence']:.2f}x threshold")
    print()

    # Save results
    output_path = Path(f"models/saved/{args.city}/edge_backtest_results.parquet")
    df_results.to_parquet(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Show sample of trades
    print()
    print("=" * 60)
    print("SAMPLE TRADES")
    print("=" * 60)
    trades = df_results[df_results["signal"] != "no_trade"].head(10)
    if not trades.empty:
        for _, row in trades.iterrows():
            print(f"{row['day']} | edge={row['edge']:+.1f}°F | "
                  f"signal={row['signal']} | "
                  f"forecast={row['forecast_temp']:.1f}°F | "
                  f"market={row['market_temp']:.1f}°F | "
                  f"settle={row['settlement_temp']:.1f}°F | "
                  f"pnl={row['pnl']:+.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
