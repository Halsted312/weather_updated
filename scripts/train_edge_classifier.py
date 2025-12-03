#!/usr/bin/env python3
"""Train ML Edge Classifier using CatBoost with Optuna tuning.

This script:
1. Loads combined train+test parquet data (700+ days)
2. Runs the ordinal model to generate edge signals
3. Joins with settlement outcomes
4. Trains EdgeClassifier with Optuna hyperparameter tuning

Usage:
    PYTHONPATH=. .venv/bin/python scripts/train_edge_classifier.py
    PYTHONPATH=. .venv/bin/python scripts/train_edge_classifier.py --city chicago --trials 30
    PYTHONPATH=. .venv/bin/python scripts/train_edge_classifier.py --trials 150 --workers 12
"""

import argparse
import logging
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db import get_db_session
from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.edge.implied_temp import (
    compute_market_implied_temp,
    compute_forecast_implied_temp,
)
from models.edge.detector import detect_edge, EdgeSignal
from models.edge.classifier import EdgeClassifier

# Suppress warnings during training
warnings.filterwarnings("ignore", category=UserWarning)

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


def load_combined_data(city: str) -> pd.DataFrame:
    """Load and combine train + test parquet files.

    Args:
        city: City name (e.g., 'chicago')

    Returns:
        Combined DataFrame with all data
    """
    base_path = Path(f"models/saved/{city}")

    train_path = base_path / "train_data_full.parquet"
    test_path = base_path / "test_data_full.parquet"

    dfs = []

    if train_path.exists():
        df_train = pd.read_parquet(train_path)
        logger.info(f"Loaded train data: {len(df_train):,} rows")
        dfs.append(df_train)
    else:
        logger.warning(f"Train data not found: {train_path}")

    if test_path.exists():
        df_test = pd.read_parquet(test_path)
        logger.info(f"Loaded test data: {len(df_test):,} rows")
        dfs.append(df_test)
    else:
        logger.warning(f"Test data not found: {test_path}")

    if not dfs:
        raise FileNotFoundError(f"No data found for {city}")

    df_combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data: {len(df_combined):,} rows")

    # Sort by day and time
    if "day" in df_combined.columns and "cutoff_time" in df_combined.columns:
        df_combined = df_combined.sort_values(["day", "cutoff_time"]).reset_index(
            drop=True
        )

    return df_combined


def load_bracket_candles_for_event(
    session,
    city: str,
    event_date: date,
    snapshot_time,
) -> dict[str, pd.DataFrame]:
    """Load bracket candles for all brackets of an event up to snapshot time.

    Handles both old format (HIGHAUS-) and new format (KXHIGHAUS-) tickers.
    """
    config = CITY_CONFIG.get(city)
    if not config:
        return {}

    prefix = config["ticker_prefix"]
    event_suffix = event_date.strftime("%y%b%d").upper()

    # Query for BOTH old and new ticker formats
    # Old: HIGHAUS-23AUG01-...
    # New: KXHIGHAUS-24DEC01-...
    old_prefix = prefix.replace("KXHIGH", "HIGH")  # Remove "KX"

    query = text("""
        SELECT
            ticker,
            bucket_start,
            yes_bid_close,
            yes_ask_close,
            volume,
            open_interest
        FROM kalshi.candles_1m_dense
        WHERE (ticker LIKE :new_pattern OR ticker LIKE :old_pattern)
          AND bucket_start <= :snapshot_time
        ORDER BY ticker, bucket_start
    """)

    result = session.execute(
        query,
        {
            "new_pattern": f"{prefix}-{event_suffix}%",
            "old_pattern": f"{old_prefix}-{event_suffix}%",
            "snapshot_time": snapshot_time,
        },
    )
    rows = result.fetchall()

    if not rows:
        return {}

    df = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "bucket_start",
            "yes_bid_close",
            "yes_ask_close",
            "volume",
            "open_interest",
        ],
    )

    bracket_candles = {}
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()
        parts = ticker.split("-")
        if len(parts) >= 3:
            bracket_part = parts[-1]
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
                    strike_val = float(strike)  # Handle decimal strikes (e.g., 86.5)
                    label = f"{strike_val:g}-{strike_val + 1:g}"  # Format: "86.5-87.5" or "86-87"
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


def load_all_settlements(city: str, dates: list) -> dict:
    """Batch load all settlements in one query.

    Args:
        city: City name
        dates: List of dates to load

    Returns:
        Dictionary mapping date -> settlement temperature
    """
    if not dates:
        return {}

    with get_db_session() as session:
        # Build date list for SQL
        date_strs = [f"'{d}'" for d in dates]
        query = text(f"""
            SELECT date_local, tmax_final
            FROM wx.settlement
            WHERE city = :city AND date_local IN ({','.join(date_strs)})
        """)
        result = session.execute(query, {"city": city})
        return {row[0]: float(row[1]) for row in result.fetchall()}


def load_all_candles_batch(city: str, dates: list) -> dict:
    """Batch load ALL bracket candles for ALL days in ONE query.

    Args:
        city: City name
        dates: List of dates to load candles for

    Returns:
        Dictionary mapping (day, label) -> DataFrame of candles
    """
    if not dates:
        return {}

    config = CITY_CONFIG.get(city)
    if not config:
        return {}

    prefix = config["ticker_prefix"]
    old_prefix = prefix.replace("KXHIGH", "HIGH")  # Remove "KX" for old format

    # Build date suffix mapping
    date_to_suffix = {}
    for d in dates:
        suffix = d.strftime("%y%b%d").upper()
        date_to_suffix[suffix] = d

    logger.info(f"Loading candles for {len(dates)} days (checking both old and new ticker formats)...")

    # CRITICAL FIX: Batch queries to avoid PostgreSQL query size limits
    # With 1062 days × 2 formats = 2124 patterns = 42KB query (too large!)
    # Process in chunks of 200 days = 400 patterns per query
    BATCH_SIZE = 200
    all_rows = []

    with get_db_session() as session:
        for batch_start in range(0, len(dates), BATCH_SIZE):
            batch_dates = dates[batch_start:batch_start + BATCH_SIZE]
            ticker_patterns = []

            for d in batch_dates:
                suffix = d.strftime("%y%b%d").upper()
                # Add both formats
                ticker_patterns.append(f"{prefix}-{suffix}%")
                ticker_patterns.append(f"{old_prefix}-{suffix}%")

            patterns_sql = ",".join([f"'{p}'" for p in ticker_patterns])
            query = text(f"""
                SELECT
                    ticker,
                    bucket_start,
                    yes_bid_close,
                    yes_ask_close,
                    volume,
                    open_interest
                FROM kalshi.candles_1m_dense
                WHERE ticker LIKE ANY (ARRAY[{patterns_sql}])
                ORDER BY ticker, bucket_start
            """)

            result = session.execute(query)
            batch_rows = result.fetchall()
            all_rows.extend(batch_rows)

            if batch_rows:
                logger.info(f"  Batch {batch_start//BATCH_SIZE + 1}: Loaded {len(batch_rows):,} candles")

    if not all_rows:
        logger.error("No candles loaded from database!")
        return {}

    logger.info(f"Total loaded: {len(all_rows):,} candle rows")
    rows = all_rows

    df = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "bucket_start",
            "yes_bid_close",
            "yes_ask_close",
            "volume",
            "open_interest",
        ],
    )

    # Organize by (day, label) for fast lookup
    candle_cache = {}
    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()
        parts = ticker.split("-")
        if len(parts) >= 3:
            # Extract date suffix and bracket label
            date_suffix = parts[1]  # e.g., "24NOV15"
            day = date_to_suffix.get(date_suffix)
            if day is None:
                continue

            bracket_part = parts[-1]
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
                    strike_val = float(strike)  # Handle decimal strikes (e.g., 86.5)
                    label = f"{strike_val:g}-{strike_val + 1:g}"  # Format: "86.5-87.5" or "86-87"
                except ValueError:
                    label = bracket_part
            else:
                label = bracket_part

            candle_cache[(day, label)] = ticker_df

    logger.info(f"Organized into {len(candle_cache):,} (day, bracket) entries")
    return candle_cache


def get_candles_from_cache(
    candle_cache: dict, day: date, snapshot_time
) -> dict[str, pd.DataFrame]:
    """Get bracket candles for a specific day/time from pre-loaded cache."""
    bracket_candles = {}

    # CRITICAL: snapshot_time is in LOCAL time (naive), candles are in UTC
    # Need to convert snapshot to UTC for comparison
    snapshot_ts = pd.Timestamp(snapshot_time)

    # If snapshot is naive, assume it's in local time and convert to UTC
    if snapshot_ts.tz is None:
        try:
            # Localize to city timezone (handle DST transitions)
            # ambiguous='infer': For fall-back, infer from context
            # nonexistent='shift_forward': For spring-forward, shift to next valid time
            snapshot_ts = snapshot_ts.tz_localize("America/Chicago", ambiguous='infer', nonexistent='shift_forward')
            # Convert to UTC for comparison with candles
            snapshot_ts = snapshot_ts.tz_convert("UTC")
        except Exception:
            # DST transition edge case - return empty (happens ~6 days/year)
            return {}

    # Make naive for comparison (both are now in UTC)
    snapshot_ts = snapshot_ts.tz_convert(None) if snapshot_ts.tz else snapshot_ts

    # Find all entries for this day
    for (d, label), df in candle_cache.items():
        if d != day:
            continue
        # Filter to candles up to snapshot_time (handle timezone)
        bucket_start = df["bucket_start"]
        if bucket_start.dt.tz is not None:
            bucket_start = bucket_start.dt.tz_convert(None)
        filtered = df[bucket_start <= snapshot_ts].copy()
        if not filtered.empty:
            bracket_candles[label] = filtered

    return bracket_candles


def _process_single_day(
    day: date,
    day_df: pd.DataFrame,
    model: OrdinalDeltaTrainer,
    candle_cache: dict,
    settlement: float,
    edge_threshold: float,
    sample_rate: int,
) -> list:
    """Process a single day's edge data - runs in thread with shared memory.

    Args:
        day: The date to process
        day_df: Feature DataFrame for this day
        model: Pre-loaded ordinal model (shared)
        candle_cache: Pre-loaded candle cache (shared)
        settlement: Settlement temperature
        edge_threshold: Threshold for edge detection
        sample_rate: Sample every Nth snapshot

    Returns:
        List of result dictionaries for this day
    """
    if day_df.empty or settlement is None:
        return []

    results = []
    unique_times = day_df["cutoff_time"].unique()
    sampled_times = unique_times[::sample_rate]

    for snapshot_time in sampled_times:
        snapshot_df = day_df[day_df["cutoff_time"] == snapshot_time]
        if snapshot_df.empty:
            continue

        # Get base temp
        base_temp = snapshot_df["t_forecast_base"].iloc[0]
        if pd.isna(base_temp):
            base_temp = snapshot_df["fcst_prev_max_f"].iloc[0]
        if pd.isna(base_temp):
            continue

        # Get model prediction
        try:
            delta_probs = model.predict_proba(snapshot_df)
            forecast_result = compute_forecast_implied_temp(
                delta_probs=delta_probs[0],
                base_temp=base_temp,
            )
        except Exception:
            continue

        # Get market-implied temp from cache (NO DB QUERY!)
        bracket_candles = get_candles_from_cache(candle_cache, day, snapshot_time)
        if not bracket_candles:
            # DEBUG: This is likely where it's failing
            # logger.warning(f"No bracket candles for {day} at {snapshot_time}")
            continue

        market_result = compute_market_implied_temp(
            bracket_candles=bracket_candles,
            snapshot_time=snapshot_time,
        )
        if not market_result.valid:
            # DEBUG: Market implied temp calculation failed
            # logger.warning(f"Invalid market result for {day} at {snapshot_time}")
            continue

        # Detect edge
        edge_result = detect_edge(
            forecast_implied=forecast_result.implied_temp,
            market_implied=market_result.implied_temp,
            forecast_uncertainty=forecast_result.uncertainty,
            market_uncertainty=market_result.uncertainty,
            threshold=edge_threshold,
        )

        # Compute P&L
        pnl = None
        if edge_result.signal != EdgeSignal.NO_TRADE:
            if edge_result.signal == EdgeSignal.BUY_HIGH:
                pnl = 1.0 if settlement > market_result.implied_temp else -1.0
            else:
                pnl = 1.0 if settlement < market_result.implied_temp else -1.0

        # Extract features from original data
        row_data = {
            "day": day,
            "snapshot_time": snapshot_time,
            "forecast_temp": forecast_result.implied_temp,
            "market_temp": market_result.implied_temp,
            "edge": edge_result.edge,
            "signal": edge_result.signal.value,
            "confidence": edge_result.confidence,
            "forecast_uncertainty": forecast_result.uncertainty,
            "market_uncertainty": market_result.uncertainty,
            "base_temp": base_temp,
            "predicted_delta": forecast_result.predicted_delta,
            "settlement_temp": settlement,
            "pnl": pnl,
        }

        # Add context features from original data
        for col in [
            "snapshot_hour",
            "hours_to_event_close",
            "minutes_since_market_open",
            "obs_fcst_max_gap",
            "fcst_remaining_potential",
            "temp_volatility_30min",
            "market_bid_ask_spread",
        ]:
            if col in snapshot_df.columns:
                row_data[col] = snapshot_df[col].iloc[0]

        results.append(row_data)

    return results


def generate_edge_data(
    city: str,
    df: pd.DataFrame,
    model_path: Path,
    edge_threshold: float = 1.5,
    sample_rate: int = 12,
    max_workers: int = 14,
) -> pd.DataFrame:
    """Generate edge detection data using parallel processing.

    Uses ThreadPoolExecutor with shared memory for model and candle cache.
    All DB queries are done upfront in batch (2 queries total).

    Args:
        city: City name
        df: Combined feature DataFrame
        model_path: Path to ordinal model
        edge_threshold: Threshold for edge detection
        sample_rate: Sample every Nth snapshot per day (1 = all)
        max_workers: Number of parallel workers

    Returns:
        DataFrame with edge features and outcomes
    """
    logger.info(f"Generating edge data for {city} with {max_workers} workers...")

    unique_days = sorted(df["day"].unique())
    logger.info(f"Processing {len(unique_days)} unique days")

    # Load model ONCE (shared by all threads)
    logger.info(f"Loading model from {model_path}...")
    model = OrdinalDeltaTrainer()
    model.load(model_path)

    # Batch load all settlements (1 query)
    logger.info("Batch loading settlements...")
    settlements = load_all_settlements(city, unique_days)
    logger.info(f"Loaded {len(settlements)} settlements")

    # Filter to days with settlements
    days_with_settlement = [d for d in unique_days if d in settlements]
    logger.info(f"Days with settlement data: {len(days_with_settlement)}")

    # Batch load ALL candles (1 query - the big optimization!)
    logger.info("Batch loading ALL candles (this may take a moment)...")
    candle_cache = load_all_candles_batch(city, days_with_settlement)
    logger.info(f"Candle cache built: {len(candle_cache)} (day, bracket) entries")

    if not candle_cache:
        logger.error("Candle cache is EMPTY! No candles loaded.")
        return pd.DataFrame()

    # Show sample cache keys
    sample_keys = list(candle_cache.keys())[:5]
    logger.info(f"Sample cache keys: {sample_keys}")

    # Prepare day data
    day_data = []
    for day in days_with_settlement:
        day_df = df[df["day"] == day].copy()
        if day_df.empty:
            continue
        day_data.append((day, day_df, settlements[day]))

    if not day_data:
        logger.warning("No days to process")
        return pd.DataFrame()

    logger.info(f"Processing {len(day_data)} days with {max_workers} threads...")

    # Process in parallel with ThreadPoolExecutor (shared memory!)
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_day,
                day,
                day_df,
                model,  # Shared model
                candle_cache,  # Shared candle cache
                settlement,
                edge_threshold,
                sample_rate,
            ): day
            for day, day_df, settlement in day_data
        }

        with tqdm(total=len(futures), desc="Processing days") as pbar:
            for future in as_completed(futures):
                day = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error processing day {day}: {e}")
                pbar.update(1)

    if not all_results:
        logger.warning("No edge data generated")
        return pd.DataFrame()

    df_results = pd.DataFrame(all_results)
    logger.info(f"Generated {len(df_results):,} edge samples")

    # Filter to signals only (exclude no_trade)
    df_signals = df_results[df_results["signal"] != "no_trade"]
    logger.info(f"Signals with outcomes: {len(df_signals[df_signals['pnl'].notna()]):,}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="Train ML Edge Classifier")
    parser.add_argument(
        "--city",
        type=str,
        default="chicago",
        choices=list(CITY_CONFIG.keys()),
        help="City to train on",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for Optuna",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Edge threshold in degrees F",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=6,
        help="Sample every Nth snapshot per day (1 = all)",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for trade decision",
    )
    parser.add_argument(
        "--optuna-metric",
        type=str,
        default="filtered_precision",
        choices=["auc", "filtered_precision", "f1", "mean_pnl", "sharpe"],
        help="Optuna objective metric (default: filtered_precision)",
    )
    parser.add_argument(
        "--min-trades-for-metric",
        type=int,
        default=10,
        help="Minimum trades when optimizing precision/F1",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of CV splits for DayGroupedTimeSeriesSplit (default: 5)",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        help="Disable validation-based tuning of decision threshold",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate edge data even if cached",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Limit to first N days (for testing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ML EDGE CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"City: {args.city}")
    print(f"Optuna trials: {args.trials}")
    print(f"Workers: {args.workers}")
    print(f"Edge threshold: {args.threshold}°F")
    print(f"Sample rate: every {args.sample_rate}th snapshot")
    if args.max_days:
        print(f"Max days: {args.max_days}")
    print()

    # Check for cached edge data (skip cache if max_days is set - test mode)
    cache_path = Path(f"models/saved/{args.city}/edge_training_data.parquet")
    use_cache = cache_path.exists() and not args.regenerate and not args.max_days

    if use_cache:
        logger.info(f"Loading cached edge data from {cache_path}")
        df_edge = pd.read_parquet(cache_path)
    else:
        # Load ordinal model path (model loaded in workers)
        model_path = Path(f"models/saved/{args.city}/ordinal_catboost_optuna.pkl")
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return 1

        logger.info(f"Using ordinal model: {model_path}")

        # Load combined data
        df_combined = load_combined_data(args.city)

        # Apply max-days limit if set (use LAST N days for candle overlap)
        if args.max_days:
            unique_days = sorted(df_combined["day"].unique())
            days_to_keep = unique_days[-args.max_days:]  # Last N days (most recent)
            df_combined = df_combined[df_combined["day"].isin(days_to_keep)]
            logger.info(f"Limited to last {len(days_to_keep)} days ({len(df_combined):,} rows)")

        # Generate edge data with parallel processing
        df_edge = generate_edge_data(
            city=args.city,
            df=df_combined,
            model_path=model_path,
            edge_threshold=args.threshold,
            sample_rate=args.sample_rate,
            max_workers=args.workers,
        )

        if df_edge.empty:
            logger.error("No edge data generated")
            return 1

        # Cache for future runs (only if not in test mode)
        if not args.max_days:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_edge.to_parquet(cache_path, index=False)
            logger.info(f"Cached edge data to {cache_path}")

    # Filter to signals only (exclude no_trade for training)
    df_signals = df_edge[df_edge["signal"] != "no_trade"].copy()
    logger.info(f"Training on {len(df_signals):,} edge signals")

    # Show class balance
    n_wins = (df_signals["pnl"] > 0).sum()
    n_total = df_signals["pnl"].notna().sum()
    print(f"\nClass balance: {n_wins}/{n_total} wins ({n_wins/n_total:.1%})")
    print()

    # Train EdgeClassifier
    print("=" * 60)
    print(f"OPTUNA TRAINING ({args.trials} trials)")
    print("=" * 60)

    classifier = EdgeClassifier(
        n_trials=args.trials,
        n_jobs=args.workers,
        decision_threshold=args.decision_threshold,
        optimize_metric=args.optuna_metric,
        min_trades_for_metric=args.min_trades_for_metric,
    )

    metrics = classifier.train(
        df_signals,
        target_col="pnl",
        shuffle=False,  # CRITICAL: Must be False to prevent leakage!
        tune_threshold=not args.no_threshold_tuning,
        cv_splits=args.cv_splits,
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Test AUC: {metrics['test_auc']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.1%}")
    print()
    print(f"Baseline win rate: {metrics['baseline_win_rate']:.1%}")
    print(f"Filtered win rate: {metrics['filtered_win_rate']:.1%}")
    print(f"Improvement: +{(metrics['filtered_win_rate'] - metrics['baseline_win_rate'])*100:.1f}pp")
    print()
    print(f"Trades recommended: {metrics['n_trades_recommended']}/{metrics['n_test_total']} ({metrics['n_trades_recommended']/metrics['n_test_total']:.1%})")
    print()

    # Feature importance
    print("Feature Importance:")
    importance = metrics["feature_importance"]
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
    print()

    # Save model
    save_path = Path(f"models/saved/{args.city}/edge_classifier")
    classifier.save(save_path, city=args.city)
    print(f"Model saved to: {save_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
