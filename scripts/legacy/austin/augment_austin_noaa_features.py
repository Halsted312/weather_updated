#!/usr/bin/env python3
"""Augment Austin training parquet with NOAA + candle features.

Loads the existing full.parquet, computes NOAA and candle features,
joins them back, and saves as full_aug.parquet.

NOAA features: Computed once per day (same for all cutoff_times)
Candle features: Computed per (day, cutoff_time) pair

Usage:
    PYTHONPATH=. .venv/bin/python scripts/augment_austin_noaa_features.py
    PYTHONPATH=. .venv/bin/python scripts/augment_austin_noaa_features.py --dry-run
    PYTHONPATH=. .venv/bin/python scripts/augment_austin_noaa_features.py --noaa-only
"""

import argparse
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# City config
CITY_ID = "austin"
INPUT_PATH = Path("data/training_cache/austin/full.parquet")
OUTPUT_PATH = Path("data/training_cache/austin/full_aug.parquet")

# NOAA feature columns
NOAA_COLS = [
    "nbm_peak_window_max_f", "nbm_peak_window_revision_1h_f",
    "hrrr_peak_window_max_f", "hrrr_peak_window_revision_1h_f",
    "ndfd_tmax_T1_f", "ndfd_drift_T2_to_T1_f",
    "hrrr_minus_nbm_peak_window_max_f", "ndfd_minus_vc_T1_f",
    "nbm_t15_z_30d_f", "hrrr_t15_z_30d_f", "hrrr_minus_nbm_t15_z_30d_f",
]

# Candle feature columns
CANDLE_COLS = [
    "c_logit_mid_last", "c_logit_mom_15m", "c_logit_vol_15m", "c_logit_surprise_15m",
    "c_spread_pct_mean_15m", "c_mid_range_pct_15m", "c_trade_frac_15m", "c_synth_frac_15m",
]


def load_noaa_features_for_day(session, city_id: str, target_date: date) -> Dict[str, Any]:
    """Load NOAA guidance and compute features for a single day."""
    from models.data.loader import load_weather_more_apis_guidance, load_obs_t15_stats_30d
    from models.features.more_apis import compute_more_apis_features

    more_apis = load_weather_more_apis_guidance(session, city_id, target_date)
    obs_t15_mean, obs_t15_std = load_obs_t15_stats_30d(session, city_id, target_date)

    feature_set = compute_more_apis_features(
        more_apis=more_apis,
        vc_t1_tempmax=None,
        obs_t15_mean_30d_f=obs_t15_mean,
        obs_t15_std_30d_f=obs_t15_std,
    )
    return feature_set.to_dict()


def load_candles_for_day(session, city_id: str, event_date: date) -> Optional[pd.DataFrame]:
    """Load all 1-min candles for a single day from candles_1m_dense."""
    from sqlalchemy import text
    from models.data.dataset import get_market_clock_window

    # Get market clock window for this day
    window_start, window_end = get_market_clock_window(event_date)

    # Map city to ticker pattern
    city_ticker_map = {
        "chicago": "CHI", "austin": "AUS", "denver": "DEN",
        "los_angeles": "LAX", "miami": "MIA", "philadelphia": "PHL",
    }
    ticker_pattern = city_ticker_map.get(city_id.lower(), city_id.upper()[:3])

    query = text("""
        SELECT bucket_start, ticker, yes_bid_close, yes_ask_close, volume, open_interest
        FROM kalshi.candles_1m_dense
        WHERE ticker LIKE :ticker_pattern
          AND bucket_start >= :window_start
          AND bucket_start <= :window_end
        ORDER BY bucket_start
    """)

    try:
        result = session.execute(query, {
            "ticker_pattern": f"%{ticker_pattern}%",
            "window_start": window_start,
            "window_end": window_end,
        })
        df = pd.DataFrame(result.fetchall(), columns=[
            "bucket_start", "ticker", "yes_bid_close", "yes_ask_close", "volume", "open_interest"
        ])
        if df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"Could not load candles for {city_id}/{event_date}: {e}")
        return None


def compute_candle_features_for_snapshot(
    candles_df: Optional[pd.DataFrame],
    cutoff_time: pd.Timestamp,
) -> Dict[str, Any]:
    """Compute candle micro features for a single snapshot."""
    from models.features.candles_micro import compute_candles_micro_features

    fs = compute_candles_micro_features(
        candles_df=candles_df,
        snapshot_time=cutoff_time,
        window_minutes=15,
    )
    return fs.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Augment Austin parquet with NOAA + candle features")
    parser.add_argument("--dry-run", action="store_true", help="Load and report, don't save")
    parser.add_argument("--noaa-only", action="store_true", help="Only add NOAA features, skip candles")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH), help="Input parquet path")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output parquet path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    logger.info(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Determine date column
    date_col = 'event_date' if 'event_date' in df.columns else 'day'
    df[date_col] = pd.to_datetime(df[date_col]).dt.date

    # Get unique days
    unique_days = sorted(df[date_col].unique())
    logger.info(f"Found {len(unique_days)} unique days: {unique_days[0]} to {unique_days[-1]}")

    from src.db import get_db_session

    # =========================================================================
    # Phase 1: NOAA features (per day)
    # =========================================================================
    logger.info("\n=== Phase 1: NOAA Features ===")
    day_noaa_features = {}
    with get_db_session() as session:
        for day in tqdm(unique_days, desc="Loading NOAA features"):
            features = load_noaa_features_for_day(session, CITY_ID, day)
            day_noaa_features[day] = features

    noaa_df = pd.DataFrame.from_dict(day_noaa_features, orient='index')
    noaa_df.index.name = date_col
    noaa_df = noaa_df.reset_index()

    logger.info(f"NOAA feature null rates (per day):")
    for col in NOAA_COLS:
        if col in noaa_df.columns:
            non_null = noaa_df[col].notna().sum()
            pct = 100 * non_null / len(noaa_df)
            logger.info(f"  {col}: {non_null}/{len(noaa_df)} days non-null ({pct:.1f}%)")

    # =========================================================================
    # Phase 2: Candle features (per day+cutoff_time) - OPTIONAL
    # =========================================================================
    candle_df = None
    if not args.noaa_only:
        logger.info("\n=== Phase 2: Candle Features ===")

        # Get unique (day, cutoff_time) pairs
        unique_snapshots = df[[date_col, 'cutoff_time']].drop_duplicates()
        logger.info(f"Found {len(unique_snapshots):,} unique (day, cutoff_time) pairs")

        # Group by day for efficient candle loading
        candle_rows = []
        with get_db_session() as session:
            for day in tqdm(unique_days, desc="Loading candle features"):
                # Load candles once per day
                candles_df = load_candles_for_day(session, CITY_ID, day)

                # Get all cutoff_times for this day
                day_snapshots = unique_snapshots[unique_snapshots[date_col] == day]

                for _, row in day_snapshots.iterrows():
                    cutoff_time = pd.Timestamp(row['cutoff_time'])
                    features = compute_candle_features_for_snapshot(candles_df, cutoff_time)
                    features[date_col] = day
                    features['cutoff_time'] = row['cutoff_time']
                    candle_rows.append(features)

        candle_df = pd.DataFrame(candle_rows)

        logger.info(f"Candle feature null rates (per snapshot):")
        for col in CANDLE_COLS:
            if col in candle_df.columns:
                non_null = candle_df[col].notna().sum()
                pct = 100 * non_null / len(candle_df)
                logger.info(f"  {col}: {non_null:,}/{len(candle_df):,} non-null ({pct:.1f}%)")

    # =========================================================================
    # Phase 3: Join to main DataFrame
    # =========================================================================
    logger.info("\n=== Phase 3: Joining Features ===")

    # Drop existing NOAA columns if present
    existing_noaa = [c for c in df.columns if c in NOAA_COLS]
    if existing_noaa:
        logger.info(f"Dropping existing NOAA columns: {existing_noaa}")
        df = df.drop(columns=existing_noaa)

    # Drop existing candle columns if present
    existing_candle = [c for c in df.columns if c in CANDLE_COLS]
    if existing_candle:
        logger.info(f"Dropping existing candle columns: {existing_candle}")
        df = df.drop(columns=existing_candle)

    # Join NOAA features (by day only)
    df_aug = df.merge(noaa_df, on=date_col, how='left')
    logger.info(f"After NOAA join: {len(df_aug):,} rows, {len(df_aug.columns)} columns")

    # Join candle features (by day + cutoff_time)
    if candle_df is not None:
        df_aug = df_aug.merge(candle_df, on=[date_col, 'cutoff_time'], how='left')
        logger.info(f"After candle join: {len(df_aug):,} rows, {len(df_aug.columns)} columns")

    # =========================================================================
    # Phase 4: Verify and Save
    # =========================================================================
    logger.info("\n=== Phase 4: Verification ===")
    logger.info(f"Final dataset: {len(df_aug):,} rows, {len(df_aug.columns)} columns")

    # Verify row count unchanged
    if len(df_aug) != len(df):
        logger.warning(f"ROW COUNT CHANGED! Original: {len(df)}, Augmented: {len(df_aug)}")

    # Print final null rates
    logger.info("\nFinal NOAA null rates:")
    for col in NOAA_COLS:
        if col in df_aug.columns:
            non_null = df_aug[col].notna().sum()
            pct = 100 * non_null / len(df_aug)
            logger.info(f"  {col}: {non_null:,}/{len(df_aug):,} ({pct:.1f}%)")

    if candle_df is not None:
        logger.info("\nFinal candle null rates:")
        for col in CANDLE_COLS:
            if col in df_aug.columns:
                non_null = df_aug[col].notna().sum()
                pct = 100 * non_null / len(df_aug)
                logger.info(f"  {col}: {non_null:,}/{len(df_aug):,} ({pct:.1f}%)")

    if args.dry_run:
        logger.info("\n[DRY RUN] Skipping save")
        return 0

    # Save
    logger.info(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_parquet(output_path, index=False)
    logger.info(f"Saved {len(df_aug):,} rows to {output_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows:   {len(df_aug):,}")
    print(f"Cols:   {len(df_aug.columns)} (was {len(df.columns) + len(existing_noaa) + len(existing_candle)})")
    print(f"Days:   {len(unique_days)}")
    print("\nTo use for training:")
    print(f"  mv {input_path} {input_path.parent / 'full_original.parquet'}")
    print(f"  mv {output_path} {input_path}")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
