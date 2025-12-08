#!/usr/bin/env python3
"""Build training dataset from raw parquet files (no DB required).

This script runs on the fast machine after raw data has been extracted
from the slow machine via extract_raw_data_to_parquet.py.

It loads raw parquets and runs the full feature engineering pipeline
to produce train_data_full.parquet and test_data_full.parquet.

Usage:
    PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin
    PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city austin --workers 12

Required input files:
    models/raw_data/{city}/
        vc_observations.parquet
        vc_city_observations.parquet
        settlements.parquet
        forecasts_daily.parquet
        forecasts_hourly.parquet
        noaa_guidance.parquet
    models/candles/candles_{city}.parquet

Output:
    models/saved/{city}/
        train_data_full.parquet
        test_data_full.parquet
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_HOLDOUT_PCT = 0.20
DEFAULT_WORKERS = 8


@dataclass
class RawData:
    """Container for raw data loaded from parquets."""
    observations: pd.DataFrame
    city_observations: pd.DataFrame
    settlements: pd.DataFrame
    forecasts_daily: pd.DataFrame
    forecasts_hourly: pd.DataFrame
    noaa_guidance: pd.DataFrame
    candles: pd.DataFrame


def load_raw_data(city: str, raw_data_dir: Path, candles_dir: Path) -> tuple:
    """Load all raw data parquets for a city.

    Pre-processes and pre-groups data by date for O(1) lookups per day.
    Returns (RawData, grouped_data_dict).
    """
    city_dir = raw_data_dir / city

    logger.info(f"Loading raw data from {city_dir}...")

    obs = pd.read_parquet(city_dir / "vc_observations.parquet")
    obs['datetime_local'] = pd.to_datetime(obs['datetime_local'])
    obs['obs_date'] = obs['datetime_local'].dt.date
    logger.info(f"  Observations: {len(obs):,} rows")

    city_obs = pd.read_parquet(city_dir / "vc_city_observations.parquet")
    city_obs['datetime_local'] = pd.to_datetime(city_obs['datetime_local'])
    city_obs['obs_date'] = city_obs['datetime_local'].dt.date
    logger.info(f"  City observations: {len(city_obs):,} rows")

    settlements = pd.read_parquet(city_dir / "settlements.parquet")
    logger.info(f"  Settlements: {len(settlements):,} rows")

    fcst_daily = pd.read_parquet(city_dir / "forecasts_daily.parquet")
    logger.info(f"  Daily forecasts: {len(fcst_daily):,} rows")

    fcst_hourly = pd.read_parquet(city_dir / "forecasts_hourly.parquet")
    logger.info(f"  Hourly forecasts: {len(fcst_hourly):,} rows")

    noaa = pd.read_parquet(city_dir / "noaa_guidance.parquet")
    logger.info(f"  NOAA guidance: {len(noaa):,} rows")

    candles_path = candles_dir / f"candles_{city}.parquet"
    candles = pd.read_parquet(candles_path)
    if 'bucket_start' in candles.columns:
        candles['bucket_start'] = pd.to_datetime(candles['bucket_start'])
        candles['bucket_date'] = candles['bucket_start'].dt.date
    logger.info(f"  Candles: {len(candles):,} rows")

    # PRE-GROUP by date for O(1) lookups (CRITICAL for speed!)
    logger.info("  Pre-grouping data by date...")
    obs_by_date = {d: g for d, g in obs.groupby('obs_date')}
    city_obs_by_date = {d: g for d, g in city_obs.groupby('obs_date')}
    candles_by_date = {d: g for d, g in candles.groupby('bucket_date')} if 'bucket_date' in candles.columns else {}
    logger.info(f"  Pre-grouped: {len(obs_by_date)} obs days, {len(candles_by_date)} candle days")

    raw_data = RawData(
        observations=obs,
        city_observations=city_obs,
        settlements=settlements,
        forecasts_daily=fcst_daily,
        forecasts_hourly=fcst_hourly,
        noaa_guidance=noaa,
        candles=candles,
    )

    grouped = {
        'obs_by_date': obs_by_date,
        'city_obs_by_date': city_obs_by_date,
        'candles_by_date': candles_by_date,
    }

    return raw_data, grouped


def get_market_clock_window(event_date: date) -> tuple[datetime, datetime]:
    """Get market clock observation window for an event date."""
    # D-1 10:00 to D 23:55 (local time)
    d_minus_1 = event_date - timedelta(days=1)
    window_start = datetime.combine(d_minus_1, datetime.min.time()).replace(hour=10, minute=0)
    window_end = datetime.combine(event_date, datetime.min.time()).replace(hour=23, minute=55)
    return window_start, window_end


def get_snapshot_times(window_start: datetime, window_end: datetime, interval_min: int = 5) -> List[datetime]:
    """Generate snapshot times within window."""
    times = []
    current = window_start
    while current <= window_end:
        times.append(current)
        current += timedelta(minutes=interval_min)
    return times


def get_fcst_daily_for_day(fcst_daily: pd.DataFrame, target_date: date, lead_days: int = 1) -> Optional[dict]:
    """Get daily forecast for a specific target date and lead time."""
    mask = (fcst_daily['target_date'] == target_date) & (fcst_daily['lead_days'] == lead_days)
    rows = fcst_daily[mask]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "tempmax_f": row.get("tempmax_f"),
        "tempmin_f": row.get("tempmin_f"),
        "temp_f": row.get("temp_f"),
        "humidity": row.get("humidity"),
        "precip_in": row.get("precip_in"),
        "precipprob": row.get("precipprob"),
        "windspeed_mph": row.get("windspeed_mph"),
        "windgust_mph": row.get("windgust_mph"),
        "cloudcover": row.get("cloudcover"),
        "conditions": row.get("conditions"),
    }


def get_fcst_multi_for_day(fcst_daily: pd.DataFrame, target_date: date) -> dict[int, Optional[dict]]:
    """Get multi-horizon forecasts for a target date."""
    result = {}
    for lead in range(1, 7):
        result[lead] = get_fcst_daily_for_day(fcst_daily, target_date, lead)
    return result


def get_fcst_hourly_for_day(fcst_hourly: pd.DataFrame, target_date: date, basis_date: date) -> Optional[pd.DataFrame]:
    """Get hourly forecast for a specific target date."""
    if fcst_hourly.empty:
        return None

    # Convert to date if needed
    fcst_hourly = fcst_hourly.copy()
    fcst_hourly['target_date'] = pd.to_datetime(fcst_hourly['target_datetime_local']).dt.date

    mask = (fcst_hourly['target_date'] == target_date) & (fcst_hourly['forecast_basis_date'] == basis_date)
    rows = fcst_hourly[mask]
    if rows.empty:
        return None
    return rows


def get_noaa_for_day(noaa: pd.DataFrame, target_date: date) -> dict:
    """Get NOAA guidance for a target date."""
    result = {
        "nbm": {"latest_run": None, "prev_run": None},
        "hrrr": {"latest_run": None, "prev_run": None},
        "ndfd": {"latest_run": None, "prev_run": None},
    }

    if noaa.empty:
        return result

    for model in ["nbm", "hrrr", "ndfd"]:
        mask = (noaa['target_date'] == target_date) & (noaa['model'] == model)
        runs = noaa[mask].sort_values('run_datetime_utc', ascending=False)

        if len(runs) >= 1:
            r = runs.iloc[0]
            result[model]["latest_run"] = {
                "run_datetime_utc": r["run_datetime_utc"],
                "peak_window_max_f": float(r["peak_window_max_f"]) if pd.notna(r["peak_window_max_f"]) else None,
            }
        if len(runs) >= 2:
            r = runs.iloc[1]
            result[model]["prev_run"] = {
                "run_datetime_utc": r["run_datetime_utc"],
                "peak_window_max_f": float(r["peak_window_max_f"]) if pd.notna(r["peak_window_max_f"]) else None,
            }

    return result


def compute_obs_t15_stats(obs: pd.DataFrame, target_date: date, lookback_days: int = 30) -> tuple[Optional[float], Optional[float]]:
    """Compute rolling mean/std of observed temp at 15:00 local over previous N days."""
    start_date = target_date - timedelta(days=lookback_days)
    end_date = target_date - timedelta(days=1)

    obs = obs.copy()
    obs['obs_date'] = pd.to_datetime(obs['datetime_local']).dt.date

    mask = (obs['obs_date'] >= start_date) & (obs['obs_date'] <= end_date)
    obs_window = obs[mask]

    if obs_window.empty:
        return None, None

    obs_window = obs_window.copy()
    obs_window['hour'] = pd.to_datetime(obs_window['datetime_local']).dt.hour
    obs_window['minute'] = pd.to_datetime(obs_window['datetime_local']).dt.minute

    t15_obs = obs_window[(obs_window['hour'] == 15) & (obs_window['minute'] <= 15)]

    if len(t15_obs) < 10:
        return None, None

    daily_t15 = t15_obs.groupby('obs_date')['temp_f'].first()

    if len(daily_t15) < 10:
        return None, None

    return float(daily_t15.mean()), float(daily_t15.std(ddof=0))


def precompute_all_obs_t15_stats(obs: pd.DataFrame, all_days: List[date], lookback_days: int = 30) -> Dict[date, tuple[Optional[float], Optional[float]]]:
    """Pre-compute obs_t15 stats for ALL days in a single efficient pass.

    Instead of scanning 307k rows per day, we:
    1. Extract daily T15 temps once (O(n))
    2. Compute rolling mean/std using pandas rolling (O(days))

    This reduces O(n * days) to O(n + days).
    """
    logger.info("Pre-computing obs_t15 stats for all days...")

    # Check if dates already pre-parsed (from load_raw_data), else parse them
    if 'obs_date' not in obs.columns:
        obs = obs.copy()
        obs['datetime_local'] = pd.to_datetime(obs['datetime_local'])
        obs['obs_date'] = obs['datetime_local'].dt.date

    # Extract hour/minute for filtering (use existing datetime_local)
    hour = obs['datetime_local'].dt.hour
    minute = obs['datetime_local'].dt.minute

    # Filter to T15 observations (15:00-15:15)
    t15_obs = obs[(hour == 15) & (minute <= 15)]

    # Get first T15 temp per day
    daily_t15 = t15_obs.groupby('obs_date')['temp_f'].first()

    # Convert to DataFrame for rolling calculations
    daily_df = daily_t15.reset_index()
    daily_df.columns = ['obs_date', 'temp_f']
    daily_df = daily_df.sort_values('obs_date')
    daily_df = daily_df.set_index('obs_date')

    # Compute rolling stats (lookback_days window, shifted by 1 to exclude current day)
    # shift(1) ensures we use days BEFORE target_date
    rolling = daily_df['temp_f'].shift(1).rolling(window=lookback_days, min_periods=10)
    rolling_mean = rolling.mean()
    rolling_std = rolling.std(ddof=0)

    # Build result dict
    result = {}
    for day in all_days:
        if day in rolling_mean.index:
            mean_val = rolling_mean.loc[day]
            std_val = rolling_std.loc[day]
            if pd.notna(mean_val) and pd.notna(std_val):
                result[day] = (float(mean_val), float(std_val))
            else:
                result[day] = (None, None)
        else:
            result[day] = (None, None)

    non_null = sum(1 for v in result.values() if v[0] is not None)
    logger.info(f"  Pre-computed obs_t15 stats: {non_null}/{len(all_days)} days have valid stats")

    return result


def build_day_snapshots(
    city: str,
    event_date: date,
    raw_data: RawData,
    grouped: Dict,
    snapshot_interval_min: int = 5,
    obs_t15_stats: Optional[Dict[date, tuple[Optional[float], Optional[float]]]] = None,
) -> List[dict]:
    """Build all snapshots for a single day.

    Args:
        city: City code
        event_date: Target date
        raw_data: RawData container with all parquet data
        grouped: Pre-grouped data dict for O(1) lookups
        snapshot_interval_min: Minutes between snapshots
        obs_t15_stats: Pre-computed obs_t15 stats dict (optional, computed if not provided)
    """
    from models.data.snapshot import build_snapshot

    # Get settlement
    settle_row = raw_data.settlements[raw_data.settlements['date_local'] == event_date]
    if settle_row.empty:
        return []
    settle_f = int(settle_row.iloc[0]['tmax_final'])

    # Get window
    window_start, window_end = get_market_clock_window(event_date)
    snapshot_times = get_snapshot_times(window_start, window_end, snapshot_interval_min)

    # Get observations via O(1) dict lookup (NOT filtering!)
    d_minus_1 = event_date - timedelta(days=1)
    obs_today = grouped['obs_by_date'].get(event_date, pd.DataFrame())
    obs_yesterday = grouped['obs_by_date'].get(d_minus_1, pd.DataFrame())
    day_obs = pd.concat([obs_today, obs_yesterday], ignore_index=True) if len(obs_today) or len(obs_yesterday) else pd.DataFrame()

    city_obs_today = grouped['city_obs_by_date'].get(event_date, pd.DataFrame())
    city_obs_yesterday = grouped['city_obs_by_date'].get(d_minus_1, pd.DataFrame())
    day_city_obs = pd.concat([city_obs_today, city_obs_yesterday], ignore_index=True) if len(city_obs_today) or len(city_obs_yesterday) else pd.DataFrame()

    # Get candles via O(1) dict lookup
    candles_today = grouped['candles_by_date'].get(event_date, pd.DataFrame())
    candles_yesterday = grouped['candles_by_date'].get(d_minus_1, pd.DataFrame())
    day_candles = pd.concat([candles_today, candles_yesterday], ignore_index=True) if len(candles_today) or len(candles_yesterday) else pd.DataFrame()

    # Get forecasts
    basis_date = event_date - timedelta(days=1)
    fcst_daily = get_fcst_daily_for_day(raw_data.forecasts_daily, event_date, lead_days=1)
    fcst_hourly = get_fcst_hourly_for_day(raw_data.forecasts_hourly, event_date, basis_date)
    fcst_multi = get_fcst_multi_for_day(raw_data.forecasts_daily, event_date)

    # Get NOAA guidance
    more_apis = get_noaa_for_day(raw_data.noaa_guidance, event_date)

    # Get obs_t15 stats (use pre-computed if available)
    if obs_t15_stats is not None and event_date in obs_t15_stats:
        obs_t15_mean, obs_t15_std = obs_t15_stats[event_date]
    else:
        obs_t15_mean, obs_t15_std = compute_obs_t15_stats(raw_data.observations, event_date)

    # Build snapshots
    rows = []
    for cutoff_time in snapshot_times:
        try:
            features = build_snapshot(
                city=city,
                event_date=event_date,
                cutoff_time=cutoff_time,
                obs_df=day_obs,
                window_start=window_start,
                fcst_daily=fcst_daily,
                fcst_hourly_df=fcst_hourly,
                fcst_multi=fcst_multi,
                candles_df=day_candles if not day_candles.empty else None,
                city_obs_df=day_city_obs if not day_city_obs.empty else None,
                more_apis=more_apis,
                obs_t15_mean_30d_f=obs_t15_mean,
                obs_t15_std_30d_f=obs_t15_std,
                settle_f=settle_f,
                include_labels=True,
            )

            # Skip if insufficient observations
            n_obs = features.get("num_samples_sofar", 0) or 0
            if n_obs < 1:
                continue

            features["day"] = event_date
            rows.append(features)

        except Exception as e:
            logger.warning(f"Error building snapshot for {city}/{event_date}/{cutoff_time}: {e}")
            continue

    return rows


def process_day_chunk(
    city: str,
    days: List[date],
    raw_data_dir: Path,
    candles_dir: Path,
    snapshot_interval_min: int = 5,
    obs_t15_stats: Optional[Dict[date, tuple[Optional[float], Optional[float]]]] = None,
) -> List[dict]:
    """Process a chunk of days (for parallel execution).

    Args:
        city: City code
        days: List of dates to process
        raw_data_dir: Path to raw data directory
        candles_dir: Path to candles directory
        snapshot_interval_min: Minutes between snapshots
        obs_t15_stats: Pre-computed obs_t15 stats dict (computed per-chunk if not provided)
    """
    # Load raw data (each worker loads its own copy)
    raw_data = load_raw_data(city, raw_data_dir, candles_dir)

    # If no pre-computed stats, compute them for this chunk's days
    # (still more efficient than per-day computation)
    if obs_t15_stats is None:
        obs_t15_stats = precompute_all_obs_t15_stats(raw_data.observations, days)

    all_rows = []
    for event_date in days:
        rows = build_day_snapshots(city, event_date, raw_data, snapshot_interval_min, obs_t15_stats)
        all_rows.extend(rows)

    return all_rows


# Worker function for ProcessPoolExecutor (must be at module level for pickling)
def _process_chunk_worker(args: tuple) -> List[dict]:
    """Worker wrapper for process_day_chunk (unpacks args for ProcessPoolExecutor)."""
    city, days, raw_data_dir, candles_dir, snapshot_interval_min, obs_t15_stats = args
    return process_day_chunk(city, days, raw_data_dir, candles_dir, snapshot_interval_min, obs_t15_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Build training dataset from raw parquet files"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"],
        help="City to build dataset for",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="models/raw_data",
        help="Directory containing raw parquets",
    )
    parser.add_argument(
        "--candles-dir",
        type=str,
        default="models/candles",
        help="Directory containing candle parquets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/saved",
        help="Output directory for train/test parquets",
    )
    parser.add_argument(
        "--holdout-pct",
        type=float,
        default=DEFAULT_HOLDOUT_PCT,
        help="Holdout percentage for test set (default 0.20)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=5,
        help="Snapshot interval in minutes (default: 5)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD) - filter to this date range",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD) - filter to this date range",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail immediately if required parquets are missing (default: True)",
    )
    args = parser.parse_args()

    city = args.city
    raw_data_dir = Path(args.raw_data_dir)
    candles_dir = Path(args.candles_dir)
    output_dir = Path(args.output_dir) / city

    logger.info("=" * 60)
    logger.info(f"BUILD DATASET FROM PARQUETS: {city.upper()}")
    logger.info("=" * 60)

    # Load raw data to get date range (returns tuple now)
    raw_data, grouped = load_raw_data(city, raw_data_dir, candles_dir)

    # Get unique days with settlements
    all_days = sorted(raw_data.settlements['date_local'].unique())
    logger.info(f"\nFound {len(all_days)} days with settlements")
    logger.info(f"Date range: {all_days[0]} to {all_days[-1]}")

    # Apply date filtering if provided (CRITICAL FOR OCT 2025 TESTING)
    if args.start:
        start_date = pd.to_datetime(args.start).date()
        all_days = [d for d in all_days if d >= start_date]
        logger.info(f"Filtered start >= {start_date}")

    if args.end:
        end_date = pd.to_datetime(args.end).date()
        all_days = [d for d in all_days if d <= end_date]
        logger.info(f"Filtered end <= {end_date}")

    if args.start or args.end:
        logger.info(f"After filtering: {len(all_days)} days ({all_days[0]} to {all_days[-1]})")

    # Split into train/test
    n_days = len(all_days)
    holdout_days = max(1, int(n_days * args.holdout_pct))
    train_days = all_days[:-holdout_days]
    test_days = all_days[-holdout_days:]

    logger.info(f"\nTrain days: {len(train_days)} ({train_days[0]} to {train_days[-1]})")
    logger.info(f"Test days: {len(test_days)} ({test_days[0]} to {test_days[-1]})")

    # Pre-compute obs_t15 stats for ALL days (O(n) instead of O(n * days))
    logger.info("\n--- Pre-computing rolling stats ---")
    all_target_days = train_days + test_days
    obs_t15_stats = precompute_all_obs_t15_stats(raw_data.observations, all_target_days)

    # Build datasets - use sequential processing with pre-loaded & pre-grouped data
    # O(1) dict lookups per day instead of filtering millions of rows
    def build_dataset_sequential(days: List[date], desc: str) -> pd.DataFrame:
        """Build dataset for a list of days using pre-loaded raw_data."""
        logger.info(f"Building {desc} ({len(days)} days)...")
        all_rows = []
        for event_date in tqdm(days, desc=desc):
            rows = build_day_snapshots(city, event_date, raw_data, grouped, args.snapshot_interval, obs_t15_stats)
            all_rows.extend(rows)
        return pd.DataFrame(all_rows)

    # Build training dataset
    logger.info("\n--- Building training dataset ---")
    df_train = build_dataset_sequential(train_days, "Training days")
    logger.info(f"Training samples: {len(df_train):,}")

    # Build test dataset
    logger.info("\n--- Building test dataset ---")
    df_test = build_dataset_sequential(test_days, "Test days")
    logger.info(f"Test samples: {len(df_test):,}")

    # Add lag features
    from models.features.calendar import add_lag_features_to_dataframe

    logger.info("\n--- Adding lag features ---")
    if len(df_train) > 0 and "settle_f" in df_train.columns:
        df_train = add_lag_features_to_dataframe(df_train)
    if len(df_test) > 0 and "settle_f" in df_test.columns:
        df_test = add_lag_features_to_dataframe(df_test)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_data_full.parquet"
    test_path = output_dir / "test_data_full.parquet"

    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    logger.info(f"\nSaved train: {train_path}")
    logger.info(f"Saved test: {test_path}")

    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print(f"\nTrain: {len(df_train):,} rows, {len(df_train.columns)} columns")
    print(f"Test: {len(df_test):,} rows, {len(df_test.columns)} columns")

    # Check for key features
    print("\nFeature check:")
    key_features = ["delta", "settle_f", "vc_max_f_sofar", "fcst_prev_max_f",
                    "nbm_peak_window_max_f", "hrrr_peak_window_max_f",
                    "c_logit_mid_last", "market_yes_bid"]
    for col in key_features:
        if col in df_train.columns:
            non_null = df_train[col].notna().sum()
            pct = 100 * non_null / len(df_train)
            print(f"  {col}: {pct:.1f}% non-null")
        else:
            print(f"  {col}: MISSING")

    return 0


if __name__ == "__main__":
    sys.exit(main())
