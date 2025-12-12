#!/usr/bin/env python3
"""Build datasets for all 6 cities in serial (one at a time).

Run with:
    nohup python scripts/build_all_city_datasets.py > logs/build_all_datasets.log 2>&1 &

Check status:
    tail -f logs/build_all_datasets.log
"""

import os
import sys
import time
from datetime import date
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models.data.dataset import DatasetConfig, build_dataset

# All 6 cities (lowercase names as stored in DB)
CITIES = ["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"]

# Date range - full history
START_DATE = date(2023, 1, 1)
END_DATE = date(2024, 11, 30)

# Dataset config - hourly snapshots with forecast features
CONFIG = DatasetConfig(
    time_window="market_clock",
    snapshot_interval_min=60,  # 1-hour snapshots
    include_forecast=True,
    include_market=False,
    include_station_city=False,
)

# New feature prefixes to check
NEW_FEATURE_PREFIXES = [
    "fcst_peak_",
    "fcst_drift_",
    "fcst_humidity_",
    "fcst_cloudcover_",
    "fcst_dewpoint_",
    "fcst_prev_distance",
    "fcst_prev_near",
]


def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main():
    print(f"[{timestamp()}] === Building datasets for all 6 cities ===")
    print(f"[{timestamp()}] Date range: {START_DATE} to {END_DATE}")
    print(f"[{timestamp()}] Snapshot interval: {CONFIG.snapshot_interval_min} min")
    print(f"[{timestamp()}] Cities: {', '.join(CITIES)}")
    print()

    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://kalshi:kalshi@localhost:5434/kalshi_weather"
    )
    engine = create_engine(db_url)

    total_start = time.time()
    results = {}

    for i, city in enumerate(CITIES, 1):
        print(f"[{timestamp()}] === [{i}/6] Starting {city.upper()} ===")
        city_start = time.time()

        try:
            with Session(engine) as session:
                df = build_dataset(
                    cities=[city],
                    start_date=START_DATE,
                    end_date=END_DATE,
                    config=CONFIG,
                    session=session,
                )

            city_elapsed = time.time() - city_start

            if df.empty:
                print(f"[{timestamp()}] WARNING: {city} returned empty dataset!")
                results[city] = {"status": "empty", "time_min": city_elapsed / 60}
                continue

            # Save to parquet
            output_path = f"models/saved/{city}_dataset_with_new_features.parquet"
            df.to_parquet(output_path)

            # Check new features
            new_features = [
                c
                for c in df.columns
                if any(c.startswith(p) for p in NEW_FEATURE_PREFIXES)
            ]

            print(f"[{timestamp()}] {city.upper()} complete:")
            print(f"    Shape: {df.shape}")
            print(f"    Time: {city_elapsed/60:.1f} minutes")
            print(f"    Saved: {output_path}")
            print(f"    New features: {len(new_features)}")

            # Show feature population rates
            for f in sorted(new_features)[:5]:  # First 5 only
                non_null = df[f].notna().sum()
                pct = 100 * non_null / len(df)
                print(f"      {f}: {pct:.1f}% populated")
            if len(new_features) > 5:
                print(f"      ... and {len(new_features)-5} more")

            results[city] = {
                "status": "success",
                "shape": df.shape,
                "time_min": city_elapsed / 60,
                "new_features": len(new_features),
            }

        except Exception as e:
            city_elapsed = time.time() - city_start
            print(f"[{timestamp()}] ERROR on {city}: {e}")
            results[city] = {"status": "error", "error": str(e), "time_min": city_elapsed / 60}

        print()

    # Final summary
    total_elapsed = time.time() - total_start
    print(f"[{timestamp()}] === FINAL SUMMARY ===")
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print()

    for city, info in results.items():
        status = info["status"]
        t = info["time_min"]
        if status == "success":
            print(f"  {city:15} OK     {info['shape']}  {t:.1f} min  {info['new_features']} new features")
        elif status == "empty":
            print(f"  {city:15} EMPTY  {t:.1f} min")
        else:
            print(f"  {city:15} ERROR  {t:.1f} min  {info['error'][:50]}")

    print()
    print(f"[{timestamp()}] === DONE ===")


if __name__ == "__main__":
    main()
