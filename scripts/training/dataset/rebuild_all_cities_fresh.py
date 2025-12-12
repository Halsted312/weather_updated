#!/usr/bin/env python3
"""
Rebuild all 6 cities with fresh datasets including new features.

Saves each city's dataset as parquet in models/saved/{city}/full.parquet

Usage:
    nohup python scripts/rebuild_all_cities_fresh.py > logs/rebuild_all_cities.log 2>&1 &
"""

import logging
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path (scripts/training/dataset/ -> 3 levels up)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.core.train_city_ordinal_optuna import (
    build_dataset_parallel,
    VALID_CITIES,
)
from src.db import get_db_session
from models.data.loader import get_available_date_range

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*80)
    logger.info("REBUILDING ALL 6 CITIES WITH NEW FEATURES")
    logger.info("="*80)
    logger.info("Cities: " + ", ".join(VALID_CITIES))
    logger.info("Workers per city: 20")
    logger.info("")

    base_dir = Path("models/saved")
    all_stats = []
    total_start = datetime.now()

    for i, city in enumerate(VALID_CITIES, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[{i}/6] REBUILDING {city.upper()}")
        logger.info("="*80)

        try:
            # Get available data range (same logic as train_city_ordinal_optuna.py)
            from src.db import get_db_session
            with get_db_session() as session:
                db_min_date, max_date = get_available_date_range(session, city)

            if db_min_date is None or max_date is None:
                logger.error(f"No data available for {city}")
                continue

            # Enforce minimum date of 2023-01-01 to ensure lag features have history
            min_date = max(db_min_date, date(2023, 1, 1))

            if min_date != db_min_date:
                logger.info(f"Enforcing min date: {db_min_date} → {min_date} (need lag history)")

            total_days = (max_date - min_date).days + 1
            logger.info(f"Date range: {min_date} to {max_date} ({total_days} days)")

            start_time = datetime.now()

            # Build dataset
            df = build_dataset_parallel(
                city=city,
                start_date=min_date,
                end_date=max_date,
                n_workers=20,
                include_station_city=True,
            )

            # Save to city-specific folder
            city_dir = base_dir / city
            city_dir.mkdir(parents=True, exist_ok=True)
            output_file = city_dir / "full.parquet"

            df.to_parquet(output_file, index=False)

            elapsed = (datetime.now() - start_time).total_seconds()

            stats = {
                "city": city,
                "rows": len(df),
                "columns": len(df.columns),
                "days": total_days,
                "file": str(output_file),
                "elapsed_sec": elapsed,
            }
            all_stats.append(stats)

            logger.info(f"✓ Saved {len(df):,} rows × {len(df.columns)} columns")
            logger.info(f"✓ Output: {output_file}")
            logger.info(f"✓ Build time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

            # Verify new features
            new_features = [
                'wetbulb_last_obs', 'windchill_last_obs', 'cloudcover_rate_last_30min',
                'log_abs_obs_fcst_gap', 'fcst_multi_cv', 'humidity_x_temp_rate'
            ]
            missing = [f for f in new_features if f not in df.columns]
            present = [f for f in new_features if f in df.columns]

            if missing:
                logger.warning(f"⚠ Missing features: {missing}")

            if present:
                # Check non-null counts
                for f in present[:3]:  # Check first 3
                    non_null = df[f].notna().sum()
                    pct = 100 * non_null / len(df)
                    logger.info(f"  {f}: {non_null:,}/{len(df):,} non-null ({pct:.1f}%)")

        except Exception as e:
            logger.error(f"✗ Failed to rebuild {city}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    total_elapsed = (datetime.now() - total_start).total_seconds()

    logger.info("\n" + "="*80)
    logger.info("REBUILD COMPLETE")
    logger.info("="*80)

    total_rows = sum(s['rows'] for s in all_stats)
    logger.info(f"\n✓ Successfully rebuilt {len(all_stats)}/6 cities")
    logger.info(f"✓ Total rows: {total_rows:,}")
    logger.info(f"✓ Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    logger.info("\nPer-city summary:")
    for s in all_stats:
        logger.info(f"  {s['city']:15s}: {s['rows']:>9,} rows × {s['columns']:>3} cols, {s['elapsed_sec']:>6.1f}s, {s['file']}")

    logger.info(f"\n✓ All datasets ready for training!")
    logger.info("\nNext step: Train with new features")
    logger.info("  python scripts/train_city_ordinal_optuna.py --city austin --trials 50 --use-cached")


if __name__ == "__main__":
    main()
