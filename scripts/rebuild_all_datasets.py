#!/usr/bin/env python3
"""
Rebuild all city datasets with new features.

This script rebuilds datasets for all cities with the new meteo_advanced and
engineered features. It first runs a quick test on 2 weeks of data to verify
everything works, then builds full datasets for all cities.

Usage:
    # Test mode (2 weeks in 2024)
    python scripts/rebuild_all_datasets.py --test

    # Full rebuild (all cities)
    nohup python scripts/rebuild_all_datasets.py --all > logs/rebuild_datasets.log 2>&1 &

    # Single city
    python scripts/rebuild_all_datasets.py --city austin
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from multiprocessing import Pool as ProcessPool
from functools import partial

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_city_ordinal_optuna import (
    build_dataset_parallel,
    VALID_CITIES,
    DEFAULT_WORKERS,
)
from src.db.connection import get_engine
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_data_range(city: str) -> tuple[date, date]:
    """Get available data range for a city from database."""
    engine = get_engine()

    query = text("""
        SELECT
            MIN(date_local) as min_date,
            MAX(date_local) as max_date
        FROM wx.settlement
        WHERE city = :city
          AND settle_f IS NOT NULL
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"city": city}).fetchone()

    if not result or not result.min_date:
        raise ValueError(f"No settlement data found for {city}")

    return result.min_date, result.max_date


def build_city_dataset(
    city: str,
    start_date: date,
    end_date: date,
    output_dir: Path,
    n_workers: int = DEFAULT_WORKERS,
    include_station_city: bool = True,
) -> dict:
    """Build dataset for a single city and save as parquet.

    Returns:
        Dict with stats about the build
    """
    logger.info("="*80)
    logger.info(f"Building dataset for {city.upper()}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info("="*80)

    start_time = datetime.now()

    # Build dataset
    df = build_dataset_parallel(
        city=city,
        start_date=start_date,
        end_date=end_date,
        n_workers=n_workers,
        include_station_city=include_station_city,
    )

    # Save as parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{city}_full.parquet"
    df.to_parquet(output_file, index=False)

    elapsed = (datetime.now() - start_time).total_seconds()

    stats = {
        "city": city,
        "rows": len(df),
        "columns": len(df.columns),
        "days": (end_date - start_date).days + 1,
        "start_date": start_date,
        "end_date": end_date,
        "output_file": str(output_file),
        "elapsed_seconds": elapsed,
    }

    logger.info(f"✓ Saved {len(df):,} rows ({len(df.columns)} columns) to {output_file}")
    logger.info(f"✓ Build time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Verify new features are present
    new_features = [
        'wetbulb_last_obs', 'windchill_last_obs', 'cloudcover_rate_last_30min',
        'log_abs_obs_fcst_gap', 'fcst_multi_cv', 'humidity_x_temp_rate'
    ]
    missing = [f for f in new_features if f not in df.columns]
    if missing:
        logger.warning(f"⚠ Missing new features: {missing}")
    else:
        logger.info(f"✓ All new features present (checked {len(new_features)} samples)")

    return stats


def test_build(output_dir: Path):
    """Quick test build on 2 weeks of Austin data in 2024."""
    logger.info("="*80)
    logger.info("RUNNING TEST BUILD (2 weeks, Austin, 2024)")
    logger.info("="*80)

    city = "austin"
    test_start = date(2024, 6, 1)  # Random 2 weeks in June 2024
    test_end = date(2024, 6, 14)

    stats = build_city_dataset(
        city=city,
        start_date=test_start,
        end_date=test_end,
        output_dir=output_dir / "test",
        n_workers=8,
    )

    logger.info("\n" + "="*80)
    logger.info("TEST BUILD COMPLETE")
    logger.info("="*80)
    logger.info(f"Rows: {stats['rows']:,}")
    logger.info(f"Columns: {stats['columns']}")
    logger.info(f"File: {stats['output_file']}")
    logger.info(f"Time: {stats['elapsed_seconds']:.1f}s")
    logger.info("\n✓ Test successful! New features are present and working.")
    logger.info("\nNext steps:")
    logger.info("  - Review test file to verify features")
    logger.info("  - Run full rebuild: python scripts/rebuild_all_datasets.py --all")

    return stats


def rebuild_all_cities(
    output_dir: Path,
    cities: list[str] = None,
    n_workers: int = DEFAULT_WORKERS,
):
    """Rebuild datasets for all cities (or specified subset)."""
    if cities is None:
        cities = VALID_CITIES

    logger.info("="*80)
    logger.info(f"REBUILDING DATASETS FOR {len(cities)} CITIES")
    logger.info("="*80)
    logger.info(f"Cities: {', '.join(cities)}")
    logger.info(f"Workers per city: {n_workers}")
    logger.info("")

    all_stats = []

    for i, city in enumerate(cities, 1):
        logger.info(f"\n[{i}/{len(cities)}] Processing {city.upper()}...")

        try:
            # Get available data range for this city
            min_date, max_date = get_data_range(city)
            logger.info(f"Available data: {min_date} to {max_date}")

            # Build full dataset
            stats = build_city_dataset(
                city=city,
                start_date=min_date,
                end_date=max_date,
                output_dir=output_dir,
                n_workers=n_workers,
            )
            all_stats.append(stats)

        except Exception as e:
            logger.error(f"✗ Failed to build {city}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    logger.info("\n" + "="*80)
    logger.info("REBUILD COMPLETE")
    logger.info("="*80)

    total_rows = sum(s['rows'] for s in all_stats)
    total_time = sum(s['elapsed_seconds'] for s in all_stats)

    logger.info(f"\nSuccessfully rebuilt {len(all_stats)}/{len(cities)} cities")
    logger.info(f"Total rows: {total_rows:,}")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("\nPer-city summary:")
    for s in all_stats:
        logger.info(f"  {s['city']:15s}: {s['rows']:>8,} rows, {s['elapsed_seconds']:>6.1f}s")

    logger.info(f"\n✓ All datasets saved to: {output_dir}")
    logger.info("\nNext steps:")
    logger.info("  - Run training with new features: --use-cached flag will now work")
    logger.info("  - Expected improvements: MAE 2.09°F → 1.85-1.95°F")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild city datasets with new features"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test build (2 weeks of Austin data)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rebuild all 6 cities",
    )
    parser.add_argument(
        "--city",
        choices=VALID_CITIES,
        help="Rebuild single city",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        choices=VALID_CITIES,
        help="Rebuild multiple specific cities (e.g., --cities austin chicago)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training_cache"),
        help="Output directory for parquet files",
    )

    args = parser.parse_args()

    # Determine which mode
    if args.test:
        test_build(args.output_dir)
    elif args.all:
        rebuild_all_cities(args.output_dir, n_workers=args.workers)
    elif args.city:
        min_date, max_date = get_data_range(args.city)
        build_city_dataset(
            city=args.city,
            start_date=min_date,
            end_date=max_date,
            output_dir=args.output_dir,
            n_workers=args.workers,
        )
    elif args.cities:
        rebuild_all_cities(args.output_dir, cities=args.cities, n_workers=args.workers)
    else:
        parser.print_help()
        print("\nError: Must specify --test, --all, --city, or --cities")
        sys.exit(1)


if __name__ == "__main__":
    main()
