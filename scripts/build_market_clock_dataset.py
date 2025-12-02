#!/usr/bin/env python3
"""
Build Market-Clock TOD v1 Dataset.

Creates training datasets for the Market-Clock TOD v1 model, which spans
from market open (D-1 10:00 local) to market close (D 23:55 local).

This script is separate from training to allow:
- Retraining with different params without re-hitting the DB
- Dataset inspection and validation before training
- Faster iteration during model development

Usage:
    # Smoke test: 1 city, 30 days
    .venv/bin/python scripts/build_market_clock_dataset.py --mode smoke

    # Medium test: all cities, 60-90 days
    .venv/bin/python scripts/build_market_clock_dataset.py --mode medium

    # Full dataset: all cities, ~700 days
    .venv/bin/python scripts/build_market_clock_dataset.py --mode full

    # Custom range
    .venv/bin/python scripts/build_market_clock_dataset.py \\
        --cities chicago austin \\
        --start-date 2024-01-01 \\
        --end-date 2024-12-31 \\
        --output data/market_clock_tod_v1/custom_dataset.parquet
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_db_session
from models.data.market_clock_dataset_builder import (
    build_market_clock_snapshot_dataset,
    ALL_CITIES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path("data/market_clock_tod_v1")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build Market-Clock TOD v1 dataset'
    )

    parser.add_argument(
        '--mode',
        choices=['smoke', 'medium', 'full', 'custom'],
        default='smoke',
        help='Dataset mode: smoke (1 city, 30 days), medium (6 cities, 90 days), '
             'full (6 cities, ~700 days), custom (specify params)'
    )

    parser.add_argument(
        '--cities',
        nargs='+',
        default=None,
        help='Cities to include (default: depends on mode)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date YYYY-MM-DD (default: depends on mode)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date YYYY-MM-DD (default: yesterday)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Snapshot interval in minutes (default: 5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output parquet path (default: based on mode)'
    )

    parser.add_argument(
        '--no-forecast',
        action='store_true',
        help='Exclude forecast features (faster build)'
    )

    return parser.parse_args()


def get_mode_params(mode: str) -> dict:
    """Get parameters for each mode."""
    yesterday = date.today() - timedelta(days=1)

    if mode == 'smoke':
        return {
            'cities': ['chicago'],
            'start_date': yesterday - timedelta(days=30),
            'end_date': yesterday,
            'output_name': 'train_data_smoke.parquet',
        }
    elif mode == 'medium':
        return {
            'cities': ALL_CITIES,
            'start_date': yesterday - timedelta(days=90),
            'end_date': yesterday,
            'output_name': 'train_data_medium.parquet',
        }
    elif mode == 'full':
        return {
            'cities': ALL_CITIES,
            'start_date': yesterday - timedelta(days=700),
            'end_date': yesterday,
            'output_name': 'train_data.parquet',
        }
    else:  # custom
        return {
            'cities': None,  # Must be specified
            'start_date': None,
            'end_date': yesterday,
            'output_name': 'train_data_custom.parquet',
        }


def validate_dataset(df) -> dict:
    """Validate dataset and return statistics."""
    stats = {
        'total_rows': len(df),
        'unique_cities': df['city'].nunique() if 'city' in df else 0,
        'unique_dates': df['event_date'].nunique() if 'event_date' in df else 0,
        'date_range': None,
        'missing_delta': 0,
        'missing_t_base': 0,
        'cities_breakdown': {},
    }

    if 'event_date' in df and len(df) > 0:
        stats['date_range'] = (
            df['event_date'].min().isoformat() if hasattr(df['event_date'].min(), 'isoformat')
            else str(df['event_date'].min()),
            df['event_date'].max().isoformat() if hasattr(df['event_date'].max(), 'isoformat')
            else str(df['event_date'].max()),
        )

    if 'delta' in df:
        stats['missing_delta'] = df['delta'].isna().sum()

    if 't_base' in df:
        stats['missing_t_base'] = df['t_base'].isna().sum()

    if 'city' in df:
        for city in df['city'].unique():
            city_df = df[df['city'] == city]
            stats['cities_breakdown'][city] = {
                'rows': len(city_df),
                'dates': city_df['event_date'].nunique() if 'event_date' in city_df else 0,
            }

    # Check market-clock features
    market_clock_cols = [
        'minutes_since_market_open',
        'hours_since_market_open',
        'is_d_minus_1',
        'is_event_day',
    ]
    for col in market_clock_cols:
        if col in df:
            stats[f'{col}_present'] = True
            stats[f'{col}_nulls'] = df[col].isna().sum()
        else:
            stats[f'{col}_present'] = False

    # Check city one-hot features
    city_one_hot_cols = [
        'city_chicago', 'city_austin', 'city_denver',
        'city_los_angeles', 'city_miami', 'city_philadelphia',
    ]
    stats['city_one_hot_present'] = all(col in df for col in city_one_hot_cols)

    return stats


def main():
    args = parse_args()

    # Get mode parameters
    mode_params = get_mode_params(args.mode)

    # Override with command line args
    cities = args.cities or mode_params['cities']
    if cities is None:
        logger.error("Cities must be specified for custom mode")
        return 1

    if args.start_date:
        start_date = date.fromisoformat(args.start_date)
    else:
        start_date = mode_params['start_date']
        if start_date is None:
            logger.error("Start date must be specified for custom mode")
            return 1

    if args.end_date:
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = mode_params['end_date']

    output_name = args.output or (OUTPUT_DIR / mode_params['output_name'])
    if isinstance(output_name, str):
        output_name = Path(output_name)

    # Create output directory
    output_name.parent.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 80)
    logger.info(f"BUILDING MARKET-CLOCK TOD V1 DATASET - Mode: {args.mode.upper()}")
    logger.info("=" * 80)
    logger.info(f"Cities: {', '.join(cities)}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Snapshot interval: {args.interval} minutes")
    logger.info(f"Include forecast features: {not args.no_forecast}")
    logger.info(f"Output: {output_name}")
    logger.info("=" * 80)

    # Build dataset
    with get_db_session() as session:
        df = build_market_clock_snapshot_dataset(
            cities=cities,
            start_date=start_date,
            end_date=end_date,
            session=session,
            snapshot_interval_min=args.interval,
            include_forecast_features=not args.no_forecast,
        )

    if df.empty:
        logger.error("No data returned! Check database and date range.")
        return 1

    # Validate
    logger.info("\n" + "=" * 80)
    logger.info("DATASET VALIDATION")
    logger.info("=" * 80)

    stats = validate_dataset(df)

    logger.info(f"Total rows: {stats['total_rows']:,}")
    logger.info(f"Unique cities: {stats['unique_cities']}")
    logger.info(f"Unique dates: {stats['unique_dates']}")
    logger.info(f"Date range: {stats['date_range']}")
    logger.info(f"Missing delta: {stats['missing_delta']}")
    logger.info(f"Missing t_base: {stats['missing_t_base']}")

    logger.info("\nCity breakdown:")
    for city, city_stats in stats['cities_breakdown'].items():
        logger.info(f"  {city}: {city_stats['rows']:,} rows, {city_stats['dates']} dates")

    logger.info("\nMarket-clock features:")
    for col in ['minutes_since_market_open', 'hours_since_market_open', 'is_d_minus_1', 'is_event_day']:
        present = stats.get(f'{col}_present', False)
        nulls = stats.get(f'{col}_nulls', 0)
        logger.info(f"  {col}: {'✓' if present else '✗'} (nulls: {nulls})")

    logger.info(f"\nCity one-hot features: {'✓' if stats['city_one_hot_present'] else '✗'}")

    # Check for issues
    issues = []
    if stats['missing_delta'] > 0:
        issues.append(f"{stats['missing_delta']} rows missing delta")
    if stats['missing_t_base'] > 0:
        issues.append(f"{stats['missing_t_base']} rows missing t_base")
    if not stats.get('minutes_since_market_open_present'):
        issues.append("Missing market-clock features")
    if not stats['city_one_hot_present']:
        issues.append("Missing city one-hot features")

    if issues:
        logger.warning("\n⚠️  Issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    # Save dataset
    logger.info("\n" + "=" * 80)
    logger.info("SAVING DATASET")
    logger.info("=" * 80)

    df.to_parquet(output_name, index=False)
    file_size_mb = output_name.stat().st_size / (1024 * 1024)

    logger.info(f"Saved to: {output_name}")
    logger.info(f"File size: {file_size_mb:.1f} MB")
    logger.info(f"Columns: {len(df.columns)}")

    # Save column list for reference
    cols_path = output_name.with_suffix('.columns.txt')
    with open(cols_path, 'w') as f:
        for col in sorted(df.columns):
            f.write(f"{col}\n")
    logger.info(f"Column list saved to: {cols_path}")

    # Print sample of key columns
    logger.info("\nSample of key columns (first 5 rows):")
    key_cols = [
        'city', 'event_date', 'snapshot_datetime',
        'minutes_since_market_open', 'is_d_minus_1', 'is_event_day',
        't_base', 'delta', 'settle_f',
    ]
    key_cols = [c for c in key_cols if c in df.columns]
    print(df[key_cols].head().to_string())

    logger.info("\n" + "=" * 80)
    logger.info("DATASET BUILD COMPLETE!")
    logger.info("=" * 80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
