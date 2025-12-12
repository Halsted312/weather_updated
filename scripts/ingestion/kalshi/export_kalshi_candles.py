#!/usr/bin/env python3
"""Export Kalshi candle data to parquet for offline edge classifier training.

This script exports all 1-minute candle data for a city from the database
to a parquet file, allowing edge classifier training on machines without
database access.

Usage:
    # Export single city
    python scripts/export_kalshi_candles.py --city denver

    # Export all cities
    python scripts/export_kalshi_candles.py --all

    # Custom output path
    python scripts/export_kalshi_candles.py --city austin --output /path/to/candles.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# Add project root to path (scripts/ingestion/kalshi/ -> 3 levels up)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.db.connection import get_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Ticker patterns for each city (both old and new formats)
TICKER_PATTERNS = {
    "chicago": ["KXHIGHCHI%", "HIGHCHI%"],
    "austin": ["KXHIGHAUS%", "HIGHAUS%"],
    "denver": ["KXHIGHDEN%", "HIGHDEN%"],
    "los_angeles": ["KXHIGHLAX%", "HIGHLAX%"],
    "miami": ["KXHIGHMIA%", "HIGHMIA%"],
    "philadelphia": ["KXHIGHPHIL%", "HIGHPHIL%"],
}

ALL_CITIES = list(TICKER_PATTERNS.keys())


def export_candles(
    city: str,
    output_path: Path,
    start_date: str = None,
    end_date: str = None,
) -> int:
    """Export candle data for a city to parquet.

    Args:
        city: City name (lowercase)
        output_path: Output parquet file path
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD, exclusive)

    Returns:
        Number of rows exported
    """
    if city not in TICKER_PATTERNS:
        logger.error(f"Unknown city: {city}. Valid: {ALL_CITIES}")
        return 0

    engine = get_engine()
    patterns = TICKER_PATTERNS[city]
    pattern_clause = " OR ".join([f"ticker LIKE '{p}'" for p in patterns])

    # Build date filter clause
    date_filters = []
    if start_date:
        date_filters.append(f"bucket_start >= '{start_date}'")
    if end_date:
        date_filters.append(f"bucket_start < '{end_date}'")
    date_clause = f" AND {' AND '.join(date_filters)}" if date_filters else ""

    logger.info(f"Exporting candles for {city}...")
    logger.info(f"Patterns: {patterns}")
    if start_date or end_date:
        logger.info(f"Date range: {start_date or 'start'} to {end_date or 'end'}")

    query = f"""
        SELECT
            ticker,
            bucket_start,
            period_minutes,
            is_synthetic,
            has_trade,
            yes_bid_open,
            yes_bid_high,
            yes_bid_low,
            yes_bid_close,
            yes_ask_open,
            yes_ask_high,
            yes_ask_low,
            yes_ask_close,
            trade_open,
            trade_high,
            trade_low,
            trade_close,
            trade_mean,
            trade_previous,
            trade_min,
            trade_max,
            volume,
            open_interest
        FROM kalshi.candles_1m_dense
        WHERE ({pattern_clause}){date_clause}
        ORDER BY ticker, bucket_start
    """

    logger.info("Running query (this may take a few minutes)...")
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        logger.warning(f"No candles found for {city}")
        return 0

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet with compression
    df.to_parquet(output_path, index=False, compression="snappy")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Exported {len(df):,} candles to {output_path} ({file_size_mb:.1f} MB)")

    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Export Kalshi candle data to parquet for offline training"
    )
    parser.add_argument(
        "--city",
        type=str,
        choices=ALL_CITIES,
        help="City to export",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export all cities",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: models/candles/candles_{city}.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/candles",
        help="Output directory for --all mode (default: models/candles/)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD, exclusive)",
    )
    args = parser.parse_args()

    if not args.city and not args.all:
        parser.error("Must specify --city or --all")

    if args.all:
        # Export all cities
        output_dir = Path(args.output_dir)
        total_rows = 0
        for city in ALL_CITIES:
            output_path = output_dir / f"candles_{city}.parquet"
            rows = export_candles(city, output_path, args.start_date, args.end_date)
            total_rows += rows
        logger.info(f"Total: {total_rows:,} candles exported for {len(ALL_CITIES)} cities")
    else:
        # Export single city
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = Path(f"models/candles/candles_{args.city}.parquet")

        export_candles(args.city, output_path, args.start_date, args.end_date)

    return 0


if __name__ == "__main__":
    sys.exit(main())
