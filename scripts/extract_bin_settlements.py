#!/usr/bin/env python3
"""
Extract Kalshi bin settlement data from markets parquet files.

Creates a clean table of bin settlements with:
- ticker, series_ticker, event_ticker
- event_date_local (close_time converted to city timezone)
- settlement fields: result, settlement_value, strike_type, floor_strike, cap_strike
- status
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.date_utils import event_date_from_close_time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# City name mappings for output
CITY_NAMES = {
    "KXHIGHCHI": "chicago",
    "KXHIGHLAX": "la",
    "KXHIGHDEN": "denver",
    "KXHIGHAUS": "austin",
    "KXHIGHMIA": "miami",
    "KXHIGHPHIL": "philadelphia",
}


def extract_series_ticker(ticker: str) -> str:
    """
    Extract series ticker from market ticker.

    Example: KXHIGHCHI-24NOV13-T59 → KXHIGHCHI
    """
    return ticker.split("-")[0]


def convert_to_event_date_local(close_time_str: str, series_ticker: str):
    """
    Convert market close_time (UTC) to event_date_local (city timezone).

    Args:
        close_time_str: ISO format close time (e.g., "2024-11-14T04:59:00Z")
        series_ticker: Series ticker to determine city timezone

    Returns:
        Local date for the event
    """
    return event_date_from_close_time(series_ticker, close_time_str)


def extract_bin_settlements(markets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract bin settlement data from markets DataFrame.

    Args:
        markets_df: DataFrame from markets.parquet

    Returns:
        DataFrame with bin settlement columns
    """
    # Extract series ticker from ticker
    markets_df["series_ticker"] = markets_df["ticker"].apply(extract_series_ticker)

    # Convert close_time to event_date_local
    markets_df["event_date_local"] = markets_df.apply(
        lambda row: convert_to_event_date_local(row["close_time"], row["series_ticker"]),
        axis=1
    )

    # Select relevant columns
    settlements = markets_df[[
        "ticker",
        "series_ticker",
        "event_ticker",
        "event_date_local",
        "close_time",
        "status",
        "result",
        "settlement_value",
        "strike_type",
        "floor_strike",
        "cap_strike",
    ]].copy()

    # Filter to only settled/finalized markets
    settlements = settlements[settlements["status"].isin(["settled", "finalized"])]

    # Sort by event date and ticker
    settlements = settlements.sort_values(["event_date_local", "ticker"])

    return settlements


def process_city_data(city_dir: Path) -> pd.DataFrame:
    """
    Process markets data from a city directory.

    Args:
        city_dir: Path to city data directory (e.g., data/test_run/chicago)

    Returns:
        DataFrame with bin settlements
    """
    markets_file = city_dir / "markets.parquet"

    if not markets_file.exists():
        logger.warning(f"Markets file not found: {markets_file}")
        return pd.DataFrame()

    logger.info(f"Processing: {markets_file}")

    # Read markets
    markets_df = pd.read_parquet(markets_file)
    logger.info(f"  Loaded {len(markets_df)} markets")

    # Extract settlements
    settlements = extract_bin_settlements(markets_df)
    logger.info(f"  Extracted {len(settlements)} settled markets")

    return settlements


def main():
    parser = argparse.ArgumentParser(
        description="Extract Kalshi bin settlements from markets parquet files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/test_run",
        help="Input directory containing city subdirectories (default: ./data/test_run)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/test_run/kalshi_bin_settlements.csv",
        help="Output CSV file (default: ./data/test_run/kalshi_bin_settlements.csv)",
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=None,
        help="Cities to process (default: all subdirectories)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {output_file}")

    # Find city directories
    if args.cities:
        city_dirs = [input_dir / city for city in args.cities]
    else:
        city_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    logger.info(f"Processing {len(city_dirs)} cities...")

    # Process all cities
    all_settlements = []
    for city_dir in city_dirs:
        if not city_dir.is_dir():
            continue

        settlements = process_city_data(city_dir)
        if not settlements.empty:
            all_settlements.append(settlements)

    if not all_settlements:
        logger.error("No settlements extracted")
        return

    # Combine all settlements
    combined = pd.concat(all_settlements, ignore_index=True)

    # Sort by event date, series, and ticker
    combined = combined.sort_values(["event_date_local", "series_ticker", "ticker"])

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    combined.to_csv(output_file, index=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Saved {len(combined)} bin settlements to: {output_file}")
    logger.info(f"{'='*60}")

    # Print summary
    print("\n" + "="*60)
    print("BIN SETTLEMENTS SUMMARY")
    print("="*60)
    print(f"Total settlements: {len(combined)}")
    print(f"\nBy series:")
    print(combined.groupby("series_ticker").size())
    print(f"\nBy event date:")
    print(combined.groupby("event_date_local").size())
    print(f"\nBy strike type:")
    print(combined.groupby("strike_type").size())
    print(f"\nBy result:")
    print(combined.groupby("result").size())
    print("\nSample rows:")
    print(combined[["ticker", "event_date_local", "result", "settlement_value", "strike_type", "floor_strike", "cap_strike"]].head(10))
    print("="*60)


if __name__ == "__main__":
    main()
