#!/usr/bin/env python3
"""
Load historical parquet data into PostgreSQL database.

Reads parquet files from data/raw/{city}/ directories and loads them
into the database using idempotent upsert logic.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from db.loaders import (
    upsert_series,
    upsert_market,
    bulk_upsert_candles,
    bulk_upsert_trades,
    log_ingestion,
)
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to JSON-serializable Python types.

    Args:
        obj: Object to clean

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load historical parquet data into database"
    )
parser.add_argument(
    "--city",
    type=str,
    help="City to load (e.g., chicago, miami). If not specified, loads all cities.",
)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="Base directory containing city data folders (default: ./data/raw)",
    )
    parser.add_argument(
        "--skip-series",
        action="store_true",
        help="Skip loading series metadata",
    )
    parser.add_argument(
        "--skip-markets",
        action="store_true",
        help="Skip loading markets",
    )
    parser.add_argument(
        "--skip-candles",
        action="store_true",
        help="Skip loading candlesticks",
    )
    parser.add_argument(
        "--skip-trades",
        action="store_true",
        help="Skip loading trades",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Only load records on/after this UTC date (e.g., 2025-01-01)",
    )
    return parser.parse_args()


def load_series(session, city_dir: Path) -> Dict[str, Any]:
    """Load series metadata from parquet."""
    series_file = city_dir / "series.parquet"
    if not series_file.exists():
        logger.warning(f"Series file not found: {series_file}")
        return {}

    logger.info(f"Loading series from {series_file}...")
    df = pd.read_parquet(series_file)

    if df.empty:
        logger.warning("Series parquet is empty")
        return {}

    # Convert DataFrame row to dict and clean for JSON
    series_dict = df.iloc[0].to_dict()
    series_dict = clean_for_json(series_dict)

    # Upsert series
    upsert_series(session, series_dict)
    logger.info(f"Loaded series: {series_dict.get('ticker', 'N/A')}")

    return series_dict


def load_markets(
    session,
    city_dir: Path,
    series_ticker: str,
    start_date: Optional[pd.Timestamp] = None,
) -> int:
    """Load markets from parquet."""
    markets_file = city_dir / "markets.parquet"
    if not markets_file.exists():
        logger.warning(f"Markets file not found: {markets_file}")
        return 0

    logger.info(f"Loading markets from {markets_file}...")
    df = pd.read_parquet(markets_file)

    if start_date is not None:
        close_times = pd.to_datetime(df["close_time"], utc=True)
        before = len(df)
        df = df.loc[close_times >= start_date].copy()
        removed = before - len(df)
        logger.info(f"Filtered markets by start_date {start_date.date()}: removed {removed} rows")

    if df.empty:
        logger.warning("Markets parquet is empty")
        return 0

    # Upsert each market
    count = 0
    for _, row in df.iterrows():
        market_dict = row.to_dict()

        # Add series_ticker if not present (Kalshi API doesn't return it)
        if "series_ticker" not in market_dict or pd.isna(market_dict.get("series_ticker")):
            market_dict["series_ticker"] = series_ticker

        # Clean for JSON serializability
        market_dict = clean_for_json(market_dict)

        upsert_market(session, market_dict)
        count += 1

    logger.info(f"Loaded {count} markets for {series_ticker}")
    return count


def load_candles(
    session,
    city_dir: Path,
    period: str = "1m",
    start_date: Optional[pd.Timestamp] = None,
) -> int:
    """Load candlesticks from parquet.

    Args:
        session: Database session
        city_dir: Path to city data directory
        period: "1m" or "5m"
    """
    candles_file = city_dir / f"candles_{period}.parquet"
    if not candles_file.exists():
        logger.warning(f"Candles file not found: {candles_file}")
        return 0

    logger.info(f"Loading {period} candles from {candles_file}...")
    df = pd.read_parquet(candles_file)

    if start_date is not None:
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        before = len(df)
        df = df.loc[timestamps >= start_date].copy()
        removed = before - len(df)
        logger.info(f"Filtered {period} candles by start_date {start_date}: removed {removed} rows")

    if df.empty:
        logger.warning(f"{period} candles parquet is empty")
        return 0

    # Convert DataFrame to list of dicts
    candles = df.to_dict("records")

    # Bulk upsert
    count = bulk_upsert_candles(session, candles)
    logger.info(f"Loaded {count} {period} candles")

    return count


def load_trades(
    session,
    city_dir: Path,
    start_date: Optional[pd.Timestamp] = None,
) -> int:
    """Load trades from parquet."""
    trades_file = city_dir / "trades.parquet"
    if not trades_file.exists():
        logger.warning(f"Trades file not found: {trades_file}")
        return 0

    logger.info(f"Loading trades from {trades_file}...")
    df = pd.read_parquet(trades_file)

    if start_date is not None:
        created_times = pd.to_datetime(df["created_time"], utc=True)
        before = len(df)
        df = df.loc[created_times >= start_date].copy()
        removed = before - len(df)
        logger.info(f"Filtered trades by start_date {start_date}: removed {removed} rows")

    if df.empty:
        logger.warning("Trades parquet is empty")
        return 0

    # Convert DataFrame to list of dicts
    trades = df.to_dict("records")

    # Bulk upsert
    count = bulk_upsert_trades(session, trades)
    logger.info(f"Loaded {count} trades")

    return count


def load_city(
    city_name: str,
    data_dir: Path,
    skip_series: bool = False,
    skip_markets: bool = False,
    skip_candles: bool = False,
    skip_trades: bool = False,
    start_date: Optional[pd.Timestamp] = None,
) -> Dict[str, int]:
    """Load all data for a city.

    Returns:
        Dict with counts of loaded records
    """
    city_dir = data_dir / city_name
    if not city_dir.exists():
        logger.error(f"City directory not found: {city_dir}")
        return {}

    logger.info(f"\n{'=' * 60}")
    logger.info(f"LOADING DATA FOR: {city_name.upper()}")
    logger.info(f"{'=' * 60}\n")

    stats = {
        "city": city_name,
        "series": 0,
        "markets": 0,
        "candles_1m": 0,
        "candles_5m": 0,
        "trades": 0,
    }

    try:
        with get_session() as session:
            # Load series
            series_dict = {}
            if not skip_series:
                series_dict = load_series(session, city_dir)
                if series_dict:
                    stats["series"] = 1

            series_ticker = series_dict.get("ticker", city_name.upper())

            # Load markets
            if not skip_markets:
                stats["markets"] = load_markets(
                    session,
                    city_dir,
                    series_ticker,
                    start_date=start_date,
                )
                # Commit markets before loading candles (FK constraint)
                session.commit()

            # Load candlesticks (both 1m and 5m)
            if not skip_candles:
                stats["candles_1m"] = load_candles(session, city_dir, "1m", start_date=start_date)
                stats["candles_5m"] = load_candles(session, city_dir, "5m", start_date=start_date)

            # Load trades
            if not skip_trades:
                stats["trades"] = load_trades(session, city_dir, start_date=start_date)

            # Log ingestion
            if series_ticker:
                log_ingestion(
                    session,
                    series_ticker=series_ticker,
                    markets_fetched=stats["markets"],
                    candles_1m=stats["candles_1m"],
                    candles_5m=stats["candles_5m"],
                    trades_fetched=stats["trades"],
                    status="success",
                    error_message=None,
                )

        logger.info(f"\n✓ {city_name.upper()} LOAD COMPLETE!")
        logger.info(f"  Series: {stats['series']}")
        logger.info(f"  Markets: {stats['markets']}")
        logger.info(f"  1-min candles: {stats['candles_1m']}")
        logger.info(f"  5-min candles: {stats['candles_5m']}")
        logger.info(f"  Trades: {stats['trades']}\n")

    except Exception as e:
        logger.error(f"Error loading {city_name}: {e}", exc_info=True)

        # Try to log error
        try:
            with get_session() as session:
                log_ingestion(
                    session,
                    series_ticker=series_ticker if series_dict else city_name.upper(),
                    markets_fetched=0,
                    candles_1m=0,
                    candles_5m=0,
                    trades_fetched=0,
                    status="failed",
                    error_message=str(e),
                )
        except:
            pass

        stats["error"] = str(e)

    return stats


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)

    # Determine which cities to load
    if args.city:
        cities = [args.city]
    else:
        # Auto-discover city directories
        cities = [
            d.name
            for d in data_dir.iterdir()
            if d.is_dir() and (d / "series.parquet").exists()
        ]
        if not cities:
            logger.error(f"No city data found in {data_dir}")
            sys.exit(1)

    logger.info(f"Cities to load: {', '.join(cities)}\n")

    start_date = None
    if args.start_date:
        try:
            start_date = pd.to_datetime(args.start_date, utc=True)
        except Exception as exc:
            logger.error(f"Invalid --start-date '{args.start_date}': {exc}")
            sys.exit(1)

    # Load each city
    all_stats = []
    for city in cities:
        stats = load_city(
            city,
            data_dir,
            skip_series=args.skip_series,
            skip_markets=args.skip_markets,
            skip_candles=args.skip_candles,
            skip_trades=args.skip_trades,
            start_date=start_date,
        )
        all_stats.append(stats)

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("LOAD SUMMARY")
    logger.info(f"{'=' * 60}\n")

    total_markets = sum(s.get("markets", 0) for s in all_stats)
    total_candles_1m = sum(s.get("candles_1m", 0) for s in all_stats)
    total_candles_5m = sum(s.get("candles_5m", 0) for s in all_stats)
    total_trades = sum(s.get("trades", 0) for s in all_stats)

    logger.info(f"Total cities loaded: {len(all_stats)}")
    logger.info(f"Total markets: {total_markets:,}")
    logger.info(f"Total 1-min candles: {total_candles_1m:,}")
    logger.info(f"Total 5-min candles: {total_candles_5m:,}")
    logger.info(f"Total trades: {total_trades:,}")

    # Check for errors
    errors = [s for s in all_stats if "error" in s]
    if errors:
        logger.warning(f"\n{len(errors)} cities had errors:")
        for s in errors:
            logger.warning(f"  {s['city']}: {s['error']}")

    logger.info("\n✓ DATABASE LOAD COMPLETE!\n")


if __name__ == "__main__":
    main()
