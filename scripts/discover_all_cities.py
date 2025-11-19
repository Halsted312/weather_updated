#!/usr/bin/env python3
"""
Fetch Kalshi weather market data for all cities.

Downloads series metadata, markets, and trades for multiple cities,
then aggregates into 1-minute and 5-minute OHLCV bars.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.discover_chicago import (
    fetch_series_metadata,
    fetch_market_trades,
    create_ohlcv_bars,
    save_to_parquet,
    generate_summary_report,
)
from kalshi.client import KalshiClient
from kalshi.strike_parser import ensure_strike_metadata
from dotenv import load_dotenv
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# City configurations
CITIES = {
    "chicago": {
        "series": "KXHIGHCHI",
        "name": "Chicago",
        "station": "GHCND:USW00014819",  # Midway
    },
    "miami": {
        "series": "KXHIGHMIA",
        "name": "Miami",
        "station": "GHCND:USW00012839",  # Miami Airport
    },
    "austin": {
        "series": "KXHIGHAUS",
        "name": "Austin",
        "station": "GHCND:USW00013958",  # Austin Airport
    },
    "la": {
        "series": "KXHIGHLAX",
        "name": "Los Angeles",
        "station": "GHCND:USW00023174",  # LAX
    },
    "denver": {
        "series": "KXHIGHDEN",
        "name": "Denver",
        "station": "GHCND:USW00003017",  # Denver Airport
    },
    "philadelphia": {
        "series": "KXHIGHPHIL",
        "name": "Philadelphia",
        "station": "GHCND:USW00013739",  # Philly Airport
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi weather market data for all cities"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days of historical data to fetch (relative, mutually exclusive with --start-date/--end-date)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format (mutually exclusive with --days)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (defaults to today if --start-date is provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/raw",
        help="Output directory for city subdirectories (default: ./data/raw)",
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        choices=list(CITIES.keys()),
        default=list(CITIES.keys()),
        help="Cities to fetch (default: all)",
    )

    args = parser.parse_args()

    # Validate date arguments
    if args.start_date and args.days:
        parser.error("Cannot specify both --start-date and --days")

    if not args.start_date and not args.days:
        args.days = 100  # Default to 100 days if nothing specified

    return args


def fetch_markets(
    client: KalshiClient,
    series_ticker: str,
    days: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> List[Dict]:
    """
    Fetch markets for a series.

    Args:
        client: Kalshi API client
        series_ticker: Series ticker (e.g., "KXHIGHCHI")
        days: Number of days back from now (mutually exclusive with start_date/end_date)
        start_date: Start date (mutually exclusive with days)
        end_date: End date (defaults to now if start_date provided)

    Returns:
        List of market dicts
    """
    # Calculate date range
    if days is not None:
        end = datetime.now()
        start = end - timedelta(days=days)
    elif start_date is not None:
        start = start_date
        end = end_date if end_date else datetime.now()
    else:
        raise ValueError("Must provide either 'days' or 'start_date'")

    # Convert to Unix timestamps
    min_close_ts = int(start.timestamp())
    max_close_ts = int(end.timestamp())

    logger.info(
        f"Fetching markets from {start.date()} to {end.date()} "
        f"({(end - start).days} days)..."
    )

    markets = client.get_all_markets(
        series_ticker=series_ticker,
        status="closed,settled",
        min_close_ts=min_close_ts,
        max_close_ts=max_close_ts,
    )

    markets = [ensure_strike_metadata(dict(m)) for m in markets]

    logger.info(f"Found {len(markets)} markets")
    return markets


def fetch_city_data(
    client: KalshiClient,
    city_key: str,
    city_config: Dict,
    days: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_base: Path = None,
) -> None:
    """Fetch data for a single city."""
    city_name = city_config["name"]
    series_ticker = city_config["series"]
    output_dir = output_base / city_key

    logger.info(f"\n{'='*60}")
    logger.info(f"FETCHING DATA FOR {city_name.upper()}")
    logger.info(f"Series: {series_ticker}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}\n")

    # Fetch series metadata
    series = fetch_series_metadata(client, series_ticker)
    save_to_parquet([series], output_dir, "series.parquet")

    # Fetch markets
    markets = fetch_markets(
        client, series_ticker, days=days, start_date=start_date, end_date=end_date
    )
    save_to_parquet(markets, output_dir, "markets.parquet")

    # Fetch trades and aggregate
    all_trades = []
    all_candles_1m = []
    all_candles_5m = []

    for i, market in enumerate(markets, 1):
        ticker = market["ticker"]
        logger.info(f"[{city_name}] Processing market {i}/{len(markets)}: {ticker}")

        # Fetch trades
        trades = fetch_market_trades(client, market)
        all_trades.extend(trades)

        # Aggregate into candles
        if trades:
            candles_dict = create_ohlcv_bars(trades, ticker)

            if "1min" in candles_dict:
                candles_1m_df = candles_dict["1min"]
                all_candles_1m.append(candles_1m_df)
                logger.info(f"  → Generated {len(candles_1m_df)} 1-minute candles")

            if "5min" in candles_dict:
                candles_5m_df = candles_dict["5min"]
                all_candles_5m.append(candles_5m_df)
                logger.info(f"  → Generated {len(candles_5m_df)} 5-minute candles")

    # Save all data
    save_to_parquet(all_trades, output_dir, "trades.parquet")

    if all_candles_1m:
        candles_1m_combined = pd.concat(all_candles_1m, ignore_index=True)
        candles_1m_combined.to_parquet(output_dir / "candles_1m.parquet", index=False)
        logger.info(f"Saved {len(candles_1m_combined)} 1-minute candles")

    if all_candles_5m:
        candles_5m_combined = pd.concat(all_candles_5m, ignore_index=True)
        candles_5m_combined.to_parquet(output_dir / "candles_5m.parquet", index=False)
        logger.info(f"Saved {len(candles_5m_combined)} 5-minute candles")

    # Generate report
    all_candles = candles_1m_combined.to_dict("records") if all_candles_1m else []
    generate_summary_report(series, markets, all_candles, output_dir)

    logger.info(f"✓ {city_name} data fetch complete!\n")


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    api_key = os.getenv("KALSHI_API_KEY")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    base_url = os.getenv(
        "KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
    )

    if not api_key or not private_key_path:
        logger.error("Missing KALSHI_API_KEY or KALSHI_PRIVATE_KEY_PATH in environment")
        sys.exit(1)

    output_base = Path(args.output)
    logger.info(f"Output base directory: {output_base}")
    logger.info(f"Cities to fetch: {', '.join(args.cities)}")

    # Parse dates if provided
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        logger.info(f"Start date: {start_date.date()}")

    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        logger.info(f"End date: {end_date.date()}")
    elif start_date:
        end_date = datetime.now()
        logger.info(f"End date: {end_date.date()} (default to today)")

    if args.days:
        logger.info(f"Days: {args.days} (relative from now)")

    # Initialize client
    logger.info("Initializing Kalshi client...")
    client = KalshiClient(
        api_key=api_key,
        private_key_path=private_key_path,
        base_url=base_url,
    )

    # Fetch data for each city
    for city_key in args.cities:
        city_config = CITIES[city_key]
        try:
            fetch_city_data(
                client,
                city_key,
                city_config,
                days=args.days,
                start_date=start_date,
                end_date=end_date,
                output_base=output_base,
            )
        except Exception as e:
            logger.error(f"Error fetching {city_config['name']}: {e}")
            continue

    logger.info("\n" + "="*60)
    logger.info("ALL CITIES COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
