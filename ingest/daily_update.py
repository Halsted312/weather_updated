#!/usr/bin/env python3
"""
Daily incremental ingestion script.

Fetches new Kalshi weather market data since the last ingestion
and loads it into the database. Designed to be run daily via cron.
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi.client import KalshiClient
from db.connection import get_session
from db.models import IngestionLog
from db.loaders import (
    upsert_series,
    upsert_market,
    bulk_upsert_candles,
    bulk_upsert_trades,
    log_ingestion,
)
from weather.time_utils import coerce_datetime_to_utc, utc_now
from sqlalchemy import select, desc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# City configuration
CITIES = {
    "chicago": {"series": "KXHIGHCHI", "station": "GHCND:USW00014819"},
    "miami": {"series": "KXHIGHMIA", "station": "GHCND:USW00012839"},
    "austin": {"series": "KXHIGHAUS", "station": "GHCND:USW00013958"},
    "la": {"series": "KXHIGHLAX", "station": "GHCND:USW00023174"},
    "denver": {"series": "KXHIGHDEN", "station": "GHCND:USW00003017"},
    "philadelphia": {"series": "KXHIGHPHIL", "station": "GHCND:USW00013739"},
}


def get_last_ingestion_date(session, series_ticker: str) -> Optional[datetime]:
    """
    Get the last successful ingestion date for a series.

    Args:
        session: Database session
        series_ticker: Series to check

    Returns:
        Last max_close_date from successful ingestions, or None
    """
    stmt = (
        select(IngestionLog.max_close_date)
        .where(
            IngestionLog.series_ticker == series_ticker,
            IngestionLog.status == "success",
        )
        .order_by(desc(IngestionLog.max_close_date))
        .limit(1)
    )

    result = session.execute(stmt).scalar_one_or_none()
    return result


def aggregate_trades_to_candles(
    trades: List[Dict[str, Any]],
    period_minutes: int = 1,
) -> pd.DataFrame:
    """
    Aggregate trades into OHLCV candlesticks.

    Args:
        trades: List of trade dicts with created_time, yes_price, count
        period_minutes: Candle period in minutes (1 or 5)

    Returns:
        DataFrame with OHLCV data per period
    """
    if not trades:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(trades)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["created_time"])

    # Use yes_price as the main price (in cents)
    df["price"] = df["yes_price"]

    # Sort by timestamp
    df = df.sort_values("timestamp")

    # Create period buckets
    df["period"] = df["timestamp"].dt.floor(f"{period_minutes}min")

    # Aggregate by period
    candles = df.groupby("period").agg(
        {
            "price": ["first", "max", "min", "last"],  # OHLC
            "count": "sum",  # Volume (total contracts traded)
            "trade_id": "count",  # Number of trades
        }
    )

    # Flatten column names
    candles.columns = ["_".join(col).strip() for col in candles.columns.values]
    candles = candles.rename(
        columns={
            "price_first": "open",
            "price_max": "high",
            "price_min": "low",
            "price_last": "close",
            "count_sum": "volume",
            "trade_id_count": "num_trades",
        }
    )

    # Reset index to make period a column
    candles = candles.reset_index()
    candles = candles.rename(columns={"period": "timestamp"})

    # Add period_minutes column
    candles["period_minutes"] = period_minutes

    return candles


def ingest_city(
    session,
    client: KalshiClient,
    city_name: str,
    series_ticker: str,
    lookback_days: int = 7,
) -> Dict[str, int]:
    """
    Ingest new data for a city since last ingestion.

    Args:
        session: Database session
        client: Kalshi API client
        city_name: City name (for logging)
        series_ticker: Series ticker (e.g., KXHIGHCHI)
        lookback_days: Days to look back if no previous ingestion

    Returns:
        Dict with counts of fetched data
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"INGESTING: {city_name.upper()} ({series_ticker})")
    logger.info(f"{'=' * 60}\n")

    # Get last ingestion date
    last_date = get_last_ingestion_date(session, series_ticker)

    if last_date:
        # Fetch since last ingestion
        start_date = last_date
        logger.info(f"Last ingestion: {start_date.date()}")
        logger.info(f"Fetching new markets since {start_date.date()}...")
    else:
        # First ingestion - fetch last N days
        start_date = utc_now() - timedelta(days=lookback_days)
        logger.info(f"No previous ingestion found")
        logger.info(f"Fetching last {lookback_days} days...")

    end_date = utc_now()

    # Convert to Unix timestamps
    min_close_ts = int(start_date.timestamp())
    max_close_ts = int(end_date.timestamp())

    # Fetch markets
    logger.info(f"Fetching markets from {start_date.date()} to {end_date.date()}...")
    markets = client.get_all_markets(
        series_ticker=series_ticker,
        status="closed,settled,finalized",
        min_close_ts=min_close_ts,
        max_close_ts=max_close_ts,
    )

    logger.info(f"Found {len(markets)} markets")

    if not markets:
        logger.info("No new markets to process")
        return {
            "markets": 0,
            "candles_1m": 0,
            "candles_5m": 0,
            "trades": 0,
        }

    # Upsert series (in case metadata changed)
    series_data = client.get_series(series_ticker)
    upsert_series(session, series_data.get("series", {}))
    session.commit()

    # Upsert markets
    for market in markets:
        # Add series_ticker (API doesn't return it)
        market["series_ticker"] = series_ticker
        upsert_market(session, market)

    session.commit()
    logger.info(f"Loaded {len(markets)} markets")

    # Fetch trades and generate candles
    all_candles_1m = []
    all_candles_5m = []
    all_trades = []

    for i, market in enumerate(markets, 1):
        ticker = market["ticker"]
        logger.info(f"Processing market {i}/{len(markets)}: {ticker}")

        # Parse timestamps
        open_time = coerce_datetime_to_utc(market["open_time"])
        close_time = coerce_datetime_to_utc(market["close_time"])

        # Fetch trades
        try:
            trades = client.get_all_trades(
                ticker=ticker,
                min_ts=int(open_time.timestamp()),
                max_ts=int(close_time.timestamp()),
            )

            if trades:
                logger.info(f"  → Got {len(trades)} trades")
                all_trades.extend(trades)

                # Generate 1-min candles
                candles_1m = aggregate_trades_to_candles(trades, period_minutes=1)
                if not candles_1m.empty:
                    candles_1m["market_ticker"] = ticker
                    all_candles_1m.append(candles_1m)
                    logger.info(f"  → Generated {len(candles_1m)} 1-minute candles")

                # Generate 5-min candles
                candles_5m = aggregate_trades_to_candles(trades, period_minutes=5)
                if not candles_5m.empty:
                    candles_5m["market_ticker"] = ticker
                    all_candles_5m.append(candles_5m)
                    logger.info(f"  → Generated {len(candles_5m)} 5-minute candles")
            else:
                logger.info(f"  → No trades")

        except Exception as e:
            logger.error(f"  → Error fetching trades: {e}")
            continue

    # Load candles
    candles_1m_count = 0
    candles_5m_count = 0

    if all_candles_1m:
        candles_1m_df = pd.concat(all_candles_1m, ignore_index=True)
        candles_1m_list = candles_1m_df.to_dict("records")
        candles_1m_count = bulk_upsert_candles(session, candles_1m_list)
        logger.info(f"Loaded {candles_1m_count} 1-minute candles")

    if all_candles_5m:
        candles_5m_df = pd.concat(all_candles_5m, ignore_index=True)
        candles_5m_list = candles_5m_df.to_dict("records")
        candles_5m_count = bulk_upsert_candles(session, candles_5m_list)
        logger.info(f"Loaded {candles_5m_count} 5-minute candles")

    # Load trades
    trades_count = 0
    if all_trades:
        trades_count = bulk_upsert_trades(session, all_trades)
        logger.info(f"Loaded {trades_count} trades")

    # Get date range
    min_close_date = None
    max_close_date = None
    if markets:
        close_times = [
            coerce_datetime_to_utc(m["close_time"])
            for m in markets
        ]
        min_close_date = min(close_times)
        max_close_date = max(close_times)

    # Log ingestion
    log_ingestion(
        session,
        series_ticker=series_ticker,
        markets_fetched=len(markets),
        candles_1m=candles_1m_count,
        candles_5m=candles_5m_count,
        trades_fetched=trades_count,
        min_close_date=min_close_date,
        max_close_date=max_close_date,
        status="success",
        error_message=None,
    )

    logger.info(f"\n✓ {city_name.upper()} INGESTION COMPLETE!")
    logger.info(f"  Markets: {len(markets)}")
    logger.info(f"  1-min candles: {candles_1m_count}")
    logger.info(f"  5-min candles: {candles_5m_count}")
    logger.info(f"  Trades: {trades_count}\n")

    return {
        "markets": len(markets),
        "candles_1m": candles_1m_count,
        "candles_5m": candles_5m_count,
        "trades": trades_count,
    }


def main():
    """Main execution function."""
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

    # Initialize client
    logger.info("Initializing Kalshi client...")
    client = KalshiClient(
        api_key=api_key,
        private_key_path=private_key_path,
        base_url=base_url,
    )

    # Ingest each city
    total_stats = {
        "markets": 0,
        "candles_1m": 0,
        "candles_5m": 0,
        "trades": 0,
    }

    try:
        with get_session() as session:
            for city_name, config in CITIES.items():
                try:
                    stats = ingest_city(
                        session,
                        client,
                        city_name,
                        config["series"],
                        lookback_days=7,
                    )

                    total_stats["markets"] += stats["markets"]
                    total_stats["candles_1m"] += stats["candles_1m"]
                    total_stats["candles_5m"] += stats["candles_5m"]
                    total_stats["trades"] += stats["trades"]

                except Exception as e:
                    logger.error(f"Error ingesting {city_name}: {e}", exc_info=True)

                    # Log error
                    try:
                        log_ingestion(
                            session,
                            series_ticker=config["series"],
                            markets_fetched=0,
                            candles_1m=0,
                            candles_5m=0,
                            trades_fetched=0,
                            status="failed",
                            error_message=str(e),
                        )
                    except:
                        pass

        # Print summary
        logger.info(f"\n{'=' * 60}")
        logger.info("DAILY INGESTION SUMMARY")
        logger.info(f"{'=' * 60}\n")
        logger.info(f"Total markets: {total_stats['markets']:,}")
        logger.info(f"Total 1-min candles: {total_stats['candles_1m']:,}")
        logger.info(f"Total 5-min candles: {total_stats['candles_5m']:,}")
        logger.info(f"Total trades: {total_stats['trades']:,}")
        logger.info(f"\n✓ DAILY INGESTION COMPLETE!\n")

    except Exception as e:
        logger.error(f"Fatal error in daily ingestion: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
