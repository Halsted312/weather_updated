#!/usr/bin/env python3
"""
Live Kalshi candlestick polling daemon.

Continuously polls the Kalshi API for recent candlesticks on active weather markets.
Uses the same data source as backfill_kalshi_candles.py for consistency.

Usage:
    # Poll every 60 seconds (default)
    python scripts/poll_kalshi_candles.py

    # Faster polling (every 30 seconds)
    python scripts/poll_kalshi_candles.py --interval 30

    # Specific city only
    python scripts/poll_kalshi_candles.py --city chicago
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_settings
from src.db import get_db_session, KalshiCandle1m, KalshiMarket
from src.kalshi.client import KalshiClient
from src.kalshi.schemas import Candle, Market
from src.utils import KALSHI_LIMITER

# Import market conversion functions from backfill script
from scripts.ingestion.kalshi.backfill_kalshi_markets import market_to_db_dict, upsert_markets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


def candle_to_db_dict(ticker: str, candle: Candle, source: str) -> dict:
    """
    Convert Kalshi Candle schema to database dict for upsert.
    Same logic as backfill_kalshi_candles.py for consistency.
    """
    # API returns end_period_ts (bar end), subtract 60s to get true bar start
    bucket_start = (
        datetime.fromtimestamp(candle.end_period_ts - 60, tz=timezone.utc)
        if candle.end_period_ts
        else None
    )

    return {
        "ticker": ticker,
        "bucket_start": bucket_start,
        "source": source,
        "period_minutes": candle.period_minutes or 1,
        # Last Trade OHLC + statistics
        "trade_open": candle.price_open,
        "trade_high": candle.price_high,
        "trade_low": candle.price_low,
        "trade_close": candle.price_close,
        "trade_mean": candle.price_mean,
        "trade_previous": candle.price_previous,
        "trade_min": candle.price_min,
        "trade_max": candle.price_max,
        # YES Bid OHLC (FULL)
        "yes_bid_open": candle.yes_bid_open,
        "yes_bid_high": candle.yes_bid_high,
        "yes_bid_low": candle.yes_bid_low,
        "yes_bid_close": candle.yes_bid_close,
        # YES Ask OHLC (FULL)
        "yes_ask_open": candle.yes_ask_open,
        "yes_ask_high": candle.yes_ask_high,
        "yes_ask_low": candle.yes_ask_low,
        "yes_ask_close": candle.yes_ask_close,
        # Volume/OI
        "volume": candle.volume,
        "open_interest": candle.open_interest,
    }


def upsert_candles(session, candles: List[dict]) -> int:
    """Upsert candles into kalshi.candles_1m."""
    if not candles:
        return 0

    valid_candles = [c for c in candles if c["bucket_start"] is not None]
    if not valid_candles:
        return 0

    stmt = insert(KalshiCandle1m).values(valid_candles)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "bucket_start", "source"],
        set_={
            "trade_open": stmt.excluded.trade_open,
            "trade_high": stmt.excluded.trade_high,
            "trade_low": stmt.excluded.trade_low,
            "trade_close": stmt.excluded.trade_close,
            "trade_mean": stmt.excluded.trade_mean,
            "trade_previous": stmt.excluded.trade_previous,
            "trade_min": stmt.excluded.trade_min,
            "trade_max": stmt.excluded.trade_max,
            "yes_bid_open": stmt.excluded.yes_bid_open,
            "yes_bid_high": stmt.excluded.yes_bid_high,
            "yes_bid_low": stmt.excluded.yes_bid_low,
            "yes_bid_close": stmt.excluded.yes_bid_close,
            "yes_ask_open": stmt.excluded.yes_ask_open,
            "yes_ask_high": stmt.excluded.yes_ask_high,
            "yes_ask_low": stmt.excluded.yes_ask_low,
            "yes_ask_close": stmt.excluded.yes_ask_close,
            "volume": stmt.excluded.volume,
            "open_interest": stmt.excluded.open_interest,
            "period_minutes": stmt.excluded.period_minutes,
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def get_active_weather_markets_from_api(
    client: KalshiClient,
    city_filter: Optional[str] = None,
) -> List[Market]:
    """
    Get currently active (open) weather markets from the Kalshi API.

    Returns markets where:
    - Status is 'open'
    - Series is a weather series (KXHIGH*)
    """
    # Weather series tickers
    weather_series = [city.series_ticker for city in CITIES.values()]

    if city_filter:
        # Filter to specific city
        city_info = CITIES.get(city_filter)
        if city_info:
            weather_series = [city_info.series_ticker]
        else:
            logger.warning(f"Unknown city filter: {city_filter}")
            return []

    all_markets: List[Market] = []
    now_ts = int(datetime.now(timezone.utc).timestamp())

    for series_ticker in weather_series:
        try:
            # Fetch open markets for this series with close time in future
            markets = client.get_all_markets(
                series_ticker=series_ticker,
                status="open",
                min_close_ts=now_ts,
            )
            all_markets.extend(markets)
            if markets:
                logger.debug(f"Found {len(markets)} open markets for {series_ticker}")
        except Exception as e:
            logger.error(f"Error fetching markets for {series_ticker}: {e}")

    return all_markets


def group_markets_by_event(markets: List[Market]) -> Dict[str, List[Market]]:
    """Group markets by their event_ticker."""
    from collections import defaultdict
    events: Dict[str, List[Market]] = defaultdict(list)

    for market in markets:
        parts = market.ticker.split("-")
        if len(parts) >= 2:
            event_ticker = f"{parts[0]}-{parts[1]}"
            events[event_ticker].append(market)

    return dict(events)


def poll_event_candles(
    client: KalshiClient,
    session,
    event_ticker: str,
    series_ticker: str,
    lookback_minutes: int = 10,
) -> int:
    """
    Poll recent candlesticks for an event.

    Args:
        client: Kalshi API client
        session: Database session
        event_ticker: Event ticker
        series_ticker: Series ticker
        lookback_minutes: How many minutes back to fetch

    Returns:
        Number of candles upserted
    """
    now = datetime.now(timezone.utc)
    start_ts = int((now - timedelta(minutes=lookback_minutes)).timestamp())
    end_ts = int(now.timestamp())

    try:
        # Fetch recent candles for event
        market_candles = client.get_all_event_candlesticks(
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=1,
        )
    except Exception as e:
        logger.error(f"Error fetching candles for {event_ticker}: {e}")
        return 0

    if not market_candles:
        return 0

    # Convert and upsert
    total_candles = 0
    for ticker, candles in market_candles.items():
        if not candles:
            continue

        db_records = [candle_to_db_dict(ticker, c, "api_event") for c in candles]
        if db_records:
            rows = upsert_candles(session, db_records)
            total_candles += rows

    return total_candles


def run_poll_cycle(
    client: KalshiClient,
    city_filter: Optional[str] = None,
    lookback_minutes: int = 10,
) -> Dict[str, int]:
    """
    Run a single poll cycle for all active markets.

    Returns:
        Dict with stats: events_polled, candles_upserted, markets_upserted
    """
    stats = {"events_polled": 0, "candles_upserted": 0, "markets_active": 0, "markets_upserted": 0}

    # Get active markets from API (not database)
    markets = get_active_weather_markets_from_api(client, city_filter)
    stats["markets_active"] = len(markets)

    if not markets:
        logger.debug("No active weather markets found")
        return stats

    # Group by event
    events = group_markets_by_event(markets)
    logger.info(f"Found {len(markets)} active markets in {len(events)} events")

    with get_db_session() as session:
        # First, upsert all markets to satisfy FK constraint
        market_dicts = [market_to_db_dict(m) for m in markets]
        stats["markets_upserted"] = upsert_markets(session, market_dicts)
        session.commit()  # Commit markets before inserting candles

        for event_ticker, event_markets in events.items():
            if shutdown_requested:
                break

            series_ticker = event_markets[0].ticker.split("-")[0]

            candles = poll_event_candles(
                client=client,
                session=session,
                event_ticker=event_ticker,
                series_ticker=series_ticker,
                lookback_minutes=lookback_minutes,
            )

            stats["events_polled"] += 1
            stats["candles_upserted"] += candles

        session.commit()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Live Kalshi candlestick polling daemon")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between poll cycles (default: 60)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=10,
        help="Minutes of history to fetch each cycle (default: 10)",
    )
    parser.add_argument(
        "--city",
        type=str,
        help="Filter to specific city (e.g., chicago)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (for testing)",
    )

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 60)
    logger.info("Kalshi Candlestick Polling Daemon Starting")
    logger.info(f"  Poll interval: {args.interval}s")
    logger.info(f"  Lookback: {args.lookback} minutes")
    logger.info(f"  City filter: {args.city or 'all'}")
    logger.info("=" * 60)

    # Create client
    settings = get_settings()
    client = KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url,
        rate_limiter=KALSHI_LIMITER,
    )

    # Stats
    total_cycles = 0
    total_candles = 0

    try:
        while not shutdown_requested:
            cycle_start = time.time()

            stats = run_poll_cycle(
                client=client,
                city_filter=args.city,
                lookback_minutes=args.lookback,
            )

            total_cycles += 1
            total_candles += stats["candles_upserted"]

            if stats["candles_upserted"] > 0:
                logger.info(
                    f"Cycle {total_cycles}: {stats['events_polled']} events, "
                    f"{stats['candles_upserted']} candles "
                    f"({stats['markets_active']} active markets)"
                )
            else:
                logger.debug(
                    f"Cycle {total_cycles}: {stats['markets_active']} active markets, "
                    f"no new candles"
                )

            if args.once:
                break

            # Sleep until next cycle
            elapsed = time.time() - cycle_start
            sleep_time = max(0, args.interval - elapsed)

            if sleep_time > 0 and not shutdown_requested:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise
    finally:
        logger.info("=" * 60)
        logger.info("Kalshi Candlestick Polling Daemon Stopped")
        logger.info(f"  Total cycles: {total_cycles}")
        logger.info(f"  Total candles: {total_candles}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
