#!/usr/bin/env python3
"""
Backfill Kalshi 1-minute candlesticks from the API.

Supports dual storage of candlestick data:
- api_event: From Kalshi Event Candlesticks API (more efficient, one call per event)
- trades: Aggregated from individual trades (fallback/audit)

Supports resume-on-crash via checkpoints in meta.ingestion_checkpoint.

Usage:
    # Backfill both sources (default)
    python scripts/backfill_kalshi_candles.py --days 60

    # Backfill only API event candlesticks
    python scripts/backfill_kalshi_candles.py --source api_event --days 30

    # Backfill only trades-derived candles
    python scripts/backfill_kalshi_candles.py --source trades --days 30

    # Specific city or market
    python scripts/backfill_kalshi_candles.py --city chicago --days 30
    python scripts/backfill_kalshi_candles.py --market KXHIGHCHI-25NOV26-B50

    # All history
    python scripts/backfill_kalshi_candles.py --all-history
"""

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_city, get_settings
from src.db import get_db_session, KalshiMarket, KalshiCandle1m
from src.db.checkpoint import (
    get_or_create_checkpoint,
    update_checkpoint,
    complete_checkpoint,
)
from src.kalshi.client import KalshiClient
from src.kalshi.schemas import Candle
from src.utils import KALSHI_LIMITER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def candle_to_db_dict(ticker: str, candle: Candle, source: str) -> dict:
    """
    Convert Kalshi Candle schema to database dict for upsert.

    Args:
        ticker: Market ticker
        candle: Candle object from API
        source: Data source ('api_event' or 'trades')

    Returns:
        Dict ready for database upsert
    """
    bucket_start = (
        datetime.fromtimestamp(candle.end_period_ts, tz=timezone.utc)
        if candle.end_period_ts
        else None
    )

    return {
        "ticker": ticker,
        "bucket_start": bucket_start,
        "source": source,
        # Use price fields for OHLC (these are the actual traded prices)
        "open_c": candle.price_open,
        "high_c": candle.price_high,
        "low_c": candle.price_low,
        "close_c": candle.price_close,
        # Bid/Ask (use close values for snapshot)
        "yes_bid_c": candle.yes_bid_close,
        "yes_ask_c": candle.yes_ask_close,
        # Volume/OI
        "volume": candle.volume,
        "open_interest": candle.open_interest,
    }


def upsert_candles(session, candles: List[dict]) -> int:
    """
    Upsert candles into kalshi.candles_1m.

    Returns:
        Number of rows affected
    """
    if not candles:
        return 0

    # Filter out candles with None bucket_start
    valid_candles = [c for c in candles if c["bucket_start"] is not None]
    if not valid_candles:
        return 0

    stmt = insert(KalshiCandle1m).values(valid_candles)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker", "bucket_start", "source"],  # New composite PK
        set_={
            "open_c": stmt.excluded.open_c,
            "high_c": stmt.excluded.high_c,
            "low_c": stmt.excluded.low_c,
            "close_c": stmt.excluded.close_c,
            "yes_bid_c": stmt.excluded.yes_bid_c,
            "yes_ask_c": stmt.excluded.yes_ask_c,
            "volume": stmt.excluded.volume,
            "open_interest": stmt.excluded.open_interest,
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def get_market_time_range(market: KalshiMarket) -> Tuple[Optional[int], Optional[int]]:
    """Get the time range for fetching candlesticks based on market times."""
    # Use listed_at to close_time (or expiration_time) as the range
    if market.listed_at:
        start_ts = int(market.listed_at.timestamp())
    else:
        # Fallback: 7 days before close
        if market.close_time:
            start_ts = int(market.close_time.timestamp()) - (7 * 24 * 60 * 60)
        else:
            return None, None

    if market.close_time:
        end_ts = int(market.close_time.timestamp())
    elif market.expiration_time:
        end_ts = int(market.expiration_time.timestamp())
    else:
        return None, None

    return start_ts, end_ts


def trades_to_candles(ticker: str, trades: list, source: str = "trades") -> List[dict]:
    """
    Aggregate trades into 1-minute OHLCV candles.

    Args:
        ticker: Market ticker
        trades: List of Trade objects from API
        source: Data source identifier

    Returns:
        List of dicts ready for database upsert
    """
    if not trades:
        return []

    # Group trades by minute bucket
    buckets = defaultdict(list)
    for trade in trades:
        if trade.created_time:
            # Round down to minute
            bucket_ts = (trade.created_time // 60) * 60
            buckets[bucket_ts].append(trade)

    # Aggregate each bucket into OHLCV
    candles = []
    for bucket_ts, bucket_trades in sorted(buckets.items()):
        prices = [t.price for t in bucket_trades if t.price is not None]
        volumes = [t.count for t in bucket_trades]

        if not prices:
            continue

        bucket_start = datetime.fromtimestamp(bucket_ts, tz=timezone.utc)

        candle = {
            "ticker": ticker,
            "bucket_start": bucket_start,
            "source": source,
            "open_c": prices[0],
            "high_c": max(prices),
            "low_c": min(prices),
            "close_c": prices[-1],
            "yes_bid_c": None,  # Not available from trades
            "yes_ask_c": None,
            "volume": sum(volumes),
            "open_interest": None,
        }
        candles.append(candle)

    return candles


def backfill_event_candles(
    client: KalshiClient,
    session,
    event_ticker: str,
    series_ticker: str,
    markets: List[KalshiMarket],
) -> Tuple[int, int]:
    """
    Fetch and upsert candlesticks for all markets in an event using Event Candlesticks API.

    This is more efficient than per-market calls (one API call for all brackets).

    Args:
        client: Kalshi API client
        session: Database session
        event_ticker: Event ticker (e.g., "KXHIGHCHI-25NOV24")
        series_ticker: Series ticker (e.g., "KXHIGHCHI")
        markets: List of markets in this event

    Returns:
        Tuple of (candles_upserted, markets_with_data)
    """
    # Determine time range from all markets in event
    start_ts = None
    end_ts = None
    for market in markets:
        m_start, m_end = get_market_time_range(market)
        if m_start and m_end:
            if start_ts is None or m_start < start_ts:
                start_ts = m_start
            if end_ts is None or m_end > end_ts:
                end_ts = m_end

    if start_ts is None or end_ts is None:
        logger.warning(f"Cannot determine time range for event {event_ticker}")
        return 0, 0

    logger.info(f"Fetching event candlesticks for {event_ticker} ({len(markets)} markets)")

    try:
        # Fetch all candles for event in one call
        market_candles = client.get_all_event_candlesticks(
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            start_ts=start_ts,
            end_ts=end_ts,
            period_interval=1,
        )
    except Exception as e:
        logger.error(f"Error fetching event candlesticks for {event_ticker}: {e}")
        return 0, 0

    if not market_candles:
        logger.warning(f"No candlestick data returned for event {event_ticker}")
        return 0, 0

    # Convert to database format
    total_candles = 0
    markets_with_data = 0

    for ticker, candles in market_candles.items():
        if not candles:
            continue

        db_records = [candle_to_db_dict(ticker, c, "api_event") for c in candles]

        # Upsert in batches
        batch_size = 500
        for i in range(0, len(db_records), batch_size):
            batch = db_records[i : i + batch_size]
            rows = upsert_candles(session, batch)
            total_candles += rows

        if db_records:
            markets_with_data += 1
            logger.debug(f"  {ticker}: {len(db_records)} candles from API")

    return total_candles, markets_with_data


def backfill_trades_candles(
    client: KalshiClient,
    session,
    market: KalshiMarket,
) -> int:
    """
    Fetch trades and aggregate into 1-minute candles for a market.

    Args:
        client: Kalshi API client
        session: Database session
        market: Market to backfill

    Returns:
        Number of candles upserted
    """
    ticker = market.ticker
    start_ts, end_ts = get_market_time_range(market)

    if start_ts is None or end_ts is None:
        logger.warning(f"Cannot determine time range for {ticker}, skipping")
        return 0

    try:
        trades = client.get_all_trades(
            ticker=ticker,
            min_ts=start_ts,
            max_ts=end_ts,
        )
    except Exception as e:
        logger.error(f"Error fetching trades for {ticker}: {e}")
        return 0

    if not trades:
        logger.debug(f"No trades found for {ticker}")
        return 0

    db_records = trades_to_candles(ticker, trades, source="trades")
    if not db_records:
        return 0

    # Upsert in batches
    batch_size = 500
    total_upserted = 0

    for i in range(0, len(db_records), batch_size):
        batch = db_records[i : i + batch_size]
        rows = upsert_candles(session, batch)
        total_upserted += rows

    logger.debug(f"{ticker}: aggregated {len(trades)} trades into {total_upserted} candles")
    return total_upserted


def get_markets_to_backfill(
    session,
    city_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    ticker: Optional[str] = None,
) -> List[KalshiMarket]:
    """Get markets that need candle backfill."""
    query = select(KalshiMarket)

    if ticker:
        query = query.where(KalshiMarket.ticker == ticker)
    else:
        # Only get completed markets (closed/settled/determined/finalized)
        query = query.where(
            KalshiMarket.status.in_(["closed", "settled", "determined", "finalized"])
        )

        if city_id:
            query = query.where(KalshiMarket.city == city_id)

        if start_date:
            query = query.where(KalshiMarket.close_time >= start_date)

        if end_date:
            query = query.where(KalshiMarket.close_time <= end_date)

    query = query.order_by(KalshiMarket.close_time.desc())

    return list(session.execute(query).scalars().all())


def group_markets_by_event(markets: List[KalshiMarket]) -> Dict[str, List[KalshiMarket]]:
    """Group markets by their event_ticker."""
    events = defaultdict(list)

    for market in markets:
        # Derive event_ticker from market ticker: KXHIGHCHI-25NOV24-B50 -> KXHIGHCHI-25NOV24
        parts = market.ticker.split("-")
        if len(parts) >= 2:
            event_ticker = f"{parts[0]}-{parts[1]}"
            events[event_ticker].append(market)
        else:
            logger.warning(f"Cannot derive event_ticker from {market.ticker}")

    return dict(events)


def main():
    parser = argparse.ArgumentParser(description="Backfill Kalshi candlesticks")
    parser.add_argument(
        "--days", type=int, default=60, help="Number of days to backfill (default: 60)"
    )
    parser.add_argument(
        "--start-date", type=str, help="Start date (YYYY-MM-DD), overrides --days"
    )
    parser.add_argument(
        "--end-date", type=str, help="End date (YYYY-MM-DD), default: today"
    )
    parser.add_argument(
        "--all-history", action="store_true", help="Fetch all available history"
    )
    parser.add_argument(
        "--city", type=str, help="Single city to backfill (default: all)"
    )
    parser.add_argument("--market", type=str, help="Single market ticker to backfill")
    parser.add_argument(
        "--source",
        type=str,
        choices=["api_event", "trades", "both"],
        default="both",
        help="Data source to backfill: api_event, trades, or both (default: both)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't write to database"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of events to process"
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true", help="Disable checkpoint tracking"
    )

    args = parser.parse_args()

    # Calculate date range
    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end_date
        else datetime.now(timezone.utc)
    )

    if args.all_history:
        # Fetch all history from database (no date filter)
        start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
        logger.info("Fetching ALL available history")
    elif args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    else:
        start_date = end_date - timedelta(days=args.days)

    logger.info(
        f"Backfilling candles for markets from {start_date.date()} to {end_date.date()}"
    )
    logger.info(f"Source mode: {args.source}")

    # Get settings and create client with rate limiter
    settings = get_settings()
    client = KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url,
        rate_limiter=KALSHI_LIMITER,
    )

    with get_db_session() as session:
        # Get markets to process
        markets = get_markets_to_backfill(
            session=session,
            city_id=args.city,
            start_date=start_date,
            end_date=end_date,
            ticker=args.market,
        )

        if not markets:
            logger.warning(
                "No markets found to backfill. Run backfill_kalshi_markets.py first."
            )
            return

        # Group markets by event
        events = group_markets_by_event(markets)

        if args.limit:
            events = dict(list(events.items())[: args.limit])

        logger.info(f"Found {len(markets)} markets in {len(events)} events")

        if args.dry_run:
            logger.info("DRY RUN - not writing to database")
            for event_ticker, event_markets in list(events.items())[:5]:
                logger.info(f"  {event_ticker}: {len(event_markets)} markets")
                for m in event_markets[:3]:
                    start_ts, end_ts = get_market_time_range(m)
                    if start_ts and end_ts:
                        duration_hours = (end_ts - start_ts) / 3600
                        logger.info(f"    {m.ticker}: {duration_hours:.1f} hours")
            return

        # Backfill statistics
        total_api_candles = 0
        total_trades_candles = 0
        api_events_success = 0
        trades_markets_success = 0
        error_count = 0

        # Get or create checkpoint
        checkpoint = None
        pipeline_name = f"kalshi_candles_{args.source}"
        if not args.no_checkpoint and not args.market:
            checkpoint = get_or_create_checkpoint(
                session=session,
                pipeline_name=pipeline_name,
                city=args.city,
            )
            session.commit()

        try:
            # Process by event for API candlesticks (more efficient)
            if args.source in ("api_event", "both"):
                logger.info("\n=== Backfilling from Event Candlesticks API ===")
                for i, (event_ticker, event_markets) in enumerate(events.items()):
                    try:
                        series_ticker = event_markets[0].ticker.split("-")[0]
                        candles, markets_with_data = backfill_event_candles(
                            client=client,
                            session=session,
                            event_ticker=event_ticker,
                            series_ticker=series_ticker,
                            markets=event_markets,
                        )
                        total_api_candles += candles
                        if candles > 0:
                            api_events_success += 1

                        if (i + 1) % 5 == 0:
                            logger.info(
                                f"Progress: {i + 1}/{len(events)} events, "
                                f"{total_api_candles} API candles"
                            )
                            # Update checkpoint
                            if checkpoint:
                                update_checkpoint(
                                    session=session,
                                    checkpoint_id=checkpoint.id,
                                    last_ticker=event_ticker,
                                    processed_count=candles,
                                )
                            session.commit()

                    except Exception as e:
                        logger.error(f"Error processing event {event_ticker}: {e}")
                        error_count += 1
                        if checkpoint:
                            update_checkpoint(
                                session=session,
                                checkpoint_id=checkpoint.id,
                                error=str(e),
                            )

                session.commit()

            # Process by market for trades (need per-market trades)
            if args.source in ("trades", "both"):
                logger.info("\n=== Backfilling from Trades ===")
                for i, market in enumerate(markets):
                    try:
                        candles = backfill_trades_candles(
                            client=client,
                            session=session,
                            market=market,
                        )
                        total_trades_candles += candles
                        if candles > 0:
                            trades_markets_success += 1

                        if (i + 1) % 20 == 0:
                            logger.info(
                                f"Progress: {i + 1}/{len(markets)} markets, "
                                f"{total_trades_candles} trades candles"
                            )
                            # Update checkpoint
                            if checkpoint:
                                update_checkpoint(
                                    session=session,
                                    checkpoint_id=checkpoint.id,
                                    last_ticker=market.ticker,
                                    processed_count=candles,
                                )
                            session.commit()

                    except Exception as e:
                        logger.error(f"Error processing {market.ticker}: {e}")
                        error_count += 1
                        if checkpoint:
                            update_checkpoint(
                                session=session,
                                checkpoint_id=checkpoint.id,
                                error=str(e),
                            )

                session.commit()

            # Mark checkpoint complete
            if checkpoint:
                complete_checkpoint(
                    session=session,
                    checkpoint_id=checkpoint.id,
                    status="completed",
                )
                session.commit()

        except Exception as e:
            if checkpoint:
                update_checkpoint(
                    session=session,
                    checkpoint_id=checkpoint.id,
                    error=str(e),
                )
                complete_checkpoint(
                    session=session,
                    checkpoint_id=checkpoint.id,
                    status="failed",
                )
                session.commit()
            raise

        # Summary
        logger.info("\n=== Backfill Complete ===")
        logger.info(f"Events processed: {len(events)}")
        logger.info(f"Markets processed: {len(markets)}")

        if args.source in ("api_event", "both"):
            logger.info(f"API candlesticks:")
            logger.info(f"  Events with data: {api_events_success}/{len(events)}")
            logger.info(f"  Total candles: {total_api_candles}")

        if args.source in ("trades", "both"):
            logger.info(f"Trades-derived candles:")
            logger.info(f"  Markets with data: {trades_markets_success}/{len(markets)}")
            logger.info(f"  Total candles: {total_trades_candles}")

        logger.info(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
