#!/usr/bin/env python3
"""
Dense candle daemon - continuously fills gaps in kalshi.candles_1m_dense.

Runs every interval (default 60s) and:
1. Finds active markets (open, close_time in future)
2. For each market, checks if dense candles are up to date
3. Forward-fills any gaps from the last dense candle to now

This ensures that even when there's no trading activity, the dense candle
table always has a row for every minute with the last known prices.

Usage:
    python scripts/dense_candle_daemon.py
    python scripts/dense_candle_daemon.py --interval 30  # Every 30 seconds
    python scripts/dense_candle_daemon.py --once         # Run once and exit
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES
from src.db.connection import get_engine

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


# Price columns to forward-fill
PRICE_COLS = [
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "trade_open", "trade_high", "trade_low", "trade_close",
    "trade_mean", "trade_previous", "trade_min", "trade_max",
    "open_interest",
]


def get_active_markets(engine) -> pd.DataFrame:
    """Get currently active weather markets (open, close_time in future)."""
    now = datetime.now(timezone.utc)

    # Get city names from config
    city_names = list(CITIES.keys())

    sql = text("""
        SELECT ticker, listed_at, close_time, event_date, city
        FROM kalshi.markets
        WHERE status IN ('open', 'active')
          AND close_time > :now
          AND city = ANY(:cities)
        ORDER BY event_date, ticker
    """)

    return pd.read_sql(sql, engine, params={"now": now, "cities": city_names})


def get_last_dense_candle(engine, ticker: str) -> Optional[Tuple[datetime, dict]]:
    """Get the most recent dense candle for a ticker."""
    sql = text("""
        SELECT bucket_start,
               yes_bid_close, yes_ask_close,
               trade_close, open_interest, volume
        FROM kalshi.candles_1m_dense
        WHERE ticker = :ticker
        ORDER BY bucket_start DESC
        LIMIT 1
    """)

    result = pd.read_sql(sql, engine, params={"ticker": ticker})
    if result.empty:
        return None

    row = result.iloc[0]
    return row["bucket_start"], row.to_dict()


def get_last_sparse_candle(engine, ticker: str) -> Optional[Tuple[datetime, dict]]:
    """Get the most recent sparse candle for a ticker (source of truth for prices)."""
    sql = text("""
        SELECT bucket_start, source,
               yes_bid_open, yes_bid_high, yes_bid_low, yes_bid_close,
               yes_ask_open, yes_ask_high, yes_ask_low, yes_ask_close,
               trade_open, trade_high, trade_low, trade_close,
               trade_mean, trade_previous, trade_min, trade_max,
               volume, open_interest
        FROM kalshi.candles_1m
        WHERE ticker = :ticker
        ORDER BY bucket_start DESC
        LIMIT 1
    """)

    result = pd.read_sql(sql, engine, params={"ticker": ticker})
    if result.empty:
        return None

    row = result.iloc[0]
    return row["bucket_start"], row.to_dict()


def fill_dense_gaps(
    engine,
    ticker: str,
    from_ts: datetime,
    to_ts: datetime,
    last_prices: dict,
) -> int:
    """
    Fill gaps in dense candles from from_ts to to_ts using last_prices.

    Creates synthetic candles with is_synthetic=True, has_trade=False, volume=0.
    Uses last known prices for all price columns.

    Returns number of rows inserted.
    """
    # Generate minute grid
    # Ensure both timestamps are timezone-aware
    if from_ts.tzinfo is None:
        from_ts = from_ts.replace(tzinfo=timezone.utc)
    if to_ts.tzinfo is None:
        to_ts = to_ts.replace(tzinfo=timezone.utc)

    # Floor to minute boundaries
    from_minute = from_ts.replace(second=0, microsecond=0)
    to_minute = to_ts.replace(second=0, microsecond=0)

    # Start from the next minute after from_ts
    start = from_minute + timedelta(minutes=1)

    if start > to_minute:
        return 0  # Nothing to fill

    # Generate minute range
    minutes = pd.date_range(start=start, end=to_minute, freq="1min", tz="UTC")

    if len(minutes) == 0:
        return 0

    # Build synthetic candle rows
    rows = []
    for bucket_start in minutes:
        row = {
            "ticker": ticker,
            "bucket_start": bucket_start,
            "period_minutes": 1,
            "is_synthetic": True,
            "has_trade": False,
            "volume": 0,
        }
        # Copy price columns from last known values
        for col in PRICE_COLS:
            row[col] = last_prices.get(col)

        rows.append(row)

    if not rows:
        return 0

    df = pd.DataFrame(rows)

    # Upsert into dense table (don't overwrite real data)
    # Use INSERT ... ON CONFLICT DO NOTHING to avoid overwriting
    with engine.begin() as conn:
        # Delete any existing synthetic rows in this range (we'll replace them)
        conn.execute(
            text("""
                DELETE FROM kalshi.candles_1m_dense
                WHERE ticker = :ticker
                  AND bucket_start >= :start
                  AND bucket_start <= :end
                  AND is_synthetic = true
            """),
            {"ticker": ticker, "start": start, "end": to_minute},
        )

    # Insert new synthetic rows
    df.to_sql(
        "candles_1m_dense",
        engine,
        schema="kalshi",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )

    return len(df)


def process_market(engine, ticker: str, close_time: datetime) -> int:
    """
    Process a single market - fill gaps up to now (or close_time).

    Returns number of synthetic candles created.
    """
    now = datetime.now(timezone.utc)

    # Don't fill past close_time
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    target_time = min(now, close_time)

    # Get last dense candle
    last_dense = get_last_dense_candle(engine, ticker)

    if last_dense is None:
        # No dense candles yet - need to bootstrap from sparse
        last_sparse = get_last_sparse_candle(engine, ticker)
        if last_sparse is None:
            return 0  # No data at all

        from_ts, last_prices = last_sparse
    else:
        from_ts, last_prices = last_dense

    # Ensure from_ts is timezone-aware
    if isinstance(from_ts, pd.Timestamp):
        from_ts = from_ts.to_pydatetime()
    if from_ts.tzinfo is None:
        from_ts = from_ts.replace(tzinfo=timezone.utc)

    # Check if we need to fill
    gap_minutes = (target_time - from_ts).total_seconds() / 60

    if gap_minutes <= 1:
        return 0  # Already up to date

    # Get the most recent sparse candle for accurate prices
    last_sparse = get_last_sparse_candle(engine, ticker)
    if last_sparse:
        _, sparse_prices = last_sparse
        # Use sparse prices (more complete) but fall back to dense
        for col in PRICE_COLS:
            if sparse_prices.get(col) is not None:
                last_prices[col] = sparse_prices[col]

    # Fill the gaps
    return fill_dense_gaps(engine, ticker, from_ts, target_time, last_prices)


def run_cycle(engine) -> Dict[str, int]:
    """
    Run one fill cycle for all active markets.

    Returns stats dict.
    """
    stats = {"markets_checked": 0, "markets_filled": 0, "candles_created": 0}

    markets = get_active_markets(engine)
    stats["markets_checked"] = len(markets)

    if markets.empty:
        return stats

    for _, row in markets.iterrows():
        if shutdown_requested:
            break

        ticker = row["ticker"]
        close_time = row["close_time"]

        try:
            candles = process_market(engine, ticker, close_time)
            if candles > 0:
                stats["markets_filled"] += 1
                stats["candles_created"] += candles
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Dense candle gap-filler daemon")
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between fill cycles (default: 60)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit",
    )
    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("=" * 60)
    logger.info("Dense Candle Gap-Filler Daemon Starting")
    logger.info(f"  Fill interval: {args.interval}s")
    logger.info("=" * 60)

    engine = get_engine()

    total_cycles = 0
    total_candles = 0

    try:
        while not shutdown_requested:
            cycle_start = time.time()

            stats = run_cycle(engine)

            total_cycles += 1
            total_candles += stats["candles_created"]

            if stats["candles_created"] > 0:
                logger.info(
                    f"Cycle {total_cycles}: filled {stats['markets_filled']}/{stats['markets_checked']} markets, "
                    f"{stats['candles_created']} candles created"
                )
            else:
                logger.debug(
                    f"Cycle {total_cycles}: {stats['markets_checked']} markets checked, all up to date"
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
        logger.info("Dense Candle Gap-Filler Daemon Stopped")
        logger.info(f"  Total cycles: {total_cycles}")
        logger.info(f"  Total candles created: {total_candles}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
