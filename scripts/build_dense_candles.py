#!/usr/bin/env python3
"""
Build dense 1-minute candles from sparse Kalshi event/trade candles.

Outputs to kalshi.candles_1m_dense:
- One row per (ticker, minute) within observed candle bounds.
- Forward-fills prices/open_interest.
- Sets volume=0 and is_synthetic=True where no raw candle existed.
- has_trade=True when volume > 0.

Optimized for low memory usage:
- Processes in 10-day chunks (configurable via --chunk-days)
- Writes each ticker immediately to DB (no accumulation in RAM)
- Progress logging per ticker

Intended usage:
    python scripts/build_dense_candles.py --city chicago --start-date 2025-01-01 --end-date 2025-01-31
    python scripts/build_dense_candles.py --city chicago --start-date 2025-05-01 --end-date 2025-08-31 --chunk-days 7
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DENSE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS kalshi.candles_1m_dense (
    ticker TEXT NOT NULL,
    bucket_start TIMESTAMPTZ NOT NULL,
    period_minutes SMALLINT NOT NULL DEFAULT 1,
    is_synthetic BOOLEAN NOT NULL,
    has_trade BOOLEAN NOT NULL,
    yes_bid_open SMALLINT,
    yes_bid_high SMALLINT,
    yes_bid_low SMALLINT,
    yes_bid_close SMALLINT,
    yes_ask_open SMALLINT,
    yes_ask_high SMALLINT,
    yes_ask_low SMALLINT,
    yes_ask_close SMALLINT,
    trade_open SMALLINT,
    trade_high SMALLINT,
    trade_low SMALLINT,
    trade_close SMALLINT,
    trade_mean SMALLINT,
    trade_previous SMALLINT,
    trade_min SMALLINT,
    trade_max SMALLINT,
    volume INTEGER,
    open_interest INTEGER,
    PRIMARY KEY (ticker, bucket_start)
);
"""

# Columns to select and their order
PRICE_COLS = [
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "trade_open", "trade_high", "trade_low", "trade_close",
    "trade_mean", "trade_previous", "trade_min", "trade_max",
    "open_interest",
]

OUTPUT_COLS = [
    "ticker", "bucket_start", "period_minutes", "is_synthetic", "has_trade",
    "yes_bid_open", "yes_bid_high", "yes_bid_low", "yes_bid_close",
    "yes_ask_open", "yes_ask_high", "yes_ask_low", "yes_ask_close",
    "trade_open", "trade_high", "trade_low", "trade_close",
    "trade_mean", "trade_previous", "trade_min", "trade_max",
    "volume", "open_interest",
]


def ensure_dense_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(DENSE_TABLE_SQL))


def load_markets_for_chunk(engine, city: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Load markets for a specific date chunk."""
    sql = text("""
        SELECT ticker, listed_at, close_time, event_date
        FROM kalshi.markets
        WHERE city = :city AND event_date BETWEEN :start_date AND :end_date
        ORDER BY event_date, ticker
    """)
    return pd.read_sql(sql, engine, params={"city": city, "start_date": start_date, "end_date": end_date})


def load_sparse_for_ticker(engine, ticker: str) -> pd.DataFrame:
    """Load sparse candles for a single ticker - minimal memory footprint."""
    sql = text("""
        SELECT bucket_start, source,
               yes_bid_open, yes_bid_high, yes_bid_low, yes_bid_close,
               yes_ask_open, yes_ask_high, yes_ask_low, yes_ask_close,
               trade_open, trade_high, trade_low, trade_close,
               trade_mean, trade_previous, trade_min, trade_max,
               volume, open_interest
        FROM kalshi.candles_1m
        WHERE ticker = :ticker
        ORDER BY bucket_start
    """)
    return pd.read_sql(sql, engine, params={"ticker": ticker}, parse_dates=["bucket_start"])


def build_dense_for_ticker(
    sparse: pd.DataFrame,
    ticker: str,
    listed_at,
    close_time,
) -> pd.DataFrame:
    """Build dense candles for a single ticker. Returns empty DataFrame if no data."""
    if sparse.empty:
        return pd.DataFrame()

    df = sparse.copy()

    # Prefer api_event over trades when both exist for same minute
    source_order = {"api_event": 0, "trades": 1}
    df["source_rank"] = df["source"].map(source_order).fillna(99).astype(np.int8)
    df = df.sort_values(["bucket_start", "source_rank"])
    df = df.groupby("bucket_start").first().reset_index()

    first_candle = df["bucket_start"].min().floor("min")
    last_candle = df["bucket_start"].max().floor("min")

    # Determine grid bounds
    start_grid = first_candle
    end_grid = last_candle

    if listed_at is not None:
        listed_ts = pd.to_datetime(listed_at).floor("min")
        if pd.notna(listed_ts):
            start_grid = max(first_candle, listed_ts)

    if close_time is not None:
        close_ts = pd.to_datetime(close_time).floor("min")
        if pd.notna(close_ts):
            end_grid = min(last_candle, close_ts)

    if start_grid > end_grid:
        start_grid, end_grid = first_candle, last_candle

    # Create minute grid and reindex
    index = pd.date_range(start=start_grid, end=end_grid, freq="1min", tz="UTC")
    df = df.set_index("bucket_start").reindex(index)
    df["ticker"] = ticker

    # Mark synthetic rows
    df["is_synthetic"] = df["source"].isna()

    # Forward-fill prices (this is the core densification)
    df[PRICE_COLS] = df[PRICE_COLS].ffill()

    # Handle volume
    df["volume"] = df["volume"].fillna(0).astype(np.int32)
    df.loc[df["is_synthetic"], "volume"] = 0
    df["has_trade"] = df["volume"] > 0
    df["period_minutes"] = np.int16(1)

    # Reset index and select output columns
    df = df.reset_index().rename(columns={"index": "bucket_start"})

    # Only keep needed columns
    return df[OUTPUT_COLS]


def upsert_dense_for_ticker(engine, ticker: str, dense_df: pd.DataFrame) -> int:
    """Upsert dense candles for a single ticker. Returns row count."""
    if dense_df.empty:
        return 0

    start_ts = dense_df["bucket_start"].min()
    end_ts = dense_df["bucket_start"].max()

    with engine.begin() as conn:
        # Delete existing rows for this ticker in the time range
        conn.execute(
            text("""
                DELETE FROM kalshi.candles_1m_dense
                WHERE ticker = :ticker AND bucket_start BETWEEN :start_ts AND :end_ts
            """),
            {"ticker": ticker, "start_ts": start_ts, "end_ts": end_ts},
        )

    # Insert new rows
    dense_df.to_sql(
        "candles_1m_dense",
        engine,
        schema="kalshi",
        if_exists="append",
        index=False,
        method="multi",
        chunksize=5000,  # Insert in smaller chunks for better memory
    )
    return len(dense_df)


def date_chunks(start_date: date, end_date: date, chunk_days: int):
    """Generate date range chunks."""
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description="Build dense 1-minute candles from sparse Kalshi data.")
    parser.add_argument("--city", required=True, help="City id (e.g., chicago)")
    parser.add_argument("--start-date", required=True, help="Start date YYYY-MM-DD (event_date filter)")
    parser.add_argument("--end-date", required=True, help="End date YYYY-MM-DD (event_date filter)")
    parser.add_argument("--chunk-days", type=int, default=10, help="Days per chunk (default: 10)")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    chunk_days = args.chunk_days

    engine = get_engine()
    ensure_dense_table(engine)

    # Calculate totals for progress
    total_days = (end_date - start_date).days + 1
    logger.info(
        "Building dense candles for %s: %s to %s (%d days, %d-day chunks)",
        args.city, start_date, end_date, total_days, chunk_days
    )

    total_tickers = 0
    total_rows = 0
    start_time = time.time()

    # Process in date chunks
    for chunk_start, chunk_end in date_chunks(start_date, end_date, chunk_days):
        chunk_start_time = time.time()

        # Load markets for this chunk only
        markets = load_markets_for_chunk(engine, args.city, chunk_start, chunk_end)

        if markets.empty:
            logger.info("Chunk %s to %s: no markets", chunk_start, chunk_end)
            continue

        chunk_tickers = 0
        chunk_rows = 0

        # Process each ticker individually - load, build, write, free memory
        for _, row in markets.iterrows():
            ticker = row["ticker"]

            # Load sparse candles for this ticker only
            sparse = load_sparse_for_ticker(engine, ticker)

            if sparse.empty:
                continue

            # Build dense candles
            dense_df = build_dense_for_ticker(
                sparse=sparse,
                ticker=ticker,
                listed_at=row["listed_at"],
                close_time=row["close_time"],
            )

            # Write immediately to DB
            rows_written = upsert_dense_for_ticker(engine, ticker, dense_df)

            if rows_written > 0:
                chunk_tickers += 1
                chunk_rows += rows_written

            # Explicitly free memory
            del sparse, dense_df

        total_tickers += chunk_tickers
        total_rows += chunk_rows

        chunk_elapsed = time.time() - chunk_start_time
        logger.info(
            "Chunk %s to %s: %d tickers, %d rows (%.1fs)",
            chunk_start, chunk_end, chunk_tickers, chunk_rows, chunk_elapsed
        )

    elapsed = time.time() - start_time
    rate = total_rows / elapsed if elapsed > 0 else 0

    logger.info(
        "Done: %d tickers, %d rows in %.1fs (%.0f rows/sec)",
        total_tickers, total_rows, elapsed, rate
    )


if __name__ == "__main__":
    main()
