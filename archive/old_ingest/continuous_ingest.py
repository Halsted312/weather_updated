#!/usr/bin/env python3
"""
Continuous real-time ingestion - runs every minute.

Fetches ALL Kalshi weather markets (open, closed, finalized) and loads into database.
Designed to run continuously every minute to maintain a complete real-time dataset.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi.client import KalshiClient
from db.connection import get_session
from db.loaders import (
    upsert_series,
    upsert_market,
    bulk_upsert_candles,
    bulk_upsert_trades,
    log_ingestion,
)
import numpy as np
from weather.time_utils import coerce_datetime_to_utc, utc_now

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# City configuration (non-NYC focus)
CITIES = {
    "chicago": {"series": "KXHIGHCHI", "station": "GHCND:USW00014819"},
    "miami": {"series": "KXHIGHMIA", "station": "GHCND:USW00012839"},
    "austin": {"series": "KXHIGHAUS", "station": "GHCND:USW00013958"},
    "la": {"series": "KXHIGHLAX", "station": "GHCND:USW00023174"},
    "denver": {"series": "KXHIGHDEN", "station": "GHCND:USW00003017"},
    "philadelphia": {"series": "KXHIGHPHIL", "station": "GHCND:USW00013739"},
}


DEFAULT_LOOKBACK_DAYS = float(os.getenv("KALSHI_INGEST_LOOKBACK_DAYS", "30"))


def normalize_market_window(market: Dict[str, Any]) -> tuple[datetime, datetime]:
    """Return UTC-aware open/close timestamps for a Kalshi market payload."""

    open_time = coerce_datetime_to_utc(market["open_time"])
    raw_close = market.get("close_time") or market.get("expiration_time")
    if raw_close is None:
        raise ValueError(f"Market {market.get('ticker')} missing close_time/expiration_time")
    close_time = coerce_datetime_to_utc(raw_close)
    return open_time, close_time


def clean_for_json(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-serializable Python types."""
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


def aggregate_trades_to_candles(
    trades: List[Dict[str, Any]],
    period_minutes: int = 1,
) -> pd.DataFrame:
    """Aggregate trades into OHLCV candlesticks."""
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["timestamp"] = pd.to_datetime(df["created_time"])
    df["price"] = df["yes_price"]
    df = df.sort_values("timestamp")
    df["period"] = df["timestamp"].dt.floor(f"{period_minutes}min")

    candles = df.groupby("period").agg(
        {
            "price": ["first", "max", "min", "last"],
            "count": "sum",
            "trade_id": "count",
        }
    )

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

    candles = candles.reset_index()
    candles = candles.rename(columns={"period": "timestamp"})
    candles["period_minutes"] = period_minutes

    # Filter out incomplete candles (only keep completed periods)
    # A candle is complete when: candle_start + period_duration < current_time
    from datetime import timezone
    current_time = pd.Timestamp.now(tz=timezone.utc)
    candle_end_time = candles["timestamp"] + pd.Timedelta(minutes=period_minutes)
    candles = candles[candle_end_time < current_time].copy()

    return candles


def ingest_city(
    session,
    client: KalshiClient,
    city_name: str,
    series_ticker: str,
    lookback_days: float = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, int]:
    """
    Ingest ALL markets for a city (open, closed, finalized).

    Args:
        session: Database session
        client: Kalshi API client
        city_name: City name
        series_ticker: Series ticker
        lookback_days: Days to look back

    Returns:
        Dict with counts
    """
    logger.info(f"[{city_name.upper()}] Fetching markets...")

    # Fetch ALL markets from last N days (allow deep historical backfill)
    start_date = utc_now() - timedelta(days=lookback_days)
    min_close_ts = int(start_date.timestamp())

    try:
        # Get all markets - don't filter by status to get everything
        markets = client.get_all_markets(
            series_ticker=series_ticker,
            min_close_ts=min_close_ts,
        )

        logger.info(f"[{city_name.upper()}] Found {len(markets)} markets")

        if not markets:
            return {"markets": 0, "candles_1m": 0, "candles_5m": 0, "trades": 0}

        # Upsert series
        series_data = client.get_series(series_ticker)
        series_dict = clean_for_json(series_data.get("series", {}))
        upsert_series(session, series_dict)

        # Upsert all markets
        for market in markets:
            market["series_ticker"] = series_ticker
            market_clean = clean_for_json(market)
            upsert_market(session, market_clean)

        session.commit()

        # Process trades and candles for markets
        all_candles_1m = []
        all_candles_5m = []
        all_trades = []

        for i, market in enumerate(markets, 1):
            ticker = market["ticker"]
            status = market.get("status", "unknown")

            # Parse timestamps
            open_time, close_time = normalize_market_window(market)

            # Only fetch trades for markets that have trades (closed or active with volume)
            if status in ["closed", "finalized", "settled"] or market.get("volume", 0) > 0:
                try:
                    trades = client.get_all_trades(
                        ticker=ticker,
                        min_ts=int(open_time.timestamp()),
                        max_ts=int(close_time.timestamp()),
                    )

                    if trades:
                        all_trades.extend(trades)

                        # Generate candles
                        candles_1m = aggregate_trades_to_candles(trades, 1)
                        if not candles_1m.empty:
                            candles_1m["market_ticker"] = ticker
                            all_candles_1m.append(candles_1m)

                        candles_5m = aggregate_trades_to_candles(trades, 5)
                        if not candles_5m.empty:
                            candles_5m["market_ticker"] = ticker
                            all_candles_5m.append(candles_5m)

                except Exception as e:
                    logger.debug(f"[{city_name.upper()}] {ticker}: {e}")
                    continue

        # Load candles
        candles_1m_count = 0
        candles_5m_count = 0

        if all_candles_1m:
            candles_1m_df = pd.concat(all_candles_1m, ignore_index=True)
            candles_1m_count = bulk_upsert_candles(session, candles_1m_df.to_dict("records"))

        if all_candles_5m:
            candles_5m_df = pd.concat(all_candles_5m, ignore_index=True)
            candles_5m_count = bulk_upsert_candles(session, candles_5m_df.to_dict("records"))

        # Load trades
        trades_count = 0
        if all_trades:
            trades_count = bulk_upsert_trades(session, all_trades)

        # Get date range
        min_close_date = None
        max_close_date = None
        if markets:
            close_times = [
                normalize_market_window(m)[1]
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
        )

        logger.info(
            f"[{city_name.upper()}] ✓ Markets:{len(markets)} "
            f"1m:{candles_1m_count} 5m:{candles_5m_count} Trades:{trades_count}"
        )

        return {
            "markets": len(markets),
            "candles_1m": candles_1m_count,
            "candles_5m": candles_5m_count,
            "trades": trades_count,
        }

    except Exception as e:
        logger.error(f"[{city_name.upper()}] Error: {e}")

        try:
            log_ingestion(
                session,
                series_ticker=series_ticker,
                markets_fetched=0,
                candles_1m=0,
                candles_5m=0,
                trades_fetched=0,
                status="failed",
                error_message=str(e),
            )
        except:
            pass

        return {"markets": 0, "candles_1m": 0, "candles_5m": 0, "trades": 0}


def detect_and_fill_gaps(session, client: KalshiClient) -> Dict[str, int]:
    """
    Detect gaps in candle data and backfill missing periods.

    Returns:
        Dict with counts of filled candles
    """
    from db.models import Market, Candle
    from sqlalchemy import func

    logger.info("Checking for gaps in candle data...")

    filled_stats = {"candles_1m": 0, "candles_5m": 0, "trades": 0}

    # Get all active or recently closed markets (last 3 days)
    cutoff_time = utc_now() - timedelta(days=3)
    markets = session.query(Market).filter(
        Market.close_time > cutoff_time
    ).all()

    logger.info(f"Scanning {len(markets)} recent markets for gaps...")

    for market in markets:
        try:
            # Check if market has any candles
            latest_candle = session.query(Candle).filter(
                Candle.market_ticker == market.ticker
            ).order_by(Candle.timestamp.desc()).first()

            # Parse market times
            open_time = coerce_datetime_to_utc(market.open_time)
            close_time = coerce_datetime_to_utc(market.close_time)
            current_time = utc_now()

            # Determine the expected latest candle time
            # For 1-min candles: should have candles up to (min(close_time, current_time) - 1 min)
            expected_latest_1m = min(close_time, current_time) - timedelta(minutes=1)

            # If no candles exist, or if there's a gap > 10 minutes
            needs_backfill = False
            if latest_candle is None:
                # No candles at all - backfill from open_time
                gap_start = open_time
                needs_backfill = True
            else:
                latest_time = coerce_datetime_to_utc(latest_candle.timestamp)
                gap_duration = (expected_latest_1m - latest_time).total_seconds() / 60

                if gap_duration > 10:  # Gap larger than 10 minutes
                    gap_start = latest_time + timedelta(minutes=1)
                    needs_backfill = True

            if needs_backfill:
                logger.info(f"Gap detected for {market.ticker}: backfilling from {gap_start} to {expected_latest_1m}")

                # Fetch all trades in the gap period
                trades = client.get_all_trades(
                    ticker=market.ticker,
                    min_ts=int(gap_start.timestamp()),
                    max_ts=int(min(close_time, current_time).timestamp()),
                )

                if trades:
                    # Generate candles
                    candles_1m = aggregate_trades_to_candles(trades, 1)
                    candles_5m = aggregate_trades_to_candles(trades, 5)

                    # Upsert candles
                    if not candles_1m.empty:
                        candles_1m["market_ticker"] = market.ticker
                        count = bulk_upsert_candles(session, candles_1m.to_dict("records"))
                        filled_stats["candles_1m"] += count
                        logger.info(f"  → Filled {count} 1-min candles")

                    if not candles_5m.empty:
                        candles_5m["market_ticker"] = market.ticker
                        count = bulk_upsert_candles(session, candles_5m.to_dict("records"))
                        filled_stats["candles_5m"] += count
                        logger.info(f"  → Filled {count} 5-min candles")

                    # Upsert trades
                    count = bulk_upsert_trades(session, trades)
                    filled_stats["trades"] += count

        except Exception as e:
            logger.debug(f"Error backfilling {market.ticker}: {e}")
            continue

    if filled_stats["candles_1m"] > 0 or filled_stats["candles_5m"] > 0:
        logger.info(f"Gap filling complete: {filled_stats['candles_1m']} 1-min, {filled_stats['candles_5m']} 5-min candles")
    else:
        logger.info("No gaps detected")

    return filled_stats


def run_ingestion_cycle(client: KalshiClient):
    """Run one complete ingestion cycle for all cities (parallel processing)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING INGESTION CYCLE - {utc_now().isoformat()}")
    logger.info(f"{'='*60}\n")

    total_stats = {
        "markets": 0,
        "candles_1m": 0,
        "candles_5m": 0,
        "trades": 0,
    }

    try:
        # Process cities in parallel (4 workers for balance of speed vs API limits)
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all city ingestion tasks
            future_to_city = {}
            for city_name, config in CITIES.items():
                # Each city gets its own session to avoid conflicts
                future = executor.submit(
                    _ingest_city_with_session,
                    client,
                    city_name,
                    config["series"],
                )
                future_to_city[future] = city_name

            # Collect results as they complete
            for future in as_completed(future_to_city):
                city_name = future_to_city[future]
                try:
                    stats = future.result()
                    total_stats["markets"] += stats["markets"]
                    total_stats["candles_1m"] += stats["candles_1m"]
                    total_stats["candles_5m"] += stats["candles_5m"]
                    total_stats["trades"] += stats["trades"]
                except Exception as e:
                    logger.error(f"Error processing {city_name}: {e}")

        logger.info(f"\n{'='*60}")
        logger.info(
            f"CYCLE COMPLETE - Markets:{total_stats['markets']:,} "
            f"1m:{total_stats['candles_1m']:,} "
            f"5m:{total_stats['candles_5m']:,} "
            f"Trades:{total_stats['trades']:,}"
        )
        logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Cycle error: {e}", exc_info=True)


def _ingest_city_with_session(
    client: KalshiClient,
    city_name: str,
    series_ticker: str,
) -> Dict[str, int]:
    """Wrapper to ingest a city with its own database session (for parallel execution)."""
    with get_session() as session:
        return ingest_city(
            session,
            client,
            city_name,
            series_ticker,
            lookback_days=DEFAULT_LOOKBACK_DAYS,
        )


def main():
    """Main execution - runs continuously every minute."""
    load_dotenv()

    api_key = os.getenv("KALSHI_API_KEY")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
    base_url = os.getenv(
        "KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
    )

    if not api_key or not private_key_path:
        logger.error("Missing KALSHI_API_KEY or KALSHI_PRIVATE_KEY_PATH")
        sys.exit(1)

    # Initialize client
    client = KalshiClient(
        api_key=api_key,
        private_key_path=private_key_path,
        base_url=base_url,
    )

    logger.info("CONTINUOUS INGESTION STARTED")
    logger.info("Running every 5 seconds. Press Ctrl+C to stop.\n")

    # Check for gaps on startup (handles restarts/downtime)
    logger.info("Running initial gap detection...")
    try:
        with get_session() as session:
            detect_and_fill_gaps(session, client)
    except Exception as e:
        logger.error(f"Gap detection failed on startup: {e}")

    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            logger.info(f"=== CYCLE {cycle_count} ===")

            start_time = time.time()
            run_ingestion_cycle(client)
            elapsed = time.time() - start_time

            logger.info(f"Cycle took {elapsed:.1f}s")

            # Every 60 cycles (5 minutes), check for gaps
            if cycle_count % 60 == 0:
                logger.info("Running periodic gap detection...")
                try:
                    with get_session() as session:
                        detect_and_fill_gaps(session, client)
                except Exception as e:
                    logger.error(f"Gap detection failed: {e}")

            # Wait until next cycle (5 seconds for fast updates)
            sleep_time = max(0, 5 - elapsed)
            if sleep_time > 0:
                logger.info(f"Sleeping {sleep_time:.1f}s until next cycle...\n")
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\n\nStopping continuous ingestion...")
        logger.info(f"Completed {cycle_count} cycles")


if __name__ == "__main__":
    main()
