#!/usr/bin/env python3
"""
Backfill Kalshi markets from the API.

Fetches all markets for the 6 weather series (KXHIGHCHI, KXHIGHAUS, etc.)
and upserts them into kalshi.markets.

Supports resume-on-crash via checkpoints in meta.ingestion_checkpoint.

Usage:
    python scripts/backfill_kalshi_markets.py --days 60
    python scripts/backfill_kalshi_markets.py --start-date 2024-01-01
    python scripts/backfill_kalshi_markets.py --city chicago --days 30
    python scripts/backfill_kalshi_markets.py --all-history  # Fetch all available history
"""

import argparse
import logging
import re
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_city, get_settings
from src.db import get_db_session, KalshiMarket
from src.db.checkpoint import (
    get_or_create_checkpoint,
    update_checkpoint,
    complete_checkpoint,
)
from src.kalshi.client import KalshiClient
from src.kalshi.schemas import Market
from src.utils import KALSHI_LIMITER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_market_ticker(ticker: str) -> dict:
    """
    Parse market ticker to extract city, event_date, strike info.

    Examples:
        KXHIGHCHI-25NOV26-B50 -> between 50-54
        KXHIGHCHI-25NOV26-T90 -> greater than 90
        KXHIGHCHI-25NOV26-T34LO -> less than 34

    Returns:
        Dict with city, event_date, strike_type, floor_strike, cap_strike
    """
    parts = ticker.split("-")
    if len(parts) < 3:
        return {}

    series_ticker = parts[0]
    date_part = parts[1]
    strike_part = parts[2]

    # Parse date: 25NOV26 -> 2025-11-26 (format: YYMONDD)
    # Note: First 2 digits are YEAR (25=2025), then 3-char MONTH, then 2 digits DAY
    try:
        year = 2000 + int(date_part[:2])  # 25 -> 2025
        month_str = date_part[2:5]         # NOV
        day = int(date_part[5:7])          # 26

        months = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
            "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        month = months.get(month_str.upper())
        if month:
            event_date = date(year, month, day)
        else:
            event_date = None
    except (ValueError, IndexError):
        event_date = None

    # Parse strike: B50 (between), T90 (top/greater), T34LO (low/less)
    strike_type = None
    floor_strike = None
    cap_strike = None

    if strike_part.startswith("B"):
        # Between bracket: B55.5 means 55.5-57.5 (2-degree buckets for new markets)
        # Legacy format: B50 means 50-54 (5-degree buckets)
        strike_type = "between"
        try:
            m = re.search(r"B(\d+(?:\.\d+)?)", strike_part)
            if m:
                floor_val = float(m.group(1))
                # Detect bucket width: if floor has decimal (.5), it's new 2°F format
                # Otherwise it's legacy 5°F format
                if "." in m.group(1):
                    bucket_width = 2.0  # New 2°F buckets (B55.5 -> 55.5-57.5)
                else:
                    bucket_width = 4.0  # Legacy 5°F buckets (B50 -> 50-54)
                floor_strike = floor_val
                cap_strike = floor_val + bucket_width
        except (AttributeError, ValueError):
            pass
    elif strike_part.endswith("LO"):
        # Less than: T34LO or T34.5LO means < 34 or < 34.5
        strike_type = "less"
        try:
            m = re.search(r"T(\d+(?:\.\d+)?)LO", strike_part)
            if m:
                cap_strike = float(m.group(1))
        except (AttributeError, ValueError):
            pass
    elif strike_part.startswith("T"):
        # Greater than: T90 or T90.5 means >= 90 or >= 90.5
        strike_type = "greater"
        try:
            m = re.search(r"T(\d+(?:\.\d+)?)", strike_part)
            if m:
                floor_strike = float(m.group(1))
        except (AttributeError, ValueError):
            pass

    # Map series to city (handle both old format HIGHCHI and new format KXHIGHCHI)
    city_map = {city.series_ticker: city.city_id for city in CITIES.values()}
    # Also add legacy format (without KX prefix)
    legacy_map = {
        "HIGHCHI": "chicago",
        "HIGHAUS": "austin",
        "HIGHDEN": "denver",
        "HIGHLAX": "los_angeles",
        "HIGHMIA": "miami",
        "HIGHPHL": "philadelphia",   # Legacy format
        "HIGHPHIL": "philadelphia",  # Kalshi actually uses PHIL not PHL
        "HIGHNYC": "new_york",       # Even though excluded, map it
    }
    city_map.update(legacy_map)
    city = city_map.get(series_ticker)

    return {
        "city": city,
        "event_date": event_date,
        "strike_type": strike_type,
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
    }


def market_to_db_dict(market: Market) -> dict:
    """Convert Kalshi Market schema to database dict for upsert."""
    parsed = parse_market_ticker(market.ticker)

    # Convert timestamps
    listed_at = datetime.fromtimestamp(market.open_time, tz=timezone.utc) if market.open_time else None
    close_time = datetime.fromtimestamp(market.close_time, tz=timezone.utc) if market.close_time else None
    expiration_time = datetime.fromtimestamp(market.expiration_time, tz=timezone.utc) if market.expiration_time else None

    # Settlement value (1 = YES, 0 = NO)
    settlement_value = None
    if market.result == "yes":
        settlement_value = 1
    elif market.result == "no":
        settlement_value = 0

    return {
        "ticker": market.ticker,
        "city": parsed.get("city"),
        "event_date": parsed.get("event_date"),
        "exchange_market_id": None,  # Not in API response
        "strike_type": parsed.get("strike_type") or market.strike_type,
        "floor_strike": float(parsed.get("floor_strike") or market.floor_strike or 0) if (parsed.get("floor_strike") or market.floor_strike) else None,
        "cap_strike": float(parsed.get("cap_strike") or market.cap_strike or 0) if (parsed.get("cap_strike") or market.cap_strike) else None,
        "listed_at": listed_at,
        "close_time": close_time,
        "expiration_time": expiration_time,
        "status": market.status,
        "result": market.result,
        "settlement_value": settlement_value,
        "raw_json": market.model_dump(mode="json"),
    }


def upsert_markets(session, markets: list[dict]) -> int:
    """
    Upsert markets into kalshi.markets.

    Returns:
        Number of rows affected
    """
    if not markets:
        return 0

    stmt = insert(KalshiMarket).values(markets)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={
            "city": stmt.excluded.city,
            "event_date": stmt.excluded.event_date,
            "strike_type": stmt.excluded.strike_type,
            "floor_strike": stmt.excluded.floor_strike,
            "cap_strike": stmt.excluded.cap_strike,
            "close_time": stmt.excluded.close_time,
            "expiration_time": stmt.excluded.expiration_time,
            "status": stmt.excluded.status,
            "result": stmt.excluded.result,
            "settlement_value": stmt.excluded.settlement_value,
            "raw_json": stmt.excluded.raw_json,
            "updated_at": text("NOW()"),
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def backfill_markets(
    client: KalshiClient,
    session,
    series_ticker: str,
    min_close_ts: int,
    max_close_ts: int,
) -> int:
    """
    Fetch and upsert all markets for a series within date range.

    Returns:
        Number of markets upserted
    """
    logger.info(f"Fetching markets for {series_ticker} from {min_close_ts} to {max_close_ts}")

    # Fetch all markets - Kalshi API requires separate calls per status
    # (comma-separated status no longer supported)
    markets = []
    for status in ["settled", "closed"]:
        status_markets = client.get_all_markets(
            series_ticker=series_ticker,
            status=status,
            min_close_ts=min_close_ts,
            max_close_ts=max_close_ts,
        )
        logger.info(f"  {status}: {len(status_markets)} markets")
        markets.extend(status_markets)

    # Deduplicate by ticker (in case of overlap)
    seen_tickers = set()
    unique_markets = []
    for m in markets:
        if m.ticker not in seen_tickers:
            seen_tickers.add(m.ticker)
            unique_markets.append(m)
    markets = unique_markets

    if not markets:
        logger.info(f"No markets found for {series_ticker}")
        return 0

    logger.info(f"Found {len(markets)} markets for {series_ticker}")

    # Convert to DB format
    db_records = [market_to_db_dict(m) for m in markets]

    # Upsert in batches
    batch_size = 100
    total_upserted = 0

    for i in range(0, len(db_records), batch_size):
        batch = db_records[i:i + batch_size]
        rows = upsert_markets(session, batch)
        total_upserted += rows
        logger.debug(f"Upserted batch {i // batch_size + 1}: {rows} rows")

    return total_upserted


def main():
    parser = argparse.ArgumentParser(description="Backfill Kalshi markets")
    parser.add_argument("--days", type=int, default=60, help="Number of days to backfill (default: 60)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD), overrides --days")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD), default: today")
    parser.add_argument("--all-history", action="store_true", help="Fetch all available history (2+ years)")
    parser.add_argument("--city", type=str, help="Single city to backfill (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint tracking")

    args = parser.parse_args()

    # Calculate date range
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.now(timezone.utc)

    if args.all_history:
        # Kalshi weather markets started ~mid 2023, go back 3 years to be safe
        start_date = datetime(2022, 1, 1, tzinfo=timezone.utc)
        logger.info("Fetching ALL available history (since 2022)")
    elif args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=args.days)

    # Make timezone-aware
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    min_close_ts = int(start_date.timestamp())
    max_close_ts = int(end_date.timestamp())

    logger.info(f"Backfilling markets from {start_date.date()} to {end_date.date()}")

    # Get settings and create client with rate limiter
    settings = get_settings()
    client = KalshiClient(
        api_key=settings.kalshi_api_key,
        private_key_path=settings.kalshi_private_key_path,
        base_url=settings.kalshi_base_url,
        rate_limiter=KALSHI_LIMITER,
    )

    # Determine which cities to process
    if args.city:
        city_config = get_city(args.city)
        cities_to_process = {args.city: city_config}
    else:
        cities_to_process = CITIES

    series_tickers = [city.series_ticker for city in cities_to_process.values()]
    logger.info(f"Processing series: {series_tickers}")

    if args.dry_run:
        logger.info("DRY RUN - not writing to database")
        for series_ticker in series_tickers:
            # Kalshi API requires separate calls per status
            total = 0
            for status in ["settled", "closed"]:
                markets = client.get_all_markets(
                    series_ticker=series_ticker,
                    status=status,
                    min_close_ts=min_close_ts,
                    max_close_ts=max_close_ts,
                )
                total += len(markets)
            logger.info(f"{series_ticker}: {total} markets found")
        return

    # Backfill each series with checkpoint tracking
    total_markets = 0
    with get_db_session() as session:
        for city_id, city_config in cities_to_process.items():
            series_ticker = city_config.series_ticker

            # Get or create checkpoint for this city
            checkpoint = None
            if not args.no_checkpoint:
                checkpoint = get_or_create_checkpoint(
                    session=session,
                    pipeline_name="kalshi_markets",
                    city=city_id,
                )
                session.commit()

            try:
                count = backfill_markets(
                    client=client,
                    session=session,
                    series_ticker=series_ticker,
                    min_close_ts=min_close_ts,
                    max_close_ts=max_close_ts,
                )
                total_markets += count
                logger.info(f"{series_ticker} ({city_id}): upserted {count} markets")

                # Update checkpoint on success
                if checkpoint:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        last_date=end_date.date(),
                        processed_count=count,
                    )
                    session.commit()

            except Exception as e:
                logger.error(f"Error processing {series_ticker}: {e}")
                if checkpoint:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        error=str(e),
                    )
                    session.commit()
                raise

            # Mark checkpoint completed for this city
            if checkpoint:
                complete_checkpoint(
                    session=session,
                    checkpoint_id=checkpoint.id,
                    status="completed",
                )
                session.commit()

    logger.info(f"Backfill complete: {total_markets} total markets upserted")


if __name__ == "__main__":
    main()
