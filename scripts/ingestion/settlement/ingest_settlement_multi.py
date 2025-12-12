#!/usr/bin/env python3
"""
Multi-source settlement data ingestion script.

Fetches historical TMAX data from multiple sources and reconciles them:
1. IEM CLI JSON API (primary - historical NWS CLI data)
2. NCEI Access Data Service (validator/fallback)

Also links Kalshi market settlement results to temperature data.

Usage:
    python scripts/ingest_settlement_multi.py --all-cities --all-history
    python scripts/ingest_settlement_multi.py --city chicago --start-date 2024-01-01 --end-date 2024-12-31
"""

import argparse
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional

from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert

from src.db import get_db_session
from src.db.models import WxSettlement, KalshiMarket
from src.db.checkpoint import get_or_create_checkpoint, update_checkpoint, complete_checkpoint
from src.weather.iem_cli import IEMCliClient
from src.weather.noaa_ncei import NCEIAccessClient
from src.weather.nws_cli import SETTLEMENT_STATIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_kalshi_settlement_for_date(
    session, city: str, target_date: date
) -> Optional[Dict]:
    """
    Find the Kalshi market that settled YES for a given city and date.

    For overlapping "greater than X" markets, picks the most specific one
    (highest floor_strike). For "less than X" markets, picks lowest cap_strike.

    Returns dict with settled_ticker, settled_bucket_type, floor_strike, cap_strike, bucket_label.
    """
    # Query for all markets that settled YES on this date
    stmt = select(KalshiMarket).where(
        and_(
            KalshiMarket.city == city,
            KalshiMarket.event_date == target_date,
            KalshiMarket.result == "yes",
        )
    )

    results = session.execute(stmt).scalars().all()

    if not results:
        return None

    # If multiple results, pick the most specific bucket
    if len(results) > 1:
        # Group by strike_type and pick most specific
        between = [r for r in results if r.strike_type == "between"]
        greater = [r for r in results if r.strike_type == "greater"]
        less = [r for r in results if r.strike_type == "less"]

        # "between" buckets are most specific, prefer them
        if between:
            result = between[0]
        # For "greater", pick highest floor_strike (most restrictive)
        elif greater:
            result = max(greater, key=lambda r: r.floor_strike or 0)
        # For "less", pick lowest cap_strike (most restrictive)
        elif less:
            result = min(less, key=lambda r: r.cap_strike or 999)
        else:
            result = results[0]

        logger.debug(f"Multiple YES markets for {city} {target_date}, picked {result.ticker}")
    else:
        result = results[0]

    # Build human-readable bucket label
    bucket_label = None
    if result.strike_type == "greater":
        bucket_label = f"above {result.floor_strike}°F"
    elif result.strike_type == "less":
        bucket_label = f"below {result.cap_strike}°F"
    elif result.strike_type == "between":
        bucket_label = f"{result.floor_strike}-{result.cap_strike}°F"

    return {
        "settled_ticker": result.ticker,
        "settled_bucket_type": result.strike_type,
        "settled_floor_strike": result.floor_strike,
        "settled_cap_strike": result.cap_strike,
        "settled_bucket_label": bucket_label,
    }


def choose_tmax_final(
    tmax_iem: Optional[int],
    tmax_ncei: Optional[int],
    tmax_cli: Optional[int] = None,
    tmax_cf6: Optional[int] = None,
) -> tuple[int, str]:
    """
    Choose the final TMAX value based on precedence.

    Precedence: CLI > IEM (same as CLI) > CF6 > NCEI

    Returns (tmax_final, source_final)
    """
    # IEM CLI JSON is the parsed NWS CLI - highest priority for historical
    if tmax_iem is not None:
        return tmax_iem, "iem"

    # Direct CLI (only works for current day)
    if tmax_cli is not None:
        return tmax_cli, "cli"

    # CF6 (preliminary monthly)
    if tmax_cf6 is not None:
        return tmax_cf6, "cf6"

    # NCEI (canonical fallback)
    if tmax_ncei is not None:
        return tmax_ncei, "ncei"

    raise ValueError("No TMAX data available from any source")


def ingest_settlement_for_city(
    session,
    city: str,
    start_date: date,
    end_date: date,
    iem_client: IEMCliClient,
    ncei_client: NCEIAccessClient,
    checkpoint_id: Optional[int] = None,
) -> int:
    """
    Ingest settlement data for a city from multiple sources.

    Returns number of records processed.
    """
    logger.info(f"Ingesting settlement for {city} from {start_date} to {end_date}")

    # Fetch IEM CLI data for date range
    logger.info(f"  Fetching IEM CLI data...")
    iem_data = iem_client.fetch_city_history(city, start_date, end_date)
    iem_by_date = {r["date_local"]: r for r in iem_data}
    logger.info(f"  Got {len(iem_data)} IEM records")

    # Fetch NCEI data for date range
    logger.info(f"  Fetching NCEI data...")
    ncei_data = ncei_client.fetch_city_history(city, start_date, end_date)
    ncei_by_date = {r["date_local"]: r for r in ncei_data}
    logger.info(f"  Got {len(ncei_data)} NCEI records")

    # Get all dates in range
    all_dates = set(iem_by_date.keys()) | set(ncei_by_date.keys())
    logger.info(f"  Processing {len(all_dates)} unique dates")

    processed = 0
    for target_date in sorted(all_dates):
        iem_record = iem_by_date.get(target_date)
        ncei_record = ncei_by_date.get(target_date)

        tmax_iem = iem_record["tmax_f"] if iem_record else None
        tmax_ncei = ncei_record["tmax_f"] if ncei_record else None

        # Skip if no data from any source
        if tmax_iem is None and tmax_ncei is None:
            continue

        # Choose final TMAX
        try:
            tmax_final, source_final = choose_tmax_final(tmax_iem, tmax_ncei)
        except ValueError:
            logger.warning(f"  No TMAX data for {city} on {target_date}")
            continue

        # Get Kalshi settlement info
        kalshi_info = get_kalshi_settlement_for_date(session, city, target_date)

        # Build record
        record = {
            "city": city,
            "date_local": target_date,
            "tmax_iem_f": tmax_iem,
            "tmax_ncei_f": tmax_ncei,
            "tmax_final": tmax_final,
            "source_final": source_final,
            "raw_payload_iem": iem_record["raw_json"] if iem_record else None,
            "raw_payload_ncei": ncei_record["raw_json"] if ncei_record else None,
        }

        # Add Kalshi settlement info if available
        if kalshi_info:
            record.update(kalshi_info)

        # Upsert record
        stmt = insert(WxSettlement).values(**record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["city", "date_local"],
            set_={
                "tmax_iem_f": stmt.excluded.tmax_iem_f,
                "tmax_ncei_f": stmt.excluded.tmax_ncei_f,
                "tmax_final": stmt.excluded.tmax_final,
                "source_final": stmt.excluded.source_final,
                "raw_payload_iem": stmt.excluded.raw_payload_iem,
                "raw_payload_ncei": stmt.excluded.raw_payload_ncei,
                "settled_ticker": stmt.excluded.settled_ticker,
                "settled_bucket_type": stmt.excluded.settled_bucket_type,
                "settled_floor_strike": stmt.excluded.settled_floor_strike,
                "settled_cap_strike": stmt.excluded.settled_cap_strike,
                "settled_bucket_label": stmt.excluded.settled_bucket_label,
            },
        )
        session.execute(stmt)
        processed += 1

        # Update checkpoint periodically
        if checkpoint_id and processed % 100 == 0:
            session.commit()
            update_checkpoint(session, checkpoint_id, target_date, processed)
            logger.info(f"  Processed {processed} records (through {target_date})")

    session.commit()
    logger.info(f"  Completed {city}: {processed} records")
    return processed


def main():
    parser = argparse.ArgumentParser(description="Ingest multi-source settlement data")
    parser.add_argument("--city", type=str, help="Single city to process")
    parser.add_argument("--all-cities", action="store_true", help="Process all cities")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--all-history", action="store_true", help="Process all available history (since 2022)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Determine cities to process
    if args.all_cities:
        cities = list(SETTLEMENT_STATIONS.keys())
    elif args.city:
        cities = [args.city]
    else:
        parser.error("Must specify --city or --all-cities")

    # Determine date range
    if args.all_history:
        start_date = date(2022, 1, 1)
        end_date = date.today() - timedelta(days=1)
        logger.info(f"Processing all history from {start_date} to {end_date}")
    elif args.start_date and args.end_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    else:
        parser.error("Must specify --start-date/--end-date or --all-history")

    # Initialize clients
    iem_client = IEMCliClient()
    ncei_client = NCEIAccessClient()

    total_processed = 0

    with get_db_session() as session:
        for city in cities:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {city.upper()}")
            logger.info(f"{'='*60}")

            # Get or create checkpoint
            checkpoint_id = None
            city_start = start_date

            if args.resume:
                checkpoint = get_or_create_checkpoint(session, f"settlement_multi/{city}", city)
                checkpoint_id = checkpoint.id

                if checkpoint.last_processed_date:
                    city_start = checkpoint.last_processed_date + timedelta(days=1)
                    logger.info(f"Resuming from {city_start}")

                    if city_start > end_date:
                        logger.info(f"Already completed {city}")
                        continue
            else:
                checkpoint = get_or_create_checkpoint(session, f"settlement_multi/{city}", city)
                checkpoint_id = checkpoint.id

            try:
                processed = ingest_settlement_for_city(
                    session=session,
                    city=city,
                    start_date=city_start,
                    end_date=end_date,
                    iem_client=iem_client,
                    ncei_client=ncei_client,
                    checkpoint_id=checkpoint_id,
                )
                total_processed += processed

                # Mark checkpoint complete
                if checkpoint_id:
                    complete_checkpoint(session, checkpoint_id, "completed")

            except Exception as e:
                logger.error(f"Error processing {city}: {e}", exc_info=True)
                if checkpoint_id:
                    update_checkpoint(session, checkpoint_id, error=str(e))
                    complete_checkpoint(session, checkpoint_id, "failed")
                continue

    logger.info(f"\n{'='*60}")
    logger.info(f"COMPLETE: Processed {total_processed} total records")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
