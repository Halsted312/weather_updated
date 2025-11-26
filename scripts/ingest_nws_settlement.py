#!/usr/bin/env python3
"""
Ingest NWS settlement temperature data (CLI and CF6).

Fetches official TMAX values from NWS and upserts into wx.settlement.
Uses cascading sources: CLI (final) > CF6 (preliminary) > ADS (historical).

CLI is the official settlement source for Kalshi weather markets.
CF6 provides preliminary values that can be used before CLI is available.

Supports resume-on-crash via checkpoints in meta.ingestion_checkpoint.

Usage:
    python scripts/ingest_nws_settlement.py                    # Today's data
    python scripts/ingest_nws_settlement.py --date 2025-11-25  # Specific date
    python scripts/ingest_nws_settlement.py --city chicago     # Single city
    python scripts/ingest_nws_settlement.py --all-cities       # All cities
    python scripts/ingest_nws_settlement.py --start-date 2024-01-01 --end-date 2024-12-31  # Range
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, get_city, get_settings
from src.db import get_db_session, WxSettlement
from src.db.checkpoint import (
    get_or_create_checkpoint,
    update_checkpoint,
    complete_checkpoint,
)
from src.weather.nws_cli import NWSCliClient, SETTLEMENT_STATIONS
from src.weather.nws_cf6 import NWSCF6Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def result_to_db_dict(city_id: str, result: Dict[str, Any]) -> dict:
    """
    Convert NWS fetch result to database record.

    Args:
        city_id: City ID
        result: Dict from NWSCliClient or NWSCF6Client

    Returns:
        Dict ready for upsert into wx.settlement
    """
    source = result.get("source", "unknown").lower()

    # Map to column based on source
    record = {
        "city": city_id,
        "date_local": result["date_local"],
        "tmax_cli_f": int(result["tmax_f"]) if source == "cli" else None,
        "tmax_cf6_f": int(result["tmax_f"]) if source == "cf6" else None,
        "tmax_ads_f": None,  # Would be populated from NOAA ADS if needed
        "tmax_final": int(result["tmax_f"]),
        "source_final": source,
        "raw_payload": {
            "source": source,
            "tmax_f": result["tmax_f"],
            "is_preliminary": result.get("is_preliminary", False),
            "icao": result.get("icao"),
            "issuedby": result.get("issuedby"),
            "ghcnd": result.get("ghcnd"),
        },
    }

    return record


def upsert_settlement(session, records: List[dict]) -> int:
    """
    Upsert settlement records into wx.settlement.

    Logic for updates:
    - If source is CLI, always update (it's final)
    - If source is CF6, only update if no CLI value exists

    Returns:
        Number of rows affected
    """
    if not records:
        return 0

    total_rows = 0

    for record in records:
        stmt = insert(WxSettlement).values(record)

        # Smart upsert: CLI overwrites everything, CF6 only fills gaps
        source = record["source_final"]

        if source == "cli":
            # CLI is authoritative - update everything
            stmt = stmt.on_conflict_do_update(
                index_elements=["city", "date_local"],
                set_={
                    "tmax_cli_f": stmt.excluded.tmax_cli_f,
                    "tmax_final": stmt.excluded.tmax_final,
                    "source_final": stmt.excluded.source_final,
                    "raw_payload": stmt.excluded.raw_payload,
                    "updated_at": text("NOW()"),
                },
            )
        else:
            # CF6/ADS - only update if no CLI value exists
            stmt = stmt.on_conflict_do_update(
                index_elements=["city", "date_local"],
                set_={
                    "tmax_cf6_f": stmt.excluded.tmax_cf6_f,
                    # Only update final if CLI is not set
                    "tmax_final": text(
                        "CASE WHEN wx.settlement.tmax_cli_f IS NOT NULL "
                        "THEN wx.settlement.tmax_final "
                        "ELSE EXCLUDED.tmax_final END"
                    ),
                    "source_final": text(
                        "CASE WHEN wx.settlement.tmax_cli_f IS NOT NULL "
                        "THEN wx.settlement.source_final "
                        "ELSE EXCLUDED.source_final END"
                    ),
                    "raw_payload": text(
                        "CASE WHEN wx.settlement.tmax_cli_f IS NOT NULL "
                        "THEN wx.settlement.raw_payload "
                        "ELSE EXCLUDED.raw_payload END"
                    ),
                    "updated_at": text("NOW()"),
                },
            )

        result = session.execute(stmt)
        total_rows += result.rowcount

    return total_rows


def fetch_settlement_for_city(
    cli_client: NWSCliClient,
    cf6_client: NWSCF6Client,
    city_id: str,
    target_date: date,
) -> Optional[Dict[str, Any]]:
    """
    Fetch settlement TMAX for a city, trying CLI first then CF6.

    Args:
        cli_client: NWS CLI client
        cf6_client: NWS CF6 client
        city_id: City ID
        target_date: Target date

    Returns:
        Settlement result dict, or None if neither source available
    """
    logger.info(f"Fetching settlement for {city_id} on {target_date}")

    # Try CLI first (final values)
    try:
        cli_result = cli_client.get_tmax_for_city(city_id, target_date)
        if cli_result:
            logger.info(f"  CLI: {cli_result['tmax_f']}F (final)")
            return cli_result
        else:
            logger.debug(f"  CLI: not available")
    except Exception as e:
        logger.error(f"  CLI error: {e}")

    # Fall back to CF6 (preliminary)
    try:
        cf6_result = cf6_client.get_tmax_for_city(city_id, target_date)
        if cf6_result:
            logger.info(f"  CF6: {cf6_result['tmax_f']}F (preliminary)")
            return cf6_result
        else:
            logger.debug(f"  CF6: not available")
    except Exception as e:
        logger.error(f"  CF6 error: {e}")

    logger.warning(f"  No settlement data available for {city_id} on {target_date}")
    return None


def ingest_settlement(
    session,
    cli_client: NWSCliClient,
    cf6_client: NWSCF6Client,
    city_ids: List[str],
    target_date: date,
) -> int:
    """
    Ingest settlement data for multiple cities.

    Returns:
        Number of records upserted
    """
    records = []

    for city_id in city_ids:
        result = fetch_settlement_for_city(
            cli_client, cf6_client, city_id, target_date
        )

        if result:
            record = result_to_db_dict(city_id, result)
            records.append(record)

    if not records:
        logger.warning(f"No settlement data found for {target_date}")
        return 0

    # Upsert all records
    rows = upsert_settlement(session, records)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Ingest NWS settlement temperature data")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD), default: yesterday")
    parser.add_argument("--start-date", type=str, help="Start date for range backfill (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for range backfill (YYYY-MM-DD)")
    parser.add_argument("--city", type=str, help="Single city to ingest")
    parser.add_argument("--all-cities", action="store_true", help="Ingest all cities")
    parser.add_argument("--days-back", type=int, default=1, help="Number of days back from today")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint tracking")

    args = parser.parse_args()

    # Determine date range
    if args.start_date and args.end_date:
        # Range mode
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        dates_to_process = []
        current = start_date
        while current <= end_date:
            dates_to_process.append(current)
            current += timedelta(days=1)
        logger.info(f"Processing date range: {start_date} to {end_date} ({len(dates_to_process)} days)")
    elif args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
        dates_to_process = [target_date]
    else:
        # Default to yesterday (CLI is posted next morning)
        target_date = date.today() - timedelta(days=args.days_back)
        dates_to_process = [target_date]

    logger.info(f"Target date(s): {dates_to_process[0]} to {dates_to_process[-1]}")

    # Determine which cities to process
    if args.city:
        if args.city not in SETTLEMENT_STATIONS:
            logger.error(f"Unknown city: {args.city}. Available: {list(SETTLEMENT_STATIONS.keys())}")
            return
        city_ids = [args.city]
    else:
        city_ids = list(SETTLEMENT_STATIONS.keys())

    logger.info(f"Processing cities: {city_ids}")

    # Create clients
    cli_client = NWSCliClient()
    cf6_client = NWSCF6Client()

    if args.dry_run:
        logger.info("DRY RUN - not writing to database")
        for target_date in dates_to_process[:3]:  # Sample first 3 dates
            for city_id in city_ids:
                result = fetch_settlement_for_city(
                    cli_client, cf6_client, city_id, target_date
                )
                if result:
                    logger.info(f"  Would insert: {city_id} {target_date} -> {result['tmax_f']}F ({result['source']})")
        return

    # Ingest with checkpoint tracking
    total_rows = 0
    with get_db_session() as session:
        # Get or create checkpoint
        checkpoint = None
        if not args.no_checkpoint:
            checkpoint = get_or_create_checkpoint(
                session=session,
                pipeline_name="nws_settlement",
                city=args.city,
            )
            session.commit()

        try:
            for target_date in dates_to_process:
                rows = ingest_settlement(
                    session=session,
                    cli_client=cli_client,
                    cf6_client=cf6_client,
                    city_ids=city_ids,
                    target_date=target_date,
                )
                total_rows += rows

                # Update checkpoint periodically
                if checkpoint and rows > 0:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint.id,
                        last_date=target_date,
                        processed_count=rows,
                    )
                session.commit()

                if len(dates_to_process) > 1:
                    logger.info(f"  {target_date}: {rows} records")

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

        logger.info(f"Upserted {total_rows} total settlement records")


if __name__ == "__main__":
    main()
