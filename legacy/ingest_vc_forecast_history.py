#!/usr/bin/env python3
"""
Ingest Visual Crossing historical daily forecasts.

Fetches daily forecast snapshots using the `forecastBasisDate` parameter
to retrieve what the forecast was on specific past dates.

This data enables Option-1 backtesting: "did the forecast beat the market?"

Usage:
    python scripts/ingest_vc_forecast_history.py --days 60
    python scripts/ingest_vc_forecast_history.py --start-date 2024-01-01 --end-date 2024-03-31
    python scripts/ingest_vc_forecast_history.py --city chicago --days 7
    python scripts/ingest_vc_forecast_history.py --all-history  # Fetch all (since 2022-01-01)
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITIES, EXCLUDED_VC_CITIES, get_city, get_settings
from src.db import get_db_session, WxForecastSnapshot
from src.db.checkpoint import (
    get_or_create_checkpoint,
    update_checkpoint,
    complete_checkpoint,
)
from src.weather.visual_crossing import VisualCrossingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_forecast_to_records(
    city_id: str,
    basis_date: date,
    payload: Dict[str, Any],
    horizon_days: int = 15,
) -> List[dict]:
    """
    Convert Visual Crossing forecast payload to database records.

    Args:
        city_id: City ID (e.g., "chicago")
        basis_date: The date the forecast was made
        payload: Raw API response from fetch_historical_daily_forecast
        horizon_days: Max lead days to keep (default: 15)

    Returns:
        List of dicts ready for upsert into wx.forecast_snapshot
    """
    records = []

    for day in payload.get("days", []):
        target_str = day.get("datetime")
        if not target_str:
            continue

        target_date = date.fromisoformat(target_str)
        lead_days = (target_date - basis_date).days

        # Only keep reasonable horizon (0 to horizon_days-1)
        if lead_days < 0 or lead_days >= horizon_days:
            continue

        record = {
            "city": city_id,
            "target_date": target_date,
            "basis_date": basis_date,
            "lead_days": lead_days,
            "provider": "visualcrossing",
            # Map VC API fields to model columns
            "tempmax_fcst_f": day.get("tempmax"),
            "tempmin_fcst_f": day.get("tempmin"),
            "precip_fcst_in": day.get("precip"),
            "precip_prob_fcst": day.get("precipprob"),
            "humidity_fcst": day.get("humidity"),
            "windspeed_fcst_mph": day.get("windspeed"),
            "conditions_fcst": day.get("conditions"),
            # Store full payload for future use
            "raw_json": day,
        }
        records.append(record)

    return records


def upsert_forecast_snapshots(session, records: List[dict]) -> int:
    """
    Upsert forecast snapshots into wx.forecast_snapshot.

    Returns:
        Number of rows affected
    """
    if not records:
        return 0

    stmt = insert(WxForecastSnapshot).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["city", "target_date", "basis_date"],
        set_={
            "lead_days": stmt.excluded.lead_days,
            "provider": stmt.excluded.provider,
            "tempmax_fcst_f": stmt.excluded.tempmax_fcst_f,
            "tempmin_fcst_f": stmt.excluded.tempmin_fcst_f,
            "precip_fcst_in": stmt.excluded.precip_fcst_in,
            "precip_prob_fcst": stmt.excluded.precip_prob_fcst,
            "humidity_fcst": stmt.excluded.humidity_fcst,
            "windspeed_fcst_mph": stmt.excluded.windspeed_fcst_mph,
            "conditions_fcst": stmt.excluded.conditions_fcst,
            "raw_json": stmt.excluded.raw_json,
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def ingest_city_forecasts(
    client: VisualCrossingClient,
    session,
    city_id: str,
    start_date: date,
    end_date: date,
    horizon_days: int = 15,
    checkpoint_id: int = None,
) -> int:
    """
    Fetch and upsert historical forecasts for a single city.

    Args:
        client: Visual Crossing client
        session: Database session
        city_id: City ID (e.g., "chicago")
        start_date: First basis_date to fetch
        end_date: Last basis_date to fetch
        horizon_days: Forecast horizon days (default: 15)
        checkpoint_id: Optional checkpoint ID for progress tracking

    Returns:
        Total number of forecast records upserted
    """
    city = get_city(city_id)
    location = f"stn:{city.icao}"

    total_days = (end_date - start_date).days + 1
    logger.info(f"Ingesting {city_id} ({city.icao}) forecasts: {total_days} basis dates")

    total_upserted = 0
    current_date = start_date
    processed_count = 0

    while current_date <= end_date:
        basis_str = current_date.isoformat()

        try:
            # Fetch 15-day forecast as it looked on this basis_date
            payload = client.fetch_historical_daily_forecast(
                location=location,
                basis_date=basis_str,
                horizon_days=horizon_days,
            )

            # Convert to DB records
            records = parse_forecast_to_records(
                city_id=city_id,
                basis_date=current_date,
                payload=payload,
                horizon_days=horizon_days,
            )

            # Upsert batch
            if records:
                rows = upsert_forecast_snapshots(session, records)
                total_upserted += rows

            processed_count += 1

            # Progress logging every 50 basis dates
            if processed_count % 50 == 0:
                logger.info(
                    f"  {city_id}: {processed_count}/{total_days} basis dates, "
                    f"{total_upserted} forecast rows"
                )
                session.commit()

                # Update checkpoint
                if checkpoint_id:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint_id,
                        last_date=current_date,
                        processed_count=total_upserted,
                    )
                    session.commit()

        except Exception as e:
            logger.error(f"Error fetching forecast for {city_id} basis={basis_str}: {e}")
            # Continue with next date instead of failing completely
            pass

        current_date += timedelta(days=1)

    return total_upserted


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Visual Crossing historical daily forecasts"
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="Number of basis days to ingest (default: 60)"
    )
    parser.add_argument(
        "--start-date", type=str,
        help="Start basis date (YYYY-MM-DD), overrides --days"
    )
    parser.add_argument(
        "--end-date", type=str,
        help="End basis date (YYYY-MM-DD), default: yesterday"
    )
    parser.add_argument(
        "--all-history", action="store_true",
        help="Fetch all available history (since 2022-01-01)"
    )
    parser.add_argument(
        "--city", type=str,
        help="Single city to ingest (default: all)"
    )
    parser.add_argument(
        "--horizon-days", type=int, default=15,
        help="Forecast horizon days (default: 15)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't write to database"
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="Disable checkpoint tracking"
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.05,
        help="Delay between API calls in seconds (default: 0.05)"
    )

    args = parser.parse_args()

    # Calculate date range (use yesterday as default end to ensure forecasts are finalized)
    if args.end_date:
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    if args.all_history:
        # Visual Crossing historical forecasts available since ~2020
        # We use 2022-01-01 to match our settlement data range
        start_date = date(2022, 1, 1)
        logger.info("Fetching ALL available forecast history (since 2022-01-01)")
    elif args.start_date:
        start_date = date.fromisoformat(args.start_date)
    else:
        start_date = end_date - timedelta(days=args.days - 1)

    logger.info(f"Ingesting VC historical forecasts from {start_date} to {end_date}")
    logger.info(f"Forecast horizon: {args.horizon_days} days")

    # Get settings and create client
    settings = get_settings()
    client = VisualCrossingClient(
        api_key=settings.vc_api_key,
        base_url=settings.vc_base_url,
        rate_limit_delay=args.rate_limit,
    )

    # Determine which cities to process
    if args.city:
        if args.city in EXCLUDED_VC_CITIES:
            logger.error(f"{args.city} is excluded from Visual Crossing")
            return
        cities = [args.city]
    else:
        cities = [c for c in CITIES.keys() if c not in EXCLUDED_VC_CITIES]

    logger.info(f"Processing cities: {cities}")

    # Calculate expected totals
    total_basis_days = (end_date - start_date).days + 1
    expected_rows = len(cities) * total_basis_days * args.horizon_days
    expected_api_calls = len(cities) * total_basis_days
    logger.info(f"Expected: ~{expected_api_calls:,} API calls, ~{expected_rows:,} rows")

    if args.dry_run:
        logger.info("DRY RUN - would fetch but not write to database")
        for city_id in cities:
            city = get_city(city_id)
            logger.info(
                f"Would ingest {city_id} ({city.icao}) forecasts "
                f"from {start_date} to {end_date}"
            )
        return

    # Ingest each city with checkpoint tracking
    total_records = 0
    with get_db_session() as session:
        for city_id in cities:
            # Get or create checkpoint for this city
            checkpoint = None
            checkpoint_id = None
            if not args.no_checkpoint:
                checkpoint = get_or_create_checkpoint(
                    session=session,
                    pipeline_name="vc_forecast_history",
                    city=city_id,
                )
                checkpoint_id = checkpoint.id
                session.commit()

            try:
                count = ingest_city_forecasts(
                    client=client,
                    session=session,
                    city_id=city_id,
                    start_date=start_date,
                    end_date=end_date,
                    horizon_days=args.horizon_days,
                    checkpoint_id=checkpoint_id,
                )
                total_records += count
                logger.info(f"{city_id}: upserted {count} forecast records")

                # Final commit for this city
                session.commit()

                # Mark checkpoint completed for this city
                if checkpoint_id:
                    complete_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint_id,
                        status="completed",
                    )
                    session.commit()

            except Exception as e:
                logger.error(f"Error ingesting forecasts for {city_id}: {e}")
                if checkpoint_id:
                    update_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint_id,
                        error=str(e),
                    )
                    complete_checkpoint(
                        session=session,
                        checkpoint_id=checkpoint_id,
                        status="failed",
                    )
                    session.commit()
                session.rollback()
                raise

    logger.info(f"Forecast ingestion complete: {total_records:,} total records upserted")


if __name__ == "__main__":
    main()
