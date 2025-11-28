#!/usr/bin/env python3
"""
Ingest Visual Crossing historical hourly forecasts.

Fetches hourly forecast curves (72 hours / 3 days) using the `forecastBasisDate`
parameter to retrieve what the forecast looked like on specific past dates.

This data enables trend analysis: comparing forecasted temperature curves
against actual observations and Kalshi market prices.

Usage:
    python scripts/ingest_vc_forecast_hourly.py --days 60
    python scripts/ingest_vc_forecast_hourly.py --start-date 2024-01-01 --end-date 2024-03-31
    python scripts/ingest_vc_forecast_hourly.py --city chicago --days 7
    python scripts/ingest_vc_forecast_hourly.py --all-history  # Fetch all (since 2022-01-01)
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
from src.db import get_db_session, WxForecastSnapshotHourly
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

# Configurable horizons (per design decision)
MAX_HOUR_HORIZON = 72  # 3 days of hourly data
MAX_DAY_HORIZON = 7    # 7 days of daily data (used in API call)


def parse_hourly_forecast_to_records(
    city_id: str,
    tz_name: str,
    basis_date: date,
    payload: Dict[str, Any],
    max_hours: int = MAX_HOUR_HORIZON,
) -> List[dict]:
    """
    Convert Visual Crossing hourly forecast payload to database records.

    Args:
        city_id: City ID (e.g., "chicago")
        tz_name: IANA timezone (e.g., "America/Chicago")
        basis_date: The date the forecast was made
        payload: Raw API response from fetch_historical_hourly_forecast
        max_hours: Max lead hours to keep (default: 72)

    Returns:
        List of dicts ready for upsert into wx.forecast_snapshot_hourly
    """
    records = []
    hour_count = 0

    # Track seen local hours to handle DST fall-back duplicates
    # When DST ends, 1:00 AM occurs twice - we keep the first and skip the second
    seen_local_hours: set = set()

    for day in payload.get("days", []):
        hours = day.get("hours", [])

        for hour in hours:
            if hour_count >= max_hours:
                break

            datetime_str = hour.get("datetime")
            epoch = hour.get("datetimeEpoch")

            if not datetime_str or not epoch:
                continue

            # Parse local datetime (VC returns local time by default)
            # Format: "HH:00:00" within the day context
            day_date = day.get("datetime")  # "YYYY-MM-DD"
            hour_time = datetime_str  # "HH:00:00"
            target_hour_local = datetime.fromisoformat(f"{day_date}T{hour_time}")

            # Skip duplicate local hour (DST fall-back)
            # This keeps the first 1:00 AM and drops the second during fall-back
            if target_hour_local in seen_local_hours:
                logger.debug(f"Skipping duplicate local hour {target_hour_local} (DST fall-back)")
                continue
            seen_local_hours.add(target_hour_local)

            record = {
                "city": city_id,
                "target_hour_local": target_hour_local,
                "target_hour_epoch": epoch,
                "basis_date": basis_date,
                "lead_hours": hour_count,
                "provider": "visualcrossing",
                "tz_name": tz_name,
                # Hourly forecast values
                "temp_fcst_f": hour.get("temp"),
                "feelslike_fcst_f": hour.get("feelslike"),
                "humidity_fcst": hour.get("humidity"),
                "precip_fcst_in": hour.get("precip"),
                "precip_prob_fcst": hour.get("precipprob"),
                "windspeed_fcst_mph": hour.get("windspeed"),
                "conditions_fcst": hour.get("conditions"),
                # Store full payload for future use
                "raw_json": hour,
            }
            records.append(record)
            hour_count += 1

        if hour_count >= max_hours:
            break

    return records


def upsert_hourly_forecasts(session, records: List[dict]) -> int:
    """
    Upsert hourly forecast snapshots into wx.forecast_snapshot_hourly.

    Returns:
        Number of rows affected
    """
    if not records:
        return 0

    stmt = insert(WxForecastSnapshotHourly).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["city", "target_hour_local", "basis_date"],
        set_={
            "target_hour_epoch": stmt.excluded.target_hour_epoch,
            "lead_hours": stmt.excluded.lead_hours,
            "provider": stmt.excluded.provider,
            "tz_name": stmt.excluded.tz_name,
            "temp_fcst_f": stmt.excluded.temp_fcst_f,
            "feelslike_fcst_f": stmt.excluded.feelslike_fcst_f,
            "humidity_fcst": stmt.excluded.humidity_fcst,
            "precip_fcst_in": stmt.excluded.precip_fcst_in,
            "precip_prob_fcst": stmt.excluded.precip_prob_fcst,
            "windspeed_fcst_mph": stmt.excluded.windspeed_fcst_mph,
            "conditions_fcst": stmt.excluded.conditions_fcst,
            "raw_json": stmt.excluded.raw_json,
        },
    )

    result = session.execute(stmt)
    return result.rowcount


def ingest_city_hourly_forecasts(
    client: VisualCrossingClient,
    session,
    city_id: str,
    start_date: date,
    end_date: date,
    max_hours: int = MAX_HOUR_HORIZON,
    max_days: int = MAX_DAY_HORIZON,
    checkpoint_id: int = None,
) -> int:
    """
    Fetch and upsert historical hourly forecasts for a single city.

    Args:
        client: Visual Crossing client
        session: Database session
        city_id: City ID (e.g., "chicago")
        start_date: First basis_date to fetch
        end_date: Last basis_date to fetch
        max_hours: Hourly horizon (default: 72)
        max_days: Daily horizon for API (default: 7)
        checkpoint_id: Optional checkpoint ID for progress tracking

    Returns:
        Total number of hourly forecast records upserted
    """
    city = get_city(city_id)
    location = f"stn:{city.icao}"
    tz_name = city.timezone

    total_days = (end_date - start_date).days + 1
    logger.info(f"Ingesting {city_id} ({city.icao}) hourly forecasts: {total_days} basis dates")

    total_upserted = 0
    current_date = start_date
    processed_count = 0

    while current_date <= end_date:
        basis_str = current_date.isoformat()

        try:
            # Fetch hourly + daily forecast as it looked on this basis_date
            payload = client.fetch_historical_hourly_forecast(
                location=location,
                basis_date=basis_str,
                horizon_hours=max_hours,
                horizon_days=max_days,
            )

            # Convert to DB records (hourly only)
            records = parse_hourly_forecast_to_records(
                city_id=city_id,
                tz_name=tz_name,
                basis_date=current_date,
                payload=payload,
                max_hours=max_hours,
            )

            # Upsert batch
            if records:
                rows = upsert_hourly_forecasts(session, records)
                total_upserted += rows

            processed_count += 1

            # Progress logging every 50 basis dates
            if processed_count % 50 == 0:
                logger.info(
                    f"  {city_id}: {processed_count}/{total_days} basis dates, "
                    f"{total_upserted} hourly rows"
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
            logger.error(f"Error fetching hourly forecast for {city_id} basis={basis_str}: {e}")
            # Continue with next date instead of failing completely
            pass

        current_date += timedelta(days=1)

    return total_upserted


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Visual Crossing historical hourly forecasts"
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
        "--max-hours", type=int, default=MAX_HOUR_HORIZON,
        help=f"Max hourly horizon (default: {MAX_HOUR_HORIZON})"
    )
    parser.add_argument(
        "--max-days", type=int, default=MAX_DAY_HORIZON,
        help=f"Max daily horizon for API (default: {MAX_DAY_HORIZON})"
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
        logger.info("Fetching ALL available hourly forecast history (since 2022-01-01)")
    elif args.start_date:
        start_date = date.fromisoformat(args.start_date)
    else:
        start_date = end_date - timedelta(days=args.days - 1)

    logger.info(f"Ingesting VC hourly forecasts from {start_date} to {end_date}")
    logger.info(f"Hourly horizon: {args.max_hours}h, Daily horizon: {args.max_days}d")

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
    expected_rows = len(cities) * total_basis_days * args.max_hours
    expected_api_calls = len(cities) * total_basis_days
    logger.info(f"Expected: ~{expected_api_calls:,} API calls, ~{expected_rows:,} hourly rows")

    if args.dry_run:
        logger.info("DRY RUN - would fetch but not write to database")
        for city_id in cities:
            city = get_city(city_id)
            logger.info(
                f"Would ingest {city_id} ({city.icao}) hourly forecasts "
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
                    pipeline_name="vc_forecast_hourly",
                    city=city_id,
                )
                checkpoint_id = checkpoint.id
                session.commit()

            try:
                count = ingest_city_hourly_forecasts(
                    client=client,
                    session=session,
                    city_id=city_id,
                    start_date=start_date,
                    end_date=end_date,
                    max_hours=args.max_hours,
                    max_days=args.max_days,
                    checkpoint_id=checkpoint_id,
                )
                total_records += count
                logger.info(f"{city_id}: upserted {count} hourly forecast records")

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
                logger.error(f"Error ingesting hourly forecasts for {city_id}: {e}")
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

    logger.info(f"Hourly forecast ingestion complete: {total_records:,} total records upserted")


if __name__ == "__main__":
    main()
