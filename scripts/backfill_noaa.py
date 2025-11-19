#!/usr/bin/env python3
"""
Backfill NOAA official TMAX (settlement) data for all cities.

Fetches daily observed maximum temperature from NCEI Access Data Service
and loads into weather_observed table using idempotent upserts.

Kalshi markets settle to these official NOAA readings (not Visual Crossing).

Usage:
    python scripts/backfill_noaa.py --start-date 2025-08-05 --end-date 2025-11-12 --cities all
    python scripts/backfill_noaa.py --start-date 2025-11-10 --end-date 2025-11-12 --cities chicago
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather.noaa_ads import NOAAClient, CITY_STATIONS
from db.connection import get_session
from db.loaders import upsert_weather

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill NOAA official TMAX settlement data"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--cities",
        type=str,
        default="all",
        help="Comma-separated city names or 'all' (default: all)",
    )
    return parser.parse_args()


def backfill_city(
    client: NOAAClient,
    city_name: str,
    station_id: str,
    start_date: str,
    end_date: str,
) -> Dict[str, int]:
    """
    Backfill NOAA TMAX data for a single city.

    Args:
        client: NOAA client
        city_name: City name (for logging)
        station_id: GHCND station ID (e.g., "GHCND:USW00014819")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dict with counts of loaded records
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILLING: {city_name.upper()} ({station_id})")
    logger.info(f"{'='*60}\n")

    total_records = 0

    try:
        with get_session() as session:
            logger.info(f"Fetching NOAA data from {start_date} to {end_date}...")

            # Fetch TMAX data (in Fahrenheit)
            data = client.get_daily_tmax(
                station_id,
                start_date,
                end_date,
                units="standard"  # Fahrenheit
            )

            if not data:
                logger.warning(f"No data fetched for {city_name}")
                return {"records": 0}

            logger.info(f"Fetched {len(data)} daily TMAX records")

            # Upsert each day
            for record in data:
                # Normalize date/tmax fields (NOAA ADS returns uppercase keys)
                record_date = record.get("date") or record.get("DATE")

                tmax_raw = record.get("TMAX")
                tmax_f = float(tmax_raw) if tmax_raw not in (None, "") else None
                tmax_c = (tmax_f - 32) * 5 / 9 if tmax_f is not None else None

                weather_data = {
                    "station_id": station_id,
                    "date": record_date,
                    "tmax_f": tmax_f,
                    "tmax_c": tmax_c,
                    "source": "noaa_ads",
                    "raw_json": record,
                }

                upsert_weather(session, weather_data)
                total_records += 1

            session.commit()
            logger.info(f"✓ Loaded {total_records} daily observations")

    except Exception as e:
        logger.error(f"Error backfilling {city_name}: {e}", exc_info=True)
        return {"records": 0, "error": str(e)}

    logger.info(f"\n✓ {city_name.upper()} BACKFILL COMPLETE!")
    logger.info(f"  Total records: {total_records}\n")

    return {"records": total_records}


def main():
    """Main execution function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Parse city list
    if args.cities == "all":
        cities = list(CITY_STATIONS.keys())
    else:
        cities = [c.strip() for c in args.cities.split(",")]

    # Validate cities
    for city in cities:
        if city not in CITY_STATIONS:
            logger.error(f"Unknown city: {city}. Available: {list(CITY_STATIONS.keys())}")
            sys.exit(1)

    # Initialize NOAA client
    logger.info("Initializing NOAA client...")
    client = NOAAClient()

    # Calculate date range
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    num_days = (end_dt - start_dt).days + 1

    logger.info(f"\n{'='*60}")
    logger.info(f"NOAA OFFICIAL TMAX BACKFILL")
    logger.info(f"{'='*60}\n")
    logger.info(f"Date range: {args.start_date} to {args.end_date} ({num_days} days)")
    logger.info(f"Cities: {', '.join(cities)}")
    logger.info(f"Expected records: ~{num_days * len(cities)} (1 per day per city)")
    logger.info(f"\n{'='*60}\n")

    # Backfill each city
    total_stats = {"records": 0, "cities": 0, "errors": 0}

    for city in cities:
        station_id = CITY_STATIONS[city]

        try:
            stats = backfill_city(
                client,
                city,
                station_id,
                args.start_date,
                args.end_date,
            )

            total_stats["records"] += stats.get("records", 0)
            if "error" not in stats:
                total_stats["cities"] += 1
            else:
                total_stats["errors"] += 1

        except Exception as e:
            logger.error(f"Fatal error backfilling {city}: {e}", exc_info=True)
            total_stats["errors"] += 1

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BACKFILL SUMMARY")
    logger.info(f"{'='*60}\n")
    logger.info(f"Cities processed: {total_stats['cities']}/{len(cities)}")
    logger.info(f"Total records loaded: {total_stats['records']:,}")
    logger.info(f"Errors: {total_stats['errors']}")

    # Validate data
    logger.info(f"\n{'='*60}")
    logger.info("DATA VALIDATION")
    logger.info(f"{'='*60}\n")

    with get_session() as session:
        from sqlalchemy import text

        # Check record counts
        result = session.execute(text("""
            SELECT station_id, COUNT(*) as count,
                   MIN(date)::date as earliest,
                   MAX(date)::date as latest
            FROM weather_observed
            WHERE source = 'noaa_ads'
            GROUP BY station_id
            ORDER BY station_id
        """)).fetchall()

        for row in result:
            city = next((k for k, v in CITY_STATIONS.items() if v == row[0]), "Unknown")
            logger.info(f"  {city:15} ({row[0]}): {row[1]:>3} records  [{row[2]} to {row[3]}]")

        # Check for missing dates
        logger.info(f"\nMissing dates check:")
        result = session.execute(text(f"""
            WITH expected AS (
                SELECT unnest(ARRAY{list(CITY_STATIONS.values())}) as station_id,
                       generate_series('{args.start_date}'::date, '{args.end_date}'::date, '1 day'::interval)::date as date
            ),
            actual AS (
                SELECT station_id, date
                FROM weather_observed
                WHERE source = 'noaa_ads'
            )
            SELECT e.station_id, COUNT(*) as missing_days
            FROM expected e
            LEFT JOIN actual a ON e.station_id = a.station_id AND e.date = a.date
            WHERE a.date IS NULL
            GROUP BY e.station_id
        """)).fetchall()

        if result:
            for row in result:
                city = next((k for k, v in CITY_STATIONS.items() if v == row[0]), "Unknown")
                logger.warning(f"  {city:15} ({row[0]}): {row[1]} missing days")
        else:
            logger.info(f"  ✓ No missing dates - complete coverage!")

    if total_stats["errors"] > 0:
        logger.warning(f"\n⚠ {total_stats['errors']} cities had errors")
        sys.exit(1)
    else:
        logger.info("\n✓ BACKFILL COMPLETE!\n")


if __name__ == "__main__":
    main()
