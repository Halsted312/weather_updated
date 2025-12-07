#!/usr/bin/env python3
"""
Ingest NOAA model guidance (NBM, HRRR, NDFD) for Kalshi weather cities.

This script downloads GRIB2 files from public NOAA S3 buckets, extracts
temperature forecasts at city locations, and stores scalar summaries
in wx.weather_more_apis_guidance.

Usage:
    # Backfill Austin NBM for May-Dec 2025
    python scripts/ingest_weather_more_apis_guidance.py \\
        --city austin --start 2025-05-01 --end 2025-12-01 --model nbm

    # Backfill all three models for Austin
    python scripts/ingest_weather_more_apis_guidance.py \\
        --city austin --start 2025-05-01 --end 2025-12-01 --all-models

Requirements:
    - wgrib2 must be installed and in PATH
    - Database must be running (TimescaleDB)
    - City config must exist in src.config.cities
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytz

from src.db.connection import get_db_session
from src.config.cities import CITIES, get_city
from src.weather_more_apis.ingest import ingest_guidance_for_city_date

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Set up logging
log_file = LOGS_DIR / f"noaa_guidance_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest NOAA model guidance (NBM, HRRR, NDFD)"
    )
    parser.add_argument(
        "--city", required=True, choices=list(CITIES.keys()), help="City ID"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--model",
        choices=["nbm", "hrrr", "ndfd"],
        help="Single model to ingest (or use --all-models)",
    )
    parser.add_argument(
        "--all-models", action="store_true", help="Ingest all three models"
    )
    parser.add_argument(
        "--run-hour",
        type=int,
        help="Specific UTC run hour (0-23). Default: 12Z for NBM/HRRR, 00Z for NDFD",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model and not args.all_models:
        logger.error("Must specify either --model or --all-models")
        sys.exit(1)

    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    models = ["nbm", "hrrr", "ndfd"] if args.all_models else [args.model]

    city_id = args.city
    city_obj = get_city(city_id)
    city_config = {
        "city_id": city_obj.city_id,
        "timezone": city_obj.timezone,
        "latitude": city_obj.latitude,
        "longitude": city_obj.longitude,
    }

    logger.info("="*80)
    logger.info(f"NOAA MODEL GUIDANCE INGESTION - {city_id.upper()}")
    logger.info("="*80)
    logger.info(f"Models: {', '.join(m.upper() for m in models)}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"City: {city_obj.city_id} ({city_obj.timezone})")
    logger.info(f"Location: {city_obj.latitude:.2f}, {city_obj.longitude:.2f}")
    logger.info("")

    total_days = (end_date - start_date).days + 1
    start_time = time.time()

    success_count = 0
    fail_count = 0
    failed_entries = []

    with get_db_session() as session:
        current_date = start_date
        processed = 0

        while current_date <= end_date:
            processed += 1

            # Progress indicator every 10 days
            if processed % 10 == 0 or processed == 1:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (total_days - processed) / rate if rate > 0 else 0
                eta_minutes = eta_seconds / 60
                logger.info(
                    f"Progress: {processed}/{total_days} ({100*processed/total_days:.1f}%) "
                    f"| Rate: {rate:.2f} days/sec | ETA: {eta_minutes:.1f} min"
                )

            for model in models:
                # Determine run hour
                if args.run_hour is not None:
                    run_hour = args.run_hour
                else:
                    # Default: 12Z for NBM/HRRR, 00Z for NDFD
                    run_hour = 0 if model == "ndfd" else 12

                # Run datetime is on the day before target date (T-1 run forecasts target date)
                run_date = current_date - timedelta(days=1)
                run_datetime = datetime(
                    run_date.year, run_date.month, run_date.day, run_hour, tzinfo=pytz.UTC
                )

                try:
                    success = ingest_guidance_for_city_date(
                        session, city_id, current_date, model, run_datetime, city_config
                    )

                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        failed_entries.append((current_date, model, "Extraction failed"))

                except Exception as e:
                    logger.error(f"Error ingesting {model} {city_id} {current_date}: {e}")
                    fail_count += 1
                    failed_entries.append((current_date, model, str(e)))

            current_date += timedelta(days=1)

    # Summary
    elapsed_total = time.time() - start_time
    logger.info("")
    logger.info("="*80)
    logger.info("INGESTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Processed: {processed} days in {elapsed_total/60:.1f} minutes")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Success rate: {100*success_count/(success_count+fail_count):.1f}%")

    if failed_entries:
        logger.warning(f"Failed entries ({len(failed_entries)}):")
        for date, model, error in failed_entries[:20]:  # Show first 20
            logger.warning(f"  - {date} {model}: {error}")
        if len(failed_entries) > 20:
            logger.warning(f"  ... and {len(failed_entries) - 20} more")

    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
