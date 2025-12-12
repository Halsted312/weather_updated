#!/usr/bin/env python3
"""
FAST parallel ingestion of NOAA model guidance using threading.

This version parallelizes both:
1. Across days (process multiple days concurrently)
2. Within each day (download 6 peak-hour GRIBs concurrently)

Target: 200+ days in 2-5 minutes instead of 1 hour.

Usage:
    python scripts/ingest_weather_more_apis_guidance_FAST.py \\
        --city austin --start 2025-05-01 --end 2025-12-01 --model nbm --workers 16
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pytz

from src.db.connection import get_db_session
from src.config.cities import CITIES, get_city
from src.weather_more_apis.ingest import ingest_guidance_for_city_date

# Logging setup
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

log_file = LOGS_DIR / f"noaa_guidance_fast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def ingest_single_day(city_id, target_date, model, city_config, run_hour):
    """Worker function to ingest one day (thread-safe with separate session)."""
    run_date = target_date - timedelta(days=1)
    run_datetime = datetime(
        run_date.year, run_date.month, run_date.day, run_hour, tzinfo=pytz.UTC
    )

    with get_db_session() as session:
        try:
            success = ingest_guidance_for_city_date(
                session, city_id, target_date, model, run_datetime, city_config
            )
            return (target_date, model, success, None)
        except Exception as e:
            logger.error(f"Error {model} {target_date}: {e}")
            return (target_date, model, False, str(e))


def parse_args():
    parser = argparse.ArgumentParser(description="FAST parallel NOAA guidance ingestion")
    parser.add_argument("--city", required=True, choices=list(CITIES.keys()))
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--model", choices=["nbm", "hrrr", "ndfd"], help="Single model")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers (default: 16)")
    parser.add_argument("--run-hour", type=int, help="UTC run hour (default: 12 for NBM/HRRR, 0 for NDFD)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model and not args.all_models:
        logger.error("Specify --model or --all-models")
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

    total_days = (end_date - start_date).days + 1

    logger.info("="*80)
    logger.info(f"FAST NOAA INGESTION - {city_id.upper()}")
    logger.info("="*80)
    logger.info(f"Models: {', '.join(m.upper() for m in models)}")
    logger.info(f"Date range: {start_date} to {end_date} ({total_days} days)")
    logger.info(f"Workers: {args.workers}")
    logger.info("")

    # Build task list (all date+model combinations)
    tasks = []
    current = start_date
    while current <= end_date:
        for model in models:
            run_hour = args.run_hour if args.run_hour is not None else (0 if model == "ndfd" else 12)
            tasks.append((city_id, current, model, city_config, run_hour))
        current += timedelta(days=1)

    logger.info(f"Total tasks: {len(tasks)} ({total_days} days × {len(models)} models)")

    start_time = time.time()
    success_count = 0
    fail_count = 0
    failed_entries = []

    # Parallel execution
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(ingest_single_day, *task): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(futures):
            target_date, model, success, error = future.result()
            completed += 1

            if success:
                success_count += 1
            else:
                fail_count += 1
                failed_entries.append((target_date, model, error or "Unknown"))

            # Progress every 10 completions
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%) "
                    f"| Rate: {rate:.1f} tasks/sec | ETA: {eta:.0f}s"
                )

    # Summary
    elapsed_total = time.time() - start_time
    logger.info("")
    logger.info("="*80)
    logger.info("INGESTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total:.0f} seconds)")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Success rate: {100*success_count/(success_count+fail_count):.1f}%")
    logger.info(f"Throughput: {success_count/elapsed_total:.2f} tasks/second")

    if failed_entries:
        logger.warning(f"Failed ({len(failed_entries)}):")
        for date, model, error in failed_entries[:10]:
            logger.warning(f"  {date} {model}: {error}")
        if len(failed_entries) > 10:
            logger.warning(f"  ... +{len(failed_entries)-10} more")

    logger.info(f"Log: {log_file}")


if __name__ == "__main__":
    main()
