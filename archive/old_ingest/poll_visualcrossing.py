#!/usr/bin/env python3
"""
Real-time Visual Crossing weather poller.

Polls every minute for today's weather data across all stations,
upserts new minutes, and periodically refreshes the 1-minute grid.

Usage:
    python ingest/poll_visualcrossing.py
    nohup python ingest/poll_visualcrossing.py > /tmp/wx_poller.log 2>&1 &
"""

import os
import sys
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather.visual_crossing import VisualCrossingClient, STATION_MAP
from weather.time_utils import utc_now
from db.connection import get_session
from db.loaders import upsert_wx_location, bulk_upsert_wx_minutes, refresh_1m_grid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def poll_city(
    client: VisualCrossingClient,
    city_name: str,
    loc_id: str,
    vc_key: str,
) -> Dict[str, Any]:
    """
    Poll weather data for a single city (today's data).

    Args:
        client: Visual Crossing client
        city_name: City name
        loc_id: Location ID (e.g., "KMDW")
        vc_key: Visual Crossing location key (e.g., "stn:KMDW")

    Returns:
        Dict with counts of loaded records
    """
    try:
        # Get today's date
        today = utc_now().date().strftime("%Y-%m-%d")

        # Fetch today's data
        df = client.fetch_day_for_station(loc_id, vc_key, today)

        if df.empty:
            logger.debug(f"{city_name}: No new data")
            return {"city": city_name, "records": 0}

        # Upsert to database (each city gets own session)
        with get_session() as session:
            # Ensure location exists
            upsert_wx_location(session, loc_id, vc_key, city_name)

            # Upsert minute observations
            count = bulk_upsert_wx_minutes(session, loc_id, df)
            session.commit()

        logger.info(f"{city_name}: {count} records")
        return {"city": city_name, "records": count}

    except Exception as e:
        logger.error(f"{city_name}: Error - {e}")
        return {"city": city_name, "records": 0, "error": str(e)}


def run_poll_cycle(client: VisualCrossingClient) -> Dict[str, int]:
    """
    Run one poll cycle for all cities (parallel).

    Args:
        client: Visual Crossing client

    Returns:
        Dict with total stats
    """
    total_stats = {"records": 0, "cities": 0, "errors": 0}

    # Process cities in parallel (4 workers for balance)
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all city poll tasks
        future_to_city = {}
        for city_name, (loc_id, vc_key) in STATION_MAP.items():
            future = executor.submit(poll_city, client, city_name, loc_id, vc_key)
            future_to_city[future] = city_name

        # Collect results as they complete
        for future in as_completed(future_to_city):
            city_name = future_to_city[future]
            try:
                result = future.result()
                total_stats["records"] += result.get("records", 0)
                if "error" not in result:
                    total_stats["cities"] += 1
                else:
                    total_stats["errors"] += 1
            except Exception as e:
                logger.error(f"Error processing {city_name}: {e}")
                total_stats["errors"] += 1

    return total_stats


def main():
    """Main execution - runs continuously every minute."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("VC_API_KEY")
    if not api_key:
        logger.error("VC_API_KEY not found in environment")
        sys.exit(1)

    # Initialize Visual Crossing client
    logger.info("Initializing Visual Crossing poller...")
    client = VisualCrossingClient(api_key=api_key)

    logger.info("VISUAL CROSSING REAL-TIME POLLER STARTED")
    logger.info("Polling every 60 seconds. Press Ctrl+C to stop.\n")

    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            logger.info(f"=== CYCLE {cycle_count} - {utc_now().isoformat()} ===")

            start_time = time.time()

            # Run poll cycle
            stats = run_poll_cycle(client)

            elapsed = time.time() - start_time

            logger.info(
                f"Cycle complete: {stats['cities']}/{len(STATION_MAP)} cities, "
                f"{stats['records']} records, "
                f"{stats['errors']} errors ({elapsed:.1f}s)"
            )

            # Every 5 cycles (5 minutes), refresh materialized view
            if cycle_count % 5 == 0:
                logger.info("Refreshing 1-minute grid...")
                try:
                    with get_session() as session:
                        refresh_1m_grid(session)
                        session.commit()
                    logger.info("✓ Grid refreshed")
                except Exception as e:
                    logger.error(f"Error refreshing grid: {e}")

            # Wait until next cycle (60 seconds)
            sleep_time = max(0, 60 - elapsed)
            if sleep_time > 0:
                logger.info(f"Sleeping {sleep_time:.1f}s until next cycle...\n")
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\n\nStopping Visual Crossing poller...")
        logger.info(f"Completed {cycle_count} cycles")


if __name__ == "__main__":
    main()
