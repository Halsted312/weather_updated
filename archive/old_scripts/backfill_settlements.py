#!/usr/bin/env python3
"""
Backfill settlement data for all cities using NWS CLI, CF6, and GHCND.

Precedence (official Kalshi settlement source):
1. CLI (Daily Climate Report) - final, official
2. CF6 (WS Form F-6) - preliminary monthly table
3. GHCND (NCEI Access Data Service) - audit/backfill only

Usage:
    python scripts/backfill_settlements.py --start-date 2025-08-05 --end-date 2025-11-12 --cities all
    python scripts/backfill_settlements.py --start-date 2025-11-10 --end-date 2025-11-12 --cities chicago
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather.nws_cli import NWSCliClient, SETTLEMENT_STATIONS
from weather.nws_cf6 import NWSCF6Client
from weather.noaa_ads import NOAAClient
from db.connection import get_session
from db.loaders import upsert_settlement

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill settlement data (CLI → CF6 → GHCND)"
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


def fetch_settlement_with_precedence(
    cli_client: NWSCliClient,
    cf6_client: NWSCF6Client,
    noaa_client: NOAAClient,
    city: str,
    target_date: date,
) -> Optional[Dict[str, Any]]:
    """
    Fetch settlement for a city/date using precedence logic.

    Precedence: CLI (final) → CF6 (preliminary) → GHCND (audit)

    Args:
        cli_client: NWS CLI client
        cf6_client: NWS CF6 client
        noaa_client: NOAA GHCND client
        city: City name
        target_date: Target date

    Returns:
        Settlement dict, or None if no data found from any source
    """
    station = SETTLEMENT_STATIONS[city]

    # Try CLI first (official, final)
    logger.info(f"  Trying CLI for {target_date}...")
    try:
        cli_result = cli_client.get_tmax_for_city(city, target_date)
        if cli_result:
            logger.info(f"  ✓ CLI: {cli_result['tmax_f']}°F")
            return {
                "city": city,
                "icao": cli_result["icao"],
                "issuedby": cli_result["issuedby"],
                "date_local": cli_result["date_local"],
                "tmax_f": cli_result["tmax_f"],
                "source": "CLI",
                "is_preliminary": False,
                "raw_payload": cli_result["raw_html"],
            }
        else:
            logger.info("  ✗ CLI not available")
    except Exception as e:
        logger.warning(f"  ✗ CLI error: {e}")

    # Try CF6 (preliminary)
    logger.info(f"  Trying CF6 for {target_date}...")
    try:
        cf6_result = cf6_client.get_tmax_for_city(city, target_date)
        if cf6_result:
            logger.info(f"  ✓ CF6: {cf6_result['tmax_f']}°F (preliminary)")
            return {
                "city": city,
                "icao": cf6_result["icao"],
                "issuedby": cf6_result["issuedby"],
                "date_local": cf6_result["date_local"],
                "tmax_f": cf6_result["tmax_f"],
                "source": "CF6",
                "is_preliminary": True,
                "raw_payload": cf6_result["raw_html"],
            }
        else:
            logger.info("  ✗ CF6 not available")
    except Exception as e:
        logger.warning(f"  ✗ CF6 error: {e}")

    # Fall back to GHCND (audit/backfill)
    logger.info(f"  Trying GHCND for {target_date}...")
    try:
        ghcnd_station = station["ghcnd"]
        date_str = target_date.isoformat()

        # GHCND expects date range, so we fetch single day
        data = noaa_client.get_daily_tmax(
            ghcnd_station,
            date_str,
            date_str,
            units="standard"  # Fahrenheit
        )

        if data and len(data) > 0:
            record = data[0]
            tmax_f = record.get("TMAX")

            if tmax_f is not None:
                logger.info(f"  ✓ GHCND: {tmax_f}°F (audit)")
                return {
                    "city": city,
                    "icao": station["icao"],
                    "issuedby": station["issuedby"],
                    "date_local": target_date,
                    "tmax_f": tmax_f,
                    "source": "GHCND",
                    "is_preliminary": False,
                    "raw_payload": str(record),
                }
            else:
                logger.info("  ✗ GHCND returned NULL TMAX")
        else:
            logger.info("  ✗ GHCND no data")
    except Exception as e:
        logger.warning(f"  ✗ GHCND error: {e}")

    logger.warning(f"  ✗ No settlement data found from any source")
    return None


def backfill_city(
    cli_client: NWSCliClient,
    cf6_client: NWSCF6Client,
    noaa_client: NOAAClient,
    city: str,
    start_date: date,
    end_date: date,
) -> Dict[str, int]:
    """
    Backfill settlement data for a single city.

    Args:
        cli_client: NWS CLI client
        cf6_client: NWS CF6 client
        noaa_client: NOAA GHCND client
        city: City name
        start_date: Start date
        end_date: End date

    Returns:
        Dict with counts: {cli: N, cf6: N, ghcnd: N, missing: N}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILLING: {city.upper()}")
    logger.info(f"{'='*60}\n")

    stats = {"cli": 0, "cf6": 0, "ghcnd": 0, "missing": 0}

    try:
        with get_session() as session:
            # Iterate through each date
            current_date = start_date
            while current_date <= end_date:
                logger.info(f"Date: {current_date}")

                # Fetch settlement with precedence logic
                result = fetch_settlement_with_precedence(
                    cli_client, cf6_client, noaa_client, city, current_date
                )

                if result:
                    # Upsert to database
                    upsert_settlement(session, result)
                    stats[result["source"].lower()] += 1
                else:
                    stats["missing"] += 1

                # Rate limiting (be nice to NWS servers)
                time.sleep(0.5)

                current_date += timedelta(days=1)

            session.commit()
            logger.info(f"\n✓ {city.upper()} BACKFILL COMPLETE!")
            logger.info(f"  CLI: {stats['cli']}")
            logger.info(f"  CF6: {stats['cf6']}")
            logger.info(f"  GHCND: {stats['ghcnd']}")
            logger.info(f"  Missing: {stats['missing']}\n")

    except Exception as e:
        logger.error(f"Error backfilling {city}: {e}", exc_info=True)

    return stats


def main():
    """Main execution function."""
    args = parse_args()

    # Load environment variables
    load_dotenv()

    # Parse city list
    if args.cities == "all":
        cities = list(SETTLEMENT_STATIONS.keys())
    else:
        cities = [c.strip() for c in args.cities.split(",")]

    # Validate cities
    for city in cities:
        if city not in SETTLEMENT_STATIONS:
            logger.error(f"Unknown city: {city}. Available: {list(SETTLEMENT_STATIONS.keys())}")
            sys.exit(1)

    # Initialize clients
    logger.info("Initializing clients...")
    cli_client = NWSCliClient()
    cf6_client = NWSCF6Client()
    noaa_client = NOAAClient()

    # Parse dates
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    num_days = (end_dt - start_dt).days + 1

    logger.info(f"\n{'='*60}")
    logger.info(f"SETTLEMENT DATA BACKFILL (CLI → CF6 → GHCND)")
    logger.info(f"{'='*60}\n")
    logger.info(f"Date range: {args.start_date} to {args.end_date} ({num_days} days)")
    logger.info(f"Cities: {', '.join(cities)}")
    logger.info(f"Expected records: {num_days * len(cities)}")
    logger.info(f"\n{'='*60}\n")

    # Backfill each city
    total_stats = {"cli": 0, "cf6": 0, "ghcnd": 0, "missing": 0, "cities": 0}

    for city in cities:
        try:
            stats = backfill_city(
                cli_client,
                cf6_client,
                noaa_client,
                city,
                start_dt,
                end_dt,
            )

            total_stats["cli"] += stats["cli"]
            total_stats["cf6"] += stats["cf6"]
            total_stats["ghcnd"] += stats["ghcnd"]
            total_stats["missing"] += stats["missing"]
            total_stats["cities"] += 1

        except Exception as e:
            logger.error(f"Fatal error backfilling {city}: {e}", exc_info=True)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BACKFILL SUMMARY")
    logger.info(f"{'='*60}\n")
    logger.info(f"Cities processed: {total_stats['cities']}/{len(cities)}")
    logger.info(f"CLI records: {total_stats['cli']:,}")
    logger.info(f"CF6 records: {total_stats['cf6']:,}")
    logger.info(f"GHCND records: {total_stats['ghcnd']:,}")
    logger.info(f"Missing: {total_stats['missing']:,}")

    total_found = total_stats["cli"] + total_stats["cf6"] + total_stats["ghcnd"]
    expected = num_days * len(cities)
    coverage_pct = (total_found / expected * 100) if expected > 0 else 0

    logger.info(f"\nTotal coverage: {total_found}/{expected} ({coverage_pct:.1f}%)")

    if total_stats["cli"] > 0:
        cli_pct = (total_stats["cli"] / total_found * 100)
        logger.info(f"CLI (official): {cli_pct:.1f}%")

    if total_stats["missing"] > 0:
        logger.warning(f"\n⚠ {total_stats['missing']} dates missing settlement data")

    logger.info("\n✓ BACKFILL COMPLETE!\n")


if __name__ == "__main__":
    main()
