#!/usr/bin/env python3
"""
Backfill settlement data using IEM CF6 JSON API (primary source).

NEW STRATEGY (based on research findings):
1. IEM CF6 JSON API - bulk fetch full years, fast and reliable
2. CLI (optional) - upgrade recent dates to final values
3. GHCND (optional) - audit/verification only

This is MUCH faster than the old day-by-day scraping approach.

Usage:
    # Backfill 100 days for all cities (recommended)
    python scripts/backfill_settlements_iem.py --start-date 2025-08-05 --end-date 2025-11-12 --cities all

    # Test with Chicago only
    python scripts/backfill_settlements_iem.py --start-date 2025-11-01 --end-date 2025-11-11 --cities chicago

    # Include raw CF6 text for audit (slower)
    python scripts/backfill_settlements_iem.py --start-date 2025-11-01 --end-date 2025-11-11 --cities all --fetch-raw-text
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from weather.iem_cf6 import IEMClient, SETTLEMENT_STATIONS
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
        description="Backfill settlement data using IEM CF6 JSON API"
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
    parser.add_argument(
        "--fetch-raw-text",
        action="store_true",
        help="Fetch raw CF6 text products for audit (slower but more complete)",
    )
    return parser.parse_args()


def backfill_city_iem(
    client: IEMClient,
    city: str,
    start_date: date,
    end_date: date,
    fetch_raw_text: bool = False
) -> Dict[str, int]:
    """
    Backfill settlement data for a single city using IEM.

    Args:
        client: IEM client
        city: City name
        start_date: Start date
        end_date: End date
        fetch_raw_text: Whether to fetch raw CF6 text for audit

    Returns:
        Dict with counts: {records: N, errors: N}
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILLING: {city.upper()} (IEM CF6)")
    logger.info(f"{'='*60}\n")

    stats = {"records": 0, "errors": 0}

    try:
        with get_session() as session:
            # Fetch settlements from IEM
            settlements = client.get_settlements_for_city(
                city, start_date, end_date, fetch_raw_text
            )

            if not settlements:
                logger.warning(f"No IEM CF6 data found for {city}")
                return stats

            # Upsert each settlement to database
            for settlement in settlements:
                try:
                    upsert_settlement(session, settlement)
                    stats["records"] += 1
                except Exception as e:
                    logger.error(
                        f"Error upserting {city} {settlement['date_local']}: {e}"
                    )
                    stats["errors"] += 1

            session.commit()

            logger.info(f"\n✓ {city.upper()} BACKFILL COMPLETE!")
            logger.info(f"  Records: {stats['records']}")
            logger.info(f"  Errors: {stats['errors']}\n")

    except Exception as e:
        logger.error(f"Fatal error backfilling {city}: {e}", exc_info=True)
        stats["errors"] += 1

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
            logger.error(
                f"Unknown city: {city}. Available: {list(SETTLEMENT_STATIONS.keys())}"
            )
            sys.exit(1)

    # Parse dates
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    num_days = (end_dt - start_dt).days + 1

    # Initialize IEM client
    logger.info("Initializing IEM CF6 client...")
    client = IEMClient()

    logger.info(f"\n{'='*60}")
    logger.info(f"IEM CF6 SETTLEMENT BACKFILL")
    logger.info(f"{'='*60}\n")
    logger.info(f"Date range: {args.start_date} to {args.end_date} ({num_days} days)")
    logger.info(f"Cities: {', '.join(cities)}")
    logger.info(f"Expected records: {num_days * len(cities)}")
    logger.info(f"Fetch raw CF6 text: {args.fetch_raw_text}")
    logger.info(f"\n{'='*60}\n")

    # Backfill each city
    total_stats = {"records": 0, "errors": 0, "cities": 0}

    for city in cities:
        try:
            stats = backfill_city_iem(
                client, city, start_dt, end_dt, args.fetch_raw_text
            )

            total_stats["records"] += stats["records"]
            total_stats["errors"] += stats["errors"]
            total_stats["cities"] += 1

        except Exception as e:
            logger.error(f"Fatal error backfilling {city}: {e}", exc_info=True)
            total_stats["errors"] += 1

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BACKFILL SUMMARY")
    logger.info(f"{'='*60}\n")
    logger.info(f"Cities processed: {total_stats['cities']}/{len(cities)}")
    logger.info(f"Total records: {total_stats['records']:,}")
    logger.info(f"Errors: {total_stats['errors']}")

    expected = num_days * len(cities)
    coverage_pct = (total_stats["records"] / expected * 100) if expected > 0 else 0

    logger.info(f"\nCoverage: {total_stats['records']}/{expected} ({coverage_pct:.1f}%)")

    # Validation query
    logger.info(f"\n{'='*60}")
    logger.info("DATABASE VALIDATION")
    logger.info(f"{'='*60}\n")

    try:
        with get_session() as session:
            from sqlalchemy import text

            # Check coverage per city (columnar schema)
            result = session.execute(text("""
                SELECT
                    city,
                    COUNT(*) as total_rows,
                    COUNT(tmax_iem_cf6) as iem_count,
                    COUNT(tmax_cli) as cli_count,
                    COUNT(tmax_cf6) as cf6_count,
                    COUNT(tmax_ghcnd) as ghcnd_count,
                    COUNT(tmax_vc) as vc_count,
                    MIN(date_local)::date as earliest,
                    MAX(date_local)::date as latest
                FROM wx.settlement
                WHERE date_local >= :start_date AND date_local <= :end_date
                GROUP BY city
                ORDER BY city
            """), {"start_date": start_dt, "end_date": end_dt}).fetchall()

            logger.info("Coverage by city:")
            for row in result:
                logger.info(
                    f"  {row[0]:15}: {row[1]:>4} rows  "
                    f"[IEM:{row[2]} CLI:{row[3]} CF6:{row[4]} GHCND:{row[5]} VC:{row[6]}]  "
                    f"{row[7]} to {row[8]}"
                )

            # Check tmax_final and source_final
            result = session.execute(text("""
                SELECT
                    city,
                    source_final,
                    COUNT(*) as count
                FROM wx.settlement
                WHERE date_local >= :start_date AND date_local <= :end_date
                GROUP BY city, source_final
                ORDER BY city, source_final
            """), {"start_date": start_dt, "end_date": end_dt}).fetchall()

            logger.info("\nFinal settlement sources:")
            for row in result:
                logger.info(f"  {row[0]:15} ({row[1]:10}): {row[2]:>4} records")

            logger.info("\n✓ Validation complete (columnar schema)")

    except Exception as e:
        logger.error(f"Error running validation queries: {e}")

    if total_stats["errors"] > 0:
        logger.warning(f"\n⚠ {total_stats['errors']} errors occurred during backfill")
        sys.exit(1)
    else:
        logger.info("\n✓ BACKFILL COMPLETE!\n")


if __name__ == "__main__":
    main()
