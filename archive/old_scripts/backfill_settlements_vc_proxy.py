#!/usr/bin/env python3
"""
Backfill settlement gaps using Visual Crossing daily max as proxy.

Uses minute-level observations from wx.minute_obs, aggregates to daily max
temperature, and fills gaps in wx.settlement where official sources
(NWS CLI/CF6, GHCND) are missing.

Precedence: CLI (final) → CF6 (preliminary) → GHCND (audit) → VC (this, proxy)

Usage:
    # Dry run - show gaps without inserting
    python scripts/backfill_settlements_vc_proxy.py --start-date 2025-08-05 --end-date 2025-11-12 --cities all --dry-run

    # Fill gaps for all cities
    python scripts/backfill_settlements_vc_proxy.py --start-date 2025-08-05 --end-date 2025-11-12 --cities all

    # Fill gaps for Chicago only
    python scripts/backfill_settlements_vc_proxy.py --start-date 2025-11-01 --end-date 2025-11-12 --cities chicago
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

from weather.vc_daily_max import (
    fill_settlement_gaps_with_vc,
    compare_vc_vs_official,
)
from weather.nws_cli import SETTLEMENT_STATIONS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill settlement gaps using Visual Crossing proxy"
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
        "--dry-run",
        action="store_true",
        help="Show gaps without inserting (default: false)",
    )
    parser.add_argument(
        "--show-comparison",
        action="store_true",
        help="Show VC vs official comparison table (default: false)",
    )
    return parser.parse_args()


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

    logger.info(f"\n{'='*60}")
    logger.info(f"VC PROXY SETTLEMENT BACKFILL")
    logger.info(f"{'='*60}\n")
    logger.info(f"Date range: {args.start_date} to {args.end_date} ({num_days} days)")
    logger.info(f"Cities: {', '.join(cities)}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Show comparison: {args.show_comparison}")
    logger.info(f"\n{'='*60}\n")

    # Show VC vs official comparison if requested
    if args.show_comparison:
        logger.info("VC vs Official Settlement Comparison")
        logger.info(f"{'='*60}\n")

        for city in cities:
            logger.info(f"\n{city.upper()}:")
            logger.info("-" * 65)

            comparisons = compare_vc_vs_official(city, start_dt, end_dt)

            print("\nDate       | VC Max | Official | Source    | Diff   | Obs")
            print("-" * 65)

            for comp in comparisons:
                vc_str = f"{comp['vc_tmax']:.1f}°F" if comp['vc_tmax'] else "N/A    "
                off_str = f"{comp['official_tmax']:.1f}°F" if comp['official_tmax'] else "N/A    "
                src_str = comp['official_source'] or "N/A      "
                diff_str = f"{comp['diff']:+.1f}°F" if comp['diff'] else "N/A   "
                obs_str = str(comp['num_obs']) if comp['num_obs'] else "N/A"

                print(
                    f"{comp['date_local']} | {vc_str:6} | {off_str:8} | "
                    f"{src_str:9} | {diff_str:6} | {obs_str}"
                )

        logger.info(f"\n{'='*60}\n")

    # Fill gaps with VC proxy
    logger.info("Filling settlement gaps with VC proxy...")
    logger.info(f"{'='*60}\n")

    stats = fill_settlement_gaps_with_vc(
        start_dt, end_dt, cities, dry_run=args.dry_run
    )

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BACKFILL SUMMARY")
    logger.info(f"{'='*60}\n")
    logger.info(f"Total gaps found: {stats['total_gaps']}")
    logger.info(f"Filled with VC proxy: {stats['filled']}")
    logger.info(f"Errors: {stats['errors']}")

    if args.dry_run:
        logger.info("\n✓ DRY RUN COMPLETE - No data inserted")
    else:
        if stats['errors'] > 0:
            logger.warning(f"\n⚠ {stats['errors']} errors occurred during backfill")
            sys.exit(1)
        else:
            logger.info("\n✓ BACKFILL COMPLETE!")


if __name__ == "__main__":
    main()
