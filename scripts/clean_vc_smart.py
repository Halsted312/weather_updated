#!/usr/bin/env python3
"""
Smart VC temperature cleanup with ML exclusion logic.

Two-tier approach:
1. Days with <5% NULL temps → Forward-fill gaps, mark is_forward_filled=TRUE
2. Days with ≥5% NULL temps → Mark exclude_from_ml=TRUE, don't forward-fill

This ensures ML training uses only high-quality data while preserving
usable days with minor gaps.

Usage:
    # Dry run to see what would happen
    python scripts/clean_vc_smart.py --all-cities --dry-run

    # Apply cleanup
    python scripts/clean_vc_smart.py --all-cities

    # Single city
    python scripts/clean_vc_smart.py --city-code MIA
"""

import argparse
import logging
import sys
from datetime import date
from typing import Dict, List, Optional, Set

from sqlalchemy import text

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.db import get_db_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NULL_THRESHOLD_PCT = 5.0  # Days with ≥5% NULL should be excluded from ML


def analyze_day_quality(session, city_code: str) -> Dict[date, float]:
    """Calculate NULL percentage for each day.

    Returns:
        Dict mapping date → NULL percentage
    """
    query = text("""
        SELECT
            vmw.datetime_local::date as day,
            ROUND(100.0 * COUNT(CASE WHEN vmw.temp_f IS NULL OR vmw.temp_f < -70 THEN 1 END) / COUNT(*), 2) as null_pct
        FROM wx.vc_minute_weather vmw
        JOIN wx.vc_location loc ON vmw.vc_location_id = loc.id
        WHERE loc.city_code = :city_code
          AND loc.location_type = 'station'
          AND vmw.data_type = 'actual_obs'
        GROUP BY vmw.datetime_local::date
        HAVING COUNT(CASE WHEN vmw.temp_f IS NULL OR vmw.temp_f < -70 THEN 1 END) > 0
    """)

    results = session.execute(query, {"city_code": city_code}).fetchall()
    return {row[0]: row[1] for row in results}


def get_bad_days(day_quality: Dict[date, float], threshold_pct: float) -> Set[date]:
    """Get set of days that should be excluded from ML (≥threshold% NULL)."""
    return {day for day, null_pct in day_quality.items() if null_pct >= threshold_pct}


def forward_fill_city(
    session,
    city_code: str,
    bad_days: Set[date],
    dry_run: bool = False,
) -> Dict[str, int]:
    """Forward-fill temperature errors for one city.

    Only forward-fills days with <5% NULL (good days with minor gaps).
    Days with ≥5% NULL are marked exclude_from_ml=TRUE instead.

    Returns:
        Dict with counts: ffilled, excluded, total_fixed
    """
    logger.info(f"Processing {city_code}...")

    # Get location ID
    query_loc = text("""
        SELECT id FROM wx.vc_location
        WHERE city_code = :city_code AND location_type = 'station'
    """)
    loc_result = session.execute(query_loc, {"city_code": city_code}).fetchone()

    if not loc_result:
        logger.error(f"  Location not found for {city_code}")
        return {"ffilled": 0, "excluded": 0, "total_fixed": 0}

    location_id = loc_result[0]

    # Load all observations
    query_obs = text("""
        SELECT
            id,
            datetime_local::date as day,
            datetime_utc,
            temp_f
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :location_id
          AND data_type = 'actual_obs'
        ORDER BY datetime_utc
    """)

    rows = session.execute(query_obs, {"location_id": location_id}).fetchall()
    logger.info(f"  Loaded {len(rows)} observations")

    # Track what we'll do
    records_to_ffill: List[tuple] = []  # (id, new_temp)
    records_to_exclude: List[int] = []  # id

    last_valid_temp: Optional[float] = None

    for row_id, day, dt_utc, temp_f in rows:
        # Check if this day should be excluded from ML
        is_bad_day = day in bad_days

        # Check if temp is valid
        is_valid = temp_f is not None and temp_f > -50 and temp_f < 150

        if is_valid:
            last_valid_temp = temp_f
        elif is_bad_day:
            # Bad day (≥5% NULL) - mark for ML exclusion, don't forward-fill
            records_to_exclude.append(row_id)
        elif last_valid_temp is not None:
            # Good day with minor gap (<5% NULL) - forward-fill
            records_to_ffill.append((row_id, last_valid_temp))

    logger.info(f"  Found:")
    logger.info(f"    {len(records_to_ffill)} records to forward-fill (good days with minor gaps)")
    logger.info(f"    {len(records_to_exclude)} records to mark exclude_from_ml (bad days)")

    if dry_run:
        return {"ffilled": 0, "excluded": 0, "total_fixed": 0}

    # Apply forward-fills
    if records_to_ffill:
        for row_id, new_temp in records_to_ffill:
            update_stmt = text("""
                UPDATE wx.vc_minute_weather
                SET temp_f = :new_temp,
                    is_forward_filled = TRUE
                WHERE id = :row_id
            """)
            session.execute(update_stmt, {"new_temp": new_temp, "row_id": row_id})

    # Mark ML exclusions
    if records_to_exclude:
        for row_id in records_to_exclude:
            update_stmt = text("""
                UPDATE wx.vc_minute_weather
                SET exclude_from_ml = TRUE
                WHERE id = :row_id
            """)
            session.execute(update_stmt, {"row_id": row_id})

    session.commit()

    logger.info(f"  ✅ Fixed {len(records_to_ffill)} records (forward-filled)")
    logger.info(f"  ✅ Flagged {len(records_to_exclude)} records (exclude from ML)")

    return {
        "ffilled": len(records_to_ffill),
        "excluded": len(records_to_exclude),
        "total_fixed": len(records_to_ffill) + len(records_to_exclude),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Smart cleanup: ffill minor gaps, exclude bad days from ML"
    )
    parser.add_argument(
        "--city-code",
        type=str,
        help="Process single city (e.g., CHI, MIA)",
    )
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help="Process all cities",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=NULL_THRESHOLD_PCT,
        help=f"NULL%% threshold for ML exclusion (default: {NULL_THRESHOLD_PCT}%%)",
    )

    args = parser.parse_args()

    # Determine cities
    if args.all_cities:
        city_codes = ["CHI", "AUS", "DEN", "LAX", "MIA", "PHL"]
    elif args.city_code:
        city_codes = [args.city_code]
    else:
        logger.error("Must specify --city-code or --all-cities")
        sys.exit(1)

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    logger.info(f"NULL threshold for ML exclusion: {args.threshold}%")
    logger.info("")

    total_ffilled = 0
    total_excluded = 0
    total_fixed = 0

    with get_db_session() as session:
        for city_code in city_codes:
            # Analyze day quality
            day_quality = analyze_day_quality(session, city_code)
            bad_days = get_bad_days(day_quality, args.threshold)

            logger.info(f"{city_code}: {len(bad_days)} days to exclude (≥{args.threshold}% NULL)")

            # Apply fixes
            stats = forward_fill_city(session, city_code, bad_days, dry_run=args.dry_run)

            total_ffilled += stats["ffilled"]
            total_excluded += stats["excluded"]
            total_fixed += stats["total_fixed"]

            logger.info("")

    logger.info("="*60)
    logger.info("CLEANUP COMPLETE")
    logger.info("="*60)
    logger.info(f"Total records forward-filled: {total_ffilled:,}")
    logger.info(f"Total records excluded from ML: {total_excluded:,}")
    logger.info(f"Total records processed: {total_fixed:,}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
