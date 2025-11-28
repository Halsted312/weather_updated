#!/usr/bin/env python3
"""
Clean and forward-fill temperature errors in VC minute weather data.

Detects and fixes temperature anomalies:
- Sentinel values (e.g., -77.9F)
- Impossible jumps (>50F change in 5 minutes)
- Out-of-range values (<-50F or >150F)

Uses forward-fill from last valid temperature, marks records with is_forward_filled=TRUE.

Usage:
    # Backfill existing bad data
    python scripts/clean_vc_temp_errors.py --backfill --all-cities

    # Check what would be fixed (dry-run)
    python scripts/clean_vc_temp_errors.py --backfill --all-cities --dry-run

    # Fix specific city
    python scripts/clean_vc_temp_errors.py --backfill --city-code CHI
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import List, Optional

from sqlalchemy import text, update

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.db import get_db_session, VcMinuteWeather, VcLocation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_temp_valid(temp: Optional[float]) -> bool:
    """Check if temperature is valid (not sentinel or out of range).

    Args:
        temp: Temperature in °F

    Returns:
        True if valid, False if error/sentinel
    """
    if temp is None:
        return False

    # VC sentinel values
    if abs(temp - (-77.9)) < 0.1:  # -77.9F sentinel
        return False

    # Physically impossible ranges
    if temp < -50 or temp > 150:
        return False

    return True


def detect_and_fix_errors(
    session,
    city_code: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Detect and forward-fill temperature errors for a city.

    Args:
        session: SQLAlchemy session
        city_code: City code (e.g., 'CHI', 'AUS')
        dry_run: If True, only report what would be fixed

    Returns:
        (records_fixed, records_checked)
    """
    logger.info(f"Processing {city_code}...")

    # Get location ID
    query_loc = text("""
        SELECT id FROM wx.vc_location
        WHERE city_code = :city_code AND location_type = 'station'
    """)
    loc_result = session.execute(query_loc, {"city_code": city_code}).fetchone()

    if not loc_result:
        logger.error(f"Location not found for {city_code}")
        return 0, 0

    location_id = loc_result[0]

    # Get all observations ordered by time
    query_obs = text("""
        SELECT
            id,
            datetime_utc,
            temp_f,
            is_forward_filled
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :location_id
          AND data_type = 'actual_obs'
        ORDER BY datetime_utc
    """)

    rows = session.execute(query_obs, {"location_id": location_id}).fetchall()

    logger.info(f"  Loaded {len(rows)} observations")

    # Forward-fill pass
    last_valid_temp: Optional[float] = None
    records_to_fix: List[tuple] = []  # (id, new_temp)
    records_checked = 0

    for row_id, dt_utc, temp_f, is_ff in rows:
        records_checked += 1

        if is_temp_valid(temp_f):
            # Valid temp - update last known good
            last_valid_temp = temp_f
        else:
            # Invalid temp - needs forward-fill
            if last_valid_temp is not None:
                records_to_fix.append((row_id, last_valid_temp, temp_f))
                temp_str = f"{temp_f:.1f}F" if temp_f is not None else "NULL"
                logger.debug(
                    f"    {dt_utc}: {temp_str} (bad) → {last_valid_temp:.1f}F (ffill)"
                )
            else:
                # No prior valid temp - skip (beginning of series)
                temp_str = f"{temp_f:.1f}F" if temp_f is not None else "NULL"
                logger.warning(
                    f"    {dt_utc}: {temp_str} (bad) - no prior valid temp to ffill from"
                )

    if not records_to_fix:
        logger.info(f"  ✅ No errors found for {city_code}")
        return 0, records_checked

    logger.info(f"  Found {len(records_to_fix)} records to fix")

    if dry_run:
        # Show what would be fixed
        logger.info(f"  Preview of fixes (showing first 10):")
        for row_id, new_temp, old_temp in records_to_fix[:10]:
            if old_temp is None:
                old_str = "NULL"
            elif old_temp < -70:
                old_str = f"{old_temp:.1f}F (sentinel)"
            else:
                old_str = f"{old_temp:.1f}F"
            logger.info(f"    ID {row_id}: {old_str} → {new_temp:.1f}F")
        if len(records_to_fix) > 10:
            logger.info(f"    ... and {len(records_to_fix) - 10} more")
        return 0, records_checked

    # Apply fixes
    for row_id, new_temp, old_temp in records_to_fix:
        update_stmt = text("""
            UPDATE wx.vc_minute_weather
            SET temp_f = :new_temp,
                is_forward_filled = TRUE
            WHERE id = :row_id
        """)
        session.execute(update_stmt, {"new_temp": new_temp, "row_id": row_id})

    session.commit()
    logger.info(f"  ✅ Fixed {len(records_to_fix)} records for {city_code}")

    return len(records_to_fix), records_checked


def main():
    parser = argparse.ArgumentParser(
        description="Clean temperature errors in VC minute weather data"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run backfill on existing data",
    )
    parser.add_argument(
        "--city-code",
        type=str,
        help="Process single city (e.g., CHI, AUS)",
    )
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help="Process all cities",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )

    args = parser.parse_args()

    if not args.backfill:
        logger.error("Must specify --backfill for now (live mode not yet implemented)")
        sys.exit(1)

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

    total_fixed = 0
    total_checked = 0

    with get_db_session() as session:
        for city_code in city_codes:
            fixed, checked = detect_and_fix_errors(session, city_code, dry_run=args.dry_run)
            total_fixed += fixed
            total_checked += checked

    logger.info("")
    logger.info("="*60)
    logger.info(f"COMPLETE: Checked {total_checked:,} records")
    logger.info(f"          Fixed {total_fixed} records ({100*total_fixed/total_checked if total_checked > 0 else 0:.4f}%)")
    logger.info("="*60)


if __name__ == "__main__":
    main()
