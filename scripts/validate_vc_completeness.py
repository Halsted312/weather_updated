#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate Visual Crossing minute observation completeness.

Checks:
- Coverage: how many days have complete 5-min grid (288 rows)
- Forward-fill percentage per day and overall
- VC daily max vs official settlement temperature differences

Usage:
    python scripts/validate_vc_completeness.py \
        --start-date 2024-01-10 \
        --end-date 2024-01-16 \
        --cities chicago \
        --output data/vc_validation_report.csv
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, date
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from weather.vc_daily_max import compare_vc_vs_official, CITY_TIMEZONES
from weather.visual_crossing import STATION_MAP
from sqlalchemy import text

logger = logging.getLogger(__name__)


def validate_vc_coverage(
    start_date: date,
    end_date: date,
    cities: list[str],
) -> pd.DataFrame:
    """
    Validate VC data coverage for date range.

    Returns:
        DataFrame with columns: city, date_local, total_rows, real_rows, ffilled_rows, ffilled_pct, coverage_complete
    """
    with get_session() as session:
        # Query per-city, per-day coverage
        query = text("""
            WITH daily_stats AS (
                SELECT
                    loc_id,
                    DATE(ts_utc AT TIME ZONE :timezone) as date_local,
                    COUNT(*) as total_rows,
                    SUM(CASE WHEN NOT ffilled THEN 1 ELSE 0 END) as real_rows,
                    SUM(CASE WHEN ffilled THEN 1 ELSE 0 END) as ffilled_rows
                FROM wx.minute_obs
                WHERE loc_id = :loc_id
                  AND DATE(ts_utc AT TIME ZONE :timezone) >= :start_date
                  AND DATE(ts_utc AT TIME ZONE :timezone) <= :end_date
                GROUP BY loc_id, DATE(ts_utc AT TIME ZONE :timezone)
            )
            SELECT
                :city as city,
                date_local,
                total_rows,
                real_rows,
                ffilled_rows,
                ROUND(100.0 * ffilled_rows / NULLIF(total_rows, 0), 1) as ffilled_pct,
                (total_rows = 288) as coverage_complete
            FROM daily_stats
            ORDER BY date_local
        """)

        all_results = []

        for city in cities:
            if city not in STATION_MAP:
                logger.warning(f"Unknown city: {city}")
                continue

            station = STATION_MAP[city]
            loc_id = station[0]
            timezone = CITY_TIMEZONES[city]

            result = session.execute(query, {
                "city": city,
                "loc_id": loc_id,
                "timezone": timezone,
                "start_date": start_date,
                "end_date": end_date,
            }).fetchall()

            for row in result:
                all_results.append({
                    "city": row.city,
                    "date_local": row.date_local,
                    "total_rows": row.total_rows,
                    "real_rows": row.real_rows,
                    "ffilled_rows": row.ffilled_rows,
                    "ffilled_pct": float(row.ffilled_pct) if row.ffilled_pct else 0.0,
                    "coverage_complete": row.coverage_complete,
                })

        return pd.DataFrame(all_results)


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Visual Crossing minute observation completeness"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--cities",
        type=str,
        nargs="+",
        default=["all"],
        help="Cities to validate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file (optional)"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    # Get cities list
    if "all" in args.cities:
        cities = list(STATION_MAP.keys())
    else:
        cities = args.cities

    logger.info(f"\n{'='*60}")
    logger.info("VISUAL CROSSING COVERAGE VALIDATION")
    logger.info(f"{'='*60}\n")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Cities: {', '.join(cities)}")
    logger.info(f"\n{'='*60}\n")

    # Validate coverage
    logger.info("Validating VC data coverage...")
    coverage_df = validate_vc_coverage(start_date, end_date, cities)

    if coverage_df.empty:
        logger.warning("No VC data found in database for this date range")
        return 1

    # Calculate overall stats
    overall_stats = {
        "total_days": len(coverage_df),
        "complete_days": int(coverage_df['coverage_complete'].sum()),
        "total_rows": int(coverage_df['total_rows'].sum()),
        "real_rows": int(coverage_df['real_rows'].sum()),
        "ffilled_rows": int(coverage_df['ffilled_rows'].sum()),
    }
    overall_stats["coverage_pct"] = 100.0 * overall_stats["complete_days"] / overall_stats["total_days"]
    overall_stats["ffilled_pct"] = 100.0 * overall_stats["ffilled_rows"] / overall_stats["total_rows"]

    # Print summary
    print("\n" + "="*60)
    print("VC COVERAGE REPORT")
    print("="*60)
    print(f"Total days: {overall_stats['total_days']}")
    print(f"Complete days (288 rows): {overall_stats['complete_days']} ({overall_stats['coverage_pct']:.1f}%)")
    print(f"Total rows: {overall_stats['total_rows']}")
    print(f"  Real observations: {overall_stats['real_rows']}")
    print(f"  Forward-filled: {overall_stats['ffilled_rows']} ({overall_stats['ffilled_pct']:.1f}%)")
    print("="*60 + "\n")

    # Show per-city stats
    print("Per-city stats:")
    for city in cities:
        city_df = coverage_df[coverage_df['city'] == city]
        if city_df.empty:
            continue

        city_stats = {
            "days": len(city_df),
            "complete": int(city_df['coverage_complete'].sum()),
            "avg_ffilled_pct": city_df['ffilled_pct'].mean(),
        }
        print(f"  {city}: {city_stats['complete']}/{city_stats['days']} complete, {city_stats['avg_ffilled_pct']:.1f}% avg ffilled")

    print("\n" + "="*60 + "\n")

    # Identify poor coverage days (> 50% ffilled)
    poor_coverage = coverage_df[coverage_df['ffilled_pct'] > 50.0]
    if not poor_coverage.empty:
        logger.warning(f"Found {len(poor_coverage)} days with > 50% forward-filled data:")
        for _, row in poor_coverage.iterrows():
            logger.warning(
                f"  {row['city']} {row['date_local']}: {row['ffilled_pct']:.1f}% ffilled "
                f"({row['ffilled_rows']}/{row['total_rows']} rows)"
            )
    else:
        logger.info(" All days have d 50% forward-filled data (good coverage)")

    # Compare VC daily max vs official
    logger.info(f"\n{'='*60}")
    logger.info("Comparing VC daily max vs official settlement temps...")
    logger.info(f"{'='*60}\n")

    all_comparisons = []
    for city in cities:
        try:
            comparisons = compare_vc_vs_official(city, start_date, end_date)
            all_comparisons.extend(comparisons)
        except Exception as e:
            logger.error(f"Error comparing {city}: {e}")

    if all_comparisons:
        comp_df = pd.DataFrame(all_comparisons)

        # Filter to days with both VC and official data
        valid_comp = comp_df.dropna(subset=['vc_tmax', 'official_tmax'])

        if not valid_comp.empty:
            avg_diff = valid_comp['diff'].abs().mean()
            max_diff = valid_comp['diff'].abs().max()
            count_within_2f = (valid_comp['diff'].abs() <= 2).sum()

            print("VC vs Official Daily Max:")
            print(f"  Days compared: {len(valid_comp)}")
            print(f"  Avg absolute diff: {avg_diff:.1f}�F")
            print(f"  Max absolute diff: {max_diff:.1f}�F")
            print(f"  Within �2�F: {count_within_2f}/{len(valid_comp)} ({100.0*count_within_2f/len(valid_comp):.1f}%)")
            print()

        # Add to coverage report
        if args.output:
            # Merge coverage with comparisons
            coverage_df = coverage_df.merge(
                comp_df[['city', 'date_local', 'vc_tmax', 'official_tmax', 'official_source', 'diff']],
                on=['city', 'date_local'],
                how='left'
            )

    # Save to CSV if requested
    if args.output:
        coverage_df.to_csv(args.output, index=False)
        logger.info(f"Saved validation report to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
