#!/usr/bin/env python3
"""
Visual Crossing daily max aggregator.

Aggregates minute-level observations from wx.minute_obs to compute daily
maximum temperature. Used as PROXY when official settlement sources
(NWS CLI/CF6, GHCND) are null or unavailable.

Precedence: CLI (final) → CF6 (preliminary) → GHCND (audit) → VC (this, proxy)

Important: VC data should match official sources within ±1-2°F.
"""

import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import text

from db.connection import get_session
from db.loaders import upsert_settlement
from weather.nws_cli import SETTLEMENT_STATIONS
from weather.time_utils import get_timezone_name

logger = logging.getLogger(__name__)


def aggregate_daily_max_for_city(
    city: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """
    Aggregate Visual Crossing minute observations to daily max temp.

    Args:
        city: City name (chicago, new_york, etc.)
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        List of dicts with keys: city, icao, date_local, tmax_f, num_obs
    """
    if city not in SETTLEMENT_STATIONS:
        raise ValueError(
            f"Unknown city: {city}. Available: {list(SETTLEMENT_STATIONS.keys())}"
        )

    station = SETTLEMENT_STATIONS[city]
    timezone = get_timezone_name(city)

    logger.info(f"Aggregating VC daily max for {city} ({station['icao']}) in {timezone}")

    with get_session() as session:
        # Query: aggregate MAX(temp_f) by local date
        # Convert ts_utc to local timezone, extract date, then MAX(temp_f)
        query = text("""
            SELECT
                :loc_id as loc_id,
                DATE(ts_utc AT TIME ZONE :timezone) as date_local,
                MAX(temp_f) as tmax_f,
                COUNT(*) as num_obs
            FROM wx.minute_obs
            WHERE loc_id = :loc_id
              AND DATE(ts_utc AT TIME ZONE :timezone) >= :start_date
              AND DATE(ts_utc AT TIME ZONE :timezone) <= :end_date
              AND temp_f IS NOT NULL
              AND NOT (temp_f = 'NaN'::float)
            GROUP BY loc_id, DATE(ts_utc AT TIME ZONE :timezone)
            ORDER BY date_local
        """)

        result = session.execute(query, {
            "loc_id": station["icao"],
            "timezone": timezone,
            "start_date": start_date,
            "end_date": end_date,
        }).fetchall()

        # Convert to list of dicts
        daily_maxes = []
        for row in result:
            daily_maxes.append({
                "city": city,
                "icao": row.loc_id,
                "issuedby": station["issuedby"],
                "date_local": row.date_local,
                "tmax_f": float(row.tmax_f),
                "num_obs": row.num_obs,
            })

        logger.info(f"Aggregated {len(daily_maxes)} daily max temps for {city}")

        return daily_maxes


def aggregate_all_cities(
    start_date: date,
    end_date: date,
    cities: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Aggregate VC daily max for all cities.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        cities: List of city names (default: all)

    Returns:
        Dict mapping city name to list of daily max dicts
    """
    if cities is None:
        cities = list(SETTLEMENT_STATIONS.keys())

    results = {}

    for city in cities:
        try:
            daily_maxes = aggregate_daily_max_for_city(city, start_date, end_date)
            results[city] = daily_maxes
        except Exception as e:
            logger.error(f"Error aggregating {city}: {e}", exc_info=True)
            results[city] = []

    return results


def compare_vc_vs_official(
    city: str,
    start_date: date,
    end_date: date,
) -> List[Dict[str, Any]]:
    """
    Compare Visual Crossing daily max vs official settlement sources.

    Args:
        city: City name
        start_date: Start date
        end_date: End date

    Returns:
        List of comparison dicts with keys:
            city, date_local, vc_tmax, official_tmax, official_source, diff, num_obs
    """
    if city not in SETTLEMENT_STATIONS:
        raise ValueError(f"Unknown city: {city}")

    station = SETTLEMENT_STATIONS[city]

    with get_session() as session:
        # Join VC aggregated max with official settlements
        query = text("""
            WITH vc_daily AS (
                SELECT
                    loc_id,
                    DATE(ts_utc AT TIME ZONE :timezone) as date_local,
                    MAX(temp_f) as tmax_f,
                    COUNT(*) as num_obs
                FROM wx.minute_obs
                WHERE loc_id = :loc_id
                  AND DATE(ts_utc AT TIME ZONE :timezone) >= :start_date
                  AND DATE(ts_utc AT TIME ZONE :timezone) <= :end_date
                  AND temp_f IS NOT NULL
                  AND NOT (temp_f = 'NaN'::float)
                GROUP BY loc_id, DATE(ts_utc AT TIME ZONE :timezone)
            )
            SELECT
                :city as city,
                COALESCE(v.date_local, s.date_local) as date_local,
                v.tmax_f as vc_tmax,
                s.tmax_final as official_tmax,
                s.source_final as official_source,
                v.num_obs,
                ROUND((v.tmax_f - s.tmax_final)::numeric, 1) as diff
            FROM vc_daily v
            FULL OUTER JOIN (
                SELECT * FROM wx.settlement
                WHERE city = :city
                  AND date_local >= :start_date
                  AND date_local <= :end_date
            ) s ON s.date_local = v.date_local
            WHERE COALESCE(v.date_local, s.date_local) >= :start_date
              AND COALESCE(v.date_local, s.date_local) <= :end_date
            ORDER BY date_local
        """)

        result = session.execute(query, {
            "city": city,
            "loc_id": station["icao"],
            "timezone": CITY_TIMEZONES[city],
            "start_date": start_date,
            "end_date": end_date,
        }).fetchall()

        # Convert to list of dicts
        comparisons = []
        for row in result:
            comparisons.append({
                "city": row.city,
                "date_local": row.date_local,
                "vc_tmax": float(row.vc_tmax) if row.vc_tmax else None,
                "official_tmax": float(row.official_tmax) if row.official_tmax else None,
                "official_source": row.official_source,
                "diff": float(row.diff) if row.diff else None,
                "num_obs": row.num_obs,
            })

        return comparisons


def fill_settlement_gaps_with_vc(
    start_date: date,
    end_date: date,
    cities: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Fill gaps in wx.settlement using VC daily max as proxy.

    Only inserts VC data when no official settlement exists (CLI/CF6/GHCND null).

    Args:
        start_date: Start date
        end_date: End date
        cities: List of cities (default: all)
        dry_run: If True, only report what would be inserted

    Returns:
        Dict with stats: {total_gaps: N, filled: N, errors: N}
    """
    if cities is None:
        cities = list(SETTLEMENT_STATIONS.keys())

    stats = {"total_gaps": 0, "filled": 0, "errors": 0}

    for city in cities:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Checking gaps for {city.upper()}")
            logger.info(f"{'='*60}\n")

            # Get VC daily maxes
            vc_maxes = aggregate_daily_max_for_city(city, start_date, end_date)

            with get_session() as session:
                for vc_record in vc_maxes:
                    # Check if official settlement exists (any non-VC source in columnar schema)
                    check = session.execute(text("""
                        SELECT COUNT(*) as count
                        FROM wx.settlement
                        WHERE city = :city
                          AND date_local = :date_local
                          AND (tmax_cli IS NOT NULL
                               OR tmax_cf6 IS NOT NULL
                               OR tmax_iem_cf6 IS NOT NULL
                               OR tmax_ghcnd IS NOT NULL)
                    """), {
                        "city": city,
                        "date_local": vc_record["date_local"],
                    }).fetchone()

                    if check.count == 0:
                        # Gap found - official source missing
                        stats["total_gaps"] += 1

                        logger.info(
                            f"Gap found: {city} {vc_record['date_local']} "
                            f"(VC: {vc_record['tmax_f']:.1f}°F, {vc_record['num_obs']} obs)"
                        )

                        if not dry_run:
                            # Insert VC as proxy using columnar schema
                            upsert_settlement(session, {
                                "city": vc_record["city"],
                                "date_local": vc_record["date_local"],
                                "tmax_f": vc_record["tmax_f"],
                                "source": "VC",  # Maps to tmax_vc, is_prelim_vc, retrieved_at_vc
                                "is_preliminary": True,
                                "raw_payload": f"Aggregated from {vc_record['num_obs']} minute observations",
                            })

                            stats["filled"] += 1
                            logger.info(f"  → Filled with VC proxy")

                if not dry_run:
                    session.commit()

        except Exception as e:
            logger.error(f"Error filling gaps for {city}: {e}", exc_info=True)
            stats["errors"] += 1

    return stats


def main():
    """Demo: Compare VC vs official settlements for Chicago."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from datetime import timedelta

    # Compare last 14 days
    end_date = date.today()
    start_date = end_date - timedelta(days=14)

    logger.info(f"\n{'='*60}")
    logger.info("VC vs Official Settlement Comparison")
    logger.info(f"{'='*60}\n")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"City: Chicago\n")

    comparisons = compare_vc_vs_official("chicago", start_date, end_date)

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

    # Check for gaps
    logger.info(f"\n{'='*60}")
    logger.info("Checking for settlement gaps...")
    logger.info(f"{'='*60}\n")

    stats = fill_settlement_gaps_with_vc(start_date, end_date, ["chicago"], dry_run=True)

    logger.info(f"\nGap Analysis:")
    logger.info(f"  Total gaps found: {stats['total_gaps']}")
    logger.info(f"  Would fill: {stats['filled']}")
    logger.info(f"  Errors: {stats['errors']}")


if __name__ == "__main__":
    main()
