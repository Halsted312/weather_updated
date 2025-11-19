#!/usr/bin/env python3
"""
Check Phase 4 coverage: verify that each city/date has complete data.

Checks that for each city/date with settled markets:
- Markets exist (6 bins per day)
- 1-minute candles covering the trading day
- NWS settlement in wx.settlement
- VC minutes (for 6 good cities) in wx.minute_obs
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, date
from typing import List
import pandas as pd
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# City configuration
CITY_CONFIG = {
    "chicago": {"series": "KXHIGHCHI", "loc_id": "KMDW", "has_vc": True},
    "miami": {"series": "KXHIGHMIA", "loc_id": "KMIA", "has_vc": True},
    "austin": {"series": "KXHIGHAUS", "loc_id": "KAUS", "has_vc": True},
    "la": {"series": "KXHIGHLAX", "loc_id": "KLAX", "has_vc": True},
    "denver": {"series": "KXHIGHDEN", "loc_id": "KDEN", "has_vc": True},
    "philadelphia": {"series": "KXHIGHPHIL", "loc_id": "KPHL", "has_vc": True},
}


def check_market_coverage(start_date: date, end_date: date) -> pd.DataFrame:
    """Check market coverage per city/date."""
    logger.info("Checking market coverage...")

    with get_session() as session:
        query = text("""
            SELECT
                series_ticker,
                DATE(close_time AT TIME ZONE 'America/Chicago') as market_date,
                COUNT(*) as market_count,
                COUNT(*) FILTER (WHERE status = 'settled') as settled_count,
                MIN(ticker) as example_ticker
            FROM markets
            WHERE series_ticker IN :series_list
              AND close_time >= :start_date
              AND close_time <= :end_date
            GROUP BY series_ticker, DATE(close_time AT TIME ZONE 'America/Chicago')
            ORDER BY series_ticker, market_date
        """)

        result = session.execute(
            query,
            {
                "series_list": tuple(c["series"] for c in CITY_CONFIG.values()),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    return df


def check_candle_coverage(start_date: date, end_date: date) -> pd.DataFrame:
    """Check 1-minute candle coverage."""
    logger.info("Checking 1-minute candle coverage...")

    with get_session() as session:
        query = text("""
            SELECT
                m.series_ticker,
                DATE(m.close_time AT TIME ZONE 'America/Chicago') as market_date,
                COUNT(DISTINCT m.ticker) as markets_with_candles,
                COUNT(c.market_ticker) as total_candles,
                AVG(
                    CASE WHEN c.market_ticker IS NOT NULL THEN 1 ELSE 0 END
                ) as avg_candles_per_market
            FROM markets m
            LEFT JOIN candles c ON m.ticker = c.market_ticker AND c.period_minutes = 1
            WHERE m.series_ticker IN :series_list
              AND m.close_time >= :start_date
              AND m.close_time <= :end_date
              AND m.status = 'settled'
            GROUP BY m.series_ticker, DATE(m.close_time AT TIME ZONE 'America/Chicago')
            ORDER BY m.series_ticker, market_date
        """)

        result = session.execute(
            query,
            {
                "series_list": tuple(c["series"] for c in CITY_CONFIG.values()),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    return df


def check_settlement_coverage(start_date: date, end_date: date) -> pd.DataFrame:
    """Check NWS settlement coverage."""
    logger.info("Checking NWS settlement coverage...")

    with get_session() as session:
        query = text("""
            SELECT
                loc_id,
                date_local,
                tmax_final_f,
                CASE
                    WHEN tmax_cli_f IS NOT NULL THEN 'CLI'
                    WHEN tmax_cf6_f IS NOT NULL THEN 'CF6'
                    WHEN tmax_iem_cf6_f IS NOT NULL THEN 'IEM_CF6'
                    WHEN tmax_ghcnd_f IS NOT NULL THEN 'GHCND'
                    ELSE 'None'
                END as source
            FROM wx.settlement
            WHERE loc_id IN :loc_ids
              AND date_local >= :start_date
              AND date_local <= :end_date
            ORDER BY loc_id, date_local
        """)

        result = session.execute(
            query,
            {
                "loc_ids": tuple(c["loc_id"] for c in CITY_CONFIG.values()),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    return df


def check_vc_coverage(start_date: date, end_date: date) -> pd.DataFrame:
    """Check VC minute coverage for good cities."""
    logger.info("Checking VC minute coverage...")

    # Get only cities with VC data
    vc_cities = {k: v for k, v in CITY_CONFIG.items() if v["has_vc"]}

    with get_session() as session:
        query = text("""
            SELECT
                loc_id,
                DATE(ts_utc AT TIME ZONE 'America/Chicago') as date_local,
                COUNT(*) as total_rows,
                SUM(CASE WHEN NOT ffilled THEN 1 ELSE 0 END) as real_rows,
                SUM(CASE WHEN ffilled THEN 1 ELSE 0 END) as ffilled_rows,
                ROUND(100.0 * SUM(CASE WHEN ffilled THEN 1 ELSE 0 END) / COUNT(*), 1) as ffilled_pct
            FROM wx.minute_obs
            WHERE loc_id IN :loc_ids
              AND DATE(ts_utc AT TIME ZONE 'America/Chicago') >= :start_date
              AND DATE(ts_utc AT TIME ZONE 'America/Chicago') <= :end_date
            GROUP BY loc_id, DATE(ts_utc AT TIME ZONE 'America/Chicago')
            ORDER BY loc_id, date_local
        """)

        result = session.execute(
            query,
            {
                "loc_ids": tuple(c["loc_id"] for c in vc_cities.values()),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

        df = pd.DataFrame(result.fetchall(), columns=result.keys())

    return df


def summarize_coverage(
    markets_df: pd.DataFrame,
    candles_df: pd.DataFrame,
    settlement_df: pd.DataFrame,
    vc_df: pd.DataFrame,
) -> None:
    """Print coverage summary."""
    logger.info("\n" + "="*60)
    logger.info("PHASE 4 COVERAGE SUMMARY")
    logger.info("="*60)

    # Markets summary
    logger.info("\n1. MARKETS")
    logger.info("-" * 40)
    for series in sorted(markets_df["series_ticker"].unique()):
        series_df = markets_df[markets_df["series_ticker"] == series]
        total_days = len(series_df)
        days_with_6_markets = len(series_df[series_df["market_count"] == 6])
        days_settled = len(series_df[series_df["settled_count"] == 6])

        logger.info(f"\n{series}:")
        logger.info(f"  Total days: {total_days}")
        logger.info(f"  Days with 6 markets: {days_with_6_markets} ({100*days_with_6_markets/total_days:.1f}%)")
        logger.info(f"  Days fully settled: {days_settled} ({100*days_settled/total_days:.1f}%)")

    # Candles summary
    logger.info("\n2. 1-MINUTE CANDLES")
    logger.info("-" * 40)
    for series in sorted(candles_df["series_ticker"].unique()):
        series_df = candles_df[candles_df["series_ticker"] == series]
        total_candles = series_df["total_candles"].sum()
        avg_per_day = series_df["total_candles"].mean()

        logger.info(f"\n{series}:")
        logger.info(f"  Total candles: {total_candles:,}")
        logger.info(f"  Avg candles/day: {avg_per_day:.1f}")

    # Settlement summary
    logger.info("\n3. NWS SETTLEMENT")
    logger.info("-" * 40)
    for loc_id in sorted(settlement_df["loc_id"].unique()):
        loc_df = settlement_df[settlement_df["loc_id"] == loc_id]
        total_days = len(loc_df)
        source_counts = loc_df["source"].value_counts()

        logger.info(f"\n{loc_id}:")
        logger.info(f"  Total days: {total_days}")
        logger.info(f"  Sources:")
        for source, count in source_counts.items():
            logger.info(f"    {source}: {count} ({100*count/total_days:.1f}%)")

    # VC summary
    logger.info("\n4. VC MINUTE DATA (6 cities)")
    logger.info("-" * 40)
    for loc_id in sorted(vc_df["loc_id"].unique()):
        loc_df = vc_df[vc_df["loc_id"] == loc_id]
        total_days = len(loc_df)
        complete_days = len(loc_df[loc_df["total_rows"] == 288])
        avg_ffilled = loc_df["ffilled_pct"].mean()

        logger.info(f"\n{loc_id}:")
        logger.info(f"  Total days: {total_days}")
        logger.info(f"  Complete days (288 rows): {complete_days} ({100*complete_days/total_days:.1f}%)")
        logger.info(f"  Avg forward-fill: {avg_ffilled:.1f}%")

    logger.info("\n" + "="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Check Phase 4 coverage"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-11-14",
        help="End date in YYYY-MM-DD format",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    logger.info(f"Checking coverage from {start_date} to {end_date}")

    # Run checks
    markets_df = check_market_coverage(start_date, end_date)
    candles_df = check_candle_coverage(start_date, end_date)
    settlement_df = check_settlement_coverage(start_date, end_date)
    vc_df = check_vc_coverage(start_date, end_date)

    # Summary
    summarize_coverage(markets_df, candles_df, settlement_df, vc_df)


if __name__ == "__main__":
    main()
