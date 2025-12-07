#!/usr/bin/env python3
"""Extract raw data from DB to parquet files for fast machine processing.

This script dumps all necessary raw data for a city to parquet files:
- VC observations (5-min temps)
- Settlements
- Daily forecasts (T-1 to T-6)
- Hourly forecasts
- NOAA guidance (NBM, HRRR, NDFD)
- City observations (for station-city gap)

The fast machine can then load these parquets and run feature engineering
without needing database access.

Usage:
    # Extract all raw data for austin
    PYTHONPATH=. python scripts/extract_raw_data_to_parquet.py --city austin

    # Extract with custom date range
    PYTHONPATH=. python scripts/extract_raw_data_to_parquet.py --city austin \
        --start-date 2023-01-01 --end-date 2025-12-01

Output:
    models/raw_data/{city}/
        vc_observations.parquet      # 5-min station observations
        vc_city_observations.parquet # 5-min city aggregate observations
        settlements.parquet          # Daily settlements
        forecasts_daily.parquet      # Daily forecasts (all lead days)
        forecasts_hourly.parquet     # Hourly forecasts
        noaa_guidance.parquet        # NBM, HRRR, NDFD guidance
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_vc_observations(engine, city_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Extract VC 5-minute observations for station."""
    query = text("""
        SELECT
            vm.datetime_local,
            vm.datetime_utc,
            vm.temp_f,
            vm.humidity,
            vm.dew_f,
            vm.windspeed_mph,
            vm.windgust_mph,
            vm.winddir,
            vm.cloudcover,
            vm.pressure_mb,
            vm.visibility_miles,
            vm.conditions,
            vm.solarradiation
        FROM wx.vc_minute_weather vm
        JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
        WHERE vl.city_code = :city_code
          AND vl.location_type = 'station'
          AND vm.data_type = 'actual_obs'
          AND vm.datetime_local >= :start_date
          AND vm.datetime_local < :end_date_plus
        ORDER BY vm.datetime_local
    """)

    df = pd.read_sql(query, engine, params={
        "city_code": city_code,
        "start_date": start_date,
        "end_date_plus": end_date + timedelta(days=1),
    })
    logger.info(f"Extracted {len(df):,} station observations")
    return df


def extract_city_observations(engine, city_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Extract VC 5-minute observations for city aggregate."""
    query = text("""
        SELECT
            vm.datetime_local,
            vm.datetime_utc,
            vm.temp_f,
            vm.humidity,
            vm.dew_f,
            vm.windspeed_mph,
            vm.cloudcover
        FROM wx.vc_minute_weather vm
        JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
        WHERE vl.city_code = :city_code
          AND vl.location_type = 'city'
          AND vm.data_type = 'actual_obs'
          AND vm.datetime_local >= :start_date
          AND vm.datetime_local < :end_date_plus
        ORDER BY vm.datetime_local
    """)

    df = pd.read_sql(query, engine, params={
        "city_code": city_code,
        "start_date": start_date,
        "end_date_plus": end_date + timedelta(days=1),
    })
    logger.info(f"Extracted {len(df):,} city observations")
    return df


def extract_settlements(engine, city_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Extract settlement data."""
    query = text("""
        SELECT
            date_local,
            tmax_final,
            source_final,
            tmax_cli_f,
            tmax_iem_f,
            tmax_ncei_f
        FROM wx.settlement
        WHERE city = :city_id
          AND date_local >= :start_date
          AND date_local <= :end_date
        ORDER BY date_local
    """)

    df = pd.read_sql(query, engine, params={
        "city_id": city_id,
        "start_date": start_date,
        "end_date": end_date,
    })
    logger.info(f"Extracted {len(df):,} settlements")
    return df


def extract_daily_forecasts(engine, city_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Extract daily forecasts (all lead days T-0 to T-6)."""
    query = text("""
        SELECT
            fd.target_date,
            fd.forecast_basis_date,
            fd.lead_days,
            fd.tempmax_f,
            fd.tempmin_f,
            fd.temp_f,
            fd.humidity,
            fd.precip_in,
            fd.precipprob,
            fd.windspeed_mph,
            fd.windgust_mph,
            fd.cloudcover,
            fd.conditions
        FROM wx.vc_forecast_daily fd
        JOIN wx.vc_location vl ON fd.vc_location_id = vl.id
        WHERE vl.city_code = :city_code
          AND vl.location_type = 'city'
          AND fd.data_type = 'historical_forecast'
          AND fd.target_date >= :start_date
          AND fd.target_date <= :end_date
        ORDER BY fd.target_date, fd.lead_days
    """)

    df = pd.read_sql(query, engine, params={
        "city_code": city_code,
        "start_date": start_date,
        "end_date": end_date,
    })
    logger.info(f"Extracted {len(df):,} daily forecasts")
    return df


def extract_hourly_forecasts(engine, city_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Extract hourly forecasts (T-1 only for simplicity)."""
    query = text("""
        SELECT
            fh.target_datetime_local,
            fh.target_datetime_utc,
            fh.forecast_basis_date,
            fh.lead_hours,
            fh.temp_f,
            fh.humidity,
            fh.dew_f,
            fh.precip_in,
            fh.precipprob,
            fh.windspeed_mph,
            fh.cloudcover,
            fh.conditions
        FROM wx.vc_forecast_hourly fh
        JOIN wx.vc_location vl ON fh.vc_location_id = vl.id
        WHERE vl.city_code = :city_code
          AND vl.location_type = 'city'
          AND fh.data_type = 'historical_forecast'
          AND DATE(fh.target_datetime_local) >= :start_date
          AND DATE(fh.target_datetime_local) <= :end_date
        ORDER BY fh.target_datetime_local
    """)

    df = pd.read_sql(query, engine, params={
        "city_code": city_code,
        "start_date": start_date,
        "end_date": end_date,
    })
    logger.info(f"Extracted {len(df):,} hourly forecasts")
    return df


def extract_noaa_guidance(engine, city_id: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Extract NOAA model guidance (NBM, HRRR, NDFD)."""
    query = text("""
        SELECT
            target_date,
            model,
            run_datetime_utc,
            peak_window_max_f,
            timezone
        FROM wx.weather_more_apis_guidance
        WHERE city_id = :city_id
          AND target_date >= :start_date
          AND target_date <= :end_date
        ORDER BY target_date, model, run_datetime_utc
    """)

    df = pd.read_sql(query, engine, params={
        "city_id": city_id,
        "start_date": start_date,
        "end_date": end_date,
    })
    logger.info(f"Extracted {len(df):,} NOAA guidance records")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract raw data from DB to parquet files"
    )
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"],
        help="City to extract data for",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: earliest available",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Default: latest available",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/raw_data",
        help="Output directory (default: models/raw_data)",
    )
    args = parser.parse_args()

    from src.db.connection import get_engine, get_db_session
    from src.config.cities import get_city
    from models.data.loader import get_available_date_range

    city_config = get_city(args.city)
    city_code = city_config.city_code  # e.g., 'AUS'
    city_id = args.city  # e.g., 'austin'

    engine = get_engine()

    # Determine date range
    if args.start_date and args.end_date:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    else:
        with get_db_session() as session:
            start_date, end_date = get_available_date_range(session, city_id)
        if start_date is None:
            logger.error(f"No data available for {city_id}")
            return 1

    logger.info("=" * 60)
    logger.info(f"RAW DATA EXTRACTION: {city_id.upper()}")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"City code: {city_code}")

    # Create output directory
    output_dir = Path(args.output_dir) / city_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Extract all data
    print("\n--- Extracting data ---")

    # 1. Station observations
    logger.info("\n[1/6] Extracting station observations...")
    df_obs = extract_vc_observations(engine, city_code, start_date, end_date)
    obs_path = output_dir / "vc_observations.parquet"
    df_obs.to_parquet(obs_path, index=False)
    logger.info(f"  Saved to {obs_path}")

    # 2. City observations
    logger.info("\n[2/6] Extracting city observations...")
    df_city_obs = extract_city_observations(engine, city_code, start_date, end_date)
    city_obs_path = output_dir / "vc_city_observations.parquet"
    df_city_obs.to_parquet(city_obs_path, index=False)
    logger.info(f"  Saved to {city_obs_path}")

    # 3. Settlements
    logger.info("\n[3/6] Extracting settlements...")
    df_settle = extract_settlements(engine, city_id, start_date, end_date)
    settle_path = output_dir / "settlements.parquet"
    df_settle.to_parquet(settle_path, index=False)
    logger.info(f"  Saved to {settle_path}")

    # 4. Daily forecasts
    logger.info("\n[4/6] Extracting daily forecasts...")
    df_fcst_daily = extract_daily_forecasts(engine, city_code, start_date, end_date)
    fcst_daily_path = output_dir / "forecasts_daily.parquet"
    df_fcst_daily.to_parquet(fcst_daily_path, index=False)
    logger.info(f"  Saved to {fcst_daily_path}")

    # 5. Hourly forecasts
    logger.info("\n[5/6] Extracting hourly forecasts...")
    df_fcst_hourly = extract_hourly_forecasts(engine, city_code, start_date, end_date)
    fcst_hourly_path = output_dir / "forecasts_hourly.parquet"
    df_fcst_hourly.to_parquet(fcst_hourly_path, index=False)
    logger.info(f"  Saved to {fcst_hourly_path}")

    # 6. NOAA guidance
    logger.info("\n[6/6] Extracting NOAA guidance...")
    df_noaa = extract_noaa_guidance(engine, city_id, start_date, end_date)
    noaa_path = output_dir / "noaa_guidance.parquet"
    df_noaa.to_parquet(noaa_path, index=False)
    logger.info(f"  Saved to {noaa_path}")

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    total_size = 0
    for f in sorted(output_dir.glob("*.parquet")):
        size_mb = f.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"\nTotal size: {total_size:.1f} MB")

    print(f"\nDate range: {start_date} to {end_date}")
    print(f"Days: {(end_date - start_date).days + 1}")

    print("\n--- Files to copy to fast machine ---")
    print(f"  {output_dir}/")
    print(f"  models/candles/candles_{city_id}.parquet")

    print("\n--- On fast machine, run ---")
    print(f"  PYTHONPATH=. python scripts/build_dataset_from_parquets.py --city {city_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
