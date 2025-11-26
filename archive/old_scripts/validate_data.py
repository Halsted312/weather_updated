#!/usr/bin/env python3
"""
Data validation script for Kalshi weather ML pipeline.

Verifies infrastructure before building models:
- 1-min candles for Chicago markets
- 5-min weather data with ffill join
- Market metadata (close_time, strikes)
- Timezone handling
- Settlement data joins
"""

import logging
import argparse
from datetime import date, datetime, timedelta
from typing import Dict, List, Any
import pandas as pd
import pytz
from sqlalchemy import text

from db.connection import get_session

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validator for ML pipeline data requirements."""

    def __init__(self, city: str, start_date: date, end_date: date):
        self.city = city
        self.start_date = start_date
        self.end_date = end_date
        self.num_days = (end_date - start_date).days + 1
        self.warnings = []
        self.errors = []

        # City-specific config
        self.series_map = {
            "chicago": ("KXHIGHCHI%", "KMDW", "America/Chicago"),
        }

        if city not in self.series_map:
            raise ValueError(f"City {city} not supported yet. Available: {list(self.series_map.keys())}")

        self.series_pattern, self.station_icao, self.timezone = self.series_map[city]

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("="*60)
        logger.info("DATA VALIDATION REPORT")
        logger.info("="*60)
        logger.info(f"City: {self.city}")
        logger.info(f"Date range: {self.start_date} to {self.end_date} ({self.num_days} days)")
        logger.info(f"Series pattern: {self.series_pattern}")
        logger.info(f"Station: {self.station_icao}")
        logger.info(f"Timezone: {self.timezone}")
        logger.info("")

        results = {}

        # Run checks in sequence
        results['markets'] = self.check_markets()
        results['candles'] = self.check_candles()
        results['weather'] = self.check_weather()
        results['weather_join'] = self.check_weather_join()
        results['settlement'] = self.check_settlement()
        results['timezone'] = self.check_timezone()

        # Summary
        self.print_summary(results)

        return results

    def check_markets(self) -> Dict[str, Any]:
        """Validate markets table has required fields."""
        logger.info("\n" + "-"*60)
        logger.info("CHECK 1: Markets Table")
        logger.info("-"*60)

        with get_session() as session:
            query = text("""
                SELECT
                    COUNT(*) as total_markets,
                    COUNT(DISTINCT event_ticker) as num_events,
                    COUNT(DISTINCT DATE(expiration_time AT TIME ZONE :timezone)) as num_days,
                    SUM(CASE WHEN close_time IS NULL THEN 1 ELSE 0 END) as null_close_time,
                    SUM(CASE WHEN expiration_time IS NULL THEN 1 ELSE 0 END) as null_expiration_time,
                    SUM(CASE WHEN strike_type IS NULL THEN 1 ELSE 0 END) as null_strike_type,
                    SUM(CASE WHEN strike_type = 'greater' THEN 1 ELSE 0 END) as count_greater,
                    SUM(CASE WHEN strike_type = 'less' THEN 1 ELSE 0 END) as count_less,
                    SUM(CASE WHEN strike_type = 'between' THEN 1 ELSE 0 END) as count_between,
                    MIN(open_time) as earliest_open,
                    MAX(close_time) as latest_close
                FROM markets
                WHERE series_ticker LIKE :series_pattern
                  AND DATE(expiration_time AT TIME ZONE :timezone) >= :start_date
                  AND DATE(expiration_time AT TIME ZONE :timezone) <= :end_date
            """)

            result = session.execute(query, {
                'series_pattern': self.series_pattern,
                'timezone': self.timezone,
                'start_date': self.start_date,
                'end_date': self.end_date
            }).fetchone()

        stats = {
            'total_markets': result.total_markets or 0,
            'num_events': result.num_events or 0,
            'num_days': result.num_days or 0,
            'null_close_time': result.null_close_time or 0,
            'null_expiration_time': result.null_expiration_time or 0,
            'null_strike_type': result.null_strike_type or 0,
            'count_greater': result.count_greater or 0,
            'count_less': result.count_less or 0,
            'count_between': result.count_between or 0,
            'earliest_open': result.earliest_open,
            'latest_close': result.latest_close
        }

        logger.info(f"  Total markets: {stats['total_markets']}")
        logger.info(f"  Unique events: {stats['num_events']}")
        logger.info(f"  Days covered: {stats['num_days']}/{self.num_days}")
        logger.info(f"  Bracket types:")
        logger.info(f"    - greater: {stats['count_greater']}")
        logger.info(f"    - less: {stats['count_less']}")
        logger.info(f"    - between: {stats['count_between']}")

        # Check for issues
        if stats['total_markets'] == 0:
            self.errors.append("No markets found for specified date range")
            return stats  # Early return if no markets

        if stats['null_close_time'] > 0:
            self.warnings.append(f"{stats['null_close_time']} markets missing close_time")

        if stats['null_expiration_time'] > 0:
            self.warnings.append(f"{stats['null_expiration_time']} markets missing expiration_time")

        if stats['null_strike_type'] > 0:
            self.warnings.append(f"{stats['null_strike_type']} markets missing strike_type")

        # Sample a few markets
        with get_session() as session:
            query = text("""
                SELECT ticker, title, strike_type, floor_strike, cap_strike,
                       close_time, expiration_time
                FROM markets
                WHERE series_ticker LIKE :series_pattern
                  AND DATE(expiration_time AT TIME ZONE :timezone) >= :start_date
                  AND DATE(expiration_time AT TIME ZONE :timezone) <= :end_date
                ORDER BY expiration_time
                LIMIT 3
            """)
            sample = session.execute(query, {
                'series_pattern': self.series_pattern,
                'timezone': self.timezone,
                'start_date': self.start_date,
                'end_date': self.end_date
            }).fetchall()

        logger.info(f"\n  Sample markets (first 3):")
        for row in sample:
            logger.info(f"    {row.ticker}: {row.strike_type} [{row.floor_strike}, {row.cap_strike}]")
            logger.info(f"      close: {row.close_time}, expire: {row.expiration_time}")

        return stats

    def check_candles(self) -> Dict[str, Any]:
        """Validate 1-min candles exist."""
        logger.info("\n" + "-"*60)
        logger.info("CHECK 2: 1-Minute Candles")
        logger.info("-"*60)

        with get_session() as session:
            # Get markets for this period
            markets_query = text("""
                SELECT ticker, close_time
                FROM markets
                WHERE series_ticker LIKE :series_pattern
                  AND DATE(expiration_time AT TIME ZONE :timezone) >= :start_date
                  AND DATE(expiration_time AT TIME ZONE :timezone) <= :end_date
                LIMIT 5
            """)
            markets = session.execute(markets_query, {
                'series_pattern': self.series_pattern,
                'timezone': self.timezone,
                'start_date': self.start_date,
                'end_date': self.end_date
            }).fetchall()

            if not markets:
                self.errors.append("No markets found to check candles")
                return {'status': 'error'}

            # Check candles for first few markets
            sample_ticker = markets[0].ticker

            candles_query = text("""
                SELECT
                    COUNT(*) as total_candles,
                    MIN(timestamp) as earliest_candle,
                    MAX(timestamp) as latest_candle,
                    SUM(CASE WHEN open IS NULL THEN 1 ELSE 0 END) as null_open,
                    SUM(CASE WHEN high IS NULL THEN 1 ELSE 0 END) as null_high,
                    SUM(CASE WHEN low IS NULL THEN 1 ELSE 0 END) as null_low,
                    SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
                    AVG(high - low) as avg_spread_cents,
                    MAX(high - low) as max_spread_cents
                FROM candles
                WHERE market_ticker = :ticker
            """)

            candles_result = session.execute(candles_query, {
                'ticker': sample_ticker
            }).fetchone()

        stats = {
            'sample_ticker': sample_ticker,
            'total_candles': candles_result.total_candles or 0,
            'earliest_candle': candles_result.earliest_candle,
            'latest_candle': candles_result.latest_candle,
            'null_open': candles_result.null_open or 0,
            'null_high': candles_result.null_high or 0,
            'null_low': candles_result.null_low or 0,
            'null_close': candles_result.null_close or 0,
            'avg_spread_cents': candles_result.avg_spread_cents,
            'max_spread_cents': candles_result.max_spread_cents
        }

        logger.info(f"  Sample ticker: {stats['sample_ticker']}")
        logger.info(f"  Total candles: {stats['total_candles']}")
        logger.info(f"  Candle range: {stats['earliest_candle']} to {stats['latest_candle']}")
        logger.info(f"  OHLC completeness:")
        logger.info(f"    Missing open: {stats['null_open']}")
        logger.info(f"    Missing high: {stats['null_high']}")
        logger.info(f"    Missing low: {stats['null_low']}")
        logger.info(f"    Missing close: {stats['null_close']}")
        if stats['avg_spread_cents']:
            logger.info(f"  Spread (high-low): avg={stats['avg_spread_cents']:.1f}¢, max={stats['max_spread_cents']}¢")

        if stats['total_candles'] == 0:
            self.errors.append(f"No candles found for {sample_ticker}")

        # Check OHLC completeness
        total_missing = stats['null_open'] + stats['null_high'] + stats['null_low'] + stats['null_close']
        if total_missing > stats['total_candles'] * 0.1:
            self.warnings.append(f">10% of OHLC values missing ({total_missing}/{stats['total_candles']*4})")

        return stats

    def check_weather(self) -> Dict[str, Any]:
        """Validate 5-min weather data exists."""
        logger.info("\n" + "-"*60)
        logger.info("CHECK 3: 5-Minute Weather Data")
        logger.info("-"*60)

        with get_session() as session:
            query = text("""
                SELECT
                    COUNT(*) as total_obs,
                    COUNT(DISTINCT DATE(ts_utc AT TIME ZONE :timezone)) as num_days,
                    MIN(ts_utc) as earliest_obs,
                    MAX(ts_utc) as latest_obs,
                    SUM(CASE WHEN temp_f IS NULL THEN 1 ELSE 0 END) as null_temp,
                    SUM(CASE WHEN temp_f = 'NaN'::float THEN 1 ELSE 0 END) as nan_temp,
                    AVG(temp_f) FILTER (WHERE temp_f IS NOT NULL AND temp_f != 'NaN'::float) as avg_temp,
                    MIN(temp_f) FILTER (WHERE temp_f IS NOT NULL AND temp_f != 'NaN'::float) as min_temp,
                    MAX(temp_f) FILTER (WHERE temp_f IS NOT NULL AND temp_f != 'NaN'::float) as max_temp
                FROM wx.minute_obs
                WHERE loc_id = :station
                  AND DATE(ts_utc AT TIME ZONE :timezone) >= :start_date
                  AND DATE(ts_utc AT TIME ZONE :timezone) <= :end_date
            """)

            result = session.execute(query, {
                'station': self.station_icao,
                'timezone': self.timezone,
                'start_date': self.start_date,
                'end_date': self.end_date
            }).fetchone()

        stats = {
            'total_obs': result.total_obs,
            'num_days': result.num_days,
            'earliest_obs': result.earliest_obs,
            'latest_obs': result.latest_obs,
            'null_temp': result.null_temp,
            'nan_temp': result.nan_temp,
            'avg_temp': result.avg_temp,
            'min_temp': result.min_temp,
            'max_temp': result.max_temp
        }

        expected_obs_per_day = 288  # 24 * 60 / 5 = 288 5-min intervals
        expected_total = self.num_days * expected_obs_per_day
        coverage_pct = (stats['total_obs'] / expected_total * 100) if expected_total > 0 else 0

        logger.info(f"  Station: {self.station_icao}")
        logger.info(f"  Total observations: {stats['total_obs']:,}")
        logger.info(f"  Expected: {expected_total:,} ({expected_obs_per_day}/day × {self.num_days} days)")
        logger.info(f"  Coverage: {coverage_pct:.1f}%")
        logger.info(f"  Days covered: {stats['num_days']}/{self.num_days}")
        logger.info(f"  Temp range: {stats['min_temp']:.1f}°F to {stats['max_temp']:.1f}°F (avg: {stats['avg_temp']:.1f}°F)")
        logger.info(f"  Missing/NaN temps: {stats['null_temp'] + stats['nan_temp']}")

        if stats['total_obs'] == 0:
            self.errors.append("No weather observations found")

        if coverage_pct < 80:
            self.warnings.append(f"Weather data coverage only {coverage_pct:.1f}% (expected ~100%)")

        return stats

    def check_weather_join(self) -> Dict[str, Any]:
        """Test ffill ≤4min join from 5-min weather to 1-min grid."""
        logger.info("\n" + "-"*60)
        logger.info("CHECK 4: Weather FFill Join (5-min → 1-min)")
        logger.info("-"*60)

        # Test join for a single day
        test_date = self.start_date

        with get_session() as session:
            query = text("""
                WITH minute_grid AS (
                    SELECT ts FROM generate_series(
                        CAST(:start_dt AS timestamp),
                        CAST(:end_dt AS timestamp),
                        '1 minute'::interval
                    ) AS ts
                ),
                weather_5min AS (
                    SELECT ts_utc, temp_f
                    FROM wx.minute_obs
                    WHERE loc_id = :station
                      AND DATE(ts_utc AT TIME ZONE :timezone) = :test_date
                      AND temp_f IS NOT NULL
                      AND NOT (temp_f = 'NaN'::float)
                )
                SELECT
                    COUNT(*) as total_minutes,
                    COUNT(w.temp_f) as minutes_with_temp,
                    COUNT(*) - COUNT(w.temp_f) as minutes_missing_temp
                FROM minute_grid g
                LEFT JOIN LATERAL (
                    SELECT temp_f, ts_utc
                    FROM weather_5min
                    WHERE ts_utc <= g.ts
                    ORDER BY ts_utc DESC
                    LIMIT 1
                ) w ON (g.ts - w.ts_utc) <= INTERVAL '4 minutes'
            """)

            # Build timestamps for test day
            tz = pytz.timezone(self.timezone)
            start_dt = tz.localize(datetime.combine(test_date, datetime.min.time())).astimezone(pytz.UTC)
            end_dt = start_dt + timedelta(days=1) - timedelta(minutes=1)

            result = session.execute(query, {
                'start_dt': start_dt,
                'end_dt': end_dt,
                'station': self.station_icao,
                'timezone': self.timezone,
                'test_date': test_date
            }).fetchone()

        stats = {
            'test_date': test_date,
            'total_minutes': result.total_minutes,
            'minutes_with_temp': result.minutes_with_temp,
            'minutes_missing_temp': result.minutes_missing_temp
        }

        coverage_pct = (stats['minutes_with_temp'] / stats['total_minutes'] * 100) if stats['total_minutes'] > 0 else 0

        logger.info(f"  Test date: {test_date}")
        logger.info(f"  Total 1-min intervals: {stats['total_minutes']}")
        logger.info(f"  Intervals with temp (ffill ≤4min): {stats['minutes_with_temp']}")
        logger.info(f"  Intervals missing temp: {stats['minutes_missing_temp']}")
        logger.info(f"  Coverage: {coverage_pct:.1f}%")

        if coverage_pct < 95:
            self.warnings.append(f"FFill join coverage only {coverage_pct:.1f}% (target >95%)")

        return stats

    def check_settlement(self) -> Dict[str, Any]:
        """Validate settlement data joins correctly."""
        logger.info("\n" + "-"*60)
        logger.info("CHECK 5: Settlement Data")
        logger.info("-"*60)

        with get_session() as session:
            query = text("""
                SELECT
                    COUNT(DISTINCT s.date_local) as days_with_settlement,
                    COUNT(DISTINCT m.event_ticker) as num_events,
                    SUM(CASE WHEN s.tmax_final IS NULL THEN 1 ELSE 0 END) as events_missing_tmax,
                    AVG(s.tmax_final) FILTER (WHERE s.tmax_final IS NOT NULL) as avg_tmax,
                    MIN(s.tmax_final) FILTER (WHERE s.tmax_final IS NOT NULL) as min_tmax,
                    MAX(s.tmax_final) FILTER (WHERE s.tmax_final IS NOT NULL) as max_tmax,
                    array_agg(DISTINCT s.source_final) FILTER (WHERE s.source_final IS NOT NULL) as sources_used
                FROM markets m
                LEFT JOIN wx.settlement s ON (
                    s.city = :city
                    AND s.date_local = DATE(m.expiration_time AT TIME ZONE :timezone)
                )
                WHERE m.series_ticker LIKE :series_pattern
                  AND DATE(m.expiration_time AT TIME ZONE :timezone) >= :start_date
                  AND DATE(m.expiration_time AT TIME ZONE :timezone) <= :end_date
            """)

            result = session.execute(query, {
                'city': self.city,
                'series_pattern': self.series_pattern,
                'timezone': self.timezone,
                'start_date': self.start_date,
                'end_date': self.end_date
            }).fetchone()

        stats = {
            'days_with_settlement': result.days_with_settlement,
            'num_events': result.num_events,
            'events_missing_tmax': result.events_missing_tmax,
            'avg_tmax': result.avg_tmax,
            'min_tmax': result.min_tmax,
            'max_tmax': result.max_tmax,
            'sources_used': result.sources_used or []
        }

        coverage_pct = (stats['days_with_settlement'] / self.num_days * 100) if self.num_days > 0 else 0

        logger.info(f"  Days with settlement: {stats['days_with_settlement']}/{self.num_days} ({coverage_pct:.1f}%)")
        logger.info(f"  Events: {stats['num_events']}")
        logger.info(f"  Events missing tmax_final: {stats['events_missing_tmax']}")
        if stats['avg_tmax']:
            logger.info(f"  TMAX range: {stats['min_tmax']:.1f}°F to {stats['max_tmax']:.1f}°F (avg: {stats['avg_tmax']:.1f}°F)")
        logger.info(f"  Sources used: {', '.join(stats['sources_used'])}")

        if coverage_pct < 90:
            self.warnings.append(f"Settlement coverage only {coverage_pct:.1f}% (target >90%)")

        # Sample join
        with get_session() as session:
            query = text("""
                SELECT
                    m.ticker, m.strike_type, m.floor_strike, m.cap_strike,
                    s.date_local, s.tmax_final, s.source_final,
                    m.expiration_time
                FROM markets m
                LEFT JOIN wx.settlement s ON (
                    s.city = :city
                    AND s.date_local = DATE(m.expiration_time AT TIME ZONE :timezone)
                )
                WHERE m.series_ticker LIKE :series_pattern
                  AND DATE(m.expiration_time AT TIME ZONE :timezone) >= :start_date
                  AND DATE(m.expiration_time AT TIME ZONE :timezone) <= :end_date
                ORDER BY m.expiration_time
                LIMIT 3
            """)

            sample = session.execute(query, {
                'city': self.city,
                'series_pattern': self.series_pattern,
                'timezone': self.timezone,
                'start_date': self.start_date,
                'end_date': self.end_date
            }).fetchall()

        logger.info(f"\n  Sample settlement joins (first 3):")
        for row in sample:
            tmax_str = f"{row.tmax_final:.1f}°F" if row.tmax_final else "NULL"
            logger.info(f"    {row.ticker}: {row.strike_type} [{row.floor_strike}, {row.cap_strike}]")
            logger.info(f"      date_local: {row.date_local}, tmax: {tmax_str}, source: {row.source_final}")

        return stats

    def check_timezone(self) -> Dict[str, Any]:
        """Validate timezone handling."""
        logger.info("\n" + "-"*60)
        logger.info("CHECK 6: Timezone Handling")
        logger.info("-"*60)

        with get_session() as session:
            # Get a sample market
            query = text("""
                SELECT ticker, open_time, close_time, expiration_time
                FROM markets
                WHERE series_ticker LIKE :series_pattern
                  AND DATE(expiration_time AT TIME ZONE :timezone) >= :start_date
                  AND DATE(expiration_time AT TIME ZONE :timezone) <= :end_date
                LIMIT 1
            """)

            sample = session.execute(query, {
                'series_pattern': self.series_pattern,
                'timezone': self.timezone,
                'start_date': self.start_date,
                'end_date': self.end_date
            }).fetchone()

            if not sample:
                return {'status': 'no_data'}

            # Convert times
            query = text("""
                SELECT
                    CAST(:open_time AS timestamptz) as open_utc,
                    (CAST(:open_time AS timestamptz) AT TIME ZONE :timezone) as open_local,
                    CAST(:close_time AS timestamptz) as close_utc,
                    (CAST(:close_time AS timestamptz) AT TIME ZONE :timezone) as close_local,
                    DATE(CAST(:close_time AS timestamptz) AT TIME ZONE :timezone) as date_local
            """)

            result = session.execute(query, {
                'open_time': sample.open_time,
                'close_time': sample.close_time,
                'timezone': self.timezone
            }).fetchone()

        logger.info(f"  Sample ticker: {sample.ticker}")
        logger.info(f"  Open time:")
        logger.info(f"    UTC: {result.open_utc}")
        logger.info(f"    Local ({self.timezone}): {result.open_local}")
        logger.info(f"  Close time:")
        logger.info(f"    UTC: {result.close_utc}")
        logger.info(f"    Local ({self.timezone}): {result.close_local}")
        logger.info(f"  Date local: {result.date_local}")

        return {
            'sample_ticker': sample.ticker,
            'open_utc': result.open_utc,
            'open_local': result.open_local,
            'close_utc': result.close_utc,
            'close_local': result.close_local,
            'date_local': result.date_local
        }

    def print_summary(self, results: Dict[str, Any]):
        """Print validation summary."""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        if self.errors:
            logger.error(f"ERRORS: {len(self.errors)}")
            for err in self.errors:
                logger.error(f"  ❌ {err}")
        else:
            logger.info("✅ No errors found")

        if self.warnings:
            logger.warning(f"\nWARNINGS: {len(self.warnings)}")
            for warn in self.warnings:
                logger.warning(f"  ⚠️  {warn}")
        else:
            logger.info("✅ No warnings")

        if not self.errors and not self.warnings:
            logger.info("\n✅ ALL CHECKS PASSED - Ready for ML pipeline")
        elif not self.errors:
            logger.info("\n⚠️  CHECKS PASSED WITH WARNINGS - Review warnings before proceeding")
        else:
            logger.error("\n❌ VALIDATION FAILED - Fix errors before proceeding")

        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate data infrastructure for Kalshi weather ML pipeline"
    )
    parser.add_argument(
        '--city',
        type=str,
        default='chicago',
        help='City to validate (default: chicago)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    validator = DataValidator(args.city, start_date, end_date)
    results = validator.run_all_checks()

    # Exit with error code if validation failed
    if validator.errors:
        exit(1)


if __name__ == '__main__':
    main()
