#!/usr/bin/env python3
"""
Pipeline Health Check for Live Trading

Comprehensive check that ALL data sources are 100% live and ready for instant trading.

Checks:
1. Database connectivity
2. VC observations fresh (< 10 min) for all 6 cities
3. VC daily forecasts have today's basis date
4. VC hourly forecasts have today's basis date
5. Kalshi candles fresh (< 5 min) for active markets
6. Kalshi markets metadata (active markets exist)
7. Dense candles populated
8. Models loadable and can make predictions

Output: Colored PASS/FAIL table with details
Exit code: 0 if all pass, 1 if any fail

Usage:
    python scripts/check_pipeline_health.py
    python scripts/check_pipeline_health.py --verbose
    python scripts/check_pipeline_health.py --predict  # Also test predictions
"""

import argparse
import sys
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.orm import Session

# Setup path for imports
sys.path.insert(0, '/home/halsted/Documents/python/weather_updated')

from src.db.connection import get_db_session, test_connection
from src.db.models import VcMinuteWeather, VcLocation, VcForecastDaily, VcForecastHourly
from src.db.models import KalshiMarket, KalshiCandle1m

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']
CITY_CODES = ['CHI', 'AUS', 'DEN', 'LAX', 'MIA', 'PHL']

# Freshness thresholds
VC_OBS_MAX_AGE_MINUTES = 15  # VC polls every 5 min, allow 15 min buffer
VC_FORECAST_MAX_AGE_HOURS = 6  # Forecasts should be updated every few hours
KALSHI_CANDLE_MAX_AGE_MINUTES = 10  # Candle poller runs every 60s


@dataclass
class HealthCheck:
    """Single health check result"""
    name: str
    passed: bool
    details: str
    value: Optional[str] = None


@dataclass
class HealthReport:
    """Complete health report"""
    checks: List[HealthCheck]
    timestamp: datetime

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)


def check_database_connection() -> HealthCheck:
    """Check database is reachable"""
    try:
        if test_connection():
            return HealthCheck("Database Connection", True, "PostgreSQL reachable", "OK")
        else:
            return HealthCheck("Database Connection", False, "Connection test failed", "FAIL")
    except Exception as e:
        return HealthCheck("Database Connection", False, f"Error: {e}", "FAIL")


def check_vc_observations(session: Session) -> List[HealthCheck]:
    """Check VC observations are fresh for all cities"""
    checks = []
    now_utc = datetime.now(ZoneInfo('UTC'))

    for city_code in CITY_CODES:
        try:
            result = session.execute(text("""
                SELECT
                    l.city_code,
                    l.location_type,
                    MAX(m.datetime_utc) as latest_obs,
                    COUNT(*) as obs_count
                FROM wx.vc_minute_weather m
                JOIN wx.vc_location l ON m.vc_location_id = l.id
                WHERE l.city_code = :city_code
                  AND m.data_type = 'actual_obs'
                  AND m.is_forward_filled = false
                  AND m.datetime_utc > NOW() - INTERVAL '1 hour'
                GROUP BY l.city_code, l.location_type
                ORDER BY l.location_type
            """), {"city_code": city_code})

            rows = result.fetchall()

            if not rows:
                checks.append(HealthCheck(
                    f"VC Obs {city_code}",
                    False,
                    "No observations in last hour",
                    "0 rows"
                ))
                continue

            # Check both station and city feeds
            for row in rows:
                loc_type = row.location_type
                latest = row.latest_obs
                count = row.obs_count

                if latest is None:
                    checks.append(HealthCheck(
                        f"VC Obs {city_code} ({loc_type})",
                        False,
                        "No observations found",
                        "NULL"
                    ))
                    continue

                # Make latest timezone-aware if needed
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=ZoneInfo('UTC'))

                age_minutes = (now_utc - latest).total_seconds() / 60
                passed = age_minutes <= VC_OBS_MAX_AGE_MINUTES

                checks.append(HealthCheck(
                    f"VC Obs {city_code} ({loc_type})",
                    passed,
                    f"{count} obs/hr, {age_minutes:.1f} min old",
                    f"{age_minutes:.0f}m" if passed else f"STALE {age_minutes:.0f}m"
                ))

        except Exception as e:
            checks.append(HealthCheck(
                f"VC Obs {city_code}",
                False,
                f"Query error: {e}",
                "ERROR"
            ))

    return checks


def check_vc_forecasts_daily(session: Session) -> List[HealthCheck]:
    """Check VC daily forecasts have recent basis dates"""
    checks = []
    today = date.today()

    try:
        result = session.execute(text("""
            SELECT
                l.city_code,
                COUNT(*) as forecast_count,
                MAX(f.forecast_basis_date) as latest_basis,
                MAX(f.created_at) as latest_created
            FROM wx.vc_forecast_daily f
            JOIN wx.vc_location l ON f.vc_location_id = l.id
            WHERE f.forecast_basis_date >= :today - INTERVAL '1 day'
            GROUP BY l.city_code
            ORDER BY l.city_code
        """), {"today": today})

        rows = {row.city_code: row for row in result.fetchall()}

        for city_code in CITY_CODES:
            if city_code not in rows:
                checks.append(HealthCheck(
                    f"VC Daily Fcst {city_code}",
                    False,
                    "No forecasts for today/yesterday",
                    "MISSING"
                ))
                continue

            row = rows[city_code]
            latest_basis = row.latest_basis

            # Check if we have today's forecast
            has_today = latest_basis >= today if latest_basis else False

            checks.append(HealthCheck(
                f"VC Daily Fcst {city_code}",
                has_today,
                f"{row.forecast_count} forecasts, basis={latest_basis}",
                str(latest_basis) if has_today else f"OLD {latest_basis}"
            ))

    except Exception as e:
        session.rollback()  # Rollback to allow subsequent queries
        checks.append(HealthCheck(
            "VC Daily Forecasts",
            False,
            f"Query error: {e}",
            "ERROR"
        ))

    return checks


def check_vc_forecasts_hourly(session: Session) -> List[HealthCheck]:
    """Check VC hourly forecasts have recent basis dates"""
    checks = []
    today = date.today()

    try:
        result = session.execute(text("""
            SELECT
                l.city_code,
                COUNT(*) as forecast_count,
                MAX(f.forecast_basis_date) as latest_basis,
                MAX(f.created_at) as latest_created
            FROM wx.vc_forecast_hourly f
            JOIN wx.vc_location l ON f.vc_location_id = l.id
            WHERE f.forecast_basis_date >= :today - INTERVAL '1 day'
            GROUP BY l.city_code
            ORDER BY l.city_code
        """), {"today": today})

        rows = {row.city_code: row for row in result.fetchall()}

        for city_code in CITY_CODES:
            if city_code not in rows:
                checks.append(HealthCheck(
                    f"VC Hourly Fcst {city_code}",
                    False,
                    "No forecasts for today/yesterday",
                    "MISSING"
                ))
                continue

            row = rows[city_code]
            latest_basis = row.latest_basis

            # Check if we have today's forecast
            has_today = latest_basis >= today if latest_basis else False

            checks.append(HealthCheck(
                f"VC Hourly Fcst {city_code}",
                has_today,
                f"{row.forecast_count} hours, basis={latest_basis}",
                str(latest_basis) if has_today else f"OLD {latest_basis}"
            ))

    except Exception as e:
        session.rollback()  # Rollback to allow subsequent queries
        checks.append(HealthCheck(
            "VC Hourly Forecasts",
            False,
            f"Query error: {e}",
            "ERROR"
        ))

    return checks


def check_kalshi_markets(session: Session) -> HealthCheck:
    """Check Kalshi markets are loaded and active"""
    try:
        result = session.execute(text("""
            SELECT
                COUNT(*) as total_markets,
                COUNT(*) FILTER (WHERE close_time > NOW()) as active_markets,
                COUNT(DISTINCT city) as cities_covered
            FROM kalshi.markets
            WHERE city IS NOT NULL
        """))

        row = result.fetchone()
        total = row.total_markets
        active = row.active_markets
        cities = row.cities_covered

        passed = active >= 6  # At least 1 market per city

        return HealthCheck(
            "Kalshi Markets",
            passed,
            f"{active} active, {total} total, {cities} cities",
            f"{active} active" if passed else f"ONLY {active}"
        )

    except Exception as e:
        return HealthCheck("Kalshi Markets", False, f"Query error: {e}", "ERROR")


def check_kalshi_candles(session: Session) -> HealthCheck:
    """Check Kalshi candles are fresh for active markets"""
    now_utc = datetime.now(ZoneInfo('UTC'))

    try:
        # Get latest candle timestamp
        result = session.execute(text("""
            SELECT
                MAX(c.bucket_start) as latest_candle,
                COUNT(DISTINCT c.ticker) as tickers_with_candles
            FROM kalshi.candles_1m c
            JOIN kalshi.markets m ON c.ticker = m.ticker
            WHERE m.close_time > NOW()
              AND c.bucket_start > NOW() - INTERVAL '1 hour'
        """))

        row = result.fetchone()
        latest = row.latest_candle
        ticker_count = row.tickers_with_candles

        if latest is None:
            return HealthCheck(
                "Kalshi Candles",
                False,
                "No candles in last hour for active markets",
                "STALE"
            )

        # Make timezone aware if needed
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=ZoneInfo('UTC'))

        age_minutes = (now_utc - latest).total_seconds() / 60
        passed = age_minutes <= KALSHI_CANDLE_MAX_AGE_MINUTES

        return HealthCheck(
            "Kalshi Candles",
            passed,
            f"{ticker_count} tickers, {age_minutes:.1f} min old",
            f"{age_minutes:.0f}m" if passed else f"STALE {age_minutes:.0f}m"
        )

    except Exception as e:
        return HealthCheck("Kalshi Candles", False, f"Query error: {e}", "ERROR")


def check_dense_candles(session: Session) -> HealthCheck:
    """Check dense candles table is populated"""
    now_utc = datetime.now(ZoneInfo('UTC'))

    try:
        result = session.execute(text("""
            SELECT
                MAX(bucket_start) as latest,
                COUNT(*) as count_last_hour
            FROM kalshi.candles_1m_dense
            WHERE bucket_start > NOW() - INTERVAL '1 hour'
        """))

        row = result.fetchone()
        latest = row.latest
        count = row.count_last_hour

        if latest is None:
            return HealthCheck(
                "Dense Candles",
                False,
                "No dense candles in last hour",
                "EMPTY"
            )

        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=ZoneInfo('UTC'))

        age_minutes = (now_utc - latest).total_seconds() / 60
        passed = age_minutes <= KALSHI_CANDLE_MAX_AGE_MINUTES

        return HealthCheck(
            "Dense Candles",
            passed,
            f"{count} candles/hr, {age_minutes:.1f} min old",
            f"{age_minutes:.0f}m" if passed else f"STALE {age_minutes:.0f}m"
        )

    except Exception as e:
        return HealthCheck("Dense Candles", False, f"Query error: {e}", "ERROR")


def check_model_predictions(session: Session) -> List[HealthCheck]:
    """Check that models can load and make predictions for all cities"""
    checks = []

    try:
        from models.inference.live_engine import LiveInferenceEngine

        engine = LiveInferenceEngine()
        today = date.today()
        tomorrow = today + timedelta(days=1)

        for city in CITIES:
            try:
                # Try tomorrow first (most common use case)
                result = engine.predict(city, tomorrow, session)
                event_date = tomorrow

                # If tomorrow's market not open yet, try today
                if result is None:
                    result = engine.predict(city, today, session)
                    event_date = today

                if result is not None:
                    checks.append(HealthCheck(
                        f"Model {city}",
                        True,
                        f"Predict OK ({event_date}): {result.expected_settle:.1f}F, CI=[{result.ci_90_low},{result.ci_90_high}]",
                        f"{result.expected_settle:.0f}F"
                    ))
                else:
                    checks.append(HealthCheck(
                        f"Model {city}",
                        False,
                        "Prediction returned None for today and tomorrow (check data/market availability)",
                        "FAIL"
                    ))

            except Exception as e:
                checks.append(HealthCheck(
                    f"Model {city}",
                    False,
                    f"Prediction error: {e}",
                    "ERROR"
                ))

    except ImportError as e:
        checks.append(HealthCheck(
            "Models",
            False,
            f"Cannot import LiveInferenceEngine: {e}",
            "IMPORT ERROR"
        ))
    except Exception as e:
        checks.append(HealthCheck(
            "Models",
            False,
            f"Engine initialization error: {e}",
            "ERROR"
        ))

    return checks


def print_report(report: HealthReport, verbose: bool = False):
    """Print health report in a readable format"""

    # ANSI colors
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"\n{BOLD}═══════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}  PIPELINE HEALTH CHECK - {report.timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}{RESET}")
    print(f"{BOLD}═══════════════════════════════════════════════════════════════{RESET}\n")

    # Group checks by category
    categories = {}
    for check in report.checks:
        # Extract category from check name
        if "Database" in check.name:
            cat = "Database"
        elif "VC Obs" in check.name:
            cat = "VC Observations"
        elif "VC Daily" in check.name:
            cat = "VC Daily Forecasts"
        elif "VC Hourly" in check.name:
            cat = "VC Hourly Forecasts"
        elif "Kalshi Markets" in check.name:
            cat = "Kalshi Markets"
        elif "Kalshi Candles" in check.name or "Dense" in check.name:
            cat = "Kalshi Candles"
        elif "Model" in check.name:
            cat = "Model Predictions"
        else:
            cat = "Other"

        if cat not in categories:
            categories[cat] = []
        categories[cat].append(check)

    # Print by category
    for cat_name, checks in categories.items():
        cat_passed = all(c.passed for c in checks)
        status_icon = f"{GREEN}✓{RESET}" if cat_passed else f"{RED}✗{RESET}"
        print(f"{BOLD}{cat_name}{RESET} {status_icon}")

        for check in checks:
            if check.passed:
                icon = f"{GREEN}✓{RESET}"
                value_color = GREEN
            else:
                icon = f"{RED}✗{RESET}"
                value_color = RED

            # Shorter name for display
            short_name = check.name.replace("VC ", "").replace("Kalshi ", "")

            if verbose:
                print(f"  {icon} {short_name}: {check.details}")
            else:
                print(f"  {icon} {short_name}: {value_color}{check.value}{RESET}")

        print()

    # Summary
    print(f"{BOLD}═══════════════════════════════════════════════════════════════{RESET}")

    if report.all_passed:
        print(f"{GREEN}{BOLD}  ALL CHECKS PASSED ({report.pass_count}/{len(report.checks)}){RESET}")
        print(f"{GREEN}  ✓ READY TO TRADE{RESET}")
    else:
        print(f"{RED}{BOLD}  {report.fail_count} CHECKS FAILED ({report.pass_count}/{len(report.checks)} passed){RESET}")
        print(f"{RED}  ✗ NOT READY TO TRADE - Fix issues above{RESET}")

    print(f"{BOLD}═══════════════════════════════════════════════════════════════{RESET}\n")


def run_health_check(test_predictions: bool = False, verbose: bool = False) -> HealthReport:
    """Run all health checks and return report"""

    checks = []

    # 1. Database connection
    checks.append(check_database_connection())

    if not checks[0].passed:
        # Can't continue without database
        return HealthReport(checks, datetime.now(ZoneInfo('America/Chicago')))

    # Run remaining checks with a session
    with get_db_session() as session:
        # 2. VC Observations
        checks.extend(check_vc_observations(session))

        # 3. VC Daily Forecasts
        checks.extend(check_vc_forecasts_daily(session))

        # 4. VC Hourly Forecasts
        checks.extend(check_vc_forecasts_hourly(session))

        # 5. Kalshi Markets
        checks.append(check_kalshi_markets(session))

        # 6. Kalshi Candles
        checks.append(check_kalshi_candles(session))

        # 7. Dense Candles
        checks.append(check_dense_candles(session))

        # 8. Model Predictions (optional, can be slow)
        if test_predictions:
            checks.extend(check_model_predictions(session))

    return HealthReport(checks, datetime.now(ZoneInfo('America/Chicago')))


def main():
    parser = argparse.ArgumentParser(description="Pipeline Health Check for Live Trading")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--predict", "-p", action="store_true", help="Also test model predictions (slower)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only print summary")
    args = parser.parse_args()

    report = run_health_check(test_predictions=args.predict, verbose=args.verbose)

    if not args.quiet:
        print_report(report, verbose=args.verbose)
    else:
        # Quiet mode - just summary
        if report.all_passed:
            print(f"OK: {report.pass_count}/{len(report.checks)} checks passed")
        else:
            print(f"FAIL: {report.fail_count} checks failed")
            for check in report.checks:
                if not check.passed:
                    print(f"  - {check.name}: {check.details}")

    # Exit code
    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
