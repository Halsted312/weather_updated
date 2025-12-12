#!/usr/bin/env python3
"""
Quick validation script for 15-minute historical forecast ingestion.

Validates that minute-level forecast data was ingested correctly.
"""

import sys
from datetime import date

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from sqlalchemy import func, cast, Date, text
from src.db import get_db_session, VcLocation, VcMinuteWeather


def validate_minute_forecast_ingestion(
    city_code: str,
    start_date: date,
    end_date: date,
):
    """Validate minute-level historical forecast ingestion."""

    print(f"Validating 15-min forecast ingestion for {city_code}")
    print(f"Date range: {start_date} to {end_date}")
    print("=" * 80)
    print()

    with get_db_session() as session:
        # Get location ID
        location = session.query(VcLocation).filter(
            VcLocation.city_code == city_code,
            VcLocation.location_type == "city",
        ).first()

        if not location:
            print(f"❌ ERROR: No VcLocation found for {city_code}/city")
            return

        print(f"✅ Location found: {location.vc_location_query}")
        print(f"   IANA Timezone: {location.iana_timezone}")
        print()

        # Query minute forecast data
        query = session.query(
            func.count().label("n_rows"),
            func.min(VcMinuteWeather.datetime_local).label("min_local"),
            func.max(VcMinuteWeather.datetime_local).label("max_local"),
            func.min(VcMinuteWeather.forecast_basis_date).label("min_basis"),
            func.max(VcMinuteWeather.forecast_basis_date).label("max_basis"),
            func.avg(VcMinuteWeather.temp_f).label("avg_temp"),
            func.count().filter(VcMinuteWeather.temp_f.is_(None)).label("null_temps"),
            func.count().filter(VcMinuteWeather.humidity.is_(None)).label("null_humidity"),
            func.count().filter(VcMinuteWeather.cloudcover.is_(None)).label("null_cloudcover"),
        ).filter(
            VcMinuteWeather.vc_location_id == location.id,
            VcMinuteWeather.data_type == "historical_forecast",
            cast(VcMinuteWeather.datetime_local, Date) >= start_date,
            cast(VcMinuteWeather.datetime_local, Date) <= end_date,
        )

        result = query.one()

        # Print results
        print("📊 Query Results:")
        print(f"   Total rows: {result.n_rows}")
        print(f"   Min datetime_local: {result.min_local}")
        print(f"   Max datetime_local: {result.max_local}")
        print(f"   Min basis_date: {result.min_basis}")
        print(f"   Max basis_date: {result.max_basis}")
        print(f"   Avg temperature: {result.avg_temp:.2f}°F" if result.avg_temp else "   Avg temperature: N/A")
        print()

        print("🔍 Data Quality:")
        print(f"   Null temps: {result.null_temps} / {result.n_rows}")
        print(f"   Null humidity: {result.null_humidity} / {result.n_rows}")
        print(f"   Null cloudcover: {result.null_cloudcover} / {result.n_rows}")
        print()

        # Sample rows
        print("📋 Sample Rows (first 5):")
        sample_query = session.query(VcMinuteWeather).filter(
            VcMinuteWeather.vc_location_id == location.id,
            VcMinuteWeather.data_type == "historical_forecast",
            cast(VcMinuteWeather.datetime_local, Date) >= start_date,
            cast(VcMinuteWeather.datetime_local, Date) <= end_date,
        ).order_by(VcMinuteWeather.datetime_local).limit(5)

        for i, row in enumerate(sample_query, 1):
            print(f"   Row {i}:")
            print(f"     datetime_local: {row.datetime_local}")
            print(f"     basis_date: {row.forecast_basis_date}")
            print(f"     lead_hours: {row.lead_hours}")
            print(f"     temp: {row.temp_f}°F, humidity: {row.humidity}%, dew: {row.dew_f}°F")
            print(f"     windspeed: {row.windspeed_mph}mph, pressure: {row.pressure_mb}mb")
            print()

        # Validation checks
        print("=" * 80)
        print("VALIDATION SUMMARY:")
        print("=" * 80)

        days_span = (end_date - start_date).days + 1
        expected_rows = days_span * 96  # 96 minutes per day (15-min intervals)

        checks = []

        if result.n_rows > 0:
            checks.append(("✅", f"Data ingested: {result.n_rows} rows"))
        else:
            checks.append(("❌", "No data found"))

        if result.n_rows >= expected_rows * 0.9:  # Allow 10% variance
            checks.append(("✅", f"Row count reasonable: {result.n_rows} vs ~{expected_rows} expected"))
        else:
            checks.append(("⚠️ ", f"Row count low: {result.n_rows} vs ~{expected_rows} expected"))

        if result.null_temps == 0:
            checks.append(("✅", "No null temperatures"))
        else:
            checks.append(("❌", f"{result.null_temps} null temperatures found"))

        if result.avg_temp and 30 <= result.avg_temp <= 100:
            checks.append(("✅", f"Temperature range reasonable: avg {result.avg_temp:.1f}°F"))
        else:
            checks.append(("⚠️ ", f"Temperature range suspicious: avg {result.avg_temp}°F"))

        if result.min_basis and result.max_basis:
            expected_min_basis = start_date.replace(day=1) - timedelta(days=1)  # T-1 of first day
            checks.append(("✅", f"Basis dates present: {result.min_basis} to {result.max_basis}"))

        for icon, msg in checks:
            print(f"{icon} {msg}")

        print()
        if all(check[0] == "✅" for check in checks):
            print("🎉 ALL VALIDATIONS PASSED!")
        else:
            print("⚠️  Some validations need attention")


if __name__ == "__main__":
    from datetime import timedelta

    # Validate Austin November 1-7
    validate_minute_forecast_ingestion(
        city_code="AUS",
        start_date=date(2024, 11, 1),
        end_date=date(2024, 11, 7),
    )
