"""Add Visual Crossing greenfield tables.

Revision ID: 007
Revises: 006
Create Date: 2025-11-28

Creates new VC schema tables:
- wx.vc_location: Location dimension (station + city for each market city)
- wx.vc_minute_weather: Minute-level observations/forecasts (TimescaleDB hypertable)
- wx.vc_forecast_daily: Daily forecast snapshots
- wx.vc_forecast_hourly: Hourly forecast snapshots

Includes CHECK constraints, UNIQUE constraints, indexes, and seed data.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ==========================================================================
    # wx.vc_location - Location dimension table
    # ==========================================================================
    op.create_table(
        "vc_location",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("city_code", sa.Text(), nullable=False),  # 'CHI', 'DEN', etc.
        sa.Column("kalshi_code", sa.Text(), nullable=False),  # 'CHI', 'PHIL', etc.
        sa.Column("location_type", sa.Text(), nullable=False),  # 'station' | 'city'
        sa.Column("vc_location_query", sa.Text(), nullable=False),  # 'stn:KMDW' or 'Chicago,IL'
        sa.Column("station_id", sa.Text(), nullable=True),  # 'KMDW' (null for city type)
        sa.Column("iana_timezone", sa.Text(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("elevation_m", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("city_code", "location_type", name="uq_vc_location_city_type"),
        sa.UniqueConstraint("vc_location_query", name="uq_vc_location_query"),
        sa.CheckConstraint("location_type IN ('station', 'city')", name="ck_vc_location_type"),
        schema="wx",
    )

    # ==========================================================================
    # wx.vc_minute_weather - Minute-level weather data (fact table, HYPERTABLE)
    # ==========================================================================
    op.create_table(
        "vc_minute_weather",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("vc_location_id", sa.Integer(), sa.ForeignKey("wx.vc_location.id"), nullable=False),
        # Classification
        sa.Column("data_type", sa.Text(), nullable=False),  # 'actual_obs' | 'current_snapshot' | 'forecast' | 'historical_forecast'
        sa.Column("forecast_basis_date", sa.Date(), nullable=True),
        sa.Column("forecast_basis_datetime_utc", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lead_hours", sa.Integer(), nullable=True),
        # Time fields
        sa.Column("datetime_epoch_utc", sa.BigInteger(), nullable=False),
        sa.Column("datetime_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("datetime_local", sa.DateTime(), nullable=False),
        sa.Column("timezone", sa.Text(), nullable=False),
        sa.Column("tzoffset_minutes", sa.SmallInteger(), nullable=False),
        # Core weather
        sa.Column("temp_f", sa.Float(), nullable=True),
        sa.Column("tempmax_f", sa.Float(), nullable=True),
        sa.Column("tempmin_f", sa.Float(), nullable=True),
        sa.Column("feelslike_f", sa.Float(), nullable=True),
        sa.Column("feelslikemax_f", sa.Float(), nullable=True),
        sa.Column("feelslikemin_f", sa.Float(), nullable=True),
        sa.Column("dew_f", sa.Float(), nullable=True),
        sa.Column("humidity", sa.Float(), nullable=True),
        # Precipitation
        sa.Column("precip_in", sa.Float(), nullable=True),
        sa.Column("precipprob", sa.Float(), nullable=True),
        sa.Column("preciptype", sa.Text(), nullable=True),
        sa.Column("precipcover", sa.Float(), nullable=True),
        sa.Column("snow_in", sa.Float(), nullable=True),
        sa.Column("snowdepth_in", sa.Float(), nullable=True),
        sa.Column("precipremote", sa.Float(), nullable=True),
        # Wind (10m)
        sa.Column("windspeed_mph", sa.Float(), nullable=True),
        sa.Column("windgust_mph", sa.Float(), nullable=True),
        sa.Column("winddir", sa.Float(), nullable=True),
        sa.Column("windspeedmean_mph", sa.Float(), nullable=True),
        sa.Column("windspeedmin_mph", sa.Float(), nullable=True),
        sa.Column("windspeedmax_mph", sa.Float(), nullable=True),
        # Extended wind (50/80/100m)
        sa.Column("windspeed50_mph", sa.Float(), nullable=True),
        sa.Column("winddir50", sa.Float(), nullable=True),
        sa.Column("windspeed80_mph", sa.Float(), nullable=True),
        sa.Column("winddir80", sa.Float(), nullable=True),
        sa.Column("windspeed100_mph", sa.Float(), nullable=True),
        sa.Column("winddir100", sa.Float(), nullable=True),
        # Atmosphere
        sa.Column("cloudcover", sa.Float(), nullable=True),
        sa.Column("visibility_miles", sa.Float(), nullable=True),
        sa.Column("pressure_mb", sa.Float(), nullable=True),
        # Solar/Radiation
        sa.Column("uvindex", sa.Float(), nullable=True),
        sa.Column("solarradiation", sa.Float(), nullable=True),
        sa.Column("solarenergy", sa.Float(), nullable=True),
        sa.Column("dniradiation", sa.Float(), nullable=True),
        sa.Column("difradiation", sa.Float(), nullable=True),
        sa.Column("ghiradiation", sa.Float(), nullable=True),
        sa.Column("gtiradiation", sa.Float(), nullable=True),
        sa.Column("sunelevation", sa.Float(), nullable=True),
        sa.Column("sunazimuth", sa.Float(), nullable=True),
        # Instability/Energy
        sa.Column("cape", sa.Float(), nullable=True),
        sa.Column("cin", sa.Float(), nullable=True),
        sa.Column("deltat", sa.Float(), nullable=True),
        sa.Column("degreedays", sa.Float(), nullable=True),
        sa.Column("accdegreedays", sa.Float(), nullable=True),
        # Text/Flags
        sa.Column("conditions", sa.Text(), nullable=True),
        sa.Column("icon", sa.Text(), nullable=True),
        sa.Column("stations", sa.Text(), nullable=True),
        sa.Column("resolved_address", sa.Text(), nullable=True),
        # Metadata
        sa.Column("source_system", sa.Text(), server_default="vc_timeline", nullable=True),
        sa.Column("raw_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        # TimescaleDB requires time column in primary key for hypertables
        sa.PrimaryKeyConstraint("id", "datetime_utc"),
        sa.CheckConstraint(
            "data_type IN ('actual_obs', 'current_snapshot', 'forecast', 'historical_forecast')",
            name="ck_vc_minute_data_type",
        ),
        schema="wx",
    )

    # Partial unique indexes for idempotent UPSERTs
    # Obs index: for actual_obs where forecast_basis_date IS NULL
    op.execute("""
        CREATE UNIQUE INDEX uq_vc_minute_obs
        ON wx.vc_minute_weather (vc_location_id, data_type, datetime_utc)
        WHERE forecast_basis_date IS NULL
    """)

    # Forecast index: for forecasts where forecast_basis_date IS NOT NULL
    op.execute("""
        CREATE UNIQUE INDEX uq_vc_minute_fcst
        ON wx.vc_minute_weather (vc_location_id, data_type, forecast_basis_date, datetime_utc)
        WHERE forecast_basis_date IS NOT NULL
    """)

    # Query indexes
    op.create_index(
        "ix_vc_minute_weather_loc_datetime",
        "vc_minute_weather",
        ["vc_location_id", "datetime_utc"],
        schema="wx",
    )
    op.create_index(
        "ix_vc_minute_weather_loc_type_basis",
        "vc_minute_weather",
        ["vc_location_id", "data_type", "forecast_basis_date"],
        schema="wx",
    )

    # Convert to TimescaleDB hypertable
    op.execute("""
        SELECT create_hypertable(
            'wx.vc_minute_weather',
            'datetime_utc',
            chunk_time_interval => INTERVAL '7 days',
            if_not_exists => TRUE
        )
    """)

    # ==========================================================================
    # wx.vc_forecast_daily - Daily forecast snapshots
    # ==========================================================================
    op.create_table(
        "vc_forecast_daily",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("vc_location_id", sa.Integer(), sa.ForeignKey("wx.vc_location.id"), nullable=False),
        # Classification
        sa.Column("data_type", sa.Text(), nullable=False),  # 'forecast' | 'historical_forecast'
        sa.Column("forecast_basis_date", sa.Date(), nullable=False),
        sa.Column("forecast_basis_datetime_utc", sa.DateTime(timezone=True), nullable=True),
        # Target
        sa.Column("target_date", sa.Date(), nullable=False),
        sa.Column("lead_days", sa.Integer(), nullable=False),
        # Weather fields
        sa.Column("tempmax_f", sa.Float(), nullable=True),
        sa.Column("tempmin_f", sa.Float(), nullable=True),
        sa.Column("temp_f", sa.Float(), nullable=True),
        sa.Column("feelslikemax_f", sa.Float(), nullable=True),
        sa.Column("feelslikemin_f", sa.Float(), nullable=True),
        sa.Column("feelslike_f", sa.Float(), nullable=True),
        sa.Column("dew_f", sa.Float(), nullable=True),
        sa.Column("humidity", sa.Float(), nullable=True),
        sa.Column("precip_in", sa.Float(), nullable=True),
        sa.Column("precipprob", sa.Float(), nullable=True),
        sa.Column("preciptype", sa.Text(), nullable=True),
        sa.Column("precipcover", sa.Float(), nullable=True),
        sa.Column("snow_in", sa.Float(), nullable=True),
        sa.Column("snowdepth_in", sa.Float(), nullable=True),
        sa.Column("windspeed_mph", sa.Float(), nullable=True),
        sa.Column("windgust_mph", sa.Float(), nullable=True),
        sa.Column("winddir", sa.Float(), nullable=True),
        sa.Column("windspeedmean_mph", sa.Float(), nullable=True),
        sa.Column("windspeedmin_mph", sa.Float(), nullable=True),
        sa.Column("windspeedmax_mph", sa.Float(), nullable=True),
        sa.Column("windspeed50_mph", sa.Float(), nullable=True),
        sa.Column("winddir50", sa.Float(), nullable=True),
        sa.Column("windspeed80_mph", sa.Float(), nullable=True),
        sa.Column("winddir80", sa.Float(), nullable=True),
        sa.Column("windspeed100_mph", sa.Float(), nullable=True),
        sa.Column("winddir100", sa.Float(), nullable=True),
        sa.Column("cloudcover", sa.Float(), nullable=True),
        sa.Column("visibility_miles", sa.Float(), nullable=True),
        sa.Column("pressure_mb", sa.Float(), nullable=True),
        sa.Column("uvindex", sa.Float(), nullable=True),
        sa.Column("solarradiation", sa.Float(), nullable=True),
        sa.Column("solarenergy", sa.Float(), nullable=True),
        sa.Column("dniradiation", sa.Float(), nullable=True),
        sa.Column("difradiation", sa.Float(), nullable=True),
        sa.Column("ghiradiation", sa.Float(), nullable=True),
        sa.Column("gtiradiation", sa.Float(), nullable=True),
        sa.Column("cape", sa.Float(), nullable=True),
        sa.Column("cin", sa.Float(), nullable=True),
        sa.Column("deltat", sa.Float(), nullable=True),
        sa.Column("degreedays", sa.Float(), nullable=True),
        sa.Column("accdegreedays", sa.Float(), nullable=True),
        sa.Column("conditions", sa.Text(), nullable=True),
        sa.Column("icon", sa.Text(), nullable=True),
        # Metadata
        sa.Column("source_system", sa.Text(), server_default="vc_timeline", nullable=True),
        sa.Column("raw_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "vc_location_id", "target_date", "forecast_basis_date", "data_type",
            name="uq_vc_daily_row",
        ),
        sa.CheckConstraint(
            "data_type IN ('forecast', 'historical_forecast')",
            name="ck_vc_daily_data_type",
        ),
        schema="wx",
    )

    op.create_index(
        "ix_vc_forecast_daily_loc_target",
        "vc_forecast_daily",
        ["vc_location_id", "target_date", "forecast_basis_date"],
        schema="wx",
    )

    # ==========================================================================
    # wx.vc_forecast_hourly - Hourly forecast snapshots
    # ==========================================================================
    op.create_table(
        "vc_forecast_hourly",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("vc_location_id", sa.Integer(), sa.ForeignKey("wx.vc_location.id"), nullable=False),
        # Classification
        sa.Column("data_type", sa.Text(), nullable=False),  # 'forecast' | 'historical_forecast'
        sa.Column("forecast_basis_date", sa.Date(), nullable=False),
        sa.Column("forecast_basis_datetime_utc", sa.DateTime(timezone=True), nullable=True),
        # Target time
        sa.Column("target_datetime_epoch_utc", sa.BigInteger(), nullable=False),
        sa.Column("target_datetime_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_datetime_local", sa.DateTime(), nullable=False),
        sa.Column("timezone", sa.Text(), nullable=False),
        sa.Column("tzoffset_minutes", sa.SmallInteger(), nullable=False),
        sa.Column("lead_hours", sa.Integer(), nullable=False),
        # Weather fields
        sa.Column("temp_f", sa.Float(), nullable=True),
        sa.Column("feelslike_f", sa.Float(), nullable=True),
        sa.Column("dew_f", sa.Float(), nullable=True),
        sa.Column("humidity", sa.Float(), nullable=True),
        sa.Column("precip_in", sa.Float(), nullable=True),
        sa.Column("precipprob", sa.Float(), nullable=True),
        sa.Column("preciptype", sa.Text(), nullable=True),
        sa.Column("snow_in", sa.Float(), nullable=True),
        sa.Column("windspeed_mph", sa.Float(), nullable=True),
        sa.Column("windgust_mph", sa.Float(), nullable=True),
        sa.Column("winddir", sa.Float(), nullable=True),
        sa.Column("windspeed50_mph", sa.Float(), nullable=True),
        sa.Column("winddir50", sa.Float(), nullable=True),
        sa.Column("windspeed80_mph", sa.Float(), nullable=True),
        sa.Column("winddir80", sa.Float(), nullable=True),
        sa.Column("windspeed100_mph", sa.Float(), nullable=True),
        sa.Column("winddir100", sa.Float(), nullable=True),
        sa.Column("cloudcover", sa.Float(), nullable=True),
        sa.Column("visibility_miles", sa.Float(), nullable=True),
        sa.Column("pressure_mb", sa.Float(), nullable=True),
        sa.Column("uvindex", sa.Float(), nullable=True),
        sa.Column("solarradiation", sa.Float(), nullable=True),
        sa.Column("solarenergy", sa.Float(), nullable=True),
        sa.Column("dniradiation", sa.Float(), nullable=True),
        sa.Column("difradiation", sa.Float(), nullable=True),
        sa.Column("ghiradiation", sa.Float(), nullable=True),
        sa.Column("gtiradiation", sa.Float(), nullable=True),
        sa.Column("sunelevation", sa.Float(), nullable=True),
        sa.Column("sunazimuth", sa.Float(), nullable=True),
        sa.Column("cape", sa.Float(), nullable=True),
        sa.Column("cin", sa.Float(), nullable=True),
        sa.Column("conditions", sa.Text(), nullable=True),
        sa.Column("icon", sa.Text(), nullable=True),
        # Metadata
        sa.Column("source_system", sa.Text(), server_default="vc_timeline", nullable=True),
        sa.Column("raw_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "vc_location_id", "target_datetime_utc", "forecast_basis_date", "data_type",
            name="uq_vc_hourly_row",
        ),
        sa.CheckConstraint(
            "data_type IN ('forecast', 'historical_forecast')",
            name="ck_vc_hourly_data_type",
        ),
        schema="wx",
    )

    op.create_index(
        "ix_vc_forecast_hourly_loc_target",
        "vc_forecast_hourly",
        ["vc_location_id", "target_datetime_utc", "forecast_basis_date"],
        schema="wx",
    )

    # ==========================================================================
    # Seed wx.vc_location with 12 rows (6 cities × 2 types)
    # ==========================================================================
    op.execute("""
        INSERT INTO wx.vc_location (city_code, kalshi_code, location_type, vc_location_query, station_id, iana_timezone)
        VALUES
            -- Chicago
            ('CHI', 'CHI', 'station', 'stn:KMDW', 'KMDW', 'America/Chicago'),
            ('CHI', 'CHI', 'city', 'Chicago,IL', NULL, 'America/Chicago'),
            -- Denver
            ('DEN', 'DEN', 'station', 'stn:KDEN', 'KDEN', 'America/Denver'),
            ('DEN', 'DEN', 'city', 'Denver,CO', NULL, 'America/Denver'),
            -- Austin
            ('AUS', 'AUS', 'station', 'stn:KAUS', 'KAUS', 'America/Chicago'),
            ('AUS', 'AUS', 'city', 'Austin,TX', NULL, 'America/Chicago'),
            -- Los Angeles
            ('LAX', 'LAX', 'station', 'stn:KLAX', 'KLAX', 'America/Los_Angeles'),
            ('LAX', 'LAX', 'city', 'Los Angeles,CA', NULL, 'America/Los_Angeles'),
            -- Miami
            ('MIA', 'MIA', 'station', 'stn:KMIA', 'KMIA', 'America/New_York'),
            ('MIA', 'MIA', 'city', 'Miami,FL', NULL, 'America/New_York'),
            -- Philadelphia (note: kalshi_code is 'PHIL', not 'PHL')
            ('PHL', 'PHIL', 'station', 'stn:KPHL', 'KPHL', 'America/New_York'),
            ('PHL', 'PHIL', 'city', 'Philadelphia,PA', NULL, 'America/New_York')
    """)


def downgrade() -> None:
    # Drop tables in reverse order (respecting foreign keys)
    # Note: indexes are dropped automatically when table is dropped
    op.drop_table("vc_forecast_hourly", schema="wx")
    op.drop_table("vc_forecast_daily", schema="wx")
    op.drop_table("vc_minute_weather", schema="wx")
    op.drop_table("vc_location", schema="wx")
