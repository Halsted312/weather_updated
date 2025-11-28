"""
SQLAlchemy ORM models for all three schemas: wx, kalshi, sim.
"""

from datetime import date, datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# Schema: wx (Weather/Labels)
# =============================================================================


class WxSettlement(Base):
    """Official NWS daily max temperature (ground truth for Kalshi settlement).

    Multi-source TMAX tracking:
    - tmax_cli_f: NWS CLI (via IEM parsed JSON)
    - tmax_cf6_f: NWS CF6 (preliminary monthly)
    - tmax_iem_f: IEM CLI JSON API (historical)
    - tmax_ncei_f: NCEI daily-summaries (canonical fallback)
    - tmax_ads_f: Legacy ADS data

    Kalshi bucket tracking:
    - settled_ticker: The winning market ticker
    - settled_bucket_type: 'between' | 'less' | 'greater'
    - settled_floor_strike, settled_cap_strike: Temperature bounds
    - settled_bucket_label: Human-readable label
    """

    __tablename__ = "settlement"
    __table_args__ = {"schema": "wx"}

    city: Mapped[str] = mapped_column(Text, primary_key=True)
    date_local: Mapped[date] = mapped_column(Date, primary_key=True)

    # Source-specific temperatures (integer °F)
    tmax_cli_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_cf6_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_iem_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_ncei_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    tmax_ads_f: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # Chosen settlement value
    tmax_final: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    source_final: Mapped[str] = mapped_column(Text, nullable=False)  # 'cli' | 'cf6' | 'iem' | 'ncei' | 'ads'

    # Legacy raw payload (kept for backwards compatibility)
    raw_payload: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Per-source raw payloads
    raw_payload_cli: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    raw_payload_cf6: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    raw_payload_iem: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    raw_payload_ncei: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Kalshi settlement bucket tracking
    settled_ticker: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    settled_bucket_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'between' | 'less' | 'greater'
    settled_floor_strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    settled_cap_strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    settled_bucket_label: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )


class WxMinuteObs(Base):
    """[LEGACY] Visual Crossing 5-minute weather observations.

    DEPRECATION NOTICE: This model is superseded by VcMinuteWeather which provides:
    - ALL 47+ weather fields (vs ~15 in this legacy model)
    - Both station-locked and city-aggregate feeds
    - Proper datetime/timezone handling
    - Unified schema for observations + forecasts + historical forecasts
    - Foreign key relationship to VcLocation

    Use VcMinuteWeather for all new development. This model will be removed
    in a future release once migration is complete.
    """

    __tablename__ = "minute_obs"
    __table_args__ = {"schema": "wx"}

    loc_id: Mapped[str] = mapped_column(String(10), primary_key=True)  # KMDW, KDEN, etc.
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)

    # Core weather data
    temp_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslike_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dew_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Precipitation
    precip_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_prob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snow_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snow_depth_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Wind
    windspeed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windgust_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Atmospheric
    pressure_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    visibility_mi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cloud_cover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Solar
    solar_radiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solar_energy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uv_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Conditions
    conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    source: Mapped[str] = mapped_column(String(20), default="visualcrossing")
    stations: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    ffilled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)


class WxForecastSnapshot(Base):
    """Visual Crossing historical forecast snapshots for Option-1 backtest."""

    __tablename__ = "forecast_snapshot"
    __table_args__ = {"schema": "wx"}

    city: Mapped[str] = mapped_column(Text, primary_key=True)
    target_date: Mapped[date] = mapped_column(Date, primary_key=True)
    basis_date: Mapped[date] = mapped_column(Date, primary_key=True)

    lead_days: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(Text, default="visualcrossing")

    # Forecast values
    tempmax_fcst_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tempmin_fcst_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_fcst_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_prob_fcst: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_fcst: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed_fcst_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    conditions_fcst: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )


class WxForecastSnapshotHourly(Base):
    """Visual Crossing hourly forecast snapshots for ML/trend analysis.

    Stores 72-hour (3-day) forecast curves at local midnight basis times.
    Used for temperature trend analysis and comparison with Kalshi prices.
    """

    __tablename__ = "forecast_snapshot_hourly"
    __table_args__ = {"schema": "wx"}

    city: Mapped[str] = mapped_column(Text, primary_key=True)
    target_hour_local: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    basis_date: Mapped[date] = mapped_column(Date, primary_key=True)

    target_hour_epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    lead_hours: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-71
    provider: Mapped[str] = mapped_column(Text, default="visualcrossing")
    tz_name: Mapped[str] = mapped_column(Text, nullable=False)  # IANA timezone

    # Forecast values (hourly, not daily max/min)
    temp_fcst_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslike_fcst_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity_fcst: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_fcst_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_prob_fcst: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed_fcst_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    conditions_fcst: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )


# =============================================================================
# Schema: wx (Visual Crossing - New Greenfield Tables)
# =============================================================================


class VcLocation(Base):
    """Visual Crossing location dimension table.

    Tracks both station-locked (e.g., stn:KMDW) and city-aggregate (e.g., Chicago,IL)
    locations for each Kalshi weather market city.

    - location_type='station': Single airport station, no interpolation
    - location_type='city': City-aggregate with VC's multi-station interpolation
    """

    __tablename__ = "vc_location"
    __table_args__ = (
        {"schema": "wx"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # City identifiers
    city_code: Mapped[str] = mapped_column(Text, nullable=False)  # 'CHI', 'DEN', 'AUS', 'LAX', 'MIA', 'PHL'
    kalshi_code: Mapped[str] = mapped_column(Text, nullable=False)  # 'CHI', 'DEN', 'AUS', 'LAX', 'MIA', 'PHIL'
    location_type: Mapped[str] = mapped_column(Text, nullable=False)  # 'station' | 'city'

    # Visual Crossing query parameters
    vc_location_query: Mapped[str] = mapped_column(Text, nullable=False)  # 'stn:KMDW' or 'Chicago,IL'
    station_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'KMDW' (nullable for city type)

    # Timezone
    iana_timezone: Mapped[str] = mapped_column(Text, nullable=False)  # 'America/Chicago'

    # Location metadata (populated from VC add: fields on first fetch)
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    elevation_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )

    # Relationships
    minute_weather: Mapped[list["VcMinuteWeather"]] = relationship(back_populates="location")
    forecast_daily: Mapped[list["VcForecastDaily"]] = relationship(back_populates="location")
    forecast_hourly: Mapped[list["VcForecastHourly"]] = relationship(back_populates="location")


class VcMinuteWeather(Base):
    """Visual Crossing minute-level weather data (fact table).

    Stores both observations and forecasts at 5-min (obs) or 15-min (forecast) resolution.
    Data type classification:
    - 'actual_obs': Historical observations (forecast_basis_date=NULL)
    - 'current_snapshot': Current conditions snapshot
    - 'forecast': Future forecast data
    - 'historical_forecast': Past forecast (for backtesting)

    Note: Degreeday fields (degreedays, accdegreedays) are daily constructs -
    at minute resolution they'll be constant per day.
    """

    __tablename__ = "vc_minute_weather"
    __table_args__ = (
        {"schema": "wx"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vc_location_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("wx.vc_location.id"), nullable=False
    )

    # Classification
    data_type: Mapped[str] = mapped_column(Text, nullable=False)  # 'actual_obs' | 'current_snapshot' | 'forecast' | 'historical_forecast'
    forecast_basis_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)  # NULL for actual_obs
    forecast_basis_datetime_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    lead_hours: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # (target - basis) / 3600

    # Time fields (from VC datetime/datetimeEpoch/timezone/tzoffset)
    datetime_epoch_utc: Mapped[int] = mapped_column(BigInteger, nullable=False)
    datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    datetime_local: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # WITHOUT timezone
    timezone: Mapped[str] = mapped_column(Text, nullable=False)  # 'America/Chicago'
    tzoffset_minutes: Mapped[int] = mapped_column(SmallInteger, nullable=False)  # e.g., -360 for CST

    # Core weather
    temp_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tempmax_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tempmin_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslike_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslikemax_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslikemin_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dew_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Precipitation
    precip_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipprob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    preciptype: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    precipcover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snow_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snowdepth_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipremote: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Wind (10m standard)
    windspeed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windgust_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeedmean_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeedmin_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeedmax_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Extended wind (50/80/100m)
    windspeed50_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed80_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir80: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed100_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir100: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Atmosphere
    cloudcover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    visibility_miles: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pressure_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Solar/Radiation
    uvindex: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solarradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solarenergy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dniradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    difradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ghiradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gtiradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sunelevation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sunazimuth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Instability/Energy
    cape: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    deltat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    degreedays: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accdegreedays: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Text/Flags
    conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolved_address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Forward-fill tracking (for when VC returns no temp reading)
    is_forward_filled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("FALSE"))

    # Metadata
    source_system: Mapped[str] = mapped_column(Text, default="vc_timeline")
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )

    # Relationships
    location: Mapped["VcLocation"] = relationship(back_populates="minute_weather")


class VcForecastDaily(Base):
    """Visual Crossing daily forecast snapshots.

    Stores daily-level forecasts with full extended weather fields.
    data_type: 'forecast' (current) | 'historical_forecast' (for backtesting)
    """

    __tablename__ = "vc_forecast_daily"
    __table_args__ = (
        {"schema": "wx"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vc_location_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("wx.vc_location.id"), nullable=False
    )

    # Classification
    data_type: Mapped[str] = mapped_column(Text, nullable=False)  # 'forecast' | 'historical_forecast'
    forecast_basis_date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_basis_datetime_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Target
    target_date: Mapped[date] = mapped_column(Date, nullable=False)
    lead_days: Mapped[int] = mapped_column(Integer, nullable=False)  # target_date - forecast_basis_date

    # Daily weather fields
    tempmax_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tempmin_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temp_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslikemax_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslikemin_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslike_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dew_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipprob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    preciptype: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    precipcover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snow_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    snowdepth_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windgust_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeedmean_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeedmin_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeedmax_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed50_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed80_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir80: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed100_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir100: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cloudcover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    visibility_miles: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pressure_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uvindex: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solarradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solarenergy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dniradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    difradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ghiradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gtiradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cape: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    deltat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    degreedays: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accdegreedays: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    source_system: Mapped[str] = mapped_column(Text, default="vc_timeline")
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )

    # Relationships
    location: Mapped["VcLocation"] = relationship(back_populates="forecast_daily")


class VcForecastHourly(Base):
    """Visual Crossing hourly forecast snapshots.

    Stores hourly-level forecasts with full extended weather fields and time metadata.
    data_type: 'forecast' (current) | 'historical_forecast' (for backtesting)
    """

    __tablename__ = "vc_forecast_hourly"
    __table_args__ = (
        {"schema": "wx"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vc_location_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("wx.vc_location.id"), nullable=False
    )

    # Classification
    data_type: Mapped[str] = mapped_column(Text, nullable=False)  # 'forecast' | 'historical_forecast'
    forecast_basis_date: Mapped[date] = mapped_column(Date, nullable=False)
    forecast_basis_datetime_utc: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Target time
    target_datetime_epoch_utc: Mapped[int] = mapped_column(BigInteger, nullable=False)
    target_datetime_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    target_datetime_local: Mapped[datetime] = mapped_column(DateTime, nullable=False)  # WITHOUT timezone
    timezone: Mapped[str] = mapped_column(Text, nullable=False)
    tzoffset_minutes: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    lead_hours: Mapped[int] = mapped_column(Integer, nullable=False)

    # Hourly weather fields
    temp_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    feelslike_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dew_f: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    humidity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precip_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precipprob: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    preciptype: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    snow_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windgust_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed50_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed80_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir80: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    windspeed100_mph: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    winddir100: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cloudcover: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    visibility_miles: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pressure_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uvindex: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solarradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    solarenergy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dniradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    difradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ghiradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    gtiradiation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sunelevation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sunazimuth: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cape: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    conditions: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    icon: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    source_system: Mapped[str] = mapped_column(Text, default="vc_timeline")
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )

    # Relationships
    location: Mapped["VcLocation"] = relationship(back_populates="forecast_hourly")


# =============================================================================
# Schema: kalshi (Market Data)
# =============================================================================


class KalshiMarket(Base):
    """Kalshi market/contract metadata."""

    __tablename__ = "markets"
    __table_args__ = {"schema": "kalshi"}

    ticker: Mapped[str] = mapped_column(Text, primary_key=True)
    city: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    event_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    exchange_market_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    strike_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'between' | 'less' | 'greater'
    floor_strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cap_strike: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    listed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    close_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    expiration_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'open' | 'closed' | 'settled'
    result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'yes' | 'no'
    settlement_value: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )

    # Relationships
    candles: Mapped[list["KalshiCandle1m"]] = relationship(back_populates="market")


class KalshiCandle1m(Base):
    """1-minute OHLCV candlesticks for Kalshi markets.

    Supports dual storage with `source` column:
    - 'api_event': From Kalshi Event Candlesticks API (primary)
    - 'trades': Aggregated from individual trades (fallback/audit)
    """

    __tablename__ = "candles_1m"
    __table_args__ = {"schema": "kalshi"}

    ticker: Mapped[str] = mapped_column(
        Text, ForeignKey("kalshi.markets.ticker"), primary_key=True
    )
    bucket_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    # Source of candle data: 'api_event' | 'trades'
    source: Mapped[str] = mapped_column(
        String(20), primary_key=True, default="trades"
    )

    # OHLC prices (0-100 cents)
    open_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    high_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    low_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    close_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # Bid/Ask
    yes_bid_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    yes_ask_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)

    # Volume
    volume: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    open_interest: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    market: Mapped["KalshiMarket"] = relationship(back_populates="candles")


class KalshiWsRaw(Base):
    """Raw WebSocket message log - store EVERYTHING."""

    __tablename__ = "ws_raw"
    __table_args__ = {"schema": "kalshi"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), primary_key=True
    )

    source: Mapped[str] = mapped_column(Text, default="kalshi", nullable=False)
    stream: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'market-data', 'trades', etc.
    topic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # ticker or channel
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)


class KalshiOrder(Base):
    """My orders on Kalshi."""

    __tablename__ = "orders"
    __table_args__ = {"schema": "kalshi"}

    order_id: Mapped[str] = mapped_column(Text, primary_key=True)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)  # 'buy' | 'sell'
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price_c: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    order_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'limit' | 'market'
    status: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # 'pending' | 'filled' | 'cancelled'

    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )

    # Relationships
    fills: Mapped[list["KalshiFill"]] = relationship(back_populates="order")


class KalshiFill(Base):
    """My fills on Kalshi."""

    __tablename__ = "fills"
    __table_args__ = {"schema": "kalshi"}

    fill_id: Mapped[str] = mapped_column(Text, primary_key=True)
    order_id: Mapped[str] = mapped_column(
        Text, ForeignKey("kalshi.orders.order_id"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    side: Mapped[str] = mapped_column(Text, nullable=False)
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price_c: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fee_c: Mapped[Optional[int]] = mapped_column(SmallInteger, nullable=True)
    ts_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    order: Mapped["KalshiOrder"] = relationship(back_populates="fills")


# =============================================================================
# Schema: sim (Backtest/Simulation)
# =============================================================================


class SimRun(Base):
    """Backtest run metadata."""

    __tablename__ = "run"
    __table_args__ = {"schema": "sim"}

    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    strategy_name: Mapped[str] = mapped_column(Text, nullable=False)
    params_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    train_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    train_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    test_start: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    test_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )

    # Relationships
    trades: Mapped[list["SimTrade"]] = relationship(back_populates="run")


class SimTrade(Base):
    """Simulated trades from backtests."""

    __tablename__ = "trade"
    __table_args__ = {"schema": "sim"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_ts_utc: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    run_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("sim.run.run_id"), nullable=False
    )
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    event_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)

    side: Mapped[str] = mapped_column(Text, nullable=False)  # 'buy' | 'sell'
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    price_c: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    fee_c: Mapped[int] = mapped_column(SmallInteger, default=0)
    pnl_c: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    position_after: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    run: Mapped["SimRun"] = relationship(back_populates="trades")


# =============================================================================
# Schema: meta (Operations/Infrastructure)
# =============================================================================


class IngestCheckpoint(Base):
    """Track ingestion progress for resumable backfills.

    Each (pipeline_name, city) combo tracks where we left off,
    enabling resume on crash/restart.
    """

    __tablename__ = "ingestion_checkpoint"
    __table_args__ = {"schema": "meta"}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # What are we tracking?
    pipeline_name: Mapped[str] = mapped_column(Text, nullable=False)
    city: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Where did we get to?
    last_processed_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    last_processed_cursor: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_processed_ticker: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status tracking
    status: Mapped[str] = mapped_column(Text, default="running")  # running, completed, failed
    total_processed: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), onupdate=text("NOW()")
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
