#!/usr/bin/env python3
"""
Visual Crossing live data polling daemon.

Continuously polls Visual Crossing for:
- Observations: Every 5 minutes (station-locked + city-aggregate)
- Forecasts: Every 15 minutes (station-locked + city-aggregate)

Auto-backfills gaps up to 3 hours on startup.

Usage:
    python scripts/poll_vc_live_daemon.py
    python scripts/poll_vc_live_daemon.py --once       # Single poll cycle, then exit
    python scripts/poll_vc_live_daemon.py --obs-only   # Only observations
    python scripts/poll_vc_live_daemon.py --forecast-only  # Only forecasts
    python scripts/poll_vc_live_daemon.py --city-code CHI  # Single city

Deployment:
    systemctl start vc-live-daemon
"""

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select, func, update, and_
from sqlalchemy.dialects.postgresql import insert

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import get_settings
from src.db import get_db_session, VcLocation, VcMinuteWeather, VcForecastHourly, VcForecastDaily
from src.weather.visual_crossing import VisualCrossingClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Shutdown flag for graceful termination
shutdown_requested = False

# Polling intervals
OBS_INTERVAL_SECONDS = 300      # 5 minutes
FORECAST_INTERVAL_SECONDS = 900  # 15 minutes
API_RATE_LIMIT_SECONDS = 0.3    # Time between API calls
GAP_LOOKBACK_HOURS = 3          # Max gap to backfill


@dataclass
class LocationInfo:
    """Lightweight location data for daemon operations."""
    id: int
    city_code: str
    location_type: str
    station_id: Optional[str]
    vc_location_query: str
    iana_timezone: str

    @classmethod
    def from_db(cls, loc: VcLocation) -> "LocationInfo":
        """Create from SQLAlchemy model while session is active."""
        return cls(
            id=loc.id,
            city_code=loc.city_code,
            location_type=loc.location_type,
            station_id=loc.station_id,
            vc_location_query=loc.vc_location_query,
            iana_timezone=loc.iana_timezone,
        )


def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting graceful shutdown...")
    shutdown_requested = True


def _format_list(value: Any) -> Optional[str]:
    """Format list as comma-separated string."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else None
    elif isinstance(value, str):
        return value
    return None


def parse_vc_minute_to_record(
    minute_data: Dict[str, Any],
    vc_location_id: int,
    iana_timezone: str,
    data_type: str = "actual_obs",
    forecast_basis_date: Optional[date] = None,
    forecast_basis_datetime_utc: Optional[datetime] = None,
    lead_hours: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse a single VC minute record to database format.

    Args:
        minute_data: Raw minute data from VC API
        vc_location_id: Foreign key to vc_location
        iana_timezone: IANA timezone name from location
        data_type: 'actual_obs' or 'forecast'
        forecast_basis_date: For forecasts, when the forecast was made
        forecast_basis_datetime_utc: UTC datetime when forecast was made
        lead_hours: Hours ahead this forecast is for

    Returns:
        Dict ready for insertion into vc_minute_weather
    """
    epoch = minute_data.get("datetimeEpoch")
    if not epoch:
        return None

    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)

    # Convert to local time using IANA timezone
    tz_local = ZoneInfo(iana_timezone)
    dt_local_aware = dt_utc.astimezone(tz_local)
    dt_local_naive = dt_local_aware.replace(tzinfo=None)

    # Calculate actual offset in minutes
    offset_seconds = dt_local_aware.utcoffset().total_seconds()
    tzoffset_minutes = int(offset_seconds / 60)

    record = {
        "vc_location_id": vc_location_id,
        "data_type": data_type,
        "forecast_basis_date": forecast_basis_date,
        "forecast_basis_datetime_utc": forecast_basis_datetime_utc,
        "lead_hours": lead_hours,
        # Time fields
        "datetime_epoch_utc": epoch,
        "datetime_utc": dt_utc,
        "datetime_local": dt_local_naive,
        "timezone": minute_data.get("timezone") or iana_timezone,
        "tzoffset_minutes": tzoffset_minutes,
        # Weather fields
        "temp_f": minute_data.get("temp"),
        "tempmax_f": minute_data.get("tempmax"),
        "tempmin_f": minute_data.get("tempmin"),
        "feelslike_f": minute_data.get("feelslike"),
        "feelslikemax_f": minute_data.get("feelslikemax"),
        "feelslikemin_f": minute_data.get("feelslikemin"),
        "dew_f": minute_data.get("dew"),
        "humidity": minute_data.get("humidity"),
        "precip_in": minute_data.get("precip"),
        "precipprob": minute_data.get("precipprob"),
        "preciptype": _format_list(minute_data.get("preciptype")),
        "precipcover": minute_data.get("precipcover"),
        "snow_in": minute_data.get("snow"),
        "snowdepth_in": minute_data.get("snowdepth"),
        "precipremote": minute_data.get("precipremote"),
        "windspeed_mph": minute_data.get("windspeed"),
        "windgust_mph": minute_data.get("windgust"),
        "winddir": minute_data.get("winddir"),
        "windspeedmean_mph": minute_data.get("windspeedmean"),
        "windspeedmin_mph": minute_data.get("windspeedmin"),
        "windspeedmax_mph": minute_data.get("windspeedmax"),
        "windspeed50_mph": minute_data.get("windspeed50"),
        "winddir50": minute_data.get("winddir50"),
        "windspeed80_mph": minute_data.get("windspeed80"),
        "winddir80": minute_data.get("winddir80"),
        "windspeed100_mph": minute_data.get("windspeed100"),
        "winddir100": minute_data.get("winddir100"),
        "cloudcover": minute_data.get("cloudcover"),
        "visibility_miles": minute_data.get("visibility"),
        "pressure_mb": minute_data.get("pressure"),
        "uvindex": minute_data.get("uvindex"),
        "solarradiation": minute_data.get("solarradiation"),
        "solarenergy": minute_data.get("solarenergy"),
        "dniradiation": minute_data.get("dniradiation"),
        "difradiation": minute_data.get("difradiation"),
        "ghiradiation": minute_data.get("ghiradiation"),
        "gtiradiation": minute_data.get("gtiradiation"),
        "sunelevation": minute_data.get("sunelevation"),
        "sunazimuth": minute_data.get("sunazimuth"),
        "cape": minute_data.get("cape"),
        "cin": minute_data.get("cin"),
        "deltat": minute_data.get("deltat"),
        "degreedays": minute_data.get("degreedays"),
        "accdegreedays": minute_data.get("accdegreedays"),
        "conditions": minute_data.get("conditions"),
        "icon": minute_data.get("icon"),
        "stations": _format_list(minute_data.get("stations")),
        "resolved_address": minute_data.get("resolvedAddress"),
        "source_system": "vc_live_daemon",
        "raw_json": minute_data,
        "is_forward_filled": False,  # Real data from VC
    }

    return record


def flatten_vc_response_to_minutes(
    data: Dict[str, Any],
    vc_location_id: int,
    iana_timezone: str,
    data_type: str = "actual_obs",
    forecast_basis_datetime_utc: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """
    Flatten VC API response to list of minute records.

    Works for both observations (5-min in hours->minutes)
    and forecasts (15-min in hours->minutes).
    """
    records = []
    forecast_basis_date = forecast_basis_datetime_utc.date() if forecast_basis_datetime_utc else None

    for day in data.get("days", []):
        for hour in day.get("hours", []):
            # Check for minutes array (5-min or 15-min intervals)
            minutes_data = hour.get("minutes", [])
            if minutes_data:
                for minute in minutes_data:
                    # Calculate lead hours if this is a forecast
                    lead_hours = None
                    if forecast_basis_datetime_utc and minute.get("datetimeEpoch"):
                        forecast_dt = datetime.fromtimestamp(minute["datetimeEpoch"], tz=timezone.utc)
                        lead_hours = int((forecast_dt - forecast_basis_datetime_utc).total_seconds() / 3600)

                    record = parse_vc_minute_to_record(
                        minute,
                        vc_location_id,
                        iana_timezone,
                        data_type=data_type,
                        forecast_basis_date=forecast_basis_date,
                        forecast_basis_datetime_utc=forecast_basis_datetime_utc,
                        lead_hours=lead_hours,
                    )
                    if record:
                        records.append(record)
            else:
                # No minutes, use hour-level data (store as single record)
                lead_hours = None
                if forecast_basis_datetime_utc and hour.get("datetimeEpoch"):
                    forecast_dt = datetime.fromtimestamp(hour["datetimeEpoch"], tz=timezone.utc)
                    lead_hours = int((forecast_dt - forecast_basis_datetime_utc).total_seconds() / 3600)

                record = parse_vc_minute_to_record(
                    hour,
                    vc_location_id,
                    iana_timezone,
                    data_type=data_type,
                    forecast_basis_date=forecast_basis_date,
                    forecast_basis_datetime_utc=forecast_basis_datetime_utc,
                    lead_hours=lead_hours,
                )
                if record:
                    records.append(record)

    return records


def parse_vc_hourly_forecast(
    hour_data: Dict[str, Any],
    vc_location_id: int,
    iana_timezone: str,
    forecast_basis_datetime_utc: datetime,
) -> Optional[Dict[str, Any]]:
    """Parse hourly forecast data for vc_forecast_hourly table."""
    epoch = hour_data.get("datetimeEpoch")
    if not epoch:
        return None

    dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
    tz_local = ZoneInfo(iana_timezone)
    dt_local_aware = dt_utc.astimezone(tz_local)
    dt_local_naive = dt_local_aware.replace(tzinfo=None)

    # Calculate offset in minutes
    offset_seconds = dt_local_aware.utcoffset().total_seconds()
    tzoffset_minutes = int(offset_seconds / 60)

    lead_hours = int((dt_utc - forecast_basis_datetime_utc).total_seconds() / 3600)

    return {
        "vc_location_id": vc_location_id,
        "data_type": "forecast",
        "forecast_basis_date": forecast_basis_datetime_utc.date(),
        "forecast_basis_datetime_utc": forecast_basis_datetime_utc,
        # Target time fields
        "target_datetime_epoch_utc": epoch,
        "target_datetime_utc": dt_utc,
        "target_datetime_local": dt_local_naive,
        "timezone": iana_timezone,
        "tzoffset_minutes": tzoffset_minutes,
        "lead_hours": lead_hours,
        # Weather fields
        "temp_f": hour_data.get("temp"),
        "feelslike_f": hour_data.get("feelslike"),
        "dew_f": hour_data.get("dew"),
        "humidity": hour_data.get("humidity"),
        "precip_in": hour_data.get("precip"),
        "precipprob": hour_data.get("precipprob"),
        "preciptype": _format_list(hour_data.get("preciptype")),
        "snow_in": hour_data.get("snow"),
        "windspeed_mph": hour_data.get("windspeed"),
        "windgust_mph": hour_data.get("windgust"),
        "winddir": hour_data.get("winddir"),
        "windspeed50_mph": hour_data.get("windspeed50"),
        "winddir50": hour_data.get("winddir50"),
        "windspeed80_mph": hour_data.get("windspeed80"),
        "winddir80": hour_data.get("winddir80"),
        "windspeed100_mph": hour_data.get("windspeed100"),
        "winddir100": hour_data.get("winddir100"),
        "cloudcover": hour_data.get("cloudcover"),
        "visibility_miles": hour_data.get("visibility"),
        "pressure_mb": hour_data.get("pressure"),
        "uvindex": hour_data.get("uvindex"),
        "solarradiation": hour_data.get("solarradiation"),
        "solarenergy": hour_data.get("solarenergy"),
        "dniradiation": hour_data.get("dniradiation"),
        "difradiation": hour_data.get("difradiation"),
        "ghiradiation": hour_data.get("ghiradiation"),
        "gtiradiation": hour_data.get("gtiradiation"),
        "sunelevation": hour_data.get("sunelevation"),
        "sunazimuth": hour_data.get("sunazimuth"),
        "cape": hour_data.get("cape"),
        "cin": hour_data.get("cin"),
        "conditions": hour_data.get("conditions"),
        "icon": hour_data.get("icon"),
        "source_system": "vc_live_daemon",
        "raw_json": hour_data,
    }


def parse_vc_daily_forecast(
    day_data: Dict[str, Any],
    vc_location_id: int,
    iana_timezone: str,
    forecast_basis_datetime_utc: datetime,
) -> Optional[Dict[str, Any]]:
    """Parse daily forecast data for vc_forecast_daily table."""
    date_str = day_data.get("datetime")
    if not date_str:
        return None

    target_date = date.fromisoformat(date_str)
    lead_days = (target_date - forecast_basis_datetime_utc.date()).days

    return {
        "vc_location_id": vc_location_id,
        "data_type": "forecast",
        "forecast_basis_date": forecast_basis_datetime_utc.date(),
        "forecast_basis_datetime_utc": forecast_basis_datetime_utc,
        "target_date": target_date,
        "lead_days": lead_days,
        # Daily weather fields
        "tempmax_f": day_data.get("tempmax"),
        "tempmin_f": day_data.get("tempmin"),
        "temp_f": day_data.get("temp"),
        "feelslikemax_f": day_data.get("feelslikemax"),
        "feelslikemin_f": day_data.get("feelslikemin"),
        "feelslike_f": day_data.get("feelslike"),
        "dew_f": day_data.get("dew"),
        "humidity": day_data.get("humidity"),
        "precip_in": day_data.get("precip"),
        "precipprob": day_data.get("precipprob"),
        "precipcover": day_data.get("precipcover"),
        "preciptype": _format_list(day_data.get("preciptype")),
        "snow_in": day_data.get("snow"),
        "snowdepth_in": day_data.get("snowdepth"),
        "windspeed_mph": day_data.get("windspeed"),
        "windgust_mph": day_data.get("windgust"),
        "winddir": day_data.get("winddir"),
        "windspeedmean_mph": day_data.get("windspeedmean"),
        "windspeedmin_mph": day_data.get("windspeedmin"),
        "windspeedmax_mph": day_data.get("windspeedmax"),
        "windspeed50_mph": day_data.get("windspeed50"),
        "winddir50": day_data.get("winddir50"),
        "windspeed80_mph": day_data.get("windspeed80"),
        "winddir80": day_data.get("winddir80"),
        "windspeed100_mph": day_data.get("windspeed100"),
        "winddir100": day_data.get("winddir100"),
        "cloudcover": day_data.get("cloudcover"),
        "visibility_miles": day_data.get("visibility"),
        "pressure_mb": day_data.get("pressure"),
        "uvindex": day_data.get("uvindex"),
        "solarradiation": day_data.get("solarradiation"),
        "solarenergy": day_data.get("solarenergy"),
        "dniradiation": day_data.get("dniradiation"),
        "difradiation": day_data.get("difradiation"),
        "ghiradiation": day_data.get("ghiradiation"),
        "gtiradiation": day_data.get("gtiradiation"),
        "cape": day_data.get("cape"),
        "cin": day_data.get("cin"),
        "deltat": day_data.get("deltat"),
        "degreedays": day_data.get("degreedays"),
        "accdegreedays": day_data.get("accdegreedays"),
        "conditions": day_data.get("conditions"),
        "icon": day_data.get("icon"),
        "source_system": "vc_live_daemon",
        "raw_json": day_data,
    }


def upsert_vc_minute_weather(session, records: List[Dict[str, Any]]) -> int:
    """Upsert minute weather records (observations).

    Only overwrites existing records if the new data has a real (non-forward-filled) temp,
    OR if the existing record was forward-filled.
    """
    if not records:
        return 0

    stmt = insert(VcMinuteWeather).values(records)

    # Build update columns - exclude key/immutable columns
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcMinuteWeather.__table__.columns
        if col.name not in ("id", "vc_location_id", "data_type", "forecast_basis_date", "datetime_utc", "created_at")
    }

    # Use partial index for obs (WHERE forecast_basis_date IS NULL)
    stmt = stmt.on_conflict_do_update(
        index_elements=["vc_location_id", "data_type", "datetime_utc"],
        index_where=VcMinuteWeather.forecast_basis_date.is_(None),
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


def get_last_known_temp(session, location_id: int) -> Optional[float]:
    """Get the most recent non-NULL, non-forward-filled temperature for a location."""
    result = session.execute(
        select(VcMinuteWeather.temp_f)
        .where(VcMinuteWeather.vc_location_id == location_id)
        .where(VcMinuteWeather.data_type == "actual_obs")
        .where(VcMinuteWeather.temp_f.isnot(None))
        .where(VcMinuteWeather.is_forward_filled == False)
        .order_by(VcMinuteWeather.datetime_utc.desc())
        .limit(1)
    ).scalar()
    return result


def forward_fill_null_temps(session, location_id: int, since_utc: datetime) -> int:
    """
    Forward-fill NULL temps with the last known real temp.

    Args:
        session: Database session
        location_id: vc_location_id to forward-fill
        since_utc: Only update records newer than this time

    Returns:
        Number of records forward-filled
    """
    # Get last known real temp
    last_temp = get_last_known_temp(session, location_id)

    if last_temp is None:
        logger.debug(f"Location {location_id}: No historical temp to forward-fill from")
        return 0

    # Update all NULL temps since the given time
    stmt = (
        update(VcMinuteWeather)
        .where(
            and_(
                VcMinuteWeather.vc_location_id == location_id,
                VcMinuteWeather.data_type == "actual_obs",
                VcMinuteWeather.datetime_utc >= since_utc,
                VcMinuteWeather.temp_f.is_(None),
            )
        )
        .values(temp_f=last_temp, is_forward_filled=True)
    )

    result = session.execute(stmt)
    return result.rowcount


def upsert_vc_forecast_hourly(session, records: List[Dict[str, Any]]) -> int:
    """Upsert hourly forecast records."""
    if not records:
        return 0

    stmt = insert(VcForecastHourly).values(records)

    # Unique index: uq_vc_hourly_row (vc_location_id, target_datetime_utc, forecast_basis_date, data_type)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcForecastHourly.__table__.columns
        if col.name not in ("id", "vc_location_id", "target_datetime_utc", "forecast_basis_date", "data_type", "created_at")
    }

    stmt = stmt.on_conflict_do_update(
        index_elements=["vc_location_id", "target_datetime_utc", "forecast_basis_date", "data_type"],
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


def upsert_vc_forecast_daily(session, records: List[Dict[str, Any]]) -> int:
    """Upsert daily forecast records."""
    if not records:
        return 0

    stmt = insert(VcForecastDaily).values(records)

    # Unique index: uq_vc_daily_row (vc_location_id, target_date, forecast_basis_date, data_type)
    update_cols = {
        col.name: stmt.excluded[col.name]
        for col in VcForecastDaily.__table__.columns
        if col.name not in ("id", "vc_location_id", "target_date", "forecast_basis_date", "data_type", "created_at")
    }

    stmt = stmt.on_conflict_do_update(
        index_elements=["vc_location_id", "target_date", "forecast_basis_date", "data_type"],
        set_=update_cols,
    )

    result = session.execute(stmt)
    return result.rowcount


class VcLiveDaemon:
    """
    Visual Crossing live data polling daemon.

    Polls observations (5-min intervals) and forecasts (15-min intervals)
    for all configured locations.
    """

    def __init__(
        self,
        client: VisualCrossingClient,
        locations: List[LocationInfo],
        enable_obs: bool = True,
        enable_forecasts: bool = True,
        api_delay: float = API_RATE_LIMIT_SECONDS,
    ):
        self.client = client
        self.locations = locations
        self.enable_obs = enable_obs
        self.enable_forecasts = enable_forecasts
        self.api_delay = api_delay

        # Track last poll times per location
        self.last_obs_poll: Dict[int, datetime] = {}
        self.last_forecast_poll: Dict[int, datetime] = {}

        # Stats
        self.total_obs_records = 0
        self.total_forecast_records = 0
        self.poll_cycles = 0

    def get_latest_obs_time(self, session, location_id: int) -> Optional[datetime]:
        """Get the most recent observation timestamp for a location."""
        result = session.execute(
            select(func.max(VcMinuteWeather.datetime_utc))
            .where(VcMinuteWeather.vc_location_id == location_id)
            .where(VcMinuteWeather.data_type == "actual_obs")
        ).scalar()
        return result

    def backfill_gaps(self, session) -> int:
        """
        Check for gaps in observations and backfill up to GAP_LOOKBACK_HOURS.

        Returns total records backfilled.
        """
        now_utc = datetime.now(timezone.utc)
        lookback_start = now_utc - timedelta(hours=GAP_LOOKBACK_HOURS)
        total_backfilled = 0

        for location in self.locations:
            latest = self.get_latest_obs_time(session, location.id)

            if latest is None:
                # No data at all - backfill from lookback start
                gap_start = lookback_start
                logger.info(f"{location.city_code}/{location.location_type}: No data, backfilling from {gap_start}")
            elif latest < lookback_start:
                # Data is older than lookback window - backfill from lookback start
                gap_start = lookback_start
                logger.info(f"{location.city_code}/{location.location_type}: Data older than {GAP_LOOKBACK_HOURS}h, backfilling from {gap_start}")
            elif latest < now_utc - timedelta(minutes=10):
                # Gap exists within lookback window
                gap_start = latest
                logger.info(f"{location.city_code}/{location.location_type}: Gap detected, backfilling from {gap_start}")
            else:
                # Data is recent enough
                logger.debug(f"{location.city_code}/{location.location_type}: Data current as of {latest}")
                continue

            # Fetch and upsert observations for the gap
            try:
                start_date = gap_start.date()
                end_date = now_utc.date()

                if location.location_type == "station":
                    data = self.client.fetch_station_history_minutes(
                        station_id=location.station_id,
                        start_date=start_date,
                        end_date=end_date,
                    )
                else:
                    data = self.client.fetch_city_history_minutes(
                        city_query=location.vc_location_query,
                        start_date=start_date,
                        end_date=end_date,
                    )

                records = flatten_vc_response_to_minutes(
                    data, location.id, location.iana_timezone, data_type="actual_obs"
                )

                # Filter to only records after the gap start
                records = [r for r in records if r["datetime_utc"] > gap_start]

                if records:
                    rows = upsert_vc_minute_weather(session, records)
                    total_backfilled += rows
                    logger.info(f"{location.city_code}/{location.location_type}: Backfilled {rows} records")

                    # Forward-fill any NULL temps from backfill
                    ff_count = forward_fill_null_temps(session, location.id, gap_start)
                    if ff_count > 0:
                        logger.info(f"{location.city_code}/{location.location_type}: Forward-filled {ff_count} NULL temps during backfill")

                time.sleep(self.api_delay)

            except Exception as e:
                logger.error(f"Error backfilling {location.city_code}/{location.location_type}: {e}")

        session.commit()
        return total_backfilled

    def poll_observations(self, session) -> int:
        """
        Poll current observations for all locations.

        After upserting, forward-fills any NULL temps with the last known real temp.
        Returns number of records upserted.
        """
        total_records = 0
        total_forward_filled = 0
        now_utc = datetime.now(timezone.utc)
        # Forward-fill window: look back 1 hour to catch any recently inserted NULL temps
        forward_fill_since = now_utc - timedelta(hours=1)

        for location in self.locations:
            # Check if enough time has passed since last poll
            last_poll = self.last_obs_poll.get(location.id)
            if last_poll and (now_utc - last_poll).total_seconds() < OBS_INTERVAL_SECONDS:
                continue

            try:
                # Fetch today's observations (VC returns 5-min intervals)
                today = now_utc.date()

                if location.location_type == "station":
                    data = self.client.fetch_station_history_minutes(
                        station_id=location.station_id,
                        start_date=today,
                        end_date=today,
                    )
                else:
                    data = self.client.fetch_city_history_minutes(
                        city_query=location.vc_location_query,
                        start_date=today,
                        end_date=today,
                    )

                records = flatten_vc_response_to_minutes(
                    data, location.id, location.iana_timezone, data_type="actual_obs"
                )

                if records:
                    rows = upsert_vc_minute_weather(session, records)
                    total_records += rows
                    logger.debug(f"{location.city_code}/{location.location_type}: {rows} obs records")

                # Forward-fill any NULL temps for this location
                ff_count = forward_fill_null_temps(session, location.id, forward_fill_since)
                if ff_count > 0:
                    total_forward_filled += ff_count
                    logger.info(f"{location.city_code}/{location.location_type}: Forward-filled {ff_count} NULL temps")

                self.last_obs_poll[location.id] = now_utc
                time.sleep(self.api_delay)

            except Exception as e:
                logger.error(f"Error polling obs for {location.city_code}/{location.location_type}: {e}")

        if total_records > 0 or total_forward_filled > 0:
            session.commit()
            self.total_obs_records += total_records

        return total_records

    def poll_forecasts(self, session) -> int:
        """
        Poll current forecasts for all locations.

        Returns number of records upserted.
        """
        total_hourly = 0
        total_daily = 0
        now_utc = datetime.now(timezone.utc)

        for location in self.locations:
            # Check if enough time has passed since last poll
            last_poll = self.last_forecast_poll.get(location.id)
            if last_poll and (now_utc - last_poll).total_seconds() < FORECAST_INTERVAL_SECONDS:
                continue

            try:
                # Fetch current forecast (includes 15-day forecast)
                if location.location_type == "station":
                    data = self.client.fetch_station_current_and_forecast(
                        station_id=location.station_id,
                    )
                else:
                    data = self.client.fetch_city_current_and_forecast(
                        city_query=location.vc_location_query,
                    )

                # Parse hourly forecasts
                hourly_records = []
                daily_records = []

                for day in data.get("days", []):
                    # Daily forecast
                    daily_record = parse_vc_daily_forecast(
                        day, location.id, location.iana_timezone, now_utc
                    )
                    if daily_record:
                        daily_records.append(daily_record)

                    # Hourly forecasts
                    for hour in day.get("hours", []):
                        hourly_record = parse_vc_hourly_forecast(
                            hour, location.id, location.iana_timezone, now_utc
                        )
                        if hourly_record:
                            hourly_records.append(hourly_record)

                if hourly_records:
                    rows = upsert_vc_forecast_hourly(session, hourly_records)
                    total_hourly += rows

                if daily_records:
                    rows = upsert_vc_forecast_daily(session, daily_records)
                    total_daily += rows

                logger.debug(
                    f"{location.city_code}/{location.location_type}: "
                    f"{len(hourly_records)} hourly, {len(daily_records)} daily forecast records"
                )

                self.last_forecast_poll[location.id] = now_utc
                time.sleep(self.api_delay)

            except Exception as e:
                logger.error(f"Error polling forecast for {location.city_code}/{location.location_type}: {e}")

        total = total_hourly + total_daily
        if total > 0:
            session.commit()
            self.total_forecast_records += total

        return total

    def run_once(self) -> Dict[str, int]:
        """
        Run a single poll cycle.

        Returns dict with counts of records processed.
        """
        results = {"obs": 0, "forecasts": 0, "backfill": 0}

        with get_db_session() as session:
            # First, check for and fill gaps
            results["backfill"] = self.backfill_gaps(session)

            # Poll observations
            if self.enable_obs:
                results["obs"] = self.poll_observations(session)

            # Poll forecasts
            if self.enable_forecasts:
                results["forecasts"] = self.poll_forecasts(session)

        self.poll_cycles += 1
        return results

    def run_forever(self):
        """
        Run the daemon continuously until shutdown is requested.
        """
        global shutdown_requested

        logger.info("=" * 60)
        logger.info("Visual Crossing Live Daemon Starting")
        logger.info(f"  Locations: {len(self.locations)}")
        logger.info(f"  Poll observations: {self.enable_obs}")
        logger.info(f"  Poll forecasts: {self.enable_forecasts}")
        logger.info(f"  Obs interval: {OBS_INTERVAL_SECONDS}s")
        logger.info(f"  Forecast interval: {FORECAST_INTERVAL_SECONDS}s")
        logger.info(f"  API delay: {self.api_delay}s")
        logger.info("=" * 60)

        # Initial backfill
        logger.info("Running initial gap backfill...")
        with get_db_session() as session:
            backfilled = self.backfill_gaps(session)
            logger.info(f"Initial backfill complete: {backfilled} records")

        # Main loop
        while not shutdown_requested:
            try:
                cycle_start = time.time()

                with get_db_session() as session:
                    if self.enable_obs:
                        obs_count = self.poll_observations(session)
                        if obs_count > 0:
                            logger.info(f"Polled {obs_count} observation records")

                    if self.enable_forecasts:
                        forecast_count = self.poll_forecasts(session)
                        if forecast_count > 0:
                            logger.info(f"Polled {forecast_count} forecast records")

                self.poll_cycles += 1

                # Log periodic status
                if self.poll_cycles % 100 == 0:
                    logger.info(
                        f"Status: {self.poll_cycles} cycles, "
                        f"{self.total_obs_records} obs, "
                        f"{self.total_forecast_records} forecasts"
                    )

                # Sleep until next cycle (minimum 1 second)
                cycle_duration = time.time() - cycle_start
                sleep_time = max(1.0, OBS_INTERVAL_SECONDS - cycle_duration)

                # Sleep in small increments to check shutdown flag
                sleep_end = time.time() + sleep_time
                while time.time() < sleep_end and not shutdown_requested:
                    time.sleep(min(1.0, sleep_end - time.time()))

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)  # Brief pause on error

        logger.info("=" * 60)
        logger.info("Visual Crossing Live Daemon Stopped")
        logger.info(f"  Total cycles: {self.poll_cycles}")
        logger.info(f"  Total obs records: {self.total_obs_records}")
        logger.info(f"  Total forecast records: {self.total_forecast_records}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Visual Crossing live data polling daemon"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run single poll cycle then exit"
    )
    parser.add_argument(
        "--obs-only", action="store_true",
        help="Only poll observations"
    )
    parser.add_argument(
        "--forecast-only", action="store_true",
        help="Only poll forecasts"
    )
    parser.add_argument(
        "--city-code", type=str,
        help="Filter by city code (e.g., CHI, DEN)"
    )
    parser.add_argument(
        "--location-type", type=str, choices=["station", "city"],
        help="Filter by location type"
    )
    parser.add_argument(
        "--api-delay", type=float, default=API_RATE_LIMIT_SECONDS,
        help=f"Seconds between API calls (default: {API_RATE_LIMIT_SECONDS})"
    )

    args = parser.parse_args()

    # Set up signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Determine what to poll
    poll_obs = not args.forecast_only
    poll_forecasts = not args.obs_only

    # Get settings and create client
    settings = get_settings()
    client = VisualCrossingClient(
        api_key=settings.vc_api_key,
        base_url=settings.vc_base_url,
    )

    # Load locations from database (convert to dataclass to avoid session detachment)
    with get_db_session() as session:
        query = select(VcLocation)

        if args.city_code:
            query = query.where(VcLocation.city_code == args.city_code.upper())
        if args.location_type:
            query = query.where(VcLocation.location_type == args.location_type)

        db_locations = list(session.execute(query).scalars().all())
        locations = [LocationInfo.from_db(loc) for loc in db_locations]

    if not locations:
        logger.error("No locations found matching criteria")
        sys.exit(1)

    logger.info(f"Found {len(locations)} locations to poll")
    for loc in locations:
        logger.info(f"  - {loc.city_code}/{loc.location_type}: {loc.vc_location_query}")

    # Create and run daemon
    daemon = VcLiveDaemon(
        client=client,
        locations=locations,
        enable_obs=poll_obs,
        enable_forecasts=poll_forecasts,
        api_delay=args.api_delay,
    )

    if args.once:
        logger.info("Running single poll cycle...")
        results = daemon.run_once()
        logger.info(f"Results: backfill={results['backfill']}, obs={results['obs']}, forecasts={results['forecasts']}")
    else:
        daemon.run_forever()


if __name__ == "__main__":
    main()
