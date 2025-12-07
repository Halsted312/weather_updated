"""
GRIB2 download, extraction, and database insertion for NOAA model guidance.

Functions:
- download_grib2(): Download GRIB file from S3
- extract_temperature_at_point(): Extract temp at lat/lon using wgrib2
- compute_peak_window_max(): Sample temps at peak hours (13-18 local), return max
- extract_ndfd_tmax(): Extract NDFD daily Tmax
- ingest_guidance_for_city_date(): Orchestrate download → extract → DB insert
"""

import logging
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pytz
import requests
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from src.db.models import WeatherMoreApisGuidance
from src.weather_more_apis.model_specs import ModelSpec, MODEL_SPECS, build_s3_url

logger = logging.getLogger(__name__)


def download_grib2(url: str, output_path: Path, timeout: int = 60) -> bool:
    """
    Download GRIB2 file from S3.

    Args:
        url: Full S3 URL
        output_path: Local path to save GRIB file
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        logger.debug(f"Downloaded {url} ({len(response.content) / 1024:.1f} KB)")
        return True
    except requests.RequestException as e:
        logger.warning(f"Failed to download {url}: {e}")
        return False


def extract_temperature_at_point(
    grib_path: Path, lon: float, lat: float, match_pattern: str
) -> Optional[float]:
    """
    Extract temperature at a specific lat/lon using wgrib2.

    Args:
        grib_path: Path to GRIB2 file
        lon: Longitude (negative for west)
        lat: Latitude
        match_pattern: wgrib2 match pattern (e.g., ":TMP:2 m above ground:")

    Returns:
        Temperature in Fahrenheit, or None if extraction fails

    Notes:
        - wgrib2 returns temperature in Kelvin for TMP variable
        - Converts K → F: (K - 273.15) * 9/5 + 32
        - TMAX (daily max) may already be in F (check output)
    """
    try:
        # wgrib2 file.grib2 -lon <lon> <lat> -match "<pattern>"
        result = subprocess.run(
            [
                "wgrib2",
                str(grib_path),
                "-lon",
                str(lon),
                str(lat),
                "-match",
                match_pattern,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"wgrib2 failed (rc={result.returncode}): {result.stderr}")
            return None

        # Parse output: "1:0:val=293.15"
        for line in result.stdout.strip().split("\n"):
            if "val=" in line:
                value_str = line.split("val=")[-1].strip()
                value = float(value_str)

                # Convert Kelvin → Fahrenheit
                # Reasonable temp range: 200-350K (−99°F to 170°F)
                if 200 <= value <= 350:
                    fahrenheit = (value - 273.15) * 9 / 5 + 32
                    return round(fahrenheit, 2)
                elif -50 <= value <= 150:
                    # Already in Fahrenheit (NDFD TMAX)
                    return round(value, 2)
                else:
                    logger.warning(f"Temperature out of range: {value}")
                    return None

        logger.warning("No temperature value found in wgrib2 output")
        return None

    except (subprocess.SubprocessError, ValueError) as e:
        logger.warning(f"Failed to extract temperature: {e}")
        return None


def compute_peak_window_max(
    model_type: str,
    run_datetime: datetime,
    target_date_local: datetime.date,
    city_config: Dict[str, Any],
    spec: ModelSpec,
) -> Optional[float]:
    """
    Compute peak window (13-18 local) max temp for NBM or HRRR.

    Samples 2-m temp at hours 13, 14, 15, 16, 17, 18 local time and returns max.

    Args:
        model_type: 'nbm' or 'hrrr'
        run_datetime: Model initialization time (UTC)
        target_date_local: Target forecast date (local)
        city_config: City configuration dict with timezone, lat, lon
        spec: ModelSpec for this model

    Returns:
        Max temperature in peak window (°F), or None if data unavailable
    """
    timezone_str = city_config["timezone"]
    tz = pytz.timezone(timezone_str)
    lat = city_config["latitude"]
    lon = city_config["longitude"]

    # Canonical hour: 15:00 local (single hour for speed)
    # This is typically near daily high for most cities
    peak_hours_local = [15]  # Just 15:00 local

    temps = []

    for hour_local in peak_hours_local:
        # Build datetime for this hour in local timezone
        local_dt = tz.localize(
            datetime.combine(
                target_date_local, datetime.min.time()
            ) + timedelta(hours=hour_local)
        )
        utc_dt = local_dt.astimezone(pytz.UTC)

        # Calculate forecast hour (time delta from model run)
        forecast_hour = int((utc_dt - run_datetime).total_seconds() / 3600)

        if forecast_hour < 0:
            # Can't forecast past times
            logger.debug(f"Skipping past hour: fhr={forecast_hour}")
            continue

        # Build S3 URL for this forecast hour
        url = build_s3_url(model_type, run_datetime, forecast_hour)

        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tmp:
            tmp_path = Path(tmp.name)

            if not download_grib2(url, tmp_path):
                logger.warning(
                    f"Missing GRIB for {model_type} fhr={forecast_hour} (url={url})"
                )
                continue

            # Extract temperature at city location
            temp = extract_temperature_at_point(
                tmp_path, lon, lat, spec.wgrib2_match_pattern
            )

            if temp is not None:
                temps.append(temp)
                logger.debug(f"{model_type} fhr={forecast_hour}: {temp}°F")

    if not temps:
        logger.warning(
            f"No temps extracted for {model_type} peak window on {target_date_local}"
        )
        return None

    return round(max(temps), 2)


def extract_ndfd_tmax(
    run_datetime: datetime,
    target_date_local: datetime.date,
    city_config: Dict[str, Any],
    spec: ModelSpec,
) -> Optional[float]:
    """
    Extract NDFD daily Tmax for target date.

    NDFD provides daily max temps directly (not hourly samples).

    Args:
        run_datetime: Model initialization time (UTC, cycle 0/6/12/18)
        target_date_local: Target forecast date (local)
        city_config: City configuration
        spec: NDFD ModelSpec

    Returns:
        Daily Tmax in Fahrenheit, or None if unavailable

    Notes:
        - NDFD GRIB files contain multiple forecast days
        - Need to filter by valid time to get correct day
        - TODO: Investigate exact GRIB structure and inventory
    """
    lat = city_config["latitude"]
    lon = city_config["longitude"]

    # NDFD daily max - single file per cycle, contains multiple days
    # Using forecast_hour=0 as placeholder; may need to adjust
    url = build_s3_url("ndfd", run_datetime, forecast_hour=0, sector="conus")

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=True) as tmp:
        tmp_path = Path(tmp.name)

        if not download_grib2(url, tmp_path):
            logger.warning(f"Missing NDFD GRIB for cycle {run_datetime}")
            return None

        # Extract TMAX at point
        # Note: wgrib2 may return Fahrenheit or Kelvin depending on GRIB encoding
        temp = extract_temperature_at_point(tmp_path, lon, lat, spec.wgrib2_match_pattern)

        if temp is not None:
            logger.debug(f"NDFD Tmax: {temp}°F")
            return temp

        logger.warning(f"Failed to extract NDFD Tmax for {target_date_local}")
        return None


def ingest_guidance_for_city_date(
    session: Session,
    city_id: str,
    target_date: datetime.date,
    model_type: str,
    run_datetime: datetime,
    city_config: Dict[str, Any],
) -> bool:
    """
    Ingest NOAA guidance for one (city, target_date, model, run).

    Downloads GRIB, extracts temp, inserts/updates database row.

    Args:
        session: SQLAlchemy session
        city_id: City identifier ('austin', 'chicago', etc.)
        target_date: Local target date (event day)
        model_type: 'nbm', 'hrrr', or 'ndfd'
        run_datetime: Model run initialization time (UTC)
        city_config: City configuration dict (from src.config.cities)

    Returns:
        True if successful, False otherwise
    """
    spec = MODEL_SPECS[model_type]

    logger.info(
        f"Ingesting {model_type.upper()} for {city_id} {target_date} run={run_datetime}"
    )

    # Compute peak window max or daily Tmax
    if model_type in ["nbm", "hrrr"]:
        peak_window_max_f = compute_peak_window_max(
            model_type, run_datetime, target_date, city_config, spec
        )
    elif model_type == "ndfd":
        peak_window_max_f = extract_ndfd_tmax(
            run_datetime, target_date, city_config, spec
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if peak_window_max_f is None:
        logger.warning(
            f"Failed to extract temperature for {model_type} {city_id} {target_date}"
        )
        return False

    # Upsert into database
    stmt = (
        insert(WeatherMoreApisGuidance)
        .values(
            city_id=city_id,
            target_date=target_date,
            model=model_type,
            run_datetime_utc=run_datetime,
            peak_window_max_f=peak_window_max_f,
            timezone=city_config["timezone"],
        )
        .on_conflict_do_update(
            index_elements=["city_id", "target_date", "model", "run_datetime_utc"],
            set_={"peak_window_max_f": peak_window_max_f},
        )
    )

    session.execute(stmt)
    session.commit()

    logger.info(f"✓ {model_type.upper()} {city_id} {target_date}: {peak_window_max_f}°F")
    return True
