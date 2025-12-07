"""
NOAA weather model specifications for NBM, HRRR, and NDFD.

Defines S3 bucket paths, GRIB key patterns, and wgrib2 extraction patterns
for each forecast model.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

ModelType = Literal["nbm", "hrrr", "ndfd"]


@dataclass
class ModelSpec:
    """Specification for a NOAA weather model."""

    name: str
    description: str
    grid_resolution_km: float
    update_frequency_hours: int
    s3_bucket: str
    s3_key_pattern: str  # f-string pattern with {date}, {hour}, {fhr}
    wgrib2_match_pattern: str  # Pattern for -match flag
    variable_name: str  # TMP, TMAX, etc.
    level: str  # "2 m above ground", "surface", etc.


# NBM (National Blend of Models)
NBM_SPEC = ModelSpec(
    name="nbm",
    description="National Blend of Models - ensemble blend",
    grid_resolution_km=2.5,
    update_frequency_hours=1,
    s3_bucket="noaa-nbm-grib2-pds",
    s3_key_pattern="blend.{date:%Y%m%d}/{hour:02d}/core/blend.t{hour:02d}z.core.f{fhr:03d}.co.grib2",
    wgrib2_match_pattern=":TMP:2 m above ground:",
    variable_name="TMP",
    level="2 m above ground",
)

# HRRR (High-Resolution Rapid Refresh)
HRRR_SPEC = ModelSpec(
    name="hrrr",
    description="High-Resolution Rapid Refresh - hourly updated",
    grid_resolution_km=3.0,
    update_frequency_hours=1,
    s3_bucket="noaa-hrrr-bdp-pds",
    s3_key_pattern="hrrr.{date:%Y%m%d}/conus/hrrr.t{hour:02d}z.wrfsfcf{fhr:02d}.grib2",
    wgrib2_match_pattern=":TMP:2 m above ground:",
    variable_name="TMP",
    level="2 m above ground",
)

# NDFD (National Digital Forecast Database)
NDFD_SPEC = ModelSpec(
    name="ndfd",
    description="National Digital Forecast Database - 4x daily",
    grid_resolution_km=2.5,
    update_frequency_hours=6,
    s3_bucket="noaa-ndfd-pds",
    # NOTE: Exact path TBD - will investigate during implementation
    s3_key_pattern="NDFD_{sector}_T{cycle:02d}Z/NDFD_{sector}_T{cycle:02d}Z.grib2",
    wgrib2_match_pattern=":TMAX:surface:",  # Daily max temp
    variable_name="TMAX",
    level="surface",
)

MODEL_SPECS = {
    "nbm": NBM_SPEC,
    "hrrr": HRRR_SPEC,
    "ndfd": NDFD_SPEC,
}


def build_s3_url(
    model_type: ModelType, run_datetime: datetime, forecast_hour: int, sector: str = "conus"
) -> str:
    """
    Build S3 URL for GRIB2 file.

    Args:
        model_type: 'nbm', 'hrrr', or 'ndfd'
        run_datetime: Model initialization time (UTC)
        forecast_hour: Forecast lead time in hours
        sector: Geographic sector for NDFD ('conus', 'alaska', etc.)

    Returns:
        Full S3 URL (https://...)

    Examples:
        >>> from datetime import datetime
        >>> build_s3_url("nbm", datetime(2025, 6, 15, 12), 6)
        'https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.20250615/12/core/blend.t12z.core.f006.co.grib2'

        >>> build_s3_url("hrrr", datetime(2025, 6, 15, 12), 6)
        'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.20250615/conus/hrrr.t12z.wrfsfcf06.grib2'
    """
    spec = MODEL_SPECS[model_type]

    if model_type == "ndfd":
        # NDFD uses cycles (0, 6, 12, 18) and sector
        cycle = run_datetime.hour
        key = spec.s3_key_pattern.format(sector=sector, cycle=cycle)
    else:
        # NBM and HRRR use date/hour/forecast_hour
        key = spec.s3_key_pattern.format(
            date=run_datetime, hour=run_datetime.hour, fhr=forecast_hour
        )

    return f"https://{spec.s3_bucket}.s3.amazonaws.com/{key}"
