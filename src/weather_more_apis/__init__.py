"""
NOAA model guidance integration for Kalshi Weather Trading System.

This package provides ingestion and feature computation for three NOAA forecast sources:
- NBM (National Blend of Models): Ensemble blend, hourly updates
- HRRR (High-Resolution Rapid Refresh): 3km grid, hourly updates
- NDFD (National Digital Forecast Database): Official NWS forecast, 4x daily

Data flow:
1. Download GRIB2 files from public NOAA S3 buckets
2. Extract temperature at city lat/lon using wgrib2
3. Compute peak window (13-18 local) max temp
4. Store scalar summaries in wx.weather_more_apis_guidance
5. Expose 8 features via models/features/more_apis.py

All three models are stored in a unified table and exposed as a single feature block.
"""

__version__ = "0.1.0"

from src.weather_more_apis.model_specs import (
    NBM_SPEC,
    HRRR_SPEC,
    NDFD_SPEC,
    MODEL_SPECS,
    build_s3_url,
)

__all__ = [
    "NBM_SPEC",
    "HRRR_SPEC",
    "NDFD_SPEC",
    "MODEL_SPECS",
    "build_s3_url",
]
