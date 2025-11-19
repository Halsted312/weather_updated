"""Canonical city configuration shared across ML utilities."""

CITY_CONFIG = {
    "chicago": {"loc_id": "KMDW", "series_code": "CHI", "timezone": "America/Chicago"},
    "miami": {"loc_id": "KMIA", "series_code": "MIA", "timezone": "America/New_York"},
    "austin": {"loc_id": "KAUS", "series_code": "AUS", "timezone": "America/Chicago"},
    "la": {"loc_id": "KLAX", "series_code": "LAX", "timezone": "America/Los_Angeles"},
    "denver": {"loc_id": "KDEN", "series_code": "DEN", "timezone": "America/Denver"},
    "philadelphia": {"loc_id": "KPHL", "series_code": "PHIL", "timezone": "America/New_York"},
}

# All supported cities use Visual Crossing minutes (no exclusions today)
EXCLUDED_VC_CITIES: set[str] = set()
