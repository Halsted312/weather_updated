"""Visual Crossing API elements configuration.

Centralized elements list for all Visual Crossing API calls.
This ensures consistent field requests across all ingestion scripts.

Field names verified against:
- https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/
- https://www.visualcrossing.com/resources/documentation/weather-api/energy-elements-in-the-timeline-weather-api/
"""

# Core weather elements available at all time resolutions
CORE_ELEMENTS = [
    "datetime",
    "datetimeEpoch",
    "temp",
    "tempmax",
    "tempmin",
    "feelslike",
    "feelslikemax",
    "feelslikemin",
    "dew",
    "humidity",
    "precip",
    "precipprob",
    "preciptype",
    "precipcover",
    "snow",
    "snowdepth",
    "windspeed",
    "windgust",
    "winddir",
    "cloudcover",
    "visibility",
    "pressure",
    "uvindex",
    "solarradiation",
    "solarenergy",
    "conditions",
    "icon",
    "stations",
]

# Extended wind elements at different heights (50m, 80m, 100m)
# Useful for energy/wind farm applications and ML features
EXTENDED_WIND = [
    "windspeed50",
    "winddir50",
    "windspeed80",
    "winddir80",
    "windspeed100",
    "winddir100",
]

# Extended solar/radiation elements
# DNI = Direct Normal Irradiance, DIF = Diffuse, GHI = Global Horizontal, GTI = Global Tilted
EXTENDED_SOLAR = [
    "dniradiation",
    "difradiation",
    "ghiradiation",
    "gtiradiation",
    "sunelevation",
    "sunazimuth",
]

# "add:" prefix fields - additional computed or metadata fields
# Note: degreedays/accdegreedays are daily constructs - constant per day at minute resolution
ADD_FIELDS = [
    "add:cape",           # Convective Available Potential Energy
    "add:cin",            # Convective Inhibition
    "add:deltat",         # Temperature change
    "add:degreedays",     # Heating/cooling degree days
    "add:accdegreedays",  # Accumulated degree days
    "add:elevation",      # Location elevation in meters
    "add:latitude",       # Location latitude
    "add:longitude",      # Location longitude
    "add:timezone",       # IANA timezone name
    "add:tzoffset",       # Timezone offset from UTC in hours
    "add:windspeedmean",  # Mean wind speed
    "add:windspeedmin",   # Minimum wind speed
    "add:windspeedmax",   # Maximum wind speed
    "add:precipremote",   # Remote/radar precipitation
    "add:resolvedAddress",  # Resolved address string from VC
]


def build_elements_string() -> str:
    """Build comma-separated elements string for VC API calls.

    Returns:
        Comma-separated string of all elements to request from Visual Crossing API.

    Example:
        >>> elements = build_elements_string()
        >>> url = f"...?elements={elements}&..."
    """
    return ",".join(CORE_ELEMENTS + EXTENDED_WIND + EXTENDED_SOLAR + ADD_FIELDS)


def get_all_elements() -> list[str]:
    """Get list of all element names.

    Returns:
        List of all element names (without 'add:' prefix stripped).
    """
    return CORE_ELEMENTS + EXTENDED_WIND + EXTENDED_SOLAR + ADD_FIELDS


# Mapping from VC response field names to database column names
# VC uses camelCase, we use snake_case with unit suffixes
VC_TO_DB_FIELD_MAP = {
    # Core weather
    "temp": "temp_f",
    "tempmax": "tempmax_f",
    "tempmin": "tempmin_f",
    "feelslike": "feelslike_f",
    "feelslikemax": "feelslikemax_f",
    "feelslikemin": "feelslikemin_f",
    "dew": "dew_f",
    "humidity": "humidity",
    "precip": "precip_in",
    "precipprob": "precipprob",
    "preciptype": "preciptype",
    "precipcover": "precipcover",
    "snow": "snow_in",
    "snowdepth": "snowdepth_in",
    "windspeed": "windspeed_mph",
    "windgust": "windgust_mph",
    "winddir": "winddir",
    "cloudcover": "cloudcover",
    "visibility": "visibility_miles",
    "pressure": "pressure_mb",
    "uvindex": "uvindex",
    "solarradiation": "solarradiation",
    "solarenergy": "solarenergy",
    "conditions": "conditions",
    "icon": "icon",
    "stations": "stations",

    # Extended wind
    "windspeed50": "windspeed50_mph",
    "winddir50": "winddir50",
    "windspeed80": "windspeed80_mph",
    "winddir80": "winddir80",
    "windspeed100": "windspeed100_mph",
    "winddir100": "winddir100",

    # Extended solar
    "dniradiation": "dniradiation",
    "difradiation": "difradiation",
    "ghiradiation": "ghiradiation",
    "gtiradiation": "gtiradiation",
    "sunelevation": "sunelevation",
    "sunazimuth": "sunazimuth",

    # Add fields (note: VC returns these without 'add:' prefix in response)
    "cape": "cape",
    "cin": "cin",
    "deltat": "deltat",
    "degreedays": "degreedays",
    "accdegreedays": "accdegreedays",
    "elevation": "elevation_m",  # Will be stored in vc_location
    "latitude": "latitude",      # Will be stored in vc_location
    "longitude": "longitude",    # Will be stored in vc_location
    "timezone": "timezone",
    "tzoffset": "tzoffset_minutes",  # Multiply by 60 when storing
    "windspeedmean": "windspeedmean_mph",
    "windspeedmin": "windspeedmin_mph",
    "windspeedmax": "windspeedmax_mph",
    "precipremote": "precipremote",
    "resolvedAddress": "resolved_address",
}
