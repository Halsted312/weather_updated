"""
City configuration - Single source of truth for all city/station mappings.

All 6 cities use airport stations with dense ASOS data (<2% forward-fill in VC).
NYC is excluded due to 82% forward-fill at Central Park (KNYC).
"""

from dataclasses import dataclass
from typing import Dict, Set


@dataclass(frozen=True)
class CityConfig:
    """Configuration for a single city."""

    city_id: str  # Canonical city identifier: 'chicago', 'austin', etc.
    icao: str  # Airport station code: 'KMDW', 'KAUS', etc.
    nws_office: str  # NWS CLI issuing office: 'MDW', 'AUS', etc.
    series_ticker: str  # Kalshi series ticker: 'KXHIGHCHI', etc.
    timezone: str  # IANA timezone: 'America/Chicago', etc.
    ghcnd_station: str  # GHCND station ID for NOAA data
    # New fields for VC greenfield schema
    city_code: str  # 3-letter code: 'CHI', 'DEN', 'AUS', 'LAX', 'MIA', 'PHL'
    kalshi_code: str  # Kalshi market code: 'CHI', 'DEN', 'AUS', 'LAX', 'MIA', 'PHIL'
    vc_city_query: str  # Visual Crossing city query: 'Chicago,IL', etc.
    # Station coordinates from NOAA/NCEI (for historical forecast queries)
    latitude: float  # Station latitude (decimal degrees)
    longitude: float  # Station longitude (decimal degrees)
    # Kalshi 1-min candle coverage (as of 2025-12-11)
    kalshi_candles_first: str  # First date with 1-min candles (may have gaps)
    kalshi_candles_continuous_from: str  # Continuous coverage starts (no gaps after)
    # Recommended training start (earliest date with all data sources available)
    training_start_date: str  # Default start date for model training
    # Delta range for ordinal model training (min, max)
    delta_range: tuple[int, int] = (-3, 3)  # Station forecasts are accurate: 99.4% within ±3°F

    @property
    def vc_location(self) -> str:
        """Visual Crossing station-pinned location format."""
        return f"stn:{self.icao}"

    @property
    def vc_station_query(self) -> str:
        """Visual Crossing station query (alias for vc_location)."""
        return f"stn:{self.icao}"

    @property
    def vc_latlon_query(self) -> str:
        """Visual Crossing lat/lon query for historical forecasts anchored at station."""
        return f"{self.latitude},{self.longitude}"


# All supported cities (6 airport stations with excellent VC coverage)
CITIES: Dict[str, CityConfig] = {
    "chicago": CityConfig(
        city_id="chicago",
        icao="KMDW",
        nws_office="MDW",
        series_ticker="KXHIGHCHI",
        timezone="America/Chicago",
        ghcnd_station="GHCND:USW00014819",
        city_code="CHI",
        kalshi_code="CHI",
        vc_city_query="Chicago,IL",
        latitude=41.78412,
        longitude=-87.75514,
        kalshi_candles_first="2021-12-31",
        kalshi_candles_continuous_from="2023-12-25",  # 1 gap: 2023-12-24
        training_start_date="2023-12-25",  # 716 days, continuous coverage
    ),
    "austin": CityConfig(
        city_id="austin",
        icao="KAUS",
        nws_office="AUS",
        series_ticker="KXHIGHAUS",
        timezone="America/Chicago",
        ghcnd_station="GHCND:USW00013904",  # Austin Bergstrom (Kalshi official), NOT Camp Mabry USW00013958
        city_code="AUS",
        kalshi_code="AUS",
        vc_city_query="Austin,TX",
        latitude=30.18311,
        longitude=-97.67989,
        kalshi_candles_first="2023-05-12",
        kalshi_candles_continuous_from="2025-12-01",  # 5 gaps before this
        training_start_date="2023-05-12",  # 941 usable days (5 gaps - auto filtered)
    ),
    "denver": CityConfig(
        city_id="denver",
        icao="KDEN",
        nws_office="DEN",
        series_ticker="KXHIGHDEN",
        timezone="America/Denver",
        ghcnd_station="GHCND:USW00003017",
        city_code="DEN",
        kalshi_code="DEN",
        vc_city_query="Denver,CO",
        latitude=39.84657,
        longitude=-104.65623,
        kalshi_candles_first="2024-11-20",
        kalshi_candles_continuous_from="2025-12-01",  # 1 gap: 2025-11-30
        training_start_date="2024-11-20",  # 387 usable days (1 gap - auto filtered)
    ),
    "los_angeles": CityConfig(
        city_id="los_angeles",
        icao="KLAX",
        nws_office="LAX",
        series_ticker="KXHIGHLAX",
        timezone="America/Los_Angeles",
        ghcnd_station="GHCND:USW00023174",
        city_code="LAX",
        kalshi_code="LAX",
        vc_city_query="Los Angeles,CA",
        latitude=33.93816,
        longitude=-118.38660,
        kalshi_candles_first="2025-01-05",
        kalshi_candles_continuous_from="2025-01-05",  # No gaps (newest city)
        training_start_date="2025-01-05",  # 342 days, perfect coverage
    ),
    "miami": CityConfig(
        city_id="miami",
        icao="KMIA",
        nws_office="MIA",
        series_ticker="KXHIGHMIA",
        timezone="America/New_York",
        ghcnd_station="GHCND:USW00012839",
        city_code="MIA",
        kalshi_code="MIA",
        vc_city_query="Miami,FL",
        latitude=25.78805,
        longitude=-80.31694,
        kalshi_candles_first="2023-05-12",
        kalshi_candles_continuous_from="2025-12-01",  # 2 gaps: 2023-12-24, 2025-11-30
        training_start_date="2023-05-12",  # 944 usable days (2 gaps - auto filtered)
    ),
    "philadelphia": CityConfig(
        city_id="philadelphia",
        icao="KPHL",
        nws_office="PHL",
        series_ticker="KXHIGHPHIL",  # Note: Kalshi uses "PHIL" not "PHL"
        timezone="America/New_York",
        ghcnd_station="GHCND:USW00013739",
        city_code="PHL",
        kalshi_code="PHIL",  # Kalshi uses "PHIL" not "PHL"
        vc_city_query="Philadelphia,PA",
        latitude=39.87326,
        longitude=-75.22681,
        kalshi_candles_first="2024-11-20",
        kalshi_candles_continuous_from="2025-12-01",  # 1 gap: 2025-11-30
        training_start_date="2024-11-20",  # 387 usable days (1 gap - auto filtered)
    ),
}

# Cities excluded from Visual Crossing minute data
# NYC excluded due to 82% forward-fill (Central Park is a climate station, not ASOS)
EXCLUDED_VC_CITIES: Set[str] = {"new_york"}

# All city IDs
CITY_IDS = list(CITIES.keys())

# All ICAO station codes
STATION_IDS = [city.icao for city in CITIES.values()]

# All series tickers
SERIES_TICKERS = [city.series_ticker for city in CITIES.values()]


def get_city(city_id: str) -> CityConfig:
    """Get city configuration by ID."""
    if city_id not in CITIES:
        raise ValueError(f"Unknown city: {city_id}. Available: {CITY_IDS}")
    return CITIES[city_id]


def get_city_by_icao(icao: str) -> CityConfig:
    """Get city configuration by ICAO station code."""
    for city in CITIES.values():
        if city.icao == icao:
            return city
    raise ValueError(f"Unknown ICAO: {icao}. Available: {STATION_IDS}")


def get_city_by_series(series_ticker: str) -> CityConfig:
    """Get city configuration by Kalshi series ticker."""
    for city in CITIES.values():
        if city.series_ticker == series_ticker:
            return city
    raise ValueError(f"Unknown series: {series_ticker}. Available: {SERIES_TICKERS}")
