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

    @property
    def vc_location(self) -> str:
        """Visual Crossing station-pinned location format."""
        return f"stn:{self.icao}"

    @property
    def vc_station_query(self) -> str:
        """Visual Crossing station query (alias for vc_location)."""
        return f"stn:{self.icao}"


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
    ),
    "austin": CityConfig(
        city_id="austin",
        icao="KAUS",
        nws_office="AUS",
        series_ticker="KXHIGHAUS",
        timezone="America/Chicago",
        ghcnd_station="GHCND:USW00013958",
        city_code="AUS",
        kalshi_code="AUS",
        vc_city_query="Austin,TX",
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
