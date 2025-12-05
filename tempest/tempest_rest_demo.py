#!/usr/bin/env python3
"""
Quick REST demo for a Tempest weather station.

- Prints basic station info
- Prints current conditions & forecast for your station
- Prints current conditions for a nearby point ("surrounding area" example)
"""

import os
from typing import Dict, Any

import requests  # pip install requests


BASE_URL = "https://swd.weatherflow.com/swd/rest"


def get_token() -> str:
    """Read a Tempest access token from common env var names."""
    for name in ("TEMPEST_API_KEY", "TEMPEST_API_TOKEN", "tempest_api_key"):
        value = os.getenv(name)
        if value:
            return value
    raise SystemExit(
        "Set your Tempest token in one of TEMPEST_API_KEY, "
        "TEMPEST_API_TOKEN, or tempest_api_key."
    )


def get_default_station(token: str) -> Dict[str, Any]:
    """Return the first station associated with this account."""
    resp = requests.get(f"{BASE_URL}/stations", params={"token": token}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    stations = data.get("stations") or []
    if not stations:
        raise RuntimeError("No stations found for this token.")
    # Just pick the first station for this demo
    return stations[0]


def get_better_forecast_for_station(
    station_id: int, token: str, units: Dict[str, str]
) -> Dict[str, Any]:
    params = {"station_id": station_id, "token": token}
    params.update(units)
    resp = requests.get(f"{BASE_URL}/better_forecast", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_better_forecast_for_point(
    lat: float, lon: float, token: str, units: Dict[str, str], snap_to_station: bool = True
) -> Dict[str, Any]:
    """
    Forecast for an arbitrary lat/lon ("surrounding area").

    snap_to_station=False gives you the model forecast for that point;
    True will snap to your nearest owned station within ~5km per docs.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "token": token,
        "snap_to_nearest_owned_station": str(snap_to_station).lower(),
    }
    params.update(units)
    resp = requests.get(f"{BASE_URL}/better_forecast", params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    token = get_token()
    station = get_default_station(token)
    station_id = station["station_id"]
    name = station.get("public_name") or station.get("name")
    lat = station["latitude"]
    lon = station["longitude"]

    print(f"Using station {station_id} ({name}) at ({lat:.4f}, {lon:.4f})")

    units = {
        "units_temp": "f",      # 'c' for Celsius
        "units_wind": "mph",    # 'mps', 'kph', 'kts', 'bft', 'lfm'
        "units_pressure": "mb", # 'mb', 'inhg', 'mmhg', 'hpa'
        "units_precip": "in",   # 'mm', 'cm', 'in'
        "units_distance": "mi", # 'km', 'mi'
    }

    # 1) Forecast + current conditions for YOUR station
    station_fc = get_better_forecast_for_station(station_id, token, units)
    cc_station = station_fc["current_conditions"]

    print("\n--- Current conditions at YOUR station ---")
    print(
        f"{cc_station['conditions']}, "
        f"{cc_station['air_temperature']}°F, "
        f"RH {cc_station['relative_humidity']}%, "
        f"wind {cc_station['wind_avg']} {units['units_wind']} "
        f"from {cc_station['wind_direction_cardinal']}"
    )

    # 2) "Surrounding area" – example: slightly north of your station
    nearby_lat = lat + 0.1    # ~10–12 km; tweak however you like
    nearby_lon = lon

    point_fc = get_better_forecast_for_point(
        nearby_lat, nearby_lon, token, units, snap_to_station=False
    )
    cc_point = point_fc["current_conditions"]

    print("\n--- Current conditions at NEARBY point ---")
    print(
        f"{cc_point['conditions']}, "
        f"{cc_point['air_temperature']}°F, "
        f"RH {cc_point['relative_humidity']}%, "
        f"wind {cc_point['wind_avg']} {units['units_wind']} "
        f"from {cc_point['wind_direction_cardinal']}"
    )

    # 3) Today’s high/low from daily forecast at your station
    today = station_fc["forecast"]["daily"][0]
    print("\n--- Today at your station (forecast) ---")
    print(
        f"High {today['air_temp_high']}°F, "
        f"Low {today['air_temp_low']}°F, "
        f"Precip prob {today['precip_probability']}%"
    )


if __name__ == "__main__":
    main()
