#!/usr/bin/env python3
import os
import time
import datetime as dt
import requests
from dotenv import load_dotenv

# Load only the Tempest-specific env file from repo root
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env.tempest")
load_dotenv(DOTENV_PATH)

BF_URL = "https://swd.weatherflow.com/swd/rest/better_forecast"

# Midway / KMDW area (airport ref point)
MDW_LAT = 41.7856
MDW_LON = -87.7527

INTERVAL_SECONDS = 5


def get_env():
    token = "7457427a-c7a9-4b5c-bd9c-3b722d740f86"
    
    station_id = 200439

    device_id = os.environ.get("TEMPEST_DEVICE_ID")  # not used here, but validated

    if not token:
        raise SystemExit("TEMPEST_API_KEY env var is missing.")
    if not station_id:
        raise SystemExit("TEMPEST_STATION_ID env var is missing.")
    if not device_id:
        print("Warning: TEMPEST_DEVICE_ID not set (not needed for this script).")

    try:
        station_id_int = int(station_id)
    except ValueError:
        raise SystemExit("TEMPEST_STATION_ID must be an integer.")

    return token, station_id_int


def fetch_better_forecast_station(token, station_id, units):
    """Call /better_forecast for your station_id."""
    params = {"station_id": station_id, **units}
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    resp = requests.get(BF_URL, params=params, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason}: {resp.text}", response=resp
        )
    return resp.json()


def fetch_better_forecast_point(token, lat, lon, units):
    """Call /better_forecast for a lat/lon point near Midway."""
    params = {"lat": lat, "lon": lon, **units}
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    resp = requests.get(BF_URL, params=params, headers=headers, timeout=10)
    if resp.status_code != 200:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason}: {resp.text}", response=resp
        )
    return resp.json()


def summarize_current(label, data, units_temp, units_wind):
    cc = data["current_conditions"]
    ts = dt.datetime.fromtimestamp(cc["time"])
    print(f"\n[{label}] CURRENT @ {ts} local:")
    print(
        f"  Temp: {cc['air_temperature']}°{units_temp.upper()} "
        f"(feels_like={cc['feels_like']}°{units_temp.upper()})"
    )
    print(
        f"  Cond: {cc['conditions']} (icon={cc['icon']})"
        f"\n  Wind: {cc['wind_avg']} {units_wind} "
        f"(gust {cc['wind_gust']} {units_wind}) "
        f"{cc['wind_direction_cardinal']} ({cc['wind_direction']}°)"
    )
    print(
        f"  RH:   {cc['relative_humidity']}%   "
        f"SLP: {cc['sea_level_pressure']} mb   UV: {cc['uv']}"
    )


def summarize_today(label, data, units_temp, units_wind):
    daily0 = data["forecast"]["daily"][0]
    sunrise = dt.datetime.fromtimestamp(daily0["sunrise"])
    sunset = dt.datetime.fromtimestamp(daily0["sunset"])
    print(f"\n[{label}] TODAY FORECAST:")
    print(
        f"  High/Low: {daily0['air_temp_high']} / {daily0['air_temp_low']}°{units_temp.upper()}"
    )
    print(
        f"  Cond:     {daily0['conditions']} (icon={daily0['icon']})"
        f"\n  Precip:   {daily0['precip_probability']}% "
        f"type={daily0.get('precip_type', 'n/a')}"
    )
    print(
        f"  Sunrise:  {sunrise}   Sunset: {sunset}"
        f"\n  Avg wind: {daily0.get('wind_avg', 'n/a')} {units_wind}"
    )


def summarize_next_hours(label, data, units_temp, units_wind, hours=3):
    hourly = data["forecast"]["hourly"][:hours]
    print(f"\n[{label}] NEXT {hours} HOURLY:")
    for h in hourly:
        t = dt.datetime.fromtimestamp(h["time"])
        print(
            f"  {t}: {h['air_temperature']}°{units_temp.upper()} "
            f"{h['conditions']}  wind {h['wind_avg']} {units_wind} "
            f"{h['wind_direction_cardinal']} "
            f"precip_prob={h['precip_probablity']}%"
        )


def stream_loop():
    token, station_id = get_env()

    units = {
        "units_temp": "f",
        "units_wind": "mph",
        "units_pressure": "mb",
        "units_precip": "in",
        "units_distance": "mi",
    }

    print(
        f"Starting Tempest 5-second stream.\n"
        f"Station ID: {station_id}\n"
        f"Midway point: lat={MDW_LAT}, lon={MDW_LON}\n"
        f"Press Ctrl+C to stop.\n"
    )

    while True:
        loop_start = dt.datetime.now()
        print("\n" + "=" * 80)
        print(f"SNAPSHOT @ {loop_start}:\n")

        try:
            station_data = fetch_better_forecast_station(token, station_id, units)
            mdw_data = fetch_better_forecast_point(token, MDW_LAT, MDW_LON, units)

            summarize_current("YOUR STATION", station_data, units["units_temp"], units["units_wind"])
            summarize_today("YOUR STATION", station_data, units["units_temp"], units["units_wind"])
            summarize_next_hours("YOUR STATION", station_data, units["units_temp"], units["units_wind"])

            summarize_current("MIDWAY LAT/LON", mdw_data, units["units_temp"], units["units_wind"])
            summarize_today("MIDWAY LAT/LON", mdw_data, units["units_temp"], units["units_wind"])
            summarize_next_hours("MIDWAY LAT/LON", mdw_data, units["units_temp"], units["units_wind"])

        except requests.HTTPError as e:
            print("HTTP ERROR:", e)
        except Exception as e:
            print("OTHER ERROR:", repr(e))

        elapsed = (dt.datetime.now() - loop_start).total_seconds()
        to_sleep = max(0, INTERVAL_SECONDS - elapsed)
        time.sleep(to_sleep)


if __name__ == "__main__":
    try:
        stream_loop()
    except KeyboardInterrupt:
        print("\nStopping stream. Bye!")
