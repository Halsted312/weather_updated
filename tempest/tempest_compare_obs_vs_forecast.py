#!/usr/bin/env python3
import os
import datetime as dt
import requests
from dotenv import load_dotenv

# Load env from ../.env.tempest
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env.tempest")
load_dotenv(DOTENV_PATH)

TOKEN = os.environ.get("TEMPEST_API_KEY")
STATION_ID = int(os.environ.get("TEMPEST_STATION_ID", "0"))
DEVICE_ID = int(os.environ.get("TEMPEST_DEVICE_ID", "0"))

if not TOKEN or not STATION_ID or not DEVICE_ID:
    raise SystemExit("Check TEMPEST_API_KEY / TEMPEST_STATION_ID / TEMPEST_DEVICE_ID in .env.tempest")

BF_URL = "https://swd.weatherflow.com/swd/rest/better_forecast"
OBS_STATION_URL = f"https://swd.weatherflow.com/swd/rest/observations/station/{STATION_ID}"
OBS_DEVICE_URL = "https://swd.weatherflow.com/swd/rest/observations"


def get_forecast_current():
    params = {
        "station_id": STATION_ID,
        "units_temp": "f",
        "units_wind": "mph",
        "units_pressure": "mb",
        "units_precip": "in",
        "units_distance": "mi",
    }
    headers = {"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"}
    r = requests.get(BF_URL, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    cc = data["current_conditions"]
    t = dt.datetime.fromtimestamp(cc["time"])
    return t, cc["air_temperature"], cc["conditions"]


def get_station_obs():
    params = {"token": TOKEN}
    r = requests.get(OBS_STATION_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    # Latest obs is first element of "obs"
    obs = data["obs"][0]
    # Per docs, station obs fields include "air_temperature" (or "air_temp") etc.
    # In the Home API, keys are usually named (not positional).
    # Print raw dict so you can inspect.
    return data


def get_device_obs():
    params = {"device_id": DEVICE_ID, "token": TOKEN}
    r = requests.get(OBS_DEVICE_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    # For Tempest, you'll get "type": "obs_st" and "obs": [[...]]
    obs_arr = data["obs"][0]
    ts_epoch = obs_arr[0]
    temp_c = obs_arr[7]   # index 7 = air temperature (°C) for obs_st
    rh = obs_arr[8]
    ts = dt.datetime.fromtimestamp(ts_epoch)
    return ts, temp_c, rh, data


def main():
    print(f"Using station_id={STATION_ID}, device_id={DEVICE_ID}\n")

    # Forecast/current-conditions view (outdoor model)
    fc_time, fc_temp_f, fc_cond = get_forecast_current()
    print("FORECAST current_conditions (outdoor, better_forecast):")
    print(f"  time: {fc_time}")
    print(f"  temp: {fc_temp_f:.1f} °F")
    print(f"  cond: {fc_cond}\n")

    # Raw device obs (your Tempest hardware)
    dev_time, dev_temp_c, dev_rh, dev_raw = get_device_obs()
    dev_temp_f = dev_temp_c * 9.0 / 5.0 + 32.0
    print("DEVICE obs_st (raw Tempest sensor):")
    print(f"  time: {dev_time}")
    print(f"  temp: {dev_temp_c:.1f} °C  ({dev_temp_f:.1f} °F)")
    print(f"  RH:   {dev_rh:.0f}%\n")

    # If you want, also dump the station obs dict to inspect structure
    try:
        station_raw = get_station_obs()
        print("STATION obs (full JSON):")
        print(station_raw)
    except Exception as e:
        print("Could not fetch station obs:", e)


if __name__ == "__main__":
    main()
