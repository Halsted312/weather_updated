#!/usr/bin/env python3
import os
import requests
from dotenv import load_dotenv
import datetime as dt

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DOTENV_PATH = os.path.join(BASE_DIR, ".env.tempest")
load_dotenv(DOTENV_PATH)

BF_URL = "https://swd.weatherflow.com/swd/rest/better_forecast"

token = os.environ.get("TEMPEST_API_KEY")
station_id = os.environ.get("TEMPEST_STATION_ID")
device_id = os.environ.get("TEMPEST_DEVICE_ID")

print("ENV TEMPEST_API_KEY set?", bool(token))
print("ENV TEMPEST_STATION_ID:", station_id)
print("ENV TEMPEST_DEVICE_ID :", device_id)

if not token or not station_id:
    raise SystemExit("Missing token or station_id")

station_id = int(station_id)

units = {
    "units_temp": "f",
    "units_wind": "mph",
    "units_pressure": "mb",
    "units_precip": "in",
    "units_distance": "mi",
}

# Hit better_forecast for your station
params = {"station_id": station_id, **units}
headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

resp = requests.get(BF_URL, params=params, headers=headers, timeout=10)
print("HTTP status:", resp.status_code)
print("Raw snippet:", resp.text[:300], "...\n")

resp.raise_for_status()
data = resp.json()

cc = data["current_conditions"]
print("OK. Current temp at your station:",
      cc["air_temperature"], "°F",
      "at", dt.datetime.fromtimestamp(cc["time"]))
