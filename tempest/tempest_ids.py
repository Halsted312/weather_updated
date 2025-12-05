#!/usr/bin/env python3
import os
import requests
import pprint

TOKEN = os.environ.get("TEMPEST_API_KEY")
if not TOKEN:
    raise SystemExit("Set TEMPEST_API_KEY env var to your Tempest personal access token.")

STATIONS_URL = "https://swd.weatherflow.com/swd/rest/stations"


def get_stations():
    resp = requests.get(STATIONS_URL, params={"token": TOKEN})
    resp.raise_for_status()
    data = resp.json()
    return data["stations"]   # 'stations' list per API docs 


def main():
    stations = get_stations()
    print("Found stations:\n")
    for st in stations:
        print(
            f"- {st['name']} "
            f"(station_id={st['station_id']}, "
            f"lat={st['latitude']}, lon={st['longitude']})"
        )
        for dev in st.get("devices", []):
            print(
                f"    device_id={dev['device_id']} "
                f"type={dev['device_type']} serial={dev.get('serial_number')}"
            )

    print("\nPick the station_id you care about and the Tempest device_id (type 'ST').")


if __name__ == "__main__":
    main()
