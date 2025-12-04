Perfect, let’s turn this into concrete, copy-pasteable code stubs you can drop into your repo.

I’ll assume:

* You’re using **SQLAlchemy ORM**, with models roughly like `VcLocation` and `VcMinuteWeather` (schema from `007_add_vc_tables.py` / `008_add_forward_fill_tracking.py`).
* You want **pandas DataFrames**, not just ORM objects.
* You need **T-1 15-minute historical forecast series** for **both**:

  * VC **station** (Kalshi’s reference station), and
  * VC **city** (Visual Crossing weighted city aggregate),
    for a given `city_code` and `target_date`.

Below I’ll give you a small module you can add as e.g.:

```text
src/db/vc_minute_queries.py
```

You (or the agent) can adjust imports / paths to match your actual package layout.

---

## 1. ORM model assumptions

I’ll write the stubs assuming these ORM classes (adapt names if they differ):

```python
# models/vc.py (for example)

from sqlalchemy import Column, Integer, Text, Date, DateTime, Float, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class VcLocation(Base):
    __tablename__ = "vc_location"
    __table_args__ = {"schema": "wx"}

    id = Column(Integer, primary_key=True)
    city_code = Column(Text)        # 'AUS', 'CHI', etc.
    kalshi_code = Column(Text)
    location_type = Column(Text)    # 'station' or 'city'
    vc_location_query = Column(Text)
    station_id = Column(Text)
    iana_timezone = Column(Text)
    # ... other columns ...


class VcMinuteWeather(Base):
    __tablename__ = "vc_minute_weather"
    __table_args__ = {"schema": "wx"}

    id = Column(Integer, primary_key=True)
    vc_location_id = Column(Integer, ForeignKey("wx.vc_location.id"))
    data_type = Column(Text)                 # 'actual_obs', 'historical_forecast', etc.
    forecast_basis_date = Column(Date)       # only for forecast/historical_forecast
    datetime_local = Column(DateTime)
    datetime_utc = Column(DateTime)
    temp_f = Column(Float)
    humidity = Column(Float)
    cloudcover = Column(Float)
    dew_f = Column(Float)
    precip_in = Column(Float)
    windspeed_mph = Column(Float)
    uvindex = Column(Float)
    is_forward_filled = Column(Boolean)
    # ... other fields ...
```

If your actual model names or module paths differ, the agent can rename accordingly.

---

## 2. Core helper: fetch T-1 15-minute series for one location_type

Module: `src/db/vc_minute_queries.py`

```python
# src/db/vc_minute_queries.py

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal, Tuple, Dict

import pandas as pd
from sqlalchemy import cast, Date
from sqlalchemy.orm import Session

from models.vc import VcLocation, VcMinuteWeather  # adjust import path as needed


LocationType = Literal["station", "city"]


def get_vc_locations_for_city(
    session: Session,
    city_code: str,
) -> Dict[str, VcLocation]:
    """
    Return a mapping {location_type: VcLocation} for a given city_code.

    location_type will typically be:
      - 'station'  -> VC station (e.g. 'stn:KMDW')
      - 'city'     -> VC city aggregate (e.g. 'Chicago,IL')

    Raises if either type is missing.
    """
    locs = (
        session.query(VcLocation)
        .filter(
            VcLocation.city_code == city_code,
            VcLocation.location_type.in_(["station", "city"]),
        )
        .all()
    )

    by_type: Dict[str, VcLocation] = {}
    for loc in locs:
        by_type[loc.location_type] = loc

    # Optional: safety checks
    expected_types = {"station", "city"}
    missing = expected_types.difference(by_type.keys())
    if missing:
        raise ValueError(
            f"Missing VcLocation rows for city_code={city_code}, "
            f"missing types={missing}. Got types={list(by_type.keys())}."
        )

    return by_type


def fetch_tminus1_minute_forecast_df(
    session: Session,
    city_code: str,
    target_date: date,
    location_type: LocationType,
    include_forward_filled: bool = True,
) -> pd.DataFrame:
    """
    Fetch T-1 15-min historical forecast series from wx.vc_minute_weather
    for a given Kalshi city_code, target_date, and VC location_type.

    - Uses data_type='historical_forecast'
    - forecast_basis_date = target_date - 1
    - DATE(datetime_local) = target_date

    Returns a pandas DataFrame sorted by datetime_local with at least:
      - datetime_local
      - datetime_utc
      - temp_f, humidity, cloudcover, dew_f, precip_in, windspeed_mph, uvindex
    """
    basis_date = target_date - timedelta(days=1)

    # Get the specific VcLocation (station or city) for this city_code
    loc = (
        session.query(VcLocation)
        .filter(
            VcLocation.city_code == city_code,
            VcLocation.location_type == location_type,
        )
        .one()
    )

    q = (
        session.query(
            VcMinuteWeather.datetime_local,
            VcMinuteWeather.datetime_utc,
            VcMinuteWeather.temp_f,
            VcMinuteWeather.humidity,
            VcMinuteWeather.cloudcover,
            VcMinuteWeather.dew_f,
            VcMinuteWeather.precip_in,
            VcMinuteWeather.windspeed_mph,
            VcMinuteWeather.uvindex,
            VcMinuteWeather.is_forward_filled,
        )
        .filter(
            VcMinuteWeather.vc_location_id == loc.id,
            VcMinuteWeather.data_type == "historical_forecast",
            VcMinuteWeather.forecast_basis_date == basis_date,
            cast(VcMinuteWeather.datetime_local, Date) == target_date,
        )
        .order_by(VcMinuteWeather.datetime_local)
    )

    rows = q.all()
    if not rows:
        # Decide how strict you want to be here: raise vs return empty DF
        return pd.DataFrame(
            columns=[
                "datetime_local",
                "datetime_utc",
                "temp_f",
                "humidity",
                "cloudcover",
                "dew_f",
                "precip_in",
                "windspeed_mph",
                "uvindex",
                "is_forward_filled",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "datetime_local",
            "datetime_utc",
            "temp_f",
            "humidity",
            "cloudcover",
            "dew_f",
            "precip_in",
            "windspeed_mph",
            "uvindex",
            "is_forward_filled",
        ],
    )

    if not include_forward_filled and "is_forward_filled" in df.columns:
        df = df[(df["is_forward_filled"] == False) | (df["is_forward_filled"].isna())]

    # Ensure sorted by time just in case
    df = df.sort_values("datetime_local").reset_index(drop=True)

    return df
```

This gives you a clean `DataFrame` for either station or city.

---

## 3. Fetch **both** station & city series together as DataFrames

You often want both series aligned by timestamp to feed `station_city.py` or to compute gaps.

```python
def fetch_station_and_city_tminus1_minute_forecasts(
    session: Session,
    city_code: str,
    target_date: date,
    include_forward_filled: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to fetch T-1 15-min historical forecast series
    for both station and city locations.

    Returns:
      (station_df, city_df)

    Each has at least:
      - datetime_local
      - datetime_utc
      - temp_f, humidity, cloudcover, dew_f, precip_in, windspeed_mph, uvindex
    """
    # Get both vc_location rows up front (sanity check)
    locs_by_type = get_vc_locations_for_city(session, city_code)

    station_df = fetch_tminus1_minute_forecast_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="station",
        include_forward_filled=include_forward_filled,
    )

    city_df = fetch_tminus1_minute_forecast_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="city",
        include_forward_filled=include_forward_filled,
    )

    return station_df, city_df
```

If you want them **merged on timestamps** in a single frame (very handy for station–city gap features), you can add:

```python
def build_station_city_gap_df(
    session: Session,
    city_code: str,
    target_date: date,
    include_forward_filled: bool = True,
) -> pd.DataFrame:
    """
    Fetch T-1 15-min forecast series for station & city and merge into a single DF.

    Result columns:
      - datetime_local
      - datetime_utc
      - temp_station, temp_city, temp_gap
      - humidity_station, humidity_city, humidity_gap
      - cloudcover_station, cloudcover_city, ...
      (etc.)
    """
    station_df, city_df = fetch_station_and_city_tminus1_minute_forecasts(
        session=session,
        city_code=city_code,
        target_date=target_date,
        include_forward_filled=include_forward_filled,
    )

    if station_df.empty or city_df.empty:
        # Either return empty or decide to raise, depending on how strict you want
        return pd.DataFrame()

    s = station_df.rename(
        columns={
            "temp_f": "temp_station",
            "humidity": "humidity_station",
            "cloudcover": "cloudcover_station",
            "dew_f": "dew_station",
            "precip_in": "precip_station",
            "windspeed_mph": "windspeed_station",
            "uvindex": "uvindex_station",
        }
    )
    c = city_df.rename(
        columns={
            "temp_f": "temp_city",
            "humidity": "humidity_city",
            "cloudcover": "cloudcover_city",
            "dew_f": "dew_city",
            "precip_in": "precip_city",
            "windspeed_mph": "windspeed_city",
            "uvindex": "uvindex_city",
        }
    )

    # Outer join on datetime_local to keep full union of timestamps
    merged = pd.merge_asof(
        s.sort_values("datetime_local"),
        c.sort_values("datetime_local"),
        on="datetime_local",
        direction="nearest",  # or "exact" if you expect perfect alignment
        tolerance=pd.Timedelta("7min"),  # 15-min grid → ±7min is safe
    )

    # Compute gaps (station - city)
    merged["temp_gap"] = merged["temp_station"] - merged["temp_city"]
    merged["humidity_gap"] = merged["humidity_station"] - merged["humidity_city"]
    merged["cloudcover_gap"] = merged["cloudcover_station"] - merged["cloudcover_city"]
    merged["windspeed_gap"] = merged["windspeed_station"] - merged["windspeed_city"]

    return merged
```

That `gap_df` is pretty much plug-and-play with your existing `station_city` feature logic.

---

## 4. Helper: infer step_minutes for your 15-min features

Your feature functions (like `forecast_static`, `shape`) can accept a `step_minutes` argument. You can compute it from the DataFrame:

```python
def infer_step_minutes(df: pd.DataFrame) -> int:
    """
    Infer the step size in minutes from a datetime_local Series.
    Assumes regular spacing and at least 2 rows.
    """
    if df.empty or len(df) < 2:
        return 60  # default / fallback

    diffs = df["datetime_local"].diff().dropna().dt.total_seconds() / 60.0
    # Use median to be robust to occasional irregularities
    return int(diffs.median())
```

Usage:

```python
df_fcst = fetch_tminus1_minute_forecast_df(...)
step_minutes = infer_step_minutes(df_fcst)

fcst_series = df_fcst["temp_f"].tolist()
features = compute_forecast_static_features(fcst_series, step_minutes=step_minutes)
```

That way you don’t hardcode “15” anywhere – if Visual Crossing returns 5-min or 10-min intervals for some cases, your functions adapt automatically.

---

## 5. How this ties into your existing feature modules

With these stubs in place, your agent can:

1. For each city + event date:

   * Call `build_station_city_gap_df(...)` to get a ready-made DataFrame with station/city temps, humidity, cloudcover, etc.
   * Extract:

     * `temp_station` series for station-based predictor features.
     * `temp_city` series for city-based features.
     * `temp_gap` series for station-city gap features (e.g. in `station_city.py`).

2. For the **high-temp predictor**:

   * Use `fetch_tminus1_minute_forecast_df(..., location_type="city")` (or station, or both) as the raw 15-min path to feed:

     * `compute_forecast_static_features(fcst_series, step_minutes)`
     * `compute_shape_features` from `shape.py` (spike vs plateau of forecast curve; you already have this machinery).

3. For the **edge classifier**:

   * Use the summarized outputs (forecast_high, forecast_high_time, plateau/spike flags, station-city gaps), not the full minute series.

---

## 6. What to tell the agent

You can tell the agent something like (in your MD plan):

> * Implement `src/db/vc_minute_queries.py` with:
>
>   * `get_vc_locations_for_city(session, city_code)`
>   * `fetch_tminus1_minute_forecast_df(session, city_code, target_date, location_type, include_forward_filled=True)`
>   * `fetch_station_and_city_tminus1_minute_forecasts(...)`
>   * `build_station_city_gap_df(...)`
> * Use these helpers as the **only** place where you touch `wx.vc_minute_weather` directly for forecast series.
> * Everywhere else (ordinal trainer, feature builders, edge pipeline) operate on **pandas DataFrames** returned by these helpers.

Once those are in place, it will be straightforward to plug them into `forecast.py`, `shape.py`, and `station_city.py` like we’ve been discussing.

If you want, next step we can sketch a small `ForecastFeatureBuilder` class that takes those DataFrames and returns a unified `dict[str, float]` of features for a given city/day, so you can wire it cleanly into your ordinal trainer and edge pipeline.
