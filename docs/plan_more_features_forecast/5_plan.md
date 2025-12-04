Great, this is a nice concrete layer to nail down.

You already did the hard part (schema + ingestion design). Now it’s just: “how do I pull a T-1 15-minute series for *both* VC station and VC city for a given Kalshi city + event date?”

I’ll give you:

1. **How the tables fit together (recap)**
2. **Plain SQL examples** for T-1 15-minute series (station vs city)
3. **SQLAlchemy ORM-style queries** (if you have `VcLocation` / `VcMinuteWeather` models)
4. **SQLAlchemy Core-style query** (autoloading tables)
5. **A concrete prompt + doc references** to hand to your coding agent

---

## 1. How your VC tables fit together (recap)

From `007_add_vc_tables.py` and the ingest scripts:

* `wx.vc_location`

  * `id` (PK)
  * `city_code` (e.g. `'AUS'`, `'CHI'`)
  * `kalshi_code` (Kalshi’s code, often same as city_code)
  * `location_type` – `'station'` or `'city'`
  * `vc_location_query` – what you pass to VC:

    * `'stn:KMDW'` for station
    * `'Chicago,IL'` for city aggregate
  * `station_id` – ICAO (e.g. `'KMDW'`) for `location_type='station'`
  * `iana_timezone` – e.g. `'America/Chicago'`

* `wx.vc_minute_weather`

  * `id`
  * `vc_location_id` → FK to `wx.vc_location.id`
  * `data_type` – `'actual_obs' | 'current_snapshot' | 'forecast' | 'historical_forecast'`
  * `forecast_basis_date` – date of the model run (for forecasts / historical_forecasts)
  * `forecast_basis_datetime_utc`
  * `lead_hours`
  * `datetime_utc`, `datetime_local`
  * all the weather fields: `temp_f`, `humidity`, `cloudcover`, `precip_in`, `dew_f`, `windspeed_mph`, `uvindex`, etc.
  * `is_forward_filled` (from rev 008) to flag FF’d temps

So:

* **Station vs city** is controlled by `vc_location.location_type` and `vc_location.vc_location_query`.
* **T-1 forecast** is `data_type='historical_forecast'` + `forecast_basis_date = target_date - 1`.
* **Target day** is `DATE(datetime_local) = target_date` for that minute series.

---

## 2. Raw SQL: T-1 15-minute series for station / city

Assuming you’ve ingested **15-min historical forecasts** into `wx.vc_minute_weather` (via `include=minutes&forecastBasisDay=1`), here’s the basic pattern.

### 2.1 One series for a single location_type

**Inputs:**

* `:city_code` – e.g. `'AUS'`
* `:location_type` – `'station'` or `'city'`
* `:target_date` – the event date (date of the Kalshi market)
* `:basis_date` – `target_date - interval '1 day'` (you can compute either in SQL or in Python)

```sql
-- T-1 historical forecast 15-min series
SELECT
    m.datetime_local,
    m.temp_f,
    m.humidity,
    m.cloudcover,
    m.dew_f,
    m.precip_in,
    m.windspeed_mph,
    m.uvindex
FROM wx.vc_minute_weather AS m
JOIN wx.vc_location AS loc
  ON m.vc_location_id = loc.id
WHERE
    loc.city_code      = :city_code
    AND loc.location_type = :location_type     -- 'station' or 'city'
    AND m.data_type        = 'historical_forecast'
    AND m.forecast_basis_date = :basis_date    -- T-1 basis
    AND m.datetime_local::date = :target_date
ORDER BY
    m.datetime_local;
```

You can decide whether to also filter on `is_forward_filled = FALSE` for “clean-only” series:

```sql
    AND (m.is_forward_filled = FALSE OR m.is_forward_filled IS NULL)
```

### 2.2 Get both station & city in one shot

If you want both series in one query, you can either:

* Run the above twice with `location_type='station'` / `'city'`, or
* Grab them together and distinguish via `location_type` in the result set.

Example:

```sql
SELECT
    loc.location_type,
    loc.vc_location_query,
    m.datetime_local,
    m.temp_f,
    m.humidity,
    m.cloudcover,
    m.dew_f,
    m.precip_in,
    m.windspeed_mph,
    m.uvindex
FROM wx.vc_minute_weather AS m
JOIN wx.vc_location AS loc
  ON m.vc_location_id = loc.id
WHERE
    loc.city_code      = :city_code
    AND loc.location_type IN ('station', 'city')
    AND m.data_type        = 'historical_forecast'
    AND m.forecast_basis_date = :basis_date
    AND m.datetime_local::date = :target_date
ORDER BY
    loc.location_type,
    m.datetime_local;
```

In Python you can then slice this into two series:

* `location_type='station'` → Kalshi station source
* `location_type='city'` → VC weighted city aggregate

---

## 3. SQLAlchemy ORM-style (if you have `VcLocation`, `VcMinuteWeather` models)

If you already have ORM classes like:

```python
class VcLocation(Base):
    __tablename__ = "vc_location"
    __table_args__ = {"schema": "wx"}

    id = Column(Integer, primary_key=True)
    city_code = Column(Text)
    kalshi_code = Column(Text)
    location_type = Column(Text)   # 'station' or 'city'
    vc_location_query = Column(Text)
    station_id = Column(Text)
    iana_timezone = Column(Text)
    ...

class VcMinuteWeather(Base):
    __tablename__ = "vc_minute_weather"
    __table_args__ = {"schema": "wx"}

    id = Column(Integer, primary_key=True)
    vc_location_id = Column(Integer, ForeignKey("wx.vc_location.id"))
    data_type = Column(Text)
    forecast_basis_date = Column(Date)
    datetime_local = Column(DateTime)
    temp_f = Column(Float)
    humidity = Column(Float)
    cloudcover = Column(Float)
    dew_f = Column(Float)
    precip_in = Column(Float)
    windspeed_mph = Column(Float)
    uvindex = Column(Float)
    is_forward_filled = Column(Boolean)
    ...
```

then a nice helper looks like:

```python
from datetime import date, timedelta
from sqlalchemy import cast, Date
from sqlalchemy.orm import Session


def get_tminus1_minute_forecast(
    session: Session,
    city_code: str,
    target_date: date,
    location_type: str,  # 'station' or 'city'
):
    """Return ordered rows for the T-1 15-min historical forecast series."""
    basis_date = target_date - timedelta(days=1)

    loc = (
        session.query(VcLocation)
        .filter(
            VcLocation.city_code == city_code,
            VcLocation.location_type == location_type,
        )
        .one()
    )

    q = (
        session.query(VcMinuteWeather)
        .filter(
            VcMinuteWeather.vc_location_id == loc.id,
            VcMinuteWeather.data_type == "historical_forecast",
            VcMinuteWeather.forecast_basis_date == basis_date,
            cast(VcMinuteWeather.datetime_local, Date) == target_date,
            # Optionally filter out forward-filled records:
            # (VcMinuteWeather.is_forward_filled == False) | (VcMinuteWeather.is_forward_filled.is_(None)),
        )
        .order_by(VcMinuteWeather.datetime_local)
    )

    return q.all()
```

To get **both station & city** at once:

```python
def get_station_and_city_series(
    session: Session,
    city_code: str,
    target_date: date,
):
    basis_date = target_date - timedelta(days=1)

    # Fetch both locations
    locs = (
        session.query(VcLocation)
        .filter(
            VcLocation.city_code == city_code,
            VcLocation.location_type.in_(["station", "city"]),
        )
        .all()
    )
    loc_by_type = {loc.location_type: loc for loc in locs}

    def fetch_for(loc):
        return (
            session.query(VcMinuteWeather)
            .filter(
                VcMinuteWeather.vc_location_id == loc.id,
                VcMinuteWeather.data_type == "historical_forecast",
                VcMinuteWeather.forecast_basis_date == basis_date,
                cast(VcMinuteWeather.datetime_local, Date) == target_date,
            )
            .order_by(VcMinuteWeather.datetime_local)
            .all()
        )

    station_series = fetch_for(loc_by_type["station"])
    city_series = fetch_for(loc_by_type["city"])

    return station_series, city_series
```

You can then convert each list of ORM objects into a `DataFrame` and feed that into your feature functions (`forecast_static`, `shape`, `station_city`, etc.).

---

## 4. SQLAlchemy Core-style (autoloading the tables)

If you don’t have ORM classes or prefer Core, you can:

```python
from datetime import date, timedelta
from sqlalchemy import create_engine, MetaData, Table, select, func, bindparam

engine = create_engine(DATABASE_URL)
metadata = MetaData()
vc_location = Table("vc_location", metadata, schema="wx", autoload_with=engine)
vc_minute_weather = Table("vc_minute_weather", metadata, schema="wx", autoload_with=engine)


def fetch_tminus1_series_core(
    city_code: str,
    target_date: date,
    location_type: str,
):
    basis_date = target_date - timedelta(days=1)

    loc = (
        select(vc_location.c.id)
        .where(
            vc_location.c.city_code == city_code,
            vc_location.c.location_type == location_type,
        )
        .limit(1)
    )

    with engine.begin() as conn:
        loc_id = conn.execute(loc).scalar_one()

        stmt = (
            select(
                vc_minute_weather.c.datetime_local,
                vc_minute_weather.c.temp_f,
                vc_minute_weather.c.humidity,
                vc_minute_weather.c.cloudcover,
                vc_minute_weather.c.dew_f,
                vc_minute_weather.c.precip_in,
                vc_minute_weather.c.windspeed_mph,
                vc_minute_weather.c.uvindex,
            )
            .where(
                vc_minute_weather.c.vc_location_id == loc_id,
                vc_minute_weather.c.data_type == "historical_forecast",
                vc_minute_weather.c.forecast_basis_date == basis_date,
                func.date(vc_minute_weather.c.datetime_local) == target_date,
            )
            .order_by(vc_minute_weather.c.datetime_local)
        )

        rows = conn.execute(stmt).fetchall()

    return rows
```

Again, for both station/city you can just call twice or widen the `where` clause on location_type.

---

## 5. What to tell your coding agent (prompt + doc refs)

Here’s a prompt you can drop into a Markdown doc for the agent, plus which files to study:

---

### Prompt for agent

> **Goal**
> Extend our Visual Crossing pipeline so that, for each Kalshi city + event date, we can pull **T-1 15-minute historical forecast series** for both:
>
> * the **Kalshi station** (e.g. KMDW / `stn:KMDW`), and
> * the **Visual Crossing city aggregate** (e.g. `"Chicago,IL"`).
>
> These series must come from `wx.vc_minute_weather` with `data_type='historical_forecast'` and `forecast_basis_date = target_date - 1`, and must be aligned by `datetime_local` so we can:
>
> * feed them into the **ordinal high-temperature predictor** (forecast_static + shape features)
> * compute **station vs city gap** features over time (using `station_city.py`).
>
> **Tasks**
>
> 1. **Verify schema & ingestion**
>
>    * Confirm that `wx.vc_location` and `wx.vc_minute_weather` are created as in `007_add_vc_tables.py` and that `008_add_forward_fill_tracking.py` has been applied.
>    * Ensure our historical forecast ingestion code (`ingest_vc_hist_forecast_v2.py`) is calling Visual Crossing Timeline with `include=minutes` and `forecastBasisDay/forecastBasisDate` so that **minute-level historical forecasts** are saved into `vc_minute_weather` with `data_type='historical_forecast'`.
> 2. **Add helpers to fetch T-1 15-min series**
>
>    * Implement SQLAlchemy helpers (ORM or Core) that:
>
>      * Given `city_code`, `target_date`, and `location_type` (`'station'` or `'city'`),
>      * Query `wx.vc_location` to get the corresponding `vc_location.id`,
>      * Query `wx.vc_minute_weather` for:
>
>        * `vc_location_id = loc.id`
>        * `data_type = 'historical_forecast'`
>        * `forecast_basis_date = target_date - 1`
>        * `DATE(datetime_local) = target_date`
>      * Return an ordered 15-minute series with at least:
>
>        * `datetime_local`, `temp_f`, `humidity`, `cloudcover`, `dew_f`, `precip_in`, `windspeed_mph`, `uvindex`.
>    * Provide convenience wrappers:
>
>      * `get_station_and_city_series(session, city_code, target_date)` → returns `(station_df, city_df)` with aligned timestamps.
> 3. **Integrate into feature pipeline**
>
>    * In `forecast.py`, generalize `compute_forecast_static_features` to accept `step_minutes` so we can compute `fcst_prev_hour_of_max` as a float (e.g. 14.25 hours for 14:15).
>    * Add new feature groups that operate on the T-1 15-min forecast series:
>
>      * Forecast shape features via `shape.py` (e.g. plateau vs spike of forecast curve).
>      * Optional humidity / cloudcover static features (min/max/mean/range) for the forecast day.
>    * In `station_city.py`, extend the gap features to accept both **minute-level obs** and **minute-level forecast** series for station vs city if helpful.
>
> **Key files to study**
>
> * `007_add_vc_tables.py` – defines `wx.vc_location` and `wx.vc_minute_weather` schema and indexes.
> * `008_add_forward_fill_tracking.py` – adds `is_forward_filled` to `vc_minute_weather`.
> * `visual_crossing.py` – VC API client, shows how we call Timeline for minutes/forecasts.
> * `ingest_vc_hist_forecast_v2.py` – current historical forecast ingestion (days/hours + basisDate/basisDay).
> * `ingest_vc_obs_parallel.py` – minute observation ingestion and tz handling; mirror its patterns for minute-level historical forecasts.
> * `forecast.py` – existing hourly forecast / forecast error features; we’ll adapt these to work with 15-minute series.
> * `shape.py` – shape-of-day features (plateau vs spike); use this for both obs and forecast series.
> * `station_city.py` – station vs city gap features; ensures the station vs city comparison is wired correctly.
>
> **Deliverables**
>
> * A reusable function (and unit tests) to fetch T-1 15-minute series from `vc_minute_weather` for both station and city.
> * Updated forecast feature code to use these series (with minimal disruption to existing pipeline).
> * Small debug notebook or script that, for one city and a few days, plots:
>
>   * station forecast 15-min curve vs city forecast curve,
>   * station vs city gap vs observed temps.

---

If you want, I can next help you sketch the exact query wrappers that return `pandas` DataFrames ready to plug straight into your existing feature functions (`forecast_static`, `shape`, `station_city`) so you don’t have to reshape in a bunch of different places.
