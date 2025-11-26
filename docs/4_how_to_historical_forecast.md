# How to Store and Use Historical Forecasts from Visual Crossing
---

## 0. What we’re trying to store

From Visual Crossing’s own docs:

* Historical forecasts are based on **full forecast model runs stored at midnight UTC each day**.
* You request them using the **Timeline API** plus `forecastBasisDate` (or `forecastBasisDay`) parameters.
* Example from docs:

```text
https://weather.visualcrossing.com/.../timeline/London,UK/2023-05-01/2023-05-15
  ?unitGroup=us
  &include=days
  &key=YOUR_API_KEY
  &forecastBasisDate=2023-05-01
```

That returns the **entire 15-day forecast as it looked on 2023-05-01**.

We want to:

1. Store, for each city and **basis date**, the daily forecast for each **target date** in that 15-day window.
2. Join, for a given Kalshi `event_date`, the **midnight-run forecast high** (`tempmax`) versus the **bracket range**.
3. Use that in backtests and live logic (e.g., “if midnight forecast high is 35°F and all Chicago brackets are 45–50°F, immediately hammer the low bin”).

---

## 1. DB design – reuse `wx` schema, no separate DB

Tell the agent:

> **Don’t create a new database.**
> Keep using the existing Postgres + Timescale instance. We’ll:
>
> * Add a `wx.forecast_snapshot` table for **daily forecast history** (basis_date × target_date).
> * Optionally add `wx.hourly_fcst_basis0` later for hourly Option-2 features.
> * Build a `feature` view to join midnight forecasts vs Kalshi brackets.

### 1.1. Table `wx.forecast_snapshot`

This is the core table: *“what was the forecast for date D at basis date B for city C?”*

```sql
-- migrations/versions/xxxx_add_wx_forecast_snapshot.sql
CREATE TABLE IF NOT EXISTS wx.forecast_snapshot (
    city           TEXT        NOT NULL,  -- 'chicago', 'denver', etc. (same keys as kalshi.markets)
    vc_location    TEXT        NOT NULL,  -- e.g. 'stn:KMDW' or '41.78,-87.75'
    basis_date     DATE        NOT NULL,  -- forecastBasisDate in local date sense
    target_date    DATE        NOT NULL,  -- day being forecast
    lead_days      INT         NOT NULL,  -- (target_date - basis_date)
    -- key forecast fields (unitGroup = us)
    tempmax_f      DOUBLE PRECISION,
    tempmin_f      DOUBLE PRECISION,
    temp_f         DOUBLE PRECISION,      -- mean daily temp, if returned
    feelslikemax_f DOUBLE PRECISION,
    feelslikemin_f DOUBLE PRECISION,
    precip_in      DOUBLE PRECISION,
    precipchance   DOUBLE PRECISION,
    cloudcover_pct DOUBLE PRECISION,
    humidity_pct   DOUBLE PRECISION,
    windspeed_mph  DOUBLE PRECISION,
    wgust_mph      DOUBLE PRECISION,
    conditions     TEXT,
    icon           TEXT,
    -- bookkeeping
    query_cost     INT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_json       JSONB       NOT NULL,  -- full 'day' object from VC

    PRIMARY KEY (city, basis_date, target_date)
);
```

> Agent: add this via Alembic in the `wx` schema, and update tests to assert the table exists.

Later we’ll also plumb `wx.hourly_fcst_basis0`, but you can start with the daily forecast table.

---

## 2. VC client – add historical forecast method

You already have a `VisualCrossingClient` for minute obs. Now add a **historical forecast** method using `forecastBasisDate`. The docs show we should hit the Timeline API and pass `forecastBasisDate` in query params.

### 2.1. URL pattern

For a given city/location and basis_date:

```text
GET https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/
    {location}/{basis_date}/{basis_date+14}
    ?unitGroup=us
    &include=days
    &elements=datetime,datetimeEpoch,tempmax,tempmin,temp,feelslikemax,feelslikemin,
              precip,precipprob,cloudcover,humidity,windspeed,windgust,conditions,icon
    &forecastBasisDate={basis_date}
    &key=YOUR_API_KEY
    &options=histfcst
```

(You can include the `options=` only if needed; the main magic is `forecastBasisDate`.)

### 2.2. Code stub for client

```python
# src/weather/visual_crossing.py

from __future__ import annotations
import datetime as dt
from typing import Any, Dict, List
import logging
import requests

log = logging.getLogger(__name__)

class VisualCrossingClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline",
        session: requests.Session | None = None,
        unit_group: str = "us",
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.unit_group = unit_group

    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(params)
        params.setdefault("key", self.api_key)
        params.setdefault("unitGroup", self.unit_group)
        resp = self.session.get(f"{self.base_url}/{path.lstrip('/')}", params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def fetch_historical_daily_forecast(
        self,
        vc_location: str,         # e.g. 'stn:KMDW'
        basis_date: dt.date,
        horizon_days: int = 15,
    ) -> Dict[str, Any]:
        """
        Fetch the 15-day forecast as it looked on basis_date.

        Uses forecastBasisDate as documented by Visual Crossing:
        https://www.visualcrossing.com/resources/documentation/weather-data/how-to-query-weather-forecasts-from-the-past-historical-forecasts/
        """
        end_date = basis_date + dt.timedelta(days=horizon_days - 1)
        path = f"{vc_location}/{basis_date.isoformat()}/{end_date.isoformat()}"
        params = {
            "include": "days",
            "forecastBasisDate": basis_date.isoformat(),
            # keep payload tight but rich enough for future use:
            "elements": ",".join([
                "datetime",
                "datetimeEpoch",
                "tempmax",
                "tempmin",
                "temp",
                "feelslikemax",
                "feelslikemin",
                "precip",
                "precipprob",
                "cloudcover",
                "humidity",
                "windspeed",
                "windgust",
                "conditions",
                "icon",
                "queryCost",
            ]),
        }
        payload = self._request(path, params=params)
        return payload
```

> Agent: write a small test that calls `fetch_historical_daily_forecast("stn:KMDW", date(2024,1,1))` and asserts `payload["days"]` is non-empty and has `tempmax` fields.

---

## 3. Ingestion script – `scripts/ingest_vc_forecast_history.py`

We now need a script that:

* Loops over **city × basis_date**.
* Calls the VC client.
* Inserts one row per `(city, basis_date, target_date)` into `wx.forecast_snapshot`.

### 3.1. High-level behavior

Tell the agent:

> Implement `scripts/ingest_vc_forecast_history.py` with:
>
> * CLI args:
>
>   * `--start-date` / `--end-date` (ISO) OR `--days` (back from today).
>   * `--city` (repeatable) or `--all-cities`.
>   * `--dry-run`.
> * For each city + basis_date:
>
>   * Call `VisualCrossingClient.fetch_historical_daily_forecast(vc_location, basis_date)`.
>   * For each `day` in `payload["days"]`:
>
>     * `target_date = day["datetime"]` (YYYY-MM-DD, local).
>     * `lead_days = (target_date - basis_date).days`.
>     * Insert/upsert into `wx.forecast_snapshot`.

### 3.2. Code stub

```python
# scripts/ingest_vc_forecast_history.py

import argparse
import datetime as dt
import logging
from typing import Iterable

from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.config import get_settings
from src.db.connection import get_engine
from src.weather.visual_crossing import VisualCrossingClient
from src.models.wx import ForecastSnapshot  # SQLAlchemy model we'll define

log = logging.getLogger(__name__)


CITY_TO_VC_LOCATION = {
    "chicago": "stn:KMDW",
    "denver": "stn:KDEN",
    "austin": "stn:KAUS",
    "los_angeles": "stn:KLAX",
    "miami": "stn:KMIA",
    "philadelphia": "stn:KPHL",
}


def iter_basis_dates(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, help="YYYY-MM-DD (basis date)")
    parser.add_argument("--end-date", type=str, help="YYYY-MM-DD (basis date)")
    parser.add_argument("--days", type=int, help="Backfill N days before today if start/end not given")
    parser.add_argument("--city", action="append", help="City key, e.g. 'chicago'. Repeatable.")
    parser.add_argument("--all-cities", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    client = VisualCrossingClient(api_key=settings.visual_crossing_api_key)

    today = dt.date.today()
    if args.start_date and args.end_date:
        start = dt.date.fromisoformat(args.start_date)
        end = dt.date.fromisoformat(args.end_date)
    else:
        days = args.days or 30
        end = today
        start = today - dt.timedelta(days=days)

    if args.all_cities:
        cities = list(CITY_TO_VC_LOCATION.keys())
    else:
        if not args.city:
            raise SystemExit("Specify --city one or more times or --all-cities")
        cities = args.city

    engine = get_engine()
    with engine.begin() as conn:
        for city in cities:
            vc_loc = CITY_TO_VC_LOCATION[city]
            log.info("Ingesting VC forecast history for %s (%s) from %s to %s", city, vc_loc, start, end)

            for basis_date in iter_basis_dates(start, end):
                # Optional: skip basis_date < 2020-01-01 since historical forecast only starts then
                if basis_date < dt.date(2020, 1, 1):
                    continue

                payload = client.fetch_historical_daily_forecast(vc_loc, basis_date)
                days = payload.get("days", [])
                query_cost = payload.get("queryCost")

                if not days:
                    log.warning("No forecast days for %s basis=%s", city, basis_date)
                    continue

                rows = []
                for day in days:
                    target_str = day["datetime"]  # 'YYYY-MM-DD'
                    target_date = dt.date.fromisoformat(target_str)
                    lead_days = (target_date - basis_date).days

                    # Only keep a reasonable horizon, e.g. 0..14
                    if lead_days < 0 or lead_days > 14:
                        continue

                    rows.append({
                        "city": city,
                        "vc_location": vc_loc,
                        "basis_date": basis_date,
                        "target_date": target_date,
                        "lead_days": lead_days,
                        "tempmax_f": day.get("tempmax"),
                        "tempmin_f": day.get("tempmin"),
                        "temp_f": day.get("temp"),
                        "feelslikemax_f": day.get("feelslikemax"),
                        "feelslikemin_f": day.get("feelslikemin"),
                        "precip_in": day.get("precip"),
                        "precipchance": day.get("precipprob"),
                        "cloudcover_pct": day.get("cloudcover"),
                        "humidity_pct": day.get("humidity"),
                        "windspeed_mph": day.get("windspeed"),
                        "wgust_mph": day.get("windgust"),
                        "conditions": day.get("conditions"),
                        "icon": day.get("icon"),
                        "query_cost": query_cost,
                        "raw_json": day,
                    })

                if not rows:
                    continue

                if args.dry_run:
                    log.info("DRY RUN - would insert %d forecast rows for %s basis=%s", len(rows), city, basis_date)
                else:
                    stmt = pg_insert(ForecastSnapshot.__table__).values(rows)
                    # On conflict, update in case we re-run
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["city", "basis_date", "target_date"],
                        set_={c: stmt.excluded[c] for c in ForecastSnapshot.__table__.columns.keys()
                              if c not in ("city", "basis_date", "target_date")}
                    )
                    conn.execute(stmt)

    log.info("Forecast history ingestion complete.")


if __name__ == "__main__":
    main()
```

> Agent: adapt imports & config to match the actual repo structure, and add unit/integration tests.

---

## 4. Join with Kalshi brackets – view for “midnight forecast vs bracket geometry”

Now we build the feature you actually want:

* For each `(city, event_date, ticker)`:

  * Get **midnight forecast high** (basis_date = target_date, lead_days = 0).
  * Compare it to the **bracket range** from `kalshi.markets`.

Tell the agent to create something like:

```sql
-- migrations/versions/xxxx_add_midnight_forecast_view.sql
CREATE SCHEMA IF NOT EXISTS feature;

CREATE OR REPLACE VIEW feature.midnight_forecast_vs_brackets AS
SELECT
    m.city,
    m.event_date,
    m.ticker,
    m.strike_type,
    m.floor_strike,
    m.cap_strike,
    fs.tempmax_f              AS tempmax_midnight_fcst_f,
    fs.lead_days,
    -- geometry relative to this bin
    (fs.tempmax_f - m.floor_strike) AS delta_to_floor,
    (fs.tempmax_f - m.cap_strike)   AS delta_to_cap,
    -- helpful booleans
    CASE WHEN fs.tempmax_f < m.floor_strike THEN 1 ELSE 0 END AS fcst_below_bin,
    CASE WHEN fs.tempmax_f > m.cap_strike   THEN 1 ELSE 0 END AS fcst_above_bin
FROM kalshi.markets m
JOIN wx.forecast_snapshot fs
  ON fs.city       = m.city
 AND fs.target_date = m.event_date
 AND fs.basis_date  = m.event_date  -- "basis = target" = midnight-of-day forecast
 AND fs.lead_days   = 0;
```

Now you can do queries like:

* “Show me all days where the midnight forecast high is **below the lowest bracket**”:

```sql
SELECT city, event_date, MIN(tempmax_midnight_fcst_f) AS fcst_high,
       MIN(floor_strike) AS lowest_floor
FROM feature.midnight_forecast_vs_brackets
GROUP BY city, event_date
HAVING MIN(tempmax_midnight_fcst_f) < MIN(floor_strike) - 1;  -- 1°F margin
```

* Or at the per-ticker level for your “auto bet” rule.

Later, for Option-1 backtesting, you can join this view with `wx.settlement` and `kalshi.candles_1m` to simulate:

> “At midnight, if `tempmax_midnight_fcst_f` is 10°F below all brackets, take size in the lowest bin at its mid price and hold to settlement.”

---

## 5. What to literally tell the agent

If you want a short, pasteable spec:

> 1. **No new DB.** Add this to the existing Postgres/Timescale instance:
>
>    * Table `wx.forecast_snapshot` (daily forecast history with basis_date, target_date, lead_days, key fields, raw_json).
> 2. **Extend VisualCrossingClient** with `fetch_historical_daily_forecast(vc_location, basis_date)` using the Timeline API and `forecastBasisDate` as described here:
>    [https://www.visualcrossing.com/resources/documentation/weather-data/how-to-query-weather-forecasts-from-the-past-historical-forecasts/](https://www.visualcrossing.com/resources/documentation/weather-data/how-to-query-weather-forecasts-from-the-past-historical-forecasts/)
> 3. **Implement `scripts/ingest_vc_forecast_history.py`**:
>
>    * Args: `--start-date` / `--end-date` or `--days`, `--city`/`--all-cities`, `--dry-run`.
>    * For each `(city, basis_date)`:
>
>      * Call `fetch_historical_daily_forecast(vc_location, basis_date)`.
>      * For each `day` in `payload["days"]`, insert/upsert a row into `wx.forecast_snapshot` with `(city, vc_location, basis_date, target_date, lead_days, tempmax_f, tempmin_f, ..., raw_json)`.
> 4. **Create a view `feature.midnight_forecast_vs_brackets`** joining:
>
>    * `kalshi.markets` (city, event_date, floor/cap strikes) and
>    * `wx.forecast_snapshot` where `basis_date = target_date` and `lead_days = 0` (midnight-of-day forecast),
>    * plus derived columns: `delta_to_floor`, `delta_to_cap`, `fcst_below_bin`, `fcst_above_bin`.
> 5. Ensure ingestion populates `raw_json` fully so we have the complete VC daily forecast payload stored for future expansion.

Once that’s in place, we can write very simple SQL (or Python) to backtest the “midnight forecast vs bracket range” heuristic and then hook it into your live engine.

And yes: your “if midnight forecast is far above/below all brackets, immediately bet” idea becomes trivial once this view exists—you’ll just be reading `feature.midnight_forecast_vs_brackets` at runtime instead of trying to recompute everything by hand.
