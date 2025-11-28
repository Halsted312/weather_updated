### Querying Visual Crossing Weather Data   
---

## 1. Objectives

We want the agent to:

1. **Lock onto a single airport station per city** using Visual Crossing’s station‑ID syntax (`stn:<stationid>`) so there is **no interpolation across multiple stations** for that feed. ([Visual Crossing][1])
2. **Also pull a “city aggregate” feed** per city (e.g., `"Chicago,IL"`), letting Visual Crossing’s normal multi‑station interpolation run with the default `maxDistance`/`maxStations` and options. ([Visual Crossing][1])
3. For both the station and city feeds, ingest:

   * **Historical actuals** (obs)
   * **Current conditions + near‑term forecast**
   * **Historical forecasts** via `forecastBasisDate` (basis date concept) ([Visual Crossing][2])
4. Use **sub‑hourly data** wherever possible:

   * `include=minutes` for minute‑level grids, with `options=minuteinterval_5` (5‑min) where allowed. ([Visual Crossing][3])
5. Capture **rich weather elements** (especially wind direction/speed at 10m and 50/80/100m, cloud cover, humidity, precip, radiation, CAPE/CIN, etc.). ([Visual Crossing][4])
6. Store this in a clean schema that explicitly tags:

   * **location_type** = `station` vs `city`
   * **data_type** = `actual_obs`, `current_snapshot`, `forecast`, `historical_forecast`
   * **forecast_basis_date** (for forecast data)

---

## 2. Canonical Locations (6 airports)

Define a small config for the six Kalshi cities, each with:

* `city_name`
* `vc_city_query` (string Visual Crossing expects for city aggregate)
* `station_id` (airport METAR code, used as VC station ID)
* `vc_station_query` = `f"stn:{station_id}"`

Recommended airports:

* **Chicago** – Midway: `KMDW`
* **Denver** – Denver Intl: `KDEN`
* **Austin** – Austin Bergstrom: `KAUS`
* **Los Angeles** – LAX: `KLAX`
* **Miami** – Miami Intl: `KMIA`
* **Philadelphia** – Philadelphia Intl: `KPHL`

Agent should implement this as a **single source‑of‑truth config** (YAML or Python dict) that everything (ingestion + features) reads from.

---

## 3. Visual Crossing API Patterns

### 3.1. General notes from docs

1. **Base endpoint**

```text
https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/
```

Requests look like:
`/timeline/[location]/[start]/[end]?key=...` ([Visual Crossing][5])

2. **Includes**

Use `include` to control sections: `days`, `hours`, `minutes`, `current`, `obs`, `fcst`, `stats`, etc. ([Visual Crossing][5])

* Minute‑level data requires `include=minutes` (in JSON it nests under each `hour`). ([Visual Crossing][3])

3. **Sub‑hourly interval**

* Default interval is 15 min.
* Change with `options=minuteinterval_5`, `minuteinterval_10`, `minuteinterval_30`, … ([Visual Crossing][3])

4. **Station selection**

* `maxDistance` – meters; default ~50 miles ≈ 80,467m. ([Visual Crossing][5])
* `maxStations` – default 3; VC interpolates across them. ([Visual Crossing][5])
* `elevationDifference` – max elevation offset allowed. ([Visual Crossing][5])

5. **Use a specific station**

* Set location to `stn:<stationid>`, e.g. `stn:KMDW`. ([Visual Crossing][1])

6. **Historical forecasts**

* Add `forecastBasisDate` (or `forecastBasisDay`) to the query.
* Example:
  `.../London,UK/2023-05-01/2023-05-15?include=days&forecastBasisDate=2023-05-01` ([Visual Crossing][2])

7. **Sub‑hourly availability**

* Historical: as fine as 5–10 minutes.
* Forecast: minimum 15 minutes, typically 12–24 hours into future. ([Visual Crossing][3])

8. **Energy / extended wind elements**

* `windspeed50`, `windspeed80`, `windspeed100`, `winddir50`, `winddir80`, `winddir100` (hourly + daily) ([Visual Crossing][6])

9. **Core elements** (temp, dew, humidity, wind, precip, cloudcover, etc.) are listed in the data dictionary. ([Visual Crossing][4])

---

### 3.2. Station vs City query templates

Below are the **canonical GET patterns** the agent should implement as Python helpers.

#### A. Station‑locked **historical actuals** (5‑min if possible)

Goal: A “truth‑like” series for each airport station, separate from NOAA but directly comparable.

Template:

```text
GET /timeline/stn:{STATION_ID}/{start}/{end}
    ?unitGroup=us
    &include=obs,minutes
    &options=minuteinterval_5,stnslevel1,useobs
    &maxDistance=1609        # 1 mile – essentially only that airport
    &maxStations=1
    &elevationDifference=50  # 50m or similar conservative cap
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

Key points:

* `stn:{STATION_ID}` forces VC to use that station. ([Visual Crossing][1])
* `include=obs,minutes` → hourly obs + nested minutes. ([Visual Crossing][3])
* `useobs` ensures we only use station observations, not remote-only grids. ([Visual Crossing][7])
* `stnslevel1` → “Include only level 1 stations” (from UI / your example query).
* `maxDistance=1609` (1 mile) + `maxStations=1` makes interpolation effectively impossible unless there’s another station colocated with the airport.

For **hourly‑only** backfills, simply drop `include=minutes` and the `minuteinterval` option.

#### B. City‑aggregate historical actuals

Goal: “What Visual Crossing thinks the weather is at the city centroid”, possibly blended across multiple stations.

Template:

```text
GET /timeline/{VC_CITY_QUERY}/{start}/{end}
    ?unitGroup=us
    &include=obs,minutes
    &options=minuteinterval_5,useobs
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

Notes:

* Do *not* set tiny `maxDistance` or `maxStations`; let defaults (up to 50 miles, up to 3 stations) drive interpolation. ([Visual Crossing][1])

#### C. Station‑locked **current + 7‑day forecast** (sub‑hour where available)

Template:

```text
GET /timeline/stn:{STATION_ID}/next7days
    ?unitGroup=us
    &include=fcst,current,minutes
    &options=minuteinterval_15,stnslevel1,usefcst
    &maxDistance=1609
    &maxStations=1
    &elevationDifference=50
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

* Use `minuteinterval_15` to respect the 15‑minute minimum for forecast minutes. ([Visual Crossing][3])
* `include=current` gives “right now” conditions. ([Visual Crossing][5])

City‑aggregate forecast is the same call but with the city query instead of `stn:` and without the tight `maxDistance`/`maxStations`.

#### D. Station‑locked **historical forecasts** (daily + hourly, optionally minutes)

We need: “What did VC’s model say for **target date T** when the forecast was created on **basis date B**?”

Template:

```text
GET /timeline/stn:{STATION_ID}/{target_start}/{target_end}
    ?unitGroup=us
    &include=days,hours
    &forecastBasisDate={basis_date}
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

* `forecastBasisDate` is the main knob for historical forecasts. ([Visual Crossing][2])
* For short horizons where minute forecasts are available, we *can* experiment with `include=hours,minutes&options=minuteinterval_15,usefcst`, but the agent should gate that behind a feature flag because support may be more limited.

---

### 3.3. Elements to request

Start with a **rich, but not insane** element set that covers:

**Core daily/hourly/minute:**

* `datetime`, `temp`, `tempmax`, `tempmin`
* `dew`, `humidity`
* `feelslike`, `feelslikemax`, `feelslikemin`
* `precip`, `precipprob`, `preciptype`, `precipcover`
* `snow`, `snowdepth`
* `windspeed`, `windgust`, `winddir`
* `cloudcover`, `visibility`, `pressure`
* `uvindex`, `solarradiation`, `solarenergy`
* `conditions`, `icon`
* `stations` (to audit what VC actually used) ([Visual Crossing][4])

**Extended wind / solar (energy elements):** ([Visual Crossing][6])

* `windspeed50`, `windspeed80`, `windspeed100`
* `winddir50`, `winddir80`, `winddir100`
* `dniradiation`, `difradiation`, `ghiradiation`, `gtiradiation`, `sunelevation`, `sunazimuth`

**Thermo / instability (add:) – as in your sample query:**

* `add:cape`, `add:cin`
* `add:deltat`
* `add:degreedays`, `add:accdegreedays`
* `add:elevation`, `add:latitude`, `add:longitude`
* `add:timezone`, `add:tzoffset`
* `add:windspeedmean`, `add:windspeedmin`, `add:windspeedmax`
* `add:precipremote`, `add:resolvedAddress`

Agent should build a small helper that assembles this `elements` string so we have **identical element lists** across all queries (station vs city, obs vs forecast).

---

## 4. Proposed DB Schema Extension

The agent doesn’t need to change NOAA / Kalshi tables. Instead, add **Visual Crossing‑specific tables** that are:

* Narrowly scoped to VC,
* Explicit about location_type and data_type,
* Normalized enough to stay sane.

### 4.1. Locations table

`wx.vc_location` (dimension)

* `id` (PK)
* `city_code` (e.g. `CHI`, `DEN`, `AUS`, `LAX`, `MIA`, `PHL`)
* `location_type` (`station` | `city`)
* `vc_location_query` (e.g. `stn:KMDW` or `Chicago,IL`)
* `station_id` (nullable, airport code when `location_type='station'`)
* `latitude`, `longitude`, `elevation_m` (optional; can pull from VC’s `add:latitude` etc.)

### 4.2. Minute‑level table

`wx.vc_minute_weather`

* `id` (PK)
* `vc_location_id` (FK → `wx.vc_location`)
* `data_type` (`actual_obs` | `current_snapshot` | `forecast` | `historical_forecast`)
* `forecast_basis_date` (DATE, nullable – set for `forecast` / `historical_forecast`)
* `target_datetime_local` (TIMESTAMP, local wall time per VC docs) ([Visual Crossing][5])
* `target_datetime_utc` (TIMESTAMP, derived via VC timezone)
* **Core elements**: temp, dew, humidity, precip, snow, windspeed, winddir, windgust, cloudcover, visibility, pressure, etc.
* **Extended wind**: windspeed50/80/100, winddir50/80/100
* **Radiation / energy**: solarradiation, solarenergy, dniradiation, difradiation, ghiradiation, gtiradiation, sunelevation, sunazimuth
* **Flags**: preciptype, conditions, icon
* `raw_json` (JSONB, optional for debugging; consider pruning in prod)

Indexes:

* (`vc_location_id`, `target_datetime_utc`)
* (`vc_location_id`, `data_type`, `forecast_basis_date`, `target_datetime_utc`)

### 4.3. Hourly + Daily forecast tables (optional but nice)

You may already have `wx.forecast_snapshot` and `wx.forecast_snapshot_hourly`. The agent should:

* **Add columns** there (or create parallel `wx.vc_forecast_snapshot[_hourly]`) to store the same enriched elements as above.
* Add `location_type` / `vc_location_id` and `source = 'visual_crossing'` if not already present.
* Add `forecast_basis_date` and `data_type` to clearly distinguish:

  * `data_type='forecast'` (current best forecast)
  * `data_type='historical_forecast'` (basis date in the past)

---

## 5. Ingestion Jobs the Agent Should Build

The agent should write separate, idempotent pipelines (scripts or jobs) using the VC Timeline API.

### 5.1. Backfill station & city historical actuals

For each city:

1. For each `vc_location` of type `station` and `city`:

   * Loop over date ranges (e.g., 7–30 day chunks) from desired start date to “yesterday”.
   * Call the **historical actuals** template (section 3.2.A/B).
   * Parse both `hours` and nested `minutes` arrays. ([Visual Crossing][3])
2. Upsert into `wx.vc_minute_weather` with:

   * `data_type='actual_obs'`
   * `forecast_basis_date=NULL`
3. Handle rate limiting / query cost (sub‑hourly is expensive; VC counts each minute row as a record). ([Visual Crossing][3])

### 5.2. Nightly current + near‑term forecast snapshots

Once or several times per day:

1. For each `vc_location` (station + city):

   * Call **current + forecast** template (3.2.C).
   * Store:

     * Minute‑level forecasts for the next 12–24h in `wx.vc_minute_weather` with `data_type='forecast'` and `forecast_basis_date = today()`.
     * Daily and hourly data in existing forecast tables (enriched columns).
2. Use a simple “latest snapshot per (location, basis_date)” view that strategies can query.

### 5.3. Historical forecasts backfill

For each `target_date` in your Kalshi trading history and each location:

* For selected lead times (e.g., 0, 1, 2, 3, 5, 7 days before target), compute `basis_date = target_date - lead`.
* Call **historical forecast** template (3.2.D) for station & city.
* Store daily/hourly forecasts with:

  * `data_type='historical_forecast'`
  * `forecast_basis_date = basis_date`
  * Derived `lead_days = target_date - basis_date`.

Minutes can be added later as a second pass.

---

## 6. How Strategies Will Use This

The agent doesn’t need to touch trading code now, but the schema should anticipate:

* Ability to join **NOAA settlement TMAX** and Kalshi brackets to:

  * Station obs (VC vs NOAA)
  * City obs
  * Forecast & historical forecasts at specific lead times
  * Minute‑level wind direction / speeds around the high‑temperature time

A simple view per city/day like:

```sql
SELECT *
FROM wx.vc_minute_weather
WHERE vc_location_id IN (station_for_city, city_for_city)
  AND data_type IN ('actual_obs', 'forecast', 'historical_forecast')
  AND target_datetime_utc BETWEEN event_date_start_utc - INTERVAL '1 day'
                              AND event_date_end_utc + INTERVAL '1 day';
```

…gives your ML stack rich features around each Kalshi event.

---

## 7. Final Copy‑Pasteable Agent Prompt

Here’s the actual prompt you can give to a coding agent:

---

### AGENT PROMPT (for another model)

You are a senior Python + SQL data engineer working on a Kalshi weather trading system.

Your job: **upgrade our Visual Crossing ingestion and schema** to provide much richer, higher‑frequency data for 6 US cities, at both **single airport station** and **city‑aggregate** levels.

#### Context

* We use **Visual Crossing Timeline API** for:

  * Historical actuals (`include=obs`),
  * Current conditions + 15‑day forecast,
  * Historical forecasts via `forecastBasisDate`. ([Visual Crossing][5])
* We trade Kalshi temperature markets for **six cities**, and we want:

  * A clean **airport‑station feed** per city (no interpolation), and
  * A **city aggregate feed** per city (Visual Crossing’s multi‑station interpolation).
* Repo: `Halsted312/weather_updated` (treat as canonical).

  * Weather models currently include `wx.settlement`, `wx.minute_obs`, `wx.forecast_snapshot`, `wx.forecast_snapshot_hourly`.
  * Do **not** modify Kalshi tables or NOAA ingestion; only extend Visual Crossing side.

#### Target Locations

Define a central config mapping:

* `city_code`: `CHI`, `DEN`, `AUS`, `LAX`, `MIA`, `PHL`
* `vc_city_query`: `"Chicago,IL"`, `"Denver,CO"`, `"Austin,TX"`, `"Los Angeles,CA"`, `"Miami,FL"`, `"Philadelphia,PA"`
* `station_id`: `KMDW`, `KDEN`, `KAUS`, `KLAX`, `KMIA`, `KPHL`
* `vc_station_query`: `f"stn:{station_id}"`

We must always use the **station query** for the single‑station feed, per Visual Crossing docs on requesting data for a specific station: `stn:<stationid>`. ([Visual Crossing][1])

#### Visual Crossing request patterns to implement

Create a small client (or extend `src/weather/visual_crossing.py`) exposing functions with signatures like:

* `fetch_station_history_minutes(station_id, start_date, end_date, elements)`
* `fetch_city_history_minutes(city_query, start_date, end_date, elements)`
* `fetch_station_current_and_fcst(station_id, horizon_days, elements)`
* `fetch_city_current_and_fcst(city_query, horizon_days, elements)`
* `fetch_station_historical_forecast(station_id, target_start, target_end, basis_date, elements)`
* (optional) city version of historical forecast

**Use these canonical URL patterns:**

1. **Station historical actuals (5‑min, obs only)**

```http
GET /timeline/stn:{STATION_ID}/{start}/{end}
    ?unitGroup=us
    &include=obs,minutes
    &options=minuteinterval_5,stnslevel1,useobs
    &maxDistance=1609
    &maxStations=1
    &elevationDifference=50
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

2. **City historical actuals (5‑min, interpolated)**

```http
GET /timeline/{VC_CITY_QUERY}/{start}/{end}
    ?unitGroup=us
    &include=obs,minutes
    &options=minuteinterval_5,useobs
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

3. **Station current + next 7 days (15‑min forecast + current)**

```http
GET /timeline/stn:{STATION_ID}/next7days
    ?unitGroup=us
    &include=fcst,current,minutes
    &options=minuteinterval_15,stnslevel1,usefcst
    &maxDistance=1609
    &maxStations=1
    &elevationDifference=50
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

4. **City current + next 7 days**

Same as above but with `{VC_CITY_QUERY}` and without tight `maxDistance`/`maxStations`.

5. **Station historical forecasts (daily+hourly)**

```http
GET /timeline/stn:{STATION_ID}/{target_start}/{target_end}
    ?unitGroup=us
    &include=days,hours
    &forecastBasisDate={basis_date}
    &elements={ELEMENTS}
    &key=API_KEY
    &contentType=json
```

Optionally, add `minutes` + `minuteinterval_15` for short‑horizon sub‑hourly historical forecasts where available.

**Elements set**

Implement a helper that returns a comma‑separated `elements` string with at least:

* Core: `datetime,temp,tempmax,tempmin,dew,humidity,feelslike,feelslikemax,feelslikemin,precip,precipprob,preciptype,precipcover,snow,snowdepth,windspeed,windgust,winddir,visibility,cloudcover,pressure,uvindex,solarradiation,solarenergy,conditions,icon,stations`
* Extended wind: `windspeed50,winddir50,windspeed80,winddir80,windspeed100,winddir100`
* Solar: `dniradiation,difradiation,ghiradiation,gtiradiation,sunelevation,sunazimuth`
* “add:” fields: `add:cape,add:cin,add:deltat,add:degreedays,add:accdegreedays,add:elevation,add:latitude,add:longitude,add:timezone,add:tzoffset,add:windspeedmean,add:windspeedmin,add:windspeedmax,add:precipremote,add:resolvedAddress`

Use `include=minutes` per timeline docs; minute data will be nested under each hour. ([Visual Crossing][3])

#### Schema work

Add Visual Crossing–specific tables instead of overloading NOAA tables:

1. **`wx.vc_location`**

* `id` (PK)
* `city_code`
* `location_type` (`'station'` | `'city'`)
* `vc_location_query`
* `station_id` (nullable)
* `latitude`, `longitude`, `elevation_m` (from VC `add:` fields when first seen)

2. **`wx.vc_minute_weather`**

* `id` (PK)
* `vc_location_id` (FK)
* `data_type` (`'actual_obs' | 'current_snapshot' | 'forecast' | 'historical_forecast'`)
* `forecast_basis_date` (nullable)
* `target_datetime_local`
* `target_datetime_utc`
* Core weather columns (temp, dew, humidity, precip, snow, windspeed, winddir, windgust, cloudcover, visibility, pressure, uvindex, etc.)
* Extended wind columns (windspeed50/80/100, winddir50/80/100)
* Radiation columns (solarradiation, solarenergy, dniradiation, difradiation, ghiradiation, gtiradiation, sunelevation, sunazimuth)
* Text flags (preciptype, conditions, icon)
* `raw_json` (JSONB) for debugging (can be nullable / optional)

Indexes:

* (`vc_location_id`, `target_datetime_utc`)
* (`vc_location_id`, `data_type`, `forecast_basis_date`, `target_datetime_utc`)

3. **Forecast summary tables**

Either:

* Extend existing `wx.forecast_snapshot` and `wx.forecast_snapshot_hourly` with:

  * `source` (enum, include `'visual_crossing'`)
  * `vc_location_id`
  * extended weather fields (wind, humidity, precip, etc.)
  * `forecast_basis_date`, `data_type`

Or:

* Add parallel `wx.vc_forecast_snapshot` and `wx.vc_forecast_snapshot_hourly` with that structure.

#### Pipelines to implement

1. **Historical actuals backfill (station + city)**

* For each city and both `location_type`s:

  * Loop from chosen start date → yesterday in 7–30 day chunks.
  * Use the station / city historical templates above.
  * Parse JSON, flatten hours → minutes, and upsert into `wx.vc_minute_weather` with `data_type='actual_obs'`.
* Handle VC query costs for sub‑hourly data; use chunk sizes that keep within limits.

2. **Nightly current + forecast snapshot (station + city)**

* Once per day (or more often), for each `vc_location`:

  * Call current+forecast template.
  * Insert:

    * Minute grid: `wx.vc_minute_weather` with `data_type='forecast'` and `forecast_basis_date = today`.
    * Daily/hourly: forecast snapshot tables.
* Provide a view or helper returning “latest snapshot for each (vc_location, basis_date)” for strategy code.

3. **Historical forecast backfill (station, optionally city)**

* For each event date in our Kalshi dataset and each city:

  * For selected lead times (0, 1, 2, 3, 5, 7 days, etc.), compute `basis_date`.
  * Query VC historical forecast for that station using `forecastBasisDate`.
  * Store daily/hourly curves in forecast snapshot tables with `data_type='historical_forecast'`, `forecast_basis_date=basis_date`, derived `lead_days`.
* (Minutes can be added later as a second backfill step.)

#### Coding guidelines

* Use pure `requests` or a light wrapper; you may look at Visual Crossing’s Python examples for reference but don’t pull in heavy external deps. ([Visual Crossing][8])
* Make the ingestion functions **idempotent**: use UPSERTs keyed on `(vc_location_id, data_type, forecast_basis_date, target_datetime_utc)`.
* Centralize:

  * VC API key loading,
  * Retry / backoff,
  * Logging of query cost (from VC response) for monitoring. ([Visual Crossing][3])
* Don’t touch strategy code yet; just make sure what you build will allow future joins of:

  * NOAA settlement (`wx.settlement`)
  * Kalshi markets & candles
  * Visual Crossing station vs city data at minute/hour/day resolution.

Deliverables:

1. New/updated DB migrations for `wx.vc_location`, `wx.vc_minute_weather`, and VC forecast tables.
2. A robust Visual Crossing client module implementing the query patterns above.
3. Three ingestion scripts (or scheduled jobs) for:

   * Historical actuals backfill,
   * Nightly current+forecast snapshots,
   * Historical forecast backfill.
4. Basic unit/integration tests that:

   * Call the client for a short date range for `stn:KMDW` and `"Chicago,IL"`,
   * Assert correct parsing into `wx.vc_minute_weather`,
   * Verify that station vs city results differ when VC is obviously interpolating.

---



[1]: https://www.visualcrossing.com/resources/documentation/weather-data/how-do-i-find-my-nearest-weather-station/ "How do I find my nearest weather station? | Visual Crossing"
[2]: https://www.visualcrossing.com/resources/documentation/weather-data/how-to-query-weather-forecasts-from-the-past-historical-forecasts/ "How to query weather forecasts from the past - Historical Forecasts | Visual Crossing"
[3]: https://www.visualcrossing.com/resources/documentation/weather-api/sub-hourly-data-in-the-timeline-weather-api-2/ "Requesting sub-hourly data in the Timeline Weather API | Visual Crossing"
[4]: https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/ "Weather Data Documentation | Visual Crossing"
[5]: https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/ "Weather API Documentation | Visual Crossing"
[6]: https://www.visualcrossing.com/resources/documentation/weather-api/energy-elements-in-the-timeline-weather-api/ "Energy elements in the Timeline Weather API | Visual Crossing"
[7]: https://www.visualcrossing.com/resources/documentation/weather-api/using-remote-data-sources-in-the-weather-api/ "Using remote data sources in the Weather API | Visual Crossing"
[8]: https://www.visualcrossing.com/resources/blog/how-to-load-historical-weather-data-using-python-without-scraping/?utm_source=chatgpt.com "How to load weather data using Python"
