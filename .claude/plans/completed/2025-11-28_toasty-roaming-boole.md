# Plan: Visual Crossing Data Enhancement - Full Implementation

---
plan_id: toasty-roaming-boole
created: 2025-11-27
completed: 2025-11-28
status: completed
priority: high
agent: kalshi-weather-quant
---

## Completion Summary

**Phase 1 completed on 2025-11-28:**
- ✅ Migration 007 applied - all 4 new tables created (`wx.vc_location`, `wx.vc_minute_weather`, `wx.vc_forecast_daily`, `wx.vc_forecast_hourly`)
- ✅ Partial unique indexes for idempotent upserts
- ✅ 12 seed locations (6 cities × 2 types)
- ✅ ORM models in `src/db/models.py`
- ✅ Elements config in `src/config/vc_elements.py`
- ✅ Visual Crossing client extended with new methods
- ✅ Ingestion scripts created and tested:
  - `scripts/ingest_vc_obs_backfill.py`
  - `scripts/ingest_vc_forecast_snapshot.py`
  - `scripts/ingest_vc_historical_forecast.py`
- ✅ Legacy scripts archived to `legacy/` folder
- ✅ Documentation updated

---

## User Request
- Greenfield Visual Crossing schema redesign
- 5-min observations, 15-min forecast minutes
- Store ALL data possible from Visual Crossing
- Both station-locked AND city-aggregate feeds
- Proper datetime/timezone handling from day one
- No migration needed - fresh start

---

## Verified Station & City Mappings

### Kalshi Weather Markets - Official NWS Stations
Sources: [Reddit Guide](https://www.reddit.com/r/Kalshi/comments/1hfvnmj/an_incomplete_and_unofficial_guide_to_temperature/), [Wethr City Resources](https://wethr.net/edu/city-resources), [NWS](https://www.weather.gov/phi/PHLdashboard)

| City | Station ID | Kalshi Ticker | Kalshi Code | IANA Timezone |
|------|------------|---------------|-------------|---------------|
| Chicago | KMDW (Midway) | kxhighchi | CHI | America/Chicago |
| Denver | KDEN | kxhighden | DEN | America/Denver |
| Austin | KAUS (Bergstrom) | kxhighaus | AUS | America/Chicago |
| Los Angeles | KLAX | kxhighla | LAX | America/Los_Angeles |
| Miami | KMIA | kxhighmia | MIA | America/New_York |
| Philadelphia | KPHL | kxhighphil | PHIL | America/New_York |

**Important:** Philadelphia's station ID is `KPHL` but Kalshi's market code is `PHIL` (with the extra "I"). The `kalshi_code` column tracks Kalshi's ticker format separately from the NWS station ID.

---

## Implementation Overview

### New Schema Design

#### 1. `wx.vc_location` (Dimension Table)
```sql
CREATE TABLE wx.vc_location (
    id SERIAL PRIMARY KEY,
    city_code TEXT NOT NULL,              -- 'CHI', 'DEN', 'AUS', 'LAX', 'MIA', 'PHL'
    kalshi_code TEXT NOT NULL,            -- 'CHI', 'DEN', 'AUS', 'LAX', 'MIA', 'PHIL' (matches Kalshi tickers)
    location_type TEXT NOT NULL,          -- 'station' | 'city'
    vc_location_query TEXT NOT NULL,      -- 'stn:KMDW' or 'Chicago,IL'
    station_id TEXT,                      -- 'KMDW' (nullable for city type)
    iana_timezone TEXT NOT NULL,          -- 'America/Chicago'
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    elevation_m DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (city_code, location_type),
    UNIQUE (vc_location_query),
    CONSTRAINT vc_location_type_chk CHECK (location_type IN ('station', 'city'))
);
```

#### 2. `wx.vc_minute_weather` (Minute-Level Fact Table)

**Note:** Degreeday fields (`degreedays`, `accdegreedays`) are daily constructs - at minute resolution they'll be constant per day (not native per-minute signals).

```sql
CREATE TABLE wx.vc_minute_weather (
    id BIGSERIAL PRIMARY KEY,
    vc_location_id INTEGER NOT NULL REFERENCES wx.vc_location(id),

    -- Classification
    data_type TEXT NOT NULL,              -- 'actual_obs' | 'current_snapshot' | 'forecast' | 'historical_forecast'
    forecast_basis_date DATE,             -- NULL for actual_obs
    forecast_basis_datetime_utc TIMESTAMPTZ,
    lead_hours INTEGER,                   -- computed: (target - basis) / 3600

    -- Time Fields (from VC datetime/datetimeEpoch/timezone/tzoffset)
    datetime_epoch_utc BIGINT NOT NULL,
    datetime_utc TIMESTAMPTZ NOT NULL,
    datetime_local TIMESTAMP NOT NULL,
    timezone TEXT NOT NULL,               -- 'America/Chicago'
    tzoffset_minutes SMALLINT NOT NULL,   -- e.g., -360 for CST

    -- Core Weather
    temp_f FLOAT,
    tempmax_f FLOAT,
    tempmin_f FLOAT,
    feelslike_f FLOAT,
    feelslikemax_f FLOAT,
    feelslikemin_f FLOAT,
    dew_f FLOAT,
    humidity FLOAT,

    -- Precipitation
    precip_in FLOAT,
    precipprob FLOAT,
    preciptype TEXT,
    precipcover FLOAT,
    snow_in FLOAT,
    snowdepth_in FLOAT,
    precipremote FLOAT,

    -- Wind (10m standard)
    windspeed_mph FLOAT,
    windgust_mph FLOAT,
    winddir FLOAT,
    windspeedmean_mph FLOAT,
    windspeedmin_mph FLOAT,
    windspeedmax_mph FLOAT,

    -- Extended Wind (50/80/100m)
    windspeed50_mph FLOAT,
    winddir50 FLOAT,
    windspeed80_mph FLOAT,
    winddir80 FLOAT,
    windspeed100_mph FLOAT,
    winddir100 FLOAT,

    -- Atmosphere
    cloudcover FLOAT,
    visibility_miles FLOAT,
    pressure_mb FLOAT,

    -- Solar/Radiation
    uvindex FLOAT,
    solarradiation FLOAT,
    solarenergy FLOAT,
    dniradiation FLOAT,
    difradiation FLOAT,
    ghiradiation FLOAT,
    gtiradiation FLOAT,
    sunelevation FLOAT,
    sunazimuth FLOAT,

    -- Instability/Energy
    cape FLOAT,
    cin FLOAT,
    deltat FLOAT,
    degreedays FLOAT,
    accdegreedays FLOAT,

    -- Text/Flags
    conditions TEXT,
    icon TEXT,
    stations TEXT,
    resolved_address TEXT,

    -- Metadata
    source_system TEXT DEFAULT 'vc_timeline',
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT vc_minute_data_type_chk CHECK (data_type IN ('actual_obs', 'current_snapshot', 'forecast', 'historical_forecast')),
    CONSTRAINT vc_minute_unique_row UNIQUE (vc_location_id, data_type, forecast_basis_date, datetime_utc)
);

-- Indexes
CREATE INDEX idx_vc_minute_location_time ON wx.vc_minute_weather(vc_location_id, datetime_utc);
CREATE INDEX idx_vc_minute_datatype_basis ON wx.vc_minute_weather(vc_location_id, data_type, forecast_basis_date, datetime_utc);
```

#### 3. `wx.vc_forecast_daily` (Daily Forecast Snapshots)
```sql
CREATE TABLE wx.vc_forecast_daily (
    id BIGSERIAL PRIMARY KEY,
    vc_location_id INTEGER NOT NULL REFERENCES wx.vc_location(id),

    -- Classification
    data_type TEXT NOT NULL,              -- 'forecast' | 'historical_forecast'
    forecast_basis_date DATE NOT NULL,
    forecast_basis_datetime_utc TIMESTAMPTZ,

    -- Target
    target_date DATE NOT NULL,
    lead_days INTEGER NOT NULL,           -- target_date - forecast_basis_date

    -- Daily Weather Fields (same extended set)
    tempmax_f FLOAT,
    tempmin_f FLOAT,
    temp_f FLOAT,
    feelslikemax_f FLOAT,
    feelslikemin_f FLOAT,
    feelslike_f FLOAT,
    dew_f FLOAT,
    humidity FLOAT,
    precip_in FLOAT,
    precipprob FLOAT,
    preciptype TEXT,
    precipcover FLOAT,
    snow_in FLOAT,
    snowdepth_in FLOAT,
    windspeed_mph FLOAT,
    windgust_mph FLOAT,
    winddir FLOAT,
    windspeedmean_mph FLOAT,
    windspeedmin_mph FLOAT,
    windspeedmax_mph FLOAT,
    windspeed50_mph FLOAT,
    winddir50 FLOAT,
    windspeed80_mph FLOAT,
    winddir80 FLOAT,
    windspeed100_mph FLOAT,
    winddir100 FLOAT,
    cloudcover FLOAT,
    visibility_miles FLOAT,
    pressure_mb FLOAT,
    uvindex FLOAT,
    solarradiation FLOAT,
    solarenergy FLOAT,
    dniradiation FLOAT,
    difradiation FLOAT,
    ghiradiation FLOAT,
    gtiradiation FLOAT,
    cape FLOAT,
    cin FLOAT,
    deltat FLOAT,
    degreedays FLOAT,
    accdegreedays FLOAT,
    conditions TEXT,
    icon TEXT,

    -- Metadata
    source_system TEXT DEFAULT 'vc_timeline',
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT vc_daily_data_type_chk CHECK (data_type IN ('forecast', 'historical_forecast')),
    CONSTRAINT vc_daily_unique_row UNIQUE (vc_location_id, target_date, forecast_basis_date, data_type)
);

-- Indexes for backtest queries ("give me T-1 day forecast for this event_date")
CREATE INDEX idx_vc_daily_location_target_basis ON wx.vc_forecast_daily(vc_location_id, target_date, forecast_basis_date);
```

#### 4. `wx.vc_forecast_hourly` (Hourly Forecast Snapshots)
```sql
CREATE TABLE wx.vc_forecast_hourly (
    id BIGSERIAL PRIMARY KEY,
    vc_location_id INTEGER NOT NULL REFERENCES wx.vc_location(id),

    -- Classification
    data_type TEXT NOT NULL,
    forecast_basis_date DATE NOT NULL,
    forecast_basis_datetime_utc TIMESTAMPTZ,

    -- Target Time
    target_datetime_epoch_utc BIGINT NOT NULL,
    target_datetime_utc TIMESTAMPTZ NOT NULL,
    target_datetime_local TIMESTAMP NOT NULL,
    timezone TEXT NOT NULL,
    tzoffset_minutes SMALLINT NOT NULL,
    lead_hours INTEGER NOT NULL,

    -- Hourly Weather Fields (same extended set as minute table)
    temp_f FLOAT,
    feelslike_f FLOAT,
    dew_f FLOAT,
    humidity FLOAT,
    precip_in FLOAT,
    precipprob FLOAT,
    preciptype TEXT,
    snow_in FLOAT,
    windspeed_mph FLOAT,
    windgust_mph FLOAT,
    winddir FLOAT,
    windspeed50_mph FLOAT,
    winddir50 FLOAT,
    windspeed80_mph FLOAT,
    winddir80 FLOAT,
    windspeed100_mph FLOAT,
    winddir100 FLOAT,
    cloudcover FLOAT,
    visibility_miles FLOAT,
    pressure_mb FLOAT,
    uvindex FLOAT,
    solarradiation FLOAT,
    solarenergy FLOAT,
    dniradiation FLOAT,
    difradiation FLOAT,
    ghiradiation FLOAT,
    gtiradiation FLOAT,
    sunelevation FLOAT,
    sunazimuth FLOAT,
    cape FLOAT,
    cin FLOAT,
    conditions TEXT,
    icon TEXT,

    -- Metadata
    source_system TEXT DEFAULT 'vc_timeline',
    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT vc_hourly_data_type_chk CHECK (data_type IN ('forecast', 'historical_forecast')),
    CONSTRAINT vc_hourly_unique_row UNIQUE (vc_location_id, target_datetime_utc, forecast_basis_date, data_type)
);
```

---

## Files to Create/Modify

### Phase 1: Schema & Models

| File | Action | Description |
|------|--------|-------------|
| `src/db/models.py` | MODIFY | Add VcLocation, VcMinuteWeather, VcForecastDaily, VcForecastHourly models |
| `src/db/migrations/` | CREATE | Alembic migration for new tables |

### Phase 2: Configuration

| File | Action | Description |
|------|--------|-------------|
| `src/config/cities.py` | MODIFY | Add `vc_city_query` field to CityConfig |
| `src/config/vc_elements.py` | CREATE | Centralized elements list builder |

### Phase 3: Visual Crossing Client

| File | Action | Description |
|------|--------|-------------|
| `src/weather/visual_crossing.py` | MAJOR REWRITE | Add station vs city query methods, extended elements, proper API params |

**New client methods:**
- `fetch_station_history_minutes(station_id, start, end)` - 5-min obs, maxDistance=1609, maxStations=1
- `fetch_city_history_minutes(city_query, start, end)` - 5-min obs, default interpolation
- `fetch_station_current_and_forecast(station_id, horizon_days)` - 15-min forecast
- `fetch_city_current_and_forecast(city_query, horizon_days)` - 15-min forecast
- `fetch_station_historical_forecast(station_id, target_start, target_end, basis_date)` - days+hours
- `fetch_city_historical_forecast(city_query, target_start, target_end, basis_date)`
- `build_elements_string()` - returns full 47-field elements list

### Phase 4: Ingestion Scripts

| File | Action | Description |
|------|--------|-------------|
| `scripts/ingest_vc_obs_backfill.py` | CREATE | Backfill station + city historical observations |
| `scripts/ingest_vc_forecast_snapshot.py` | CREATE | Nightly current + forecast snapshots |
| `scripts/ingest_vc_historical_forecast.py` | CREATE | Historical forecast backfill |
| `scripts/poll_vc_daemon.py` | CREATE | Combined 24/7 daemon for all VC data |

---

## API Query Patterns

### Station Historical Observations (5-min)
```
GET /timeline/stn:{STATION_ID}/{start}/{end}
    ?unitGroup=us
    &include=obs,minutes
    &options=minuteinterval_5,stnslevel1,useobs
    &maxDistance=1609
    &maxStations=1
    &elevationDifference=50
    &elements={FULL_ELEMENTS}
    &key=API_KEY
    &contentType=json
```

### City Historical Observations (5-min, interpolated)
```
GET /timeline/{CITY_QUERY}/{start}/{end}
    ?unitGroup=us
    &include=obs,minutes
    &options=minuteinterval_5,useobs
    &elements={FULL_ELEMENTS}
    &key=API_KEY
    &contentType=json
```

### Station Current + Forecast (15-min)
```
GET /timeline/stn:{STATION_ID}/next7days
    ?unitGroup=us
    &include=fcst,current,minutes
    &options=minuteinterval_15,stnslevel1,usefcst
    &maxDistance=1609
    &maxStations=1
    &elevationDifference=50
    &elements={FULL_ELEMENTS}
    &key=API_KEY
    &contentType=json
```

### Historical Forecast (via forecastBasisDate)
```
GET /timeline/stn:{STATION_ID}/{target_start}/{target_end}
    ?unitGroup=us
    &include=days,hours
    &forecastBasisDate={basis_date}
    &elements={FULL_ELEMENTS}
    &key=API_KEY
    &contentType=json
```

---

## Elements String (Full 47 Fields)

```python
CORE_ELEMENTS = [
    "datetime", "datetimeEpoch",
    "temp", "tempmax", "tempmin",
    "feelslike", "feelslikemax", "feelslikemin",
    "dew", "humidity",
    "precip", "precipprob", "preciptype", "precipcover",
    "snow", "snowdepth",
    "windspeed", "windgust", "winddir",
    "cloudcover", "visibility", "pressure",
    "uvindex", "solarradiation", "solarenergy",
    "conditions", "icon", "stations",
]

EXTENDED_WIND = [
    "windspeed50", "winddir50",
    "windspeed80", "winddir80",
    "windspeed100", "winddir100",
]

EXTENDED_SOLAR = [
    "dniradiation", "difradiation",
    "ghiradiation", "gtiradiation",
    "sunelevation", "sunazimuth",
]

ADD_FIELDS = [
    "add:cape", "add:cin", "add:deltat",
    "add:degreedays", "add:accdegreedays",
    "add:elevation", "add:latitude", "add:longitude",
    "add:timezone", "add:tzoffset",
    "add:windspeedmean", "add:windspeedmin", "add:windspeedmax",
    "add:precipremote", "add:resolvedAddress",
]
```

---

## Location Configuration

### Seed Data for `wx.vc_location`

```sql
INSERT INTO wx.vc_location (city_code, kalshi_code, location_type, vc_location_query, station_id, iana_timezone)
VALUES
  ('CHI', 'CHI',  'station', 'stn:KMDW',         'KMDW', 'America/Chicago'),
  ('CHI', 'CHI',  'city',    'Chicago,IL',       NULL,   'America/Chicago'),
  ('DEN', 'DEN',  'station', 'stn:KDEN',         'KDEN', 'America/Denver'),
  ('DEN', 'DEN',  'city',    'Denver,CO',        NULL,   'America/Denver'),
  ('AUS', 'AUS',  'station', 'stn:KAUS',         'KAUS', 'America/Chicago'),
  ('AUS', 'AUS',  'city',    'Austin,TX',        NULL,   'America/Chicago'),
  ('LAX', 'LAX',  'station', 'stn:KLAX',         'KLAX', 'America/Los_Angeles'),
  ('LAX', 'LAX',  'city',    'Los Angeles,CA',   NULL,   'America/Los_Angeles'),
  ('MIA', 'MIA',  'station', 'stn:KMIA',         'KMIA', 'America/New_York'),
  ('MIA', 'MIA',  'city',    'Miami,FL',         NULL,   'America/New_York'),
  ('PHL', 'PHIL', 'station', 'stn:KPHL',         'KPHL', 'America/New_York'),
  ('PHL', 'PHIL', 'city',    'Philadelphia,PA',  NULL,   'America/New_York');
```

**Note:** Philadelphia uses `kalshi_code='PHIL'` to match Kalshi's `kxhighphil` ticker, while `city_code='PHL'` matches the standard abbreviation.

---

## Step-by-Step Implementation Guide (Fresh Database)

### Step 0: Schema Setup

1. **Ensure `wx` schema exists:**
   ```sql
   CREATE SCHEMA IF NOT EXISTS wx;
   ```

2. **Create Alembic migration** with all 4 tables + constraints + indexes:
   - `wx.vc_location`
   - `wx.vc_minute_weather`
   - `wx.vc_forecast_daily`
   - `wx.vc_forecast_hourly`

3. **Define ORM models in `src/db/models.py`:**
   - `VcLocation`
   - `VcMinuteWeather`
   - `VcForecastDaily`
   - `VcForecastHourly`
   - Use `sqlalchemy.CheckConstraint` or Python Enum for `location_type` and `data_type`

### Step 1: Seed Location Dimension

Run the INSERT statement from "Seed Data" section above to populate `wx.vc_location` with 12 rows (6 cities × 2 types).

### Step 2: Create Elements Config Module

Create `src/config/vc_elements.py`:

```python
"""Visual Crossing API elements configuration.

Field names verified against:
- https://www.visualcrossing.com/resources/documentation/weather-data/weather-data-documentation/
- https://www.visualcrossing.com/resources/documentation/weather-api/energy-elements-in-the-timeline-weather-api/
"""

CORE_ELEMENTS = [
    "datetime", "datetimeEpoch",
    "temp", "tempmax", "tempmin",
    "feelslike", "feelslikemax", "feelslikemin",
    "dew", "humidity",
    "precip", "precipprob", "preciptype", "precipcover",
    "snow", "snowdepth",
    "windspeed", "windgust", "winddir",
    "cloudcover", "visibility", "pressure",
    "uvindex", "solarradiation", "solarenergy",
    "conditions", "icon", "stations",
]

EXTENDED_WIND = [
    "windspeed50", "winddir50",
    "windspeed80", "winddir80",
    "windspeed100", "winddir100",
]

EXTENDED_SOLAR = [
    "dniradiation", "difradiation",
    "ghiradiation", "gtiradiation",
    "sunelevation", "sunazimuth",
]

ADD_FIELDS = [
    "add:cape", "add:cin", "add:deltat",
    "add:degreedays", "add:accdegreedays",
    "add:elevation", "add:latitude", "add:longitude",
    "add:timezone", "add:tzoffset",
    "add:windspeedmean", "add:windspeedmin", "add:windspeedmax",
    "add:precipremote", "add:resolvedAddress",
]

def build_elements_string() -> str:
    """Build comma-separated elements string for VC API calls."""
    return ",".join(CORE_ELEMENTS + EXTENDED_WIND + EXTENDED_SOLAR + ADD_FIELDS)
```

### Step 3: Rewrite Visual Crossing Client

Rewrite `src/weather/visual_crossing.py` with clear separation:

**Station-locked calls** (no interpolation):
- `location=f"stn:{station_id}"`
- `maxDistance=1609` (1 mile)
- `maxStations=1`
- `elevationDifference=50`
- `options` includes `stnslevel1` and either `useobs` or `usefcst`

**City-aggregate calls** (let VC interpolate):
- `location=city_query` (e.g., "Chicago,IL")
- Do NOT set `maxDistance` or `maxStations`
- Let VC use default multi-station interpolation

**Methods to implement:**
```python
# Historical observations (5-min)
def fetch_station_history_minutes(station_id: str, start: date, end: date) -> dict
def fetch_city_history_minutes(city_query: str, start: date, end: date) -> dict

# Current + forecast (15-min for forecast minutes)
def fetch_station_current_and_forecast(station_id: str, horizon_days: int = 7) -> dict
def fetch_city_current_and_forecast(city_query: str, horizon_days: int = 7) -> dict

# Historical forecasts (days + hours)
def fetch_station_historical_forecast(station_id: str, target_start: date, target_end: date, basis_date: date) -> dict
def fetch_city_historical_forecast(city_query: str, target_start: date, target_end: date, basis_date: date) -> dict
```

**Parsing requirements:**
- For each minute/hour/day object, extract BOTH:
  - `datetimeEpoch`, `timezone`, `tzoffset` → compute `datetime_utc`, `datetime_local`, `tzoffset_minutes`
  - All weather fields into appropriate model

### Step 4: Ingestion Scripts

#### 4.1 Station Historical Obs Backfill

Create `scripts/ingest_vc_obs_backfill.py`:

1. For each `vc_location` where `location_type='station'`:
   - Loop from desired start date (e.g., 2020-01-01) to yesterday in 7-30 day chunks
   - Call `fetch_station_history_minutes()`
   - Insert into `wx.vc_minute_weather` with:
     - `data_type='actual_obs'`
     - `forecast_basis_date=NULL`
   - Use UPSERT on unique key `(vc_location_id, data_type, forecast_basis_date, datetime_utc)`

2. Mirror for `location_type='city'` using `fetch_city_history_minutes()`

#### 4.2 Nightly Current + Forecast Snapshots

Create `scripts/ingest_vc_forecast_snapshot.py`:

For each `vc_location`:
1. Call `fetch_station_current_and_forecast()` or `fetch_city_current_and_forecast()`
2. Insert:
   - Minute rows → `wx.vc_minute_weather` with `data_type='forecast'`, `forecast_basis_date=today`
   - Hour rows → `wx.vc_forecast_hourly` with `data_type='forecast'`
   - Day rows → `wx.vc_forecast_daily` with `data_type='forecast'`
3. Schedule around local midnight for each timezone (or run hourly and dedupe by basis_date)

#### 4.3 Historical Forecasts Backfill

Create `scripts/ingest_vc_historical_forecast.py`:

For each `event_date` in Kalshi history and each station location:
1. Pick lead times: 0, 1, 2, 3, 5, 7 days
2. Compute `basis_date = event_date - lead`
3. Call `fetch_station_historical_forecast(..., basis_date)`
4. Insert into `vc_forecast_daily` / `vc_forecast_hourly` with:
   - `data_type='historical_forecast'`
   - `forecast_basis_date=basis_date`
   - Derived `lead_days` / `lead_hours`

#### 4.4 Combined Daemon

Create `scripts/poll_vc_daemon.py`:
- 24/7 daemon that triggers data collection
- Runs nightly snapshot at local midnight per timezone
- Uses checkpoint tracking for resumability

---

## Visual Crossing Client Implementation Details

### API Parameter Reference

| Use Case | include | options | maxDistance | maxStations |
|----------|---------|---------|-------------|-------------|
| Station historical obs (5-min) | `obs,minutes` | `minuteinterval_5,stnslevel1,useobs` | 1609 | 1 |
| City historical obs (5-min) | `obs,minutes` | `minuteinterval_5,useobs` | (default) | (default) |
| Station current+forecast (15-min) | `fcst,current,minutes` | `minuteinterval_15,stnslevel1,usefcst` | 1609 | 1 |
| City current+forecast (15-min) | `fcst,current,minutes` | `minuteinterval_15,usefcst` | (default) | (default) |
| Historical forecast (days+hours) | `days,hours` | (none needed) | (default) | (default) |

### Datetime Parsing

From each VC response object, extract:
```python
datetime_epoch_utc = obj['datetimeEpoch']  # Unix seconds
datetime_utc = datetime.fromtimestamp(datetime_epoch_utc, tz=timezone.utc)
datetime_local = datetime.fromisoformat(obj['datetime'])  # VC returns local
tz_name = obj.get('timezone', location_iana_tz)  # from location or response
tzoffset_minutes = int(obj.get('tzoffset', 0) * 60)
```

---

## Key Design Decisions

1. **Greenfield schema** - New VC-specific tables, old tables remain but unused
2. **Both feeds** - Station + city-aggregate for each city (12 locations total)
3. **Full datetime handling** - epoch_utc, datetime_utc, datetime_local, timezone, tzoffset_minutes
4. **Classification system** - data_type enum + forecast_basis_date + lead_days/hours
5. **ML-ready fields** - lead_days, lead_hours computed and stored at insert time
6. **Extensible** - source_system field for future providers
7. **5-min obs, 15-min forecast** - respecting API limits
8. **UPSERT logic** - Unique constraints enable clean idempotent ingestion
9. **CHECK constraints** - Prevent invalid data_type values at DB level

---

## Smoke Test Plan

After implementation, validate with a small test:
1. **One city (Chicago)**, a few days of data
2. Verify:
   - Station vs city results differ when VC is interpolating
   - All weather fields populated (non-null where expected)
   - Datetime fields correctly computed
   - lead_hours/lead_days calculated properly
3. Then scale to all 6 cities and full date range
