# Visual Crossing Minute Data - Implementation Guide

## Overview

Visual Crossing (VC) provides 5-minute weather observations used as **features only** (never for settlement). This document explains our station-pinned implementation and NYC exclusion.

## Station-Pinned Best Practices

### Why Station Locking Matters

NWS CF6/CLI and Kalshi both reference **specific stations** (KMDW for Chicago, KDEN for Denver, etc.). To ensure VC features align with official climate records, we:

1. **Use `stn:<stationid>` location format** - Explicitly requests a specific station
2. **Set `maxStations=1` and `maxDistance=0`** - Disables multi-station blending
3. **Include `stations` in elements** - Tracks which station VC used for each minute (for diagnostics)

### VC Client Parameters

```python
params = {
    "key": API_KEY,
    "unitGroup": "us",                              # Fahrenheit, mph
    "include": "minutes",                            # Sub-hourly data
    "options": "useobs,minuteinterval_5,nonulls",    # Observed data, 5-min, suppress nulls
    "elements": "datetimeEpoch,temp,dew,humidity,windspeed,stations",  # Core + diagnostics
    "timezone": "Z",                                 # UTC
    "maxStations": "1",                              # Single station only
    "maxDistance": "0",                              # No multi-station blending
    "contentType": "json",
}
```

**URL Format:** `{base}/stn:{station}/{start}/{end}?{params}`

Example: `https://weather.visualcrossing.com/.../stn:KMDW/2024-01-10/2024-01-10?...`

## Forward-Fill Implementation

VC data may have gaps. We create a complete 5-minute UTC grid (288 slots/day) using forward-fill:

1. **Build grid:** `pd.date_range(start_utc, end_utc, freq='5min', tz='UTC')`
2. **Merge** actual VC observations
3. **Mark synthetic rows:** `ffilled=TRUE` for forward-filled, `FALSE` for real observations
4. **Forward-fill numeric columns:** `temp_f`, `humidity`, `dew_f`, `windspeed_mph`

### Quality Metrics

- **Good coverage:** >99% complete days, <2% forward-fill
- **Temperature agreement:** Avg |VC daily max - CF6| ≈ 0.5°F
- **Acceptance:** 93-97% of days within ±2°F

## Cities: 6 Good, 1 Excluded

### Good Cities (Airport Stations with Dense ASOS Data)

| City         | Station | Coverage | Avg Ffilled | Avg Δ vs CF6 |
|--------------|---------|----------|-------------|--------------|
| Austin       | KAUS    | >99%     | <2%         | ~0.5°F       |
| Chicago      | KMDW    | >99%     | <2%         | ~0.5°F       |
| Los Angeles  | KLAX    | >99%     | <2%         | ~0.5°F       |
| Miami        | KMIA    | >99%     | <2%         | ~0.5°F       |
| Denver       | KDEN    | >99%     | <2%         | ~0.5°F       |
| Philadelphia | KPHL    | >99%     | <2%         | ~0.5°F       |

### Excluded City: NYC

**Station:** KNYC (Central Park)
**Why excluded:**
- Central Park is a **climate station**, not an ASOS airport station
- Lacks dense sub-hourly observations
- VC must heavily interpolate/blend from remote stations
- Result: **~82% forward-fill** vs <2% for airport stations

**Action (updated 2025-11-18):**
- ❌ Drop NYC entirely from the pipeline (no new ingestion/backtests)
- ✅ Keep historical data archived for reference only

```python
# In weather/visual_crossing.py
EXCLUDED_VC_CITIES: set[str] = set()  # NYC removed from active city list
```

## Database Schema

### WxMinuteObs Table

```sql
CREATE TABLE wx.minute_obs (
    loc_id       VARCHAR(10) NOT NULL,  -- e.g., "KMDW"
    ts_utc       TIMESTAMP WITH TIME ZONE NOT NULL,
    temp_f       FLOAT,
    humidity     FLOAT,
    dew_f        FLOAT,
    windspeed_mph FLOAT,
    source       VARCHAR(20) DEFAULT 'visualcrossing',
    stations     VARCHAR(50),           -- Station ID used by VC (e.g., "KMDW")
    ffilled      BOOLEAN NOT NULL DEFAULT FALSE,  -- TRUE if forward-filled, FALSE if real
    raw_json     JSON,
    PRIMARY KEY (loc_id, ts_utc)
);
```

## Usage in ML Pipeline

### Feature Extraction

```python
from weather.visual_crossing import EXCLUDED_VC_CITIES

def extract_vc_features(city: str, date: date) -> pd.DataFrame:
    if city in EXCLUDED_VC_CITIES:
        # Skip VC features for NYC
        return pd.DataFrame()

    # For good cities: use VC minute data
    # ... query wx.minute_obs WHERE loc_id = ... AND ffilled = FALSE ...
```

### Quality Filters

```python
# Per-day quality check
def is_vc_day_acceptable(df: pd.DataFrame) -> bool:
    if len(df) != 288:
        return False  # Incomplete grid

    ffilled_pct = 100.0 * df['ffilled'].sum() / len(df)
    if ffilled_pct > 50.0:
        return False  # Too much forward-fill

    return True
```

## Backfill Commands

### Test (1 week, 1 city)
```bash
python ingest/backfill_visualcrossing.py \
    --start-date 2024-01-10 \
    --end-date 2024-01-16 \
    --cities chicago \
    --ffill \
    --replace
```

### Production (6 cities, full range)
```bash
python ingest/backfill_visualcrossing.py \
    --start-date 2024-01-01 \
    --end-date 2025-11-14 \
    --cities austin chicago los_angeles miami denver philadelphia \
    --ffill \
    --replace
```

## Validation

```bash
python scripts/validate_vc_completeness.py \
    --start-date 2024-01-01 \
    --end-date 2025-11-14 \
    --cities austin chicago los_angeles miami denver philadelphia \
    --output data/vc_validation_6cities.csv
```

Expected output:
- Coverage per city: ≥99% complete days
- Average ffilled%: <2%
- Average |vc_tmax - tmax_final_f|: ≈0.5°F
- Days within ±2°F: >93%

## Key Principles

1. **CF6/CLI is ground truth** - VC is features only, never settlement
2. **Station locking** - Use `stn:{id}` + `maxDistance=0` to match official station
3. **Forward-fill tracking** - `ffilled` column distinguishes real from synthetic
4. **Quality gates** - Filter out days with >50% forward-fill or >3°F delta
5. **NYC exclusion** - Keep labels and market features, skip VC minutes

## References

- [VC Station Selection](https://www.visualcrossing.com/resources/documentation/weather-data/how-do-i-find-my-nearest-weather-station/)
- [VC Sub-Hourly Data](https://www.visualcrossing.com/resources/documentation/weather-api/requesting-sub-hourly-weather-data-using-the-weather-api/)
- [VC Timeline API](https://www.visualcrossing.com/resources/documentation/weather-api/timeline-weather-api/)
