# Historical Forecast Ingestion Plan
**Created**: 2025-11-28
**Purpose**: Backfill Visual Crossing historical forecasts for ML modeling and backtesting

---

## Executive Summary

Ingest **Visual Crossing historical forecasts** using `forecastBasisDate` to reconstruct what the forecast looked like on past dates. This enables:

1. **ML training**: Compare forecasts vs actuals to learn systematic biases
2. **Backtesting**: Use only information available at decision time (no look-ahead)
3. **Intraday modeling**: Track how forecast-actual deltas evolve through the day

**Target**: All 6 cities, 2023-01-01 through 2025-11-27 (~1,062 days each)

---

## Data Model

### Two New Tables (Already in Schema from Migration 007)

#### 1. `wx.vc_forecast_daily` - Daily Forecast Snapshots

```sql
CREATE TABLE wx.vc_forecast_daily (
    id BIGSERIAL PRIMARY KEY,
    vc_location_id INTEGER REFERENCES wx.vc_location(id),
    data_type TEXT CHECK (data_type IN ('forecast', 'historical_forecast')),
    forecast_basis_date DATE NOT NULL,
    target_date DATE NOT NULL,
    lead_days INTEGER NOT NULL,  -- target_date - forecast_basis_date

    -- Daily forecast fields
    tempmax_f FLOAT,
    tempmin_f FLOAT,
    temp_f FLOAT,
    humidity FLOAT,
    precip_in FLOAT,
    precipprob FLOAT,
    windspeed_mph FLOAT,
    cloudcover FLOAT,
    conditions TEXT,
    -- ... (47+ weather fields)

    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (vc_location_id, target_date, forecast_basis_date, data_type)
);
```

**Key**: Daily high/low forecasts for backtesting bracket selection

#### 2. `wx.vc_forecast_hourly` - Hourly/Sub-hourly Forecast Curves

```sql
CREATE TABLE wx.vc_forecast_hourly (
    id BIGSERIAL PRIMARY KEY,
    vc_location_id INTEGER REFERENCES wx.vc_location(id),
    data_type TEXT CHECK (data_type IN ('forecast', 'historical_forecast')),
    forecast_basis_date DATE NOT NULL,
    target_datetime_utc TIMESTAMPTZ NOT NULL,
    target_datetime_local TIMESTAMP NOT NULL,
    timezone TEXT NOT NULL,
    lead_hours INTEGER NOT NULL,

    -- Hourly forecast fields
    temp_f FLOAT,
    feelslike_f FLOAT,
    humidity FLOAT,
    windspeed_mph FLOAT,
    -- ... (47+ weather fields)

    raw_json JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (vc_location_id, target_datetime_utc, forecast_basis_date, data_type)
);
```

**Key**: Hour-by-hour forecast temps for intraday feature engineering

---

## Ingestion Strategy

### Phase 1: Historical Daily Forecasts (Critical for Backtesting)

**Goal**: For each `(city, target_date)`, backfill forecasts issued 1, 2, 3, 5, 7 days prior

**API Pattern**:
```
GET /timeline/{station}/{target_date}/{target_date}
    ?key=API_KEY
    &unitGroup=us
    &include=days
    &forecastBasisDate={basis_date}
    &elements={ALL_ELEMENTS}
```

**Example**: Chicago, target 2025-11-20, basis 2025-11-19 (1-day ahead)

**Lead days to backfill**: [0, 1, 2, 3, 5, 7]
- 0 = same-day forecast (issued at midnight)
- 1 = 1-day ahead (most common for open_maker)
- 7 = week-ahead (for longer-term strategies)

**Work items**:
- 6 cities × 1,062 days × 6 lead_days = **38,232 API calls**
- Batching: Can request multiple target dates in one call (7-day batches)
- Estimated: ~38,232 / 7 = **5,462 API calls total**
- Time: ~2.5 sec/call × 5,462 = **~3.8 hours sequential**
- **Parallel (28 workers)**: ~8 minutes

### Phase 2: Historical Hourly Forecasts (For ML Features)

**Goal**: For subset of dates, backfill hourly forecast curves

**API Pattern**:
```
GET /timeline/{station}/{target_start}/{target_end}
    ?key=API_KEY
    &unitGroup=us
    &include=days,hours
    &forecastBasisDate={basis_date}
    &elements={ALL_ELEMENTS}
```

**Sampling strategy**:
- Every 7th day for full period (reduce API load)
- OR all days for recent period (e.g., 2024-2025 only)
- OR on-demand for specific backtests

**Work items** (if every 7th day):
- 6 cities × 152 days (every 7th) × 6 lead_days = **5,472 API calls**
- Time parallel: ~10 minutes

---

## Implementation Plan

### Step 1: Verify Schema (Already Done ✅)

Migration 007 created:
- `wx.vc_location` (12 seed locations)
- `wx.vc_forecast_daily`
- `wx.vc_forecast_hourly`

### Step 2: Create Historical Forecast Ingestion Script

**File**: `scripts/ingest_vc_historical_forecast.py` (already exists)

**Parameters**:
```bash
--city-code CHI              # Single city
--all-cities                 # All 6 cities
--start-date 2023-01-01      # First target date
--end-date 2025-11-27        # Last target date
--lead-days 0,1,2,3,5,7      # Which forecast horizons
--granularity daily          # 'daily' or 'hourly'
--batch-days 30              # Days per API call (max 30 for VC)
--dry-run                    # Preview only
```

**Example command**:
```bash
python scripts/ingest_vc_historical_forecast.py \
    --city-code CHI \
    --start-date 2023-01-01 \
    --end-date 2025-11-27 \
    --lead-days 0,1,2,3,5,7 \
    --granularity daily
```

### Step 3: Parallel Ingestion (Using Existing Script Framework)

Use the same parallel pattern as observations:
- One process per city
- 28 workers total (leaving 4 cores free)
- Checkpoint tracking for resume-on-crash

### Step 4: Validation Queries

After ingestion, validate coverage:

```sql
-- Check lead_days coverage
SELECT
    loc.city_code,
    vfd.lead_days,
    COUNT(*) as forecast_days,
    MIN(vfd.target_date) as earliest_target,
    MAX(vfd.target_date) as latest_target
FROM wx.vc_forecast_daily vfd
JOIN wx.vc_location loc ON vfd.vc_location_id = loc.id
WHERE vfd.data_type = 'historical_forecast'
  AND loc.location_type = 'station'
GROUP BY loc.city_code, vfd.lead_days
ORDER BY loc.city_code, vfd.lead_days;

-- Check forecast vs settlement comparison
SELECT
    loc.city_code,
    vfd.lead_days,
    ROUND(AVG(vfd.tempmax_f - s.tmax_final)::numeric, 2) as avg_forecast_bias,
    ROUND(STDDEV(vfd.tempmax_f - s.tmax_final)::numeric, 2) as forecast_std_error,
    COUNT(*) as days
FROM wx.vc_forecast_daily vfd
JOIN wx.vc_location loc ON vfd.vc_location_id = loc.id
JOIN wx.settlement s ON loc.city_code = s.city AND vfd.target_date = s.date_local
WHERE vfd.data_type = 'historical_forecast'
  AND loc.location_type = 'station'
  AND vfd.lead_days = 1  -- 1-day ahead forecast
GROUP BY loc.city_code, vfd.lead_days
ORDER BY loc.city_code;
```

Expected: Mean bias 0±2°F, std error 3-5°F

---

## Execution Sequence

### Test Run: Chicago Only (May-Nov 2025)

**Purpose**: Validate API calls, schema, and data quality before full backfill

```bash
# Daily forecasts only, lead_days 1
python scripts/ingest_vc_historical_forecast.py \
    --city-code CHI \
    --start-date 2025-05-01 \
    --end-date 2025-11-27 \
    --lead-days 1 \
    --granularity daily

# Validate
SELECT COUNT(*), AVG(tempmax_f), MIN(target_date), MAX(target_date)
FROM wx.vc_forecast_daily vfd
JOIN wx.vc_location loc ON vfd.vc_location_id = loc.id
WHERE loc.city_code = 'CHI' AND vfd.lead_days = 1;
```

Expected: ~211 rows, avg tempmax ~63°F

### Full Backfill: All Cities, All Lead Days

**Phase 1**: Daily forecasts (critical path)
```bash
# Run 6 cities in parallel (one process per city)
for city in CHI AUS DEN LAX MIA PHL; do
    python scripts/ingest_vc_historical_forecast.py \
        --city-code $city \
        --start-date 2023-01-01 \
        --end-date 2025-11-27 \
        --lead-days 0,1,2,3,5,7 \
        --granularity daily \
        2>&1 | tee /tmp/fcst_daily_$city.log &
done
wait
```

**Estimated time**: ~8-10 minutes (parallel)

**Phase 2** (Optional): Hourly forecasts
```bash
# Sample: every 7th day, lead_days 1 only
for city in CHI AUS DEN LAX MIA PHL; do
    python scripts/ingest_vc_historical_forecast.py \
        --city-code $city \
        --start-date 2023-01-01 \
        --end-date 2025-11-27 \
        --lead-days 1 \
        --granularity hourly \
        --sample-days 7 \
        2>&1 | tee /tmp/fcst_hourly_$city.log &
done
wait
```

**Estimated time**: ~5 minutes (parallel)

---

## Expected Data Volumes

### Daily Forecasts

| Lead Days | Days per City | Total Rows (6 cities) | Storage (est) |
|-----------|---------------|------------------------|---------------|
| 0,1,2,3,5,7 | 1,062 × 6 = 6,372 | 38,232 | ~15 MB |

### Hourly Forecasts (if every 7th day, lead_days=1 only)

| Days Sampled | Hours per Day | Total Rows (6 cities) | Storage (est) |
|--------------|---------------|------------------------|---------------|
| 152 (every 7th) | 24 | 152 × 24 × 6 = 21,888 | ~25 MB |

**Total storage**: ~40 MB (negligible)

---

## Data Quality Checks

### After Ingestion, Validate:

1. **Coverage**: All cities × all lead_days × all target_dates present
2. **Forecast bias**: Mean error within ±2°F per city
3. **Forecast variance**: Std error 3-5°F (typical for 1-day ahead)
4. **Lead day progression**: Forecast gets sharper as lead_days decreases
5. **No NULLs**: tempmax_f populated for all records

### Known Issues to Handle:

- **VC API outages**: Some basis dates may return no data (station offline)
- **Future dates**: VC returns 0.0 for tempmax on dates beyond forecast horizon
- **Sentinel values**: Check for -77.9°F or other error codes

**Solution**: Same forward-fill + exclusion logic as observations

---

## Integration with ML Pipeline

### How These Forecasts Will Be Used

1. **Static features** (per day):
   - `fcst_tmax_f_prev` (D-1 forecast for target day D)
   - `fcst_tmin_f_prev`
   - `fcst_tmax_hour_prev` (predicted hour of max)

2. **Dynamic features** (per cutoff time τ):
   - `delta_temp_now = actual(τ) - forecast(τ)`
   - `mean_error_sofar = avg(actual(t) - forecast(t))` for t ≤ τ
   - `delta_vcmax_fcstmax = vc_max_sofar - fcst_tmax_f_prev`

3. **Forecast quality features**:
   - `fcst_mean_f_day`, `fcst_std_f_day`
   - `fcst_max_remaining` (what forecast expects after τ)

### Validation Before ML Training

Before building ML models, confirm:
- [ ] 95%+ of (city, target_date, lead_days=1) have valid forecasts
- [ ] Forecast-settlement correlation ≥ 0.8 per city
- [ ] No systematic date alignment issues (timezones correct)
- [ ] Forecast bias is city-specific but stable over time

---

## Success Criteria

**Minimum viable**:
- ✅ Daily forecasts (lead_days=1) for all cities, all dates
- ✅ 95%+ coverage (some gaps from VC outages acceptable)
- ✅ Forecast bias validated (within ±2°F mean)

**Full implementation**:
- ✅ Multiple lead_days (0,1,2,3,5,7) for daily forecasts
- ✅ Hourly forecasts (sampled or full) for intraday features
- ✅ Integrated into feature builder for ML

---

## Estimated Timeline

| Phase | Task | Time (Parallel) |
|-------|------|-----------------|
| 1 | Test Chicago daily (lead_days=1) | 2 min |
| 2 | Full daily backfill (6 lead_days, all cities) | 8-10 min |
| 3 | Validation queries | 2 min |
| 4 | Hourly backfill (sampled, optional) | 5 min |
| 5 | Data quality checks + cleanup | 3 min |

**Total**: ~20-30 minutes to production-ready historical forecasts

---

## Next Steps After Ingestion

1. **Feature engineering**: Build intraday dataset with forecast-actual deltas
2. **ML modeling**: Train multinomial logistic + CatBoost with forecast features
3. **Backtesting integration**: Wire forecasts into open_maker strategies
4. **Live trading**: Set up nightly forecast snapshot ingestion

---

**Status**: Ready to begin
**Prerequisite**: VC observations complete ✅
**Blocker**: None
