---
plan_id: historical-forecast-ingestion
created: 2025-11-28
status: in_progress
priority: high
agent: kalshi-weather-quant
---

# Historical Forecast Ingestion Plan

## Objective
Ingest Visual Crossing historical forecasts (lead_days 0,1,2,3) for ML model training, enabling forecast-vs-actual delta features.

## Context
The ML modeling pipeline (per `modeling2.2.md`) requires:
- **T-1 forecast**: What VC predicted yesterday for today's high
- **T-2, T-3 forecasts**: How forecasts evolved over time
- **T-0 forecast**: Same-day midnight forecast as additional signal

This data enables features like `fcst_prev_max_f`, `err_mean_so_far`, `err_last1h` for the intraday Δ-models.

**Prerequisites completed**:
- Database schema ready (Migration 007): `wx.vc_forecast_daily`, `wx.vc_forecast_hourly`
- Ingestion script exists: `scripts/ingest_vc_historical_forecast.py`
- VC observations already backfilled

## Tasks

### Phase 1: Test Run (Chicago only)
- [ ] Run Chicago test ingestion (2025-05-01 to yesterday, lead_days 0,1,2,3)
- [ ] Ingest both station (KMDW) and city aggregate
- [ ] Validate data quality (nulls, extremes, coverage)

### Phase 2: Full Parallel Backfill
- [ ] Create parallel ingestion script (city×year chunks)
- [ ] Run full backfill: 6 cities × ~3 years × lead_days 0,1,2,3
- [ ] Validate full dataset coverage and quality

### Phase 3: Data Quality Report
- [ ] Check forecast bias vs settlement (mean error per city/lead_day)
- [ ] Identify any sentinel values or missing data patterns
- [ ] Document any data issues found

## Technical Details

### Lead Days Semantics
For a target date T (e.g., Thursday):
- `lead_days=0`: Forecast issued midnight Thursday for Thursday
- `lead_days=1`: Forecast issued Wednesday for Thursday
- `lead_days=2`: Forecast issued Tuesday for Thursday
- `lead_days=3`: Forecast issued Monday for Thursday

### API Call Pattern
```
GET /timeline/{station}/{target_start}/{target_end}
    ?key=API_KEY
    &unitGroup=us
    &include=days,hours
    &forecastBasisDate={basis_date}
    &elements={ALL_ELEMENTS}
```

### Parallel Processing Strategy
- 24 threads available
- Chunk by: city × year (6 cities × 3 years = 18 chunks)
- No rate limiting concerns (unlimited VC tokens)

### Expected Data Volumes
| Lead Days | Days/City (~1,060) | Total Rows (6 cities) |
|-----------|---------------------|------------------------|
| 0,1,2,3   | 4,240               | ~25,440 daily rows     |
| Hourly    | 4,240 × 24 = 101,760| ~610,560 hourly rows   |

## Completion Criteria
- [ ] Chicago test data validated (no critical issues)
- [ ] All 6 cities × all lead_days ingested
- [ ] Coverage ≥95% (some gaps from VC outages acceptable)
- [ ] Forecast bias validated (mean error within ±2°F per city)

## Sign-off Log

### 2025-11-28 ~10:00 CST
**Status**: Starting Phase 1
**Next steps**:
1. Run Chicago test ingestion
2. Validate data quality
3. Proceed to full backfill if clean
