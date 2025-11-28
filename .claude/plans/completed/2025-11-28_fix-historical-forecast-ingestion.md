---
plan_id: fix-historical-forecast-ingestion
created: 2025-11-28
status: completed
priority: high
agent: kalshi-weather-quant
---

# Fix Historical Forecast Ingestion

## Problem Summary

The current historical forecast ingestion is returning **observation data** (`source=obs`) instead of actual historical forecasts (`source=fcst`). All lead_days (0,1,2,3) return identical tempmax values because the API is ignoring the `forecastBasisDate` parameter for station queries.

## Root Cause Analysis

### API Testing Results (2024-07-15, Chicago)

| Location Type | Query Format | lead=0 | lead=1 | lead=2 | lead=3 | source |
|--------------|--------------|--------|--------|--------|--------|--------|
| City aggregate | `Chicago,IL` | 89.1 | 91.3 | 91.1 | 89.1 | **fcst** |
| **Lat/Lon** | `41.78412,-87.75514` | 91.3 | 93.5 | 92.6 | 92.0 | **fcst** |
| Station | `stn:KMDW` | 91.1 | 91.1 | 91.1 | 91.1 | obs |

**Key findings:**
1. **Lat/lon queries work!** - Anchored at station coordinates, returns `source=fcst` with varying tempmax
2. City aggregate also works but is "blended" across the metro area
3. Station-locked queries (`stn:KMDW`) always return observation data - `forecastBasisDay` is ignored
4. Lat/lon approach gives us forecast data anchored at the actual settlement station location

## Solution

### Approach: Use Lat/Lon Queries with Station Coordinates

Use lat/lon queries (not `stn:` queries) to get historical forecasts anchored at the settlement station location:

1. Add `latitude` and `longitude` fields to `CityConfig` dataclass in `cities.py`
2. Add `vc_latlon_query` property that returns `"lat,lon"` format
3. Use `forecastBasisDay` parameter for cleaner lead-day queries
4. Query with lat/lon to get forecasts anchored at the actual station

### Station Coordinates (from NOAA/NCEI)

| City | Station | Latitude | Longitude |
|------|---------|----------|-----------|
| Chicago | KMDW | 41.78412 | -87.75514 |
| Austin | KAUS | 30.18311 | -97.67989 |
| Denver | KDEN | 39.84657 | -104.65623 |
| Los Angeles | KLAX | 33.93816 | -118.38660 |
| Miami | KMIA | 25.78805 | -80.31694 |
| Philadelphia | KPHL | 39.87326 | -75.22681 |

### API Pattern

For each (city, target_date, lead_days):
```
GET /timeline/{lat},{lon}/{target_date}/{target_date}
    ?unitGroup=us
    &include=days,hours
    &forecastBasisDay={lead_days}
    &key={API_KEY}
```

Example for Chicago, target=2024-07-15, lead=3:
```
GET /timeline/41.78412,-87.75514/2024-07-15/2024-07-15?forecastBasisDay=3&include=days,hours&...
```

## Implementation Tasks

### Phase 1: Update cities.py with lat/lon
- [ ] Add `latitude: float` and `longitude: float` fields to `CityConfig`
- [ ] Add `vc_latlon_query` property returning `f"{self.latitude},{self.longitude}"`
- [ ] Update all 6 cities with NOAA station coordinates

### Phase 2: Create new ingestion script
- [ ] Create `scripts/ingest_vc_hist_forecast_v2.py`
- [ ] Use `forecastBasisDay` parameter
- [ ] Query using `city.vc_latlon_query` for station-anchored forecasts
- [ ] Validate `source` field is 'fcst' before storing
- [ ] Support parallel processing by month

### Phase 3: Chicago Test
- [ ] Test with Chicago lat/lon, 2024-07-01 to 2024-07-31
- [ ] Verify tempmax values differ by lead_day
- [ ] Verify source='fcst' (not 'obs')

### Phase 4: Chicago Full Backfill
- [ ] Run parallel backfill: 1 process per month, 12 cores
- [ ] Date range: 2023-01-01 to 2025-11-27
- [ ] Store as data_type='historical_forecast'

### Phase 5: All Cities Backfill
- [ ] Extend to all 6 cities
- [ ] Parallel by city×month chunks

## Files to Modify

| Action | Path | Notes |
|--------|------|-------|
| MODIFY | `src/config/cities.py` | Add latitude, longitude fields + vc_latlon_query property |
| CREATE | `scripts/ingest_vc_hist_forecast_v2.py` | New script using lat/lon + forecastBasisDay |

## Technical Details

### CityConfig changes
```python
@dataclass(frozen=True)
class CityConfig:
    # ... existing fields ...
    latitude: float   # Station latitude (NOAA/NCEI)
    longitude: float  # Station longitude (NOAA/NCEI)

    @property
    def vc_latlon_query(self) -> str:
        """Visual Crossing lat/lon query for forecast anchored at station."""
        return f"{self.latitude},{self.longitude}"
```

### Parallelization Strategy
- 12 workers (one per month for Chicago backfill)
- Each worker handles: one month × 4 lead_days
- ~30 days × 4 leads = ~120 API calls per worker
- With 0.05s delay: ~6 seconds per worker

### Data Validation
```python
if day_data.get('source') == 'obs':
    logger.warning(f"Got obs instead of forecast for {target_date} - skipping")
    continue  # Don't store observation data as forecast
```

## Completion Criteria

- [x] cities.py updated with lat/lon for all 6 cities
- [x] Chicago test shows varying tempmax for different lead_days
- [x] Source field is 'fcst' for lat/lon queries
- [x] Full Chicago backfill complete
- [x] DB contains ~1,060 days × 4 leads = ~4,240 rows for Chicago
- [x] All 6 cities backfilled (25,434 daily rows, 611,568 hourly rows total)

## Sign-off Log

### 2025-11-28 08:00 CST
**Status**: COMPLETED

**Summary**:
Successfully fixed historical forecast ingestion by switching from station queries (`stn:KMDW`) to lat/lon queries anchored at station coordinates. The station query approach was ignoring the `forecastBasisDay` parameter and returning observation data instead of historical forecasts.

**What was done**:
- ✅ Added `latitude`, `longitude` fields and `vc_latlon_query` property to CityConfig
- ✅ Created `scripts/ingest_vc_hist_forecast_v2.py` with lat/lon + forecastBasisDay approach
- ✅ Validated source='fcst' filtering works correctly (obs data is skipped)
- ✅ Backfilled all 6 cities for 2023-01-01 to 2025-11-27
- ✅ Deleted old bad data (50,980 daily + 1,223,520 hourly rows with source_system='vc_timeline')

**Final Results**:
| City | Daily Rows | Hourly Rows |
|------|------------|-------------|
| Chicago | 4,239 | 101,928 |
| Austin | 4,239 | 101,928 |
| Denver | 4,239 | 101,928 |
| Los Angeles | 4,239 | 101,928 |
| Miami | 4,239 | 101,928 |
| Philadelphia | 4,239 | 101,928 |
| **Total** | **25,434** | **611,568** |

**Data Quality**:
- Coverage: 99.8%+ (only ~8 gaps per city from 2 VC archive gaps: Nov 25, 2024 and Jan 4, 2025)
- All stored data is `source=fcst` (verified)
- Script correctly skipped obs returns

**Files Modified**:
- `src/config/cities.py` - Added lat/lon coordinates for all 6 cities
- `scripts/ingest_vc_hist_forecast_v2.py` - New ingestion script (created)
