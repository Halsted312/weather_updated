---
plan_id: cloudcover-interpolation-fix
created: 2025-12-04
status: draft
priority: critical
agent: kalshi-weather-quant
---

# Cloudcover Interpolation Fix - Proper Implementation

## Problem

**Error:** `ValueError: cannot reindex on an axis with duplicate labels`

**Root cause:** Multiple forecasts exist for the same `target_datetime_local` (from different `forecast_basis_date` values), creating duplicate timestamps in the interpolation index.

**Example:** For 2024-06-01 14:00, we might have:
- Forecast from basis 2024-05-31 → cloudcover=50%
- Forecast from basis 2024-05-30 → cloudcover=55%
- Forecast from basis 2024-05-29 → cloudcover=60%

All three have `target_datetime_local = 2024-06-01 14:00` → duplicates.

---

## Schema Analysis (Verified from models.py)

### wx.vc_forecast_hourly

**Primary key:** `id` (auto-increment)

**Key columns:**
- `vc_location_id` (FK to vc_location)
- `data_type` ('forecast' | 'historical_forecast')
- `forecast_basis_date` (DATE) - when forecast was issued
- `target_datetime_local` (TIMESTAMP) - what hour is being predicted
- `cloudcover` (FLOAT, nullable)

**Critical insight:** For historical backtesting, we want T-1 forecast (basis_date = event_date - 1 day)

---

## Correct Approach - Use Most Recent Forecast (T-1) Only

### Key Principle
**Never mix forecasts from different basis dates for the same event**

For each event_date:
- Use T-1 forecast only (`forecast_basis_date = event_date - 1`)
- This gives 24 hourly cloudcover values for that day
- NO duplicates (one forecast basis per event)
- Respects information timing (no future data leakage)

### Implementation Strategy

**Process per event_date** (not all dates at once):
```python
for event_date in unique_event_dates:
    # Get T-1 forecast for this event
    basis_date = event_date - timedelta(days=1)

    query = """
        SELECT target_datetime_local, cloudcover
        FROM wx.vc_forecast_hourly
        WHERE vc_location_id = :vc_location_id
          AND forecast_basis_date = :basis_date
          AND DATE(target_datetime_local) = :event_date
          AND cloudcover IS NOT NULL
          AND data_type = 'historical_forecast'
        ORDER BY target_datetime_local
    """
    # Returns ~24 hourly values, NO duplicates

    # Interpolate these 24 values to match snapshot timestamps for this day
    # Linear interpolation: hour 14 → 15, snapshots at 14.083, 14.167, etc.
```

### Why This Works

**Advantages:**
1. ✅ No duplicate timestamps (one basis date per event)
2. ✅ Respects information timing (T-1 forecast available at market open)
3. ✅ Consistent with temperature forecast usage elsewhere
4. ✅ Simple, robust, no arbitrary averaging

**Handles edge cases:**
- Missing T-1 forecast → cloudcover remains None (CatBoost handles)
- Multiple forecasts per hour (shouldn't happen with basis_date filter) → take first

### Step 3: Interpolate Per Day

For each day's snapshots:
1. Load T-1 hourly forecast for that day (24 hours, no duplicates)
2. Filter dataset rows for that event_date
3. Interpolate cloudcover from 24 hourly values to match snapshot timestamps
4. Merge back into main dataset

### Step 4: Handle Edge Cases

**Missing T-1 forecast:**
- Fallback to T-2 if T-1 unavailable
- Or leave as None (CatBoost handles nulls)

**Timezone mismatches:**
- Ensure both dataframes use naive datetimes (no tz)
- Or both use same timezone

**Duplicate target_datetime_local despite basis filter:**
- Drop duplicates, keep first/last, or average

---

## Implementation Plan

### File: scripts/patch_cloudcover_all_cities.py

**Function 1: load_hourly_cloudcover_for_day**
```python
def load_hourly_cloudcover_for_day(
    session,
    vc_location_id: int,
    event_date: date
) -> pd.DataFrame:
    """Load T-1 hourly cloudcover for a specific day (no duplicates)."""
    basis_date = event_date - timedelta(days=1)

    query = text("""
        SELECT DISTINCT ON (target_datetime_local)
            target_datetime_local,
            cloudcover
        FROM wx.vc_forecast_hourly
        WHERE vc_location_id = :vc_location_id
          AND forecast_basis_date = :basis_date
          AND DATE(target_datetime_local) = :event_date
          AND cloudcover IS NOT NULL
          AND data_type = 'historical_forecast'
        ORDER BY target_datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "basis_date": basis_date,
        "event_date": event_date,
    })

    return pd.DataFrame(result.fetchall(), columns=['target_datetime_local', 'cloudcover'])
```

**Function 2: patch_cloudcover_for_city**
```python
def patch_cloudcover_for_city(city: str):
    # 1. Load existing parquet
    df = pd.read_parquet(f'data/training_cache/{city}/full.parquet')

    # 2. Get vc_location_id
    with get_db_session() as session:
        vc_location_id = get_vc_location_id(session, city, "city")

    # 3. Process each event_date
    unique_dates = df['event_date'].unique()

    cloudcover_map = {}  # {(event_date, snapshot_hour): cloudcover_value}

    for event_date in unique_dates:
        # Load T-1 hourly cloudcover for this day
        hourly_cc = load_hourly_cloudcover_for_day(session, vc_location_id, event_date)

        if hourly_cc.empty:
            continue

        # Create hourly series (indexed by hour)
        hourly_series = hourly_cc.set_index('target_datetime_local')['cloudcover']

        # Interpolate to snapshot hours
        day_snapshots = df[df['event_date'] == event_date]

        for _, row in day_snapshots.iterrows():
            snapshot_hour = row['snapshot_hour']

            # Linear interpolation between surrounding hours
            hour_floor = int(snapshot_hour)
            hour_ceil = hour_floor + 1

            # Get cloudcover at floor and ceil hours
            # ... interpolate ...

            cloudcover_map[(event_date, snapshot_hour)] = interpolated_value

    # 4. Map cloudcover back to df
    df['cloudcover_last_obs'] = df.apply(
        lambda row: cloudcover_map.get((row['event_date'], row['snapshot_hour'])),
        axis=1
    )

    # 5. Recompute derived features
    df['clear_sky_flag'] = (df['cloudcover_last_obs'] < 20).astype(float)
    df['high_cloud_flag'] = (df['cloudcover_last_obs'] > 70).astype(float)
    # ... etc ...

    # 6. Save back to parquet
    df.to_parquet(f'data/training_cache/{city}/full.parquet', index=False)
```

---

## FINAL IMPLEMENTATION - T-1 Forecast Aligned with System

### Load Function (Corrected)
```python
def load_hourly_cloudcover_for_event(
    session, vc_location_id: int, event_date: date
) -> pd.DataFrame:
    """Load T-1 hourly cloudcover for specific event_date."""
    from datetime import timedelta
    from sqlalchemy import text

    basis_date = event_date - timedelta(days=1)  # T-1 forecast

    query = text("""
        SELECT
            target_datetime_local,
            cloudcover,
            forecast_basis_date
        FROM wx.vc_forecast_hourly
        WHERE vc_location_id = :vc_location_id
          AND forecast_basis_date = :basis_date
          AND DATE(target_datetime_local) = :event_date
          AND cloudcover IS NOT NULL
          AND data_type = 'historical_forecast'
        ORDER BY target_datetime_local
    """)

    result = session.execute(query, {
        "vc_location_id": vc_location_id,
        "basis_date": basis_date,
        "event_date": event_date,
    })

    rows = result.fetchall()
    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=['target_datetime_local', 'cloudcover', 'forecast_basis_date'])
```

### Patch Function (Corrected)
```python
def patch_city_cloudcover(city: str):
    # Load existing parquet
    df = pd.read_parquet(f'data/training_cache/{city}/full.parquet')

    # Get vc_location_id
    with get_db_session() as session:
        vc_location_id = get_vc_location_id(session, city, "city")

        cloudcover_data = []

        # Process each unique event_date
        for event_date in df['event_date'].unique():
            # Load T-1 hourly forecast for this event
            hourly_cc = load_hourly_cloudcover_for_event(session, vc_location_id, event_date)

            if hourly_cc.empty:
                continue

            # Interpolate to 5-min intervals
            hourly_cc['target_datetime_local'] = pd.to_datetime(hourly_cc['target_datetime_local'])
            if hourly_cc['target_datetime_local'].dt.tz is not None:
                hourly_cc['target_datetime_local'] = hourly_cc['target_datetime_local'].dt.tz_localize(None)

            # Create time series for this day
            hourly_series = hourly_cc.set_index('target_datetime_local')['cloudcover'].sort_index()

            # Get all snapshot times for this event_date
            day_snapshots = df[df['event_date'] == event_date].copy()
            day_snapshots['snapshot_dt'] = pd.to_datetime(day_snapshots['event_date']) + pd.to_timedelta(day_snapshots['snapshot_hour'], unit='h')

            # Interpolate
            all_times = pd.concat([
                pd.Series(day_snapshots['snapshot_dt']),
                pd.Series(hourly_series.index)
            ]).drop_duplicates().sort_values()

            interpolated = hourly_series.reindex(all_times).interpolate(method='linear')

            # Map back
            for idx, row in day_snapshots.iterrows():
                cc_value = interpolated.loc[row['snapshot_dt']] if row['snapshot_dt'] in interpolated.index else None
                cloudcover_data.append((idx, cc_value))

        # Update dataframe
        for idx, cc_value in cloudcover_data:
            df.at[idx, 'cloudcover_last_obs'] = cc_value
            if pd.notna(cc_value):
                df.at[idx, 'clear_sky_flag'] = 1.0 if cc_value < 20 else 0.0
                df.at[idx, 'high_cloud_flag'] = 1.0 if cc_value > 70 else 0.0
                df.at[idx, 'cloud_regime'] = 0.0 if cc_value < 20 else (1.0 if cc_value < 70 else 2.0)
                if 'hour' in df.columns:
                    df.at[idx, 'cloudcover_x_hour'] = cc_value * df.at[idx, 'hour']

    # Save
    df.to_parquet(f'data/training_cache/{city}/full.parquet', index=False)
```

## Testing Strategy

### Test 1: Verify No Duplicates in Hourly Data

```python
# For Austin, event_date=2024-06-01, basis_date=2024-05-31
# Should get exactly ~24 hourly records (0:00 to 23:00)
# NO duplicates because basis_date is fixed
```

### Test 2: Verify Interpolation Works

```python
# Input: Hourly cloudcover [10, 20, 30, ...] at hours [0, 1, 2, ...]
# Output: snapshot_hour=0.5 → cloudcover=15 (linear interpolation)
```

### Test 3: Verify Patch Doesn't Break Existing Data

```python
# Before: 237 columns
# After: 237 columns (cloudcover columns updated, not added)
# Before: 486,912 rows
# After: 486,912 rows (no row changes)
```

---

## Estimated Time

- Fix implementation: 30 minutes
- Test on Austin: 5 minutes
- Patch all 6 cities: 10-15 minutes

---

## Next Steps

1. **Implement fix** with proper duplicate handling
2. **Test on Austin** dataset only
3. **Verify cloudcover features** populated (>90%)
4. **Patch all cities** if Austin succeeds
5. **Run ordinal training** with cloudcover features included

---

**STOP - Do not proceed without user approval of this plan**
