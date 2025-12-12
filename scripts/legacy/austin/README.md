# Austin Scripts - Archived 2025-12-11

## Purpose
Austin-specific experiments for 15-minute forecast features using additional
NOAA model guidance (NBM, HRRR, NDFD).

## Why Archived
The 15-minute feature experiments are paused. The standard 5-minute feature
pipeline in `models/features/` is currently used for all cities.

## Files

| File | Description | Was Imported By |
|------|-------------|-----------------|
| `backfill_vc_historical_forecast_minutes_austin.py` | Backfill 15-min VC forecasts | Nothing |
| `ingest_weather_more_apis_guidance.py` | Ingest NOAA guidance (NBM/HRRR/NDFD) | Nothing |
| `ingest_weather_more_apis_guidance_FAST.py` | Fast version of above | Nothing |
| `validate_15min_ingestion.py` | Validate 15-min data quality | Nothing |
| `debug_vc_15min_forecast.py` | Debug 15-min forecasts | Nothing |
| `augment_austin_noaa_features.py` | Add NOAA features to Austin | Nothing |
| `verify_austin_augmented.py` | Verify augmentation | Nothing |
| `train_austin_more_apis_6m.py` | Train with 6-month data | Nothing |
| `debug_austin_features_snapshot.py` | Debug feature snapshots | Nothing |
| `plot_austin_forecast_vs_obs.py` | Plotting utility | Nothing |

## Data Dependencies
These scripts populated:
- `wx.weather_more_apis_guidance` table (NOAA guidance)
- `wx.vc_forecast_minute` table (15-min forecasts)

## Revival Notes
To resume 15-minute feature work:
1. Check `src/weather_more_apis/` for NOAA ingestion utilities
2. The `models/features/more_apis.py` already supports these features
3. May need to update database schema for new data
