# Legacy Visual Crossing Ingestion Scripts

This folder contains **archived** Visual Crossing ingestion scripts that use the **old database schema**. They are kept for reference only and are **not used by the current pipeline**.

## Old Schema (deprecated)

These scripts wrote to:
- `wx.minute_obs` - 5-minute observations (limited fields, no timezone handling)
- `wx.forecast_snapshot` - Daily forecast snapshots
- `wx.forecast_snapshot_hourly` - Hourly forecast snapshots

## Scripts in this folder

| Script | Original Purpose | Superseded By |
|--------|------------------|---------------|
| `ingest_vc_minutes.py` | 5-minute observations | `scripts/ingest_vc_obs_backfill.py` |
| `ingest_vc_forecast_history.py` | Historical daily forecasts | `scripts/ingest_vc_historical_forecast.py` |
| `ingest_vc_forecast_hourly.py` | Historical hourly forecasts | `scripts/ingest_vc_historical_forecast.py` |
| `poll_vc_forecast_daemon.py` | 24/7 nightly snapshots | `scripts/ingest_vc_forecast_snapshot.py` |

## New Schema (active)

The current pipeline uses the Phase 1 VC schema:
- `wx.vc_location` - Location dimension (station + city for each market)
- `wx.vc_minute_weather` - Unified minute-level data (47+ fields, proper timezone handling)
- `wx.vc_forecast_daily` - Daily forecast snapshots (full field support)
- `wx.vc_forecast_hourly` - Hourly forecast snapshots (full field support)

## Active Ingestion Scripts

| Script | Purpose |
|--------|---------|
| `scripts/ingest_vc_obs_backfill.py` | Historical 5-min observations |
| `scripts/ingest_vc_forecast_snapshot.py` | Nightly current+forecast snapshots |
| `scripts/ingest_vc_historical_forecast.py` | Historical forecast backfill |

## Why Keep These?

- Reference for understanding the old pipeline
- Comparison if debugging data migration issues
- Documentation of previous approach
