---
plan_id: live-vc-data-daemon
created: 2025-11-28
status: in_progress
priority: critical
agent: kalshi-weather-quant
---

# Live Visual Crossing Data Daemon

## Objective

Create a 24/7 background daemon that continuously ingests Visual Crossing observations (5-min) and forecasts (15-min) for all 6 cities, auto-backfills gaps up to 3 hours, and stores data in the same format as existing backfill tables.

## Context

- **Existing backfill scripts**: `ingest_vc_obs_backfill.py`, `ingest_vc_forecast_snapshot.py`
- **Existing tables**: `wx.vc_minute_weather`, `wx.vc_forecast_daily`, `wx.vc_forecast_hourly`
- **Daemon pattern**: `legacy/poll_vc_forecast_daemon.py` provides signal handling, retry logic
- **VC client**: `src/weather/visual_crossing.py` has all needed methods
- **Rate limit**: User allows 0.3s between calls (very fast), VC has no hard limit

## Requirements

| Requirement | Value |
|-------------|-------|
| Data types | Observations (actual_obs) + Forecasts (forecast) |
| Observation interval | 5 minutes (match existing) |
| Forecast interval | 15 minutes (match existing) |
| Poll frequency | Every 0.3 seconds allowed |
| Cities | All 6 (CHI, AUS, DEN, LAX, MIA, PHL) |
| Gap backfill | Up to 3 hours on startup |
| Deployment | systemd/docker daemon, 24/7 |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    poll_vc_live_daemon.py                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Obs Poller │    │ Fcst Poller │    │ Gap Filler  │     │
│  │  (5-min)    │    │  (15-min)   │    │ (startup)   │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Visual Crossing Client                  │   │
│  │         (0.3s rate limit between calls)             │   │
│  └─────────────────────────────────────────────────────┘   │
│         │                  │                  │             │
│         ▼                  ▼                  ▼             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   Database                           │   │
│  │  wx.vc_minute_weather | wx.vc_forecast_hourly/daily │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Core Daemon Script

**File: `scripts/poll_vc_live_daemon.py`**

```python
# Core structure (pseudocode)

POLL_INTERVAL_SECONDS = 0.3  # Between API calls
OBS_FETCH_INTERVAL_MINUTES = 5  # How often to fetch new obs
FCST_FETCH_INTERVAL_MINUTES = 15  # How often to fetch forecasts
GAP_BACKFILL_HOURS = 3  # Max lookback on startup

class LiveVCDaemon:
    def __init__(self):
        self.vc_client = VisualCrossingClient(rate_limit_delay=POLL_INTERVAL_SECONDS)
        self.last_obs_fetch = {}  # city -> datetime
        self.last_fcst_fetch = {}  # city -> datetime

    def startup_gap_fill(self):
        """Check for gaps up to 3 hours, backfill if needed"""
        for location in get_all_vc_locations():
            last_obs = get_latest_obs_time(location.id)
            if last_obs is None or (now - last_obs) > timedelta(minutes=5):
                # Backfill from max(last_obs, now - 3 hours) to now
                self.backfill_obs(location, start=max(last_obs, now - 3h))

            last_fcst = get_latest_fcst_time(location.id)
            if last_fcst is None or (now - last_fcst) > timedelta(minutes=15):
                self.backfill_fcst(location)

    def run_forever(self):
        """Main loop - cycle through cities, fetch obs and forecasts"""
        self.startup_gap_fill()

        while not shutdown_requested:
            now = datetime.now(timezone.utc)

            for location in get_all_vc_locations():
                # Check if we need new observations
                if self.should_fetch_obs(location, now):
                    self.fetch_and_store_obs(location)
                    time.sleep(POLL_INTERVAL_SECONDS)

                # Check if we need new forecasts
                if self.should_fetch_fcst(location, now):
                    self.fetch_and_store_fcst(location)
                    time.sleep(POLL_INTERVAL_SECONDS)

            # Small sleep before next cycle
            time.sleep(1)
```

### Phase 2: Observation Fetching

**Strategy**: Fetch "today" observations every 5 minutes per city

```python
def fetch_and_store_obs(self, location: VcLocation):
    """Fetch recent observations for one location"""
    # Get today in location's timezone
    local_tz = ZoneInfo(location.iana_timezone)
    local_now = datetime.now(local_tz)
    today_str = local_now.strftime("%Y-%m-%d")

    # Fetch today's observations (VC returns all available minutes)
    if location.location_type == "station":
        data = self.vc_client.fetch_station_history_minutes(
            station_id=location.station_id,
            start_date=today_str,
            end_date=today_str,
            minute_interval=5,
        )
    else:
        data = self.vc_client.fetch_city_history_minutes(
            city_query=location.vc_location_query,
            start_date=today_str,
            end_date=today_str,
            minute_interval=5,
        )

    # Flatten and upsert (deduplicates via ON CONFLICT)
    records = flatten_vc_response_to_minutes(
        data,
        vc_location_id=location.id,
        iana_timezone=location.iana_timezone,
        data_type="actual_obs",
    )

    with get_db_session() as session:
        upsert_vc_minute_weather(session, records)
        session.commit()

    self.last_obs_fetch[location.id] = datetime.now(timezone.utc)
    logger.info(f"[OBS] {location.city_code}: {len(records)} records")
```

### Phase 3: Forecast Fetching

**Strategy**: Fetch current+forecast every 15 minutes per city

```python
def fetch_and_store_fcst(self, location: VcLocation):
    """Fetch current forecast snapshot for one location"""
    basis_date = date.today()
    basis_datetime_utc = datetime.now(timezone.utc)

    # Fetch next 7 days forecast with 15-min granularity
    if location.location_type == "station":
        data = self.vc_client.fetch_station_current_and_forecast(
            station_id=location.station_id,
            horizon_days=7,
            minute_interval=15,
        )
    else:
        data = self.vc_client.fetch_city_current_and_forecast(
            city_query=location.vc_location_query,
            horizon_days=7,
            minute_interval=15,
        )

    # Store in three tables:
    # 1. wx.vc_minute_weather (minute-level forecast)
    # 2. wx.vc_forecast_hourly (hourly aggregates)
    # 3. wx.vc_forecast_daily (daily aggregates)

    with get_db_session() as session:
        # Minutes -> vc_minute_weather
        minute_records = flatten_vc_response_to_minutes(
            data,
            vc_location_id=location.id,
            iana_timezone=location.iana_timezone,
            data_type="forecast",
            forecast_basis_date=basis_date,
            forecast_basis_datetime_utc=basis_datetime_utc,
        )
        upsert_vc_minute_weather(session, minute_records)

        # Hours -> vc_forecast_hourly
        hourly_records = flatten_vc_response_to_hourly(
            data,
            vc_location_id=location.id,
            data_type="forecast",
            forecast_basis_date=basis_date,
        )
        upsert_vc_forecast_hourly(session, hourly_records)

        # Days -> vc_forecast_daily
        daily_records = flatten_vc_response_to_daily(
            data,
            vc_location_id=location.id,
            data_type="forecast",
            forecast_basis_date=basis_date,
        )
        upsert_vc_forecast_daily(session, daily_records)

        session.commit()

    self.last_fcst_fetch[location.id] = datetime.now(timezone.utc)
    logger.info(f"[FCST] {location.city_code}: {len(minute_records)} min, {len(hourly_records)} hr, {len(daily_records)} daily")
```

### Phase 4: Gap Backfill Logic

```python
def startup_gap_fill(self):
    """On startup, check for gaps and backfill up to 3 hours"""
    logger.info("Checking for data gaps (max 3 hours lookback)...")

    cutoff = datetime.now(timezone.utc) - timedelta(hours=GAP_BACKFILL_HOURS)

    with get_db_session() as session:
        for location in get_all_vc_locations(session):
            # Find latest observation for this location
            latest_obs = session.execute(
                select(func.max(VcMinuteWeather.datetime_utc))
                .where(VcMinuteWeather.vc_location_id == location.id)
                .where(VcMinuteWeather.data_type == "actual_obs")
            ).scalar()

            if latest_obs is None:
                logger.warning(f"{location.city_code}: No obs found, skipping gap fill")
                continue

            gap_minutes = (datetime.now(timezone.utc) - latest_obs).total_seconds() / 60

            if gap_minutes > 5:
                # There's a gap - backfill from latest to now
                start_date = max(latest_obs.date(), cutoff.date())
                end_date = date.today()

                logger.info(f"{location.city_code}: Gap of {gap_minutes:.0f}min, backfilling {start_date} to {end_date}")

                self.backfill_obs_range(location, start_date, end_date)
                time.sleep(POLL_INTERVAL_SECONDS)
            else:
                logger.info(f"{location.city_code}: No gap (latest obs {gap_minutes:.0f}min ago)")
```

## Files to Create/Modify

| Action | Path | Description |
|--------|------|-------------|
| CREATE | `scripts/poll_vc_live_daemon.py` | Main daemon script (~400 lines) |
| CREATE | `scripts/systemd/poll-vc-live.service` | systemd unit file |
| MODIFY | `src/weather/visual_crossing.py` | Add helper for "today only" fetch if needed |
| MODIFY | `src/db/__init__.py` | Ensure upsert helpers are exported |

## Daemon CLI Interface

```bash
# Normal run (24/7 daemon)
python scripts/poll_vc_live_daemon.py

# Test mode (one cycle, then exit)
python scripts/poll_vc_live_daemon.py --once

# Specific cities only
python scripts/poll_vc_live_daemon.py --cities chicago denver

# Custom intervals
python scripts/poll_vc_live_daemon.py --obs-interval 5 --fcst-interval 15

# Skip gap fill (faster startup for testing)
python scripts/poll_vc_live_daemon.py --no-gap-fill

# Verbose logging
python scripts/poll_vc_live_daemon.py -v
```

## systemd Service File

```ini
# /etc/systemd/system/poll-vc-live.service
[Unit]
Description=Visual Crossing Live Data Daemon
After=network.target postgresql.service

[Service]
Type=simple
User=halsted
WorkingDirectory=/home/halsted/Python/weather_updated
ExecStart=/home/halsted/Python/weather_updated/.venv/bin/python scripts/poll_vc_live_daemon.py
Restart=always
RestartSec=10
Environment=PYTHONPATH=/home/halsted/Python/weather_updated

[Install]
WantedBy=multi-user.target
```

## Polling Schedule

With 0.3s between calls and 12 locations (6 cities × 2 types):

| Data Type | Interval | Calls/Cycle | Time/Cycle |
|-----------|----------|-------------|------------|
| Observations | 5 min | 12 | 3.6s |
| Forecasts | 15 min | 12 | 3.6s |
| **Total** | - | 24 | ~7.2s |

**Effective polling**:
- Observations: Every 5 minutes per city, complete cycle in <4 seconds
- Forecasts: Every 15 minutes per city, complete cycle in <4 seconds
- Plenty of margin within the 5-min and 15-min windows

## Success Criteria

- [ ] Daemon starts and runs continuously without crashes
- [ ] Observations stored every 5 minutes for all 6 cities (both station + city)
- [ ] Forecasts stored every 15 minutes for all 6 cities
- [ ] Gap backfill works on startup (fills up to 3 hours)
- [ ] Graceful shutdown on SIGTERM/SIGINT
- [ ] Logs show clear status per city
- [ ] `predict_now.py` sees fresh data within 5 minutes

## Testing Plan

1. **Unit test**: Mock VC client, verify upsert logic
2. **Integration test**: Run `--once`, verify DB records created
3. **Gap test**: Stop daemon 30 min, restart, verify backfill
4. **24-hour soak**: Run overnight, check for memory leaks or errors

## Sign-off Log

### 2025-11-28 - Plan Created
**Status**: Ready for implementation

**Requirements confirmed**:
- VC observations (5-min) + forecasts (15-min)
- 0.3s poll rate, same DB format
- All 6 cities, both station + city types
- 24/7 daemon with systemd
- Auto-backfill gaps up to 3 hours

**Next steps**:
1. Create `scripts/poll_vc_live_daemon.py`
2. Test with `--once` flag
3. Set up systemd service
4. Validate with `predict_now.py`
