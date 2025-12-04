Gotcha. Let’s close the loop with:

1. **Code stubs for `ObsFeatureBuilder`** (plus the small DB helpers it needs).
2. **How it pairs with `ForecastFeatureBuilder` and the edge pipeline.**
3. **A concise strategy summary you can drop at the bottom of your Markdown for the agent.**

I’ll stay at the “ready-to-implement” stub level, not fully wired into your exact package layout, so the agent can adapt naming.

---

## 1. DB helpers for observations (parallel to forecast ones)

Put these next to your existing `vc_minute_queries` helpers, e.g. in `db/vc_minute_queries.py`.

```python
# db/vc_minute_queries.py (add these)

from datetime import date
from typing import Literal
import pandas as pd
from sqlalchemy import cast, Date
from sqlalchemy.orm import Session

from models.vc import VcLocation, VcMinuteWeather  # adjust path

LocationType = Literal["station", "city"]


def fetch_obs_minute_df(
    session: Session,
    city_code: str,
    target_date: date,
    location_type: LocationType,
    include_forward_filled: bool = True,
) -> pd.DataFrame:
    """
    Fetch minute-level observations for a given city_code, date, and location_type
    from wx.vc_minute_weather.

    - data_type='actual_obs'
    - DATE(datetime_local) = target_date

    Returns a DataFrame with at least:
      - datetime_local
      - datetime_utc
      - temp_f, humidity, cloudcover, dew_f, precip_in, windspeed_mph, uvindex
    """
    loc = (
        session.query(VcLocation)
        .filter(
            VcLocation.city_code == city_code,
            VcLocation.location_type == location_type,
        )
        .one()
    )

    q = (
        session.query(
            VcMinuteWeather.datetime_local,
            VcMinuteWeather.datetime_utc,
            VcMinuteWeather.temp_f,
            VcMinuteWeather.humidity,
            VcMinuteWeather.cloudcover,
            VcMinuteWeather.dew_f,
            VcMinuteWeather.precip_in,
            VcMinuteWeather.windspeed_mph,
            VcMinuteWeather.uvindex,
            VcMinuteWeather.is_forward_filled,
        )
        .filter(
            VcMinuteWeather.vc_location_id == loc.id,
            VcMinuteWeather.data_type == "actual_obs",
            cast(VcMinuteWeather.datetime_local, Date) == target_date,
        )
        .order_by(VcMinuteWeather.datetime_local)
    )

    rows = q.all()
    df = pd.DataFrame(
        rows,
        columns=[
            "datetime_local",
            "datetime_utc",
            "temp_f",
            "humidity",
            "cloudcover",
            "dew_f",
            "precip_in",
            "windspeed_mph",
            "uvindex",
            "is_forward_filled",
        ],
    )

    if not include_forward_filled and "is_forward_filled" in df.columns:
        df = df[(df["is_forward_filled"] == False) | (df["is_forward_filled"].isna())]

    return df.sort_values("datetime_local").reset_index(drop=True)


def fetch_station_and_city_obs_minutes(
    session: Session,
    city_code: str,
    target_date: date,
    include_forward_filled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience: observations for both station and city as DFs.

    Returns:
      (station_df, city_df)
    """
    station_df = fetch_obs_minute_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="station",
        include_forward_filled=include_forward_filled,
    )
    city_df = fetch_obs_minute_df(
        session=session,
        city_code=city_code,
        target_date=target_date,
        location_type="city",
        include_forward_filled=include_forward_filled,
    )
    return station_df, city_df
```

---

## 2. `ObsFeatureBuilder` – daily & “up-to-snapshot” features

Now a builder class parallel to `ForecastFeatureBuilder`, using:

* `fetch_obs_minute_df` / `fetch_station_and_city_obs_minutes`
* `shape.compute_shape_features` (for realized intraday shape)
* `station_city.compute_station_city_features` (gaps over time)

File: `features/obs_builder.py`

```python
# features/obs_builder.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.vc_minute_queries import (
    fetch_obs_minute_df,
    fetch_station_and_city_obs_minutes,
)
from shape import compute_shape_features
from station_city import compute_station_city_features


@dataclass
class ObsFeatureBuilderConfig:
    """
    Controls what observation-based features we compute.
    """

    include_forward_filled: bool = True
    include_shape_features: bool = True
    include_station_city_gap_features: bool = True

    # For snapshot-specific features: how far back to look
    short_window_minutes: int = 60      # e.g. last hour
    long_window_minutes: int = 180      # e.g. last 3 hours

    default_step_minutes: int = 5       # your obs are ~5-min; we infer if possible


class ObsFeatureBuilder:
    """
    Build observation-based features from wx.vc_minute_weather for a given
    city_code and date.

    Two main uses:
      - Daily features (full-day realized shape, realized high & time)
      - Snapshot features (up to some snapshot_time, for edge classifier)
    """

    def __init__(self, session: Session, config: ObsFeatureBuilderConfig | None = None):
        self.session = session
        self.config = config or ObsFeatureBuilderConfig()

    # ---------- Public APIs ----------

    def build_daily_features(
        self,
        city_code: str,
        target_date: date,
        use_station: bool = True,
    ) -> Dict[str, Any]:
        """
        Daily realized features for the day:
          - realized_max_f, realized_min_f, realized_range_f
          - realized_hour_of_max
          - shape-of-day features (plateau vs spike)
          - optional station vs city gaps (over full day)
        """
        loc_type = "station" if use_station else "city"

        df_obs = fetch_obs_minute_df(
            session=self.session,
            city_code=city_code,
            target_date=target_date,
            location_type=loc_type,
            include_forward_filled=self.config.include_forward_filled,
        )

        if df_obs.empty:
            return {}

        feats: Dict[str, Any] = {}

        # Basic realized summary
        temps = df_obs["temp_f"].astype(float).values
        t_max = float(np.max(temps))
        t_min = float(np.min(temps))
        idx_max = int(np.argmax(temps))
        dt_max = df_obs["datetime_local"].iloc[idx_max]

        feats["obs_realized_max_f"] = t_max
        feats["obs_realized_min_f"] = t_min
        feats["obs_realized_range_f"] = t_max - t_min
        feats["obs_realized_hour_of_max"] = dt_max.hour + dt_max.minute / 60.0

        # Shape-of-day features (full day)
        if self.config.include_shape_features:
            step_minutes = self._infer_step_minutes(df_obs, self.config.default_step_minutes)
            shape_fs = compute_shape_features(
                temps_sofar=temps.tolist(),
                timestamps_local_sofar=df_obs["datetime_local"].tolist(),
                t_base=int(round(t_max)),    # use realized high as t_base
                step_minutes=step_minutes,
            )
            # Prefix to distinguish from forecast-based shape
            for k, v in shape_fs.features.items():
                feats[f"obs_{k}"] = v

        # Station vs city gap features over the *full day*
        if self.config.include_station_city_gap_features:
            gap_feats = self._build_station_city_gap_daily(city_code, target_date)
            feats.update(gap_feats)

        return feats

    def build_snapshot_features(
        self,
        city_code: str,
        target_date: date,
        snapshot_time_local: datetime,
        use_station: bool = True,
    ) -> Dict[str, Any]:
        """
        Features "as of" a given snapshot_time within the day, for edge classifier.

        Examples:
          - realized_high_so_far, diff vs forecast high (computed upstream)
          - short/long window temperature changes
          - short/long window humidity / cloud changes
        """
        loc_type = "station" if use_station else "city"

        df_obs = fetch_obs_minute_df(
            session=self.session,
            city_code=city_code,
            target_date=target_date,
            location_type=loc_type,
            include_forward_filled=self.config.include_forward_filled,
        )

        if df_obs.empty:
            return {}

        # Filter to up-to-snapshot
        df_up_to = df_obs[df_obs["datetime_local"] <= snapshot_time_local].copy()
        if df_up_to.empty:
            return {}

        feats: Dict[str, Any] = {}

        temps = df_up_to["temp_f"].astype(float).values
        t_max_sofar = float(np.max(temps))
        t_min_sofar = float(np.min(temps))
        idx_max = int(np.argmax(temps))
        dt_max = df_up_to["datetime_local"].iloc[idx_max]

        feats["obs_sofar_max_f"] = t_max_sofar
        feats["obs_sofar_min_f"] = t_min_sofar
        feats["obs_sofar_range_f"] = t_max_sofar - t_min_sofar
        feats["obs_sofar_hour_of_max"] = dt_max.hour + dt_max.minute / 60.0

        # Short/long window changes
        feats.update(
            self._compute_window_changes(df_up_to, snapshot_time_local)
        )

        return feats

    # ---------- Internal helpers ----------

    @staticmethod
    def _infer_step_minutes(df: pd.DataFrame, default_step_minutes: int) -> int:
        if df.empty or len(df) < 2:
            return default_step_minutes
        diffs = df["datetime_local"].diff().dropna().dt.total_seconds() / 60.0
        step = int(np.median(diffs))
        return step if step > 0 else default_step_minutes

    def _compute_window_changes(
        self,
        df_up_to: pd.DataFrame,
        snapshot_time_local: datetime,
    ) -> Dict[str, Any]:
        """
        Compute short- and long-window changes for temp/humidity/cloudcover/etc.
        """
        feats: Dict[str, Any] = {}

        df = df_up_to.set_index("datetime_local").sort_index()

        for window_minutes, label in [
            (self.config.short_window_minutes, "short"),
            (self.config.long_window_minutes, "long"),
        ]:
            window_start = snapshot_time_local - pd.Timedelta(minutes=window_minutes)
            df_win = df[df.index >= window_start]

            if df_win.empty:
                continue

            # Take first and last within window
            first = df_win.iloc[0]
            last = df_win.iloc[-1]

            feats[f"obs_temp_change_{label}"] = float(last["temp_f"] - first["temp_f"])
            feats[f"obs_humidity_change_{label}"] = float(
                (last["humidity"] or 0) - (first["humidity"] or 0)
            )
            feats[f"obs_cloudcover_change_{label}"] = float(
                (last["cloudcover"] or 0) - (first["cloudcover"] or 0)
            )
            feats[f"obs_precip_change_{label}"] = float(
                (last["precip_in"] or 0) - (first["precip_in"] or 0)
            )

        return feats

    def _build_station_city_gap_daily(
        self,
        city_code: str,
        target_date: date,
    ) -> Dict[str, Any]:
        """
        Use station_city.compute_station_city_features on full-day obs series
        for station vs city.
        """
        station_df, city_df = fetch_station_and_city_obs_minutes(
            session=self.session,
            city_code=city_code,
            target_date=target_date,
            include_forward_filled=self.config.include_forward_filled,
        )

        if station_df.empty or city_df.empty:
            return {}

        station_series = list(
            zip(
                station_df["datetime_local"].tolist(),
                station_df["temp_f"].astype(float).tolist(),
            )
        )
        city_series = list(
            zip(
                city_df["datetime_local"].tolist(),
                city_df["temp_f"].astype(float).tolist(),
            )
        )

        fs_gap = compute_station_city_features(station_series, city_series)
        # Prefix with 'obs_' to differentiate from forecast-based gaps
        return {f"obs_{k}": v for k, v in fs_gap.features.items()}
```

---

## 3. How everything ties together (strategy in one place)

Here’s a compact strategy statement you can paste at the end of your Markdown for the coding agent:

### Overall strategy summary

We’re standardizing the Kalshi weather pipeline into three clean layers:

1. **Data → DB**

   * Use Visual Crossing **Timeline + Historical Forecast** APIs to populate:

     * `wx.vc_minute_weather` with:

       * `data_type='actual_obs'` for minute-level obs.
       * `data_type='historical_forecast'` for minute-level (15-min) T-0..T-6 forecasts using `include=minutes` and `forecastBasisDay/Date`.
     * `wx.vc_forecast_daily` / `wx.vc_forecast_hourly` (already in place) for daily/hourly snapshots.
   * `wx.vc_location` holds both:

     * VC **station** queries (e.g. `stn:KMDW`)
     * VC **city** queries (e.g. `"Chicago,IL"`), keyed by `city_code` + `location_type`.

2. **DB → feature-ready DataFrames**

   * `db/vc_minute_queries.py` is the single source of truth for pulling minute-level series:

     * `fetch_tminus1_minute_forecast_df(session, city_code, target_date, location_type)`
     * `fetch_station_and_city_tminus1_minute_forecasts(...)`
     * `fetch_obs_minute_df(session, city_code, target_date, location_type)`
     * `fetch_station_and_city_obs_minutes(...)`
   * These helpers return **pandas DataFrames** with `datetime_local`, `temp_f`, `humidity`, `cloudcover`, etc., so downstream code never touches raw SQL.

3. **Feature builders (Python)**

   * **ForecastFeatureBuilder** (`features/forecast_builder.py`):

     * Pulls T-1 15-min **historical forecast** series (city + optionally station).
     * Uses existing modules to compute:

       * `forecast_static` features (prev forecast max/min/mean, `t_forecast_base`, `fcst_prev_hour_of_max`, etc.).
       * `shape.py` on the forecast curve (plateau vs spike of the *forecasted* day) with features prefixed `fcst_`.
       * `station_city.py` on forecast series for station vs city (gap stats over the day).
     * Returns a flat dict of forecast-based features for `(city_code, target_date)`.

   * **ObsFeatureBuilder** (`features/obs_builder.py`):

     * Pulls minute-level **observations** for station or city.
     * Computes:

       * Realized daily high/low/range, `obs_realized_hour_of_max`.
       * Observed shape-of-day features via `shape.py` (prefixed `obs_`).
       * Station vs city gaps over the realized day (via `station_city`), also prefixed `obs_`.
       * Snapshot-specific features (for edge classifier) based on obs **up to** a given `snapshot_time_local`:

         * `obs_sofar_max_f`, `obs_sofar_range_f`, short/long-window changes in temp/humidity/cloudcover/precip.

4. **Model layers**

   * **Ordinal high-temperature predictor**:

     * Uses `ForecastFeatureBuilder.build_for_day(...)` + `ObsFeatureBuilder.build_daily_features(...)` + calendar/seasonal features.
     * Learns to predict the **actual daily high**, bracket probabilities, and `t_forecast_base` using basis-aligned forecast information.
   * **Edge classifier**:

     * Takes a snapshot in time and sees:

       * Ordinal model outputs (probabilities, implied temp, `t_forecast_base`).
       * Market state (Kalshi prices/spreads/volume).
       * Snapshot `ObsFeatureBuilder.build_snapshot_features(...)`.
       * Select forecast features (e.g., `fcst_prev_max_f`, `fcst_prev_hour_of_max`, forecast shape flags) as context.
     * Decides whether the current mispricing is a real edge (trading decision).

This architecture keeps a clear separation:

* **Forecast vs Obs vs Market**
* **City vs Station**
* **Day-level predictor vs intraday edge**

and ensures everything you backtest is based only on information that actually existed at the time (basis-aligned forecasts + realized obs), while still letting you use 15-minute resolution for both physics and market microstructure.

---

If you hand the agent:

* The last few responses
* The DB helper stubs
* `ForecastFeatureBuilder` + `ObsFeatureBuilder` stubs
* And this strategy summary

you’ve given them a very solid, coherent spec to implement against.
