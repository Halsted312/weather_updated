Nice, this is the fun “glue” layer. Let’s give you a clean `ForecastFeatureBuilder` that:

* Takes a SQLAlchemy `Session`
* Uses the `vc_minute_queries` helpers we just sketched
* Calls your existing feature modules:

  * `forecast.compute_forecast_static_features`
  * `shape.compute_shape_features`
  * `station_city.compute_station_city_features`
* Returns a single `dict[str, float]` for a `(city_code, target_date)` pair

You / the agent can then drop this into your ordinal trainer and edge pipeline.

Below is a stub you can put in something like:

```text
src/features/forecast_builder.py
```

(Adjust module paths to match your repo.)

---

## 1. ForecastFeatureBuilder code stub

```python
# src/features/forecast_builder.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

# These imports assume the structure we discussed; adjust to match your repo.
from db.vc_minute_queries import (
    fetch_tminus1_minute_forecast_df,
    fetch_station_and_city_tminus1_minute_forecasts,
)
from forecast import compute_forecast_static_features
from shape import compute_shape_features
from station_city import compute_station_city_features


@dataclass
class ForecastFeatureBuilderConfig:
    """
    Configuration knobs for how we build forecast-based features.
    """

    # Whether to drop forward-filled minutes from the forecast series
    include_forward_filled: bool = True

    # Whether to compute forecast shape-of-day features (plateau/spike) from the
    # T-1 15-min forecast curve
    include_forecast_shape_features: bool = True

    # Whether to compute station vs city gap features from the T-1 forecast
    include_station_city_gap_features: bool = True

    # Default step size (minutes) if we can't infer it
    default_step_minutes: int = 15


class ForecastFeatureBuilder:
    """
    Build a unified feature dict for the high-temperature predictor
    (and optionally edge module) for a given city_code and target_date.

    This class sits on top of:
      - wx.vc_minute_weather (minute-level historical forecasts)
      - wx.vc_location (station vs city entries)
      - forecast.py (T-1 forecast static features)
      - shape.py (shape-of-day features)
      - station_city.py (station vs city gap features)

    Typical usage:

        builder = ForecastFeatureBuilder(session)
        feat_dict = builder.build_for_day(city_code="AUS", target_date=date(2025, 12, 3))
    """

    def __init__(self, session: Session, config: ForecastFeatureBuilderConfig | None = None):
        self.session = session
        self.config = config or ForecastFeatureBuilderConfig()

    # ---------- Public API ----------

    def build_for_day(self, city_code: str, target_date: date) -> Dict[str, Any]:
        """
        Build all relevant forecast-based features for a single day.

        Returns:
            A flat dict[str, Any] with keys like:
                - fcst_prev_max_f
                - fcst_prev_hour_of_max
                - t_forecast_base
                - fcst_shape_minutes_ge_base
                - station_city_temp_gap (if enabled)
                - ...etc.
        """
        feature_dict: Dict[str, Any] = {}

        # 1) T-1 15-min historical forecast for the *city* aggregate
        df_fcst_city = fetch_tminus1_minute_forecast_df(
            session=self.session,
            city_code=city_code,
            target_date=target_date,
            location_type="city",
            include_forward_filled=self.config.include_forward_filled,
        )

        if df_fcst_city.empty:
            # No forecast data: return an empty or mostly-null feature dict.
            # You can choose to log/raise here instead, depending on how strict you want to be.
            return feature_dict

        # Ensure sorted by time
        df_fcst_city = df_fcst_city.sort_values("datetime_local").reset_index(drop=True)

        # Infer step size (minutes) between samples (15, 10, 5, etc.)
        step_minutes = self._infer_step_minutes(df_fcst_city, self.config.default_step_minutes)

        # Extract forecast temperature series
        fcst_temps = df_fcst_city["temp_f"].astype(float).tolist()
        timestamps_local = df_fcst_city["datetime_local"].tolist()

        # 2) Static T-1 forecast features (existing module)
        # NOTE: compute_forecast_static_features currently only takes fcst_series.
        #       It treats indices as “hours”. If you want more precise hour_of_max
        #       for 15-min data, consider extending that function with a
        #       step_minutes argument, or we can add separate derived features here.
        fcst_static_fs = compute_forecast_static_features(fcst_temps)
        feature_dict.update(fcst_static_fs.features)  # assuming FeatureSet has .features dict

        # Grab t_forecast_base from the static features (used by shape)
        t_forecast_base = feature_dict.get("t_forecast_base")
        if t_forecast_base is None and fcst_temps:
            # Fallback: simple rounding of forecast max
            t_forecast_base = int(round(max(fcst_temps)))
            feature_dict["t_forecast_base"] = t_forecast_base

        # 3) Optional shape-of-day features from forecast curve
        if self.config.include_forecast_shape_features and t_forecast_base is not None:
            shape_fs = compute_forecast_shape_features_from_series(
                temps=fcst_temps,
                timestamps_local=timestamps_local,
                t_base=t_forecast_base,
                step_minutes=step_minutes,
            )
            feature_dict.update(shape_fs)  # already a dict

        # 4) Optional station vs city forecast gap features
        if self.config.include_station_city_gap_features:
            station_city_fs = self._compute_station_city_forecast_gap_features(
                city_code=city_code,
                target_date=target_date,
                include_forward_filled=self.config.include_forward_filled,
            )
            feature_dict.update(station_city_fs)

        return feature_dict

    # ---------- Internal helpers ----------

    @staticmethod
    def _infer_step_minutes(df: pd.DataFrame, default_step_minutes: int) -> int:
        """Infer the step size (minutes) from datetime_local, with a safe fallback."""
        if df.empty or "datetime_local" not in df.columns or len(df) < 2:
            return default_step_minutes

        diffs = df["datetime_local"].diff().dropna().dt.total_seconds() / 60.0
        # Use median to be robust to occasional irregularities
        step = int(np.median(diffs))
        if step <= 0:
            return default_step_minutes
        return step

    def _compute_station_city_forecast_gap_features(
        self,
        city_code: str,
        target_date: date,
        include_forward_filled: bool,
    ) -> Dict[str, Any]:
        """
        Build station vs city gap features from T-1 15-min historical forecast.

        Uses station_city.compute_station_city_features on (datetime, temp) pairs
        derived from the minute-level forecast series.
        """
        station_df, city_df = fetch_station_and_city_tminus1_minute_forecasts(
            session=self.session,
            city_code=city_code,
            target_date=target_date,
            include_forward_filled=include_forward_filled,
        )

        if station_df.empty or city_df.empty:
            return {}

        # Convert to lists of (datetime_local, temp_f)
        station_temps = list(
            zip(station_df["datetime_local"].tolist(), station_df["temp_f"].astype(float).tolist())
        )
        city_temps = list(
            zip(city_df["datetime_local"].tolist(), city_df["temp_f"].astype(float).tolist())
        )

        fs_gap = compute_station_city_features(station_temps, city_temps)
        return fs_gap.features  # again assuming FeatureSet has .features


# ---------- Small helper for forecast shape (using shape.py) ----------

def compute_forecast_shape_features_from_series(
    temps: list[float],
    timestamps_local: list,
    t_base: int,
    step_minutes: int,
) -> Dict[str, Any]:
    """
    Convenience wrapper to reuse shape.compute_shape_features for the T-1 forecast curve.

    This treats the 15-min forecast series as a "pseudo-observation" curve to
    characterize whether the *forecasted* day is spike-like or plateau-like.

    Returns:
        A flat dict of shape-related features with 'fcst_' prefix to distinguish
        from observed shape features.
    """
    if not temps or not timestamps_local:
        return {}

    shape_fs = compute_shape_features(
        temps_sofar=temps,
        timestamps_local_sofar=timestamps_local,
        t_base=t_base,
        step_minutes=step_minutes,
    )

    # Prefix forecast shape features so they don't collide with obs-based shape
    prefixed: Dict[str, Any] = {}
    for key, value in shape_fs.features.items():
        prefixed[f"fcst_{key}"] = value

    return prefixed
```

### Notes / TODOs for the agent

* **`compute_forecast_static_features` & `step_minutes`**

  Right now it only takes `fcst_series`. If you want `fcst_prev_hour_of_max` to be a **true hour-of-day** for 15-min data instead of an index, you can:

  * Extend it to accept `step_minutes` and set:

    ```python
    hour_of_max_index = int(np.argmax(arr))
    hour_of_max = hour_of_max_index * step_minutes / 60.0
    ```
  * Or, compute an extra derived feature in `ForecastFeatureBuilder` (e.g. `fcst_prev_hour_of_max_exact`) using the inferred `step_minutes` and the index of `max(temps)`.

* **FeatureSet interface**

  I assumed `FeatureSet` instances from `forecast`, `shape`, and `station_city` all expose a `.features` dict. If they expose `.to_dict()` instead, just change:

  ```python
  feature_dict.update(fcst_static_fs.features)
  ```

  to:

  ```python
  feature_dict.update(fcst_static_fs.to_dict())
  ```

* **Forward-filled data**

  For historical forecasts you may or may not have `is_forward_filled` populated yet (depends on how you ingest). The builder has a flag to exclude forward-filled minutes if you decide they’re too noisy.

---

## 2. How this fits into your pipeline

With this class in place, your ordinal trainer can do something like:

```python
from datetime import date
from sqlalchemy.orm import Session
from features.forecast_builder import ForecastFeatureBuilder

def build_training_row(session: Session, city_code: str, target_date: date, y_value: float):
    builder = ForecastFeatureBuilder(session)
    fcst_feats = builder.build_for_day(city_code=city_code, target_date=target_date)

    # merge with other feature groups (obs, calendar, station gaps, etc.)
    row = {
        "city_code": city_code,
        "target_date": target_date,
        "y": y_value,
        **fcst_feats,
        # **other_feature_groups...
    }
    return row
```

And later you can also pass selected pieces (like `t_forecast_base`, `fcst_prev_max_f`, `fcst_shape_*`, station-city gaps) into the edge classifier as inputs.

If you want, the next step could be a matching `ObsFeatureBuilder` for the same city/day, so you have a clean split:

* `ForecastFeatureBuilder` → what VC thought would happen (basis-aligned)
* `ObsFeatureBuilder` → what actually happened up to snapshot time
* `EdgeFeatureBuilder` (you basically already have) → deltas vs market.
