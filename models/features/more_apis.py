"""
NOAA model guidance features (NBM, HRRR, NDFD).

Exposes 8 features from weather_more_apis guidance data:
- NBM level + revision
- HRRR level + revision
- NDFD level + drift
- Disagreement metrics (HRRR vs NBM, NDFD vs VC)

All features gracefully handle missing data (return None).
"""

import logging
from typing import Optional, Dict, Any

from models.features.base import FeatureSet

logger = logging.getLogger(__name__)


def compute_more_apis_features(
    more_apis: Optional[Dict[str, Dict[str, Any]]] = None,
    vc_t1_tempmax: Optional[float] = None,
    obs_t15_mean_30d_f: Optional[float] = None,
    obs_t15_std_30d_f: Optional[float] = None,
) -> FeatureSet:
    """
    Compute NOAA model guidance features from weather_more_apis data.

    Expected structure of more_apis:
    {
        "nbm": {
            "latest_run": {"peak_window_max_f": float, "run_datetime_utc": datetime},
            "prev_run": {"peak_window_max_f": float, "run_datetime_utc": datetime} | None
        },
        "hrrr": { ... },
        "ndfd": { ... }
    }

    Args:
        more_apis: Nested dict of guidance data (from loader)
        vc_t1_tempmax: VC T-1 daily forecast tempmax for disagreement metric

    Returns:
        FeatureSet with 8 numeric features
    """
    if more_apis is None:
        return FeatureSet(name="more_apis", features=_null_features())

    # Extract model data
    nbm = more_apis.get("nbm", {})
    hrrr = more_apis.get("hrrr", {})
    ndfd = more_apis.get("ndfd", {})

    nbm_latest = nbm.get("latest_run")
    nbm_prev = nbm.get("prev_run")
    hrrr_latest = hrrr.get("latest_run")
    hrrr_prev = hrrr.get("prev_run")
    ndfd_latest = ndfd.get("latest_run")
    ndfd_prev = ndfd.get("prev_run")

    # Compute raw temp features
    nbm_t15 = _safe_get(nbm_latest, "peak_window_max_f")
    hrrr_t15 = _safe_get(hrrr_latest, "peak_window_max_f")

    # Compute z-score features (dimensionless)
    nbm_z = _compute_z(nbm_t15, obs_t15_mean_30d_f, obs_t15_std_30d_f)
    hrrr_z = _compute_z(hrrr_t15, obs_t15_mean_30d_f, obs_t15_std_30d_f)
    hrrr_minus_nbm_z = _compute_disagreement(hrrr_z, nbm_z)

    features = {
        # Raw temp features (existing)
        "nbm_peak_window_max_f": nbm_t15,
        "nbm_peak_window_revision_1h_f": _compute_revision(nbm_latest, nbm_prev),
        "hrrr_peak_window_max_f": hrrr_t15,
        "hrrr_peak_window_revision_1h_f": _compute_revision(hrrr_latest, hrrr_prev),
        "ndfd_tmax_T1_f": _safe_get(ndfd_latest, "peak_window_max_f"),
        "ndfd_drift_T2_to_T1_f": _compute_drift(ndfd_prev, ndfd_latest),
        "hrrr_minus_nbm_peak_window_max_f": _compute_disagreement(hrrr_t15, nbm_t15),
        "ndfd_minus_vc_T1_f": _compute_disagreement(
            _safe_get(ndfd_latest, "peak_window_max_f"), vc_t1_tempmax
        ),
        # NEW: Dimensionless z-score features at 15:00 local
        "nbm_t15_z_30d_f": nbm_z,
        "hrrr_t15_z_30d_f": hrrr_z,
        "hrrr_minus_nbm_t15_z_30d_f": hrrr_minus_nbm_z,
    }

    return FeatureSet(name="more_apis", features=features)


# =============================================================================
# Helper Functions
# =============================================================================


def _null_features() -> Dict[str, None]:
    """Return all features as None (when data not loaded)."""
    return {
        # Raw temp features
        "nbm_peak_window_max_f": None,
        "nbm_peak_window_revision_1h_f": None,
        "hrrr_peak_window_max_f": None,
        "hrrr_peak_window_revision_1h_f": None,
        "ndfd_tmax_T1_f": None,
        "ndfd_drift_T2_to_T1_f": None,
        "hrrr_minus_nbm_peak_window_max_f": None,
        "ndfd_minus_vc_T1_f": None,
        # Z-score features
        "nbm_t15_z_30d_f": None,
        "hrrr_t15_z_30d_f": None,
        "hrrr_minus_nbm_t15_z_30d_f": None,
    }


def _safe_get(run_dict: Optional[Dict], key: str) -> Optional[float]:
    """Safely extract value from run dict."""
    if run_dict is None:
        return None
    return run_dict.get(key)


def _compute_revision(
    latest: Optional[Dict], prev: Optional[Dict]
) -> Optional[float]:
    """
    Compute revision: latest - previous.

    Positive revision = forecast warming trend.
    Returns None if either run is missing.
    """
    if latest is None or prev is None:
        return None

    latest_temp = latest.get("peak_window_max_f")
    prev_temp = prev.get("peak_window_max_f")

    if latest_temp is None or prev_temp is None:
        return None

    return round(latest_temp - prev_temp, 2)


def _compute_drift(prev: Optional[Dict], latest: Optional[Dict]) -> Optional[float]:
    """
    Compute drift: previous - latest (opposite sign of revision).

    For NDFD: T-2 run - T-1 run (how much forecast changed between cycles).
    Positive drift = forecast cooled between runs.
    """
    if prev is None or latest is None:
        return None

    prev_temp = prev.get("peak_window_max_f")
    latest_temp = latest.get("peak_window_max_f")

    if prev_temp is None or latest_temp is None:
        return None

    return round(prev_temp - latest_temp, 2)


def _compute_disagreement(temp1: Optional[float], temp2: Optional[float]) -> Optional[float]:
    """
    Compute disagreement: temp1 - temp2.

    Larger absolute value = more forecast uncertainty.
    Returns None if either is missing.
    """
    if temp1 is None or temp2 is None:
        return None

    return round(temp1 - temp2, 2)


def _compute_z(temp: Optional[float], mean: Optional[float], std: Optional[float]) -> Optional[float]:
    """
    Compute z-score: (temp - mean) / std.

    Dimensionless measure of how many standard deviations temp is from mean.

    Args:
        temp: Temperature value
        mean: Mean of baseline distribution
        std: Standard deviation of baseline

    Returns:
        Z-score (dimensionless), or None if any input missing or std=0
    """
    if temp is None or mean is None or std is None:
        return None
    if std == 0 or std < 0.01:  # Avoid division by zero/near-zero
        return None

    return round((temp - mean) / std, 3)
