"""Test that inference produces same features as training.

STRICT: Fails on any missing features.
RELAXED: Warns on high null rates (CatBoost handles nulls natively).

Usage:
    python test_inference_parity.py [city] [date]
    python test_inference_parity.py chicago 2025-12-08
"""
from datetime import date, datetime, time
import pandas as pd

from models.data.loader import load_full_inference_data
from models.features.pipeline import SnapshotContext, compute_snapshot_features
from models.features.base import get_feature_columns
from models.features.calendar import compute_lag_features
from src.db.connection import get_db_session


def test_feature_parity(city: str = "chicago", event_date: date = None):
    """Verify inference features match training schema exactly."""
    if event_date is None:
        # Use yesterday (most likely to have data)
        event_date = date.today() - pd.Timedelta(days=1)

    # Use 14:30 local time as cutoff (naive - DB stores naive local times)
    cutoff_time = datetime.combine(event_date, time(14, 30))

    print(f"Testing inference parity for {city} on {event_date}")
    print(f"Cutoff time: {cutoff_time}")
    print("=" * 60)

    with get_db_session() as session:
        # Load all data (will raise if obs missing)
        try:
            data = load_full_inference_data(city, event_date, cutoff_time, session)
        except ValueError as e:
            print(f"\nERROR: {e}")
            print("\nCheck that data exists for this city/date in the DB.")
            return False

        print(f"\nLoaded data for {city} {event_date}:")
        print(f"  - Observations: {len(data['temps_sofar'])}")
        print(f"  - Window start: {data['window_start']}")
        print(f"  - Forecast daily: {'Yes' if data['fcst_daily'] else 'No'}")
        print(f"  - Forecast hourly: {'Yes' if data['fcst_hourly_df'] is not None else 'No'}")

        # Count multi-horizon forecasts
        multi_count = sum(1 for v in data['fcst_multi'].values() if v) if data['fcst_multi'] else 0
        print(f"  - Multi-horizon: {multi_count}/6")

        print(f"  - Candles: {'Yes' if data['candles_df'] is not None else 'No'}")
        print(f"  - City obs: {'Yes' if data['city_obs_df'] is not None else 'No'}")

        # Check NOAA guidance
        has_noaa = False
        if data['more_apis']:
            for model in ['nbm', 'hrrr', 'ndfd']:
                if data['more_apis'].get(model, {}).get('latest_run'):
                    has_noaa = True
                    break
        print(f"  - NOAA guidance: {'Yes' if has_noaa else 'No'}")
        print(f"  - 30-day stats: mean={data['obs_t15_mean']}, std={data['obs_t15_std']}")

        # Build SnapshotContext
        ctx = SnapshotContext(
            city=city,
            event_date=event_date,
            cutoff_time=cutoff_time,
            window_start=data["window_start"],
            temps_sofar=data["temps_sofar"],
            timestamps_sofar=data["timestamps_sofar"],
            obs_df=data["obs_df"],
            fcst_daily=data["fcst_daily"],
            fcst_hourly_df=data["fcst_hourly_df"],
            fcst_multi=data["fcst_multi"],
            candles_df=data["candles_df"],
            city_obs_df=data["city_obs_df"],
            more_apis=data["more_apis"],
            obs_t15_mean_30d_f=data["obs_t15_mean"],
            obs_t15_std_30d_f=data["obs_t15_std"],
            settle_f=None,  # Inference mode
        )

        # Compute features
        try:
            features = compute_snapshot_features(ctx, include_labels=False)
        except Exception as e:
            print(f"\nERROR computing features: {e}")
            return False

        # Add lag features using existing compute_lag_features()
        lag_df = data["lag_data"]
        if lag_df is not None and not lag_df.empty:
            lag_fs = compute_lag_features(lag_df, city, event_date)
            features.update(lag_fs.to_dict())

            # Compute delta_vcmax_lag1 = today's max so far - yesterday's max
            vc_max_f_lag1 = features.get("vc_max_f_lag1")
            vc_max_f_sofar = features.get("vc_max_f_sofar")
            if vc_max_f_lag1 is not None and vc_max_f_sofar is not None:
                features["delta_vcmax_lag1"] = vc_max_f_sofar - vc_max_f_lag1
            print(f"  - Lag data: {len(lag_df)} days")
        else:
            print(f"  - Lag data: None (no historical settlements)")

        # Get expected schema
        numeric_cols, categorical_cols = get_feature_columns()
        expected = set(numeric_cols + categorical_cols)
        actual = set(features.keys())

        missing = expected - actual
        extra = actual - expected

        print(f"\n{'=' * 60}")
        print("=== Feature Parity Check ===")
        print(f"Expected: {len(expected)} features")
        print(f"Actual: {len(actual)} features")
        print(f"Missing: {len(missing)}")
        print(f"Extra: {len(extra)}")

        # STRICT: Fail on missing
        if missing:
            print(f"\nMISSING FEATURES ({len(missing)}):")
            for f in sorted(missing)[:20]:
                print(f"  - {f}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            print("\nFAILED: Missing features - inference not aligned with training")
            return False

        if extra:
            print(f"\nExtra features ({len(extra)}, will be ignored by model):")
            for f in sorted(extra)[:10]:
                print(f"  - {f}")

        # Check null rates
        df = pd.DataFrame([features])
        present_cols = list(expected & actual)
        null_rates = df[present_cols].isna().mean()
        high_null = null_rates[null_rates > 0.01]
        max_null = null_rates.max()

        print(f"\n{'=' * 60}")
        print("=== Null Rate Check ===")
        print(f"Max null rate: {max_null:.4f}")
        print(f"Columns with >1% nulls: {len(high_null)}")

        # RELAXED: Warn on high nulls (CatBoost handles nulls natively)
        if len(high_null) > 0:
            print(f"\nWARNING: HIGH NULL COLUMNS ({len(high_null)}):")
            for col, rate in sorted(high_null.items(), key=lambda x: -x[1])[:10]:
                print(f"  - {col}: {rate:.4f}")
            if len(high_null) > 10:
                print(f"  ... and {len(high_null) - 10} more")
            print("\nNOTE: CatBoost handles nulls natively. This is expected for meteo features.")
            print("      Training data had similar null patterns.")

        print(f"\n{'=' * 60}")
        print("PASSED: Feature parity test!")
        print(f"  - {len(expected)} features present")
        print(f"  - 0 missing features")
        if len(high_null) > 0:
            print(f"  - {len(high_null)} columns with nulls (expected for meteo/NOAA features)")
        else:
            print(f"  - Max null rate: {max_null:.4f} (<1%)")
        print("=" * 60)
        return True


if __name__ == "__main__":
    import sys

    # Default to chicago, yesterday
    city = sys.argv[1] if len(sys.argv) > 1 else "chicago"
    event_date = None
    if len(sys.argv) > 2:
        event_date = date.fromisoformat(sys.argv[2])

    success = test_feature_parity(city, event_date)
    sys.exit(0 if success else 1)
