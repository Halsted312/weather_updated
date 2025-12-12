"""
Verification script to test feature parity between old and new pipelines.

This script:
1. Loads sample data from the database
2. Builds features using BOTH old and new pipelines
3. Compares feature counts, values, and null patterns
4. Reports any discrepancies

Run with: python scripts/verify_pipeline_parity.py
"""

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_db_session
from models.data.loader import (
    load_vc_observations,
    load_settlements,
    load_historical_forecast_daily,
    load_historical_forecast_hourly,
)

# Old pipeline imports
from models.data.snapshot_builder import (
    build_single_snapshot as old_build_snapshot,
    build_snapshot_for_inference as old_build_inference,
)

# New pipeline imports
from models.data.snapshot import (
    build_snapshot as new_build_snapshot,
    build_snapshot_for_inference as new_build_inference,
)
from models.features.pipeline import SnapshotContext, compute_snapshot_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_features(old_features: dict, new_features: dict, label: str = "") -> dict:
    """Compare two feature dictionaries and return discrepancies."""
    discrepancies = {
        'missing_in_new': [],
        'missing_in_old': [],
        'value_mismatches': [],
        'null_differences': [],
    }

    old_keys = set(old_features.keys())
    new_keys = set(new_features.keys())

    # Check for missing keys
    discrepancies['missing_in_new'] = list(old_keys - new_keys)
    discrepancies['missing_in_old'] = list(new_keys - old_keys)

    # Check values for common keys
    common_keys = old_keys & new_keys
    for key in common_keys:
        old_val = old_features[key]
        new_val = new_features[key]

        # Handle None/NaN
        old_is_null = old_val is None or (isinstance(old_val, float) and np.isnan(old_val))
        new_is_null = new_val is None or (isinstance(new_val, float) and np.isnan(new_val))

        if old_is_null != new_is_null:
            discrepancies['null_differences'].append({
                'key': key,
                'old': old_val,
                'new': new_val,
            })
        elif not old_is_null and not new_is_null:
            # Compare values
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if abs(old_val - new_val) > 1e-6:
                    discrepancies['value_mismatches'].append({
                        'key': key,
                        'old': old_val,
                        'new': new_val,
                        'diff': abs(old_val - new_val),
                    })
            elif old_val != new_val:
                discrepancies['value_mismatches'].append({
                    'key': key,
                    'old': old_val,
                    'new': new_val,
                })

    return discrepancies


def run_parity_test(city: str, event_date: date, session) -> dict:
    """Run parity test for a single city/date combination."""
    logger.info(f"Testing {city} {event_date}...")

    # Load data
    d_minus_1 = event_date - timedelta(days=1)
    obs_df = load_vc_observations(session, city, d_minus_1, event_date)

    if obs_df.empty:
        logger.warning(f"No observations for {city} {event_date}")
        return {'skipped': True, 'reason': 'no_observations'}

    # Get settlement
    settlements = load_settlements(session, city, event_date, event_date)
    if settlements.empty:
        logger.warning(f"No settlement for {city} {event_date}")
        return {'skipped': True, 'reason': 'no_settlement'}

    settle_f = int(settlements.iloc[0]['tmax_final'])

    # Load forecast
    basis_date = event_date - timedelta(days=1)
    fcst_daily = load_historical_forecast_daily(session, city, event_date, basis_date)
    fcst_hourly_df = load_historical_forecast_hourly(session, city, event_date, basis_date)

    # Test multiple cutoff times
    test_hours = [12, 15, 18, 21]
    results = []

    for hour in test_hours:
        cutoff_time = datetime.combine(event_date, datetime.min.time()).replace(hour=hour, minute=0)

        # Filter observations to cutoff
        obs_df_copy = obs_df.copy()
        obs_df_copy['datetime_local'] = pd.to_datetime(obs_df_copy['datetime_local'])
        obs_filtered = obs_df_copy[obs_df_copy['datetime_local'] <= cutoff_time]

        if obs_filtered.empty:
            continue

        # Extract temps and timestamps for old pipeline
        temps_sofar = obs_filtered['temp_f'].dropna().tolist()
        timestamps_sofar = obs_filtered['datetime_local'].tolist()

        if len(temps_sofar) < 5:
            continue

        try:
            # Build features with OLD pipeline
            old_features = old_build_inference(
                city=city,
                day=event_date,
                temps_sofar=temps_sofar,
                timestamps_sofar=timestamps_sofar,
                cutoff_time=cutoff_time,
                snapshot_hour=hour,
                fcst_daily=fcst_daily,
                fcst_hourly_df=fcst_hourly_df if fcst_hourly_df is not None and not fcst_hourly_df.empty else None,
            )
        except Exception as e:
            logger.error(f"Old pipeline failed: {e}")
            old_features = None

        try:
            # Build features with NEW pipeline
            new_features = new_build_inference(
                city=city,
                event_date=event_date,
                cutoff_time=cutoff_time,
                obs_df=obs_filtered,
                fcst_daily=fcst_daily,
                fcst_hourly_df=fcst_hourly_df if fcst_hourly_df is not None and not fcst_hourly_df.empty else None,
            )
        except Exception as e:
            logger.error(f"New pipeline failed: {e}")
            new_features = None

        if old_features is not None and new_features is not None:
            discrepancies = compare_features(old_features, new_features, f"{city}/{event_date}/{hour}:00")
            results.append({
                'hour': hour,
                'old_feature_count': len(old_features),
                'new_feature_count': len(new_features),
                'discrepancies': discrepancies,
            })

    return {'skipped': False, 'results': results}


def main():
    """Main verification routine."""
    logger.info("=" * 60)
    logger.info("Feature Pipeline Parity Verification")
    logger.info("=" * 60)

    # Test parameters
    test_cities = ['chicago', 'austin']
    test_date = date(2024, 7, 15)  # Sample date

    total_tests = 0
    total_mismatches = 0

    with get_db_session() as session:
        for city in test_cities:
            result = run_parity_test(city, test_date, session)

            if result.get('skipped'):
                logger.warning(f"Skipped {city}: {result.get('reason')}")
                continue

            for r in result.get('results', []):
                total_tests += 1
                disc = r['discrepancies']

                has_issues = (
                    len(disc['missing_in_new']) > 0 or
                    len(disc['value_mismatches']) > 0 or
                    len(disc['null_differences']) > 0
                )

                if has_issues:
                    total_mismatches += 1
                    logger.warning(f"\n{city} {test_date} hour={r['hour']}:")
                    logger.warning(f"  Old features: {r['old_feature_count']}")
                    logger.warning(f"  New features: {r['new_feature_count']}")

                    if disc['missing_in_new']:
                        logger.warning(f"  Missing in new: {disc['missing_in_new'][:10]}")
                    if disc['missing_in_old']:
                        logger.info(f"  New features: {disc['missing_in_old'][:10]}")
                    if disc['value_mismatches']:
                        logger.warning(f"  Value mismatches: {len(disc['value_mismatches'])}")
                        for m in disc['value_mismatches'][:5]:
                            logger.warning(f"    {m['key']}: {m['old']} vs {m['new']}")
                else:
                    logger.info(f"✓ {city} {test_date} hour={r['hour']}: {r['new_feature_count']} features match")

    logger.info("=" * 60)
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Tests with mismatches: {total_mismatches}")

    if total_mismatches == 0:
        logger.info("✓ All features match between old and new pipelines!")
        return 0
    else:
        logger.warning("⚠ Some features differ between pipelines")
        return 1


if __name__ == "__main__":
    sys.exit(main())
