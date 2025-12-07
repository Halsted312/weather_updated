#!/usr/bin/env python3
"""Verify Austin augmented parquet has expected NOAA and candle features.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/verify_austin_augmented.py
    PYTHONPATH=. .venv/bin/python scripts/verify_austin_augmented.py --path data/training_cache/austin/full_aug.parquet
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Expected NOAA feature columns
NOAA_FEATURES = [
    "nbm_peak_window_max_f",
    "nbm_peak_window_revision_1h_f",
    "hrrr_peak_window_max_f",
    "hrrr_peak_window_revision_1h_f",
    "ndfd_tmax_T1_f",
    "ndfd_drift_T2_to_T1_f",
    "hrrr_minus_nbm_peak_window_max_f",
    "ndfd_minus_vc_T1_f",
    "nbm_t15_z_30d_f",
    "hrrr_t15_z_30d_f",
    "hrrr_minus_nbm_t15_z_30d_f",
]

# Expected candle micro features
CANDLE_FEATURES = [
    "c_logit_mid_last",
    "c_logit_mom_15m",
    "c_logit_vol_15m",
    "c_logit_surprise_15m",
    "c_spread_pct_mean_15m",
    "c_mid_range_pct_15m",
    "c_trade_frac_15m",
    "c_synth_frac_15m",
]


def verify_parquet(path: Path) -> dict:
    """Verify parquet structure and feature null rates."""
    df = pd.read_parquet(path)

    result = {
        "path": str(path),
        "rows": len(df),
        "columns": len(df.columns),
        "noaa_present": [],
        "noaa_missing": [],
        "noaa_null_rates": {},
        "candle_present": [],
        "candle_missing": [],
        "candle_null_rates": {},
    }

    # Check NOAA features
    for col in NOAA_FEATURES:
        if col in df.columns:
            result["noaa_present"].append(col)
            non_null = df[col].notna().sum()
            result["noaa_null_rates"][col] = 1.0 - (non_null / len(df))
        else:
            result["noaa_missing"].append(col)

    # Check candle features
    for col in CANDLE_FEATURES:
        if col in df.columns:
            result["candle_present"].append(col)
            non_null = df[col].notna().sum()
            result["candle_null_rates"][col] = 1.0 - (non_null / len(df))
        else:
            result["candle_missing"].append(col)

    # Date range
    date_col = 'event_date' if 'event_date' in df.columns else 'day'
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        result["date_range"] = (df[date_col].min(), df[date_col].max())
        result["unique_days"] = df[date_col].nunique()

    return result


def print_report(result: dict):
    """Print verification report."""
    print("\n" + "=" * 70)
    print("AUSTIN PARQUET VERIFICATION REPORT")
    print("=" * 70)

    print(f"\nFile: {result['path']}")
    print(f"Rows: {result['rows']:,}")
    print(f"Columns: {result['columns']}")

    if "date_range" in result:
        print(f"Date range: {result['date_range'][0]} to {result['date_range'][1]}")
        print(f"Unique days: {result['unique_days']}")

    # NOAA Features
    print("\n" + "-" * 50)
    print("NOAA FEATURES")
    print("-" * 50)

    if result["noaa_missing"]:
        print(f"\n  MISSING ({len(result['noaa_missing'])}):")
        for col in result["noaa_missing"]:
            print(f"    - {col}")

    if result["noaa_present"]:
        print(f"\n  PRESENT ({len(result['noaa_present'])}):")
        for col in result["noaa_present"]:
            null_rate = result["noaa_null_rates"][col]
            fill_rate = (1 - null_rate) * 100
            status = "OK" if fill_rate > 50 else "LOW" if fill_rate > 0 else "EMPTY"
            print(f"    {col}: {fill_rate:.1f}% filled [{status}]")

    # Candle Features
    print("\n" + "-" * 50)
    print("CANDLE MICRO FEATURES")
    print("-" * 50)

    if result["candle_missing"]:
        print(f"\n  MISSING ({len(result['candle_missing'])}):")
        for col in result["candle_missing"]:
            print(f"    - {col}")

    if result["candle_present"]:
        print(f"\n  PRESENT ({len(result['candle_present'])}):")
        for col in result["candle_present"]:
            null_rate = result["candle_null_rates"][col]
            fill_rate = (1 - null_rate) * 100
            status = "OK" if fill_rate > 30 else "LOW" if fill_rate > 0 else "EMPTY"
            print(f"    {col}: {fill_rate:.1f}% filled [{status}]")

    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)

    noaa_ok = len(result["noaa_present"])
    noaa_total = len(NOAA_FEATURES)
    candle_ok = len(result["candle_present"])
    candle_total = len(CANDLE_FEATURES)

    print(f"  NOAA features:   {noaa_ok}/{noaa_total}")
    print(f"  Candle features: {candle_ok}/{candle_total}")

    # Expected null features (revision, NDFD)
    expected_null = [
        "nbm_peak_window_revision_1h_f",
        "hrrr_peak_window_revision_1h_f",
        "ndfd_tmax_T1_f",
        "ndfd_drift_T2_to_T1_f",
        "ndfd_minus_vc_T1_f",
    ]
    unexpected_empty = []
    for col in result["noaa_present"]:
        if col not in expected_null and result["noaa_null_rates"].get(col, 0) > 0.5:
            unexpected_empty.append(col)

    if unexpected_empty:
        print(f"\n  WARNING: Unexpectedly low fill rate:")
        for col in unexpected_empty:
            print(f"    - {col}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Verify Austin augmented parquet")
    parser.add_argument(
        "--path",
        type=str,
        default="data/training_cache/austin/full_aug.parquet",
        help="Path to parquet file to verify",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Optional: compare with another parquet (e.g., original)",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        # Try alternate paths
        alternates = [
            Path("data/training_cache/austin/full.parquet"),
            Path("models/saved/austin/train_data_full.parquet"),
        ]
        for alt in alternates:
            if alt.exists():
                logger.info(f"Found alternate: {alt}")
                path = alt
                break
        else:
            return 1

    result = verify_parquet(path)
    print_report(result)

    if args.compare:
        compare_path = Path(args.compare)
        if compare_path.exists():
            print("\n" + "=" * 70)
            print("COMPARISON")
            print("=" * 70)
            compare_result = verify_parquet(compare_path)
            print(f"\nOriginal: {compare_result['rows']:,} rows, {compare_result['columns']} cols")
            print(f"Augmented: {result['rows']:,} rows, {result['columns']} cols")
            print(f"Columns added: {result['columns'] - compare_result['columns']}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
