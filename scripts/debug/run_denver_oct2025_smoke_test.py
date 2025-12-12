#!/usr/bin/env python3
"""Denver Oct 2025 Smoke Test - End-to-End Pipeline Validation.

This script runs the complete parquet-only ML pipeline on a small dataset
(Denver, Oct 2025, 2 Optuna trials) to validate correctness before scaling.

It's your **golden path** for iterating on changes - fast, simple, fail-fast.

Stages:
1. Parquet health check (all required files exist)
2. Build dataset (with --start/--end filtering)
3. Run pytest validation
4. Train ordinal model (2 trials)
5. Train edge classifier (2 trials)
6. Summary report

Usage:
    python scripts/run_denver_oct2025_smoke_test.py
    python scripts/run_denver_oct2025_smoke_test.py --skip-dataset-build  # use existing
    python scripts/run_denver_oct2025_smoke_test.py --skip-tests  # for quick iteration
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse

CITY = "denver"
START_DATE = "2025-10-01"
END_DATE = "2025-10-31"
TRIALS = 2


def run_command(description: str, cmd: list, capture_output: bool = False):
    """Run command and fail-fast on error."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
    else:
        result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        print(f"   Exit code: {result.returncode}")
        print(f"\nPipeline aborted. Fix errors before continuing.")
        sys.exit(1)

    print(f"\n✅ PASSED: {description}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Denver Oct 2025 smoke test (parquet-only pipeline)"
    )
    parser.add_argument(
        "--skip-dataset-build",
        action="store_true",
        help="Skip dataset building (use existing parquets)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest validation",
    )
    parser.add_argument(
        "--skip-ordinal",
        action="store_true",
        help="Skip ordinal model training (test edge cache reuse)",
    )
    args = parser.parse_args()

    start_time = datetime.now()

    print("\n" + "="*70)
    print("🚀 DENVER OCT 2025 SMOKE TEST")
    print("="*70)
    print(f"City: {CITY}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Optuna trials: {TRIALS}")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # =========================================================================
    # STEP 1: PARQUET HEALTH CHECK
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: PARQUET HEALTH CHECK")
    print("="*70)

    required_parquets = [
        f"models/raw_data/{CITY}/vc_observations.parquet",
        f"models/raw_data/{CITY}/vc_city_observations.parquet",
        f"models/raw_data/{CITY}/settlements.parquet",
        f"models/raw_data/{CITY}/forecasts_daily.parquet",
        f"models/raw_data/{CITY}/forecasts_hourly.parquet",
        f"models/raw_data/{CITY}/noaa_guidance.parquet",
        f"models/candles/candles_{CITY}.parquet",
    ]

    all_exist = True
    for pq in required_parquets:
        if not Path(pq).exists():
            print(f"❌ MISSING: {pq}")
            all_exist = False
        else:
            size_mb = Path(pq).stat().st_size / (1024**2)
            print(f"✅ {pq} ({size_mb:.1f} MB)")

    if not all_exist:
        print("\n" + "="*70)
        print("❌ PARQUET FILES MISSING")
        print("="*70)
        print("\nRun extraction first:")
        print(f"  python scripts/extract_raw_data_to_parquet.py --city {CITY}")
        print(f"  python scripts/extract_raw_data_to_parquet.py --city {CITY} --start-date {START_DATE} --end-date {END_DATE}")
        sys.exit(1)

    print("\n✅ All parquet files present")

    # =========================================================================
    # STEP 2: BUILD DATASET
    # =========================================================================
    if not args.skip_dataset_build:
        run_command(
            f"Build dataset ({CITY}, {START_DATE} to {END_DATE})",
            [
                "python", "scripts/build_dataset_from_parquets.py",
                "--city", CITY,
                "--start", START_DATE,
                "--end", END_DATE,
                "--workers", "14",
            ]
        )
    else:
        print("\n" + "="*70)
        print("⏭️  SKIPPED: Dataset building (--skip-dataset-build)")
        print("="*70)
        print("Using existing train/test parquets")

        # Verify they exist
        train_path = Path(f"models/saved/{CITY}/train_data_full.parquet")
        test_path = Path(f"models/saved/{CITY}/test_data_full.parquet")

        if not train_path.exists() or not test_path.exists():
            print(f"\n❌ ERROR: Cached parquets not found!")
            print(f"   Train: {train_path} (exists: {train_path.exists()})")
            print(f"   Test: {test_path} (exists: {test_path.exists()})")
            print(f"\nRun without --skip-dataset-build to build datasets first.")
            sys.exit(1)

    # =========================================================================
    # STEP 3: RUN VALIDATION TESTS
    # =========================================================================
    if not args.skip_tests:
        run_command(
            "Run pytest validation",
            ["pytest", "tests/test_denver_oct2025.py", "-v", "--tb=short"]
        )
    else:
        print("\n" + "="*70)
        print("⏭️  SKIPPED: Tests (--skip-tests)")
        print("="*70)

    # =========================================================================
    # STEP 4: TRAIN ORDINAL MODEL
    # =========================================================================
    if not args.skip_ordinal:
        run_command(
            f"Train ordinal model ({TRIALS} trials)",
            [
                "python", "scripts/train_city_ordinal_optuna.py",
                "--city", CITY,
                "--use-cached",
                "--start-date", START_DATE,
                "--end-date", END_DATE,
                "--trials", str(TRIALS),
            ]
        )
    else:
        print("\n" + "="*70)
        print("⏭️  SKIPPED: Ordinal training (--skip-ordinal)")
        print("="*70)
        print("Using existing ordinal model")

        # Verify ordinal model exists
        ordinal_path = Path(f"models/saved/{CITY}/ordinal_catboost_optuna.pkl")
        if not ordinal_path.exists():
            print(f"\n❌ ERROR: Ordinal model not found: {ordinal_path}")
            print(f"Run without --skip-ordinal to train the model first.")
            sys.exit(1)

    # =========================================================================
    # STEP 5: TRAIN EDGE CLASSIFIER
    # =========================================================================
    run_command(
        f"Train edge classifier ({TRIALS} trials)",
        [
            "python", "scripts/train_edge_classifier.py",
            "--city", CITY,
            "--trials", str(TRIALS),
            "--threshold", "1.5",
            "--sample-rate", "4",
        ]
    )

    # =========================================================================
    # STEP 6: SUMMARY
    # =========================================================================
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*70)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"City: {CITY}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Trials per model: {TRIALS}")
    print(f"Total time: {duration}")
    print(f"\nModel outputs:")
    print(f"  models/saved/{CITY}/ordinal_catboost_optuna.pkl")
    print(f"  models/saved/{CITY}/edge_classifier.pkl")
    print(f"  models/saved/{CITY}/train_data_full.parquet")
    print(f"  models/saved/{CITY}/test_data_full.parquet")
    print(f"  models/saved/{CITY}/edge_training_data_realistic.parquet")
    print(f"  models/saved/{CITY}/edge_training_data_realistic.meta.json")

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review model metrics:")
    print(f"   cat models/saved/{CITY}/final_metrics_{CITY}.json")
    print("\n2. Scale to full date range:")
    print(f"   python scripts/run_denver_oct2025_smoke_test.py  # (without --start/--end)")
    print("\n3. Scale to all cities:")
    print(f"   python scripts/run_multi_city_pipeline.py --cities denver los_angeles miami philadelphia")


if __name__ == "__main__":
    main()
