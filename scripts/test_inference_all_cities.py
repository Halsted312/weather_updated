#!/usr/bin/env python3
"""
Test inference pipeline for all 6 cities with Ordinal CatBoost models.

Validates:
- Model loading
- Prediction shape (n, 13)
- Probability summation to 1.0
- City-specific delta range handling
"""

import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.training.ordinal_trainer import OrdinalDeltaTrainer

CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']

def test_city_model(city: str) -> dict:
    """Test model loading and inference for a city."""
    model_path = Path(f"models/saved/{city}/ordinal_catboost_optuna.pkl")

    if not model_path.exists():
        return {"city": city, "status": "FAIL", "error": f"Model not found at {model_path}"}

    try:
        # Load model
        start_time = time.time()
        trainer = OrdinalDeltaTrainer()
        trainer.load(model_path)
        load_time = time.time() - start_time

        # Check metadata
        metadata = trainer._metadata
        delta_range = metadata.get('delta_range', 'N/A')
        n_classifiers = len(trainer.classifiers)

        # Load test data
        test_data_path = Path(f"models/saved/{city}/test_data.parquet")
        if test_data_path.exists():
            df_test = pd.read_parquet(test_data_path)

            # Run prediction
            start_time = time.time()
            proba = trainer.predict_proba(df_test.head(10))
            pred_time = (time.time() - start_time) / 10  # Per sample

            # Validate
            checks = {
                "shape_correct": proba.shape == (10, 13),
                "probs_sum_to_1": np.allclose(proba.sum(axis=1), 1.0),
                "no_negatives": (proba >= 0).all(),
                "no_nans": not np.isnan(proba).any(),
            }

            # Check that missing delta classes have zero probability
            if delta_range == [-1, 10]:
                # LA/Miami/Austin/Denver should have P(delta=-2) = 0
                delta_neg2_prob = proba[:, 0].sum()  # Index 0 is delta=-2
                checks["missing_class_zero"] = np.isclose(delta_neg2_prob, 0.0)
            else:
                checks["missing_class_zero"] = True  # Chicago/Philly have all classes

            all_passed = all(checks.values())

            return {
                "city": city,
                "status": "PASS" if all_passed else "WARN",
                "load_time_ms": f"{load_time*1000:.1f}",
                "pred_time_ms": f"{pred_time*1000:.1f}",
                "delta_range": delta_range,
                "n_classifiers": n_classifiers,
                "proba_shape": str(proba.shape),
                "checks": checks,
            }
        else:
            return {
                "city": city,
                "status": "WARN",
                "load_time_ms": f"{load_time*1000:.1f}",
                "delta_range": delta_range,
                "n_classifiers": n_classifiers,
                "error": "No test data available",
            }

    except Exception as e:
        return {
            "city": city,
            "status": "FAIL",
            "error": str(e),
        }


def main():
    print("="*80)
    print("ORDINAL CATBOOST INFERENCE TESTING - ALL 6 CITIES")
    print("="*80)
    print()

    results = []
    for city in CITIES:
        print(f"Testing {city}...")
        result = test_city_model(city)
        results.append(result)

        status_emoji = "✅" if result["status"] == "PASS" else ("⚠️" if result["status"] == "WARN" else "❌")
        print(f"  {status_emoji} {result['status']}")

        if "delta_range" in result:
            print(f"     Delta range: {result['delta_range']}")
            print(f"     Classifiers: {result['n_classifiers']}")
            print(f"     Load time: {result.get('load_time_ms', 'N/A')} ms")
            print(f"     Pred time: {result.get('pred_time_ms', 'N/A')} ms/sample")

        if "checks" in result:
            print(f"     Checks:")
            for check, passed in result["checks"].items():
                check_emoji = "✅" if passed else "❌"
                print(f"       {check_emoji} {check}: {passed}")

        if "error" in result:
            print(f"     Error: {result['error']}")
        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r["status"] == "PASS")
    warned = sum(1 for r in results if r["status"] == "WARN")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print(f"✅ Passed: {passed}/6")
    print(f"⚠️  Warned: {warned}/6")
    print(f"❌ Failed: {failed}/6")
    print()

    if failed == 0 and warned == 0:
        print("🎉 ALL TESTS PASSED! Inference pipeline ready for production.")
    elif failed == 0:
        print("⚠️  All models loaded but some warnings. Review above.")
    else:
        print("❌ Some tests failed. Fix issues before deployment.")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
