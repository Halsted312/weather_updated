#!/usr/bin/env python3
"""
Calibration tooling for cross-bracket models.

Fits Platt scaling and Isotonic regression on validation predictions and
evaluates both on a hold-out test set. Outputs metrics plus the calibration
parameters to a JSON file for later use.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

LOGGER = logging.getLogger("calibrate_probs")


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")


def parse_args() -> argparse.Namespace:
    configure_logging()
    parser = argparse.ArgumentParser(description="Fit Platt and Isotonic calibrators")
    parser.add_argument("--city", required=True)
    parser.add_argument("--model", required=True, help="Model identifier (e.g., logreg, gbdt)")
    parser.add_argument("--horizon-min", type=int, choices=[1, 5], required=True)
    parser.add_argument("--val-file", required=True, help="CSV from train_cross_bracket --export-val")
    parser.add_argument("--test-file", required=True, help="CSV from train_cross_bracket --export-test")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default results/calibration_<city>_<model>_<horizon>.json)",
    )
    return parser.parse_args()


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        raise ValueError(f"{path} is missing y_true/y_prob columns")
    return df["y_true"].to_numpy(), df["y_prob"].to_numpy()


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    total = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += abs(bin_conf - bin_acc) * (mask.sum() / total)
    return float(ece)


@dataclass
class CalibrationResult:
    method: str
    brier: float
    ece: float
    params: Dict[str, object]


def fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs")
    model.fit(y_prob.reshape(-1, 1), y_true)
    return model


def apply_platt(model: LogisticRegression, y_prob: np.ndarray) -> np.ndarray:
    return model.predict_proba(y_prob.reshape(-1, 1))[:, 1]


def fit_isotonic(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso


def evaluate_calibrator(
    method: str,
    calibrator,
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    y_test_true: np.ndarray,
    y_test_prob: np.ndarray,
) -> CalibrationResult:
    if method == "platt":
        y_test_cal = apply_platt(calibrator, y_test_prob)
        params = {
            "coef": calibrator.coef_.ravel().tolist(),
            "intercept": calibrator.intercept_.ravel().tolist(),
        }
    elif method == "isotonic":
        y_test_cal = calibrator.predict(y_test_prob)
        params = {
            "x_thresholds": calibrator.X_thresholds_.tolist(),
            "y_thresholds": calibrator.y_thresholds_.tolist(),
        }
    else:
        raise ValueError(f"Unknown method {method}")

    brier = brier_score_loss(y_test_true, y_test_cal)
    ece = expected_calibration_error(y_test_true, y_test_cal)
    LOGGER.info("%s: Brier=%.4f ECE=%.4f", method, brier, ece)
    return CalibrationResult(method=method, brier=brier, ece=ece, params=params)


def main() -> None:
    args = parse_args()
    y_val_true, y_val_prob = load_predictions(args.val_file)
    y_test_true, y_test_prob = load_predictions(args.test_file)

    calibrators: Dict[str, CalibrationResult] = {}

    try:
        platt_model = fit_platt(y_val_true, y_val_prob)
        calibrators["platt"] = evaluate_calibrator(
            "platt", platt_model, y_val_true, y_val_prob, y_test_true, y_test_prob
        )
    except Exception as exc:  # pylint:disable=broad-except
        LOGGER.warning("Platt scaling failed: %s", exc)

    try:
        iso_model = fit_isotonic(y_val_true, y_val_prob)
        calibrators["isotonic"] = evaluate_calibrator(
            "isotonic", iso_model, y_val_true, y_val_prob, y_test_true, y_test_prob
        )
    except Exception as exc:  # pylint:disable=broad-except
        LOGGER.warning("Isotonic regression failed: %s", exc)

    if not calibrators:
        raise RuntimeError("No calibrator fitted successfully")

    best = min(calibrators.values(), key=lambda res: res.brier)
    LOGGER.info("Selected %s calibrator (Brier=%.4f, ECE=%.4f)", best.method, best.brier, best.ece)

    payload = {
        "city": args.city,
        "model": args.model,
        "horizon_min": args.horizon_min,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "best_method": best.method,
        "metrics": {method: {"brier": res.brier, "ece": res.ece} for method, res in calibrators.items()},
        "params": {method: res.params for method, res in calibrators.items()},
    }

    output_path = args.output
    if not output_path:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"calibration_{args.city}_{args.model}_{args.horizon_min}m.json"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Wrote calibration params to %s", output_path)


if __name__ == "__main__":
    main()
