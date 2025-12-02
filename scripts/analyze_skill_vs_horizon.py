#!/usr/bin/env python3
"""
Skill vs Horizon Analysis

Professor's Point (A): Build a unified view of model skill across time horizons.

For each hours_to_event_close bucket:
- Accuracy, Within-1, Within-2, MAE
- Std of raw delta (before clipping)
- Which model was used (Market-Clock vs TOD v1)

This validates the "distribution narrows as we approach close" story.

Usage:
    python scripts/analyze_skill_vs_horizon.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Professor's recommended bucket boundaries
BUCKET_EDGES = [0, 2, 6, 12, 18, 24, 40]  # hours to close
BUCKET_LABELS = ["0-2h", "2-6h", "6-12h", "12-18h", "18-24h", "24h+"]


def assign_bucket(hours_to_close: float) -> str:
    """Assign a time-to-close value to a bucket."""
    for i, (low, high) in enumerate(zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])):
        if low <= hours_to_close < high:
            return BUCKET_LABELS[i]
    return BUCKET_LABELS[-1]  # Catch overflow


def compute_bucket_metrics(
    df: pd.DataFrame,
    delta_col: str = "delta",
    raw_delta_col: str = "delta_raw",
    pred_col: str = "pred_delta",
) -> pd.DataFrame:
    """
    Compute metrics for each hours_to_event_close bucket.

    Returns DataFrame with columns:
        bucket, n_samples, accuracy, within_1, within_2, mae, raw_delta_std, raw_delta_mean
    """
    # Assign buckets
    df = df.copy()
    df["bucket"] = df["hours_to_event_close"].apply(assign_bucket)

    results = []
    for bucket in BUCKET_LABELS:
        bucket_df = df[df["bucket"] == bucket]
        n = len(bucket_df)

        if n == 0:
            results.append({
                "bucket": bucket,
                "n_samples": 0,
                "accuracy": None,
                "within_1": None,
                "within_2": None,
                "mae": None,
                "raw_delta_std": None,
                "raw_delta_mean": None,
                "clip_low_pct": None,
                "clip_high_pct": None,
            })
            continue

        # True delta (clipped)
        y_true = bucket_df[delta_col].values

        # Predicted delta
        if pred_col in bucket_df.columns:
            y_pred = bucket_df[pred_col].values
        else:
            y_pred = np.zeros_like(y_true)  # Placeholder if no predictions

        # Accuracy metrics
        accuracy = np.mean(y_true == y_pred) if pred_col in bucket_df.columns else None
        within_1 = np.mean(np.abs(y_true - y_pred) <= 1) if pred_col in bucket_df.columns else None
        within_2 = np.mean(np.abs(y_true - y_pred) <= 2) if pred_col in bucket_df.columns else None
        mae = np.mean(np.abs(y_true - y_pred)) if pred_col in bucket_df.columns else None

        # Raw delta statistics (before clipping) - for point (C)
        if raw_delta_col in bucket_df.columns:
            raw_delta = bucket_df[raw_delta_col].values
            raw_delta_std = np.std(raw_delta)
            raw_delta_mean = np.mean(raw_delta)
            # Clipping analysis
            clip_low_pct = np.mean(raw_delta < -2) * 100
            clip_high_pct = np.mean(raw_delta > 10) * 100
        else:
            # Use clipped delta as approximation
            raw_delta_std = np.std(y_true)
            raw_delta_mean = np.mean(y_true)
            clip_low_pct = None
            clip_high_pct = None

        results.append({
            "bucket": bucket,
            "n_samples": n,
            "accuracy": accuracy,
            "within_1": within_1,
            "within_2": within_2,
            "mae": mae,
            "raw_delta_std": raw_delta_std,
            "raw_delta_mean": raw_delta_mean,
            "clip_low_pct": clip_low_pct,
            "clip_high_pct": clip_high_pct,
        })

    return pd.DataFrame(results)


def analyze_market_clock() -> pd.DataFrame:
    """Analyze Market-Clock test data by horizon."""
    test_path = Path("models/saved/market_clock_tod_v1/test_data.parquet")
    if not test_path.exists():
        logger.warning(f"Market-Clock test data not found: {test_path}")
        return pd.DataFrame()

    logger.info("\n" + "=" * 60)
    logger.info("MARKET-CLOCK MODEL - Skill vs Horizon")
    logger.info("=" * 60)

    df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(df):,} test samples")
    logger.info(f"hours_to_event_close range: {df['hours_to_event_close'].min():.1f} - {df['hours_to_event_close'].max():.1f}")

    # Load model and get predictions
    try:
        import pickle
        model_path = Path("models/saved/market_clock_tod_v1/ordinal_catboost_market_clock_tod_v1.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Get feature columns
        feature_cols_path = Path("models/saved/market_clock_tod_v1/feature_columns.txt")
        if feature_cols_path.exists():
            feature_cols = feature_cols_path.read_text().strip().split("\n")
        else:
            # Fallback: get from model
            from models.features.base import get_feature_columns_for_market_clock
            feature_cols = get_feature_columns_for_market_clock()

        # Filter to columns that exist
        feature_cols = [c for c in feature_cols if c in df.columns]
        X_test = df[feature_cols]

        # Predict
        df["pred_delta"] = model.predict(X_test)
        logger.info(f"Generated predictions using {len(feature_cols)} features")

    except Exception as e:
        logger.warning(f"Could not load model for predictions: {e}")
        df["pred_delta"] = df["delta"]  # Use actual as fallback

    # Compute metrics by bucket
    metrics = compute_bucket_metrics(df, pred_col="pred_delta")
    metrics["model"] = "Market-Clock"

    return metrics


def analyze_tod_v1() -> pd.DataFrame:
    """Analyze TOD v1 Chicago test data by horizon."""
    test_path = Path("models/saved/chicago_tod_v1/test_data.parquet")
    if not test_path.exists():
        logger.warning(f"TOD v1 test data not found: {test_path}")
        return pd.DataFrame()

    logger.info("\n" + "=" * 60)
    logger.info("TOD v1 CHICAGO MODEL - Skill vs Horizon")
    logger.info("=" * 60)

    df = pd.read_parquet(test_path)
    logger.info(f"Loaded {len(df):,} test samples")

    # Compute hours_to_event_close from snapshot_hour
    # TOD v1 is event-day only, market close = 23:55 (23.917 hours)
    MARKET_CLOSE_HOUR = 23 + 55/60  # 23.917
    df["hours_to_event_close"] = MARKET_CLOSE_HOUR - df["snapshot_hour"]
    df["hours_to_event_close"] = df["hours_to_event_close"].clip(lower=0)

    logger.info(f"hours_to_event_close range: {df['hours_to_event_close'].min():.1f} - {df['hours_to_event_close'].max():.1f}")

    # Load model and get predictions
    try:
        import pickle
        model_path = Path("models/saved/chicago_tod_v1/ordinal_catboost_tod_v1.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Get feature columns from model or config
        from models.features.base import NUMERIC_FEATURE_COLS, CATEGORICAL_FEATURE_COLS
        feature_cols = [c for c in NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS if c in df.columns]
        X_test = df[feature_cols]

        # Predict
        df["pred_delta"] = model.predict(X_test)
        logger.info(f"Generated predictions using {len(feature_cols)} features")

    except Exception as e:
        logger.warning(f"Could not load model for predictions: {e}")
        df["pred_delta"] = df["delta"]  # Use actual as fallback

    # Compute metrics by bucket (TOD v1 only covers event day, so 0-14h range)
    metrics = compute_bucket_metrics(df, pred_col="pred_delta")
    metrics["model"] = "TOD v1"

    return metrics


def analyze_delta_range_coverage():
    """
    Professor's Point (C): Check if [-2, +10] delta range is wide enough.

    For each bucket, compute:
    - Fraction clipped at -2
    - Fraction clipped at +10
    """
    logger.info("\n" + "=" * 60)
    logger.info("DELTA RANGE COVERAGE ANALYSIS - Point (C)")
    logger.info("=" * 60)

    # Load Market-Clock test data (has delta_raw)
    test_path = Path("models/saved/market_clock_tod_v1/test_data.parquet")
    if not test_path.exists():
        logger.warning("Market-Clock test data not found")
        return

    df = pd.read_parquet(test_path)

    if "delta_raw" not in df.columns:
        logger.warning("delta_raw column not found - cannot analyze clipping")
        return

    # Assign buckets
    df["bucket"] = df["hours_to_event_close"].apply(assign_bucket)

    logger.info("\nDelta Clipping by Time Horizon:")
    logger.info("-" * 70)
    logger.info(f"{'Bucket':<10} {'N':>8} {'Mean':>8} {'Std':>8} {'<-2':>8} {'>+10':>8} {'Clipped':>10}")
    logger.info("-" * 70)

    for bucket in BUCKET_LABELS:
        bucket_df = df[df["bucket"] == bucket]
        n = len(bucket_df)
        if n == 0:
            continue

        raw_delta = bucket_df["delta_raw"].values
        mean_d = np.mean(raw_delta)
        std_d = np.std(raw_delta)
        pct_low = np.mean(raw_delta < -2) * 100
        pct_high = np.mean(raw_delta > 10) * 100
        pct_clipped = pct_low + pct_high

        logger.info(f"{bucket:<10} {n:>8,} {mean_d:>+8.2f} {std_d:>8.2f} {pct_low:>7.1f}% {pct_high:>7.1f}% {pct_clipped:>9.1f}%")

    # Overall
    raw_delta = df["delta_raw"].values
    n = len(df)
    mean_d = np.mean(raw_delta)
    std_d = np.std(raw_delta)
    pct_low = np.mean(raw_delta < -2) * 100
    pct_high = np.mean(raw_delta > 10) * 100
    pct_clipped = pct_low + pct_high

    logger.info("-" * 70)
    logger.info(f"{'OVERALL':<10} {n:>8,} {mean_d:>+8.2f} {std_d:>8.2f} {pct_low:>7.1f}% {pct_high:>7.1f}% {pct_clipped:>9.1f}%")

    # Recommendation
    if pct_clipped > 5:
        logger.info(f"\n⚠️  WARNING: {pct_clipped:.1f}% of samples are clipped!")
        logger.info("   Consider widening delta range from [-2, +10] to [-6, +10] or wider.")
    else:
        logger.info(f"\n✓ Delta range [-2, +10] covers {100-pct_clipped:.1f}% of samples - acceptable.")


def print_combined_table(mc_metrics: pd.DataFrame, tod_metrics: pd.DataFrame):
    """Print combined skill vs horizon table."""
    logger.info("\n" + "=" * 80)
    logger.info("COMBINED SKILL VS HORIZON VIEW")
    logger.info("=" * 80)

    # Merge on bucket
    combined = pd.merge(
        mc_metrics[["bucket", "n_samples", "accuracy", "within_1", "within_2", "mae", "raw_delta_std"]],
        tod_metrics[["bucket", "n_samples", "accuracy", "within_1", "within_2", "mae", "raw_delta_std"]],
        on="bucket",
        how="outer",
        suffixes=("_mc", "_tod")
    )

    logger.info("\n" + "-" * 100)
    logger.info(f"{'Bucket':<10} | {'--- Market-Clock ---':^35} | {'--- TOD v1 Chicago ---':^35}")
    logger.info(f"{'':10} | {'N':>7} {'Acc':>7} {'W-2':>7} {'MAE':>6} {'Std':>6} | {'N':>7} {'Acc':>7} {'W-2':>7} {'MAE':>6} {'Std':>6}")
    logger.info("-" * 100)

    for _, row in combined.iterrows():
        bucket = row["bucket"]

        # Market-Clock
        n_mc = row.get("n_samples_mc", 0) or 0
        acc_mc = f"{row['accuracy_mc']*100:.1f}%" if pd.notna(row.get("accuracy_mc")) else "-"
        w2_mc = f"{row['within_2_mc']*100:.1f}%" if pd.notna(row.get("within_2_mc")) else "-"
        mae_mc = f"{row['mae_mc']:.2f}" if pd.notna(row.get("mae_mc")) else "-"
        std_mc = f"{row['raw_delta_std_mc']:.2f}" if pd.notna(row.get("raw_delta_std_mc")) else "-"

        # TOD v1
        n_tod = row.get("n_samples_tod", 0) or 0
        acc_tod = f"{row['accuracy_tod']*100:.1f}%" if pd.notna(row.get("accuracy_tod")) else "-"
        w2_tod = f"{row['within_2_tod']*100:.1f}%" if pd.notna(row.get("within_2_tod")) else "-"
        mae_tod = f"{row['mae_tod']:.2f}" if pd.notna(row.get("mae_tod")) else "-"
        std_tod = f"{row['raw_delta_std_tod']:.2f}" if pd.notna(row.get("raw_delta_std_tod")) else "-"

        logger.info(f"{bucket:<10} | {n_mc:>7,} {acc_mc:>7} {w2_mc:>7} {mae_mc:>6} {std_mc:>6} | {n_tod:>7,} {acc_tod:>7} {w2_tod:>7} {mae_tod:>6} {std_tod:>6}")

    logger.info("-" * 100)

    # Derive risk schedule from std values
    logger.info("\n" + "=" * 60)
    logger.info("DERIVED RISK SCHEDULE - Point (B)")
    logger.info("=" * 60)

    # Use Market-Clock std values (covers full range)
    mc_valid = mc_metrics[mc_metrics["raw_delta_std"].notna()]
    if len(mc_valid) > 0:
        # Find variance for each bucket
        logger.info("\nVariance-based position sizing:")
        logger.info("-" * 50)
        logger.info(f"{'Bucket':<10} {'Std':>8} {'Var':>8} {'Size Mult':>12} {'Edge Mult':>12}")
        logger.info("-" * 50)

        # Baseline: 0-2h bucket (lowest std)
        baseline_row = mc_valid[mc_valid["bucket"] == "0-2h"]
        if len(baseline_row) > 0:
            baseline_std = baseline_row["raw_delta_std"].values[0]
            baseline_var = baseline_std ** 2
        else:
            baseline_std = 1.0
            baseline_var = 1.0

        for _, row in mc_valid.iterrows():
            bucket = row["bucket"]
            std = row["raw_delta_std"]
            var = std ** 2

            # Size multiplier: inverse variance relative to baseline
            size_mult = baseline_var / var
            size_mult = min(1.0, max(0.05, size_mult))  # Cap at 1.0, floor at 5%

            # Edge multiplier: require more edge when uncertain
            edge_mult = std / baseline_std
            edge_mult = max(1.0, min(3.0, edge_mult))  # Cap at 3x, floor at 1x

            logger.info(f"{bucket:<10} {std:>8.2f} {var:>8.2f} {size_mult:>11.2f}x {edge_mult:>11.2f}x")

        logger.info("-" * 50)
        logger.info("\nRecommended HorizonRiskConfig:")
        logger.info("""
@dataclass
class HorizonRiskConfig:
    bucket_edges: list[float] = field(default_factory=lambda: [2.0, 6.0, 12.0, 18.0])
    size_multipliers: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.2, 0.1, 0.05])
    edge_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.2, 1.5, 2.0, 2.5])
""")


def main():
    """Run the full skill vs horizon analysis."""
    # Analyze Market-Clock
    mc_metrics = analyze_market_clock()

    if len(mc_metrics) > 0:
        logger.info("\nMarket-Clock Metrics by Bucket:")
        logger.info(mc_metrics.to_string(index=False))

    # Analyze TOD v1
    tod_metrics = analyze_tod_v1()

    if len(tod_metrics) > 0:
        logger.info("\nTOD v1 Metrics by Bucket:")
        logger.info(tod_metrics.to_string(index=False))

    # Combined view
    if len(mc_metrics) > 0 and len(tod_metrics) > 0:
        print_combined_table(mc_metrics, tod_metrics)
    elif len(mc_metrics) > 0:
        print_combined_table(mc_metrics, pd.DataFrame())

    # Delta range coverage (Point C)
    analyze_delta_range_coverage()

    logger.info("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
