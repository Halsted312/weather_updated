"""
Evaluation harness for temperature rounding rules.

Runs all rules over historical data and computes:
- Accuracy metrics per rule
- Mismatch records for inspection
- Comparative performance analysis
"""

import csv
import logging
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from analysis.temperature.datastructures import DaySeries, RuleStats
from analysis.temperature.loader import load_day_series
from analysis.temperature.rules import ALL_RULES

logger = logging.getLogger(__name__)


def evaluate_rules(
    session: Session,
    city: str,
    start_day: date,
    end_day: date,
    verbose: bool = False,
) -> Tuple[Dict[str, RuleStats], List[dict]]:
    """Run all temperature rounding rules over a date range.

    For each day with both settlement and VC data:
    1. Apply all rules to the VC 5-minute series
    2. Compare predictions to actual settlement
    3. Update RuleStats for each rule
    4. Record mismatches for later inspection

    Args:
        session: SQLAlchemy session
        city: City identifier
        start_day: First day to evaluate (inclusive)
        end_day: Last day to evaluate (inclusive)
        verbose: Log progress every N days if True

    Returns:
        (stats_dict, mismatches_list)
        - stats_dict: RuleStats for each rule
        - mismatches_list: Records where baseline rule != settlement
    """
    # Initialize stats for each rule
    stats: Dict[str, RuleStats] = {name: RuleStats(name=name) for name in ALL_RULES}

    # Track mismatches for inspection
    mismatches: List[dict] = []

    # Count days processed
    days_processed = 0
    days_skipped = 0

    logger.info(
        f"Evaluating rules for {city} from {start_day} to {end_day}"
    )

    # Iterate over date range
    current = start_day
    while current <= end_day:
        # Load day series (settlement + VC temps)
        day_series = load_day_series(session, city, current)

        if day_series is None:
            days_skipped += 1
            current += date.resolution
            continue

        days_processed += 1

        # Apply all rules
        predictions = {}
        for rule_name, rule_fn in ALL_RULES.items():
            pred = rule_fn(day_series.temps_f)
            predictions[rule_name] = pred

            # Update stats
            stats[rule_name].update(pred, day_series.settle_f)

        # Record mismatch if baseline rule disagrees
        # Use max_of_rounded as baseline (most likely correct rule)
        baseline_pred = predictions.get("max_of_rounded")

        if baseline_pred is not None and baseline_pred != day_series.settle_f:
            mismatch_record = {
                "city": city,
                "day": current.isoformat(),
                "settle_f": day_series.settle_f,
                "vc_max_f": day_series.vc_max_f,
                "num_samples": day_series.num_samples,
                "baseline_rule": "max_of_rounded",
                "baseline_pred": baseline_pred,
                "error": baseline_pred - day_series.settle_f,
            }

            # Add predictions from all rules for comparison
            for rule_name, pred in predictions.items():
                mismatch_record[f"pred_{rule_name}"] = pred if pred is not None else "None"

            mismatches.append(mismatch_record)

        # Log progress
        if verbose and days_processed % 100 == 0:
            logger.info(f"  Processed {days_processed} days...")

        current += date.resolution

    logger.info(
        f"Evaluation complete: {days_processed} days processed, "
        f"{days_skipped} days skipped (missing data)"
    )

    return stats, mismatches


def evaluate_multiple_cities(
    session: Session,
    cities: List[str],
    start_day: date,
    end_day: date,
    verbose: bool = False,
) -> Dict[str, Tuple[Dict[str, RuleStats], List[dict]]]:
    """Run evaluation for multiple cities.

    Args:
        session: SQLAlchemy session
        cities: List of city identifiers
        start_day: First day (inclusive)
        end_day: Last day (inclusive)
        verbose: Log progress if True

    Returns:
        Dict mapping city → (stats, mismatches)
    """
    results = {}

    for city in cities:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {city.upper()}")
        logger.info(f"{'='*60}")

        stats, mismatches = evaluate_rules(
            session, city, start_day, end_day, verbose=verbose
        )

        results[city] = (stats, mismatches)

    return results


def print_stats_summary(stats: Dict[str, RuleStats]) -> None:
    """Print formatted summary of rule performance.

    Args:
        stats: Dictionary of rule_name → RuleStats
    """
    print("\n=== Temperature Rounding Rule Performance ===\n")

    # Sort by accuracy (descending)
    sorted_stats = sorted(stats.items(), key=lambda kv: kv[1].accuracy, reverse=True)

    print(f"{'Rule':<20} {'Accuracy':<10} {'MAE':<10} {'Exact':<12} {'Off±1':<8} {'Off≥2':<8}")
    print("=" * 75)

    for rule_name, st in sorted_stats:
        if st.total == 0:
            continue

        print(
            f"{rule_name:<20} "
            f"{st.accuracy:>8.2%} "
            f"{st.mae:>8.3f}°F "
            f"{st.exact_matches:>4d}/{st.total:<4d} "
            f"{st.off_by_1:>6d} "
            f"{st.off_by_2plus:>6d}"
        )

    print("=" * 75)


def write_mismatches_csv(filepath: str, mismatches: List[dict]) -> None:
    """Write mismatch records to CSV for inspection.

    Args:
        filepath: Output CSV path
        mismatches: List of mismatch dictionaries

    Creates parent directories if needed.
    """
    if not mismatches:
        print(f"No mismatches to write for {filepath}")
        return

    # Create parent directory
    parent = Path(filepath).parent
    parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    fieldnames = list(mismatches[0].keys())

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mismatches)

    print(f"Wrote {len(mismatches)} mismatch records to {filepath}")
