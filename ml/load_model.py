#!/usr/bin/env python3
"""
Model loader for real-time inference with walk-forward models.

Finds and loads the appropriate WF model for a given city/bracket/date.
Handles edge cases (before first window, after last window).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple, Dict
import logging
import joblib
import re

logger = logging.getLogger(__name__)

# Default model directory (can be overridden)
DEFAULT_MODEL_DIR = Path("models/trained")


class ModelNotFoundError(Exception):
    """Raised when no suitable model can be found for the given parameters."""
    pass


def parse_window_name(window_name: str) -> Optional[Tuple[date, date]]:
    """
    Parse WF window directory name to extract start and end dates.

    Window naming convention: win_YYYYMMDD_YYYYMMDD
    Example: win_20250802_20250919 -> (2025-08-02, 2025-09-19)

    Args:
        window_name: Directory name like "win_20250802_20250919"

    Returns:
        Tuple of (start_date, end_date) or None if parsing fails
    """
    pattern = r'win_(\d{8})_(\d{8})'
    match = re.match(pattern, window_name)

    if not match:
        return None

    start_str, end_str = match.groups()

    try:
        start_date = datetime.strptime(start_str, '%Y%m%d').date()
        end_date = datetime.strptime(end_str, '%Y%m%d').date()
        return (start_date, end_date)
    except ValueError:
        return None


def find_window_for_date(
    model_dir: Path,
    city: str,
    bracket: str,
    target_date: date
) -> Optional[Path]:
    """
    Find the walk-forward window directory containing the target date.

    Walks through models/trained/{city}/{bracket}/win_*/ directories,
    parses window dates, and finds the window whose test period contains target_date.

    Args:
        model_dir: Root model directory (e.g., models/trained)
        city: City name (e.g., "chicago")
        bracket: Bracket type ("between", "greater", "less")
        target_date: Target date for inference

    Returns:
        Path to window directory, or None if not found
    """
    bracket_dir = model_dir / city / bracket

    if not bracket_dir.exists():
        logger.warning(f"Bracket directory not found: {bracket_dir}")
        return None

    # Find all win_* directories
    window_dirs = sorted([
        d for d in bracket_dir.iterdir()
        if d.is_dir() and d.name.startswith('win_')
    ])

    if not window_dirs:
        logger.warning(f"No window directories found in {bracket_dir}")
        return None

    # Parse window dates and compute test intervals
    # Convention: Walk-forward uses 90-day train → 7-day test
    # Test period = [end_date - 6 days, end_date] (7 days total)
    TEST_DAYS = 7

    windows_with_intervals = []
    for win_dir in window_dirs:
        dates = parse_window_name(win_dir.name)
        if dates:
            train_start, test_end = dates
            # Compute test interval: 7 days ending on test_end
            test_start = test_end - timedelta(days=TEST_DAYS - 1)
            windows_with_intervals.append((win_dir, test_start, test_end))

    # Sort by test end date
    windows_with_intervals.sort(key=lambda x: x[2])

    # Find window where target_date falls within test interval [test_start, test_end]
    for win_dir, test_start, test_end in windows_with_intervals:
        if test_start <= target_date <= test_end:
            logger.info(f"Matched window: {win_dir.name} (test: {test_start} to {test_end}, target: {target_date})")
            return win_dir

    # Edge cases: before first window or after last window
    if target_date < windows_with_intervals[0][1]:
        # Before first test window → use first window
        first_window = windows_with_intervals[0][0]
        logger.warning(f"Target date {target_date} before all test windows, using first: {first_window.name}")
        return first_window
    else:
        # After last test window → use last window
        last_window = windows_with_intervals[-1][0]
        logger.warning(f"Target date {target_date} after all test windows, using last: {last_window.name}")
        return last_window


def load_model_for_date(
    city: str,
    bracket: str,
    target_date: date,
    model_dir: Path = DEFAULT_MODEL_DIR
) -> Tuple[object, str, Dict]:
    """
    Load the appropriate WF model for a given city/bracket/date.

    Finds the WF window containing target_date and loads the model from that window.
    Handles edge cases:
    - Date before first window → use first window model
    - Date after last window → use last window model
    - Date within window range → use matching window

    Args:
        city: City name (e.g., "chicago")
        bracket: Bracket type ("between", "greater", "less")
        target_date: Target date for inference (can be date or datetime)
        model_dir: Root model directory (default: models/trained)

    Returns:
        Tuple of (model, window_name, metadata_dict)
        - model: Loaded CalibratedClassifierCV or Pipeline object
        - window_name: WF window name (e.g., "win_20250802_20250919")
        - metadata: Dict with model_path, start_date, end_date

    Raises:
        ModelNotFoundError: If no suitable model can be found
    """
    # Convert datetime to date if needed
    if isinstance(target_date, datetime):
        target_date = target_date.date()

    # Find appropriate window
    window_dir = find_window_for_date(model_dir, city, bracket, target_date)

    if window_dir is None:
        raise ModelNotFoundError(
            f"No model found for {city}/{bracket} on {target_date}"
        )

    # Find model file in window directory
    model_files = list(window_dir.glob('model_*.pkl'))

    if not model_files:
        raise ModelNotFoundError(
            f"No model file found in {window_dir}"
        )

    if len(model_files) > 1:
        logger.warning(f"Multiple model files in {window_dir}, using first: {model_files[0]}")

    model_path = model_files[0]

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    # Parse window metadata
    dates = parse_window_name(window_dir.name)
    if dates:
        start_date, end_date = dates
    else:
        start_date, end_date = None, None

    metadata = {
        'model_path': str(model_path),
        'window_name': window_dir.name,
        'start_date': start_date,
        'end_date': end_date,
        'city': city,
        'bracket': bracket
    }

    return model, window_dir.name, metadata


def load_model_for_now(
    city: str,
    bracket: str,
    model_dir: Path = DEFAULT_MODEL_DIR
) -> Tuple[object, str, Dict]:
    """
    Load the appropriate WF model for the current date/time.

    Convenience wrapper around load_model_for_date() using datetime.now().

    Args:
        city: City name (e.g., "chicago")
        bracket: Bracket type ("between", "greater", "less")
        model_dir: Root model directory (default: models/trained)

    Returns:
        Tuple of (model, window_name, metadata_dict)

    Raises:
        ModelNotFoundError: If no suitable model can be found
    """
    return load_model_for_date(city, bracket, datetime.now().date(), model_dir)


def main():
    """
    Demo: Load model for Chicago/between on a specific date.
    """
    print("\n" + "="*60)
    print("Model Loader Demo")
    print("="*60 + "\n")

    # Example: Load model for Chicago/between on 2025-09-15
    target_date = date(2025, 9, 15)

    # Test with pilot models
    # Pilot structure: models/pilots/chicago/between_elasticnet/chicago/between/win_*/
    # The model_dir should point to the level where city/bracket subdirs live
    # For pilots: since structure is already .../chicago/between/win_*, pass the parent
    pilot_bracket_dir = Path("models/pilots/chicago/between_elasticnet/chicago/between")

    if pilot_bracket_dir.exists():
        print(f"Loading from pilot directory (direct bracket path): {pilot_bracket_dir}")

        # Find windows directly in this bracket directory
        window_dirs = sorted([
            d for d in pilot_bracket_dir.iterdir()
            if d.is_dir() and d.name.startswith('win_')
        ])

        if window_dirs:
            # Find appropriate window for target date
            for win_dir in window_dirs:
                dates = parse_window_name(win_dir.name)
                if dates and target_date <= dates[1]:
                    model_path = list(win_dir.glob('model_*.pkl'))[0]
                    model = joblib.load(model_path)

                    print(f"\n✓ Model loaded successfully!")
                    print(f"  Window: {win_dir.name}")
                    print(f"  Model path: {model_path}")
                    print(f"  Train start: {dates[0]}")
                    print(f"  Test end: {dates[1]}")
                    print(f"  Model type: {type(model).__name__}")

                    if hasattr(model, 'calibrated_classifiers_'):
                        print(f"  Calibration: CalibratedClassifierCV with {len(model.calibrated_classifiers_)} estimators")

                    break
        else:
            print("  ✗ No window directories found")
    else:
        print(f"  ✗ Pilot directory not found: {pilot_bracket_dir}")

    print("\n" + "="*60)
    print("Production API usage example:")
    print("="*60 + "\n")
    print("from ml.load_model import load_model_for_date")
    print("from pathlib import Path")
    print("from datetime import date")
    print("")
    print("# For production models (after promotion):")
    print('model, window, meta = load_model_for_date(')
    print('    city="chicago",')
    print('    bracket="between",')
    print('    target_date=date(2025, 9, 15),')
    print('    model_dir=Path("models/trained")')
    print(')')
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    main()
