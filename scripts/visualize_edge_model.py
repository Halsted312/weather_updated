#!/usr/bin/env python3
"""
Visualize Edge Classifier Results

Generates comprehensive diagnostic plots for edge classifier models:
- Calibration curves
- Probability histograms
- PnL vs threshold analysis
- Sharpe vs threshold analysis

Usage:
    python scripts/visualize_edge_model.py --city austin
    python scripts/visualize_edge_model.py --city chicago --test-fraction 0.25

Output:
    Plots saved to: visualizations/edge/{city}/
    - edge_calibration.png
    - edge_prob_histogram.png
    - edge_pnl_sharpe_vs_threshold.png
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from visualizations.edge_reports import edge_report_for_city

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization reports for edge classifier models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"],
        help="City to visualize",
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of days to use for test set (default: 0.2)",
    )

    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Custom save directory (default: visualizations/edge/{city})",
    )

    args = parser.parse_args()

    # Generate report
    try:
        result = edge_report_for_city(
            city=args.city,
            save_dir=args.save_dir,
            test_fraction=args.test_fraction,
        )

        logger.info(f"Successfully generated reports for {args.city}")
        logger.info(f"Plots saved to: {result['save_dir']}")

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error(f"Make sure edge classifier is trained for {args.city}")
        logger.error(f"Run: python scripts/train_edge_classifier.py --city {args.city}")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
