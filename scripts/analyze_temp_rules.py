#!/usr/bin/env python3
"""
Temperature rounding rule analysis script.

Tests multiple deterministic rules for mapping Visual Crossing 5-minute
float temperatures to NWS/Kalshi integer daily highs.

This is a CRITICAL validation step before running backtests, as incorrect
temperature-to-bracket mapping will produce false P&L signals.

Usage:
    # Single city, specific date range
    python scripts/analyze_temp_rules.py \\
        --city chicago \\
        --start 2025-05-01 \\
        --end 2025-11-27

    # All cities, full history
    python scripts/analyze_temp_rules.py \\
        --all-cities \\
        --start 2023-01-01 \\
        --end 2025-11-27

    # Custom output path
    python scripts/analyze_temp_rules.py \\
        --city austin \\
        --start 2024-01-01 \\
        --end 2024-12-31 \\
        --out reports/temp_rules/austin_2024.csv
"""

import argparse
import logging
import sys
from datetime import date

# Add src to path
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from src.config import CITY_IDS
from src.db import get_db_session
from analysis.temperature.evaluator import (
    evaluate_rules,
    evaluate_multiple_cities,
    print_stats_summary,
    write_mismatches_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze temperature rounding rules against historical settlements"
    )

    # City selection
    parser.add_argument("--city", type=str, help="Single city to analyze")
    parser.add_argument(
        "--all-cities",
        action="store_true",
        help="Analyze all 6 cities",
    )

    # Date range
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )

    # Output
    parser.add_argument(
        "--out",
        type=str,
        default="reports/temp_rules/mismatches.csv",
        help="Output CSV path for mismatches (default: reports/temp_rules/mismatches.csv)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log progress every 100 days",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Parse dates
    try:
        start_day = date.fromisoformat(args.start)
        end_day = date.fromisoformat(args.end)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.error("Use YYYY-MM-DD format (e.g., 2025-05-01)")
        sys.exit(1)

    # Validate date range
    if end_day < start_day:
        logger.error("End date must be >= start date")
        sys.exit(1)

    # Determine cities to analyze
    if args.all_cities:
        cities = CITY_IDS
        logger.info(f"Analyzing all {len(cities)} cities")
    elif args.city:
        if args.city not in CITY_IDS:
            logger.error(f"Unknown city: {args.city}")
            logger.error(f"Available: {CITY_IDS}")
            sys.exit(1)
        cities = [args.city]
    else:
        logger.error("Must specify --city or --all-cities")
        sys.exit(1)

    # Run evaluation
    with get_db_session() as session:
        if len(cities) == 1:
            # Single city analysis
            city = cities[0]
            stats, mismatches = evaluate_rules(
                session,
                city,
                start_day,
                end_day,
                verbose=args.verbose,
            )

            print_stats_summary(stats)
            write_mismatches_csv(args.out, mismatches)

        else:
            # Multi-city analysis
            all_results = evaluate_multiple_cities(
                session,
                cities,
                start_day,
                end_day,
                verbose=args.verbose,
            )

            # Print summary for each city
            for city, (stats, mismatches) in all_results.items():
                print(f"\n{'='*60}")
                print(f"{city.upper()} Results")
                print(f"{'='*60}")
                print_stats_summary(stats)

                # Write city-specific mismatch CSV
                city_csv = args.out.replace(".csv", f"_{city}.csv")
                write_mismatches_csv(city_csv, mismatches)

            # Print overall best rules
            print(f"\n{'='*60}")
            print("OVERALL BEST RULES BY CITY")
            print(f"{'='*60}")

            for city, (stats, _) in all_results.items():
                best_rule = max(stats.values(), key=lambda s: s.accuracy)
                print(
                    f"{city:15s}: {best_rule.name:20s} "
                    f"({best_rule.accuracy:.2%} accuracy, "
                    f"MAE={best_rule.mae:.3f}°F)"
                )


if __name__ == "__main__":
    main()
