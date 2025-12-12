#!/usr/bin/env python3
"""
Run full ML pipeline for multiple cities sequentially.

Pipeline phases:
  Phase 1: Build datasets + Train ordinal models for ALL cities
  Phase 2: Train edge classifiers for ALL cities (requires ordinal models)

Output:
  - Logs: logs/multi_city_pipeline_{timestamp}.log
  - Models: models/saved/{city}/ordinal_catboost_optuna.pkl
  - Models: models/saved/{city}/edge_classifier.pkl

Usage:
    # Test with 2 cities, quick run
    PYTHONPATH=. python scripts/run_multi_city_pipeline.py \
        --cities denver miami \
        --start 2025-05-01 --end 2025-05-14 \
        --ordinal-trials 2 --edge-trials 2

    # Full overnight run for 4 cities
    PYTHONPATH=. python scripts/run_multi_city_pipeline.py \
        --cities denver los_angeles miami philadelphia \
        --start 2023-01-01 --end 2025-12-03 \
        --ordinal-trials 150 --edge-trials 80 \
        --workers 8 --continue-on-error

    # Skip edge training (ordinal only)
    PYTHONPATH=. python scripts/run_multi_city_pipeline.py \
        --cities denver los_angeles miami philadelphia \
        --start 2023-01-01 --end 2025-12-03 \
        --ordinal-trials 150 --skip-edge
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure logs directory exists
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Create timestamped log file
LOG_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"multi_city_pipeline_{LOG_TIMESTAMP}.log"

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)

# Cities that need processing (excluding austin and chicago)
OTHER_CITIES = ["denver", "los_angeles", "miami", "philadelphia"]
ALL_CITIES = ["austin", "chicago", "denver", "los_angeles", "miami", "philadelphia"]


def run_command(cmd: list[str], description: str) -> tuple[bool, float]:
    """Run a command and return (success, elapsed_seconds).

    Streams output to both console and log file in real-time.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 60)

    start_time = datetime.now()

    # Run with output going to console (and our log file via tee-like behavior)
    # Using subprocess.run with no capture so child process output goes to same stdout/stderr
    result = subprocess.run(cmd)
    elapsed = (datetime.now() - start_time).total_seconds()

    if result.returncode == 0:
        logger.info(f"SUCCESS: {description} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return True, elapsed
    else:
        logger.error(f"FAILED: {description} (exit code {result.returncode}) after {elapsed:.1f}s")
        return False, elapsed


def run_city_ordinal(
    city: str,
    start_date: str,
    end_date: str,
    trials: int,
    workers: int,
    cache_dir: str = "models/saved",
    skip_build: bool = False,
) -> tuple[bool, float]:
    """Run dataset build + ordinal training for a single city.

    Returns: (success, elapsed_seconds)
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"# ORDINAL TRAINING: {city.upper()}")
    logger.info(f"# Date range: {start_date} to {end_date}")
    logger.info(f"# Trials: {trials}")
    logger.info(f"{'#'*60}")

    city_start = datetime.now()
    total_elapsed = 0.0

    # Step 1: Build dataset
    if not skip_build:
        build_cmd = [
            sys.executable,
            "models/pipeline/01_build_dataset.py",
            "--city", city,
            "--start", start_date,
            "--end", end_date,
            "--workers", str(workers),
        ]
        success, elapsed = run_command(build_cmd, f"Build dataset for {city}")
        total_elapsed += elapsed
        if not success:
            return False, total_elapsed
    else:
        logger.info(f"Skipping dataset build for {city} (--skip-build)")

    # Step 2: Train ordinal model
    train_cmd = [
        sys.executable,
        "models/pipeline/03_train_ordinal.py",
        "--city", city,
        "--trials", str(trials),
        "--workers", str(workers),
        "--cache-dir", cache_dir,
    ]
    success, elapsed = run_command(train_cmd, f"Train ordinal model for {city}")
    total_elapsed += elapsed
    if not success:
        return False, total_elapsed

    city_elapsed = (datetime.now() - city_start).total_seconds()
    logger.info(f"\n{city.upper()} ORDINAL COMPLETED in {city_elapsed:.1f}s ({city_elapsed/60:.1f} min)")

    return True, city_elapsed


def run_city_edge(
    city: str,
    trials: int,
    workers: int,
    threshold: float = 1.5,
    sample_rate: int = 4,
) -> tuple[bool, float]:
    """Run edge classifier training for a single city (requires ordinal model).

    Returns: (success, elapsed_seconds)
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"# EDGE TRAINING: {city.upper()}")
    logger.info(f"# Trials: {trials}, Threshold: {threshold}°F, Sample rate: {sample_rate}")
    logger.info(f"{'#'*60}")

    # Pre-flight check: verify ordinal model exists
    ordinal_model_path = Path(f"models/saved/{city}/ordinal_catboost_optuna.pkl")
    if not ordinal_model_path.exists():
        logger.error(f"Ordinal model not found at {ordinal_model_path}")
        logger.error("Edge classifier requires a trained ordinal model. Run ordinal training first.")
        return False, 0.0

    city_start = datetime.now()

    # Train edge classifier
    edge_cmd = [
        sys.executable,
        "scripts/train_edge_classifier.py",
        "--city", city,
        "--trials", str(trials),
        "--workers", str(workers),
        "--threshold", str(threshold),
        "--sample-rate", str(sample_rate),
        "--optuna-metric", "sharpe",
    ]
    success, elapsed = run_command(edge_cmd, f"Train edge classifier for {city}")
    if not success:
        return False, elapsed

    city_elapsed = (datetime.now() - city_start).total_seconds()
    logger.info(f"\n{city.upper()} EDGE COMPLETED in {city_elapsed:.1f}s ({city_elapsed/60:.1f} min)")

    return True, city_elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run full ML pipeline for multiple cities (ordinal + edge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test with 2 cities, quick run
    PYTHONPATH=. python scripts/run_multi_city_pipeline.py \\
        --cities denver miami \\
        --start 2025-05-01 --end 2025-05-14 \\
        --ordinal-trials 2 --edge-trials 2

    # Full overnight run for 4 cities
    PYTHONPATH=. python scripts/run_multi_city_pipeline.py \\
        --cities denver los_angeles miami philadelphia \\
        --start 2023-01-01 --end 2025-12-03 \\
        --ordinal-trials 150 --edge-trials 80 \\
        --workers 8 --continue-on-error

    # Ordinal only (skip edge training)
    PYTHONPATH=. python scripts/run_multi_city_pipeline.py \\
        --cities denver los_angeles miami philadelphia \\
        --start 2023-01-01 --end 2025-12-03 \\
        --ordinal-trials 150 --skip-edge
        """,
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        choices=ALL_CITIES,
        default=OTHER_CITIES,
        help=f"Cities to process (default: {OTHER_CITIES})",
    )
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
    parser.add_argument(
        "--ordinal-trials",
        type=int,
        default=1,
        help="Optuna trials for ordinal model (default: 1 for testing)",
    )
    parser.add_argument(
        "--edge-trials",
        type=int,
        default=1,
        help="Optuna trials for edge classifier (default: 1 for testing)",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=1.5,
        help="Edge detection threshold in °F (default: 1.5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for dataset building (default: 8)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip dataset building (use existing parquets)",
    )
    parser.add_argument(
        "--skip-edge",
        action="store_true",
        help="Skip edge training (ordinal only)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="If a city fails, continue to next city instead of stopping. "
             "Failed cities are logged and skipped for subsequent phases.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models/saved",
        help="Directory for cached datasets and models (default: models/saved)",
    )
    parser.add_argument(
        "--edge-sample-rate",
        type=int,
        default=4,
        help="Sample rate for edge classifier (every Nth snapshot, default: 4)",
    )
    args = parser.parse_args()

    cities = args.cities
    start_date = args.start
    end_date = args.end

    logger.info("=" * 60)
    logger.info("MULTI-CITY ML PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    logger.info(f"Cities: {cities}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Ordinal trials: {args.ordinal_trials}")
    logger.info(f"Edge trials: {args.edge_trials} {'(SKIPPED)' if args.skip_edge else ''}")
    logger.info(f"Edge threshold: {args.edge_threshold}°F")
    logger.info(f"Edge sample rate: {args.edge_sample_rate}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info(f"Skip build: {args.skip_build}")
    logger.info(f"Continue on error: {args.continue_on_error}")

    # Verify required directories exist
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Models output: {cache_dir.absolute()}")

    overall_start = datetime.now()
    ordinal_results = {}  # city -> (status, elapsed_seconds)
    edge_results = {}  # city -> (status, elapsed_seconds)

    # =========================================================================
    # PHASE 1: Ordinal training for ALL cities
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: ORDINAL MODEL TRAINING")
    logger.info("=" * 60)

    for city in cities:
        success, elapsed = run_city_ordinal(
            city=city,
            start_date=start_date,
            end_date=end_date,
            trials=args.ordinal_trials,
            workers=args.workers,
            cache_dir=args.cache_dir,
            skip_build=args.skip_build,
        )
        ordinal_results[city] = ("SUCCESS" if success else "FAILED", elapsed)

        if not success and not args.continue_on_error:
            logger.error(f"Ordinal training failed for {city}. Stopping.")
            logger.error(f"(Use --continue-on-error to skip failed cities)")
            break

    # =========================================================================
    # PHASE 2: Edge training for ALL cities (if not skipped)
    # =========================================================================
    if not args.skip_edge:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: EDGE CLASSIFIER TRAINING")
        logger.info("=" * 60)

        for city in cities:
            # Skip edge training if ordinal failed for this city
            ordinal_status, _ = ordinal_results.get(city, ("NOT_RUN", 0))
            if ordinal_status == "FAILED":
                logger.warning(f"Skipping edge training for {city} (ordinal failed)")
                edge_results[city] = ("SKIPPED", 0)
                continue
            if ordinal_status == "NOT_RUN":
                logger.warning(f"Skipping edge training for {city} (ordinal not run)")
                edge_results[city] = ("SKIPPED", 0)
                continue

            success, elapsed = run_city_edge(
                city=city,
                trials=args.edge_trials,
                workers=args.workers,
                threshold=args.edge_threshold,
                sample_rate=args.edge_sample_rate,
            )
            edge_results[city] = ("SUCCESS" if success else "FAILED", elapsed)

            if not success and not args.continue_on_error:
                logger.error(f"Edge training failed for {city}. Stopping.")
                break
    else:
        logger.info("\n(Skipping edge training phase)")

    overall_elapsed = (datetime.now() - overall_start).total_seconds()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min, {overall_elapsed/3600:.2f} hr)")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info("")

    logger.info("Ordinal Training Results:")
    logger.info("-" * 40)
    for city, (status, elapsed) in ordinal_results.items():
        emoji = "+" if status == "SUCCESS" else "X"
        time_str = f"{elapsed/60:.1f} min" if elapsed > 0 else "-"
        logger.info(f"  [{emoji}] {city:15} {status:10} ({time_str})")

    if not args.skip_edge:
        logger.info("")
        logger.info("Edge Training Results:")
        logger.info("-" * 40)
        for city, (status, elapsed) in edge_results.items():
            emoji = "+" if status == "SUCCESS" else ("~" if status == "SKIPPED" else "X")
            time_str = f"{elapsed/60:.1f} min" if elapsed > 0 else "-"
            logger.info(f"  [{emoji}] {city:15} {status:10} ({time_str})")

    # List output files
    logger.info("")
    logger.info("Output Files:")
    logger.info("-" * 40)
    for city in cities:
        ordinal_status, _ = ordinal_results.get(city, ("NOT_RUN", 0))
        if ordinal_status == "SUCCESS":
            logger.info(f"  models/saved/{city}/ordinal_catboost_optuna.pkl")
        if not args.skip_edge:
            edge_status, _ = edge_results.get(city, ("NOT_RUN", 0))
            if edge_status == "SUCCESS":
                logger.info(f"  models/saved/{city}/edge_classifier.pkl")

    # Check for any failures
    failures = [c for c, (s, _) in {**ordinal_results, **edge_results}.items() if s == "FAILED"]
    if failures:
        logger.error(f"\nFAILED CITIES: {failures}")
        logger.error(f"Check log file for details: {LOG_FILE}")
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("ALL CITIES COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
