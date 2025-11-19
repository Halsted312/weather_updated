#!/usr/bin/env python3
"""
Continuous retraining loop for production deployment.

This script:
1. Checks for new market data since last training
2. Retrains models if new data is available
3. Validates new models against holdout data
4. Promotes models if they pass quality gates
5. Can be run as a daily cron job

Usage:
    python scripts/continuous_retrain.py --city chicago --bracket between

Cron example (daily at 2 AM):
    0 2 * * * cd /path/to/kalshi_weather && python scripts/continuous_retrain.py >> logs/retrain.log 2>&1
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import json
import subprocess
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
from db.session import get_session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class ContinuousRetrainer:
    def __init__(
        self,
        city: str,
        bracket: str,
        model_type: str = "both",
        models_dir: str = "models/trained",
        production_dir: str = "models/production",
        min_new_days: int = 7,
        train_days: int = 90,
        holdout_days: int = 7,
    ):
        """
        Initialize continuous retraining system.

        Args:
            city: City to train for
            bracket: Bracket type
            model_type: Model type(s) to train ("elasticnet", "catboost", "both")
            models_dir: Directory for candidate models
            production_dir: Directory for production models
            min_new_days: Minimum days of new data required to trigger retrain
            train_days: Training window size
            holdout_days: Holdout validation period
        """
        self.city = city
        self.bracket = bracket
        self.model_type = model_type
        self.models_dir = Path(models_dir)
        self.production_dir = Path(production_dir)
        self.min_new_days = min_new_days
        self.train_days = train_days
        self.holdout_days = holdout_days

        # Paths for state tracking
        self.state_file = self.production_dir / f"{city}_{bracket}_state.json"
        self.production_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Dict[str, Any]:
        """Load training state from disk."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "last_train_date": None,
            "last_data_date": None,
            "production_models": {},
            "history": []
        }

    def save_state(self, state: Dict[str, Any]):
        """Save training state to disk."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def check_new_data(self) -> Tuple[Optional[date], int]:
        """
        Check for new market data since last training.

        Returns:
            Tuple of (latest_date, days_of_new_data)
        """
        from ml.dataset import CITY_CONFIG

        series_code = CITY_CONFIG[self.city]["series_code"]
        series_ticker = f"KXHIGH{series_code}"

        with get_session() as session:
            # Find latest settled market
            query = text("""
                SELECT MAX(m.event_date) as latest_date
                FROM markets m
                WHERE m.series_ticker = :series
                  AND m.status = 'settled'
                  AND m.result IS NOT NULL
            """)

            result = session.execute(query, {"series": series_ticker})
            row = result.fetchone()

            if not row or not row.latest_date:
                logger.warning("No settled markets found")
                return None, 0

            latest_date = row.latest_date
            if isinstance(latest_date, str):
                latest_date = date.fromisoformat(latest_date)

        # Load state to check last training date
        state = self.load_state()
        last_data_date = state.get("last_data_date")

        if last_data_date:
            if isinstance(last_data_date, str):
                last_data_date = date.fromisoformat(last_data_date)
            days_new = (latest_date - last_data_date).days
        else:
            days_new = self.min_new_days + 1  # Force initial training

        logger.info(f"Latest data: {latest_date}, Last trained: {last_data_date}, "
                    f"New days: {days_new}")

        return latest_date, days_new

    def train_model(
        self,
        model_type: str,
        end_date: date,
        feature_set: str = "baseline"
    ) -> bool:
        """
        Train a single model type.

        Returns:
            True if training succeeded
        """
        start_date = end_date - timedelta(days=self.train_days + self.holdout_days - 1)

        logger.info(f"Training {model_type} model from {start_date} to {end_date}")

        # Determine number of trials
        n_trials = 60 if model_type == "catboost" else 40

        cmd = [
            "python", "ml/train_walkforward.py",
            "--city", self.city,
            "--bracket", self.bracket,
            "--start", start_date.isoformat(),
            "--end", end_date.isoformat(),
            "--feature-set", feature_set,
            "--train-days", str(self.train_days),
            "--model-type", model_type,
            "--trials", str(n_trials),
            "--outdir", str(self.models_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✓ {model_type} training complete")
            return True
        else:
            logger.error(f"✗ {model_type} training failed: {result.stderr}")
            return False

    def validate_model(
        self,
        model_type: str,
        end_date: date,
        sharpe_threshold: float = 2.0,
        brier_threshold: float = 0.09
    ) -> Dict[str, Any]:
        """
        Validate model on holdout data.

        Returns:
            Dict with validation metrics
        """
        # Run backtest on holdout period
        holdout_start = end_date - timedelta(days=self.holdout_days - 1)

        output_file = f"/tmp/validate_{self.city}_{self.bracket}_{model_type}.json"

        cmd = [
            "python", "backtest/run_backtest.py",
            "--city", self.city,
            "--bracket", self.bracket,
            "--strategy", "model_kelly",
            "--models-dir", str(self.models_dir),
            "--model-type", model_type,
            "--start-date", holdout_start.isoformat(),
            "--end-date", end_date.isoformat(),
            "--output-json", output_file
        ]

        logger.info(f"Validating {model_type} on holdout: {holdout_start} to {end_date}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Validation failed: {result.stderr}")
            return {"passed": False, "error": "Backtest failed"}

        # Load metrics
        try:
            with open(output_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load validation results: {e}")
            return {"passed": False, "error": str(e)}

        # Check quality gates
        sharpe = metrics.get("sharpe", 0)
        brier = metrics.get("ece_trades") or metrics.get("ece_all_minutes")
        n_trades = metrics.get("n_trades", 0)

        passed = True
        reasons = []

        if n_trades == 0:
            passed = False
            reasons.append(f"No trades (n={n_trades})")
        elif sharpe < sharpe_threshold:
            passed = False
            reasons.append(f"Sharpe {sharpe:.2f} < {sharpe_threshold}")

        if brier and brier > brier_threshold:
            passed = False
            reasons.append(f"Brier {brier:.3f} > {brier_threshold}")

        validation_result = {
            "passed": passed,
            "sharpe": sharpe,
            "brier": brier,
            "n_trades": n_trades,
            "total_pnl": metrics.get("total_pnl_cents", 0) / 100,
            "reasons": reasons
        }

        if passed:
            logger.info(f"✓ {model_type} passed validation: Sharpe={sharpe:.2f}, "
                       f"Brier={brier:.3f if brier else 'N/A'}, Trades={n_trades}")
        else:
            logger.warning(f"✗ {model_type} failed validation: {', '.join(reasons)}")

        return validation_result

    def promote_model(self, model_type: str, end_date: date):
        """
        Promote validated model to production.
        """
        # Find latest model files
        if model_type == "catboost":
            source_dir = self.models_dir / self.city / f"{self.bracket}_catboost"
        else:
            source_dir = self.models_dir / self.city / self.bracket

        # Find most recent window directory
        window_dirs = sorted(source_dir.glob("win_*"))
        if not window_dirs:
            logger.error(f"No model windows found in {source_dir}")
            return

        latest_window = window_dirs[-1]

        # Create production directory
        if model_type == "catboost":
            prod_dir = self.production_dir / self.city / f"{self.bracket}_catboost"
        else:
            prod_dir = self.production_dir / self.city / self.bracket

        prod_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        import shutil
        for file in latest_window.glob("*"):
            dest = prod_dir / file.name
            shutil.copy2(file, dest)

        logger.info(f"✓ Promoted {model_type} model to {prod_dir}")

        # Update state
        state = self.load_state()
        state["production_models"][model_type] = {
            "date": end_date.isoformat(),
            "path": str(prod_dir),
            "window": latest_window.name
        }
        self.save_state(state)

    def run(self) -> bool:
        """
        Run the continuous retraining loop.

        Returns:
            True if any models were retrained and promoted
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"CONTINUOUS RETRAINING: {self.city} / {self.bracket}")
        logger.info(f"{'='*70}")

        # Check for new data
        latest_date, days_new = self.check_new_data()

        if days_new < self.min_new_days:
            logger.info(f"Not enough new data ({days_new} < {self.min_new_days} days). "
                       "Skipping retrain.")
            return False

        # Train models
        models_to_train = ["elasticnet", "catboost"] if self.model_type == "both" else [self.model_type]
        trained = {}
        validated = {}

        for model in models_to_train:
            # Train
            success = self.train_model(model, latest_date)
            trained[model] = success

            if success:
                # Validate
                validation = self.validate_model(model, latest_date)
                validated[model] = validation

                # Promote if passed
                if validation["passed"]:
                    self.promote_model(model, latest_date)

        # Update state
        state = self.load_state()
        state["last_train_date"] = datetime.now().isoformat()
        state["last_data_date"] = latest_date.isoformat()

        # Add to history
        state["history"].append({
            "timestamp": datetime.now().isoformat(),
            "data_date": latest_date.isoformat(),
            "trained": trained,
            "validated": validated
        })

        # Keep only last 30 history entries
        state["history"] = state["history"][-30:]

        self.save_state(state)

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("RETRAINING SUMMARY")
        logger.info(f"{'='*70}")

        promoted_count = 0
        for model in models_to_train:
            if trained.get(model) and validated.get(model, {}).get("passed"):
                logger.info(f"✓ {model}: Trained and promoted")
                promoted_count += 1
            elif trained.get(model):
                reasons = validated.get(model, {}).get("reasons", ["Unknown"])
                logger.info(f"⚠ {model}: Trained but not promoted ({', '.join(reasons)})")
            else:
                logger.info(f"✗ {model}: Training failed")

        logger.info(f"\nPromoted {promoted_count}/{len(models_to_train)} models to production")
        logger.info(f"{'='*70}\n")

        return promoted_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Continuous retraining for production models"
    )
    parser.add_argument("--city", default="chicago", help="City name")
    parser.add_argument("--bracket", choices=["between", "greater", "less"],
                        default="between", help="Bracket type")
    parser.add_argument("--model-type", choices=["elasticnet", "catboost", "both"],
                        default="both", help="Model type(s) to train")
    parser.add_argument("--models-dir", default="models/trained",
                        help="Directory for candidate models")
    parser.add_argument("--production-dir", default="models/production",
                        help="Directory for production models")
    parser.add_argument("--min-new-days", type=int, default=7,
                        help="Minimum days of new data to trigger retrain")
    parser.add_argument("--train-days", type=int, default=90,
                        help="Training window size")
    parser.add_argument("--holdout-days", type=int, default=7,
                        help="Holdout validation period")
    parser.add_argument("--sharpe-threshold", type=float, default=2.0,
                        help="Minimum Sharpe ratio for promotion")
    parser.add_argument("--brier-threshold", type=float, default=0.09,
                        help="Maximum Brier score for promotion")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create retrainer
    retrainer = ContinuousRetrainer(
        city=args.city,
        bracket=args.bracket,
        model_type=args.model_type,
        models_dir=args.models_dir,
        production_dir=args.production_dir,
        min_new_days=args.min_new_days,
        train_days=args.train_days,
        holdout_days=args.holdout_days
    )

    # Run retraining
    success = retrainer.run()

    # Exit code for scripts/monitoring
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()