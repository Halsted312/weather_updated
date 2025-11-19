#!/usr/bin/env python3
"""
Adapter to integrate ModelKellyStrategy with the backtest harness.

Loads pre-computed Ridge predictions from walk-forward training and uses
ModelKellyStrategy to generate signals based on model probabilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import math
from datetime import datetime
from typing import Dict, List, Optional
import glob
import pandas as pd
import numpy as np
from scipy.stats import norm
from zoneinfo import ZoneInfo

from backtest.strategy import Strategy, Signal, Position
from backtest.model_strategy import ModelKellyStrategy, ExecParams
from db.connection import get_session
from sqlalchemy import text
from ml.city_config import CITY_CONFIG

logger = logging.getLogger(__name__)


def probability_from_tmax(
    mu: float,
    sigma: float,
    strike_type: str,
    floor_strike: Optional[float],
    cap_strike: Optional[float],
) -> float:
    """Convert a (mu, sigma) temperature forecast into a bracket probability."""

    if sigma <= 0:
        sigma = 1.0

    if strike_type == "between" and floor_strike is not None and cap_strike is not None:
        return float(
            norm.cdf((float(cap_strike) - mu) / sigma)
            - norm.cdf((float(floor_strike) - mu) / sigma)
        )

    if strike_type == "greater" and floor_strike is not None:
        return float(1.0 - norm.cdf((float(floor_strike) - mu) / sigma))

    if strike_type == "less" and cap_strike is not None:
        return float(norm.cdf((float(cap_strike) - mu) / sigma))

    return 0.5


class ModelKellyBacktestStrategy(Strategy):
    """
    Adapter for ModelKellyStrategy that integrates with the backtest harness.

    Loads pre-computed Ridge predictions from walk-forward training CSVs
    and generates signals using ModelKellyStrategy logic.
    """

    def __init__(
        self,
        city: str,
        bracket: str,
        models_dir: str = "models/trained",
        exec_params: Optional[ExecParams] = None,
        unified_head: bool = False,
        unified_tau: float = 1.0,
        model_type: str = "elasticnet",
        ev_models_dir: Optional[str] = None,
        ev_min_delta_cents: float = 0.0,
        ev_blend_weight: float = 0.0,
        ev_max_staleness_minutes: Optional[float] = 90.0,
        ev_allow_missing: bool = False,
        tmax_preds_path: Optional[str] = None,
        tmax_min_prob: float = 0.0,
        tmax_sigma_multiplier: float = 0.0,
        hybrid_model_type: Optional[str] = None,
        hybrid_models_dir: Optional[str] = None,
        hybrid_min_prob: float = 0.0,
        market_odds_weight: float = 0.0,
    ):
        """
        Initialize model-driven strategy.

        Args:
            city: City name (e.g., "chicago")
            bracket: Bracket type ("between", "greater", "less")
            models_dir: Root directory for trained models
            exec_params: Execution parameters (uses defaults if None)
            unified_head: If True, apply unified coupling across 6 brackets
            unified_tau: Temperature parameter for unified head coupling
            model_type: Model type to load predictions from ("elasticnet" or "catboost")
        """
        self.city = city
        self.bracket = bracket
        self.models_dir = models_dir
        self.unified_head = unified_head
        self.unified_tau = unified_tau
        self.model_type = model_type
        self.ev_models_dir = ev_models_dir
        self.ev_min_delta_cents = ev_min_delta_cents
        self.ev_blend_weight = ev_blend_weight
        self.ev_max_staleness_minutes = ev_max_staleness_minutes
        self.ev_allow_missing = ev_allow_missing
        self.tmax_preds_path = tmax_preds_path
        self.tmax_min_prob = tmax_min_prob
        self.tmax_sigma_multiplier = tmax_sigma_multiplier
        self.hybrid_model_type = hybrid_model_type
        self.hybrid_models_dir = hybrid_models_dir or models_dir
        self.hybrid_min_prob = hybrid_min_prob
        self.market_odds_weight = max(0.0, min(1.0, market_odds_weight))

        if self.model_type == "tmax_reg" and not self.tmax_preds_path:
            raise ValueError("tmax_reg model_type requires --tmax-preds-csv")

        # Initialize ModelKellyStrategy
        self.kelly_strategy = ModelKellyStrategy(exec_params=exec_params)

        tz_name = CITY_CONFIG.get(city, {}).get("timezone", "UTC")
        self.city_timezone = ZoneInfo(tz_name)

        # Load all predictions from walk-forward windows
        self.predictions = self._load_predictions()

        # Optionally load EV predictions for gating/blending
        self.ev_predictions = (
            self._load_ev_predictions() if self.ev_models_dir else pd.DataFrame()
        )
        self.ev_enabled = not self.ev_predictions.empty

        if self.model_type == "tmax_reg":
            self.tmax_preds = self._load_tmax_predictions(self.tmax_preds_path)
            self.tmax_times = self.tmax_preds["timestamp"].to_numpy()
        else:
            self.tmax_preds = None
            self.tmax_times = None

        self.hybrid_predictions = pd.DataFrame()
        if self.hybrid_model_type:
            self.hybrid_predictions = self._load_additional_predictions(
                model_type=self.hybrid_model_type,
                models_dir=self.hybrid_models_dir,
            )
        self.hybrid_enabled = not self.hybrid_predictions.empty

        # Apply unified head coupling if enabled
        if self.unified_head:
            self._apply_unified_coupling()

        # Load actual minute-level candles
        self.candles = self._load_candles()

        logger.info(
            f"ModelKellyBacktestStrategy initialized: {len(self.predictions)} predictions, "
            f"{len(self.candles)} candles loaded"
        )

    def _load_predictions(self) -> pd.DataFrame:
        """
        Load all walk-forward predictions from CSV files.

        Returns:
            DataFrame with columns: market_ticker, timestamp, p_model, y_true, event_date
        """
        # Build pattern based on model type
        if self.model_type == "elasticnet":
            pattern = os.path.join(
                self.models_dir,
                self.city,
                self.bracket,
                "win_*",
                "preds_*.csv"
            )
        elif self.model_type == "catboost":
            pattern = os.path.join(
                self.models_dir,
                self.city,
                f"{self.bracket}_catboost",
                "win_*",
                "preds_*.csv"
            )
        elif self.model_type == "ev_catboost":
            pattern = os.path.join(
                self.models_dir,
                self.city,
                f"{self.bracket}_ev_catboost",
                "win_*",
                "preds_ev_*.csv"
            )
        elif self.model_type == "tmax_reg":
            return pd.DataFrame()
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        pred_files = sorted(glob.glob(pattern))

        if not pred_files:
            logger.warning(
                f"No prediction files found matching pattern: {pattern}"
            )
            return pd.DataFrame()

        logger.info(f"Loading predictions from {len(pred_files)} walk-forward windows...")

        # Load and concatenate all prediction files
        dfs = []
        for f in pred_files:
            df = pd.read_csv(f)
            dfs.append(df)

        predictions = pd.concat(dfs, ignore_index=True)
        if self.model_type == "ev_catboost":
            if "p_model" not in predictions.columns:
                if "pred_future_mid_cents" in predictions.columns:
                    predictions["p_model"] = predictions["pred_future_mid_cents"] / 100.0
                else:
                    raise ValueError("EV predictions missing pred_future_mid_cents/p_model columns")
            predictions["ev_delta_cents"] = predictions.get("pred_delta_cents")
            if "y_true" not in predictions.columns:
                if "actual_delta_cents" in predictions.columns:
                    predictions["y_true"] = (predictions["actual_delta_cents"] > 0).astype(int)
                else:
                    predictions["y_true"] = np.nan
        elif "y_true" not in predictions.columns:
            predictions["y_true"] = np.nan

        # B2 PATCH: Normalize timestamps to UTC-naive for consistent joins
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True).dt.tz_convert(None)
        predictions["event_date"] = pd.to_datetime(predictions["event_date"]).dt.date

        logger.info(
            f"Loaded {len(predictions)} predictions for {len(predictions['market_ticker'].unique())} unique markets"
        )

        # DEBUG A1: Verify predictions loaded for backtest window
        assert not predictions.empty, "No predictions loaded for this backtest window"
        print(
            f"[DEBUG A1] Loaded predictions: {len(predictions):,} rows | "
            f"ts=[{predictions['timestamp'].min()} .. {predictions['timestamp'].max()}] | "
            f"event_date_local=[{predictions['event_date'].min()} .. {predictions['event_date'].max()}]"
        )
        print(f"[DEBUG A1] Unique bracket_type in preds: {predictions.get('bracket_type', pd.Series()).dropna().unique()[:8] if 'bracket_type' in predictions else 'N/A'}")

        # Show distribution by date
        date_counts = predictions.groupby('event_date').size()
        print(f"[DEBUG A1] Predictions by date (first 10): {dict(list(date_counts.items())[:10])}")

        return predictions

    def _load_additional_predictions(self, model_type: str, models_dir: str) -> pd.DataFrame:
        """Load auxiliary settlement-model predictions for hybrid gating."""

        if model_type not in {"elasticnet", "catboost", "ev_catboost"}:
            raise ValueError(f"Unsupported hybrid model type: {model_type}")

        if model_type == "elasticnet":
            pattern = os.path.join(
                models_dir,
                self.city,
                self.bracket,
                "win_*",
                "preds_*.csv",
            )
        elif model_type == "catboost":
            pattern = os.path.join(
                models_dir,
                self.city,
                f"{self.bracket}_catboost",
                "win_*",
                "preds_*.csv",
            )
        else:  # ev_catboost
            pattern = os.path.join(
                models_dir,
                self.city,
                f"{self.bracket}_ev_catboost",
                "win_*",
                "preds_ev_*.csv",
            )

        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning("Hybrid gating enabled but no prediction files found (pattern=%s)", pattern)
            return pd.DataFrame()

        logger.info("Loading hybrid predictions (%s) from %d windows...", model_type, len(files))
        dfs = []
        for fpath in files:
            df = pd.read_csv(fpath)
            dfs.append(df)

        predictions = pd.concat(dfs, ignore_index=True)
        if model_type == "ev_catboost":
            if "p_model" not in predictions.columns and "pred_future_mid_cents" in predictions.columns:
                predictions["p_model"] = predictions["pred_future_mid_cents"] / 100.0
        if "p_model" not in predictions.columns:
            raise ValueError("Hybrid prediction files must contain a p_model column")

        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True).dt.tz_convert(None)
        if "event_date" in predictions.columns:
            predictions["event_date"] = pd.to_datetime(predictions["event_date"]).dt.date
        elif "date" in predictions.columns:
            predictions["event_date"] = pd.to_datetime(predictions["date"]).dt.date
        else:
            raise ValueError("Hybrid predictions missing event_date/date column")

        logger.info(
            "Loaded %d hybrid prediction rows for %d markets",
            len(predictions),
            predictions["market_ticker"].nunique(),
        )
        return predictions

    def _load_ev_predictions(self) -> pd.DataFrame:
        """Load EV predictions (future mid deltas) for optional gating/blending."""
        pattern = os.path.join(
            self.ev_models_dir,
            self.city,
            f"{self.bracket}_ev_catboost",
            "win_*",
            "preds_ev_*.csv",
        )
        ev_files = sorted(glob.glob(pattern))
        if not ev_files:
            logger.warning(
                "EV gating enabled but no prediction files found (pattern=%s)",
                pattern,
            )
            return pd.DataFrame()

        logger.info("Loading EV predictions from %d windows...", len(ev_files))
        dfs = []
        for fpath in ev_files:
            df = pd.read_csv(fpath)
            dfs.append(df)

        ev_preds = pd.concat(dfs, ignore_index=True)
        ev_preds["timestamp"] = pd.to_datetime(ev_preds["timestamp"], utc=True).dt.tz_convert(None)
        ev_preds["event_date"] = pd.to_datetime(ev_preds["event_date"]).dt.date
        logger.info(
            "Loaded %d EV prediction rows for %d markets",
            len(ev_preds),
            ev_preds["market_ticker"].nunique(),
        )
        return ev_preds

    def _lookup_ev_prediction(
        self,
        ticker: str,
        event_date,
        timestamp: datetime,
    ) -> Optional[pd.Series]:
        """Return the latest EV prediction for ticker/event up to timestamp."""
        if not self.ev_enabled:
            return None

        preds = self.ev_predictions[
            (self.ev_predictions["market_ticker"] == ticker)
            & (self.ev_predictions["event_date"] == event_date)
            & (self.ev_predictions["timestamp"] <= timestamp)
        ]

        if preds.empty:
            preds = self.ev_predictions[
                (self.ev_predictions["market_ticker"] == ticker)
                & (self.ev_predictions["timestamp"] <= timestamp)
            ]

        if preds.empty:
            return None

        return preds.sort_values("timestamp").iloc[-1]

    def _load_tmax_predictions(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if "sigma_est" not in df.columns or df["sigma_est"].isna().all():
            df["sigma_est"] = max(3.0, df["pred"].std())
        return df

    def _lookup_tmax_snapshot(self, timestamp: datetime) -> Optional[pd.Series]:
        if self.tmax_preds is None or self.tmax_times is None:
            return None
        idx = self.tmax_preds["timestamp"].searchsorted(timestamp, side="right") - 1
        if idx < 0:
            return None
        return self.tmax_preds.iloc[idx]

    def _prob_from_tmax(self, snapshot: pd.Series, market) -> float:
        mu = float(snapshot["pred"])
        sigma = max(1.0, float(snapshot.get("sigma_est", 4.0)))
        strike_type = market["strike_type"]
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")
        return probability_from_tmax(mu, sigma, strike_type, floor_strike, cap_strike)

    def _pool_with_market(self, p_model: float, p_market: float) -> float:
        if self.market_odds_weight <= 0:
            return p_model

        def _clip(prob: float) -> float:
            return min(max(prob, 1e-4), 1 - 1e-4)

        logit_model = math.log(_clip(p_model) / (1 - _clip(p_model)))
        logit_market = math.log(_clip(p_market) / (1 - _clip(p_market)))
        pooled = (1 - self.market_odds_weight) * logit_model + self.market_odds_weight * logit_market
        return 1.0 / (1.0 + math.exp(-pooled))

    def _tmax_distance_to_boundary(self, pred: float, market) -> float:
        """Return absolute distance to the nearest bracket boundary for gating."""

        strike_type = market["strike_type"]
        if strike_type == "between":
            lower = float(market["floor_strike"])
            upper = float(market["cap_strike"])
            return float(min(abs(pred - lower), abs(pred - upper)))
        if strike_type == "greater":
            lower = float(market["floor_strike"])
            return float(abs(pred - lower))
        if strike_type == "less":
            upper = float(market["cap_strike"])
            return float(abs(pred - upper))
        return 0.0

    def _lookup_hybrid_prediction(
        self,
        ticker: str,
        event_date,
        timestamp: datetime,
    ) -> Optional[pd.Series]:
        if not self.hybrid_enabled:
            return None

        preds = self.hybrid_predictions[
            (self.hybrid_predictions["market_ticker"] == ticker)
            & (self.hybrid_predictions["event_date"] == event_date)
            & (self.hybrid_predictions["timestamp"] <= timestamp)
        ]

        if preds.empty:
            preds = self.hybrid_predictions[
                (self.hybrid_predictions["market_ticker"] == ticker)
                & (self.hybrid_predictions["timestamp"] <= timestamp)
            ]

        if preds.empty:
            return None

        return preds.sort_values("timestamp").iloc[-1]

    def _load_candles(self) -> pd.DataFrame:
        """
        Load minute-level candles for the city from database.

        Returns:
            DataFrame with columns: market_ticker, timestamp, yes_bid_close, yes_ask_close, etc.
        """
        from ml.dataset import CITY_CONFIG

        series_code = CITY_CONFIG[self.city]["series_code"]
        series_ticker = f"KXHIGH{series_code}"

        logger.info(f"Loading 1-minute candles for {self.city}...")

        with get_session() as session:
            query = text("""
                SELECT
                    market_ticker,
                    timestamp,
                    close,
                    volume
                FROM candles
                WHERE market_ticker LIKE :series_pattern
                  AND period_minutes = 1
                ORDER BY market_ticker, timestamp
            """)

            result = session.execute(query, {"series_pattern": f"{series_ticker}%"})
            rows = result.fetchall()

            if not rows:
                logger.warning(f"No 1-minute candles found for {self.city}")
                return pd.DataFrame()

        candles = pd.DataFrame(
            rows,
            columns=["market_ticker", "timestamp", "close", "volume"]
        )
        # B2 PATCH: Normalize timestamps to UTC-naive
        candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True).dt.tz_convert(None)

        # Estimate bid/ask from close price (assume 2¢ spread)
        # close is the YES price in cents, assume bid = close - 1, ask = close + 1
        candles["yes_bid_close"] = (candles["close"] - 1).clip(lower=1)
        candles["yes_ask_close"] = (candles["close"] + 1).clip(upper=99)

        logger.info(
            f"Loaded {len(candles)} candles for {candles['market_ticker'].nunique()} markets "
            f"(date range: {candles['timestamp'].min()} to {candles['timestamp'].max()})"
        )

        return candles

    def _apply_unified_coupling(self):
        """
        Apply unified head coupling to predictions.

        Groups predictions by (event_date, timestamp) and couples the 6 bracket
        probabilities into a coherent distribution that sums to 1.
        """
        from ml.unified_head import apply_unified_head

        logger.info(f"Applying unified head coupling (tau={self.unified_tau})...")

        # Check if required columns exist
        required_cols = ["event_date", "timestamp", "p_model"]
        missing = [col for col in required_cols if col not in self.predictions.columns]
        if missing:
            logger.warning(
                f"Cannot apply unified coupling, missing columns: {missing}. "
                f"Available columns: {list(self.predictions.columns)}"
            )
            return

        # Apply coupling
        self.predictions = apply_unified_head(
            df=self.predictions,
            group_cols=["event_date", "timestamp"],
            p_col="p_model",
            method="softmax",
            tau=self.unified_tau,
            output_col="p_unified"
        )

        # Use p_unified as p_model when unified head is enabled
        if "p_unified" in self.predictions.columns:
            self.predictions["p_model_original"] = self.predictions["p_model"]
            self.predictions["p_model"] = self.predictions["p_unified"]
            logger.info(
                f"Unified head applied. Coupling status: "
                f"{self.predictions['coupling_status'].value_counts().to_dict()}"
            )
        else:
            logger.warning("Unified head failed to create p_unified column")

    def generate_signals(
        self,
        timestamp: datetime,
        market_data: pd.DataFrame,
        positions: Dict[str, Position],
        bankroll_cents: float,
    ) -> List[Signal]:
        """
        Generate trading signals for markets using model predictions.

        Args:
            timestamp: Current timestamp
            market_data: DataFrame with market metadata (from load_markets_with_settlements)
            positions: Current positions
            bankroll_cents: Available cash in cents

        Returns:
            List of Signal objects for markets with positive edge
        """
        signals = []

        # For each market in market_data, look up the latest prediction
        snapshot = None
        if self.model_type == "tmax_reg":
            snapshot = self._lookup_tmax_snapshot(timestamp)
            if snapshot is None:
                logger.debug("No Tmax snapshot available at %s", timestamp)
                return []

        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        timestamp_local = ts.tz_convert(self.city_timezone)
        minute_of_day_local = timestamp_local.hour * 60 + timestamp_local.minute

        sigma_value = None
        humidity_value = None
        if snapshot is not None:
            sigma_raw = snapshot.get("sigma_est")
            if sigma_raw is not None and not pd.isna(sigma_raw):
                sigma_value = float(sigma_raw)
            for key in ("humidity_std", "humidity", "humidity_var"):
                val = snapshot.get(key)
                if val is not None and not pd.isna(val):
                    humidity_value = float(val)
                    break

        for _, market in market_data.iterrows():
            ticker = market["ticker"]
            event_date = market["date_local"]

            # Get predictions for this market
            # First try exact match on ticker and event_date
            if self.model_type == "tmax_reg":
                p_for_signal = self._prob_from_tmax(snapshot, market)
                latest_pred = None

                if self.tmax_min_prob > 0 and max(p_for_signal, 1.0 - p_for_signal) < self.tmax_min_prob:
                    logger.debug(
                        "Skip %s: Tmax prob %.3f below min %.2f",
                        ticker,
                        p_for_signal,
                        self.tmax_min_prob,
                    )
                    continue

                if self.tmax_sigma_multiplier > 0:
                    sigma = float(snapshot.get("sigma_est", 4.0))
                    distance = self._tmax_distance_to_boundary(float(snapshot["pred"]), market)
                    if distance < self.tmax_sigma_multiplier * sigma:
                        logger.debug(
                            "Skip %s: boundary distance %.2f < %.2f * sigma %.2f",
                            ticker,
                            distance,
                            self.tmax_sigma_multiplier,
                            sigma,
                        )
                        continue
            else:
                market_preds = self.predictions[
                    (self.predictions["market_ticker"] == ticker) &
                    (self.predictions["event_date"] == event_date)
                ]

                if market_preds.empty:
                    market_preds = self.predictions[
                        self.predictions["market_ticker"] == ticker
                    ]
                    if not market_preds.empty:
                        logger.debug(f"Using ticker-only match for {ticker}: found {len(market_preds)} predictions")

                if market_preds.empty:
                    logger.debug(f"No predictions found for {ticker}")
                    continue

                latest_pred = market_preds.sort_values("timestamp").iloc[-1]
                p_for_signal = float(latest_pred["p_model"])

            if self.hybrid_enabled:
                hybrid_pred = self._lookup_hybrid_prediction(ticker, event_date, timestamp)
                if hybrid_pred is None:
                    logger.debug("Skip %s: missing hybrid prediction", ticker)
                    continue
                p_hybrid = float(hybrid_pred["p_model"])
                if (p_hybrid >= 0.5) != (p_for_signal >= 0.5):
                    logger.debug("Skip %s: hybrid disagrees with Tmax", ticker)
                    continue
                if self.hybrid_min_prob > 0 and max(p_hybrid, 1.0 - p_hybrid) < self.hybrid_min_prob:
                    logger.debug(
                        "Skip %s: hybrid prob %.3f below min %.2f",
                        ticker,
                        p_hybrid,
                        self.hybrid_min_prob,
                    )
                    continue

            # Look up actual bid/ask from candles at or before current timestamp
            # Use latest available candle (markets often don't have candles at exact close time)
            ticker_candles = self.candles[
                (self.candles["market_ticker"] == ticker) &
                (self.candles["timestamp"] <= timestamp)
            ]

            if ticker_candles.empty:
                logger.debug(f"No candle data for {ticker} at or before {timestamp}")
                continue

            # Get the most recent candle (closest to timestamp)
            candle_row = ticker_candles.loc[ticker_candles["timestamp"].idxmax()]

            # Extract real bid/ask from candles
            bid = int(candle_row["yes_bid_close"])
            ask = int(candle_row["yes_ask_close"])

            # Calculate diagnostic fields
            spread_cents = ask - bid
            mid_price = (bid + ask) / 2.0
            p_market = mid_price / 100.0

            # Calculate time to close
            close_time = pd.to_datetime(market["close_time"])
            time_to_close_minutes = (close_time - timestamp).total_seconds() / 60.0

            # Create row with real prices (probability may be blended with EV signal)
            ev_reason = ""
            ev_delta_value = np.nan
            if self.ev_enabled:
                ev_pred = self._lookup_ev_prediction(ticker, event_date, timestamp)
                if ev_pred is None:
                    if not self.ev_allow_missing:
                        logger.debug("Skipping %s: missing EV prediction", ticker)
                        continue
                else:
                    ev_ts = pd.Timestamp(ev_pred["timestamp"])
                    minutes_stale = (
                        pd.Timestamp(timestamp) - ev_ts
                    ).total_seconds() / 60.0
                    if (
                        self.ev_max_staleness_minutes is not None
                        and minutes_stale > self.ev_max_staleness_minutes
                    ):
                        logger.debug(
                            "Skipping %s: EV prediction stale (%.1f min)",
                            ticker,
                            minutes_stale,
                        )
                        continue

                    ev_delta_value = float(ev_pred.get("pred_delta_cents", np.nan))
                    if np.isnan(ev_delta_value) and not self.ev_allow_missing:
                        logger.debug("Skipping %s: EV delta missing", ticker)
                        continue

                    if self.ev_blend_weight > 0 and "p_model" in ev_pred:
                        ev_p = float(ev_pred["p_model"])
                        blend_w = self.ev_blend_weight
                        p_for_signal = (1 - blend_w) * p_for_signal + blend_w * ev_p
                    if not np.isnan(ev_delta_value):
                        ev_reason = f", ev_delta={ev_delta_value:.1f}¢"

            pooled_prob = self._pool_with_market(p_for_signal, p_market)

            row = pd.Series({
                "market_ticker": ticker,
                "yes_bid_close": bid,
                "yes_ask_close": ask,
                "p_model": pooled_prob,
                "strike_type": market.get("strike_type"),
                "minute_of_day_local": minute_of_day_local,
                "sigma_est": sigma_value,
                "humidity_var": humidity_value,
            })

            # Generate signal using ModelKellyStrategy
            signal_dict = self.kelly_strategy.signal_for_row(row)

            if signal_dict is None:
                # No signal (filtered by spread or edge)
                continue

            # Convert to Signal object
            # Map action: "BUY_YES" -> "buy", "BUY_NO" -> "sell" (short YES)
            if signal_dict["action"] == "BUY_YES":
                action = "buy"
            elif signal_dict["action"] == "BUY_NO":
                action = "sell"  # Short YES = buy NO
            else:
                continue

            if self.ev_enabled and not np.isnan(ev_delta_value):
                if action == "buy" and ev_delta_value < self.ev_min_delta_cents:
                    logger.debug(
                        "Skip %s BUY: EV delta %.2f < %.2f",
                        ticker,
                        ev_delta_value,
                        self.ev_min_delta_cents,
                    )
                    continue
                if action == "sell" and ev_delta_value > -self.ev_min_delta_cents:
                    logger.debug(
                        "Skip %s SELL: EV delta %.2f > -%.2f",
                        ticker,
                        ev_delta_value,
                        self.ev_min_delta_cents,
                    )
                    continue

            # Calculate size_fraction using fractional Kelly
            # size_fraction = alpha * kelly_frac, capped by risk limits
            size_fraction = self.kelly_strategy.params.alpha_kelly * signal_dict["kelly_frac"]

            # Cap at max bankroll % per city-day-side
            size_fraction = min(
                size_fraction,
                self.kelly_strategy.params.max_bankroll_pct_city_day_side,
                self.kelly_strategy.params.max_trade_notional_pct,
            )

            # Create signal with diagnostic fields
            reason = (
                f"model edge={signal_dict['edge_cents']:.1f}¢, "
                f"kelly={signal_dict['kelly_frac']:.3f}{ev_reason}"
            )

            signal = Signal(
                timestamp=timestamp,
                market_ticker=ticker,
                action=action,
                edge=signal_dict["edge_cents"],
                confidence=p_for_signal,
                size_fraction=size_fraction,
                reason=reason,
                spread_cents=spread_cents,
                p_market=p_market,
                time_to_close_minutes=time_to_close_minutes,
                price_cents=ask if action == "buy" else bid,
            )

            signals.append(signal)

        logger.info(
            f"Generated {len(signals)} signals from {len(market_data)} markets "
            f"(bankroll: ${bankroll_cents/100:,.2f})"
        )

        return signals

    def get_name(self) -> str:
        """Return strategy name for reporting."""
        model_suffix = f"-{self.model_type}" if self.model_type != "elasticnet" else ""
        return f"ModelKelly{model_suffix} ({self.city}, {self.bracket})"


def main():
    """Demo: Test model-driven strategy adapter."""
    print("\n" + "="*60)
    print("ModelKellyBacktestStrategy Demo")
    print("="*60 + "\n")

    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else "elasticnet"

    # Create strategy
    strategy = ModelKellyBacktestStrategy(
        city="chicago",
        bracket="between",
        models_dir="models/trained",
        model_type=model_type,
    )

    print(f"Strategy: {strategy.get_name()}")
    print(f"Loaded predictions: {len(strategy.predictions)} rows")
    print(f"Unique markets: {len(strategy.predictions['market_ticker'].unique())}")
    print()

    # Show sample predictions
    if len(strategy.predictions) > 0:
        print("Sample predictions:")
        print(strategy.predictions.head(10)[["market_ticker", "timestamp", "p_model", "y_true"]])
        print()

    print("="*60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    main()
