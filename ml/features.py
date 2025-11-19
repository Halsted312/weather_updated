#!/usr/bin/env python3
"""
Feature engineering for Kalshi weather markets.

Extracts multiple feature sets:
- Baseline: bid/ask + strike distance + coarse time features
- Ridge/ElasticNet variants: add spread_pct, microstructure, and VC weather
- NextGen: adds richer price probability, calendar, bracket-context, and
  weather-dynamics features suitable for nonlinear models (CatBoost/ordinal head)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Feature Set Definitions
FEATURE_SET_BASELINE = [
    "yes_mid", "yes_bid", "yes_ask", "spread_cents", "minutes_to_close",
    "temp_now", "temp_to_floor", "temp_to_cap",
    "hour_of_day_local", "day_of_week",
]

FEATURE_SET_A_RIDGE_CONSERVATIVE = FEATURE_SET_BASELINE + [
    "spread_pct",
    "log_minutes_to_close",
    "temp_delta_1h",
]

FEATURE_SET_B_ELASTICNET_RICH = FEATURE_SET_BASELINE + [
    # Microstructure
    "mid_chg_5m", "spread_pct", "vol_15m",
    # Weather dynamics (VC-based, excluded for NYC)
    "temp_rollmax_3h", "temp_rollmin_3h", "temp_delta_1h",
    "dewpoint", "humidity", "windspeed",
]

FEATURE_SET_NEXTGEN = FEATURE_SET_B_ELASTICNET_RICH + [
    # Additional microstructure + liquidity
    "ret_1m", "ret_5m", "ret_15m",
    "vol_5m", "vol_60m",
    "volume_5m", "volume_60m_avg", "volume_ratio",
    "log_minutes_to_close", "minutes_to_close_squared",
    "minutes_x_spread", "minutes_x_temp_dist",
    # Cross-bracket structure
    "mid_group_mean", "mid_group_std", "mid_group_min", "mid_group_max",
    "mid_zscore", "mid_rank_pct", "mid_rel_to_group",
    "mid_to_best_diff", "mid_to_worst_diff",
    "volume_share_in_timestamp",
    # Price-probability transforms
    "market_prob_mid", "market_prob_bid", "market_prob_ask",
    "market_logit_mid", "market_edge_vs_even",
    # Calendar/seasonality
    "minutes_since_midnight_local", "doy_sin", "doy_cos", "is_weekend",
    # Bracket context
    "strike_width", "strike_midpoint", "temp_to_midpoint", "temp_position_in_bracket",
    # Weather dynamics
    "prior_day_tmax", "temp_change_from_yesterday",
]


def get_feature_set(feature_set_name: str) -> List[str]:
    """
    Get list of feature names for a given feature set.

    Args:
        feature_set_name: One of 'baseline', 'ridge_conservative', 'elasticnet_rich'

    Returns:
        List of feature column names
    """
    feature_sets = {
        "baseline": FEATURE_SET_BASELINE,
        "ridge_conservative": FEATURE_SET_A_RIDGE_CONSERVATIVE,
        "elasticnet_rich": FEATURE_SET_B_ELASTICNET_RICH,
        "nextgen": FEATURE_SET_NEXTGEN,
    }

    if feature_set_name not in feature_sets:
        raise ValueError(f"Unknown feature set: {feature_set_name}. "
                        f"Choose from: {list(feature_sets.keys())}")

    return feature_sets[feature_set_name]


class FeatureBuilder:
    """
    Feature builder for Kalshi weather markets.

    Extracts minimal features from market data and weather observations.
    """

    def __init__(self, city_timezone: str = "America/Chicago"):
        """
        Initialize feature builder.

        Args:
            city_timezone: IANA timezone for the city (default: Chicago)
        """
        self.city_timezone = ZoneInfo(city_timezone)

    def build_features(
        self,
        candles_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        market_metadata: Optional[pd.DataFrame] = None,
        feature_set: str = "baseline",
    ) -> pd.DataFrame:
        """
        Build features from candles and weather data.

        Args:
            candles_df: DataFrame with columns:
                - market_ticker
                - end_period_ts (timestamp)
                - yes_bid_close, yes_ask_close, price_close
                - volume, open_interest (optional)
                - dew_f, humidity_pct, wind_mph (optional, for VC features)
            weather_df: DataFrame with columns:
                - date (date)
                - tmax_f (observed temperature)
            market_metadata: Optional DataFrame with columns:
                - ticker (or market_ticker)
                - close_time
                - strike_type, floor_strike, cap_strike
            feature_set: Feature set to build ("baseline", "ridge_conservative", "elasticnet_rich")

        Returns:
            DataFrame with features based on feature_set
        """
        if candles_df.empty:
            logger.warning("Empty candles_df provided")
            return pd.DataFrame()

        # Copy to avoid modifying original
        df = candles_df.copy()

        # Ensure timestamp column
        if "end_period_ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["end_period_ts"])
        elif "timestamp" not in df.columns:
            raise ValueError("candles_df must have 'end_period_ts' or 'timestamp' column")

        # Market features (always)
        df = self._add_market_features(df)
        df = self._add_cross_bracket_features(df)

        # Time features (always)
        df = self._add_time_features(df)

        # Create event_date early (needed by dataset.py and weather functions)
        if "timestamp_local" in df.columns:
            df["event_date"] = df["timestamp_local"].dt.date

        # Weather features (requires metadata) (always)
        if market_metadata is not None and not market_metadata.empty:
            df = self._add_weather_features(df, weather_df, market_metadata)
            df = self._add_bracket_context_features(df)
        else:
            logger.warning("No market_metadata provided, skipping weather features")
            df["temp_now"] = np.nan
            df["temp_to_floor"] = np.nan
            df["temp_to_cap"] = np.nan

        # Minutes to close (requires metadata) (always)
        if market_metadata is not None and not market_metadata.empty:
            df = self._add_minutes_to_close(df, market_metadata)
        else:
            logger.warning("No market_metadata provided, skipping minutes_to_close")
            df["minutes_to_close"] = np.nan

        # Add feature-set-specific features
        if feature_set == "ridge_conservative":
            # Feature Set A: baseline + spread_pct, log_minutes_to_close, temp_delta_1h
            df = self._add_best_features(df)  # Includes spread_pct, log_minutes_to_close
            df = self._add_vc_rich_features(df)  # For temp_delta_1h only

        elif feature_set == "elasticnet_rich":
            # Feature Set B: baseline + microstructure + VC weather dynamics
            df = self._add_microstructure_features(df)  # mid_chg_5m, spread_pct, vol_15m
            df = self._add_time_interactions(df)  # log_minutes_to_close
            df = self._add_vc_rich_features(df)  # All VC features

        elif feature_set == "nextgen":
            df = self._add_microstructure_features(df)
            df = self._add_time_interactions(df)
            df = self._add_vc_rich_features(df)
            df = self._add_weather_dynamics(df, weather_df=weather_df)
            df = self._add_price_probability_features(df)
            df = self._add_calendar_enhancements(df)

        # else: baseline - no additional features

        # CRITICAL: Ensure event_date exists (required by dataset.py)
        if "event_date" not in df.columns and "timestamp_local" in df.columns:
            df["event_date"] = df["timestamp_local"].dt.date
            logger.warning("event_date missing at end of build_features, recreated from timestamp_local")

        return df

    def _add_cross_bracket_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add group-wise statistics across all markets that share the same timestamp.

        These capture the shape of the local book (which brackets are favored) and
        allow the model to reason about relative mispricings rather than absolute prices.
        """
        if "timestamp" not in df.columns or "yes_mid" not in df.columns:
            df["mid_group_mean"] = np.nan
            df["mid_group_std"] = np.nan
            df["mid_group_min"] = np.nan
            df["mid_group_max"] = np.nan
            df["mid_zscore"] = np.nan
            df["mid_rank_pct"] = np.nan
            df["mid_rel_to_group"] = np.nan
            df["mid_to_best_diff"] = np.nan
            df["mid_to_worst_diff"] = np.nan
            df["volume_share_in_timestamp"] = np.nan
            return df

        group = df.groupby("timestamp")
        df["mid_group_mean"] = group["yes_mid"].transform("mean")
        df["mid_group_std"] = group["yes_mid"].transform("std").fillna(0.0)
        df["mid_group_min"] = group["yes_mid"].transform("min")
        df["mid_group_max"] = group["yes_mid"].transform("max")

        df["mid_rel_to_group"] = df["yes_mid"] - df["mid_group_mean"]
        denom = df["mid_group_std"].replace(0, np.nan)
        df["mid_zscore"] = df["mid_rel_to_group"] / denom
        df["mid_zscore"] = df["mid_zscore"].fillna(0.0).clip(-10, 10)

        df["mid_rank_pct"] = group["yes_mid"].rank(method="min", pct=True)
        df["mid_rank_pct"] = df["mid_rank_pct"].fillna(0.0)

        df["mid_to_best_diff"] = df["yes_mid"] - df["mid_group_max"]
        df["mid_to_worst_diff"] = df["yes_mid"] - df["mid_group_min"]

        if "volume" in df.columns:
            volume_sum = group["volume"].transform("sum").replace(0, np.nan)
            df["volume_share_in_timestamp"] = df["volume"] / volume_sum
        else:
            df["volume_share_in_timestamp"] = np.nan

        return df

    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market price features."""
        # Support both bid/ask format and OHLC format

        # Case 1: Explicit bid/ask columns
        if "yes_bid_close" in df.columns and "yes_ask_close" in df.columns:
            df["yes_bid"] = df["yes_bid_close"]
            df["yes_ask"] = df["yes_ask_close"]
            df["yes_mid"] = (df["yes_bid"] + df["yes_ask"]) / 2.0
            df["spread_cents"] = df["yes_ask"] - df["yes_bid"]

        # Case 2: OHLC format (database schema) without reliable quotes
        elif "close" in df.columns:
            # Use close as mid price and mark bid/ask as missing rather than
            # synthesizing from trade highs/lows (which can include outliers).
            df["yes_mid"] = df["close"]
            df["yes_bid"] = np.nan
            df["yes_ask"] = np.nan
            df["spread_cents"] = np.nan

        # Case 3: Fallback to price_close
        elif "price_close" in df.columns:
            logger.warning("Using price_close as fallback for yes_mid")
            df["yes_mid"] = df["price_close"]
            df["yes_bid"] = np.nan
            df["yes_ask"] = np.nan
            df["spread_cents"] = np.nan

        else:
            raise ValueError("candles_df must have bid/ask columns, OHLC columns, or price_close")

        # Ensure spread is non-negative but preserve NaN when quotes are missing
        df["spread_cents"] = df["spread_cents"].clip(lower=0)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Normalize timestamps to explicit UTC even if already tz-aware
        timestamps_utc = pd.to_datetime(df["timestamp"], utc=True)
        df["timestamp"] = timestamps_utc

        # Convert to local timezone for downstream features
        df["timestamp_local"] = timestamps_utc.dt.tz_convert(self.city_timezone)

        # hour_of_day_local (0-23)
        df["hour_of_day_local"] = df["timestamp_local"].dt.hour

        # day_of_week (0=Monday, 6=Sunday)
        df["day_of_week"] = df["timestamp_local"].dt.dayofweek

        return df

    def _add_weather_features(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame],
        market_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add weather features (temp_now, temp_to_floor, temp_to_cap)."""
        temp_source = None

        # Prefer minute-level temps already joined to the candles
        if "temp_f" in df.columns:
            temp_source = pd.to_numeric(df["temp_f"], errors="coerce")
        elif weather_df is not None and not weather_df.empty:
            weather_df = weather_df.copy()
            if "date" not in weather_df.columns:
                raise ValueError("weather_df must have 'date' column when temp_f missing")
            weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
            df["event_date"] = df["timestamp_local"].dt.date
            df = df.merge(
                weather_df[["date", "tmax_f"]].rename(columns={"date": "event_date"}),
                on="event_date",
                how="left",
            )
            temp_source = pd.to_numeric(df["tmax_f"], errors="coerce")
            df.drop(columns=["tmax_f"], inplace=True, errors="ignore")
        else:
            logger.warning("No temperature source found; temp_now will be NaN")

        df["temp_now"] = temp_source
        df.drop(columns=["temp_f"], inplace=True, errors="ignore")

        if "temp_now" in df.columns:
            df["temp_now"] = df["temp_now"].astype(float)
            df["temp_now"] = df["temp_now"].clip(lower=-80.0, upper=150.0)
        else:
            df["temp_now"] = np.nan

        # Get strikes from market_metadata
        # Support both "ticker" and "market_ticker" column names
        metadata = market_metadata.copy()
        if "ticker" in metadata.columns and "market_ticker" not in metadata.columns:
            metadata["market_ticker"] = metadata["ticker"]
        elif "market_ticker" not in metadata.columns:
            raise ValueError("market_metadata must have 'ticker' or 'market_ticker' column")

        # Merge with metadata (left join)
        df = df.merge(
            metadata[["market_ticker", "strike_type", "floor_strike", "cap_strike"]],
            on="market_ticker",
            how="left"
        )

        # temp_to_floor = floor_strike - temp_now (for "greater" markets)
        # temp_to_cap = cap_strike - temp_now (for "less" markets)
        # For "between" markets: both are relevant
        df["temp_to_floor"] = df["floor_strike"] - df["temp_now"]
        df["temp_to_cap"] = df["cap_strike"] - df["temp_now"]

        return df

    def _add_bracket_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add strike-derived context such as width, midpoint, and normalized temp.
        """
        floor_series = df["floor_strike"] if "floor_strike" in df.columns else pd.Series(np.nan, index=df.index)
        cap_series = df["cap_strike"] if "cap_strike" in df.columns else pd.Series(np.nan, index=df.index)

        floor = pd.to_numeric(floor_series, errors="coerce")
        cap = pd.to_numeric(cap_series, errors="coerce")

        strike_width = (cap - floor).where(~(cap.isna() | floor.isna()))
        strike_width = strike_width.fillna(0.0)
        df["strike_width"] = strike_width

        strike_midpoint = np.where(
            strike_width != 0,
            floor + (strike_width / 2.0),
            floor,
        )
        df["strike_midpoint"] = np.nan_to_num(strike_midpoint, nan=df.get("temp_now", 0))

        if "temp_now" in df.columns:
            df["temp_to_midpoint"] = df["strike_midpoint"] - df["temp_now"]
        else:
            df["temp_to_midpoint"] = 0.0

        denom = strike_width.replace({0: np.nan})
        if "temp_now" in df.columns and "floor_strike" in df.columns:
            df["temp_position_in_bracket"] = np.nan_to_num(
                (df["temp_now"] - floor) / denom,
                nan=0.0,
            )
            df["temp_position_in_bracket"] = df["temp_position_in_bracket"].clip(-5, 5)
        else:
            df["temp_position_in_bracket"] = 0.0

        return df

    def _add_price_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features derived from market-implied probabilities.
        """
        price_col = "yes_mid" if "yes_mid" in df.columns else "close"
        prob = pd.to_numeric(df.get(price_col), errors="coerce") / 100.0
        prob = prob.clip(1e-6, 1 - 1e-6)
        df["market_prob_mid"] = prob
        df["market_edge_vs_even"] = (prob - 0.5) * 100.0
        df["market_logit_mid"] = np.log(prob / (1 - prob))

        if "yes_bid" in df.columns and "yes_ask" in df.columns:
            df["market_prob_bid"] = (pd.to_numeric(df["yes_bid"], errors="coerce") / 100.0).clip(0, 1)
            df["market_prob_ask"] = (pd.to_numeric(df["yes_ask"], errors="coerce") / 100.0).clip(0, 1)
        else:
            df["market_prob_bid"] = np.nan
            df["market_prob_ask"] = np.nan

        return df

    def _add_calendar_enhancements(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add richer calendar/seasonality signals beyond hour/day of week.
        """
        if "timestamp_local" not in df.columns:
            logger.warning("timestamp_local missing, skipping calendar enhancements")
            df["minutes_since_midnight_local"] = np.nan
            df["doy_sin"] = np.nan
            df["doy_cos"] = np.nan
            df["is_weekend"] = 0
            return df

        ts_local = df["timestamp_local"]
        df["minutes_since_midnight_local"] = ts_local.dt.hour * 60 + ts_local.dt.minute

        day_of_year = ts_local.dt.dayofyear.astype(float)
        df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366.0)
        df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366.0)

        df["is_weekend"] = ts_local.dt.dayofweek.isin([5, 6]).astype(int)

        return df

    def _add_minutes_to_close(
        self,
        df: pd.DataFrame,
        market_metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add minutes_to_close feature."""
        # Support both "ticker" and "market_ticker" column names
        metadata = market_metadata.copy()
        if "ticker" in metadata.columns and "market_ticker" not in metadata.columns:
            metadata["market_ticker"] = metadata["ticker"]
        elif "market_ticker" not in metadata.columns:
            raise ValueError("market_metadata must have 'ticker' or 'market_ticker' column")

        # Ensure close_time is datetime
        if "close_time" not in metadata.columns:
            raise ValueError("market_metadata must have 'close_time' column")

        metadata["close_time"] = pd.to_datetime(metadata["close_time"], utc=True)

        # Merge with metadata (left join)
        df = df.merge(
            metadata[["market_ticker", "close_time"]],
            on="market_ticker",
            how="left"
        )

        close_times = pd.to_datetime(df["close_time"], utc=True)
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        # minutes_to_close = (close_time - timestamp) in minutes
        df["minutes_to_close"] = (close_times - timestamps).dt.total_seconds() / 60.0
        df["minutes_to_close"] = df["minutes_to_close"].clip(lower=0)  # Cap at 0 (already closed)

        if "event_date" not in df.columns:
            df["event_date"] = close_times.dt.tz_convert(self.city_timezone).dt.date

        # Drop intermediate columns
        df.drop(columns=["close_time"], inplace=True, errors="ignore")

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add microstructure features: returns, volatility, volume.

        Requires columns: yes_mid (or close), volume, market_ticker, timestamp
        """
        # Sort by market and timestamp for rolling calculations
        df = df.sort_values(["market_ticker", "timestamp"])

        # Use yes_mid or close for returns
        price_col = "yes_mid" if "yes_mid" in df.columns else "close"

        # Price returns (percent change)
        df["ret_1m"] = df.groupby("market_ticker")[price_col].pct_change(1).fillna(0)
        df["ret_5m"] = df.groupby("market_ticker")[price_col].pct_change(5).fillna(0)
        df["ret_15m"] = df.groupby("market_ticker")[price_col].pct_change(15).fillna(0)

        # Rolling volatility (std of returns)
        df["vol_5m"] = df.groupby("market_ticker")["ret_1m"].transform(
            lambda x: x.rolling(5, min_periods=2).std()
        ).fillna(0)
        df["vol_15m"] = df.groupby("market_ticker")["ret_1m"].transform(
            lambda x: x.rolling(15, min_periods=5).std()
        ).fillna(0)
        df["vol_60m"] = df.groupby("market_ticker")["ret_1m"].transform(
            lambda x: x.rolling(60, min_periods=10).std()
        ).fillna(0)

        # Volume features (if available)
        if "volume" in df.columns:
            # Volume sum over windows
            df["volume_5m"] = df.groupby("market_ticker")["volume"].transform(
                lambda x: x.rolling(5, min_periods=1).sum()
            ).fillna(0)
            df["volume_60m_avg"] = df.groupby("market_ticker")["volume"].transform(
                lambda x: x.rolling(60, min_periods=10).mean()
            ).fillna(0)

            # Volume momentum (recent vs historical)
            df["volume_ratio"] = (df["volume_5m"] / (df["volume_60m_avg"] + 1e-9)).clip(0, 100)
        else:
            df["volume_5m"] = 0
            df["volume_60m_avg"] = 0
            df["volume_ratio"] = 0

        # Spread pressure (relative spread)
        if "spread_cents" in df.columns and price_col in df.columns:
            df["spread_pct"] = (df["spread_cents"] / (df[price_col] + 1e-9)).clip(0, 1)
        else:
            df["spread_pct"] = 0

        return df

    def _add_time_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-to-close interaction features.

        Requires columns: minutes_to_close, spread_cents, temp_to_floor
        """
        if "minutes_to_close" not in df.columns:
            logger.warning("minutes_to_close not found, skipping time interactions")
            df["log_minutes_to_close"] = np.nan
            df["minutes_to_close_squared"] = np.nan
            df["minutes_x_spread"] = np.nan
            df["minutes_x_temp_dist"] = np.nan
            return df

        # Log of minutes to close (handles time decay non-linearity)
        df["log_minutes_to_close"] = np.log1p(df["minutes_to_close"])

        # Squared term (captures urgency near close)
        df["minutes_to_close_squared"] = df["minutes_to_close"] ** 2

        # Interaction: minutes × spread (spread tends to widen near close)
        if "spread_cents" in df.columns:
            df["minutes_x_spread"] = df["minutes_to_close"] * df["spread_cents"]
        else:
            df["minutes_x_spread"] = 0

        # Interaction: minutes × distance to strike (less uncertainty near close)
        if "temp_to_floor" in df.columns:
            df["minutes_x_temp_dist"] = df["minutes_to_close"] * df["temp_to_floor"].abs()
        else:
            df["minutes_x_temp_dist"] = 0

        return df

    def _add_weather_dynamics(
        self,
        df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Add weather dynamics features: prior day tmax, temp changes.

        Requires columns: event_date (from timestamp_local)
        Requires weather_df with columns: date, tmax_f
        """
        if "event_date" not in df.columns:
            logger.warning("Cannot add weather dynamics, missing event_date")
            df["prior_day_tmax"] = 0.0
            df["temp_change_from_yesterday"] = 0.0
            return df

        weather_input = pd.DataFrame()
        if weather_df is not None and not weather_df.empty:
            weather_input = weather_df.copy()
            weather_input["date"] = pd.to_datetime(weather_input["date"]).dt.date
        elif "temp_now" in df.columns:
            # Approximate using intraday observations
            weather_input = (
                df[["event_date", "temp_now"]]
                .dropna()
                .groupby("event_date")["temp_now"]
                .max()
                .reset_index()
                .rename(columns={"temp_now": "tmax_f", "event_date": "date"})
            )

        if weather_input.empty:
            df["prior_day_tmax"] = 0.0
            df["temp_change_from_yesterday"] = 0.0
            return df

        weather_input["prior_day"] = weather_input["date"] + timedelta(days=1)
        weather_prior = weather_input[["prior_day", "tmax_f"]].rename(
            columns={"prior_day": "event_date", "tmax_f": "prior_day_tmax"}
        )

        df = df.merge(weather_prior, on="event_date", how="left")
        df["prior_day_tmax"] = df["prior_day_tmax"].fillna(0.0)

        if "temp_now" in df.columns:
            df["temp_change_from_yesterday"] = df["temp_now"] - df["prior_day_tmax"]
            df["temp_change_from_yesterday"] = df["temp_change_from_yesterday"].fillna(0.0)
        else:
            df["temp_change_from_yesterday"] = 0.0

        return df

    def _add_single_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ONLY 1 feature: log_minutes_to_close.

        Non-linear time decay - theoretically sound transformation.
        """
        if "minutes_to_close" in df.columns:
            df["log_minutes_to_close"] = np.log1p(df["minutes_to_close"])
        else:
            logger.warning("minutes_to_close not found, skipping log_minutes_to_close")
            df["log_minutes_to_close"] = np.nan

        return df

    def _add_best_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ONLY the 3 best features (minimal expansion from baseline).

        Features:
        1. spread_pct - relative spread (execution quality indicator)
        2. log_minutes_to_close - non-linear time decay
        3. vol_15m - 15-minute price volatility (uncertainty measure)
        """
        # Sort by market and timestamp for rolling calculations
        df = df.sort_values(["market_ticker", "timestamp"])

        # 1. Spread percentage (relative spread)
        price_col = "yes_mid" if "yes_mid" in df.columns else "close"
        if "spread_cents" in df.columns and price_col in df.columns:
            df["spread_pct"] = (df["spread_cents"] / (df[price_col] + 1e-9)).clip(0, 1)
        else:
            df["spread_pct"] = 0

        # 2. Log of minutes to close (non-linear time decay)
        if "minutes_to_close" in df.columns:
            df["log_minutes_to_close"] = np.log1p(df["minutes_to_close"])
        else:
            logger.warning("minutes_to_close not found, skipping log_minutes_to_close")
            df["log_minutes_to_close"] = np.nan

        # 3. 15-minute price volatility (short-term uncertainty)
        # Calculate 1-minute returns first
        df["ret_1m"] = df.groupby("market_ticker")[price_col].pct_change(1).fillna(0)

        # Rolling std over 15 minutes
        df["vol_15m"] = df.groupby("market_ticker")["ret_1m"].transform(
            lambda x: x.rolling(15, min_periods=5).std()
        ).fillna(0)

        # Drop intermediate column
        df.drop(columns=["ret_1m"], inplace=True, errors="ignore")

        return df

    def _add_vc_rich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Visual Crossing weather features and microstructure features.

        Features:
        - mid_chg_5m: 5-minute mid price change (in cents)
        - temp_delta_1h: 1-hour temperature change (from VC minute data)
        - temp_rollmax_3h: 3-hour rolling max temperature
        - temp_rollmin_3h: 3-hour rolling min temperature
        - dewpoint: Raw dewpoint from VC (dew_f)
        - humidity: Raw humidity from VC (humidity_pct)
        - windspeed: Raw wind speed from VC (wind_mph)

        Requires columns in candles_df:
        - dew_f, humidity_pct, wind_mph (from VC minute_obs_1m)
        - temp_now (already computed from temp_f)
        """
        # Sort by market and timestamp for rolling calculations
        df = df.sort_values(["market_ticker", "timestamp"])

        # 1. Mid price change (5 minutes)
        price_col = "yes_mid" if "yes_mid" in df.columns else "close"
        if price_col in df.columns:
            df["mid_chg_5m"] = df.groupby("market_ticker")[price_col].transform(
                lambda x: x - x.shift(5)
            ).fillna(0)
        else:
            df["mid_chg_5m"] = 0

        # 2. Temperature delta (1 hour = 12 5-minute intervals)
        if "temp_now" in df.columns:
            df["temp_delta_1h"] = df.groupby("market_ticker")["temp_now"].transform(
                lambda x: x - x.shift(12)
            ).fillna(0)
        else:
            logger.warning("temp_now not found, skipping temp_delta_1h")
            df["temp_delta_1h"] = 0

        # 3. Temperature rolling max/min (3 hours = 36 5-minute intervals)
        if "temp_now" in df.columns:
            df["temp_rollmax_3h"] = df.groupby("market_ticker")["temp_now"].transform(
                lambda x: x.rolling(36, min_periods=12).max()
            ).fillna(df["temp_now"])  # Fallback to current temp if not enough data

            df["temp_rollmin_3h"] = df.groupby("market_ticker")["temp_now"].transform(
                lambda x: x.rolling(36, min_periods=12).min()
            ).fillna(df["temp_now"])  # Fallback to current temp if not enough data
        else:
            df["temp_rollmax_3h"] = 0
            df["temp_rollmin_3h"] = 0

        # 4. VC raw features (passthrough from minute_obs_1m)
        # These come from the SQL query in dataset.py
        if "dew_f" in df.columns:
            df["dewpoint"] = df["dew_f"]
        else:
            logger.warning("dew_f not found in candles, NYC or missing VC data")
            df["dewpoint"] = 0

        if "humidity_pct" in df.columns:
            df["humidity"] = df["humidity_pct"]
        else:
            logger.warning("humidity_pct not found in candles, NYC or missing VC data")
            df["humidity"] = 0

        if "wind_mph" in df.columns:
            df["windspeed"] = df["wind_mph"]
        else:
            logger.warning("wind_mph not found in candles, NYC or missing VC data")
            df["windspeed"] = 0

        return df

    def get_feature_columns(self, feature_set: str = "baseline") -> List[str]:
        """
        Return list of feature columns (for ML training).

        Args:
            feature_set: One of "baseline", "ridge_conservative", "elasticnet_rich"

        Returns:
            List of feature column names
        """
        return get_feature_set(feature_set)

    def validate_features(self, df: pd.DataFrame, feature_set: str = "baseline") -> Dict[str, int]:
        """
        Validate features and return diagnostics.

        Args:
            df: DataFrame with features
            feature_set: Feature set to validate

        Returns:
            Dict with counts of missing values per feature
        """
        feature_cols = self.get_feature_columns(feature_set)
        missing_counts = {}

        for col in feature_cols:
            if col not in df.columns:
                missing_counts[col] = len(df)  # Entire column missing
            else:
                missing_counts[col] = df[col].isna().sum()

        return missing_counts


def main():
    """Demo: Feature engineering."""
    print("\n" + "="*60)
    print("Feature Engineering Demo")
    print("="*60 + "\n")

    # Create sample data
    from datetime import date

    # Sample candles (3 markets, 2 timestamps each)
    candles_df = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "end_period_ts": datetime(2025, 8, 10, 14, 0, 0),
            "yes_bid_close": 48,
            "yes_ask_close": 52,
            "price_close": 50,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "end_period_ts": datetime(2025, 8, 10, 15, 0, 0),
            "yes_bid_close": 49,
            "yes_ask_close": 53,
            "price_close": 51,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-G75",
            "end_period_ts": datetime(2025, 8, 10, 14, 0, 0),
            "yes_bid_close": 65,
            "yes_ask_close": 69,
            "price_close": 67,
        },
    ])

    # Sample weather
    weather_df = pd.DataFrame([
        {"date": date(2025, 8, 10), "tmax_f": 78.0},
        {"date": date(2025, 8, 11), "tmax_f": 82.0},
    ])

    # Sample market metadata
    market_metadata = pd.DataFrame([
        {
            "market_ticker": "KXHIGHCHI-25AUG10-B80",
            "close_time": datetime(2025, 8, 10, 18, 0, 0),
            "strike_type": "between",
            "floor_strike": 80,
            "cap_strike": 81,
        },
        {
            "market_ticker": "KXHIGHCHI-25AUG10-G75",
            "close_time": datetime(2025, 8, 10, 18, 0, 0),
            "strike_type": "greater",
            "floor_strike": 75,
            "cap_strike": None,
        },
    ])

    # Build features
    fb = FeatureBuilder(city_timezone="America/Chicago")
    features_df = fb.build_features(candles_df, weather_df, market_metadata)

    print(f"Built {len(features_df)} feature rows from {len(candles_df)} candles")
    print(f"\nFeature columns: {fb.get_feature_columns()}")
    print(f"\nSample features:")
    print(features_df[["market_ticker", "timestamp", "yes_mid", "spread_cents",
                       "minutes_to_close", "temp_now", "temp_to_floor",
                       "hour_of_day_local"]].head())

    # Validate features
    missing = fb.validate_features(features_df)
    print(f"\nMissing value counts:")
    for col, count in missing.items():
        if count > 0:
            print(f"  {col}: {count} / {len(features_df)}")

    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
