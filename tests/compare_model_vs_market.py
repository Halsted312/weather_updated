#!/usr/bin/env python3
"""
Compare Model Predictions vs Kalshi Market Prices

Manual test script that:
1. Loads the trained Chicago model
2. Gets current observations from the database
3. Runs inference to get bracket probabilities
4. Fetches Kalshi order book prices via REST API
5. Compares and outputs the results

Usage:
    .venv/bin/python tests/compare_model_vs_market.py
"""

import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.connection import get_db_session
from src.kalshi.client import KalshiClient
from src.config.cities import CITIES
from models.data.loader import load_inference_snapshot_data, load_multi_horizon_forecasts
from models.data.snapshot import build_snapshot
from models.features.forecast import (
    compute_multi_horizon_features,
    get_forecast_evolution_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CITY = "chicago"
MODEL_PATH = Path("models/saved/chicago_v3_no_leakage/ordinal_catboost_market_clock_tod_v1.pkl")
CHICAGO_TZ = ZoneInfo("America/Chicago")


def load_model(model_path: Path) -> dict:
    """Load the trained model from pickle file."""
    logger.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    logger.info(f"Model loaded: {len(model_data['classifiers'])} classifiers, "
                f"delta range [{model_data['min_delta']}, {model_data['max_delta']}]")
    return model_data


def get_current_observations(session, city: str, event_date: date, cutoff_time: datetime) -> dict:
    """Load current observations and forecasts from database."""
    logger.info(f"Loading observations for {city} up to {cutoff_time}")

    data = load_inference_snapshot_data(
        session=session,
        city=city,
        event_date=event_date,
        cutoff_time=cutoff_time,
        include_candles=False,
        include_city_obs=False,
    )
    return data


def build_features_for_inference(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    obs_df: pd.DataFrame,
    fcst_daily: Optional[dict],
    fcst_hourly_df: Optional[pd.DataFrame],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build features for the model using the same pattern as training."""

    # Build snapshot using the unified function
    features = build_snapshot(
        city=city,
        event_date=event_date,
        cutoff_time=cutoff_time,
        obs_df=obs_df,
        fcst_daily=fcst_daily,
        fcst_hourly_df=fcst_hourly_df,
        settle_f=None,  # Not available at inference time
        include_labels=False,
    )

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Fill NaN with median (match training)
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    return df[feature_cols]


def predict_delta_probs(model_data: dict, X: pd.DataFrame) -> dict[int, float]:
    """Run ordinal model inference to get delta probabilities."""
    classifiers = model_data['classifiers']
    thresholds = model_data['thresholds']
    calibrators = model_data.get('calibrators', {})
    calibrate_method = model_data.get('calibrate', 'none')
    min_delta = model_data['min_delta']
    max_delta = model_data['max_delta']

    n = len(X)
    n_thresholds = len(thresholds)

    # Get P(delta >= k) for each threshold
    exceedance_probs = np.zeros((n, n_thresholds))

    for i, k in enumerate(thresholds):
        clf = classifiers[k]
        if isinstance(clf, dict) and clf.get("type") == "constant":
            exceedance_probs[:, i] = clf["prob"]
        else:
            raw_proba = clf.predict_proba(X)[:, 1]

            # Apply calibration if available
            if k in calibrators:
                calibrator = calibrators[k]
                if calibrate_method == 'isotonic':
                    exceedance_probs[:, i] = calibrator.predict(raw_proba)
                else:  # sigmoid
                    exceedance_probs[:, i] = calibrator.predict_proba(
                        raw_proba.reshape(-1, 1)
                    )[:, 1]
            else:
                exceedance_probs[:, i] = raw_proba

    # Enforce monotonicity: P(delta >= k) should decrease with k
    for i in range(1, n_thresholds):
        exceedance_probs[:, i] = np.minimum(
            exceedance_probs[:, i],
            exceedance_probs[:, i-1]
        )

    # Convert to class probabilities
    delta_classes = list(range(min_delta, max_delta + 1))
    n_classes = len(delta_classes)
    proba = np.zeros((n, n_classes))

    # P(delta = min_class) = 1 - P(delta >= min_class + 1)
    proba[:, 0] = 1.0 - exceedance_probs[:, 0]

    # P(delta = k) = P(delta >= k) - P(delta >= k+1)
    for i in range(n_thresholds - 1):
        proba[:, i+1] = exceedance_probs[:, i] - exceedance_probs[:, i+1]

    # P(delta = max_class) = P(delta >= max_class)
    proba[:, -1] = exceedance_probs[:, -1]

    # Clip and normalize
    proba = np.clip(proba, 0, 1)
    row_sums = proba.sum(axis=1, keepdims=True)
    proba = proba / np.maximum(row_sums, 1e-10)

    # Convert to dict
    return {delta_classes[i]: proba[0, i] for i in range(n_classes)}


def delta_probs_to_settle_probs(delta_probs: dict[int, float], t_base: int) -> dict[int, float]:
    """Convert delta probabilities to settlement temperature probabilities."""
    settle_probs = {}
    for delta, prob in delta_probs.items():
        settle_temp = t_base + delta
        settle_probs[settle_temp] = settle_probs.get(settle_temp, 0) + prob
    return settle_probs


def settle_probs_to_bracket_probs(
    settle_probs: dict[int, float],
    markets: list[dict],
) -> dict[str, dict]:
    """Convert settlement temp probs to bracket win probabilities."""
    bracket_probs = {}

    for market in markets:
        ticker = market['ticker']
        strike_type = market.get('strike_type', '')
        floor_strike = market.get('floor_strike')
        cap_strike = market.get('cap_strike')

        # Calculate probability based on strike type
        prob = 0.0

        if strike_type == 'less':
            # P(settle < cap_strike)
            for temp, p in settle_probs.items():
                if temp < cap_strike:
                    prob += p
        elif strike_type == 'greater':
            # P(settle >= floor_strike)
            for temp, p in settle_probs.items():
                if temp >= floor_strike:
                    prob += p
        elif strike_type == 'between':
            # P(floor_strike <= settle < cap_strike)
            for temp, p in settle_probs.items():
                if floor_strike <= temp < cap_strike:
                    prob += p

        bracket_probs[ticker] = {
            'prob': prob,
            'strike_type': strike_type,
            'floor_strike': floor_strike,
            'cap_strike': cap_strike,
        }

    return bracket_probs


def get_kalshi_client() -> KalshiClient:
    """Create Kalshi client from environment variables."""
    api_key = os.environ.get('KALSHI_API_KEY')
    private_key_path = os.environ.get('KALSHI_PRIVATE_KEY_PATH')

    if not api_key or not private_key_path:
        raise ValueError(
            "Set KALSHI_API_KEY and KALSHI_PRIVATE_KEY_PATH environment variables"
        )

    return KalshiClient(api_key=api_key, private_key_path=private_key_path)


def get_kalshi_markets(client: KalshiClient, series_ticker: str, event_date: date) -> list[dict]:
    """Get Kalshi markets for the given series and date."""
    logger.info(f"Fetching Kalshi markets for {series_ticker} on {event_date}")

    # Get all markets for the series (returns List[Market] directly)
    markets = client.get_all_markets(series_ticker=series_ticker, status="open")

    # Filter for today's event date
    target_date_str = event_date.strftime("%y%b%d").upper()  # e.g., "25DEC01"

    filtered = []
    for market in markets:
        ticker = market.ticker
        # Check if date is in ticker
        if target_date_str in ticker:
            filtered.append({
                'ticker': ticker,
                'strike_type': getattr(market, 'strike_type', None),
                'floor_strike': getattr(market, 'floor_strike', None),
                'cap_strike': getattr(market, 'cap_strike', None),
                'yes_bid': getattr(market, 'yes_bid', None),
                'yes_ask': getattr(market, 'yes_ask', None),
            })

    logger.info(f"Found {len(filtered)} markets for {event_date}")
    return filtered


def get_orderbook_prices(client: KalshiClient, tickers: list[str]) -> dict[str, dict]:
    """Get orderbook bid/ask for each ticker."""
    prices = {}
    for ticker in tickers:
        try:
            orderbook = client.get_orderbook(ticker, depth=5)

            # Debug: print first orderbook structure
            if len(prices) == 0:
                logger.debug(f"Orderbook structure for {ticker}: {orderbook}")

            # Kalshi orderbook structure: {'yes': [[price, qty], ...], 'no': [[price, qty], ...]}
            # or nested under 'yes'/'no' with 'bid'/'ask' keys
            yes_data = orderbook.get('yes', [])
            no_data = orderbook.get('no', [])

            # Check if it's the simple list format (yes bids) or nested
            if isinstance(yes_data, list) and yes_data:
                # Simple format: yes = [[price, qty], ...]
                best_bid = yes_data[0][0] if yes_data else None
                # For asks, we'd look at 'no' bids (buying NO = selling YES)
                best_ask = (100 - no_data[0][0]) if no_data and isinstance(no_data, list) else None
            elif isinstance(yes_data, dict):
                # Nested format with bid/ask
                yes_bids = yes_data.get('bid', [])
                yes_asks = yes_data.get('ask', [])
                best_bid = yes_bids[0][0] if yes_bids else None
                best_ask = yes_asks[0][0] if yes_asks else None
            else:
                best_bid = None
                best_ask = None

            prices[ticker] = {
                'yes_bid': best_bid,
                'yes_ask': best_ask,
            }
        except Exception as e:
            logger.warning(f"Failed to get orderbook for {ticker}: {e}")
            prices[ticker] = {'yes_bid': None, 'yes_ask': None}

    return prices


def print_forecast_evolution(session, city: str, event_date: date):
    """Load and print multi-horizon forecast evolution (T-6 through T-1)."""
    print("\n" + "-" * 70)
    print("FORECAST EVOLUTION (T-6 to T-1)")
    print("-" * 70)

    # Load multi-horizon forecasts
    fcst_multi = load_multi_horizon_forecasts(
        session=session,
        city_id=city,
        target_date=event_date,
        lead_days=[1, 2, 3, 4, 5, 6],
    )

    # Get summary
    summary = get_forecast_evolution_summary(fcst_multi)

    if not summary.get("forecasts"):
        print("  No multi-horizon forecasts available")
        print("  Run backfill: .venv/bin/python scripts/ingest_vc_hist_forecast_v2.py ...")
        return None

    # Print forecast table
    print(f"{'Lead':<6} {'Basis Date':<12} {'Forecast High':>14}")
    print("-" * 35)

    for lead in [6, 5, 4, 3, 2, 1]:  # Oldest to newest
        fcst = fcst_multi.get(lead)
        if fcst and fcst.get("tempmax_f") is not None:
            basis_date = event_date - timedelta(days=lead)
            temp = fcst["tempmax_f"]
            print(f"T-{lead:<4} {basis_date.strftime('%Y-%m-%d'):<12} {temp:>13.1f}°F")
        else:
            print(f"T-{lead:<4} {'(missing)':<12} {'-':>14}")

    # Print statistics
    stats = summary.get("stats", {})
    if stats:
        print("-" * 35)
        print(f"Mean (consensus):    {stats.get('mean', 0):.1f}°F")
        print(f"Std (uncertainty):   {stats.get('std', 0):.1f}°F")
        drift = stats.get('drift')
        if drift is not None:
            trend = "warming" if drift > 0 else "cooling" if drift < 0 else "steady"
            print(f"Drift (T-1 - T-6):   {drift:+.1f}°F ({trend})")
        print(f"Range:               {stats.get('range', 0):.1f}°F")

    # Compute features
    features = compute_multi_horizon_features(fcst_multi)
    return features


def print_comparison(
    city: str,
    event_date: date,
    cutoff_time: datetime,
    t_base: int,
    expected_settle: float,
    settle_std: float,
    bracket_probs: dict[str, dict],
    orderbook_prices: dict[str, dict],
):
    """Print formatted comparison output."""
    print("\n" + "=" * 70)
    print(f"CHICAGO HIGH TEMP - Model vs Market ({event_date})")
    print("=" * 70)

    ct_time = cutoff_time.strftime("%H:%M CT")
    print(f"Current observed max: {t_base}°F (as of {ct_time})")
    print(f"Model expected settle: {expected_settle:.1f}°F ± {settle_std:.1f}°F")
    print()

    # Sort brackets by floor_strike
    sorted_tickers = sorted(
        bracket_probs.keys(),
        key=lambda t: bracket_probs[t].get('floor_strike') or -999
    )

    print("BRACKET COMPARISON:")
    print(f"{'Ticker':<35} {'Model%':>8} {'Bid%':>6} {'Ask%':>6} {'Edge':>8}")
    print("-" * 70)

    best_edge = -999
    best_ticker = None

    for ticker in sorted_tickers:
        bp = bracket_probs[ticker]
        op = orderbook_prices.get(ticker, {})

        model_prob = bp['prob'] * 100

        bid_cents = op.get('yes_bid')
        ask_cents = op.get('yes_ask')

        bid_pct = bid_cents if bid_cents else "-"
        ask_pct = ask_cents if ask_cents else "-"

        # Calculate edge (model - market bid)
        if bid_cents is not None and bid_cents > 0:
            edge = model_prob - bid_cents
            edge_str = f"{edge:+.1f}%"
            if edge > best_edge:
                best_edge = edge
                best_ticker = ticker
        else:
            edge_str = "-"

        # Format strike info
        strike_info = ""
        if bp['strike_type'] == 'less':
            strike_info = f"<{bp['cap_strike']}"
        elif bp['strike_type'] == 'greater':
            strike_info = f"≥{bp['floor_strike']}"
        elif bp['strike_type'] == 'between':
            strike_info = f"{bp['floor_strike']}-{bp['cap_strike']}"

        # Truncate ticker for display
        short_ticker = ticker[-15:] if len(ticker) > 35 else ticker

        marker = " ← Best" if ticker == best_ticker else ""
        print(f"{short_ticker:<35} {model_prob:>7.1f}% {bid_pct:>5}  {ask_pct:>5}  {edge_str:>7}{marker}")

    print("-" * 70)
    if best_ticker and best_edge > 0:
        print(f"\nBest opportunity: {best_ticker}")
        print(f"Model: {bracket_probs[best_ticker]['prob']*100:.1f}% vs Market Bid: {orderbook_prices[best_ticker]['yes_bid']}%")
        print(f"Edge: {best_edge:.1f}%")
    else:
        print("\nNo positive edge opportunities found.")
    print()


def main():
    # Get current time in Chicago
    now_ct = datetime.now(CHICAGO_TZ)
    event_date = now_ct.date()
    cutoff_time = now_ct.replace(tzinfo=None)

    print(f"\nRunning at {now_ct.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Event date: {event_date}")

    # Load model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run training first with: .venv/bin/python scripts/train_market_clock_tod_v1.py ...")
        return 1

    model_data = load_model(MODEL_PATH)
    feature_cols = model_data['feature_cols']
    min_delta = model_data['min_delta']
    max_delta = model_data['max_delta']

    # Load current observations
    with get_db_session() as session:
        data = get_current_observations(session, CITY, event_date, cutoff_time)

        # Print forecast evolution (T-6 through T-1)
        fcst_multi_features = print_forecast_evolution(session, CITY, event_date)

    obs_df = data.get('obs_df')
    fcst_daily = data.get('fcst_daily')
    fcst_hourly_df = data.get('fcst_hourly_df')

    # Debug: check what data was loaded
    if fcst_daily is None:
        logger.warning("fcst_daily is None - check wx.vc_forecast_daily has T-1 forecasts")
        print("\n⚠️  WARNING: No T-1 forecast data found. Predictions may be inaccurate.")
        print("   To ingest forecasts: .venv/bin/python scripts/ingest_vc_forecast.py --city chicago")
    else:
        logger.info(f"fcst_daily tempmax_f: {fcst_daily.get('tempmax_f')}°F")
    if fcst_hourly_df is not None:
        logger.info(f"fcst_hourly_df: {len(fcst_hourly_df)} rows")
    else:
        logger.warning("fcst_hourly_df is None - forecast error features will be 0")

    if obs_df is None or obs_df.empty:
        print(f"ERROR: No observations found for {CITY} on {event_date}")
        return 1

    # Get current max temp
    t_base = int(round(obs_df['temp_f'].max()))
    print(f"Current observed max: {t_base}°F ({len(obs_df)} observations)")

    # Build features
    try:
        X = build_features_for_inference(
            city=CITY,
            event_date=event_date,
            cutoff_time=cutoff_time,
            obs_df=obs_df,
            fcst_daily=fcst_daily,
            fcst_hourly_df=fcst_hourly_df,
            feature_cols=feature_cols,
        )
        # Debug: check features
        logger.info(f"Feature shape: {X.shape}")
        nan_cols = [c for c in X.columns if X[c].isna().any()]
        if nan_cols:
            logger.warning(f"NaN columns after fillna: {nan_cols}")
        # Show key feature values
        key_features = [c for c in X.columns if c.startswith('err_') or c in ['vc_max_f_sofar', 'snapshot_hour']]
        for col in key_features[:10]:
            logger.info(f"  {col}: {X[col].iloc[0]:.2f}")
    except Exception as e:
        print(f"ERROR building features: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run inference
    delta_probs = predict_delta_probs(model_data, X)

    # Debug: print delta probabilities
    logger.info(f"Delta probs sum: {sum(delta_probs.values()):.4f}")
    top_deltas = sorted(delta_probs.items(), key=lambda x: -x[1])[:5]
    logger.info(f"Top 5 deltas: {[(d, f'{p:.3f}') for d, p in top_deltas]}")

    # Convert to settlement temp probs
    settle_probs = delta_probs_to_settle_probs(delta_probs, t_base)

    # Calculate expected settlement
    expected_settle = sum(temp * prob for temp, prob in settle_probs.items())
    settle_std = np.sqrt(sum((temp - expected_settle)**2 * prob for temp, prob in settle_probs.items()))

    print(f"Model expected settlement: {expected_settle:.1f}°F ± {settle_std:.1f}°F")

    # Get Kalshi markets
    try:
        client = get_kalshi_client()
        city_config = CITIES[CITY]
        series_ticker = city_config.series_ticker

        markets = get_kalshi_markets(client, series_ticker, event_date)

        if not markets:
            print(f"No open markets found for {series_ticker} on {event_date}")
            print("\nModel delta probabilities:")
            for delta, prob in sorted(delta_probs.items()):
                if prob > 0.01:
                    print(f"  delta={delta:+d}: {prob*100:.1f}%")
            return 0

        # Debug: show settle_probs
        logger.info(f"Settle probs (top 5): {sorted(settle_probs.items(), key=lambda x: -x[1])[:5]}")
        logger.info(f"Markets: {markets}")

        # Convert to bracket probs
        bracket_probs = settle_probs_to_bracket_probs(settle_probs, markets)

        # Get orderbook prices
        tickers = [m['ticker'] for m in markets]
        orderbook_prices = get_orderbook_prices(client, tickers)

        # Print comparison
        print_comparison(
            city=CITY,
            event_date=event_date,
            cutoff_time=cutoff_time,
            t_base=t_base,
            expected_settle=expected_settle,
            settle_std=settle_std,
            bracket_probs=bracket_probs,
            orderbook_prices=orderbook_prices,
        )

    except ValueError as e:
        print(f"\nKalshi API not configured: {e}")
        print("\nModel predictions only:")
        print(f"Expected settlement: {expected_settle:.1f}°F ± {settle_std:.1f}°F")
        print(f"\nTop delta probabilities:")
        for delta, prob in sorted(delta_probs.items(), key=lambda x: -x[1])[:10]:
            settle = t_base + delta
            print(f"  delta={delta:+d} (settle={settle}°F): {prob*100:.1f}%")

    except Exception as e:
        print(f"ERROR with Kalshi API: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
