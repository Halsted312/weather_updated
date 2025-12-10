"""
Manual Inference Test Script

Comprehensive test of the prediction pipeline for any city.
Outputs a full report with all model predictions, delta distributions,
bracket probabilities, and key features.

Usage:
    python -m models.inference.manual_test --city chicago
    python -m models.inference.manual_test --city denver --date 2025-12-08
    python -m models.inference.manual_test --all
    python -m models.inference.manual_test --all --yesterday
"""

import argparse
import json
import logging
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.features.pipeline import SnapshotContext, compute_snapshot_features
from models.features.calendar import compute_lag_features
from models.features.base import DELTA_CLASSES, get_feature_columns
from models.data.loader import load_full_inference_data
from models.inference.probability import (
    delta_probs_to_dict,
    delta_probs_to_temp_probs,
    expected_settlement,
    settlement_std,
    confidence_interval,
)
from src.db.connection import get_db_session
from config import live_trader_config as config

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# City timezones
CITY_TIMEZONES = {
    'chicago': 'America/Chicago',
    'austin': 'America/Chicago',
    'denver': 'America/Denver',
    'los_angeles': 'America/Los_Angeles',
    'miami': 'America/New_York',
    'philadelphia': 'America/New_York',
}

CITIES = ['chicago', 'austin', 'denver', 'los_angeles', 'miami', 'philadelphia']


def load_model(city: str) -> OrdinalDeltaTrainer:
    """Load trained model for a city."""
    model_path = Path("models/saved") / city / "ordinal_catboost_optuna.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    trainer = OrdinalDeltaTrainer()
    trainer.load(model_path)
    return trainer


def get_model_metrics(city: str) -> dict:
    """Load model metrics from JSON file."""
    metrics_path = Path("models/saved") / city / f"final_metrics_{city}.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def get_markets(session, city: str, event_date: date) -> pd.DataFrame:
    """Query Kalshi markets for this city/event with latest bid/ask from candles."""
    from sqlalchemy import text

    # Join markets with latest candle to get current bid/ask
    query = text("""
        SELECT m.ticker, m.strike_type, m.floor_strike, m.cap_strike,
               c.yes_bid_close as yes_bid, c.yes_ask_close as yes_ask
        FROM kalshi.markets m
        LEFT JOIN LATERAL (
            SELECT yes_bid_close, yes_ask_close
            FROM kalshi.candles_1m_dense
            WHERE ticker = m.ticker
            ORDER BY bucket_start DESC
            LIMIT 1
        ) c ON true
        WHERE m.city = :city AND m.event_date = :event_date
        ORDER BY m.floor_strike NULLS FIRST
    """)

    result = session.execute(query, {'city': city, 'event_date': event_date})

    rows = []
    for row in result:
        rows.append({
            'ticker': row[0],
            'strike_type': row[1],
            'floor_strike': row[2],
            'cap_strike': row[3],
            'yes_bid': row[4],
            'yes_ask': row[5],
        })

    return pd.DataFrame(rows)


def compute_bracket_prob(delta_probs: dict, t_base: int, floor_strike: Optional[int],
                         cap_strike: Optional[int], strike_type: str) -> float:
    """Compute win probability for a bracket."""
    prob = 0.0
    for delta, p in delta_probs.items():
        settled_temp = t_base + delta

        if strike_type == 'less':
            # Wins if temp <= cap
            if cap_strike is not None and settled_temp <= cap_strike:
                prob += p
        elif strike_type == 'greater':
            # Wins if temp >= floor + 1
            if floor_strike is not None and settled_temp >= floor_strike + 1:
                prob += p
        elif strike_type == 'between':
            # Wins if floor <= temp <= cap
            if (floor_strike is not None and cap_strike is not None and
                floor_strike <= settled_temp <= cap_strike):
                prob += p

    return prob


def run_inference_test(
    city: str,
    event_date: date,
    cutoff_time: Optional[datetime] = None,
    verbose: bool = False,
    output_json: bool = False,
) -> dict:
    """Run full inference test for a city and return results."""

    results = {
        'city': city,
        'event_date': str(event_date),
        'success': False,
        'error': None,
    }

    # Get current time in city timezone if not specified
    if cutoff_time is None:
        city_tz = ZoneInfo(CITY_TIMEZONES[city])
        now = datetime.now(city_tz)
        # Floor to 5-minute intervals
        total_minutes = now.hour * 60 + now.minute
        floored_minutes = (total_minutes // 5) * 5
        cutoff_time = now.replace(
            hour=floored_minutes // 60,
            minute=floored_minutes % 60,
            second=0,
            microsecond=0,
            tzinfo=None  # Remove timezone for DB queries (DB stores naive local times)
        )

    results['cutoff_time'] = str(cutoff_time)

    try:
        # Load model
        model = load_model(city)
        model_metrics = get_model_metrics(city)
        results['model_info'] = {
            'trained_at': model_metrics.get('trained_at'),
            'n_train_days': model_metrics.get('n_train_days'),
            'n_test_days': model_metrics.get('n_test_days'),
            'optuna_trials': model_metrics.get('optuna_trials'),
            'delta_accuracy': model_metrics.get('metrics', {}).get('delta_accuracy'),
            'within_2_rate': model_metrics.get('metrics', {}).get('within_2_rate'),
        }

        with get_db_session() as session:
            # Load data
            data = load_full_inference_data(
                city_id=city,
                event_date=event_date,
                cutoff_time=cutoff_time,
                session=session,
            )

            # Data summary
            results['data_loaded'] = {
                'n_observations': len(data['temps_sofar']),
                'window_start': str(data['window_start']),
                'has_fcst_daily': data['fcst_daily'] is not None,
                'has_fcst_hourly': data['fcst_hourly_df'] is not None,
                'n_multi_horizon': sum(1 for v in (data['fcst_multi'] or {}).values() if v),
                'has_candles': data['candles_df'] is not None,
                'has_city_obs': data['city_obs_df'] is not None,
                'has_noaa': any(
                    data['more_apis'].get(m, {}).get('latest_run')
                    for m in ['nbm', 'hrrr', 'ndfd']
                ) if data['more_apis'] else False,
                'obs_t15_mean': data['obs_t15_mean'],
                'obs_t15_std': data['obs_t15_std'],
            }

            # Current state
            temps_sofar = data['temps_sofar']
            if not temps_sofar:
                raise ValueError("No temperature observations found")

            vc_max_sofar = max(temps_sofar)
            vc_min_sofar = min(temps_sofar)
            t_base = round(vc_max_sofar)

            fcst_max = None
            if data['fcst_daily']:
                fcst_max = data['fcst_daily'].get('tempmax_f')

            results['current_state'] = {
                'vc_max_sofar': round(vc_max_sofar, 1),
                'vc_min_sofar': round(vc_min_sofar, 1),
                't_base': t_base,
                'fcst_max': fcst_max,
                'obs_fcst_gap': round(vc_max_sofar - fcst_max, 1) if fcst_max else None,
            }

            # Build SnapshotContext
            ctx = SnapshotContext(
                city=city,
                event_date=event_date,
                cutoff_time=cutoff_time,
                window_start=data["window_start"],
                temps_sofar=temps_sofar,
                timestamps_sofar=data["timestamps_sofar"],
                obs_df=data["obs_df"],
                fcst_daily=data["fcst_daily"],
                fcst_hourly_df=data["fcst_hourly_df"],
                fcst_multi=data["fcst_multi"],
                candles_df=data["candles_df"],
                city_obs_df=data["city_obs_df"],
                more_apis=data["more_apis"],
                obs_t15_mean_30d_f=data["obs_t15_mean"],
                obs_t15_std_30d_f=data["obs_t15_std"],
                settle_f=None,  # Inference mode
            )

            # Compute features
            features = compute_snapshot_features(ctx, include_labels=False)

            # Add lag features
            lag_df = data.get("lag_data")
            if lag_df is not None and not lag_df.empty:
                lag_fs = compute_lag_features(lag_df, city, event_date)
                features.update(lag_fs.to_dict())

                vc_max_f_lag1 = features.get("vc_max_f_lag1")
                vc_max_f_sofar = features.get("vc_max_f_sofar")
                if vc_max_f_lag1 is not None and vc_max_f_sofar is not None:
                    features["delta_vcmax_lag1"] = vc_max_f_sofar - vc_max_f_lag1

            results['n_features'] = len(features)

            # Check feature parity
            numeric_cols, categorical_cols = get_feature_columns()
            expected = set(numeric_cols + categorical_cols)
            actual = set(features.keys())
            missing = expected - actual
            extra = actual - expected

            results['feature_parity'] = {
                'expected': len(expected),
                'actual': len(actual),
                'missing': len(missing),
                'extra': len(extra),
                'missing_features': sorted(missing)[:10] if missing else [],
            }

            if missing:
                results['warnings'] = results.get('warnings', [])
                results['warnings'].append(f"Missing {len(missing)} features")

            # Run model prediction
            features_df = pd.DataFrame([features])
            delta_probs_array = model.predict_proba(features_df)[0]
            delta_probs = delta_probs_to_dict(delta_probs_array)

            # Compute statistics
            exp_settle = expected_settlement(delta_probs, t_base)
            std = settlement_std(delta_probs)
            ci_low, ci_high = confidence_interval(delta_probs, t_base, level=0.9)

            # Find mode (most likely delta)
            mode_delta = max(delta_probs, key=delta_probs.get)
            mode_prob = delta_probs[mode_delta]

            results['model_prediction'] = {
                'expected_settle': round(exp_settle, 1),
                'settlement_std': round(std, 2),
                'ci_90_low': ci_low,
                'ci_90_high': ci_high,
                'mode_delta': mode_delta,
                'mode_prob': round(mode_prob * 100, 1),
                'mode_temp': t_base + mode_delta,
            }

            # Full delta distribution
            delta_dist = []
            cumulative = 0.0
            for d in DELTA_CLASSES:
                p = delta_probs.get(d, 0)
                cumulative += p
                delta_dist.append({
                    'delta': d,
                    'prob': round(p * 100, 2),
                    'cumulative': round(cumulative * 100, 1),
                    'temp': t_base + d,
                })
            results['delta_distribution'] = delta_dist

            # Get markets and compute bracket probabilities
            markets = get_markets(session, city, event_date)

            bracket_probs = []
            for _, market in markets.iterrows():
                model_prob = compute_bracket_prob(
                    delta_probs, t_base,
                    market.get('floor_strike'),
                    market.get('cap_strike'),
                    market['strike_type']
                )

                yes_bid = market.get('yes_bid')
                yes_ask = market.get('yes_ask')

                implied_prob = None
                edge = None
                if yes_bid is not None and yes_ask is not None:
                    mid_price = (yes_bid + yes_ask) / 2
                    implied_prob = mid_price / 100.0
                    edge = model_prob - implied_prob

                bracket_probs.append({
                    'ticker': market['ticker'],
                    'strike_type': market['strike_type'],
                    'floor': market.get('floor_strike'),
                    'cap': market.get('cap_strike'),
                    'model_prob': round(model_prob * 100, 1),
                    'yes_bid': yes_bid,
                    'yes_ask': yes_ask,
                    'implied_prob': round(implied_prob * 100, 1) if implied_prob else None,
                    'edge': round(edge * 100, 1) if edge else None,
                })

            results['bracket_probs'] = bracket_probs

            # Key features (top 10 by importance if available)
            key_features = {}
            important_features = [
                't_base', 'fcst_prev_max_f', 'obs_fcst_max_gap',
                'temp_rate_last_30min', 'hours_until_fcst_max',
                'humidity_last_obs', 'cloudcover_last_obs',
                'fcst_multi_mean', 'temp_std_last_60min', 'day_fraction'
            ]
            for f in important_features:
                if f in features:
                    val = features[f]
                    if isinstance(val, float):
                        key_features[f] = round(val, 2)
                    else:
                        key_features[f] = val

            results['key_features'] = key_features
            results['success'] = True

    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        logger.exception(f"Inference failed for {city}")

    return results


def print_report(results: dict, verbose: bool = False):
    """Print a formatted report to stdout."""

    city = results['city']
    event_date = results['event_date']

    print("=" * 80)
    print(f"                    INFERENCE TEST REPORT")
    print(f"                    {city} | {event_date}")
    print("=" * 80)
    print()

    if not results['success']:
        print(f"ERROR: {results['error']}")
        print("=" * 80)
        return

    # Data Loaded
    data = results.get('data_loaded', {})
    print("DATA LOADED")
    print("-" * 80)
    print(f"  Snapshot Time:      {results.get('cutoff_time', 'N/A')}")
    print(f"  Window Start:       {data.get('window_start', 'N/A')}")
    print(f"  Observations:       {data.get('n_observations', 0)} temperature readings")
    print(f"  Forecast (T-1):     {'Yes' if data.get('has_fcst_daily') else 'No'}")
    print(f"  Hourly Forecast:    {'Yes' if data.get('has_fcst_hourly') else 'No'}")
    print(f"  Multi-Horizon:      {data.get('n_multi_horizon', 0)}/6 leads")
    print(f"  Candles:            {'Yes' if data.get('has_candles') else 'No'}")
    print(f"  City Obs:           {'Yes' if data.get('has_city_obs') else 'No'}")
    print(f"  NOAA Guidance:      {'Yes' if data.get('has_noaa') else 'No'}")
    print(f"  30-Day Stats:       mean={data.get('obs_t15_mean')}, std={data.get('obs_t15_std')}")
    print()

    # Current State
    state = results.get('current_state', {})
    print("CURRENT STATE")
    print("-" * 80)
    print(f"  Current Max (t_base):    {state.get('t_base')}F")
    print(f"  Current Min:             {state.get('vc_min_sofar')}F")
    print(f"  VC Forecast Max:         {state.get('fcst_max')}F")
    gap = state.get('obs_fcst_gap')
    gap_str = f"{gap:+.1f}F" if gap else "N/A"
    trend = ""
    if gap is not None:
        if gap > 0:
            trend = " (running warmer than forecast)"
        elif gap < 0:
            trend = " (running cooler than forecast)"
    print(f"  Obs-Fcst Gap:            {gap_str}{trend}")
    print()

    # Model Info
    model_info = results.get('model_info', {})
    print("MODEL INFO")
    print("-" * 80)
    print(f"  Trained:            {model_info.get('trained_at', 'N/A')}")
    print(f"  Train Days:         {model_info.get('n_train_days', 'N/A')}")
    print(f"  Optuna Trials:      {model_info.get('optuna_trials', 'N/A')}")
    acc = model_info.get('delta_accuracy')
    w2 = model_info.get('within_2_rate')
    print(f"  Test Accuracy:      {acc*100:.1f}%" if acc else "  Test Accuracy:      N/A")
    print(f"  Within-2 Rate:      {w2*100:.1f}%" if w2 else "  Within-2 Rate:      N/A")
    print()

    # Feature Parity
    parity = results.get('feature_parity', {})
    print("FEATURE PARITY")
    print("-" * 80)
    print(f"  Expected Features:  {parity.get('expected', 0)}")
    print(f"  Actual Features:    {parity.get('actual', 0)}")
    print(f"  Missing:            {parity.get('missing', 0)}")
    if parity.get('missing_features'):
        print(f"  Missing (first 10): {parity['missing_features']}")
    print()

    # Model Prediction
    pred = results.get('model_prediction', {})
    print("MODEL PREDICTION")
    print("-" * 80)
    print(f"  Expected Settlement:     {pred.get('expected_settle')}F")
    print(f"  Settlement Std:          {pred.get('settlement_std')}F")
    print(f"  90% Confidence Interval: [{pred.get('ci_90_low')}, {pred.get('ci_90_high')}]F")
    print(f"  Most Likely Delta:       {pred.get('mode_delta'):+d} ({pred.get('mode_prob')}%)")
    print(f"  Most Likely Temp:        {pred.get('mode_temp')}F")
    print()

    # Delta Distribution
    print("DELTA DISTRIBUTION (25 classes: -12 to +12)")
    print("-" * 80)
    print(f"  {'Delta':>6}  {'Prob':>8}  {'Cumul':>8}  |  {'Temp':>6}  Notes")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  |  {'-'*6}  {'-'*20}")

    t_base = results.get('current_state', {}).get('t_base', 0)
    mode_delta = pred.get('mode_delta')

    dist = results.get('delta_distribution', [])
    # Show significant deltas (prob > 0.1%) or in range [-5, +10]
    for item in dist:
        d = item['delta']
        p = item['prob']
        cum = item['cumulative']
        temp = item['temp']

        # Filter for display (show significant or in reasonable range)
        if verbose or p >= 0.1 or (-5 <= d <= 10):
            notes = ""
            if d == 0:
                notes = "(t_base)"
            elif d == mode_delta:
                notes = "<-- MODE"

            print(f"  {d:+6d}  {p:7.1f}%  {cum:7.1f}%  |  {temp:5d}F  {notes}")
    print()

    # Bracket Probabilities
    brackets = results.get('bracket_probs', [])
    if brackets:
        print("BRACKET PROBABILITIES (from Kalshi markets)")
        print("-" * 80)
        print(f"  {'Ticker':<28}  {'Model':>7}  {'Bid':>5}  {'Ask':>5}  {'Impl':>6}  {'Edge':>6}  Rec")
        print(f"  {'-'*28}  {'-'*7}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*10}")

        for b in brackets:
            ticker = b['ticker']
            model_p = f"{b['model_prob']:5.1f}%"

            bid = f"{b['yes_bid']:5.0f}" if b['yes_bid'] is not None else "  N/A"
            ask = f"{b['yes_ask']:5.0f}" if b['yes_ask'] is not None else "  N/A"
            impl = f"{b['implied_prob']:5.1f}%" if b['implied_prob'] is not None else "   N/A"
            edge = f"{b['edge']:+5.1f}%" if b['edge'] is not None else "   N/A"

            # Recommendation
            rec = ""
            if b['edge'] is not None:
                if b['edge'] > 3:
                    rec = "LONG"
                elif b['edge'] < -3:
                    rec = "SHORT"
                elif abs(b['edge']) > 1:
                    rec = "marginal"
                else:
                    rec = "skip"

            print(f"  {ticker:<28}  {model_p}  {bid}  {ask}  {impl}  {edge}  {rec}")
    else:
        print("BRACKET PROBABILITIES")
        print("-" * 80)
        print("  No markets found in database for this city/date")
    print()

    # Key Features
    key_feats = results.get('key_features', {})
    if key_feats:
        print("KEY FEATURES")
        print("-" * 80)
        for k, v in key_feats.items():
            print(f"  {k:<30}  {v}")
    print()

    # Warnings
    warnings = results.get('warnings', [])
    if warnings:
        print("WARNINGS")
        print("-" * 80)
        for w in warnings:
            print(f"  - {w}")
        print()

    print("=" * 80)
    print("                    END OF REPORT")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Manual inference test for Kalshi weather prediction models"
    )
    parser.add_argument(
        "--city",
        type=str,
        choices=CITIES,
        help="City to test (required unless --all is used)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all 6 cities"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Event date (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--yesterday",
        action="store_true",
        help="Use yesterday as event date"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all 25 delta classes (default: show significant only)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted report"
    )
    parser.add_argument(
        "--skip-denver",
        action="store_true",
        help="Skip Denver (useful if Denver model is broken)"
    )

    args = parser.parse_args()

    # Determine event date
    if args.date:
        event_date = date.fromisoformat(args.date)
    elif args.yesterday:
        event_date = date.today() - timedelta(days=1)
    else:
        event_date = date.today()

    # Determine cities to test
    if args.all:
        cities = CITIES.copy()
        if args.skip_denver:
            cities.remove('denver')
    elif args.city:
        cities = [args.city]
    else:
        parser.error("Either --city or --all must be specified")

    # Run tests
    all_results = []
    for city in cities:
        if not args.json:
            print(f"\n{'='*80}")
            print(f"Testing {city}...")
            print(f"{'='*80}\n")

        results = run_inference_test(
            city=city,
            event_date=event_date,
            verbose=args.verbose,
            output_json=args.json,
        )
        all_results.append(results)

        if not args.json:
            print_report(results, verbose=args.verbose)

    # JSON output
    if args.json:
        print(json.dumps(all_results, indent=2, default=str))

    # Summary for --all
    if args.all and not args.json:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        success_count = sum(1 for r in all_results if r['success'])
        print(f"Tested {len(all_results)} cities, {success_count} successful\n")

        print(f"{'City':<15} {'Success':<10} {'E[Settle]':>10} {'Std':>6} {'CI 90%':>12} {'Mode':>6}")
        print("-" * 65)

        for r in all_results:
            city = r['city']
            if r['success']:
                pred = r.get('model_prediction', {})
                exp = pred.get('expected_settle', 'N/A')
                std = pred.get('settlement_std', 'N/A')
                ci = f"[{pred.get('ci_90_low')},{pred.get('ci_90_high')}]"
                mode = f"{pred.get('mode_delta'):+d}" if pred.get('mode_delta') is not None else "N/A"
                print(f"{city:<15} {'OK':<10} {exp:>9}F {std:>5}F {ci:>12} {mode:>6}")
            else:
                print(f"{city:<15} {'FAILED':<10} {r.get('error', 'Unknown error')[:40]}")

        print()

    # Exit with error if any failed
    if any(not r['success'] for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
