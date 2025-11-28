#!/usr/bin/env python3
"""
Ad-Hoc Kalshi Weather Prediction Tool

Simple script to get bracket probabilities for any city/date/time.
Edit config.py to specify what you want to predict.

Usage:
    python tools/adhoc/predict_now.py

Or with venv:
    .venv/bin/python tools/adhoc/predict_now.py
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.features.base import DELTA_CLASSES

# Import config
try:
    from tools.adhoc import config
except ImportError:
    print("⛔ ERROR: Could not import config.py")
    print("Make sure you're running from project root or config.py exists")
    sys.exit(1)


# Training hours (models only trained on these)
TRAIN_HOURS = [10, 12, 14, 16, 18, 20, 22, 23]


def parse_time(time_str: str) -> tuple[int, int]:
    """Parse time string to (hour, minute).

    Accepts formats:
    - "1000" → (10, 0)
    - "1316" → (13, 16)
    - "10:00" → (10, 0)
    - "13:16" → (13, 16)
    """
    time_str = time_str.strip().replace(":", "")

    if len(time_str) == 3:  # "900" → 9:00
        hour = int(time_str[0])
        minute = int(time_str[1:])
    elif len(time_str) == 4:  # "1316" → 13:16
        hour = int(time_str[:2])
        minute = int(time_str[2:])
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use HHMM or HH:MM")

    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"Invalid time: {hour}:{minute:02d}")

    return hour, minute


def snap_to_nearest_snapshot_hour(hour: int) -> int:
    """Snap to nearest training hour - model only knows these hours."""
    return min(TRAIN_HOURS, key=lambda x: abs(x - hour))


def format_bar(pct: float, max_width: int = 40) -> str:
    """Create ASCII bar chart."""
    width = int(pct * max_width)
    return "|" + "█" * width


def main():
    print("="*80)
    print("KALSHI WEATHER PREDICTION - ORDINAL CATBOOST")
    print("="*80)
    print()

    # Parse config
    city = config.CITY.lower()
    date_str = config.DATE
    time_str = config.TIME

    # Parse date
    if date_str.lower() == "today":
        target_date = date.today()
    else:
        target_date = date.fromisoformat(date_str)

    # Parse time
    hour, minute = parse_time(time_str)
    snapshot_hour = snap_to_nearest_snapshot_hour(hour)

    # Load model
    model_path = Path(config.MODEL_DIR) / city / config.MODEL_FILE

    if not model_path.exists():
        print(f"⛔ ERROR: Model not found at {model_path}")
        print(f"\nAvailable cities:")
        model_dir = Path(config.MODEL_DIR)
        if model_dir.exists():
            for city_folder in sorted(model_dir.iterdir()):
                if city_folder.is_dir():
                    print(f"  - {city_folder.name}")
        return 1

    print(f"City: {city.title()} ({_get_station_code(city)})")
    print(f"Event Date: {target_date}")
    print(f"Snapshot Time: {hour}:{minute:02d} local → Using {snapshot_hour}:00 model (nearest training hour)")
    print()

    # Load model
    print("Loading model...")
    trainer = OrdinalDeltaTrainer()
    trainer.load(model_path)

    print(f"✅ Loaded {trainer._metadata.get('model_type', 'ordinal_catboost')}")
    print(f"   Delta range: {trainer._metadata.get('delta_range', 'N/A')}")
    print(f"   Classifiers: {len(trainer.classifiers)}")
    print()

    # Load today's forecast data as placeholder for current observations
    from src.db.connection import get_db_session
    from sqlalchemy import text

    with get_db_session() as session:
        # Get today's minute-level forecast data (will serve as obs placeholder)
        query = text("""
            SELECT
                vm.datetime_local,
                vm.temp_f
            FROM wx.vc_minute_weather vm
            JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
            WHERE vl.city_code = :city_code
              AND DATE(vm.datetime_local) = :target_date
              AND vm.temp_f IS NOT NULL
            ORDER BY vm.datetime_local
        """)

        city_codes = {
            "chicago": "CHI",
            "austin": "AUS",
            "denver": "DEN",
            "los_angeles": "LAX",
            "miami": "MIA",
            "philadelphia": "PHL",
        }

        result = session.execute(query, {
            "city_code": city_codes[city],
            "target_date": target_date
        }).fetchall()

        if not result:
            print(f"⛔ ERROR: No data found for {city} on {target_date}")
            print("\nOptions:")
            print(f"1. Run: .venv/bin/python scripts/ingest_vc_forecast_snapshot.py --city-code {city_codes[city]}")
            print("2. Or use yesterday's test data to demo")
            return 1

        # Convert to lists for feature computation
        all_temps = [row.temp_f for row in result]
        all_timestamps = [row.datetime_local for row in result]

        # Filter to only observations UP TO cutoff time (snapshot time)
        cutoff_dt = datetime.combine(target_date, datetime.min.time()).replace(hour=hour, minute=minute)

        temps_sofar = []
        timestamps_sofar = []
        for temp, ts in zip(all_temps, all_timestamps):
            if ts <= cutoff_dt:
                temps_sofar.append(temp)
                timestamps_sofar.append(ts)

        if len(temps_sofar) < 12:
            print(f"⚠️  WARNING: Only {len(temps_sofar)} observations up to {hour}:{minute:02d}")
            print(f"   Need at least 12 (1 hour). Proceeding anyway...")

        t_base = int(round(max(temps_sofar)))

        print(f"📊 USING TODAY'S DATA: {target_date}")
        print(f"   Observations: {len(temps_sofar)} up to {hour}:{minute:02d}")
        print(f"   Temperature range: {min(temps_sofar):.1f}°F to {max(temps_sofar):.1f}°F")
        print()
        print(f"Current Observed Max: {t_base}°F (rounded from {max(temps_sofar):.1f}°F)")
        print()

        # Build snapshot for inference
        from models.data.snapshot_builder import build_snapshot_for_inference

        snapshot = build_snapshot_for_inference(
            city=city,
            day=target_date,
            snapshot_hour=snapshot_hour,
            temps_sofar=temps_sofar,
            timestamps_sofar=timestamps_sofar,
            fcst_daily=None,  # TODO: Add T-1 forecast when available
            fcst_hourly_df=None,
        )

        # Convert to DataFrame
        snapshot_row = pd.Series(snapshot)

        # Add dummy columns for inference (not used but expected by _prepare_features)
        snapshot_row['delta'] = 0  # Dummy value
        snapshot_row['settle_f'] = t_base  # Dummy value

        actual_settle = None  # Unknown for today
        actual_delta = None

    # Predict
    print("DATA QUALITY:")
    print(f"✅ Forecast data used as observation placeholder")
    print(f"✅ Features: {len([c for c in snapshot_row.index if c not in ['city', 'day', 'snapshot_hour', 'settle_f', 'delta']])} computed")
    print()

    # Make prediction
    df_pred = pd.DataFrame([snapshot_row])
    proba = trainer.predict_proba(df_pred)[0]  # Shape: (13,) for 13 delta classes
    delta_pred = trainer.predict(df_pred)[0]

    # Compute statistics
    expected_delta = sum(DELTA_CLASSES[i] * proba[i] for i in range(len(DELTA_CLASSES)))
    expected_settle = t_base + expected_delta

    # Variance and CI
    variance = sum(((DELTA_CLASSES[i] - expected_delta) ** 2) * proba[i] for i in range(len(DELTA_CLASSES)))
    std = np.sqrt(variance)

    # 90% CI (approximate using percentiles)
    cumsum = np.cumsum(proba)
    p10_idx = np.searchsorted(cumsum, 0.10)
    p90_idx = np.searchsorted(cumsum, 0.90)
    ci_low = t_base + DELTA_CLASSES[p10_idx]
    ci_high = t_base + DELTA_CLASSES[min(p90_idx, len(DELTA_CLASSES)-1)]
    ci_span = ci_high - ci_low

    # Uncertainty warnings
    print("MODEL PREDICTION:")
    print(f"Most Likely Delta: {delta_pred:+d}°F ({proba[DELTA_CLASSES.index(delta_pred)]*100:.1f}% probability)")
    print(f"Expected Settlement: {expected_settle:.1f}°F ± {std:.1f}°F")
    print(f"90% Confidence Interval: [{ci_low}, {ci_high}] (span = {ci_span}°F)")
    print()

    # Uncertainty guardrails
    warnings = []
    if std > config.MAX_UNCERTAINTY_DEGF:
        warnings.append(f"⚠️  HIGH UNCERTAINTY: Settlement std = {std:.1f}°F (>{config.MAX_UNCERTAINTY_DEGF}°F)")
    if ci_span > config.MAX_CI_SPAN_DEGF:
        warnings.append(f"⚠️  WIDE CI: 90% interval spans {ci_span}°F (>{config.MAX_CI_SPAN_DEGF}°F)")
    if snapshot_hour < 14:
        warnings.append(f"⚠️  EARLY PREDICTION: Accuracy typically 45-55% at {snapshot_hour}:00")

    # Print delta distribution
    print("DELTA PROBABILITY DISTRIBUTION:")
    for i, delta_val in enumerate(DELTA_CLASSES):
        pct = proba[i]
        if pct < 0.005:  # Skip tiny probabilities
            continue

        bar = format_bar(pct)
        marker = " ⭐" if delta_val == delta_pred else ""
        actual_marker = f" (ACTUAL: {actual_delta:+d})" if actual_delta is not None and delta_val == actual_delta else ""
        print(f"  delta={delta_val:+2d}: {pct*100:5.1f}%  {bar}{marker}{actual_marker}")
    print()

    # Bracket probabilities
    print(f"KALSHI BRACKET PROBABILITIES (>{config.MIN_CONFIDENCE*100:.0f}% only):")
    print(f"{'Bracket':<12} {'Model Prob':>12} {'Market Price':>14} {'Edge':>10} {'Signal':<20}")
    print("-"*70)

    # Compute bracket probabilities
    brackets = []
    for threshold in range(t_base - 2, t_base + 12, 2):
        # P(settlement >= threshold)
        min_delta_needed = threshold - t_base
        mask = np.array([d >= min_delta_needed for d in DELTA_CLASSES])
        bracket_prob = proba[mask].sum()

        if bracket_prob < config.MIN_CONFIDENCE:
            continue

        bracket_label = f"[{threshold}-{threshold+1}]" if threshold < t_base + 10 else f"[{threshold}+]"
        if threshold < t_base:
            bracket_label = f"[<{t_base}]"

        # Check if market price provided
        market_price = config.MARKET_PRICES.get(bracket_label)

        if market_price is not None:
            market_prob = market_price / 100
            edge_pct = (bracket_prob - market_prob) * 100

            if edge_pct > config.MIN_EDGE_PCT:
                signal = "🟢 BUY"
            elif edge_pct < -config.MIN_EDGE_PCT:
                signal = "🔴 SELL"
            else:
                signal = "HOLD"

            print(f"{bracket_label:<12} {bracket_prob*100:>11.1f}% {market_price:>13}¢ {edge_pct:>9.1f}% {signal:<20}")
        else:
            print(f"{bracket_label:<12} {bracket_prob*100:>11.1f}% {'--':>13} {'--':>10} {'(no market price)':<20}")

        brackets.append((bracket_label, bracket_prob, market_price))

    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    trade_found = False
    for bracket_label, model_prob, market_price in brackets:
        if market_price is None:
            continue

        edge_pct = (model_prob - market_price/100) * 100

        if edge_pct > config.MIN_EDGE_PCT:
            print(f"🟢 BUY {bracket_label} at ≤{int(model_prob*100-config.MIN_EDGE_PCT)}¢ (model: {model_prob*100:.1f}%, edge: {edge_pct:+.1f}%)")
            trade_found = True
        elif edge_pct < -config.MIN_EDGE_PCT:
            print(f"🔴 SELL {bracket_label} at ≥{int(model_prob*100+config.MIN_EDGE_PCT)}¢ (model: {model_prob*100:.1f}%, edge: {edge_pct:+.1f}%)")
            trade_found = True

    if not trade_found:
        print("   No trades meet edge threshold (>{:.0f}%)".format(config.MIN_EDGE_PCT))

    print()

    # Warnings
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        print()

    # Accuracy context
    print("ACCURACY CONTEXT (from test set):")
    print(f"   Chicago @ {snapshot_hour}:00 - Typical accuracy: " + _get_accuracy_estimate(snapshot_hour))
    print(f"   Daily high usually occurs: 14:00-18:00 local")
    print()

    # Test result (if we have actual)
    if actual_settle is not None:
        print("TEST VERIFICATION:")
        print(f"   Predicted delta: {delta_pred:+d}°F")
        print(f"   Actual delta:    {actual_delta:+d}°F")
        print(f"   Error: {abs(delta_pred - actual_delta)}°F")
        if abs(delta_pred - actual_delta) <= 1:
            print(f"   ✅ Within 1°F!")
        print()
    else:
        print("🔮 LIVE PREDICTION MODE:")
        print(f"   Predicted delta: {delta_pred:+d}°F → Expected settlement: {t_base + delta_pred}°F")
        print(f"   Actual settlement: TBD (check after market settles)")
        print()

    print("="*80)
    print("NOTE: Currently using forecast data as placeholder for observations.")
    print("When live observation ingestion is set up, this will use actual temps.")
    print("="*80)

    return 0


def _get_station_code(city: str) -> str:
    """Get weather station code for city."""
    stations = {
        "chicago": "KMDW",
        "austin": "KAUS",
        "denver": "KDEN",
        "los_angeles": "KLAX",
        "miami": "KMIA",
        "philadelphia": "KPHL",
    }
    return stations.get(city, "")


def _get_accuracy_estimate(hour: int) -> str:
    """Get rough accuracy estimate for hour."""
    if hour <= 12:
        return "45-50%"
    elif hour <= 14:
        return "50-55%"
    elif hour <= 16:
        return "55-60%"
    elif hour <= 18:
        return "60-65%"
    elif hour <= 20:
        return "65-70%"
    else:
        return "70-75%"


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n⛔ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
