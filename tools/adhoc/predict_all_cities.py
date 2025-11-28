#!/usr/bin/env python3
"""
Multi-City Kalshi Weather Predictions

Input ONE time (Chicago time) and get predictions for all 6 cities
accounting for timezone differences.

Usage:
    Edit config.py to set TIME (in Chicago time)
    Then run: .venv/bin/python tools/adhoc/predict_all_cities.py
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from models.training.ordinal_trainer import OrdinalDeltaTrainer
from models.features.base import DELTA_CLASSES

try:
    from tools.adhoc import config
except ImportError:
    print("⛔ ERROR: Could not import config.py")
    sys.exit(1)


# City timezone offsets from Chicago (CST)
TIMEZONE_OFFSETS = {
    "chicago": 0,        # CST (reference)
    "austin": 0,         # CST (same as Chicago)
    "denver": -1,        # MST (1 hour behind Chicago)
    "los_angeles": -2,   # PST (2 hours behind Chicago)
    "miami": +1,         # EST (1 hour ahead of Chicago)
    "philadelphia": +1,  # EST (1 hour ahead of Chicago)
}

CITY_STATIONS = {
    "chicago": "KMDW",
    "austin": "KAUS",
    "denver": "KDEN",
    "los_angeles": "KLAX",
    "miami": "KMIA",
    "philadelphia": "KPHL",
}

CITY_CODES = {
    "chicago": "CHI",
    "austin": "AUS",
    "denver": "DEN",
    "los_angeles": "LAX",
    "miami": "MIA",
    "philadelphia": "PHL",
}

TRAIN_HOURS = [10, 12, 14, 16, 18, 20, 22, 23]


def parse_time(time_str: str) -> tuple[int, int]:
    """Parse time string to (hour, minute)."""
    time_str = time_str.strip().replace(":", "")
    if len(time_str) == 3:
        hour, minute = int(time_str[0]), int(time_str[1:])
    elif len(time_str) == 4:
        hour, minute = int(time_str[:2]), int(time_str[2:])
    else:
        raise ValueError(f"Invalid time: {time_str}")
    return hour, minute


def snap_to_nearest_snapshot_hour(hour: int) -> int:
    """Snap to nearest training hour."""
    return min(TRAIN_HOURS, key=lambda x: abs(x - hour))


def get_local_time(chicago_hour: int, chicago_minute: int, city: str) -> tuple[int, int, int]:
    """Convert Chicago time to city local time.

    Returns: (local_hour, local_minute, snapshot_hour)
    """
    offset = TIMEZONE_OFFSETS[city]
    local_hour = (chicago_hour + offset) % 24
    local_minute = chicago_minute
    snapshot_hour = snap_to_nearest_snapshot_hour(local_hour)

    return local_hour, local_minute, snapshot_hour


def predict_city(city: str, target_date: date, local_hour: int, local_minute: int, snapshot_hour: int) -> dict:
    """Run prediction for one city."""
    from src.db.connection import get_db_session
    from sqlalchemy import text
    from models.data.snapshot_builder import build_snapshot_for_inference

    # Load model
    model_path = Path(config.MODEL_DIR) / city / config.MODEL_FILE
    if not model_path.exists():
        return {"city": city, "error": "Model not found"}

    trainer = OrdinalDeltaTrainer()
    trainer.load(model_path)

    # Get data
    with get_db_session() as session:
        query = text("""
            SELECT vm.datetime_local, vm.temp_f
            FROM wx.vc_minute_weather vm
            JOIN wx.vc_location vl ON vm.vc_location_id = vl.id
            WHERE vl.city_code = :city_code
              AND DATE(vm.datetime_local) = :target_date
              AND vm.temp_f IS NOT NULL
            ORDER BY vm.datetime_local
        """)

        result = session.execute(query, {
            "city_code": CITY_CODES[city],
            "target_date": target_date
        }).fetchall()

        if not result:
            return {"city": city, "error": "No data"}

        all_temps = [row.temp_f for row in result]
        all_timestamps = [row.datetime_local for row in result]

        # Filter to cutoff time
        cutoff_dt = datetime.combine(target_date, datetime.min.time()).replace(
            hour=local_hour, minute=local_minute
        )

        temps_sofar = [t for t, ts in zip(all_temps, all_timestamps) if ts <= cutoff_dt]
        timestamps_sofar = [ts for ts in all_timestamps if ts <= cutoff_dt]

        if len(temps_sofar) < 1:
            return {"city": city, "error": f"No data before {local_hour}:{local_minute:02d}"}

        t_base = int(round(max(temps_sofar)))

        # Build snapshot
        snapshot = build_snapshot_for_inference(
            city=city,
            day=target_date,
            snapshot_hour=snapshot_hour,
            temps_sofar=temps_sofar,
            timestamps_sofar=timestamps_sofar,
            fcst_daily=None,
            fcst_hourly_df=None,
        )

        snapshot_row = pd.Series(snapshot)
        snapshot_row['delta'] = 0
        snapshot_row['settle_f'] = t_base

        # Predict
        df_pred = pd.DataFrame([snapshot_row])
        proba = trainer.predict_proba(df_pred)[0]
        delta_pred = trainer.predict(df_pred)[0]

        # Calculate statistics
        expected_delta = sum(DELTA_CLASSES[i] * proba[i] for i in range(len(DELTA_CLASSES)))
        expected_settle = t_base + expected_delta

        variance = sum(((DELTA_CLASSES[i] - expected_delta) ** 2) * proba[i]
                      for i in range(len(DELTA_CLASSES)))
        std = np.sqrt(variance)

        return {
            "city": city,
            "station": CITY_STATIONS[city],
            "local_time": f"{local_hour}:{local_minute:02d}",
            "snapshot_hour": snapshot_hour,
            "n_obs": len(temps_sofar),
            "t_base": t_base,
            "temp_range": (min(temps_sofar), max(temps_sofar)),
            "delta_pred": delta_pred,
            "expected_settle": expected_settle,
            "std": std,
            "proba": proba,
            "delta_range": trainer._metadata.get('delta_range', 'N/A'),
        }


def main():
    print("="*100)
    print("MULTI-CITY KALSHI WEATHER PREDICTIONS")
    print("="*100)
    print()

    # Parse Chicago time from config
    chicago_hour, chicago_minute = parse_time(config.TIME)
    target_date = date.fromisoformat(config.DATE) if config.DATE != "today" else date.today()

    print(f"Reference Time: {chicago_hour}:{chicago_minute:02d} Chicago (CST)")
    print(f"Event Date: {target_date}")
    print()
    print("-"*100)

    # Predict all cities
    results = []
    for city in ["chicago", "austin", "denver", "los_angeles", "miami", "philadelphia"]:
        local_hour, local_minute, snapshot_hour = get_local_time(chicago_hour, chicago_minute, city)
        result = predict_city(city, target_date, local_hour, local_minute, snapshot_hour)
        results.append(result)

    # Print summary table
    print()
    print("SUMMARY - ALL 6 CITIES:")
    print("-"*100)
    print(f"{'City':<15} {'Station':<8} {'Local Time':<12} {'Snap Hr':<8} {'Obs':<6} {'Max':<6} {'Pred Δ':<8} {'Exp Settle':<12} {'Std':<6}")
    print("-"*100)

    for r in results:
        if "error" in r:
            print(f"{r['city']:<15} {'--':<8} {'--':<12} {'--':<8} {'--':<6} {'--':<6} {'--':<8} ERROR: {r['error']}")
        else:
            print(f"{r['city']:<15} {r['station']:<8} {r['local_time']:<12} {r['snapshot_hour']:<8} "
                  f"{r['n_obs']:<6} {r['t_base']:<6}°F {r['delta_pred']:+2d}°F{' ':<4} "
                  f"{r['expected_settle']:.1f}°F{' ':<6} {r['std']:.1f}°F")

    print("-"*100)
    print()

    # Detailed breakdown for each city
    for r in results:
        if "error" in r:
            continue

        print("="*100)
        print(f"{r['city'].upper()} ({r['station']}) - Local time: {r['local_time']} → Using {r['snapshot_hour']}:00 model")
        print("="*100)

        print(f"Current Max: {r['t_base']}°F (from {r['n_obs']} observations)")
        print(f"Predicted Settlement: {r['expected_settle']:.1f}°F ± {r['std']:.1f}°F")
        print(f"Most Likely Delta: {r['delta_pred']:+d}°F")
        print()

        # Top 5 delta probabilities
        proba = r['proba']
        top_indices = np.argsort(proba)[::-1][:5]

        print("Top 5 Delta Probabilities:")
        for idx in top_indices:
            delta_val = DELTA_CLASSES[idx]
            pct = proba[idx]
            marker = " ⭐" if delta_val == r['delta_pred'] else ""
            print(f"  delta={delta_val:+2d}: {pct*100:5.1f}%{marker}")

        print()

    print("="*100)
    print("TIP: Run 'tools/adhoc/predict_now.py' for detailed bracket analysis of a specific city")
    print("="*100)

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n⛔ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
