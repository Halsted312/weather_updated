#!/usr/bin/env python3
# scripts/plot_austin_forecast_vs_obs.py

"""
Visual QA: Plot Austin T-1 forecast vs observations to verify timezone alignment.

This script loads:
- 5-minute observations from station (KAUS)
- 15-minute T-1 historical forecast from city (Austin,TX)

And plots them over a two-day window (D-1 10:00 → D 23:59) to visually confirm:
- No 6-hour timezone shift
- Curves align horizontally
- Realistic diurnal patterns

Usage:
    python scripts/plot_austin_forecast_vs_obs.py

    # Custom event date:
    python scripts/plot_austin_forecast_vs_obs.py --event-date 2025-06-15
"""

import argparse
import os
from datetime import date, datetime, time, timedelta

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db.connection import get_db_session
from src.config.cities import get_city
from models.data.loader import get_vc_location_id


def load_austin_obs_and_forecast_minutes(
    session: Session,
    event_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load station observations and city forecast for a two-day window.

    Args:
        session: Database session
        event_date: The event date (T)

    Returns:
        Tuple of (obs_df, fcst_df) DataFrames
    """
    city_id = "austin"

    stn_loc_id = get_vc_location_id(session, city_id, "station")
    city_loc_id = get_vc_location_id(session, city_id, "city")

    if stn_loc_id is None or city_loc_id is None:
        raise RuntimeError("Missing vc_location_id for Austin (station or city)")

    d_minus_1 = event_date - timedelta(days=1)
    start_dt_str = f"{d_minus_1.isoformat()} 10:00:00"
    end_dt_str = f"{event_date.isoformat()} 23:59:59"

    # Observations (station, 5-min resolution)
    obs_sql = text("""
        SELECT datetime_local, temp_f
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :stn_loc_id
          AND data_type = 'actual_obs'
          AND datetime_local >= :start_dt
          AND datetime_local <= :end_dt
        ORDER BY datetime_local
    """)

    obs_df = pd.read_sql(
        obs_sql,
        session.bind,
        params={"stn_loc_id": stn_loc_id, "start_dt": start_dt_str, "end_dt": end_dt_str},
    )

    if not obs_df.empty:
        obs_df = obs_df.rename(columns={"temp_f": "temp_obs_f"})
        print(f"Loaded {len(obs_df)} observation rows")
    else:
        print("⚠️  No observations found")

    # Historical forecast (city, 15-min resolution)
    fcst_sql = text("""
        SELECT datetime_local, temp_f
        FROM wx.vc_minute_weather
        WHERE vc_location_id = :city_loc_id
          AND data_type = 'historical_forecast'
          AND forecast_basis_date = :basis_date
          AND DATE(datetime_local) = :event_date
        ORDER BY datetime_local
    """)

    fcst_df = pd.read_sql(
        fcst_sql,
        session.bind,
        params={"city_loc_id": city_loc_id, "basis_date": d_minus_1, "event_date": event_date},
    )

    if not fcst_df.empty:
        fcst_df = fcst_df.rename(columns={"temp_f": "temp_fcst_f"})
        print(f"Loaded {len(fcst_df)} forecast rows (basis_date={d_minus_1})")
    else:
        print(f"⚠️  No forecast found for event_date={event_date}, basis_date={d_minus_1}")

    return obs_df, fcst_df


def plot_austin_forecast_vs_obs(event_date: date, save_dir: str = "visuals") -> str:
    """
    Generate and save plot comparing forecast vs observations.

    Args:
        event_date: The event date to plot (T)
        save_dir: Directory to save plot

    Returns:
        Path to saved plot file
    """
    with get_db_session() as session:
        obs_df, fcst_df = load_austin_obs_and_forecast_minutes(session, event_date)

    if obs_df.empty and fcst_df.empty:
        print("❌ No data to plot (both obs and forecast empty)")
        return None

    # Merge on datetime_local
    merged = pd.merge(obs_df, fcst_df, on="datetime_local", how="outer")
    merged = merged.sort_values("datetime_local")

    # Filter to window: D-1 10:00 → D 23:59
    d_minus_1 = event_date - timedelta(days=1)
    start_dt = datetime.combine(d_minus_1, time(10, 0))
    end_dt = datetime.combine(event_date, time(23, 59))
    mask = (merged["datetime_local"] >= start_dt) & (merged["datetime_local"] <= end_dt)
    merged = merged.loc[mask]

    if merged.empty:
        print(f"❌ No data in window {start_dt} to {end_dt}")
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot observations
    if "temp_obs_f" in merged.columns and merged["temp_obs_f"].notna().any():
        ax.plot(
            merged["datetime_local"],
            merged["temp_obs_f"],
            label="Observed temp (station, 5-min)",
            linewidth=2.0,
            color="#1f77b4",
            alpha=0.8,
        )

    # Plot forecast
    if "temp_fcst_f" in merged.columns and merged["temp_fcst_f"].notna().any():
        ax.plot(
            merged["datetime_local"],
            merged["temp_fcst_f"],
            label=f"T-1 forecast (city, 15-min, basis={d_minus_1})",
            linewidth=2.0,
            linestyle="--",
            color="#ff7f0e",
            alpha=0.8,
        )

    # Reference lines
    ax.axvline(
        datetime.combine(d_minus_1, time(10, 0)),
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        label="Market open (D-1 10:00)",
    )
    ax.axvline(
        datetime.combine(event_date, time(0, 0)),
        color="gray",
        linestyle=":",
        linewidth=1.0,
        alpha=0.5,
        label="Midnight D",
    )

    # Styling
    ax.set_title(
        f"Austin: T-1 Forecast vs Observations\n"
        f"Event Date: {event_date.isoformat()}, Basis Date: {d_minus_1.isoformat()}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Local Time (America/Chicago)", fontsize=12)
    ax.set_ylabel("Temperature (°F)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save
    os.makedirs(save_dir, exist_ok=True)
    filename = f"austin_forecast_vs_obs_{event_date.strftime('%Y%m%d')}.png"
    filepath = os.path.join(save_dir, filename)

    fig.savefig(filepath, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✅ Plot saved to: {filepath}")
    return filepath


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Austin forecast vs observations for visual QA"
    )
    parser.add_argument(
        "--event-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=date(2025, 11, 20),
        help="Event date to plot (default: 2025-11-20)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="visuals",
        help="Directory to save plot (default: visuals/)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print(f"Plotting Austin forecast vs obs for event_date={args.event_date}")
    print("=" * 70)

    filepath = plot_austin_forecast_vs_obs(args.event_date, args.save_dir)

    if filepath:
        print("\n" + "=" * 70)
        print("VISUAL QA CHECKS:")
        print("=" * 70)
        print("✅ Forecast and obs curves should align horizontally (same time axis)")
        print("✅ No 6-hour timezone shift detected (peaks/troughs at same hours)")
        print("✅ Both curves show realistic diurnal pattern (cool night, warm midday)")
        print("✅ Forecast may be slightly smoother or biased vs obs (expected)")
        print("\n⚠️  FAILURE INDICATORS:")
        print("  - Forecast peaks at 12:00 but obs peaks at 18:00 = 6-hour shift bug")
        print("  - Curves completely misaligned = wrong basis_date or event_date")
        print("  - Years showing 1900 = datetime parsing bug")
        print("\nOpen the plot and inspect visually:")
        print(f"  {filepath}")
    else:
        print("\n❌ Failed to generate plot - check data availability")


if __name__ == "__main__":
    main()
