#!/usr/bin/env python3
"""
Multi-Source Settlement Reconciliation.

Combines temperature data from multiple sources with precedence rules:
CF6 (NWS Preliminary Climate) > ADS (ASOS Daily Summary) > VC (Visual Crossing)

Key Concepts:
- Temperature Ground Truth: NWS climate products (CF6/CLI) provide integer °F
- Market Ground Truth: Kalshi settlements (which bin paid YES/NO)
- Settlement Precedence: CF6 > ADS > VC (add CLI later for CLI > CF6)

Output: Reconciled settlement table with:
- city, date_local: Primary key
- tmax_final_f: Official temperature (integer °F) using precedence
- source_final: Which source was used ("cf6", "ads", or "vc")
- Delta columns: Disagreements between sources for diagnostics
"""

from __future__ import annotations
import datetime as dt
import logging
from typing import Iterable
import pandas as pd

logger = logging.getLogger(__name__)


def build_settlement_table(
    start: dt.date,
    end: dt.date,
    cities: Iterable[str] | None = None,
    include_cf6: bool = True,
    include_ads: bool = True,
    include_vc: bool = False,  # Not implemented yet
) -> pd.DataFrame:
    """
    Build reconciled settlement table from multiple sources.

    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
        cities: List of city names (default: all cities)
        include_cf6: Fetch CF6 data (default: True)
        include_ads: Fetch ADS data (default: True)
        include_vc: Fetch Visual Crossing data (default: False, not yet implemented)

    Returns:
        DataFrame with columns:
        - city: City name
        - date_local: Local date
        - tmax_cf6_f: CF6 temperature (if fetched)
        - tmax_ads_f: ADS temperature (if fetched)
        - tmax_final_f: Final temperature using precedence (CF6 > ADS > VC)
        - source_final: Source used for tmax_final_f
        - delta_ads_minus_cf6: ADS - CF6 (for diagnostics)

    Example:
        >>> df = build_settlement_table(dt.date(2024, 11, 12), dt.date(2024, 11, 13), ["chicago"])
        >>> len(df) == 2
        True
    """
    # Import here to avoid circular dependencies
    from ingest.iem_cf6_daily import fetch_cities as fetch_cf6_cities
    from ingest.iem_ads_daily import fetch_cities as fetch_ads_cities

    # Default to all cities if not specified
    if cities is None:
        from ingest.iem_cf6_daily import CITY_TO_CF6_STATION
        cities = list(CITY_TO_CF6_STATION.keys())

    logger.info(f"Building settlement table for {len(list(cities))} cities from {start} to {end}")

    # Fetch data from each source
    dfs_to_merge = []

    if include_cf6:
        logger.info("Fetching CF6 data...")
        cf6_df = fetch_cf6_cities(list(cities), start, end)
        cf6_df = cf6_df[["city", "day", "tmax_cf6_f"]].rename(columns={"day": "date_local"})
        dfs_to_merge.append(cf6_df)
        logger.info(f"  Fetched {len(cf6_df)} CF6 records")

    if include_ads:
        logger.info("Fetching ADS data...")
        ads_df = fetch_ads_cities(list(cities), start, end)
        ads_df = ads_df[["city", "day", "tmax_ads_f"]].rename(columns={"day": "date_local"})
        dfs_to_merge.append(ads_df)
        logger.info(f"  Fetched {len(ads_df)} ADS records")

    if include_vc:
        logger.warning("Visual Crossing data not yet implemented")

    if not dfs_to_merge:
        logger.error("No data sources enabled")
        return pd.DataFrame(columns=["city", "date_local", "tmax_final_f", "source_final"])

    # Merge all sources
    logger.info("Merging sources...")
    df = dfs_to_merge[0]
    for next_df in dfs_to_merge[1:]:
        df = df.merge(next_df, on=["city", "date_local"], how="outer")

    # Apply precedence: CF6 > ADS > VC
    def choose_final_temp(row):
        """Apply settlement precedence."""
        if include_cf6 and pd.notna(row.get("tmax_cf6_f")):
            return row["tmax_cf6_f"], "cf6"
        elif include_ads and pd.notna(row.get("tmax_ads_f")):
            return row["tmax_ads_f"], "ads"
        elif include_vc and pd.notna(row.get("tmax_vc_f")):
            return row["tmax_vc_f"], "vc"
        else:
            return None, None

    df[["tmax_final_f", "source_final"]] = df.apply(
        lambda row: pd.Series(choose_final_temp(row)), axis=1
    )

    # Convert to integer (settlement values are always integers)
    df["tmax_final_f"] = df["tmax_final_f"].astype("Int64")

    # Calculate deltas for diagnostics
    if include_ads and include_cf6:
        df["delta_ads_minus_cf6"] = (df["tmax_ads_f"] - df["tmax_cf6_f"]).astype("Int64")

    # Sort by city and date
    df = df.sort_values(["city", "date_local"]).reset_index(drop=True)

    logger.info(f"Settlement table complete: {len(df)} records")

    return df


def summarize_disagreements(df: pd.DataFrame, delta_col: str = "delta_ads_minus_cf6") -> pd.DataFrame:
    """
    Summarize disagreements between sources.

    Args:
        df: Settlement table from build_settlement_table()
        delta_col: Column name for delta values (default: "delta_ads_minus_cf6")

    Returns:
        DataFrame with counts of disagreements by city and delta value

    Example:
        >>> df = build_settlement_table(dt.date(2024, 11, 12), dt.date(2024, 11, 13), ["chicago"])
        >>> summary = summarize_disagreements(df)
        >>> len(summary) >= 0  # May have disagreements or not
        True
    """
    if delta_col not in df.columns:
        logger.warning(f"Delta column '{delta_col}' not found in dataframe")
        return pd.DataFrame(columns=["city", delta_col, "count"])

    # Filter to rows where delta exists (both sources present)
    dis = df[df[delta_col].notna()]

    if len(dis) == 0:
        logger.info("No records with both sources for comparison")
        return pd.DataFrame(columns=["city", delta_col, "count"])

    # Count disagreements by city and delta value
    summary = (
        dis.groupby("city")[delta_col]
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["city", delta_col])
    )

    return summary


def print_settlement_summary(df: pd.DataFrame) -> None:
    """
    Print human-readable summary of settlement data.

    Args:
        df: Settlement table from build_settlement_table()
    """
    print("\n" + "="*60)
    print("SETTLEMENT DATA SUMMARY")
    print("="*60)

    print(f"\nTotal records: {len(df)}")

    print(f"\nBy city:")
    print(df.groupby("city").size())

    print(f"\nBy source:")
    if "source_final" in df.columns:
        print(df["source_final"].value_counts())

    print(f"\nMissing tmax_final:")
    print(df["tmax_final_f"].isna().sum())

    # Check for disagreements
    if "delta_ads_minus_cf6" in df.columns:
        print(f"\nADS vs CF6 disagreements:")
        summary = summarize_disagreements(df)
        if len(summary) > 0:
            print(summary)
        else:
            print("  No disagreements (100% agreement)")

    print(f"\nSample data:")
    cols_to_show = ["city", "date_local", "tmax_final_f", "source_final"]
    if "delta_ads_minus_cf6" in df.columns:
        cols_to_show.append("delta_ads_minus_cf6")
    print(df[cols_to_show].head(10))

    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build reconciled settlement table from multiple sources"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        help="Cities to fetch (default: all)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file (optional)",
    )
    parser.add_argument(
        "--no-cf6",
        action="store_true",
        help="Disable CF6 data source",
    )
    parser.add_argument(
        "--no-ads",
        action="store_true",
        help="Disable ADS data source",
    )

    args = parser.parse_args()

    # Parse dates
    start = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()

    print(f"\nBuilding settlement table:")
    print(f"  Cities: {', '.join(args.cities) if args.cities else 'all'}")
    print(f"  Date range: {start} to {end}")
    print(f"  Days: {(end - start).days + 1}")
    print()

    # Build settlement table
    df = build_settlement_table(
        start,
        end,
        cities=args.cities,
        include_cf6=not args.no_cf6,
        include_ads=not args.no_ads,
    )

    # Print summary
    print_settlement_summary(df)

    # Save if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to: {args.output}")
