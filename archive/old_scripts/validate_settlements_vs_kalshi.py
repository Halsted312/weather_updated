#!/usr/bin/env python3
"""
Validate NWS-derived bin outcomes against Kalshi settlements.

This script implements the critical validation check:
- Temperature Ground Truth: NWS CF6/CLI integer °F (tmax_official_f)
- Market Ground Truth: Kalshi settlement_value (which bin paid YES/NO)
- Validation: Computed bin outcome from tmax_official_f should match Kalshi's settlement

Any mismatches indicate either:
1. Data quality issues (NWS vs Kalshi disagreement)
2. Rule interpretation differences
3. Fetching/parsing errors

Expected Result: 100% agreement (or very close)
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi.bin_labels import bin_resolves_yes

logger = logging.getLogger(__name__)


def load_kalshi_settlements(csv_path: str) -> pd.DataFrame:
    """
    Load Kalshi bin settlements from CSV.

    Args:
        csv_path: Path to kalshi_bin_settlements.csv

    Returns:
        DataFrame with columns: ticker, series_ticker, event_date_local,
                                result, settlement_value, strike_type,
                                floor_strike, cap_strike
    """
    df = pd.read_parquet(csv_path) if csv_path.endswith('.parquet') else pd.read_csv(csv_path)

    # Convert result to binary (for comparison)
    # Kalshi uses "yes"/"no" strings
    df["kalshi_bin_outcome"] = (df["result"].str.lower() == "yes").astype(int)

    # Ensure event_date_local is a date
    df["event_date_local"] = pd.to_datetime(df["event_date_local"]).dt.date

    logger.info(f"Loaded {len(df)} Kalshi bin settlements")

    return df


def load_settlement_temperatures(csv_path: str) -> pd.DataFrame:
    """
    Load reconciled settlement temperatures from CSV.

    Args:
        csv_path: Path to settlements_reconciled.csv

    Returns:
        DataFrame with columns: city, date_local, tmax_final_f, source_final
    """
    df = pd.read_csv(csv_path)

    # Ensure date_local is a date
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date

    logger.info(f"Loaded {len(df)} settlement temperatures")

    return df


def map_series_to_city(series_ticker: str) -> str:
    """
    Map Kalshi series ticker to city name.

    Args:
        series_ticker: Kalshi series ticker (e.g., "KXHIGHCHI")

    Returns:
        City name (e.g., "chicago")
    """
    mapping = {
        "KXHIGHCHI": "chicago",
        "KXHIGHNY": "new_york",
        "KXHIGHLAX": "los_angeles",
        "KXHIGHDEN": "denver",
        "KXHIGHAUS": "austin",
        "KXHIGHMIA": "miami",
        "KXHIGHPHIL": "philadelphia",
    }
    return mapping.get(series_ticker, "unknown")


def validate_settlements(
    kalshi_df: pd.DataFrame,
    settlement_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate Kalshi settlements against NWS temperature data.

    Args:
        kalshi_df: Kalshi bin settlements (from load_kalshi_settlements)
        settlement_df: NWS temperatures (from load_settlement_temperatures)

    Returns:
        DataFrame with validation results including:
        - All original columns from kalshi_df
        - tmax_official_f: NWS temperature used
        - source_final: Source of NWS temperature
        - computed_bin_outcome: What the bin SHOULD resolve to (0 or 1)
        - kalshi_bin_outcome: What Kalshi actually paid (0 or 1)
        - is_match: Whether they agree (True/False)
        - mismatch_detail: Description of mismatch (if any)
    """
    # Add city column to Kalshi data
    kalshi_df["city"] = kalshi_df["series_ticker"].apply(map_series_to_city)

    # Join Kalshi settlements with NWS temperatures
    merged = kalshi_df.merge(
        settlement_df[["city", "date_local", "tmax_final_f", "source_final"]],
        left_on=["city", "event_date_local"],
        right_on=["city", "date_local"],
        how="left",
    )

    # Compute what the bin outcome SHOULD be based on NWS temperature
    merged["computed_bin_outcome"] = merged.apply(
        lambda row: bin_resolves_yes(
            row["tmax_final_f"],
            row["strike_type"],
            row["floor_strike"],
            row["cap_strike"],
        ),
        axis=1,
    )

    # Check for matches
    merged["is_match"] = merged["computed_bin_outcome"] == merged["kalshi_bin_outcome"]

    # Create mismatch detail
    def describe_mismatch(row):
        if row["is_match"]:
            return None
        if pd.isna(row["tmax_final_f"]):
            return "Missing NWS temperature"
        if pd.isna(row["computed_bin_outcome"]):
            return "Unable to compute bin outcome"
        return f"Computed={row['computed_bin_outcome']}, Kalshi={row['kalshi_bin_outcome']}, Tmax={row['tmax_final_f']}°F"

    merged["mismatch_detail"] = merged.apply(describe_mismatch, axis=1)

    logger.info(f"Validated {len(merged)} bins")

    return merged


def print_validation_report(validated_df: pd.DataFrame) -> None:
    """
    Print human-readable validation report.

    Args:
        validated_df: Output from validate_settlements()
    """
    print("\n" + "="*60)
    print("SETTLEMENT VALIDATION REPORT")
    print("="*60)

    total = len(validated_df)
    matches = validated_df["is_match"].sum()
    mismatches = total - matches
    match_rate = (matches / total * 100) if total > 0 else 0

    print(f"\nTotal bins validated: {total}")
    print(f"Matches: {matches} ({match_rate:.2f}%)")
    print(f"Mismatches: {mismatches}")

    if mismatches > 0:
        print(f"\n⚠️  MISMATCHES DETECTED:")
        mismatch_df = validated_df[~validated_df["is_match"]]

        print(f"\nBy city:")
        print(mismatch_df.groupby("city").size())

        print(f"\nBy strike type:")
        print(mismatch_df.groupby("strike_type").size())

        print(f"\nMismatch details:")
        print(mismatch_df[["ticker", "event_date_local", "tmax_final_f", "strike_type", "floor_strike", "cap_strike", "mismatch_detail"]])
    else:
        print(f"\n✓ PERFECT AGREEMENT! All computed bin outcomes match Kalshi settlements.")

    # Show source distribution
    print(f"\nNWS Temperature sources:")
    print(validated_df["source_final"].value_counts())

    # Sample matches
    print(f"\nSample matched validations:")
    sample = validated_df[validated_df["is_match"]].head(5)
    print(sample[["ticker", "event_date_local", "tmax_final_f", "result", "strike_type", "floor_strike", "cap_strike"]])

    print("="*60 + "\n")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Validate NWS-derived bin outcomes against Kalshi settlements"
    )
    parser.add_argument(
        "--kalshi-settlements",
        type=str,
        required=True,
        help="Path to Kalshi bin settlements CSV/parquet",
    )
    parser.add_argument(
        "--nws-temperatures",
        type=str,
        required=True,
        help="Path to reconciled NWS temperatures CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file with validation results (optional)",
    )

    args = parser.parse_args()

    print(f"\nValidating settlements:")
    print(f"  Kalshi settlements: {args.kalshi_settlements}")
    print(f"  NWS temperatures: {args.nws_temperatures}")
    print()

    # Load data
    kalshi_df = load_kalshi_settlements(args.kalshi_settlements)
    settlement_df = load_settlement_temperatures(args.nws_temperatures)

    # Validate
    validated_df = validate_settlements(kalshi_df, settlement_df)

    # Print report
    print_validation_report(validated_df)

    # Save if requested
    if args.output:
        validated_df.to_csv(args.output, index=False)
        print(f"Validation results saved to: {args.output}\n")
