#!/usr/bin/env python3
"""
Export Kalshi bin settlements with event_date_local conversion.

Converts UTC close_time to local event date using city timezones.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.connection import get_session
from sqlalchemy import text
from ml.date_utils import event_date_from_close_time


def export_kalshi_settlements(output_csv: str) -> int:
    """
    Export Kalshi bin settlements with event_date_local.

    Args:
        output_csv: Path to output CSV file

    Returns:
        Number of records exported
    """
    # Fetch markets from database
    with get_session() as session:
        query = text("""
            SELECT
                ticker,
                series_ticker,
                close_time,
                result,
                settlement_value,
                strike_type,
                floor_strike,
                cap_strike
            FROM markets
            WHERE status = 'finalized'
              AND settlement_value IS NOT NULL
              AND series_ticker IN (
                  'KXHIGHCHI', 'KXHIGHLAX',
                  'KXHIGHDEN', 'KXHIGHAUS', 'KXHIGHMIA', 'KXHIGHPHIL'
              )
            ORDER BY close_time
        """)

        result = session.execute(query)
        rows = result.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=result.keys())

    # Convert close_time to event_date_local
    def get_event_date_local(row):
        """Convert UTC close_time to local event date."""
        return event_date_from_close_time(row["series_ticker"], row["close_time"])

    df['event_date_local'] = df.apply(get_event_date_local, axis=1)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Exported {len(df)} Kalshi bin settlements to {output_csv}")

    return len(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Kalshi bin settlements")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )

    args = parser.parse_args()

    count = export_kalshi_settlements(args.output)
    print(f"Done! Exported {count} records")
