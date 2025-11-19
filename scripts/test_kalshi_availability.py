#!/usr/bin/env python3
"""
Test Kalshi API historical data availability.

Determines the earliest date for which Kalshi has settled markets
for weather series (KXHIGHCHI, KXHIGHMIA, etc.).
"""

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from kalshi.client import KalshiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# City series mappings
CITY_SERIES = {
    "chicago": "KXHIGHCHI",
    "los_angeles": "KXHIGHLAX",
    "denver": "KXHIGHDEN",
    "austin": "KXHIGHAUS",
    "miami": "KXHIGHMIA",
    "philadelphia": "KXHIGHPHIL",
}


def test_date_availability(client: KalshiClient, series_ticker: str, test_date: datetime) -> dict:
    """
    Test if markets exist for a specific date.

    Args:
        client: Kalshi API client
        series_ticker: Series ticker (e.g., "KXHIGHCHI")
        test_date: Date to test (local time)

    Returns:
        Dict with availability info
    """
    # Convert to UTC timestamps for API
    # Test a 2-day window around the date to account for timezone issues
    start_dt = test_date - timedelta(days=1)
    end_dt = test_date + timedelta(days=2)

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    try:
        response = client.get_markets(
            series_ticker=series_ticker,
            status="settled",  # Only look at settled markets
            min_close_ts=start_ts,
            max_close_ts=end_ts,
            limit=100,
        )

        markets = response.get("markets", [])

        if markets:
            # Get earliest and latest close times
            # Handle both timestamp formats (int or string ISO format)
            close_times = []
            for m in markets:
                close_time = m.get("close_time")
                if isinstance(close_time, str):
                    # ISO format string
                    close_times.append(datetime.fromisoformat(close_time.replace('Z', '+00:00')))
                elif isinstance(close_time, (int, float)):
                    # Unix timestamp
                    close_times.append(datetime.fromtimestamp(close_time, tz=timezone.utc))

            return {
                "available": True,
                "count": len(markets),
                "earliest": min(close_times) if close_times else None,
                "latest": max(close_times) if close_times else None,
                "sample_ticker": markets[0]["ticker"] if markets else None,
            }
        else:
            return {
                "available": False,
                "count": 0,
            }

    except Exception as e:
        logger.error(f"Error testing {test_date.date()}: {e}")
        return {
            "available": False,
            "count": 0,
            "error": str(e),
        }


def find_earliest_date(
    client: KalshiClient,
    series_ticker: str,
    earliest_guess: datetime,
    latest_guess: datetime,
) -> datetime:
    """
    Binary search to find earliest available date.

    Args:
        client: Kalshi API client
        series_ticker: Series ticker
        earliest_guess: Earliest possible date to test
        latest_guess: Latest date (known to have data)

    Returns:
        Earliest date with available markets
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Binary search for earliest date in {series_ticker}")
    logger.info(f"Range: {earliest_guess.date()} to {latest_guess.date()}")
    logger.info(f"{'='*60}\n")

    left = earliest_guess
    right = latest_guess
    earliest_found = latest_guess

    while (right - left).days > 1:
        mid = left + (right - left) / 2

        logger.info(f"Testing {mid.date()}...")
        result = test_date_availability(client, series_ticker, mid)

        if result["available"]:
            logger.info(f"  ✓ Found {result['count']} markets")
            earliest_found = mid
            right = mid  # Search earlier
        else:
            logger.info(f"  ✗ No markets found")
            left = mid  # Search later

    return earliest_found


def main():
    parser = argparse.ArgumentParser(description="Test Kalshi historical data availability")
    parser.add_argument(
        "--city",
        default="chicago",
        choices=list(CITY_SERIES.keys()),
        help="City to test (default: chicago)"
    )
    parser.add_argument(
        "--test-dates",
        type=str,
        help="Comma-separated test dates (YYYY-MM-DD format, e.g., '2024-01-01,2024-04-01')"
    )
    parser.add_argument(
        "--binary-search",
        action="store_true",
        help="Perform binary search to find exact earliest date"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save results"
    )

    args = parser.parse_args()

    # Initialize Kalshi client
    api_key = os.getenv("KALSHI_API_KEY")
    private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "./kalshi_private_key.pem")

    if not api_key:
        logger.error("KALSHI_API_KEY environment variable not set")
        sys.exit(1)

    client = KalshiClient(api_key=api_key, private_key_path=private_key_path)
    series_ticker = CITY_SERIES[args.city]

    logger.info(f"\nTesting availability for {args.city.upper()} ({series_ticker})\n")

    # Default test dates if not provided
    if args.test_dates:
        test_dates_str = args.test_dates.split(",")
    else:
        test_dates_str = [
            "2024-01-01",
            "2024-04-01",
            "2024-07-01",
            "2024-10-01",
        ]

    test_dates = [
        datetime.strptime(d.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        for d in test_dates_str
    ]

    # Test each date
    results = []
    for test_date in test_dates:
        logger.info(f"Testing {test_date.date()}...")
        result = test_date_availability(client, series_ticker, test_date)
        results.append((test_date, result))

        if result["available"]:
            logger.info(f"  ✓ {result['count']} markets found")
            if result.get("earliest"):
                logger.info(f"    Earliest close: {result['earliest'].date()}")
                logger.info(f"    Latest close: {result['latest'].date()}")
        else:
            logger.info(f"  ✗ No markets found")

        logger.info("")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}\n")

    available_dates = [date for date, res in results if res["available"]]
    unavailable_dates = [date for date, res in results if not res["available"]]

    if available_dates:
        earliest_available = min(available_dates)
        logger.info(f"✓ Data available from: {earliest_available.date()} onwards")
        logger.info(f"  Tested dates with data: {len(available_dates)}")

        if args.binary_search and unavailable_dates:
            # Find exact earliest date
            earliest_guess = min(unavailable_dates)
            latest_guess = min(available_dates)

            exact_earliest = find_earliest_date(
                client, series_ticker, earliest_guess, latest_guess
            )

            logger.info(f"\n✓ Exact earliest date (via binary search): {exact_earliest.date()}")
    else:
        logger.info(f"✗ No data found for any tested dates")
        logger.info(f"  All {len(test_dates)} tested dates returned no markets")

    if unavailable_dates:
        logger.info(f"\n✗ Tested dates WITHOUT data: {len(unavailable_dates)}")
        for date in sorted(unavailable_dates):
            logger.info(f"    {date.date()}")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(f"Kalshi Historical Availability Test - {series_ticker}\n")
            f.write(f"Test Date: {datetime.now().isoformat()}\n\n")

            if available_dates:
                earliest = min(available_dates)
                f.write(f"Earliest available date: {earliest.date()}\n")

                if args.binary_search and unavailable_dates:
                    f.write(f"Exact earliest date: {exact_earliest.date()}\n")
            else:
                f.write("No data found\n")

            f.write(f"\nTest Results:\n")
            for date, res in results:
                status = "AVAILABLE" if res["available"] else "NOT FOUND"
                f.write(f"  {date.date()}: {status}")
                if res["available"]:
                    f.write(f" ({res['count']} markets)")
                f.write("\n")

        logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
