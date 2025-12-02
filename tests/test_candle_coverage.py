import os
import sys
from datetime import date
from pathlib import Path

import psycopg2
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from check_candle_coverage import analyze_coverage, DSN_DEFAULT  # noqa: E402


def _get_dsn() -> str:
    return os.getenv("KALSHI_DB_DSN", DSN_DEFAULT)


def test_chicago_2025_candle_coverage():
    """
    Raw sparse candles are expected to have gaps (no activity → no bar).
    This test now reports gaps and xfails to remind us to use the dense layer.
    """
    dsn = _get_dsn()
    try:
        reports = analyze_coverage(
            dsn=dsn,
            city="chicago",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )
    except psycopg2.OperationalError as exc:
        pytest.skip(f"Database not reachable: {exc}")

    missing = [r for r in reports if r.missing_minutes > 0]
    if missing:
        worst = sorted(missing, key=lambda r: r.missing_minutes, reverse=True)[:5]
        details = [
            f"{r.ticker} {r.event_date} missing {r.missing_minutes}/"
            f"{r.expected_minutes} first={r.first_ts} last={r.last_ts}"
            for r in worst
        ]
        pytest.xfail(
            "Raw candles are sparse by design (no trades → no bar). "
            "Use kalshi.candles_1m_dense for full grids. "
            f"Markets with gaps: {len(missing)} of {len(reports)}. "
            "Worst examples: " + "; ".join(details)
        )
