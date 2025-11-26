from datetime import datetime, timezone

from ingest.continuous_ingest import normalize_market_window
from weather.time_utils import coerce_datetime_to_utc, local_date_from_utc


def test_coerce_datetime_to_utc_handles_naive_and_strings():
    naive = datetime(2024, 2, 1, 12, 30)
    aware = coerce_datetime_to_utc(naive)
    assert aware.tzinfo == timezone.utc
    assert aware.hour == 12

    iso = "2024-02-01T12:30:00Z"
    aware_from_str = coerce_datetime_to_utc(iso)
    assert aware_from_str == aware


def test_normalize_market_window_accepts_mixed_payloads():
    market = {
        "ticker": "KXTEST-20240201", 
        "open_time": "2024-02-01T10:00:00Z",
        "close_time": datetime(2024, 2, 1, 18, 0),
    }
    open_time, close_time = normalize_market_window(market)

    assert open_time.tzinfo == timezone.utc
    assert close_time.tzinfo == timezone.utc
    assert close_time > open_time

    market_missing_close = {
        "ticker": "KXTEST-20240202",
        "open_time": datetime(2024, 2, 1, 10, 0, tzinfo=timezone.utc),
        "expiration_time": "2024-02-01T23:00:00Z",
    }
    _, exp_close = normalize_market_window(market_missing_close)
    assert exp_close == datetime(2024, 2, 1, 23, 0, tzinfo=timezone.utc)


def test_local_date_from_utc_respects_city_timezone():
    utc_dt = datetime(2024, 1, 1, 7, 0, tzinfo=timezone.utc)
    # 7:00 UTC on Jan 1 is still Dec 31 local time in Los Angeles
    assert local_date_from_utc(utc_dt, "la").isoformat() == "2023-12-31"

    later = datetime(2024, 3, 15, 12, 0, tzinfo=timezone.utc)
    assert local_date_from_utc(later, "chicago").isoformat() == "2024-03-15"
