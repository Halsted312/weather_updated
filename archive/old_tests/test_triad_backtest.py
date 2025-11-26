import pandas as pd

from backtest.fees import net_payout_cents
from scripts.backtest_triad import BacktestConfig, prepare_panel, short_round_trip, simulate_triad
from scripts.triad_momentum import TriadConfig, compute_scores, select_triads


def _sample_panel() -> pd.DataFrame:
    ts0 = pd.Timestamp("2024-11-01 12:00:00")
    ts1 = ts0 + pd.Timedelta(minutes=1)
    rows = [
        # t0
        {
            "city": "chicago",
            "series_ticker": "SER",
            "event_ticker": "EVT",
            "market_ticker": "M1",
            "ts_utc": ts0,
            "ts_local": ts0,
            "local_date": ts0.date(),
            "bracket_idx": 1,
            "num_brackets": 3,
            "close_c": 40,
            "open_c": 40,
            "high_c": 42,
            "low_c": 39,
            "volume": 10.0,
            "mid_accel_left_diff": 0.0,
            "mid_accel_right_diff": -0.1,
            "mid_acceleration": 0.0,
            "mid_velocity": 0.0,
            "volume_delta": 0.0,
            "wx_running_max": 0.0,
            "mid_prob": 0.4,
            "mid_prob_left": None,
            "mid_prob_right": 0.5,
            "mid_velocity_left": None,
            "mid_velocity_right": 0.0,
            "mid_acceleration_left": None,
            "mid_acceleration_right": 0.0,
            "ras_accel": -0.2,
        },
        {
            "city": "chicago",
            "series_ticker": "SER",
            "event_ticker": "EVT",
            "market_ticker": "M2",
            "ts_utc": ts0,
            "ts_local": ts0,
            "local_date": ts0.date(),
            "bracket_idx": 2,
            "num_brackets": 3,
            "close_c": 50,
            "open_c": 50,
            "high_c": 52,
            "low_c": 49,
            "volume": 12.0,
            "mid_accel_left_diff": 0.2,
            "mid_accel_right_diff": 0.3,
            "mid_acceleration": 0.2,
            "mid_velocity": 0.1,
            "volume_delta": 0.0,
            "wx_running_max": 0.0,
            "mid_prob": 0.5,
            "mid_prob_left": 0.4,
            "mid_prob_right": 0.6,
            "mid_velocity_left": 0.0,
            "mid_velocity_right": 0.0,
            "mid_acceleration_left": 0.0,
            "mid_acceleration_right": 0.0,
            "ras_accel": 1.0,
        },
        {
            "city": "chicago",
            "series_ticker": "SER",
            "event_ticker": "EVT",
            "market_ticker": "M3",
            "ts_utc": ts0,
            "ts_local": ts0,
            "local_date": ts0.date(),
            "bracket_idx": 3,
            "num_brackets": 3,
            "close_c": 60,
            "open_c": 60,
            "high_c": 62,
            "low_c": 58,
            "volume": 11.0,
            "mid_accel_left_diff": -0.1,
            "mid_accel_right_diff": 0.0,
            "mid_acceleration": -0.1,
            "mid_velocity": 0.0,
            "volume_delta": 0.0,
            "wx_running_max": 0.0,
            "mid_prob": 0.6,
            "mid_prob_left": 0.5,
            "mid_prob_right": None,
            "mid_velocity_left": 0.0,
            "mid_velocity_right": None,
            "mid_acceleration_left": 0.0,
            "mid_acceleration_right": None,
            "ras_accel": -0.3,
        },
        # t1
        {
            "city": "chicago",
            "series_ticker": "SER",
            "event_ticker": "EVT",
            "market_ticker": "M1",
            "ts_utc": ts1,
            "ts_local": ts1,
            "local_date": ts1.date(),
            "bracket_idx": 1,
            "num_brackets": 3,
            "close_c": 42,
            "open_c": 41,
            "high_c": 43,
            "low_c": 40,
            "volume": 8.0,
            "mid_accel_left_diff": 0.0,
            "mid_accel_right_diff": 0.0,
            "mid_acceleration": 0.0,
            "mid_velocity": 0.0,
            "volume_delta": 0.0,
            "wx_running_max": 0.0,
            "mid_prob": 0.4,
            "mid_prob_left": None,
            "mid_prob_right": 0.55,
            "mid_velocity_left": None,
            "mid_velocity_right": 0.0,
            "mid_acceleration_left": None,
            "mid_acceleration_right": 0.0,
            "ras_accel": 0.0,
        },
        {
            "city": "chicago",
            "series_ticker": "SER",
            "event_ticker": "EVT",
            "market_ticker": "M2",
            "ts_utc": ts1,
            "ts_local": ts1,
            "local_date": ts1.date(),
            "bracket_idx": 2,
            "num_brackets": 3,
            "close_c": 58,
            "open_c": 51,
            "high_c": 60,
            "low_c": 49,
            "volume": 9.0,
            "mid_accel_left_diff": 0.0,
            "mid_accel_right_diff": 0.0,
            "mid_acceleration": 0.0,
            "mid_velocity": 0.0,
            "volume_delta": 0.0,
            "wx_running_max": 0.0,
            "mid_prob": 0.58,
            "mid_prob_left": 0.4,
            "mid_prob_right": 0.6,
            "mid_velocity_left": 0.0,
            "mid_velocity_right": 0.0,
            "mid_acceleration_left": 0.0,
            "mid_acceleration_right": 0.0,
            "ras_accel": 0.0,
        },
        {
            "city": "chicago",
            "series_ticker": "SER",
            "event_ticker": "EVT",
            "market_ticker": "M3",
            "ts_utc": ts1,
            "ts_local": ts1,
            "local_date": ts1.date(),
            "bracket_idx": 3,
            "num_brackets": 3,
            "close_c": 52,
            "open_c": 59,
            "high_c": 62,
            "low_c": 51,
            "volume": 9.0,
            "mid_accel_left_diff": 0.0,
            "mid_accel_right_diff": 0.0,
            "mid_acceleration": 0.0,
            "mid_velocity": 0.0,
            "volume_delta": 0.0,
            "wx_running_max": 0.0,
            "mid_prob": 0.52,
            "mid_prob_left": 0.6,
            "mid_prob_right": None,
            "mid_velocity_left": 0.0,
            "mid_velocity_right": None,
            "mid_acceleration_left": 0.0,
            "mid_acceleration_right": None,
            "ras_accel": 0.0,
        },
    ]
    return pd.DataFrame(rows)


def test_simulate_triad_maker_fill():
    df = _sample_panel()
    triad_cfg = TriadConfig(min_volume=0.0, min_score=-1.0, max_spread_cents=5.0)
    scored = compute_scores(df, triad_cfg)
    prepared = prepare_panel(scored, hold_minutes=1)
    intents = select_triads(prepared, min_score=-1.0)

    bt_cfg = BacktestConfig(
        hold_minutes=1,
        order_size=1,
        hedge_multiplier=0.5,
        maker_slippage_cents=0,
        taker_slippage_cents=0,
        allow_taker=False,
    )

    lookup = prepared.set_index(["ts_utc", "market_ticker"])
    trade = None
    for intent in intents:
        trade = simulate_triad(intent, lookup, bt_cfg)
        if trade:
            break

    assert trade is not None, "Expected a filled triad trade"

    expected_center = net_payout_cents(1, 50, 58, "maker", "taker")
    expected_left = short_round_trip(1, 40, 42, "maker", "taker")
    expected_right = short_round_trip(1, 60, 52, "maker", "taker")
    assert trade.pnl_cents == expected_center + expected_left + expected_right


def test_simulate_triad_requires_future_bars():
    df = _sample_panel().query("ts_utc == ts_utc.min()")  # drop future minute
    triad_cfg = TriadConfig(min_volume=0.0, min_score=-1.0, max_spread_cents=5.0)
    scored = compute_scores(df, triad_cfg)
    prepared = prepare_panel(scored, hold_minutes=1)
    intents = select_triads(prepared, min_score=-1.0)

    bt_cfg = BacktestConfig(hold_minutes=1, order_size=1, hedge_multiplier=0.0)
    lookup = prepared.set_index(["ts_utc", "market_ticker"])

    for intent in intents:
        assert simulate_triad(intent, lookup, bt_cfg) is None
