import pandas as pd

from backtest.model_strategy import ExecParams, ModelKellyStrategy


def make_row(prob: float) -> pd.Series:
    return pd.Series(
        {
            "yes_bid_close": 30,
            "yes_ask_close": 32,
            "p_model": prob,
            "strike_type": "between",
            "minute_of_day_local": 600,
            "sigma_est": 3.0,
        }
    )


def test_allowed_time_window_blocks_outside_minutes():
    params = ExecParams(allowed_time_windows={"between": [(600, 660)]})
    strategy = ModelKellyStrategy(exec_params=params)
    row = make_row(0.7)
    row["minute_of_day_local"] = 500
    assert strategy.signal_for_row(row) is None
    row["minute_of_day_local"] = 610
    assert strategy.signal_for_row(row) is not None


def test_sigma_gate_suppresses_high_uncertainty():
    params = ExecParams(sigma_gate=4.0)
    strategy = ModelKellyStrategy(exec_params=params)
    row = make_row(0.75)
    row["sigma_est"] = 5.0
    assert strategy.signal_for_row(row) is None
    row["sigma_est"] = 3.0
    assert strategy.signal_for_row(row) is not None
