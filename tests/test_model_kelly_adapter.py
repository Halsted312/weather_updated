import math

from backtest.model_kelly_adapter import ModelKellyBacktestStrategy, probability_from_tmax


def test_probability_between_bracket():
    prob = probability_from_tmax(mu=65.0, sigma=4.0, strike_type="between", floor_strike=60.0, cap_strike=70.0)
    assert 0.0 < prob < 1.0
    symmetric_prob = probability_from_tmax(mu=65.0, sigma=4.0, strike_type="between", floor_strike=55.0, cap_strike=75.0)
    assert symmetric_prob > prob


def test_probability_extremes_monotonic():
    high_prob = probability_from_tmax(mu=80.0, sigma=3.0, strike_type="greater", floor_strike=70.0, cap_strike=None)
    low_prob = probability_from_tmax(mu=60.0, sigma=3.0, strike_type="greater", floor_strike=70.0, cap_strike=None)
    assert math.isclose(high_prob, 1.0, rel_tol=0, abs_tol=1e-3)
    assert low_prob < 0.5
    less_prob = probability_from_tmax(mu=50.0, sigma=5.0, strike_type="less", floor_strike=None, cap_strike=60.0)
    assert less_prob > 0.5


def test_log_odds_pooling_moves_prob_toward_market():
    adapter = object.__new__(ModelKellyBacktestStrategy)
    adapter.market_odds_weight = 0.5
    pooled = adapter._pool_with_market(0.9, 0.2)
    assert 0.5 < pooled < 0.9
