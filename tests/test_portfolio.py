import numpy as np
import pandas as pd
import pytest

from quant_backtester.portfolio import (
    efficient_frontier, equal_weight, historical_cvar, historical_var, make_weight_fn,
    max_sharpe, mean_cov, min_variance, parametric_var, portfolio_stats,
    risk_contributions, risk_parity, run_portfolio_backtest,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2018-01-01", periods=700)
    common = rng.normal(0.0003, 0.008, (700, 1))
    data = common * [[1.0, 0.3, 0.6]] + rng.normal([[0.0002, 0.0001, 0.0003]], [[0.006, 0.004, 0.01]], (700, 3))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


@pytest.fixture
def prices(returns):
    return 100 * (1 + returns).cumprod()


# ---------- optimisation ----------
def test_min_variance_uncorrelated_is_inverse_variance():
    w = min_variance(np.diag([0.04, 0.01]))
    assert w == pytest.approx([0.2, 0.8], abs=1e-4)


def test_risk_parity_uncorrelated_is_inverse_vol():
    w = risk_parity(np.diag([0.04, 0.01]))  # vols 0.2 et 0.1
    assert w == pytest.approx([1 / 3, 2 / 3], abs=1e-4)


def test_risk_parity_equalizes_risk_contributions(returns):
    _, cov = mean_cov(returns)
    contrib = risk_contributions(risk_parity(cov), cov)
    assert contrib == pytest.approx(np.full(3, 1 / 3), abs=1e-3)


@pytest.mark.parametrize("method", ["equal_weight", "min_variance", "max_sharpe", "risk_parity"])
def test_weights_are_long_only_and_sum_to_one(returns, method):
    w = make_weight_fn(method)(returns)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= -1e-9).all()


def test_max_sharpe_beats_equal_weight_in_sample(returns):
    mu, cov = mean_cov(returns)
    s_opt = portfolio_stats(max_sharpe(mu, cov), mu, cov)[2]
    s_eq = portfolio_stats(equal_weight(3), mu, cov)[2]
    assert s_opt >= s_eq - 1e-9


def test_min_variance_has_lowest_volatility(returns):
    mu, cov = mean_cov(returns)
    vol_min = portfolio_stats(min_variance(cov), mu, cov)[1]
    for w in (equal_weight(3), max_sharpe(mu, cov), risk_parity(cov)):
        assert vol_min <= portfolio_stats(w, mu, cov)[1] + 1e-9


def test_frontier_volatility_increases_with_return(returns):
    mu, cov = mean_cov(returns)
    frontier = efficient_frontier(mu, cov, n_points=15)
    assert len(frontier) >= 10
    assert frontier["volatility"].is_monotonic_increasing or (
        frontier["volatility"].diff().dropna() > -1e-6
    ).all()


# ---------- risque ----------
def test_var_cvar_known_values():
    r = np.linspace(-0.10, 0.10, 201)
    assert historical_var(r, 0.95) == pytest.approx(0.09)
    assert historical_cvar(r, 0.95) == pytest.approx(0.095)


def test_cvar_at_least_var(returns):
    r = returns.mean(axis=1)
    assert historical_cvar(r) >= historical_var(r)


def test_parametric_var_close_to_historical_for_gaussian():
    r = np.random.default_rng(0).normal(0, 0.01, 100_000)
    assert parametric_var(r) == pytest.approx(historical_var(r), rel=0.05)


# ---------- backtest ----------
def test_daily_equal_weight_matches_average_return(prices):
    res = run_portfolio_backtest(prices, make_weight_fn("equal_weight"),
                                 lookback=20, rebalance_days=1, cost_bps=0)
    expected = prices.pct_change().dropna().mean(axis=1).iloc[20:]
    pd.testing.assert_series_equal(res.returns, expected, check_names=False, check_freq=False)


def test_backtest_alignment_and_window_length(prices):
    seen = []

    def spy(window):
        seen.append(window)
        return equal_weight(window.shape[1])

    res = run_portfolio_backtest(prices, spy, lookback=60, rebalance_days=21, cost_bps=0)
    rets = prices.pct_change().dropna()
    assert res.weights.index[0] == rets.index[59]      # 1er reequilibrage a la cloture du jour 60
    assert res.returns.index[0] == rets.index[60]      # applique le lendemain
    assert all(len(w) == 60 for w in seen)
    assert all(w.index[-1] == d for w, d in zip(seen, res.weights.index))


def test_backtest_has_no_future_leakage(prices):
    fn = make_weight_fn("min_variance")
    base = run_portfolio_backtest(prices, fn, lookback=120, rebalance_days=21)
    altered = prices.copy()
    cut = 500
    altered.iloc[cut:] *= 3
    other = run_portfolio_backtest(altered, fn, lookback=120, rebalance_days=21)
    cut_date = prices.index[cut]
    before = base.weights.index < cut_date
    pd.testing.assert_frame_equal(base.weights[before], other.weights[other.weights.index < cut_date])


def test_costs_reduce_returns(prices):
    fn = make_weight_fn("min_variance")
    free = run_portfolio_backtest(prices, fn, lookback=120, cost_bps=0)
    paid = run_portfolio_backtest(prices, fn, lookback=120, cost_bps=50)
    assert (1 + paid.returns).prod() < (1 + free.returns).prod()


def test_backtest_not_enough_data(prices):
    with pytest.raises(ValueError):
        run_portfolio_backtest(prices.iloc[:100], make_weight_fn("equal_weight"), lookback=252)
