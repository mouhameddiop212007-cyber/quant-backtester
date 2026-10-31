import numpy as np
import pandas as pd
import pytest

from quant_backtester.backtest import run_backtest
from quant_backtester.strategies import BuyAndHold, SMACrossover


@pytest.fixture
def prices():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(42)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 400)), index=idx)


def test_no_lookahead_first_day_has_no_position(prices):
    res = run_backtest(prices, BuyAndHold())
    assert res.positions.iloc[0] == 0.0  # le signal de J s'execute a J+1


def test_buy_and_hold_without_costs_matches_asset(prices):
    res = run_backtest(prices, BuyAndHold(), cost_bps=0, slippage_bps=0)
    expected = prices.pct_change().fillna(0.0)
    expected.iloc[0] = 0.0
    pd.testing.assert_series_equal(res.returns, expected, check_names=False)


def test_costs_reduce_performance(prices):
    free = run_backtest(prices, SMACrossover(10, 30), cost_bps=0, slippage_bps=0)
    paid = run_backtest(prices, SMACrossover(10, 30), cost_bps=20, slippage_bps=10)
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]


def test_sma_flat_during_warmup(prices):
    sig = SMACrossover(10, 30).generate_signals(prices)
    assert (sig.iloc[:29] == 0).all()


def test_sma_invalid_windows():
    with pytest.raises(ValueError):
        SMACrossover(50, 20)
