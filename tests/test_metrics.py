import numpy as np
import pandas as pd
import pytest

from quant_backtester.metrics import (
    annualized_return, drawdown_series, max_drawdown, sharpe_ratio,
)


def test_max_drawdown_known_value():
    r = pd.Series([0.10, -0.50])  # 1.0 -> 1.1 -> 0.55
    assert max_drawdown(r) == pytest.approx(-0.5)


def test_drawdown_is_never_positive():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    assert (drawdown_series(r) <= 1e-12).all()


def test_sharpe_nan_when_no_volatility():
    assert np.isnan(sharpe_ratio(pd.Series([0.01] * 10)))


def test_annualized_return_over_one_year():
    r = pd.Series([0.0] * 251 + [0.10])  # +10 % sur exactement 252 jours
    assert annualized_return(r) == pytest.approx(0.10)
