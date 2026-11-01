import numpy as np
import pandas as pd
import pytest

from quant_backtester.backtest import best_params, evaluate, grid_search, walk_forward
from quant_backtester.strategies import Momentum, SMACrossover


@pytest.fixture
def prices():
    idx = pd.bdate_range("2010-01-01", periods=2600)  # environ 10 ans
    rng = np.random.default_rng(7)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.01, len(idx))), index=idx)


GRID = {"lookback": [20, 60, 120]}


def test_grid_search_sorted_best_first(prices):
    table = grid_search(prices, Momentum, GRID)
    assert len(table) == 3
    assert table["score"].is_monotonic_decreasing


def test_grid_search_skips_invalid_combinations(prices):
    table = grid_search(prices, SMACrossover, {"fast": [10, 50], "slow": [30, 40]})
    assert ((table["fast"] < table["slow"])).all()
    assert len(table) == 2  # seuls (10,30) et (10,40) sont valides


def test_best_params_keeps_integer_type(prices):
    params = best_params(grid_search(prices, Momentum, GRID))
    assert isinstance(params["lookback"], int)


def test_evaluate_window_excludes_end_date(prices):
    cut = prices.index[500]
    r = evaluate(prices, Momentum(20), end=cut)
    assert r.index.max() < cut


def test_walk_forward_windows_do_not_overlap(prices):
    wf = walk_forward(prices, Momentum, GRID, train_years=3, test_years=1)
    assert wf.returns.index.is_unique and wf.returns.index.is_monotonic_increasing
    assert wf.returns.index.min() >= prices.index[0] + pd.DateOffset(years=3)
    assert len(wf.folds) >= 3


def test_walk_forward_has_no_future_leakage(prices):
    """Changer les prix APRES le 1er fold ne doit pas changer le choix du 1er fold."""
    wf1 = walk_forward(prices, Momentum, GRID, train_years=3, test_years=1)
    first_test_end = pd.Timestamp(wf1.folds.loc[0, "test_end"])

    altered = prices.copy()
    cut = altered.index.searchsorted(first_test_end)
    altered.iloc[cut:] *= 3
    wf2 = walk_forward(altered, Momentum, GRID, train_years=3, test_years=1)

    assert wf1.folds.loc[0, "params"] == wf2.folds.loc[0, "params"]
    pd.testing.assert_series_equal(
        wf1.returns[wf1.returns.index < first_test_end],
        wf2.returns[wf2.returns.index < first_test_end],
    )


def test_walk_forward_not_enough_data():
    idx = pd.bdate_range("2020-01-01", periods=100)
    short = pd.Series(np.linspace(100, 120, 100), index=idx)
    with pytest.raises(ValueError):
        walk_forward(short, Momentum, GRID, train_years=5, test_years=1)
