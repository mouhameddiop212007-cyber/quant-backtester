import numpy as np
import pandas as pd
import pytest

from quant_backtester.strategies import Momentum, RSIMeanReversion, compute_rsi


@pytest.fixture
def idx():
    return pd.bdate_range("2020-01-01", periods=100)


def test_rsi_is_100_on_rising_prices(idx):
    prices = pd.Series(np.arange(100, 200, dtype=float), index=idx)
    assert compute_rsi(prices).iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_on_falling_prices(idx):
    prices = pd.Series(np.arange(200, 100, -1, dtype=float), index=idx)
    assert compute_rsi(prices).iloc[-1] == pytest.approx(0.0)


def test_rsi_stays_between_0_and_100():
    rng = np.random.default_rng(0)
    prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 500)))
    rsi = compute_rsi(prices).dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_rsi_strategy_flat_during_warmup(idx):
    prices = pd.Series(np.linspace(100, 50, 100), index=idx)
    sig = RSIMeanReversion(period=14).generate_signals(prices)
    assert (sig.iloc[:13] == 0).all()


def test_rsi_buys_when_oversold_and_exits_when_overbought(idx):
    # chute prolongee (survente) puis forte hausse (surachat)
    prices = pd.Series(np.r_[np.linspace(100, 60, 50), np.linspace(60, 140, 50)], index=idx)
    sig = RSIMeanReversion(period=14).generate_signals(prices)
    assert sig.iloc[30] == 1.0   # achete pendant la chute
    assert sig.iloc[-1] == 0.0   # sorti apres le surachat


def test_rsi_invalid_thresholds():
    with pytest.raises(ValueError):
        RSIMeanReversion(oversold=80, overbought=20)


def test_momentum_invested_on_uptrend_and_flat_during_warmup(idx):
    prices = pd.Series(np.linspace(100, 200, 100), index=idx)
    sig = Momentum(lookback=20).generate_signals(prices)
    assert (sig.iloc[:20] == 0).all()
    assert (sig.iloc[20:] == 1).all()


def test_momentum_cash_on_downtrend(idx):
    prices = pd.Series(np.linspace(200, 100, 100), index=idx)
    assert (Momentum(lookback=20).generate_signals(prices) == 0).all()
