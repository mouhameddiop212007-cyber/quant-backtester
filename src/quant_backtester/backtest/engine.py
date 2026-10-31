"""Moteur de backtest vectorise, avec frais de transaction et slippage."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..strategies.base import Strategy


@dataclass
class BacktestResult:
    strategy_name: str
    returns: pd.Series    # rendements journaliers nets de frais
    equity: pd.Series     # valeur du portefeuille
    positions: pd.Series  # position effectivement detenue chaque jour
    turnover: pd.Series   # variation absolue de position (proxy du volume trade)


def run_backtest(
    prices: pd.Series,
    strategy: Strategy,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Simule la strategie sur ``prices``.

    - Le signal du jour t est execute a t+1 (``shift(1)``) : pas de look-ahead.
    - Chaque variation de position coute ``cost_bps + slippage_bps`` points de base.
    """
    signals = strategy.generate_signals(prices)
    positions = signals.shift(1).fillna(0.0)

    asset_returns = prices.pct_change().fillna(0.0)
    turnover = positions.diff().abs()
    turnover.iloc[0] = abs(positions.iloc[0])

    costs = turnover * (cost_bps + slippage_bps) / 10_000
    net_returns = positions * asset_returns - costs

    equity = initial_capital * (1 + net_returns).cumprod()
    return BacktestResult(strategy.name, net_returns, equity, positions, turnover)
