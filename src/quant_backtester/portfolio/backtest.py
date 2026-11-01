"""Backtest de portefeuille avec reequilibrage periodique, derive des poids et couts."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .optimization import equal_weight, max_sharpe, mean_cov, min_variance, risk_parity

WeightFn = Callable[[pd.DataFrame], np.ndarray]


@dataclass
class PortfolioResult:
    returns: pd.Series     # rendements journaliers nets, a partir du 1er jour investi
    weights: pd.DataFrame  # poids cibles a chaque date de reequilibrage
    turnover: pd.Series    # somme des |variations de poids| a chaque reequilibrage


def make_weight_fn(method: str, risk_free: float = 0.0) -> WeightFn:
    """Fonction (fenetre de rendements -> poids) pour une methode d'allocation."""
    def fn(window: pd.DataFrame) -> np.ndarray:
        n = window.shape[1]
        if method == "equal_weight":
            return equal_weight(n)
        mu, cov = mean_cov(window)
        try:
            if method == "min_variance":
                return min_variance(cov)
            if method == "max_sharpe":
                return max_sharpe(mu, cov, risk_free)
            if method == "risk_parity":
                return risk_parity(cov)
        except RuntimeError as exc:
            warnings.warn(f"{method} : {exc} ; repli sur l'equipondere.")
            return equal_weight(n)
        raise ValueError(f"Methode inconnue : {method}")
    return fn


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weight_fn: WeightFn,
    lookback: int = 252,
    rebalance_days: int = 21,
    cost_bps: float = 5.0,
) -> PortfolioResult:
    """Simule un portefeuille.

    - A la cloture du jour t, ``weight_fn`` ne voit que les ``lookback`` derniers
      rendements (jusqu'a t inclus) ; les nouveaux poids s'appliquent des t+1.
    - Entre deux reequilibrages, les poids derivent avec les prix.
    - Le cout (``cost_bps`` x turnover) est preleve le jour ou les nouveaux poids
      prennent effet.
    """
    asset_returns = prices.pct_change().dropna()
    R = asset_returns.to_numpy()
    n_days, n_assets = R.shape
    first = lookback - 1
    if n_days <= lookback:
        raise ValueError("Pas assez de donnees pour la fenetre d'estimation.")

    w = np.zeros(n_assets)  # cash avant le premier reequilibrage
    pending_cost = 0.0
    port = np.zeros(n_days)
    dates, target_rows, turnovers = [], [], []

    for t in range(n_days):
        gross = float(w @ R[t])
        port[t] = gross - pending_cost
        pending_cost = 0.0
        w_drift = w * (1.0 + R[t]) / (1.0 + gross) if w.sum() > 0 else w

        if t >= first and (t - first) % rebalance_days == 0:
            window = asset_returns.iloc[t - lookback + 1 : t + 1]
            target = np.asarray(weight_fn(window), float)
            if abs(target.sum() - 1.0) > 1e-6 or (target < -1e-9).any():
                raise ValueError("Poids invalides : long-only, somme = 1 attendue.")
            turnover = float(np.abs(target - w_drift).sum())
            pending_cost = turnover * cost_bps / 10_000
            w = target
            dates.append(asset_returns.index[t])
            target_rows.append(target)
            turnovers.append(turnover)
        else:
            w = w_drift

    returns = pd.Series(port, index=asset_returns.index).iloc[first + 1 :]
    weights = pd.DataFrame(target_rows, index=dates, columns=prices.columns)
    return PortfolioResult(returns, weights, pd.Series(turnovers, index=dates))
