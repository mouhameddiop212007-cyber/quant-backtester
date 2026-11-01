"""Optimisation de parametres et validation walk-forward.

Principe : on choisit les parametres sur une periode d'entrainement, puis on
mesure la performance sur une periode de test que l'optimisation n'a jamais vue.
Un ecart important entre les deux signale du surapprentissage.

Pas de fuite d'information : un signal a la date t n'utilise que des prix <= t,
donc on peut lancer le backtest sur toute la serie et decouper les rendements
ensuite (cela evite aussi de perdre la periode de chauffe des indicateurs).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..metrics import sharpe_ratio
from ..strategies.base import Strategy
from .engine import run_backtest

MIN_TEST_OBSERVATIONS = 20


def _window(returns: pd.Series, start=None, end=None) -> pd.Series:
    """Rendements dans [start, end[ (fin exclue : pas de chevauchement entre fenetres)."""
    mask = np.ones(len(returns), dtype=bool)
    if start is not None:
        mask &= returns.index >= pd.Timestamp(start)
    if end is not None:
        mask &= returns.index < pd.Timestamp(end)
    return returns[mask]


def evaluate(prices: pd.Series, strategy: Strategy, start=None, end=None, **bt_kwargs) -> pd.Series:
    """Rendements nets de la strategie sur la fenetre [start, end[."""
    result = run_backtest(prices, strategy, **bt_kwargs)
    return _window(result.returns, start, end)


def _to_python(value):
    return value.item() if hasattr(value, "item") else value


def grid_search(
    prices: pd.Series,
    factory: Callable[..., Strategy],
    param_grid: dict[str, list],
    start=None,
    end=None,
    metric: Callable[[pd.Series], float] = sharpe_ratio,
    **bt_kwargs,
) -> pd.DataFrame:
    """Teste toutes les combinaisons de parametres et les classe par score decroissant.

    Les combinaisons invalides (ex. SMA avec fast >= slow) sont ignorees.
    """
    keys = list(param_grid)
    rows = []
    for values in itertools.product(*param_grid.values()):
        params = dict(zip(keys, values))
        try:
            strategy = factory(**params)
        except ValueError:
            continue
        returns = evaluate(prices, strategy, start, end, **bt_kwargs)
        rows.append({**params, "score": metric(returns)})
    if not rows:
        raise ValueError("Aucune combinaison de parametres valide.")
    table = pd.DataFrame(rows)
    return table.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)


def best_params(table: pd.DataFrame) -> dict:
    """Parametres de la premiere ligne d'un resultat de grid_search."""
    return {c: _to_python(table.loc[0, c]) for c in table.columns if c != "score"}


@dataclass
class WalkForwardResult:
    returns: pd.Series   # rendements hors-echantillon, tous les folds concatenes
    folds: pd.DataFrame  # une ligne par fold : fenetres, parametres choisis, scores


def walk_forward(
    prices: pd.Series,
    factory: Callable[..., Strategy],
    param_grid: dict[str, list],
    train_years: int = 5,
    test_years: int = 1,
    metric: Callable[[pd.Series], float] = sharpe_ratio,
    **bt_kwargs,
) -> WalkForwardResult:
    """Walk-forward glissant : optimise sur ``train_years``, teste sur ``test_years``, decale."""
    idx = prices.index
    train_start = idx[0]
    fold_rows, oos_parts = [], []

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if train_end > idx[-1]:
            break

        table = grid_search(prices, factory, param_grid, train_start, train_end, metric, **bt_kwargs)
        params = best_params(table)
        test_returns = evaluate(prices, factory(**params), train_end, test_end, **bt_kwargs)
        if len(test_returns) < MIN_TEST_OBSERVATIONS:
            break

        fold_rows.append({
            "train_start": train_start.date(),
            "train_end": train_end.date(),
            "test_end": min(test_end, idx[-1]).date(),  # tronque a la fin des donnees
            "params": params,
            "train_score": table.loc[0, "score"],
            "test_score": metric(test_returns),
        })
        oos_parts.append(test_returns)
        train_start = train_start + pd.DateOffset(years=test_years)

    if not oos_parts:
        raise ValueError("Pas assez de donnees pour un seul fold walk-forward.")
    return WalkForwardResult(pd.concat(oos_parts), pd.DataFrame(fold_rows))
