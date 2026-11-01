"""Optimisation de portefeuille long-only (somme des poids = 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def mean_cov(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> tuple[pd.Series, pd.DataFrame]:
    """Rendements moyens et covariance annualises, a partir de rendements journaliers."""
    return returns.mean() * periods, returns.cov() * periods


def portfolio_stats(weights, mu, cov, risk_free: float = 0.0) -> tuple[float, float, float]:
    """(rendement, volatilite, Sharpe) annualises d'un portefeuille."""
    w, m, c = np.asarray(weights, float), np.asarray(mu, float), np.asarray(cov, float)
    ret = float(w @ m)
    vol = float(np.sqrt(w @ c @ w))
    sharpe = (ret - risk_free) / vol if vol > 1e-12 else float("nan")
    return ret, vol, sharpe


def _solve(objective, n: int, extra_constraints=(), x0=None) -> np.ndarray:
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}, *extra_constraints]
    x0 = np.full(n, 1.0 / n) if x0 is None else x0
    res = minimize(
        objective, x0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
        constraints=constraints, options={"ftol": 1e-12, "maxiter": 500},
    )
    if not res.success:
        raise RuntimeError(f"Optimisation non convergee : {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def min_variance(cov) -> np.ndarray:
    """Portefeuille de variance minimale."""
    c = np.asarray(cov, float)
    return _solve(lambda w: w @ c @ w, len(c))


def max_sharpe(mu, cov, risk_free: float = 0.0) -> np.ndarray:
    """Portefeuille tangent : maximise le ratio de Sharpe."""
    m, c = np.asarray(mu, float), np.asarray(cov, float)

    def neg_sharpe(w):
        vol = np.sqrt(w @ c @ w)
        return -(w @ m - risk_free) / max(vol, 1e-12)

    return _solve(neg_sharpe, len(m))


def risk_parity(cov) -> np.ndarray:
    """Chaque actif contribue autant au risque total.

    On minimise 0.5 w'Σw - (1/n) Σ log(w_i), un probleme convexe dont la
    solution (une fois normalisee) egalise les contributions au risque.
    """
    c = np.asarray(cov, float)
    n = len(c)
    x0 = 1.0 / (np.sqrt(np.diag(c)) * np.sqrt(n))  # solution exacte si actifs non correles

    res = minimize(
        lambda w: 0.5 * w @ c @ w - np.log(w).sum() / n,
        x0,
        jac=lambda w: c @ w - 1.0 / (n * w),
        method="L-BFGS-B",
        bounds=[(1e-10, None)] * n,
    )
    if not res.success:
        raise RuntimeError(f"Optimisation non convergee : {res.message}")
    return res.x / res.x.sum()


def risk_contributions(weights, cov) -> np.ndarray:
    """Part du risque total apportee par chaque actif (somme = 1)."""
    w, c = np.asarray(weights, float), np.asarray(cov, float)
    marginal = c @ w
    return w * marginal / (w @ marginal)


def efficient_frontier(mu, cov, n_points: int = 30) -> pd.DataFrame:
    """Frontiere efficiente long-only : variance minimale pour chaque rendement cible."""
    m, c = np.asarray(mu, float), np.asarray(cov, float)
    names = list(mu.index) if hasattr(mu, "index") else [f"w{i}" for i in range(len(m))]
    w_min = min_variance(c)

    rows = []
    for target in np.linspace(float(w_min @ m), float(m.max()), n_points):
        constraint = [{"type": "eq", "fun": lambda w, t=target: w @ m - t}]
        try:
            w = _solve(lambda w: w @ c @ w, len(m), constraint, x0=w_min)
        except RuntimeError:
            continue  # point inatteignable (bord de la frontiere)
        ret, vol, _ = portfolio_stats(w, m, c)
        rows.append({"return": ret, "volatility": vol, **dict(zip(names, w))})
    return pd.DataFrame(rows)
