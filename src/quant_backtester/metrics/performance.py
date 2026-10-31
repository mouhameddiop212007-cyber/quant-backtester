"""Metriques de performance et de risque, calculees sur des rendements journaliers."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Rendement annualise geometrique (CAGR)."""
    n = len(returns)
    if n == 0:
        return float("nan")
    growth = float((1 + returns).prod())
    return growth ** (periods / n) - 1


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Ecart-type annualise des rendements."""
    return float(returns.std(ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    returns: pd.Series, risk_free: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Ratio de Sharpe annualise. ``risk_free`` est un taux annuel."""
    excess = returns - risk_free / periods
    std = excess.std(ddof=1)
    if not np.isfinite(std) or std < 1e-12:  # tolerance : erreurs d arrondi flottant
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Serie des drawdowns : perte relative par rapport au plus haut historique."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return equity / peak - 1


def max_drawdown(returns: pd.Series) -> float:
    """Pire drawdown de la periode (valeur negative)."""
    return float(drawdown_series(returns).min())


def summary(returns: pd.Series, risk_free: float = 0.0) -> dict[str, float]:
    """Tableau de synthese des principales metriques."""
    return {
        "Rendement annualise": annualized_return(returns),
        "Volatilite annualisee": annualized_volatility(returns),
        "Sharpe": sharpe_ratio(returns, risk_free),
        "Max drawdown": max_drawdown(returns),
    }
