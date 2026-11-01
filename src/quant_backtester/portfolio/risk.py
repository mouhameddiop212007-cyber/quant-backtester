"""Mesures de risque de queue : VaR et CVaR (Expected Shortfall).

Convention : des pertes positives. Une VaR 95 % de 0.02 signifie que, 95 % du
temps, la perte quotidienne reste inferieure a 2 %.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def historical_var(returns, confidence: float = 0.95) -> float:
    """Quantile empirique des pertes."""
    return float(-np.quantile(np.asarray(returns, float), 1.0 - confidence))


def historical_cvar(returns, confidence: float = 0.95) -> float:
    """Perte moyenne dans les cas ou la perte depasse la VaR."""
    r = np.asarray(returns, float)
    threshold = np.quantile(r, 1.0 - confidence)
    return float(-r[r <= threshold + 1e-12].mean())


def parametric_var(returns, confidence: float = 0.95) -> float:
    """VaR sous hypothese de rendements gaussiens (sous-estime souvent les queues epaisses)."""
    r = np.asarray(returns, float)
    return float(-(r.mean() + r.std(ddof=1) * norm.ppf(1.0 - confidence)))
