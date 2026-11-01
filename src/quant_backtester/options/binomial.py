"""Arbre binomial de Cox-Ross-Rubinstein, options europeennes et americaines."""
from __future__ import annotations

import numpy as np

from .black_scholes import check_option


def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              option: str = "call", steps: int = 200, american: bool = False, q: float = 0.0) -> float:
    """Prix par arbre binomial CRR (induction arriere vectorisee)."""
    check_option(option)
    if S <= 0 or K <= 0 or sigma <= 0 or steps < 1:
        raise ValueError("S, K, sigma > 0 et steps >= 1 requis.")
    sign = 1.0 if option == "call" else -1.0
    if T <= 0:
        return max(sign * (S - K), 0.0)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)  # probabilite risque-neutre
    if not 0.0 < p < 1.0:
        raise ValueError("Probabilite hors de (0, 1) : augmente 'steps'.")
    disc = np.exp(-r * dt)

    j = np.arange(steps + 1)
    prices = S * u**j * d ** (steps - j)  # prix aux feuilles
    values = np.maximum(sign * (prices - K), 0.0)

    for _ in range(steps):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            prices = prices[:-1] * u  # prix des noeuds un pas plus tot
            values = np.maximum(values, sign * (prices - K))
    return float(values[0])
