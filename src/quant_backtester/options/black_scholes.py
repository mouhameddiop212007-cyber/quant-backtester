"""Black-Scholes-Merton : prix, grecques et volatilite implicite (options europeennes).

Conventions (S spot, K strike, T maturite en annees, r taux, sigma vol, q dividende continu) :
- vega : variation du prix pour +1.00 de volatilite (diviser par 100 pour +1 point)
- theta : variation du prix par ANNEE (diviser par 365 pour un jour)
- rho : variation du prix pour +1.00 de taux (diviser par 100 pour +1 point)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def check_option(option: str) -> None:
    if option not in ("call", "put"):
        raise ValueError("option doit valoir 'call' ou 'put'.")


def _check_inputs(S, K, T, sigma) -> None:
    if np.any(np.asarray(S) <= 0) or K <= 0:
        raise ValueError("S et K doivent etre strictement positifs.")
    if T < 0 or sigma < 0:
        raise ValueError("T et sigma doivent etre positifs ou nuls.")


def _d1_d2(S, K, T, r, sigma, q):
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def bs_price(S, K, T, r, sigma, option: str = "call", q: float = 0.0):
    """Prix Black-Scholes d'une option europeenne (S peut etre un tableau)."""
    check_option(option)
    S = np.asarray(S, dtype=float)
    _check_inputs(S, K, T, sigma)

    if T == 0:  # a l'echeance : valeur intrinseque
        return np.maximum(S - K, 0.0) if option == "call" else np.maximum(K - S, 0.0)
    if sigma == 0:  # monde deterministe : valeur actualisee du payoff a terme
        fwd = S * np.exp(-q * T) - K * np.exp(-r * T)
        return np.maximum(fwd, 0.0) if option == "call" else np.maximum(-fwd, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option: str = "call", q: float = 0.0) -> dict:
    """Delta, gamma, vega, theta, rho (formules fermees)."""
    check_option(option)
    S = np.asarray(S, dtype=float)
    _check_inputs(S, K, T, sigma)
    if T == 0 or sigma == 0:
        raise ValueError("Les grecques exigent T > 0 et sigma > 0.")

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf = norm.pdf(d1)
    disc_q, disc_r = np.exp(-q * T), np.exp(-r * T)
    sqrt_t = np.sqrt(T)

    gamma = disc_q * pdf / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf * sqrt_t
    common_theta = -S * disc_q * pdf * sigma / (2 * sqrt_t)

    if option == "call":
        delta = disc_q * norm.cdf(d1)
        theta = common_theta - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = common_theta + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
        rho = -K * T * disc_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_volatility(price: float, S: float, K: float, T: float, r: float,
                       option: str = "call", q: float = 0.0) -> float:
    """Volatilite implicite par recherche de racine (Brent)."""
    check_option(option)
    if T <= 0:
        raise ValueError("La volatilite implicite exige T > 0.")
    disc_s, disc_k = S * np.exp(-q * T), K * np.exp(-r * T)
    lower = max(disc_s - disc_k, 0.0) if option == "call" else max(disc_k - disc_s, 0.0)
    upper = disc_s if option == "call" else disc_k
    if not lower < price < upper:
        raise ValueError(f"Prix hors bornes d'arbitrage ({lower:.4f}, {upper:.4f}) : pas de vol implicite.")
    return float(brentq(lambda s: bs_price(S, K, T, r, s, option, q) - price, 1e-6, 10.0, xtol=1e-12))
