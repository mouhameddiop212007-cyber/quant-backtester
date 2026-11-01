"""Pricing Monte Carlo sous mouvement brownien geometrique (variables antithetiques)."""
from __future__ import annotations

import numpy as np

from .black_scholes import check_option


def _pair_stats(payoff_a: np.ndarray, payoff_b: np.ndarray, discount: float) -> tuple[float, float]:
    """Moyenne et erreur standard sur les paires antithetiques (independantes entre elles)."""
    pair = discount * 0.5 * (payoff_a + payoff_b)
    return float(pair.mean()), float(pair.std(ddof=1) / np.sqrt(len(pair)))


def mc_price(S: float, K: float, T: float, r: float, sigma: float, option: str = "call",
             n_paths: int = 100_000, q: float = 0.0, seed: int | None = 0) -> tuple[float, float]:
    """Option europeenne : (prix, erreur standard)."""
    check_option(option)
    sign = 1.0 if option == "call" else -1.0
    z = np.random.default_rng(seed).standard_normal(max(n_paths // 2, 2))
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)
    pay_a = np.maximum(sign * (S * np.exp(drift + diffusion * z) - K), 0.0)
    pay_b = np.maximum(sign * (S * np.exp(drift - diffusion * z) - K), 0.0)
    return _pair_stats(pay_a, pay_b, np.exp(-r * T))


def mc_asian_price(S: float, K: float, T: float, r: float, sigma: float, option: str = "call",
                   n_paths: int = 50_000, n_steps: int = 252, q: float = 0.0,
                   seed: int | None = 0) -> tuple[float, float]:
    """Option asiatique a moyenne arithmetique discrete : pas de formule fermee, d'ou le Monte Carlo."""
    check_option(option)
    sign = 1.0 if option == "call" else -1.0
    dt = T / n_steps
    z = np.random.default_rng(seed).standard_normal((max(n_paths // 2, 2), n_steps))

    def average_payoff(shocks: np.ndarray) -> np.ndarray:
        log_path = np.cumsum((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks, axis=1)
        avg = (S * np.exp(log_path)).mean(axis=1)
        return np.maximum(sign * (avg - K), 0.0)

    return _pair_stats(average_payoff(z), average_payoff(-z), np.exp(-r * T))
