"""Pricing d'options : Black-Scholes, arbre binomial, Monte Carlo, grecques."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quant_backtester.options import (
    bs_greeks, bs_price, crr_price, implied_volatility, mc_asian_price, mc_price,
)

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
Path("results").mkdir(exist_ok=True)

print("=== Prix d'une option at-the-money (S=K=100, T=1 an, r=5 %, vol=20 %) ===")
rows = {}
for option in ("call", "put"):
    bs = float(bs_price(S, K, T, r, sigma, option))
    mc, se = mc_price(S, K, T, r, sigma, option, n_paths=200_000)
    rows[option] = {
        "Black-Scholes": bs,
        "Binomial (500 pas)": crr_price(S, K, T, r, sigma, option, steps=500),
        "Monte Carlo": mc,
        "MC erreur std": se,
        "Americaine (500 pas)": crr_price(S, K, T, r, sigma, option, steps=500, american=True),
    }
print(pd.DataFrame(rows).T.round(4).to_string())

asian, asian_se = mc_asian_price(S, K, T, r, sigma, "call", n_paths=50_000, n_steps=252)
print(f"\nCall asiatique (moyenne arithmetique) : {asian:.4f} +/- {asian_se:.4f}"
      f"  (vs europeen {float(bs_price(S, K, T, r, sigma, 'call')):.4f})")

print("\n=== Grecques du call ===")
g = bs_greeks(S, K, T, r, sigma, "call")
print(pd.Series({"delta": g["delta"], "gamma": g["gamma"], "vega (pour +1 pt de vol)": g["vega"] / 100,
                 "theta (par jour)": g["theta"] / 365, "rho (pour +1 pt de taux)": g["rho"] / 100}).round(4).to_string())

market_price = 12.0
print(f"\nVolatilite implicite d'un call cote a {market_price} : "
      f"{implied_volatility(market_price, S, K, T, r, 'call'):.2%}")

# --- Graphique 1 : grecques en fonction du spot
spots = np.linspace(60, 140, 200)
greeks = bs_greeks(spots, K, T, r, sigma, "call")
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, (name, values) in zip(axes.ravel(), [("delta", greeks["delta"]), ("gamma", greeks["gamma"]),
                                              ("vega", greeks["vega"]), ("theta", greeks["theta"])]):
    ax.plot(spots, values)
    ax.axvline(K, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"{name} du call (K={K:g})")
    ax.set_xlabel("Spot")
fig.tight_layout()
fig.savefig("results/options_greeks.png", dpi=150)

# --- Graphique 2 : convergence de l'arbre et du Monte Carlo vers Black-Scholes
exact = float(bs_price(S, K, T, r, sigma, "call"))
steps = np.arange(10, 501, 5)
tree = [crr_price(S, K, T, r, sigma, "call", steps=int(n)) for n in steps]
paths = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000])
mc = [mc_price(S, K, T, r, sigma, "call", n_paths=int(n)) for n in paths]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(steps, tree, label="Arbre CRR")
ax1.axhline(exact, color="black", linestyle="--", label="Black-Scholes")
ax1.set_xlabel("Nombre de pas")
ax1.set_title("Convergence de l'arbre binomial")
ax1.legend()
ax2.errorbar(paths, [m[0] for m in mc], yerr=[1.96 * m[1] for m in mc], fmt="o-", capsize=3, label="Monte Carlo (IC 95 %)")
ax2.axhline(exact, color="black", linestyle="--", label="Black-Scholes")
ax2.set_xscale("log")
ax2.set_xlabel("Nombre de trajectoires")
ax2.set_title("Convergence du Monte Carlo")
ax2.legend()
fig.tight_layout()
fig.savefig("results/options_convergence.png", dpi=150)
print("\nGraphiques enregistres dans results/options_greeks.png et results/options_convergence.png")
