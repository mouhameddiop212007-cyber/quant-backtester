"""Optimisation de portefeuille : SPY (actions US), EFA (actions int.), TLT (obligations), GLD (or)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_backtester.data import load_universe
from quant_backtester.metrics import drawdown_series, summary
from quant_backtester.portfolio import (
    efficient_frontier, equal_weight, historical_cvar, historical_var, make_weight_fn,
    max_sharpe, mean_cov, min_variance, portfolio_stats, risk_parity, run_portfolio_backtest,
)

TICKERS = ["SPY", "EFA", "TLT", "GLD"]
prices = load_universe(TICKERS, start="2010-01-01")
returns = prices.pct_change().dropna()
print(f"Donnees communes : {prices.index[0].date()} -> {prices.index[-1].date()}\n")

print("=== Correlations des rendements journaliers ===")
print(returns.corr().round(2).to_string())

# 1) Frontiere efficiente sur TOUT l'echantillon (illustration : c'est du in-sample !)
mu, cov = mean_cov(returns)
frontier = efficient_frontier(mu, cov, n_points=40)
portfolios = {
    "Equipondere": equal_weight(len(TICKERS)),
    "Min variance": min_variance(cov),
    "Max Sharpe": max_sharpe(mu, cov),
    "Risk parity": risk_parity(cov),
}
print("\n=== Poids optimaux sur tout l'echantillon (in-sample) ===")
print(pd.DataFrame(portfolios, index=TICKERS).T.round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(frontier["volatility"], frontier["return"], label="Frontiere efficiente")
for t in TICKERS:
    ax.scatter(cov.loc[t, t] ** 0.5, mu[t], marker="x", color="gray")
    ax.annotate(t, (cov.loc[t, t] ** 0.5, mu[t]), textcoords="offset points", xytext=(5, 5))
for name, w in portfolios.items():
    ret, vol, _ = portfolio_stats(w, mu, cov)
    ax.scatter(vol, ret, s=60, label=name)
ax.set_xlabel("Volatilite annualisee")
ax.set_ylabel("Rendement annualise")
ax.set_title("Frontiere efficiente (in-sample)")
ax.legend()
Path("results").mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig("results/frontier.png", dpi=150)

# 2) Backtest realiste : poids estimes sur 252 jours, reequilibrage mensuel, couts inclus
methods = ["equal_weight", "min_variance", "risk_parity", "max_sharpe"]
results = {m: run_portfolio_backtest(prices, make_weight_fn(m), lookback=252, rebalance_days=21, cost_bps=5)
           for m in methods}

table = pd.DataFrame({m: summary(r.returns) for m, r in results.items()}).T
table["VaR 95% (jour)"] = [historical_var(r.returns) for r in results.values()]
table["CVaR 95% (jour)"] = [historical_cvar(r.returns) for r in results.values()]
years = len(results["equal_weight"].returns) / 252
table["Turnover/an"] = [r.turnover.sum() / years for r in results.values()]
print("\n=== Backtest hors-echantillon (poids estimes sur le passe uniquement) ===")
print(table.round(3).to_string())

print("\n=== Poids moyens dans le temps ===")
print(pd.DataFrame({m: r.weights.mean() for m, r in results.items()}).T.round(3).to_string())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for m, r in results.items():
    ax1.plot((1 + r.returns).cumprod(), label=m)
    ax2.plot(drawdown_series(r.returns), label=m)
ax1.set_title("Valeur du portefeuille (base 1)")
ax2.set_title("Drawdown")
ax1.legend()
fig.tight_layout()
fig.savefig("results/portfolio_comparison.png", dpi=150)
print("\nGraphiques enregistres dans results/frontier.png et results/portfolio_comparison.png")
