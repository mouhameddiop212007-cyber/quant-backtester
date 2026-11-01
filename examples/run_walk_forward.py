"""Validation hors-echantillon : le momentum optimise est-il robuste ?"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_backtester.backtest import (
    best_params, evaluate, grid_search, run_backtest, walk_forward,
)
from quant_backtester.data import load_prices
from quant_backtester.metrics import sharpe_ratio, summary
from quant_backtester.strategies import BuyAndHold, Momentum

prices = load_prices("SPY", start="2010-01-01")
GRID = {"lookback": [20, 40, 60, 90, 126, 180, 250]}
SPLIT = "2019-01-01"

# 1) Un seul decoupage : optimisation avant 2019, test apres
print("=== Optimisation sur la periode d'entrainement (avant", SPLIT, ") ===")
train_table = grid_search(prices, Momentum, GRID, end=SPLIT)
print(train_table.round(3).to_string())

best = best_params(train_table)
test_returns = evaluate(prices, Momentum(**best), start=SPLIT)
bh_test = evaluate(prices, BuyAndHold(), start=SPLIT)
print(f"\nMeilleur parametre en entrainement : {best}")
print(f"Sharpe entrainement : {train_table.loc[0, 'score']:.2f}")
print(f"Sharpe test         : {sharpe_ratio(test_returns):.2f}")
print(f"Sharpe Buy & Hold (test) : {sharpe_ratio(bh_test):.2f}")

# 2) Walk-forward glissant : 5 ans d'entrainement, 1 an de test
print("\n=== Walk-forward glissant (5 ans / 1 an) ===")
wf = walk_forward(prices, Momentum, GRID, train_years=5, test_years=1)
print(wf.folds.to_string(index=False))

oos = wf.returns
bh = run_backtest(prices, BuyAndHold()).returns.loc[oos.index]
table = pd.DataFrame({"Momentum walk-forward": summary(oos), "Buy & Hold": summary(bh)}).T
print("\n=== Performance hors-echantillon uniquement ===")
print(table.round(3).to_string())

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot((1 + oos).cumprod(), label="Momentum walk-forward (hors-echantillon)")
ax.plot((1 + bh).cumprod(), label="Buy & Hold")
ax.set_title("Performance hors-echantillon")
ax.legend()
Path("results").mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig("results/walk_forward.png", dpi=150)
print("Graphique enregistre dans results/walk_forward.png")
