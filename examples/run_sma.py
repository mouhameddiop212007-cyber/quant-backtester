"""Exemple : SMA 50/200 vs Buy & Hold sur le S&P 500 (ETF SPY)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_backtester.backtest import run_backtest
from quant_backtester.data import load_prices
from quant_backtester.metrics import drawdown_series, summary
from quant_backtester.strategies import BuyAndHold, SMACrossover

prices = load_prices("SPY", start="2010-01-01")

results = [run_backtest(prices, s) for s in (BuyAndHold(), SMACrossover(50, 200))]

table = pd.DataFrame({r.strategy_name: summary(r.returns) for r in results}).T
print(table.round(3))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for r in results:
    ax1.plot(r.equity, label=r.strategy_name)
    ax2.plot(drawdown_series(r.returns), label=r.strategy_name)
ax1.set_title("Valeur du portefeuille")
ax2.set_title("Drawdown")
ax1.legend()
Path("results").mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig("results/sma_vs_buyhold.png", dpi=150)
print("Graphique enregistre dans results/sma_vs_buyhold.png")
