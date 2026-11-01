"""Compare Buy & Hold, SMA, RSI et Momentum sur SPY."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from quant_backtester.backtest import run_backtest
from quant_backtester.data import load_prices
from quant_backtester.metrics import drawdown_series, summary
from quant_backtester.strategies import BuyAndHold, Momentum, RSIMeanReversion, SMACrossover

prices = load_prices("SPY", start="2010-01-01")
strategies = [BuyAndHold(), SMACrossover(50, 200), RSIMeanReversion(14, 30, 70), Momentum(126)]
results = [run_backtest(prices, s) for s in strategies]

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
fig.savefig("results/compare_strategies.png", dpi=150)
print("Graphique enregistre dans results/compare_strategies.png")

for r in results:
    print(f"{r.strategy_name}: investi {r.positions.mean():.0%} du temps, turnover {r.turnover.sum():.1f}")
