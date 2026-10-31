# quant-backtester

Framework Python de backtesting de strategies de trading, concu pour apprendre
la finance de marche et le genie logiciel en parallele. Evolution prevue vers
l'optimisation de portefeuille (Markowitz, VaR/CVaR, Risk Parity).

## Fonctionnalites actuelles
- Chargement de donnees (yfinance) avec cache local
- Metriques : rendement annualise, volatilite, Sharpe, drawdown maximum
- Classe `Strategy` extensible (Buy & Hold, croisement de moyennes mobiles)
- Moteur de backtest avec **frais de transaction, slippage et execution a J+1**
  (pas de look-ahead bias)

## Installation
```bash
git clone <ton-repo> && cd quant-backtester
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python examples/run_sma.py
```

## Utilisation
```python
from quant_backtester.data import load_prices
from quant_backtester.backtest import run_backtest
from quant_backtester.strategies import SMACrossover
from quant_backtester.metrics import summary

prices = load_prices("SPY", start="2010-01-01")
res = run_backtest(prices, SMACrossover(50, 200))
print(summary(res.returns))
```

## Creer sa propre strategie
Herite de `Strategy` et implemente `generate_signals` (position cible 0 ou 1
calculee avec les informations disponibles a la cloture du jour).

## Limites et biais (a completer avec tes resultats)
- Surapprentissage : tester beaucoup de parametres sur les memes donnees
- Biais du survivant : les tickers actuels ont "survecu"
- Frais et slippage simplifies
- Resultats passes != resultats futurs

## Feuille de route
- [x] Phase 1 : donnees et metriques
- [x] Phase 2 : moteur de backtest (socle)
- [ ] Phase 2 : strategies RSI et momentum
- [ ] Phase 3 : walk-forward, graphiques d'analyse
- [ ] Phase 4 : optimisation de portefeuille (Markowitz, VaR/CVaR, Risk Parity)
- [ ] Phase 5 : dashboard Streamlit
