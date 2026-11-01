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

## Résultats (SPY, 2010 à aujourd'hui, frais 5 bps + slippage 2 bps)

![SMA vs Buy & Hold](results/sma_vs_buyhold.png)

| | Rendement annualisé | Volatilité | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & Hold | 13,9 % | 17,0 % | 0,85 | -33,7 % |
| SMA 50/200 | 9,1 % | 13,9 % | 0,70 | -33,7 % |

La stratégie SMA réduit la volatilité mais sous-performe le Buy & Hold sur cette période haussière, et elle ne protège pas du krach de mars 2020, trop rapide pour des moyennes mobiles de cette longueur.

## Comparaison de stratégies (SPY, 2010 à aujourd'hui)

![Comparaison](results/compare_strategies.png)

| Stratégie | Rendement | Volatilité | Sharpe | Max drawdown |
|---|---|---|---|---|
| Buy & Hold | 13,9 % | 17,0 % | 0,85 | -33,7 % |
| SMA 50/200 | 9,1 % | 13,9 % | 0,70 | -33,7 % |
| RSI 14 (30/70) | 7,6 % | 13,0 % | 0,63 | -28,4 % |
| Momentum 126j | 8,4 % | 11,6 % | 0,75 | -23,8 % |

**Lecture** : aucune stratégie ne bat le Buy & Hold en rendement ajusté du risque sur cette période haussière. Le momentum offre le meilleur compromis : drawdown réduit de 10 points pour environ 5 points de rendement en moins.

**Limites** : le cash ne rapporte rien dans cette version, un seul actif et une seule période sont testés, les résultats passés ne préjugent pas des résultats futurs.



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
