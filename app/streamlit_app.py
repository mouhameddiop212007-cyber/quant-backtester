"""Dashboard interactif : strategies, portefeuilles et frontiere efficiente.

Lancer avec :  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from quant_backtester.backtest import run_backtest
from quant_backtester.data import load_universe
from quant_backtester.metrics import drawdown_series, summary
from quant_backtester.portfolio import (
    efficient_frontier, equal_weight, historical_cvar, historical_var, make_weight_fn,
    max_sharpe, mean_cov, min_variance, portfolio_stats, risk_parity, run_portfolio_backtest,
)
from quant_backtester.strategies import BuyAndHold, Momentum, RSIMeanReversion, SMACrossover

st.set_page_config(page_title="Quant Backtester", layout="wide")

METHODS = {
    "equal_weight": "Équipondéré",
    "min_variance": "Variance minimale",
    "risk_parity": "Risk parity",
    "max_sharpe": "Max Sharpe",
}
PCT_FORMAT = {
    "Rendement annualise": "{:.1%}", "Volatilite annualisee": "{:.1%}", "Sharpe": "{:.2f}",
    "Max drawdown": "{:.1%}", "VaR 95% (jour)": "{:.2%}", "CVaR 95% (jour)": "{:.2%}",
}


@st.cache_data(show_spinner="Chargement des donnees...")
def get_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return load_universe(list(tickers), start=start)


@st.cache_data(show_spinner="Backtest en cours...")
def get_portfolio_result(prices: pd.DataFrame, method: str, lookback: int, rebalance: int, cost: float):
    return run_portfolio_backtest(prices, make_weight_fn(method), lookback, rebalance, cost)


def metrics_table(returns_by_name: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, r in returns_by_name.items():
        stats = summary(r)
        stats["VaR 95% (jour)"] = historical_var(r)
        stats["CVaR 95% (jour)"] = historical_cvar(r)
        rows[name] = stats
    return pd.DataFrame(rows).T


def show_metrics(returns_by_name: dict[str, pd.Series]) -> None:
    st.dataframe(metrics_table(returns_by_name).style.format(PCT_FORMAT))


def show_curves(returns_by_name: dict[str, pd.Series]) -> None:
    equity = pd.DataFrame({n: (1 + r).cumprod() for n, r in returns_by_name.items()})
    drawdown = pd.DataFrame({n: drawdown_series(r) for n, r in returns_by_name.items()})
    left, right = st.columns(2)
    left.subheader("Valeur du portefeuille (base 1)")
    left.line_chart(equity)
    right.subheader("Drawdown")
    right.line_chart(drawdown)


# ------------------------------------------------------------------ barre laterale
st.title("Quant Backtester")
st.caption("Projet pedagogique : les performances passees ne presagent pas des performances futures.")

with st.sidebar:
    st.header("Parametres")
    raw = st.text_input("Tickers (separes par des virgules)", "SPY, EFA, TLT, GLD")
    start = st.date_input("Debut", date(2010, 1, 1), min_value=date(1995, 1, 1))
    cost_bps = st.slider("Couts de transaction (bps)", 0.0, 50.0, 5.0, 0.5)
    lookback = st.slider("Fenetre d'estimation (jours)", 60, 504, 252, 21)
    rebalance = st.slider("Reequilibrage (jours)", 5, 126, 21, 1)

tickers = tuple(dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip()))
if not tickers:
    st.info("Saisis au moins un ticker dans la barre laterale.")
    st.stop()

try:
    prices = get_prices(tickers, str(start))
except Exception as exc:  # reseau, ticker inconnu...
    st.error(f"Impossible de charger les donnees : {exc}")
    st.stop()

st.sidebar.success(f"{len(prices)} jours : {prices.index[0].date()} -> {prices.index[-1].date()}")

tab_portfolio, tab_strategy, tab_frontier = st.tabs(
    ["Portefeuille", "Strategie sur un actif", "Frontiere efficiente"]
)

# ------------------------------------------------------------------ portefeuille
with tab_portfolio:
    if len(tickers) < 2:
        st.info("Il faut au moins 2 tickers pour construire un portefeuille.")
    else:
        chosen = st.multiselect(
            "Methodes d'allocation", list(METHODS), default=list(METHODS),
            format_func=METHODS.get,
        )
        if chosen:
            try:
                results = {m: get_portfolio_result(prices, m, lookback, rebalance, cost_bps) for m in chosen}
            except ValueError as exc:
                st.error(str(exc))
            else:
                returns = {METHODS[m]: r.returns for m, r in results.items()}
                show_metrics(returns)
                show_curves(returns)

                years = len(next(iter(returns.values()))) / 252
                turnover = pd.Series({METHODS[m]: r.turnover.sum() / years for m, r in results.items()},
                                     name="Turnover / an")
                st.subheader("Turnover annuel")
                st.bar_chart(turnover)

                st.subheader("Poids dans le temps")
                focus = st.selectbox("Methode", chosen, format_func=METHODS.get)
                st.area_chart(results[focus].weights)
                st.caption("Poids estimes sur le passe uniquement, appliques des le lendemain.")

# ------------------------------------------------------------------ strategie mono-actif
def build_strategy():
    kind = st.selectbox("Strategie", ["SMA", "RSI", "Momentum"])
    if kind == "SMA":
        fast = st.slider("Moyenne courte", 5, 100, 50)
        slow = st.slider("Moyenne longue", 50, 300, 200)
        if fast >= slow:
            st.warning("La moyenne courte doit etre inferieure a la longue.")
            return None
        return SMACrossover(fast, slow)
    if kind == "RSI":
        period = st.slider("Periode RSI", 5, 30, 14)
        oversold, overbought = st.slider("Seuils (survente / surachat)", 5, 95, (30, 70))
        return RSIMeanReversion(period, oversold, overbought)
    return Momentum(st.slider("Fenetre momentum (jours)", 20, 300, 126))


with tab_strategy:
    asset = st.selectbox("Actif", list(tickers))
    strategy = build_strategy()
    if strategy is not None:
        series = prices[asset]
        res = run_backtest(series, strategy, cost_bps=cost_bps, slippage_bps=0.0)
        bench = run_backtest(series, BuyAndHold(), cost_bps=cost_bps, slippage_bps=0.0)
        returns = {strategy.name: res.returns, "Buy & Hold": bench.returns}
        c1, c2 = st.columns(2)
        c1.metric("Temps investi", f"{res.positions.mean():.0%}")
        c2.metric("Turnover total", f"{res.turnover.sum():.1f}")
        show_metrics(returns)
        show_curves(returns)

# ------------------------------------------------------------------ frontiere efficiente
with tab_frontier:
    if len(tickers) < 2:
        st.info("Il faut au moins 2 tickers.")
    else:
        st.caption("Calculee sur tout l'echantillon : c'est une illustration in-sample, "
                   "pas une allocation que l'on aurait pu choisir a l'epoque.")
        rets = prices.pct_change().dropna()
        mu, cov = mean_cov(rets)
        frontier = efficient_frontier(mu, cov, n_points=40)
        portfolios = {
            "Équipondéré": equal_weight(len(tickers)),
            "Variance minimale": min_variance(cov),
            "Max Sharpe": max_sharpe(mu, cov),
            "Risk parity": risk_parity(cov),
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(frontier["volatility"], frontier["return"], label="Frontiere efficiente")
        for t in tickers:
            vol = cov.loc[t, t] ** 0.5
            ax.scatter(vol, mu[t], marker="x", color="gray")
            ax.annotate(t, (vol, mu[t]), textcoords="offset points", xytext=(5, 5))
        for name, w in portfolios.items():
            ret, vol, _ = portfolio_stats(w, mu, cov)
            ax.scatter(vol, ret, s=60, label=name)
        ax.set_xlabel("Volatilite annualisee")
        ax.set_ylabel("Rendement annualise")
        ax.legend()
        st.pyplot(fig)
        st.dataframe(pd.DataFrame(portfolios, index=tickers).T.style.format("{:.1%}"))
