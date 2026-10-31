"""Chargement des prix de marche, avec cache local en CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path("data/raw")


def load_prices(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.Series:
    """Retourne les prix de cloture ajustes (dividendes et splits) d'un ticker.

    Les donnees sont telechargees via yfinance puis mises en cache dans
    ``cache_dir`` pour eviter de retelecharger a chaque execution.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker.replace('^', '')}_{start}_{end or 'latest'}.csv"

    if cache_file.exists() and not refresh:
        series = pd.read_csv(cache_file, index_col=0, parse_dates=True).iloc[:, 0]
        series.name = ticker
        return series

    import yfinance as yf  # import tardif : inutile pour les tests

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"Aucune donnee recue pour '{ticker}'.")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance recent : colonnes multi-niveaux
        close = close.iloc[:, 0]
    close = close.dropna().rename(ticker)
    close.to_csv(cache_file)
    return close
