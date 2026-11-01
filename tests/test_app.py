"""Test de fumee du dashboard Streamlit (donnees synthetiques, sans reseau)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def fake_universe(tickers, start="2010-01-01", end=None, **kwargs):
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rng = np.random.default_rng(1)
    data = {t: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(idx))) for t in tickers}
    return pd.DataFrame(data, index=idx)


def test_dashboard_runs_without_error(monkeypatch):
    monkeypatch.setattr("quant_backtester.data.load_universe", fake_universe)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert len(at.dataframe) >= 2  # tableaux de metriques (portefeuille + strategie)


def test_dashboard_single_ticker_shows_info(monkeypatch):
    monkeypatch.setattr("quant_backtester.data.load_universe", fake_universe)
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    at.text_input[0].set_value("SPY").run()
    assert not at.exception
