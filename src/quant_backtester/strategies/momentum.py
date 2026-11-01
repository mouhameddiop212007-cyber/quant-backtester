import pandas as pd

from .base import Strategy


class Momentum(Strategy):
    """Investi si le rendement sur les ``lookback`` derniers jours est positif."""

    def __init__(self, lookback: int = 126):
        if lookback < 1:
            raise ValueError("lookback doit etre >= 1.")
        self.lookback = lookback
        self.name = f"Momentum {lookback}j"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        past_return = prices.pct_change(self.lookback)
        return (past_return > 0).astype(float)  # NaN > 0 est False : cash en chauffe
