import pandas as pd

from .base import Strategy


class BuyAndHold(Strategy):
    """Benchmark : toujours investi."""

    name = "Buy & Hold"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        return pd.Series(1.0, index=prices.index)
