import numpy as np
import pandas as pd

from .base import Strategy


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder : 100 - 100 / (1 + gain moyen / perte moyenne)."""
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


class RSIMeanReversion(Strategy):
    """Achete en survente (RSI < oversold), vend en surachat (RSI > overbought).

    La position est conservee entre les deux seuils (hysteresis).
    """

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        if not 0 < oversold < overbought < 100:
            raise ValueError("Il faut 0 < oversold < overbought < 100.")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"RSI {period} ({oversold:g}/{overbought:g})"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        rsi = compute_rsi(prices, self.period)
        position = 0.0
        out = []
        for value in rsi.to_numpy():
            if np.isnan(value):
                out.append(0.0)  # periode de chauffe
                continue
            if value < self.oversold:
                position = 1.0
            elif value > self.overbought:
                position = 0.0
            out.append(position)
        return pd.Series(out, index=prices.index)
