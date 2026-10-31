import pandas as pd

from .base import Strategy


class SMACrossover(Strategy):
    """Investi quand la moyenne mobile courte est au-dessus de la longue, sinon cash."""

    def __init__(self, fast: int = 50, slow: int = 200):
        if fast >= slow:
            raise ValueError("fast doit etre strictement inferieur a slow.")
        self.fast = fast
        self.slow = slow
        self.name = f"SMA {fast}/{slow}"

    def generate_signals(self, prices: pd.Series) -> pd.Series:
        fast_ma = prices.rolling(self.fast).mean()
        slow_ma = prices.rolling(self.slow).mean()
        signal = (fast_ma > slow_ma).astype(float)
        signal[slow_ma.isna()] = 0.0  # periode de chauffe : pas de position
        return signal
