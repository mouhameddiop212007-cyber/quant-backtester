"""Classe de base des strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Une strategie transforme une serie de prix en positions cibles.

    Contrat important : le signal calcule a la date ``t`` ne doit utiliser que
    des informations disponibles a la cloture de ``t``. C'est le moteur de
    backtest qui decale ensuite le signal d'un jour (execution a t+1), ce qui
    evite le look-ahead bias.
    """

    name: str = "Strategy"

    @abstractmethod
    def generate_signals(self, prices: pd.Series) -> pd.Series:
        """Retourne la position cible (0 = cash, 1 = investi) pour chaque date."""
