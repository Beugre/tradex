"""
Fournisseur de données de marché pour Binance.

Couche d'abstraction au-dessus de BinanceClient pour fournir
des bougies H4 et des tickers prêts à l'emploi pour le core.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.models import Balance, Candle, TickerData
from src.exchange.binance_client import BinanceClient

logger = logging.getLogger(__name__)

# Intervalle H4 en minutes (l'API Binance attend un string mais on convertit)
H4_INTERVAL_MINUTES = 240  # 4 heures


class BinanceDataProvider:
    """Fournit les données de marché depuis l'API Binance."""

    def __init__(self, client: BinanceClient) -> None:
        self._client = client
        self._last_candle_timestamps: dict[str, int] = {}

    def get_h4_candles(
        self,
        symbol: str,
        since: Optional[int] = None,
        limit: int = 200,
    ) -> list[Candle]:
        """
        Récupère les bougies H4 pour un symbole.

        Args:
            symbol: Paire (ex: "BTCUSDC").
            since: Timestamp Unix ms à partir duquel récupérer.
            limit: Nombre max de bougies à récupérer (défaut: 200).

        Returns:
            Liste de Candle triées par timestamp croissant.
        """
        candles = self._client.get_candles(
            symbol=symbol,
            interval=H4_INTERVAL_MINUTES,
            since=since,
            limit=limit,
        )
        # Trier par timestamp croissant
        candles.sort(key=lambda c: c.timestamp)
        logger.debug(
            "[%s] %d bougies H4 récupérées (limit=%d, depuis ts=%s)",
            symbol, len(candles), limit, since,
        )
        return candles

    def has_new_candle(self, symbol: str, candles: list[Candle]) -> bool:
        """
        Vérifie si une nouvelle bougie H4 est apparue depuis le dernier check.

        Args:
            symbol: Paire.
            candles: Liste de bougies les plus récentes.

        Returns:
            True si le timestamp de la dernière bougie a changé.
        """
        if not candles:
            return False

        latest_ts = candles[-1].timestamp
        prev_ts = self._last_candle_timestamps.get(symbol)

        if prev_ts is None or latest_ts > prev_ts:
            self._last_candle_timestamps[symbol] = latest_ts
            if prev_ts is not None:
                logger.info(
                    "[%s] 🕐 Nouvelle bougie H4 détectée (ts=%d)",
                    symbol,
                    latest_ts,
                )
            return prev_ts is not None  # False au premier appel
        return False

    def get_tickers(
        self, symbols: Optional[list[str]] = None
    ) -> list[TickerData]:
        """Récupère les tickers temps réel."""
        return self._client.get_tickers(symbols)

    def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Récupère le ticker pour un seul symbole."""
        tickers = self.get_tickers([symbol])
        for t in tickers:
            if t.symbol == symbol:
                return t
        logger.warning("[%s] Ticker non trouvé", symbol)
        return None

    def get_balances(self) -> list[Balance]:
        """Récupère les soldes du compte."""
        return self._client.get_balances()
