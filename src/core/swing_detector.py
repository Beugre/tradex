"""
Détection des swing highs et swing lows sur des bougies OHLC.

Un swing high est une bougie dont le high est supérieur aux highs des N bougies
de chaque côté (lookback). Idem inversé pour un swing low.

Ce module est de la logique pure — aucun appel réseau.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.models import Candle, SwingLevel, SwingPoint

logger = logging.getLogger(__name__)


def detect_swings(candles: list[Candle], lookback: int = 3) -> list[SwingPoint]:
    """
    Détecte les swing highs et swing lows dans une série de bougies.

    Un swing high à l'index i est confirmé si :
        candles[i].high > candles[j].high pour tout j dans [i-lookback, i+lookback] (j != i)

    Un swing low à l'index i est confirmé si :
        candles[i].low < candles[j].low pour tout j dans [i-lookback, i+lookback] (j != i)

    Args:
        candles: Liste de bougies OHLCV triées par timestamp croissant.
        lookback: Nombre de bougies de chaque côté pour la confirmation.

    Returns:
        Liste de SwingPoint triés par index croissant.
    """
    if len(candles) < (2 * lookback + 1):
        logger.warning(
            "Pas assez de bougies (%d) pour lookback=%d. "
            "Minimum requis: %d",
            len(candles),
            lookback,
            2 * lookback + 1,
        )
        return []

    swings: list[SwingPoint] = []

    for i in range(lookback, len(candles) - lookback):
        swing_high = _is_swing_high(candles, i, lookback)
        swing_low = _is_swing_low(candles, i, lookback)

        if swing_high:
            point = SwingPoint(
                index=i,
                price=candles[i].high,
                level=SwingLevel.HIGH,
                timestamp=candles[i].timestamp,
            )
            swings.append(point)
            logger.debug(
                "Swing HIGH détecté à index=%d, prix=%.2f, ts=%d",
                i,
                point.price,
                point.timestamp,
            )

        if swing_low:
            point = SwingPoint(
                index=i,
                price=candles[i].low,
                level=SwingLevel.LOW,
                timestamp=candles[i].timestamp,
            )
            swings.append(point)
            logger.debug(
                "Swing LOW détecté à index=%d, prix=%.2f, ts=%d",
                i,
                point.price,
                point.timestamp,
            )

    logger.info("Détection terminée : %d swings trouvés sur %d bougies", len(swings), len(candles))
    return swings


def _is_swing_high(candles: list[Candle], index: int, lookback: int) -> bool:
    """Vérifie si la bougie à index est un swing high."""
    pivot_high = candles[index].high
    for offset in range(1, lookback + 1):
        if candles[index - offset].high >= pivot_high:
            return False
        if candles[index + offset].high >= pivot_high:
            return False
    return True


def _is_swing_low(candles: list[Candle], index: int, lookback: int) -> bool:
    """Vérifie si la bougie à index est un swing low."""
    pivot_low = candles[index].low
    for offset in range(1, lookback + 1):
        if candles[index - offset].low <= pivot_low:
            return False
        if candles[index + offset].low <= pivot_low:
            return False
    return True


def get_latest_swings(
    swings: list[SwingPoint],
    count: int = 4,
) -> list[SwingPoint]:
    """
    Retourne les N derniers swings détectés (utile pour le trend_engine).

    Args:
        swings: Liste complète des swings.
        count: Nombre de swings à retourner.

    Returns:
        Les `count` derniers swings, triés par index croissant.
    """
    return swings[-count:] if len(swings) >= count else swings
