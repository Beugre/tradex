"""
Classification de la tendance selon la Dow Theory.

Analyse les swings détectés (highs et lows) pour déterminer :
- HH/HL → BULLISH
- LH/LL → BEARISH
- Invalidation → NEUTRAL

Ce module est de la logique pure — aucun appel réseau.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.models import (
    SwingLevel,
    SwingPoint,
    SwingType,
    TrendDirection,
    TrendState,
)

logger = logging.getLogger(__name__)


def classify_swings(swings: list[SwingPoint]) -> list[SwingPoint]:
    """
    Classifie chaque swing en HH/HL/LH/LL par comparaison avec le
    swing précédent de même type (HIGH vs HIGH, LOW vs LOW).

    Args:
        swings: Liste de SwingPoint triés par index croissant.

    Returns:
        La même liste avec `swing_type` renseigné sur chaque point.
    """
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None

    for swing in swings:
        if swing.level == SwingLevel.HIGH:
            if last_high is not None:
                if swing.price > last_high.price:
                    swing.swing_type = SwingType.HH
                else:
                    swing.swing_type = SwingType.LH
            last_high = swing

        elif swing.level == SwingLevel.LOW:
            if last_low is not None:
                if swing.price > last_low.price:
                    swing.swing_type = SwingType.HL
                else:
                    swing.swing_type = SwingType.LL
            last_low = swing

    return swings


def determine_trend(swings: list[SwingPoint], symbol: str) -> TrendState:
    """
    Détermine l'état de la tendance à partir des swings classifiés.

    Règles Dow Theory :
    - BULLISH confirmé quand on a un HH suivi d'un HL (ou vice-versa).
    - BEARISH confirmé quand on a un LH suivi d'un LL (ou vice-versa).
    - Sinon NEUTRAL.

    Args:
        swings: Liste de SwingPoint avec swing_type classifié.
        symbol: Symbole de la paire (ex: "BTC-USD").

    Returns:
        TrendState avec direction, derniers swings, et historique.
    """
    classified = classify_swings(swings)

    # Trouver les deux derniers highs et les deux derniers lows
    highs = [s for s in classified if s.level == SwingLevel.HIGH]
    lows = [s for s in classified if s.level == SwingLevel.LOW]

    last_high = highs[-1] if highs else None
    prev_high = highs[-2] if len(highs) >= 2 else None
    last_low = lows[-1] if lows else None
    prev_low = lows[-2] if len(lows) >= 2 else None

    direction, neutral_reason = _compute_direction(last_high, last_low)

    state = TrendState(
        symbol=symbol,
        direction=direction,
        last_high=last_high,
        last_low=last_low,
        prev_high=prev_high,
        prev_low=prev_low,
        swings=classified,
        neutral_reason=neutral_reason,
    )

    logger.info(
        "[%s] Tendance: %s | Dernier high: %s (%.2f) | Dernier low: %s (%.2f)",
        symbol,
        direction.value,
        last_high.swing_type.value if last_high and last_high.swing_type else "?",
        last_high.price if last_high else 0,
        last_low.swing_type.value if last_low and last_low.swing_type else "?",
        last_low.price if last_low else 0,
    )

    return state


def check_trend_invalidation(
    trend: TrendState,
    current_price: float,
) -> TrendState:
    """
    Vérifie si le prix actuel invalide la tendance en cours.

    Invalidation :
    - BEARISH invalidé si prix > dernier LH → passer en NEUTRAL
    - BULLISH invalidé si prix < dernier HL → passer en NEUTRAL

    Args:
        trend: État de tendance actuel.
        current_price: Prix courant du marché.

    Returns:
        TrendState mis à jour (possiblement passé en NEUTRAL).
    """
    if trend.direction == TrendDirection.BEARISH and trend.last_high:
        # En downtrend, si le prix casse le dernier LH → invalidation
        if (
            trend.last_high.swing_type == SwingType.LH
            and current_price > trend.last_high.price
        ):
            logger.warning(
                "[%s] ⚠️ Tendance BEARISH invalidée : prix %.2f > LH %.2f",
                trend.symbol,
                current_price,
                trend.last_high.price,
            )
            trend.direction = TrendDirection.NEUTRAL
            trend.neutral_reason = f"invalidation BEARISH — prix {current_price:.2f} > LH {trend.last_high.price:.2f}"

    elif trend.direction == TrendDirection.BULLISH and trend.last_low:
        # En uptrend, si le prix casse le dernier HL → invalidation
        if (
            trend.last_low.swing_type == SwingType.HL
            and current_price < trend.last_low.price
        ):
            logger.warning(
                "[%s] ⚠️ Tendance BULLISH invalidée : prix %.2f < HL %.2f",
                trend.symbol,
                current_price,
                trend.last_low.price,
            )
            trend.direction = TrendDirection.NEUTRAL
            trend.neutral_reason = f"invalidation BULLISH — prix {current_price:.2f} < HL {trend.last_low.price:.2f}"

    return trend


def _compute_direction(
    last_high: Optional[SwingPoint],
    last_low: Optional[SwingPoint],
) -> tuple[TrendDirection, Optional[str]]:
    """
    Détermine la direction en fonction des derniers swing types.

    BULLISH = dernier high est HH ET dernier low est HL
    BEARISH = dernier high est LH ET dernier low est LL
    Sinon NEUTRAL (avec raison)

    Returns:
        Tuple (direction, neutral_reason). neutral_reason est None si non-NEUTRAL.
    """
    if last_high is None or last_low is None:
        return TrendDirection.NEUTRAL, "données insuffisantes (pas assez de swings classifiés)"

    high_type = last_high.swing_type
    low_type = last_low.swing_type

    if high_type is None or low_type is None:
        return TrendDirection.NEUTRAL, "swings non classifiés (premiers de la série)"

    if high_type == SwingType.HH and low_type == SwingType.HL:
        return TrendDirection.BULLISH, None
    elif high_type == SwingType.LH and low_type == SwingType.LL:
        return TrendDirection.BEARISH, None
    else:
        # Cas mixtes
        h = high_type.value
        l = low_type.value
        if high_type == SwingType.LH and low_type == SwingType.HL:
            return TrendDirection.NEUTRAL, f"compression ({h} + {l}) — highs ↓ mais lows ↑"
        elif high_type == SwingType.HH and low_type == SwingType.LL:
            return TrendDirection.NEUTRAL, f"expansion ({h} + {l}) — highs ↑ mais lows ↓"
        else:
            return TrendDirection.NEUTRAL, f"structure mixte ({h} + {l})"
