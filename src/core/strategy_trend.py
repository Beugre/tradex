"""
Gestion des ordres : décision d'entrée, de sortie, et construction des ordres.

Détermine quand placer un ordre limit (simulation de stop order),
quand couper une position (SL atteint), et quand ajuster le SL (zero-risk).

Ce module est de la logique pure — aucun appel réseau.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.models import (
    OrderRequest,
    OrderSide,
    Position,
    PositionStatus,
    TrendDirection,
    TrendState,
    TickerData,
)
from src.core.risk_manager import (
    calculate_position_size,
    calculate_zero_risk_sl,
    should_apply_zero_risk,
    update_trailing_stop,
)


def _fmt_price(price: float) -> str:
    """Formate un prix avec assez de décimales (min 4, max 10).
    
    Gère les prix très petits (PEPE ≈ 0.000007, SHIB ≈ 0.00001)
    sans les tronquer à 0.00, et garantit min 4 décimales pour
    éviter les écarts d'arrondi sur les ordres API.
    """
    if price <= 0:
        return "0"
    if price >= 1.0:
        return f"{price:.4f}"
    # Trouver le nombre de décimales nécessaires (au moins 2 chiffres significatifs)
    decimals = 4
    temp = price
    while temp < 0.01 and decimals < 10:
        temp *= 10
        decimals += 1
    return f"{price:.{decimals}f}"

logger = logging.getLogger(__name__)


def check_entry_signal(
    trend: TrendState,
    ticker: TickerData,
    entry_buffer_pct: float,
) -> Optional[dict]:
    """
    Vérifie si le prix actuel atteint le seuil d'entrée pour un stop simulé.

    Règles :
    - BULLISH : prix >= dernier HH * (1 + buffer%) → signal BUY
    - BEARISH : prix <= dernier LL * (1 - buffer%) → signal SELL

    Args:
        trend: État de tendance actuel.
        ticker: Données de prix temps réel.
        entry_buffer_pct: Buffer en pourcentage (ex: 0.002 = 0.2%).

    Returns:
        Dict avec 'side', 'entry_price', 'sl_price' si signal détecté, None sinon.
    """
    if trend.direction == TrendDirection.NEUTRAL:
        return None

    current_price = ticker.last_price

    if trend.direction == TrendDirection.BULLISH:
        if trend.entry_level is None or trend.sl_level is None:
            return None
        entry_buffer = trend.entry_level * entry_buffer_pct
        sl_buffer = trend.sl_level * entry_buffer_pct
        entry_threshold = trend.entry_level + entry_buffer
        if current_price >= entry_threshold:
            logger.debug(
                "[%s] 🟢 Signal BUY : prix %.4f >= HH %.4f + %.2f%% (seuil %.4f)",
                trend.symbol,
                current_price,
                trend.entry_level,
                entry_buffer_pct * 100,
                entry_threshold,
            )
            return {
                "side": OrderSide.BUY,
                "entry_price": current_price,
                "sl_price": trend.sl_level - sl_buffer,
            }

    elif trend.direction == TrendDirection.BEARISH:
        if trend.entry_level is None or trend.sl_level is None:
            return None
        entry_buffer = trend.entry_level * entry_buffer_pct
        sl_buffer = trend.sl_level * entry_buffer_pct
        entry_threshold = trend.entry_level - entry_buffer
        if current_price <= entry_threshold:
            logger.debug(
                "[%s] 🔴 Signal SELL : prix %.4f <= LL %.4f - %.2f%% (seuil %.4f)",
                trend.symbol,
                current_price,
                trend.entry_level,
                entry_buffer_pct * 100,
                entry_threshold,
            )
            return {
                "side": OrderSide.SELL,
                "entry_price": current_price,
                "sl_price": trend.sl_level + sl_buffer,
            }

    return None


def check_sl_hit(
    position: Position,
    ticker: TickerData,
    sl_buffer_pct: float,
) -> bool:
    """
    Vérifie si le stop loss est atteint pour une position ouverte.

    Args:
        position: Position ouverte.
        ticker: Données de prix temps réel.
        sl_buffer_pct: Marge en pourcentage au-delà du SL (ex: 0.003 = 0.3%).

    Returns:
        True si le SL est atteint et qu'il faut couper.
    """
    current_price = ticker.last_price
    effective_sl = position.zero_risk_sl if position.is_zero_risk_applied else position.sl_price
    buffer = effective_sl * sl_buffer_pct

    if position.side == OrderSide.BUY:
        # Long : SL touché si prix descend sous le SL
        hit = current_price <= effective_sl - buffer
    else:
        # Short : SL touché si prix monte au-dessus du SL
        hit = current_price >= effective_sl + buffer

    if hit:
        logger.warning(
            "[%s] 🛑 SL atteint : prix=%.4f, SL=%.4f (buffer=%.2f%%)",
            position.symbol,
            current_price,
            effective_sl,
            sl_buffer_pct * 100,
        )

    return hit


def build_entry_order(
    symbol: str,
    side: OrderSide,
    entry_price: float,
    position_size: float,
) -> OrderRequest:
    """
    Construit un OrderRequest pour l'entrée en position.

    Args:
        symbol: Paire (ex: "BTC-USD").
        side: BUY ou SELL.
        entry_price: Prix d'entrée.
        position_size: Taille en unités de base.

    Returns:
        OrderRequest prêt à être envoyé à l'API.
    """
    order = OrderRequest(
        symbol=symbol,
        side=side,
        base_size=f"{position_size:.8f}",
        price=_fmt_price(entry_price),
    )
    logger.info(
        "[%s] Ordre %s préparé : size=%s @ %s",
        symbol,
        side.value.upper(),
        order.base_size,
        order.price,
    )
    return order


def build_exit_order(
    position: Position,
    current_price: float,
) -> OrderRequest:
    """
    Construit un OrderRequest pour fermer une position (SL ou take profit).

    L'ordre est dans le sens opposé à la position.

    Args:
        position: Position à fermer.
        current_price: Prix actuel pour le limit.

    Returns:
        OrderRequest pour couper la position.
    """
    exit_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY

    order = OrderRequest(
        symbol=position.symbol,
        side=exit_side,
        base_size=f"{position.size:.8f}",
        price=_fmt_price(current_price),
    )
    logger.info(
        "[%s] Ordre de sortie %s préparé : size=%s @ %s",
        position.symbol,
        exit_side.value.upper(),
        order.base_size,
        order.price,
    )
    return order


def process_zero_risk(
    position: Position,
    current_price: float,
    trigger_percent: float,
    lock_percent: float,
) -> Optional[float]:
    """
    Vérifie et applique la logique zero-risk sur une position.

    Args:
        position: Position ouverte.
        current_price: Prix actuel du marché.
        trigger_percent: Seuil de déclenchement (ex: 0.02 = 2%).
        lock_percent: Pourcentage de profit à verrouiller (ex: 0.005 = 0.5%).

    Returns:
        Nouveau prix de SL si zero-risk appliqué, None sinon.
    """
    if not should_apply_zero_risk(position, current_price, trigger_percent):
        return None

    new_sl = calculate_zero_risk_sl(position, lock_percent)
    position.zero_risk_sl = new_sl
    position.is_zero_risk_applied = True
    position.status = PositionStatus.ZERO_RISK

    logger.info(
        "[%s] ✅ Zero-risk activé : nouveau SL=%.2f (lock %.1f%%)",
        position.symbol,
        new_sl,
        lock_percent * 100,
    )

    return new_sl


def process_trailing_stop(
    position: Position,
    current_price: float,
    trailing_percent: float,
) -> Optional[float]:
    """
    Met à jour le trailing stop sur une position en zero-risk.

    Args:
        position: Position avec zero-risk activé.
        current_price: Prix actuel du marché.
        trailing_percent: Distance du trailing en % (ex: 0.02 = 2%).

    Returns:
        Nouveau prix de SL si mis à jour, None sinon.
    """
    return update_trailing_stop(position, current_price, trailing_percent)
