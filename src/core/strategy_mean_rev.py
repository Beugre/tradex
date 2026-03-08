"""
Stratégie Mean-Reversion (Range Trading).

Quand la tendance Dow est NEUTRAL, on identifie un range
(borne haute = dernier high, borne basse = dernier low) et
on trade le rebond sur les extrêmes vers le milieu.

Ce module est de la logique pure — aucun appel réseau.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.core.models import (
    OrderRequest,
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    TickerData,
    TrendDirection,
    TrendState,
)

logger = logging.getLogger(__name__)


# ── Construction du range ──────────────────────────────────────────────────────

def build_range_from_trend(
    trend: TrendState,
    min_width_pct: float,
) -> Optional[RangeState]:
    """
    Construit un RangeState à partir d'une tendance NEUTRAL.

    Le range est défini par le dernier swing high et le dernier swing low.
    Si la largeur du range est inférieure à `min_width_pct`, retourne None.

    Args:
        trend: État de tendance (doit être NEUTRAL).
        min_width_pct: Largeur minimale du range en pourcentage (ex: 0.02 = 2%).

    Returns:
        RangeState si le range est valide, None sinon.
    """
    if trend.direction != TrendDirection.NEUTRAL:
        return None

    if trend.last_high is None or trend.last_low is None:
        logger.debug("[%s] Range impossible : pas de swings high/low", trend.symbol)
        return None

    range_high = trend.last_high.price
    range_low = trend.last_low.price

    if range_high <= range_low:
        logger.debug(
            "[%s] Range invalide : high %.4f <= low %.4f",
            trend.symbol, range_high, range_low,
        )
        return None

    rs = RangeState(
        symbol=trend.symbol,
        range_high=range_high,
        range_low=range_low,
    )

    if rs.range_width_pct < min_width_pct:
        logger.info(
            "[%s] 🔄 Range trop étroit : %.2f%% < %.2f%% min (H=%.4f, L=%.4f)",
            trend.symbol,
            rs.range_width_pct * 100,
            min_width_pct * 100,
            range_high,
            range_low,
        )
        return None

    logger.info(
        "[%s] 🔄 Range détecté : H=%.4f | L=%.4f | Mid=%.4f | Largeur=%.2f%%",
        trend.symbol,
        rs.range_high,
        rs.range_low,
        rs.range_mid,
        rs.range_width_pct * 100,
    )

    return rs


# ── Signaux d'entrée ──────────────────────────────────────────────────────────

def check_range_entry_signal(
    range_state: RangeState,
    ticker: TickerData,
    entry_buffer_pct: float,
    tp_ratio: float = 0.75,
) -> Optional[dict]:
    """
    Vérifie si le prix est proche d'un extrême du range pour un signal.

    - Prix ≤ range_low * (1 + buffer) → BUY (rebond sur le bas)
    - Prix ≥ range_high * (1 - buffer) → SELL (rebond sur le haut)

    Args:
        range_state: Le range actif.
        ticker: Données de prix temps réel.
        entry_buffer_pct: Zone d'entrée autour de l'extrême (ex: 0.002 = 0.2%).
        tp_ratio: Ratio du TP dans le range (0.5 = mid, 0.75 = 3/4). Défaut: 0.75.

    Returns:
        Dict avec 'side', 'entry_price', 'sl_price', 'tp_price' si signal, None sinon.
    """
    if not range_state.is_valid:
        return None

    if is_in_cooldown(range_state):
        return None

    price = ticker.last_price
    range_width = range_state.range_high - range_state.range_low

    # ── BUY au bas du range ──
    buy_zone = range_state.range_low * (1 + entry_buffer_pct)
    if price <= buy_zone:
        sl_price = range_state.range_low * (1 - entry_buffer_pct)
        tp_price = range_state.range_low + range_width * tp_ratio

        # Vérification anti-breakout : SL doit être EN-DESSOUS de l'entrée pour un BUY
        if sl_price >= price:
            logger.debug(
                "[%s] 🔄⚠️ Signal RANGE BUY rejeté (breakout) : prix %.4f déjà sous le SL %.4f",
                range_state.symbol, price, sl_price,
            )
            return None

        logger.debug(
            "[%s] 🔄🟢 RANGE BUY signal : price=%.4f ≤ buy_zone=%.4f (low=%.4f +%.2f%%) | TP=%.4f",
            range_state.symbol,
            price,
            buy_zone,
            range_state.range_low,
            entry_buffer_pct * 100,
            tp_price,
        )
        return {
            "side": OrderSide.BUY,
            "entry_price": price,
            "sl_price": sl_price,
            "tp_price": tp_price,
        }

    # ── SELL au haut du range ──
    sell_zone = range_state.range_high * (1 - entry_buffer_pct)
    if price >= sell_zone:
        sl_price = range_state.range_high * (1 + entry_buffer_pct)
        tp_price = range_state.range_high - range_width * tp_ratio

        # Vérification anti-breakout : SL doit être AU-DESSUS de l'entrée pour un SELL
        if sl_price <= price:
            logger.debug(
                "[%s] 🔄⚠️ Signal RANGE SELL rejeté (breakout) : prix %.4f déjà au-dessus du SL %.4f",
                range_state.symbol, price, sl_price,
            )
            return None

        logger.debug(
            "[%s] 🔄🔴 RANGE SELL signal : price=%.4f ≥ sell_zone=%.4f (high=%.4f -%.2f%%) | TP=%.4f",
            range_state.symbol,
            price,
            sell_zone,
            range_state.range_high,
            entry_buffer_pct * 100,
            tp_price,
        )
        return {
            "side": OrderSide.SELL,
            "entry_price": price,
            "sl_price": sl_price,
            "tp_price": tp_price,
        }

    return None


# ── Take Profit ────────────────────────────────────────────────────────────────

def check_range_tp_hit(
    position: Position,
    ticker: TickerData,
) -> bool:
    """
    Vérifie si le take profit d'une position RANGE est atteint.

    TP = range_mid stocké dans position.tp_price.

    Args:
        position: Position RANGE ouverte.
        ticker: Données de prix temps réel.

    Returns:
        True si le TP est atteint.
    """
    if position.strategy != StrategyType.RANGE:
        return False

    if position.tp_price is None:
        return False

    price = ticker.last_price

    if position.side == OrderSide.BUY:
        hit = price >= position.tp_price
    else:
        hit = price <= position.tp_price

    if hit:
        logger.info(
            "[%s] 🔄🎯 TP Range atteint : prix=%.4f, TP=%.4f",
            position.symbol,
            price,
            position.tp_price,
        )

    return hit


# ── Stop Loss (breakout du range) ─────────────────────────────────────────────

def check_range_sl_hit(
    position: Position,
    ticker: TickerData,
    sl_buffer_pct: float,
) -> bool:
    """
    Vérifie si le SL d'une position RANGE est atteint (cassure du range).

    Le SL est déjà enregistré dans position.sl_price (juste au-delà de la borne).

    Args:
        position: Position RANGE ouverte.
        ticker: Données de prix temps réel.
        sl_buffer_pct: Marge supplémentaire en pourcentage.

    Returns:
        True si le SL est atteint.
    """
    if position.strategy != StrategyType.RANGE:
        return False

    price = ticker.last_price
    sl = position.sl_price
    buffer = sl * sl_buffer_pct

    if position.side == OrderSide.BUY:
        # BUY au bas du range → SL si le prix casse encore plus bas
        hit = price <= sl - buffer
    else:
        # SELL au haut du range → SL si le prix casse encore plus haut
        hit = price >= sl + buffer

    if hit:
        logger.warning(
            "[%s] 🔄🛑 SL Range atteint (breakout) : prix=%.4f, SL=%.4f",
            position.symbol,
            price,
            sl,
        )

    return hit


# ── Cooldown ───────────────────────────────────────────────────────────────────

def is_in_cooldown(range_state: RangeState) -> bool:
    """
    Vérifie si le range est en période de cooldown (post-cassure).

    Args:
        range_state: Le range à vérifier.

    Returns:
        True si on est en cooldown (pas de nouveau trade range).
    """
    if range_state.cooldown_until <= 0:
        return False

    now = int(time.time())
    in_cd = now < range_state.cooldown_until

    if in_cd:
        remaining = range_state.cooldown_until - now
        logger.debug(
            "[%s] 🔄⏳ Cooldown actif : encore %d secondes",
            range_state.symbol,
            remaining,
        )

    return in_cd


def activate_cooldown(
    range_state: RangeState,
    cooldown_bars: int,
    bar_duration_seconds: int = 4 * 3600,
) -> None:
    """
    Active le cooldown sur un range après un SL/breakout.

    Args:
        range_state: Le range à mettre en cooldown.
        cooldown_bars: Nombre de bougies de cooldown (ex: 3 = 12h en H4).
        bar_duration_seconds: Durée d'une bougie en secondes (défaut: 4h = 14400s).
    """
    now = int(time.time())
    duration = cooldown_bars * bar_duration_seconds
    range_state.cooldown_until = now + duration
    logger.info(
        "[%s] 🔄⏳ Cooldown activé : %d bougies (%d secondes, jusqu'à %d)",
        range_state.symbol,
        cooldown_bars,
        duration,
        range_state.cooldown_until,
    )
