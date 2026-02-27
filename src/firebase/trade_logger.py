"""
Trade Logger Firebase — enregistre chaque étape du cycle de vie d'un trade.

Collections Firestore :
  - trades          : 1 doc par trade (créé à l'ouverture, mis à jour à la clôture)
  - daily_snapshots : 1 doc par jour (equity, positions, risque)
  - events          : événements système (tendance, erreurs, heartbeat)

Architecture : Bot → exécution → log immédiat Firebase → vérité indépendante du broker.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src import config
from src.core.models import (
    OrderSide,
    Position,
    PositionStatus,
    StrategyType,
    TrendDirection,
)
from src.firebase.client import add_document, update_document, get_documents

logger = logging.getLogger("tradex.firebase.trades")


def _effective_sl(sl_price: float, side: str, strategy: str) -> Optional[float]:
    """Calcule le SL effectif (avec buffer) = le vrai seuil de déclenchement."""
    if not sl_price or sl_price <= 0:
        return sl_price
    # BREAKOUT : le SL est basé sur ATR, pas de buffer additionnel
    if strategy == "BREAKOUT":
        return sl_price
    buf = (
        config.RANGE_SL_BUFFER_PERCENT
        if strategy == "RANGE"
        else config.SL_BUFFER_PERCENT
    )
    if side == "buy":
        return sl_price * (1 - buf)
    else:
        return sl_price * (1 + buf)


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════════


def log_trade_opened(
    position: Position,
    fill_type: str,
    maker_wait_seconds: int,
    risk_pct: float,
    risk_amount_usd: float,
    fiat_balance: float,
    current_equity: float,
    portfolio_risk_before: float,
    exchange: str = "revolut",
) -> Optional[str]:
    """
    Log l'ouverture d'un trade dans Firebase.
    Retourne le trade_id (= doc ID Firestore) ou None.
    """
    now = datetime.now(timezone.utc)
    trade_id = str(uuid.uuid4())

    # Contexte du signal pour audit
    signal_context = (
        f"{position.symbol}|{position.side.value}|"
        f"{position.entry_price}|{position.sl_price}|{position.strategy.value}"
    )
    signal_hash = hashlib.sha256(signal_context.encode()).hexdigest()[:16]

    doc = {
        # Identité
        "trade_id": trade_id,
        "symbol": position.symbol,
        "venue_order_id": position.venue_order_id,
        "exchange": exchange,

        # Signal
        "signal_type": position.strategy.value,              # TREND ou RANGE
        "signal_timestamp": now.isoformat(),
        "signal_context_hash": signal_hash,

        # Entry
        "side": position.side.value,
        "entry_expected": position.entry_price,
        "entry_filled": position.entry_price,                # Même chose pour limit
        "entry_slippage_pct": 0.0,                           # Limit = 0 slippage théorique
        "maker_or_taker": fill_type,
        "maker_wait_seconds": maker_wait_seconds,

        # Position
        "size": position.size,
        "size_usd": position.size * position.entry_price,
        "risk_pct": risk_pct,
        "risk_amount_usd": risk_amount_usd,
        "max_allowed_risk": config.MAX_TOTAL_RISK_PERCENT,
        "portfolio_risk_before": portfolio_risk_before,
        "fiat_balance_at_entry": fiat_balance,
        "equity_at_entry": current_equity,
        "sl_price": position.sl_price,
        "sl_price_effective": _effective_sl(
            position.sl_price, position.side.value, position.strategy.value,
        ),
        "tp_price": position.tp_price,
        "tp_price_effective": position.tp_price,  # TP n'a pas de buffer

        # Status
        "status": "OPEN",
        "is_zero_risk_applied": False,
        "opened_at": now.isoformat(),
        "closed_at": None,

        # Exit (rempli à la clôture)
        "exit_price": None,
        "exit_reason": None,
        "exit_fill_type": None,
        "holding_time_hours": None,

        # Result (rempli à la clôture)
        "fees_entry": _estimate_fee(position.size * position.entry_price, fill_type),
        "fees_exit": None,
        "fees_total": None,
        "pnl_usd": None,
        "pnl_pct": None,
        "equity_after": None,

        # Metadata
        "bot_version": "1.0",
        "dry_run": False,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    doc_id = add_document("trades", doc, doc_id=trade_id)
    if doc_id:
        logger.info(
            "🔥 [%s] Trade OPEN logged → %s (%s %s @ %s)",
            position.symbol, trade_id[:8],
            position.side.value.upper(),
            position.strategy.value,
            position.entry_price,
        )
    return doc_id


def log_trade_closed(
    trade_id: str,
    position: Position,
    exit_price: float,
    reason: str,
    fill_type: str,
    equity_after: float,
    actual_exit_size: Optional[float] = None,
) -> bool:
    """Met à jour le trade avec les données de clôture.

    Args:
        actual_exit_size: Taille réellement vendue (après ajustement au solde réel).
                          Si None, utilise position.size (comportement legacy).
    """
    now = datetime.now(timezone.utc)

    # Taille réellement échangée (peut différer de position.size à cause des fees d'achat)
    exit_size = actual_exit_size if actual_exit_size is not None else position.size

    # Calcul P&L brut — sur la taille réellement vendue
    if position.side == OrderSide.BUY:
        pnl_gross = (exit_price - position.entry_price) * exit_size
    else:
        pnl_gross = (position.entry_price - exit_price) * exit_size

    notional_entry = exit_size * position.entry_price
    pnl_pct = pnl_gross / notional_entry if notional_entry > 0 else 0

    # Fees
    fee_entry = _estimate_fee(exit_size * position.entry_price, "maker")
    fee_exit = _estimate_fee(exit_size * exit_price, fill_type)
    fees_total = fee_entry + fee_exit

    # PnL net = brut - fees
    pnl_net = pnl_gross - fees_total
    pnl_net_pct = pnl_net / notional_entry if notional_entry > 0 else 0

    # Holding time
    # On retrouve l'heure d'ouverture depuis Firebase
    trades = get_documents("trades", filters=[("trade_id", "==", trade_id)], limit=1)
    opened_at_str = trades[0].get("opened_at") if trades else None
    holding_hours = None
    if opened_at_str:
        try:
            opened_at = datetime.fromisoformat(opened_at_str)
            holding_hours = round((now - opened_at).total_seconds() / 3600, 2)
        except (ValueError, TypeError):
            pass

    updates = {
        "status": "CLOSED",
        "exit_price": exit_price,
        "exit_reason": reason,
        "exit_fill_type": fill_type,
        "holding_time_hours": holding_hours,
        "closed_at": now.isoformat(),

        "actual_exit_size": exit_size,
        "fees_entry": fee_entry,
        "fees_exit": fee_exit,
        "fees_total": fees_total,
        "pnl_usd": round(pnl_gross, 4),
        "pnl_net_usd": round(pnl_net, 4),
        "pnl_pct": round(pnl_pct, 6),
        "pnl_net_pct": round(pnl_net_pct, 6),
        "equity_after": equity_after,

        "is_zero_risk_applied": position.is_zero_risk_applied,
        "updated_at": now.isoformat(),
    }

    ok = update_document("trades", trade_id, updates)
    if ok:
        emoji = "🟢" if pnl_net >= 0 else "🔴"
        logger.info(
            "🔥 [%s] Trade CLOSED %s → PnL brut=$%+.2f | net=$%+.2f (%+.2f%%) | fees=$%.4f | %s | %.1fh",
            position.symbol, emoji, pnl_gross, pnl_net, pnl_net_pct * 100,
            fees_total, reason, holding_hours or 0,
        )
    return ok


def log_zero_risk_applied(trade_id: str, new_sl: float) -> bool:
    """Met à jour le flag zero-risk dans Firebase."""
    return update_document("trades", trade_id, {
        "is_zero_risk_applied": True,
        "sl_price": new_sl,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


def log_trailing_sl_update(trade_id: str, new_sl: float) -> bool:
    """Met à jour le SL trailing dans Firebase."""
    return update_document("trades", trade_id, {
        "sl_price": new_sl,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# EVENTS (tendance, erreurs, heartbeat)
# ═══════════════════════════════════════════════════════════════════════════════


def log_event(
    event_type: str,
    data: dict[str, Any],
    symbol: Optional[str] = None,
    exchange: str = "revolut",
) -> Optional[str]:
    """Log un événement système dans la collection `events`."""
    now = datetime.now(timezone.utc)
    doc = {
        "event_type": event_type,
        "symbol": symbol,
        "exchange": exchange,
        "timestamp": now.isoformat(),
        "data": data,
    }
    return add_document("events", doc)


def log_trend_change(
    symbol: str,
    old_direction: TrendDirection,
    new_direction: TrendDirection,
    exchange: str = "revolut",
) -> Optional[str]:
    """Log un changement de tendance."""
    return log_event("TREND_CHANGE", {
        "old": old_direction.value,
        "new": new_direction.value,
    }, symbol=symbol, exchange=exchange)


def log_heartbeat(
    open_positions: int,
    total_equity: float,
    total_risk_pct: float,
    pairs_count: int,
    exchange: str = "revolut",
) -> Optional[str]:
    """Log un heartbeat périodique."""
    return log_event("HEARTBEAT", {
        "open_positions": open_positions,
        "total_equity": total_equity,
        "total_risk_pct": round(total_risk_pct, 4),
        "pairs_count": pairs_count,
    }, exchange=exchange)


def log_close_failure(
    symbol: str,
    attempt: int,
    error: str,
    next_retry_seconds: int,
    trade_id: Optional[str] = None,
) -> Optional[str]:
    """Log un échec de clôture dans Firebase (collection `events`)."""
    doc_id = log_event("CLOSE_FAILURE", {
        "attempt": attempt,
        "error": error[:300],
        "next_retry_seconds": next_retry_seconds,
        "trade_id": trade_id,
    }, symbol=symbol)

    # Aussi marquer le trade comme bloqué si on a le trade_id
    if trade_id:
        try:
            update_document("trades", trade_id, {
                "close_blocked": True,
                "close_blocked_attempts": attempt,
                "close_blocked_error": error[:200],
                "close_blocked_at": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            logger.warning("Firebase: impossible de marquer trade bloqué: %s", e)

    return doc_id


def clear_close_failure(trade_id: str) -> None:
    """Nettoie le flag close_blocked quand la clôture réussit."""
    try:
        update_document("trades", trade_id, {
            "close_blocked": False,
            "close_blocked_attempts": 0,
            "close_blocked_error": None,
            "close_blocked_at": None,
        })
    except Exception as e:
        logger.warning("Firebase: impossible de clear close_blocked: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════


def log_daily_snapshot(
    equity: float,
    positions: list[dict[str, Any]],
    daily_pnl: float,
    trades_today: int,
    exchange: str = "revolut",
) -> Optional[str]:
    """Sauvegarde un snapshot quotidien."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    doc_id = f"{today}_{exchange}"
    doc = {
        "date": today,
        "equity": equity,
        "positions": positions,
        "daily_pnl": round(daily_pnl, 4),
        "trades_today": trades_today,
        "exchange": exchange,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return add_document("daily_snapshots", doc, doc_id=doc_id)


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════


def cleanup_old_events(retention_days: Optional[int] = None) -> int:
    """Supprime les events plus vieux que `retention_days` jours.

    Ne touche PAS aux collections `trades` et `daily_snapshots`.
    Retourne le nombre de documents supprimés.
    """
    from src.firebase.client import delete_documents_batch

    days = retention_days or config.FIREBASE_EVENTS_RETENTION_DAYS
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()

    deleted = delete_documents_batch(
        "events",
        filters=[("timestamp", "<", cutoff_iso)],
        batch_size=200,
    )

    if deleted > 0:
        logger.info(
            "🧹 Firebase cleanup: %d events supprimés (> %d jours)",
            deleted, days,
        )
    else:
        logger.debug("🧹 Firebase cleanup: rien à supprimer (rétention %dj)", days)

    return deleted


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _estimate_fee(notional_usd: float, fill_type: str) -> float:
    """Estime les fees selon maker (0%) ou taker (0.09%)."""
    if fill_type == "maker":
        return 0.0
    return round(notional_usd * 0.0009, 4)  # taker 0.09%


def get_open_trades() -> list[dict[str, Any]]:
    """Retourne les trades actuellement ouverts dans Firebase."""
    return get_documents("trades", filters=[("status", "==", "OPEN")])


def get_trades_since(iso_date: str) -> list[dict[str, Any]]:
    """Retourne les trades depuis une date ISO (ex: '2026-02-21')."""
    return get_documents(
        "trades",
        filters=[("opened_at", ">=", iso_date)],
        order_by="opened_at",
    )
