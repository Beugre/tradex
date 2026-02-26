"""
Notifications Telegram pour le bot TradeX.

Envoie des alertes formatées via le Telegram Bot API pour :
- Placement d'ordres (entrée)
- Stop loss atteint
- Changement de tendance
- Passage en zero-risk
- Clôture de position
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from src.core.models import (
    OrderSide,
    Position,
    StrategyType,
    TrendDirection,
    TrendState,
)

_STRATEGY_LABEL = {
    StrategyType.TREND: "📊 TREND",
    StrategyType.RANGE: "🔄 RANGE",
}

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org"
DASHBOARD_URL = "http://213.199.41.168:8502"


def _fp(price: float) -> str:
    """Formate un prix avec assez de décimales (min 4) pour éviter les arrondis."""
    if price <= 0:
        return "0"
    if price >= 1.0:
        return f"{price:.4f}"
    decimals = 4
    temp = price
    while temp < 0.01 and decimals < 10:
        temp *= 10
        decimals += 1
    return f"{price:.{decimals}f}"


class TelegramNotifier:
    """Envoie des notifications via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._client = httpx.Client(timeout=10.0)
        self._enabled = bool(bot_token and chat_id)

        if not self._enabled:
            logger.warning(
                "Telegram non configuré (token ou chat_id manquant). "
                "Les notifications seront désactivées."
            )

    def notify_entry(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        sl_price: float,
        size: float,
        risk_percent: float,
        risk_amount: float,
        strategy: StrategyType = StrategyType.TREND,
        tp_price: Optional[float] = None,
    ) -> None:
        """Notification de placement d'un ordre d'entrée."""
        emoji = "📈" if side == OrderSide.BUY else "📉"
        action = "BUY" if side == OrderSide.BUY else "SELL"
        base = symbol.split("-")[0] if "-" in symbol else symbol
        strat_label = _STRATEGY_LABEL.get(strategy, "")

        tp_info = ""
        if tp_price is not None:
            tp_info = f" | TP: `{_fp(tp_price)}`"

        message = (
            f"{emoji} *{action} déclenché – {symbol}* {strat_label}\n"
            f"  Entrée: `{_fp(entry_price)}` | SL: `{_fp(sl_price)}`{tp_info} | "
            f"Size: `{size:.8f} {base}`\n"
            f"  Risque: {risk_percent*100:.0f}% ({risk_amount:.2f} USD)\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_sl_hit(
        self,
        position: Position,
        exit_price: float,
    ) -> None:
        """Notification de stop loss atteint."""
        pnl = _calculate_pnl(position, exit_price)
        emoji = "✅" if pnl >= 0 else "🛑"

        message = (
            f"{emoji} *SL atteint – {position.symbol}*\n"
            f"  Entrée: `{_fp(position.entry_price)}` → Sortie: `{_fp(exit_price)}`\n"
            f"  P&L: `{pnl:+.2f} USD`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_trend_change(
        self,
        trend: TrendState,
        previous_direction: TrendDirection,
    ) -> None:
        """Notification de changement de tendance."""
        arrows = {
            TrendDirection.BULLISH: "🟢 BULLISH",
            TrendDirection.BEARISH: "🔴 BEARISH",
            TrendDirection.NEUTRAL: "⚪ NEUTRAL",
        }
        message = (
            f"🔄 *Changement de tendance – {trend.symbol}*\n"
            f"  {arrows.get(previous_direction, '?')} → "
            f"{arrows.get(trend.direction, '?')}"
        )
        self._send(message)

    def notify_zero_risk(
        self,
        position: Position,
        new_sl: float,
    ) -> None:
        """Notification de passage en zero-risk."""
        message = (
            f"🔒 *Zero-risk activé – {position.symbol}*\n"
            f"  Entrée: `{_fp(position.entry_price)}` | "
            f"Nouveau SL: `{_fp(new_sl)}`\n"
            f"  Profit verrouillé ✅\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_trailing_stop(
        self,
        position: Position,
        new_sl: float,
    ) -> None:
        """Notification de mise à jour du trailing stop."""
        emoji = "📈" if position.side == OrderSide.BUY else "📉"
        peak = position.peak_price or 0.0
        message = (
            f"{emoji} *Trailing stop – {position.symbol}*\n"
            f"  Peak: `{_fp(peak)}` | Nouveau SL: `{_fp(new_sl)}`\n"
            f"  Entrée: `{_fp(position.entry_price)}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_position_closed(
        self,
        position: Position,
        exit_price: float,
        reason: str = "Clôture",
    ) -> None:
        """Notification de clôture de position."""
        pnl = _calculate_pnl(position, exit_price)
        emoji = "💰" if pnl >= 0 else "💸"

        message = (
            f"{emoji} *Position clôturée – {position.symbol}*\n"
            f"  Raison: {reason}\n"
            f"  Entrée: `{_fp(position.entry_price)}` → Sortie: `{_fp(exit_price)}`\n"
            f"  P&L: `{pnl:+.2f} USD`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_range_tp_hit(
        self,
        position: Position,
        exit_price: float,
    ) -> None:
        """Notification de take profit range atteint."""
        pnl = _calculate_pnl(position, exit_price)
        message = (
            f"🔄🎯 *TP Range atteint – {position.symbol}*\n"
            f"  Entrée: `{_fp(position.entry_price)}` → Sortie: `{_fp(exit_price)}`\n"
            f"  P&L: `{pnl:+.2f} USD` ✅\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_forced_exit(
        self,
        position: Position,
        exit_price: float,
        reason: str,
    ) -> None:
        """Notification de sortie forcée (ex: tendance confirmée pendant range)."""
        pnl = _calculate_pnl(position, exit_price)
        emoji = "💰" if pnl >= 0 else "💸"
        message = (
            f"{emoji} *Sortie forcée – {position.symbol}* 🔄→📊\n"
            f"  Raison: {reason}\n"
            f"  Entrée: `{_fp(position.entry_price)}` → Sortie: `{_fp(exit_price)}`\n"
            f"  P&L: `{pnl:+.2f} USD`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_reconciliation(
        self,
        confirmed: int,
        removed: int,
        orphans: list[str],
        recovered: list | None = None,
    ) -> None:
        """Notification du résultat de la réconciliation au démarrage."""
        orphan_text = "\n".join(f"  • {o}" for o in orphans) if orphans else "Aucun"

        recovered_text = ""
        if recovered:
            lines = []
            for pos in recovered:
                lines.append(
                    f"  • {pos.symbol}: {pos.size:.8f} @ {_fp(pos.entry_price)} | SL={_fp(pos.sl_price)}"
                )
            recovered_text = (
                f"\n🔄 Positions reconstruites: {len(recovered)}\n"
                + "\n".join(lines)
            )

        message = (
            f"🔄 *Réconciliation au démarrage*\n"
            f"✅ Positions confirmées: {confirmed}\n"
            f"❌ Positions retirées: {removed}\n"
            f"🔍 Soldes orphelins: {len(orphans)}\n{orphan_text}"
            f"{recovered_text}"
        )
        self._send(message)

    def notify_error(self, error_message: str) -> None:
        """Notification d'erreur critique."""
        message = f"⚠️ *Erreur TradeX*\n  `{error_message}`"
        self._send(message)

    # ── Envoi ──────────────────────────────────────────────────────────────────

    def _send(self, message: str) -> None:
        """Envoie un message via Telegram Bot API."""
        if not self._enabled:
            logger.info("[Telegram OFF] %s", message)
            return

        url = f"{TELEGRAM_API_URL}/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            response = self._client.post(url, json=payload)
            if response.status_code != 200:
                logger.error(
                    "Telegram API error %d: %s",
                    response.status_code,
                    response.text,
                )
            else:
                logger.debug("Notification Telegram envoyée")
        except httpx.HTTPError as e:
            logger.error("Erreur d'envoi Telegram: %s", e)

    def close(self) -> None:
        """Ferme le client HTTP."""
        self._client.close()


def _calculate_pnl(position: Position, exit_price: float) -> float:
    """Calcule le P&L approximatif d'une position."""
    if position.side == OrderSide.BUY:
        return (exit_price - position.entry_price) * position.size
    else:
        return (position.entry_price - exit_price) * position.size
