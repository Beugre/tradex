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
    StrategyType.BREAKOUT: "🔥 BREAKOUT",
    StrategyType.CRASHBOT: "💥 CRASHBOT",
}

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org"
DASHBOARD_URL = "http://213.199.41.168:8502"
BREAKOUT_DASHBOARD_URL = "http://213.199.41.168:8504"
CRASHBOT_DASHBOARD_URL = "http://213.199.41.168:8504"


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

    # ── Breakout-specific ──────────────────────────────────────────────────────

    def notify_breakout_heartbeat(
        self,
        equity: float,
        allocated_equity: float,
        drawdown_pct: float,
        exposure_pct: float,
        open_positions: int,
        max_positions: int,
        daily_pnl: float,
        daily_pnl_pct: float,
        kill_switch: bool,
        positions_detail: list[dict] | None = None,
        signals_detected: int = 0,
        signals_rejected: int = 0,
        avg_api_latency_ms: float = 0,
    ) -> None:
        """Heartbeat périodique niveau fund pour le bot Breakout."""
        kill_emoji = "🔴 ON" if kill_switch else "🟢 OFF"
        dd_warn = " ⚠️" if drawdown_pct < -5 else ""
        lines = [
            "💓 *BREAKOUT H4*",
            f"  Equity: `${equity:,.0f}` (alloué: `${allocated_equity:,.0f}`)",
            f"  DD: `{drawdown_pct:+.1f}%`{dd_warn} | Expo: `{exposure_pct:.0f}%`",
            f"  Pos: `{open_positions}/{max_positions}` | PnL jour: `{daily_pnl:+.2f}$` (`{daily_pnl_pct:+.1f}%`)",
            f"  Kill: {kill_emoji} | Signaux: {signals_detected}📡 {signals_rejected}❌",
        ]

        if positions_detail:
            lines.append("")
            for p in positions_detail:
                gain = p.get("gain_pct", 0)
                emoji = "🟢" if gain >= 0 else "🔴"
                lines.append(
                    f"  {emoji} `{p['symbol']}` @ `{_fp(p['entry'])}` | "
                    f"SL=`{_fp(p['sl'])}` | peak=`{_fp(p['peak'])}` (`{gain:+.1f}%`)"
                )

        if avg_api_latency_ms > 0:
            lines.append(f"  ⏱️ API: `{avg_api_latency_ms:.0f}ms` moy")
        lines.append(f"[Dashboard]({BREAKOUT_DASHBOARD_URL})")
        self._send("\n".join(lines))

    def notify_signal_rejected(
        self,
        symbol: str,
        reasons: list[str],
        filters_passed: int = 0,
        filters_total: int = 4,
    ) -> None:
        """Notification quand un breakout est proche mais filtré."""
        reason_text = "\n".join(f"  → {r}" for r in reasons)
        message = (
            f"❌ *{symbol}* breakout proche ({filters_passed}/{filters_total} filtres)\n"
            f"{reason_text}\n"
            f"[Dashboard]({BREAKOUT_DASHBOARD_URL})"
        )
        self._send(message)

    def notify_breakout_entry(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        size: float,
        size_usd: float,
        risk_pct: float,
        risk_usd: float,
        sl_distance_pct: float,
        adx: float = 0,
        bb_width: float = 0,
        volume_ratio: float = 0,
    ) -> None:
        """Notification riche d'entrée Breakout."""
        base = symbol.replace("USDC", "").replace("-", "")
        message = (
            f"🚀 *LONG {symbol}* 🔥 BREAKOUT\n"
            f"  Size: `{size:.6f} {base}` (`${size_usd:.0f}`)\n"
            f"  Entrée: `{_fp(entry_price)}` | SL: `{_fp(sl_price)}` (`{sl_distance_pct:-.1f}%`)\n"
            f"  Risk: `{risk_pct*100:.1f}%` (`${risk_usd:.2f}`)\n"
            f"  ADX=`{adx:.1f}` | BBw=`{bb_width:.4f}` | Vol=`{volume_ratio:.1f}x`\n"
            f"[Dashboard]({BREAKOUT_DASHBOARD_URL})"
        )
        self._send(message)

    def notify_breakout_trail(
        self,
        symbol: str,
        entry_price: float,
        old_sl: float,
        new_sl: float,
        peak: float,
        gain_pct: float,
        palier: str = "",
    ) -> None:
        """Notification de trailing stop Breakout avec détail du palier."""
        sl_vs_entry = (new_sl - entry_price) / entry_price * 100 if entry_price > 0 else 0
        message = (
            f"🔒 *TRAIL {symbol}*\n"
            f"  SL: `{_fp(old_sl)}` → `{_fp(new_sl)}` (`{sl_vs_entry:+.1f}%` vs entry)\n"
            f"  Peak: `{_fp(peak)}` (`{gain_pct:+.1f}%`)"
        )
        if palier:
            message += f" | {palier}"
        message += f"\n[Dashboard]({BREAKOUT_DASHBOARD_URL})"
        self._send(message)

    def notify_crashbot_entry(
        self,
        symbol: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        size: float,
        size_usd: float,
        risk_pct: float,
        risk_usd: float,
        sl_distance_pct: float,
        drop_pct: float,
    ) -> None:
        """Notification riche d'entrée CrashBot (Dip Buy)."""
        base = symbol.replace("USDC", "").replace("-", "")
        message = (
            f"💥 *LONG {symbol}* 📉 DIP BUY\n"
            f"  Drop: `{drop_pct*100:+.1f}%` en 48h\n"
            f"  Size: `{size:.6f} {base}` (`${size_usd:.0f}`)\n"
            f"  Entrée: `{_fp(entry_price)}` | SL: `{_fp(sl_price)}` (`{sl_distance_pct:-.1f}%`)\n"
            f"  TP: `{_fp(tp_price)}` (`+{(tp_price/entry_price - 1)*100:.1f}%`)\n"
            f"  Risk: `{risk_pct*100:.1f}%` (`${risk_usd:.2f}`)\n"
            f"[Dashboard]({CRASHBOT_DASHBOARD_URL})"
        )
        self._send(message)

    def notify_crashbot_trail(
        self,
        symbol: str,
        entry_price: float,
        old_sl: float,
        new_sl: float,
        old_tp: float,
        new_tp: float,
        gain_pct: float,
        steps: int,
    ) -> None:
        """Notification de step trailing CrashBot."""
        sl_vs_entry = (new_sl - entry_price) / entry_price * 100 if entry_price > 0 else 0
        message = (
            f"🔒 *TRAIL {symbol}* (step {steps})\n"
            f"  SL: `{_fp(old_sl)}` → `{_fp(new_sl)}` (`{sl_vs_entry:+.1f}%` vs entry)\n"
            f"  TP: `{_fp(old_tp)}` → `{_fp(new_tp)}`\n"
            f"  Gain peak: `{gain_pct:+.1f}%`\n"
            f"[Dashboard]({CRASHBOT_DASHBOARD_URL})"
        )
        self._send(message)

    def notify_crashbot_heartbeat(
        self,
        equity: float,
        allocated_equity: float,
        drawdown_pct: float,
        exposure_pct: float,
        open_positions: int,
        max_positions: int,
        daily_pnl: float,
        daily_pnl_pct: float,
        kill_switch: bool,
        positions_detail: list[dict],
        signals_detected: int,
        avg_api_latency_ms: float = 0,
    ) -> None:
        """Heartbeat périodique CrashBot."""
        kill_emoji = "🔴 ON" if kill_switch else "🟢 OFF"
        dd_emoji = "⚠️" if drawdown_pct < -3 else ""
        lines = [
            f"💓 *CRASHBOT H4*",
            f"  Equity: `${equity:,.0f}` (alloué: `${allocated_equity:,.0f}`)",
            f"  DD mois: `{drawdown_pct:+.1f}%` {dd_emoji}",
            f"  Expo: `{exposure_pct:.0f}%` | Pos: `{open_positions}/{max_positions}`",
            f"  PnL jour: `${daily_pnl:+.2f}` (`{daily_pnl_pct:+.1f}%`)",
            f"  Kill: {kill_emoji} | Signaux: {signals_detected}📡",
        ]

        if positions_detail:
            lines.append("")
            for p in positions_detail:
                gain = p.get("gain_pct", 0)
                emoji = "🟢" if gain >= 0 else "🔴"
                lines.append(
                    f"  {emoji} `{p['symbol']}` @ `{_fp(p['entry'])}` | "
                    f"SL=`{_fp(p['sl'])}` | TP=`{_fp(p.get('tp', 0))}` (`{gain:+.1f}%`)"
                )

        if avg_api_latency_ms > 0:
            lines.append(f"  ⏱️ API: `{avg_api_latency_ms:.0f}ms` moy")
        lines.append(f"[Dashboard]({CRASHBOT_DASHBOARD_URL})")
        self._send("\n".join(lines))

    def notify_warning(self, title: str, detail: str) -> None:
        """Notification d'alerte anormale (API slow, data stale, slippage, DD)."""
        message = f"⚠️ *WARNING: {title}*\n  {detail}"
        self._send(message)

    def send_raw(self, text: str) -> None:
        """Envoie un message brut (public)."""
        self._send(text)

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
