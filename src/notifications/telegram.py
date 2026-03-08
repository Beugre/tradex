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
    StrategyType.CRASHBOT: "💥 CRASHBOT",
    StrategyType.INFINITY: "♾️ INFINITY",
    StrategyType.LONDON: "🇬🇧 LONDON",
}

logger = logging.getLogger(__name__)

TELEGRAM_API_URL = "https://api.telegram.org"
DASHBOARD_URL = "http://213.199.41.168:8502"
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

    def notify_range_heartbeat(
        self,
        equity: float,
        open_positions: list[dict],
        max_positions: int,
        neutral_count: int,
        total_pairs: int,
        neutral_transitions: int,
        cycle_count: int,
        avg_range_width_pct: float = 0.0,
        exposure_usd: float = 0.0,
        exposure_pct: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> None:
        """Heartbeat Range avec positions en cours et compteur NEUTRAL."""
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        last_update = now_utc.strftime("%H:%M UTC")

        # System status basé sur PnL open
        n_losing = sum(1 for p in open_positions if p.get("pnl_pct", 0) < -5)
        if n_losing >= 2 or unrealized_pnl < -equity * 0.05:
            sys_emoji, sys_label = "🔴", "risk mode"
        elif unrealized_pnl < 0:
            sys_emoji, sys_label = "🟡", "watching"
        else:
            sys_emoji, sys_label = "🟢", "stable"

        lines = [
            f"{sys_emoji} *RANGE Heartbeat* — {sys_label} (cycle #{cycle_count})",
            f"  💰 Equity: `${equity:,.2f}`",
            f"  📊 Pos: {len(open_positions)}/{max_positions} | "
            f"Expo: `${exposure_usd:,.0f}` (`{exposure_pct:.0f}%`)",
        ]

        if avg_range_width_pct > 0:
            lines.append(f"  📐 Range moyen: `{avg_range_width_pct:.1f}%`")

        if open_positions:
            lines.append(f"  💵 PnL open: `${unrealized_pnl:+.2f}`")
            lines.append("")
            for p in open_positions:
                emoji = "🟢" if p["pnl_pct"] >= 0 else "🔴"
                zr = " 🔒" if p["status"] == "zero_risk" else ""
                lines.append(
                    f"  {emoji} {p['symbol']} {p['side'].upper()} `{p['pnl_pct']:+.2f}%`{zr}"
                )

        lines.append("")
        lines.append(
            f"  ⚪ Neutrals: {neutral_count}/{total_pairs} "
            f"(→ neutral: {neutral_transitions})"
        )
        lines.append(f"  🕐 `{last_update}`")
        lines.append(f"[Dashboard]({DASHBOARD_URL})")

        self._send("\n".join(lines))

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
        current_risk_pct: float | None = None,
        risk_engaged_pct: float = 0.0,
        unrealized_pnl: float = 0.0,
        unrealized_pnl_pct: float = 0.0,
    ) -> None:
        """Heartbeat périodique CrashBot."""
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        last_update = now_utc.strftime("%H:%M UTC")

        # System status
        if drawdown_pct < -5:
            sys_emoji, sys_label = "🔴", "risk mode"
        elif drawdown_pct < -3:
            sys_emoji, sys_label = "🟡", "drawdown"
        else:
            sys_emoji, sys_label = "🟢", "stable"

        kill_emoji = "🔴 ON" if kill_switch else "🟢 OFF"
        dd_emoji = "⚠️" if drawdown_pct < -3 else ""

        lines = [
            f"{sys_emoji} *CRASHBOT H4* — {sys_label}",
            f"  💰 Equity: `${equity:,.0f}` (alloué: `${allocated_equity:,.0f}`)",
            f"  📉 DD mois: `{drawdown_pct:+.1f}%` {dd_emoji}",
            f"  📊 Expo: `{exposure_pct:.0f}%` | Pos: `{open_positions}/{max_positions}`",
        ]

        if current_risk_pct is not None:
            lines.append(
                f"  🎯 Risk: `{current_risk_pct*100:.1f}%`/trade | "
                f"Engagé: `{risk_engaged_pct*100:.1f}%` ({open_positions} pos)"
            )

        # PnL réalisé vs open
        lines.append(
            f"  💵 PnL jour: `${daily_pnl:+.2f}` | "
            f"Open: `${unrealized_pnl:+.2f}` (`{unrealized_pnl_pct:+.1f}%`)"
        )
        lines.append(f"  Kill: {kill_emoji} | Signaux: {signals_detected}📡")

        if positions_detail:
            lines.append("")
            for p in positions_detail:
                gain = p.get("gain_pct", 0)
                emoji = "🟢" if gain >= 0 else "🔴"
                sl_dist = p.get("sl_dist_pct", 0)
                tp_dist = p.get("tp_dist_pct", 0)
                steps = p.get("steps", 0)
                steps_tag = f" ×{steps}" if steps > 0 else ""
                lines.append(
                    f"  {emoji} `{p['symbol']}`{steps_tag} (`{gain:+.1f}%`)"
                )
                lines.append(
                    f"    SL `{_fp(p['sl'])}` (`{sl_dist:+.1f}%`) | "
                    f"TP `{_fp(p.get('tp', 0))}` (`{tp_dist:+.1f}%`)"
                )

        if avg_api_latency_ms > 0:
            lines.append(f"  ⏱️ API: `{avg_api_latency_ms:.0f}ms`")
        lines.append(f"  🕐 `{last_update}`")
        lines.append(f"[Dashboard]({CRASHBOT_DASHBOARD_URL})")
        self._send("\n".join(lines))

    def notify_warning(self, title: str, detail: str) -> None:
        """Notification d'alerte anormale (API slow, data stale, slippage, DD)."""
        message = f"⚠️ *WARNING: {title}*\n  {detail}"
        self._send(message)

    # ── Infinity Bot ───────────────────────────────────────────────────────

    def notify_infinity_buy(
        self,
        symbol: str,
        level: int,
        price: float,
        size: float,
        cost_usd: float,
        pmp: float,
        total_invested: float,
        equity: float,
    ) -> None:
        """Notification d'achat DCA Infinity Bot."""
        base = symbol.split("-")[0] if "-" in symbol else symbol
        pct_equity = cost_usd / equity * 100 if equity > 0 else 0
        message = (
            f"📌 *Infinity Buy L{level + 1} – {symbol}* ♾️\n"
            f"  Prix: `{_fp(price)}` | Size: `{size:.8f} {base}` (`${cost_usd:.2f}`)\n"
            f"  PMP: `{_fp(pmp)}` | Investi: `${total_invested:.2f}` ({pct_equity:.0f}% equity)\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_infinity_sell(
        self,
        symbol: str,
        level: int,
        price: float,
        size: float,
        proceeds_usd: float,
        pmp: float,
        pnl_pct: float,
        remaining_size: float,
    ) -> None:
        """Notification de vente palier Infinity Bot."""
        base = symbol.split("-")[0] if "-" in symbol else symbol
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        tp_label = f"TP{level + 1}"
        message = (
            f"{emoji} *Infinity {tp_label} – {symbol}* ♾️\n"
            f"  Prix: `{_fp(price)}` | Size: `{size:.8f} {base}` (`${proceeds_usd:.2f}`)\n"
            f"  PMP: `{_fp(pmp)}` | Gain: `{pnl_pct:+.2f}%`\n"
            f"  Restant: `{remaining_size:.8f} {base}`\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_infinity_stop(
        self,
        symbol: str,
        price: float,
        pmp: float,
        total_cost: float,
        proceeds: float,
        pnl_usd: float,
        pnl_pct: float,
    ) -> None:
        """Notification de stop-loss Infinity Bot."""
        message = (
            f"🛑 *Infinity STOP – {symbol}* ♾️\n"
            f"  Prix: `{_fp(price)}` | PMP: `{_fp(pmp)}`\n"
            f"  Investi: `${total_cost:.2f}` → Récupéré: `${proceeds:.2f}`\n"
            f"  P&L: `{pnl_usd:+.2f} USD` (`{pnl_pct:+.1f}%`)\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_infinity_cycle_complete(
        self,
        symbol: str,
        pmp: float,
        total_cost: float,
        total_proceeds: float,
        pnl_usd: float,
        pnl_pct: float,
        n_buys: int,
        n_sells: int,
    ) -> None:
        """Notification de fin de cycle Infinity Bot."""
        emoji = "💰" if pnl_usd >= 0 else "💸"
        message = (
            f"{emoji} *Infinity Cycle terminé – {symbol}* ♾️\n"
            f"  PMP: `{_fp(pmp)}` | Buys: {n_buys} | Sells: {n_sells}\n"
            f"  Investi: `${total_cost:.2f}` → Récupéré: `${total_proceeds:.2f}`\n"
            f"  P&L: `{pnl_usd:+.2f} USD` (`{pnl_pct:+.1f}%`)\n"
            f"[Dashboard]({DASHBOARD_URL})"
        )
        self._send(message)

    def notify_infinity_heartbeat(
        self,
        equity: float,
        allocated_equity: float,
        phase: str,
        pmp: float,
        total_invested: float,
        size_btc: float,
        buys_filled: int,
        sells_filled: int,
        pnl_latent_usd: float,
        pnl_latent_pct: float,
        trailing_high: float,
        current_price: float,
        target_price: float,
        breakeven_active: bool,
        cycle_count: int,
        ecart_usd: float = 0.0,
        ecart_pct: float = 0.0,
        countdown_str: str = "",
        last_eval: dict | None = None,
    ) -> None:
        """Heartbeat périodique Infinity Bot."""
        be_tag = "🔒 BE" if breakeven_active else ""
        pnl_emoji = "🟢" if pnl_latent_usd >= 0 else "🔴"
        lines = [
            f"💓 *INFINITY BTC* ♾️",
            f"  Equity: `${equity:,.0f}` (alloué: `${allocated_equity:,.0f}`)",
            f"  Phase: `{phase}` | Cycle: `{cycle_count}` {be_tag}",
        ]
        if phase != "WAITING":
            lines.extend([
                f"  PMP: `{_fp(pmp)}` | Investi: `${total_invested:,.2f}`",
                f"  BTC: `{size_btc:.8f}` | Buys: `{buys_filled}/5` | Sells: `{sells_filled}/5`",
                f"  {pnl_emoji} Latent: `{pnl_latent_usd:+.2f}$` (`{pnl_latent_pct:+.1f}%`)",
            ])
        # Ecart prix/cible
        ecart_emoji = "🔥" if abs(ecart_pct) < 1.0 else ""
        lines.extend([
            "",
            f"  📊 Trail High: `{_fp(trailing_high)}` | Prix: `{_fp(current_price)}`",
            f"  🎯 Cible: `{_fp(target_price)}` | Écart: `${ecart_usd:+,.0f}` (`{ecart_pct:+.1f}%`) {ecart_emoji}",
            f"  ⏳ Prochaine H4: `{countdown_str}`",
        ])
        # Last eval
        if last_eval:
            drop_icon = "✅" if last_eval.get("drop_ok") else "❌"
            rsi_icon = "✅" if last_eval.get("rsi_ok") else "❌"
            vol_icon = "✅" if last_eval.get("vol_ok") else "❌"
            result_emoji = "🟢" if last_eval.get("result") == "ENTRY" else "⏸"
            lines.extend([
                "",
                f"  🔎 Dernière éval ({last_eval.get('ts', '?')[-5:]}): {result_emoji} `{last_eval.get('result', '?')}`",
                f"    {drop_icon} Drop: `{last_eval.get('drop_pct', 0):.1f}%` (seuil: `5.0%`)",
                f"    {rsi_icon} RSI: `{last_eval.get('rsi', 0):.1f}` (max: `50`)",
                f"    {vol_icon} Volume: `{last_eval.get('volume', 0):.0f}` vs MA `{last_eval.get('volume_ma', 0):.0f}`",
            ])
        else:
            lines.append("\n  🔎 Aucune évaluation encore")
        lines.append(f"[Dashboard]({DASHBOARD_URL})")
        self._send("\n".join(lines))

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
