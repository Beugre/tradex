"""
Paper Trading Reporter — Génère des rapports de performance des bots en paper trading.

- Appelé 2× par jour (10h et 18h UTC) via le service tradex-paper-reporter
- Disponible aussi on-demand via /paper dans le Telegram Command Bot
- Requête Firebase pour les trades avec paper=True
- Calcul des métriques par bot : PnL, WR, PF, nombre de trades, equity
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from collections import defaultdict
from datetime import datetime, timezone

from src import config
from src.firebase.client import get_documents
from src.notifications.telegram import TelegramNotifier

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.paper_reporter")

# Mapping exchange → bot label
EXCHANGE_LABELS: dict[str, str] = {
    "binance": "Trail Range",
    "binance-crashbot": "CrashBot",
    "binance-listing": "Listing Bot",
    "revolut-infinity": "Infinity",
    "revolut-london": "London Breakout",
    "revolut-breakout": "Breakout Momentum",
}


def _fetch_paper_trades() -> list[dict]:
    """Récupère tous les trades paper depuis Firebase."""
    return get_documents(
        "trades",
        filters=[("paper", "==", True)],
    )


def _compute_bot_metrics(trades: list[dict]) -> dict[str, dict]:
    """Agrège les métriques par bot (exchange)."""
    by_exchange: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        exchange = t.get("exchange", "unknown")
        by_exchange[exchange].append(t)

    result: dict[str, dict] = {}
    for exchange, bot_trades in sorted(by_exchange.items()):
        label = EXCHANGE_LABELS.get(exchange, exchange)
        total = len(bot_trades)
        closed = [t for t in bot_trades if t.get("status") == "CLOSED"]
        open_trades = [t for t in bot_trades if t.get("status") == "OPEN"]

        pnl_list = []
        for t in closed:
            pnl = t.get("pnl_net_usd") or t.get("pnl_usd")
            if pnl is not None:
                pnl_list.append(float(pnl))

        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        total_pnl = sum(pnl_list)
        win_rate = (len(wins) / len(pnl_list) * 100) if pnl_list else 0.0

        sum_wins = sum(wins)
        sum_losses = abs(sum(losses))
        pf = (sum_wins / sum_losses) if sum_losses > 0 else (999.0 if sum_wins > 0 else 0.0)

        # Unrealized PnL for open trades
        unrealized = 0.0
        for t in open_trades:
            entry = t.get("entry_filled") or t.get("entry_expected", 0)
            size = t.get("size", 0)
            # We don't have current price here, so we skip unrealized
            # (this is computed only when live prices available)

        result[exchange] = {
            "label": label,
            "total_trades": total,
            "closed_trades": len(closed),
            "open_trades": len(open_trades),
            "pnl": total_pnl,
            "win_rate": win_rate,
            "profit_factor": pf,
            "wins": len(wins),
            "losses": len(losses),
            "avg_win": (sum_wins / len(wins)) if wins else 0.0,
            "avg_loss": (sum_losses / len(losses)) if losses else 0.0,
        }

    return result


def generate_paper_report() -> str:
    """Génère le rapport texte paper trading pour Telegram."""
    trades = _fetch_paper_trades()
    if not trades:
        return ""

    metrics = _compute_bot_metrics(trades)
    if not metrics:
        return ""

    lines = ["📝 *PAPER TRADING REPORT*"]
    lines.append(f"_{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")
    lines.append("")

    total_pnl = 0.0
    total_trades = 0

    for exchange, m in metrics.items():
        pnl_sign = "+" if m["pnl"] >= 0 else ""
        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] < 999 else "∞"

        lines.append(f"*{m['label']}*")
        lines.append(
            f"  PnL: {pnl_sign}${m['pnl']:.2f} | Trades: {m['closed_trades']}"
            f" ({m['open_trades']} open)"
        )
        lines.append(
            f"  WR: {m['win_rate']:.1f}% | PF: {pf_str}"
            f" | W/L: {m['wins']}/{m['losses']}"
        )
        if m["avg_win"] > 0 or m["avg_loss"] > 0:
            lines.append(
                f"  Avg win: +${m['avg_win']:.2f}"
                f" | Avg loss: -${m['avg_loss']:.2f}"
            )
        lines.append("")

        total_pnl += m["pnl"]
        total_trades += m["closed_trades"]

    # Summary
    pnl_sign = "+" if total_pnl >= 0 else ""
    lines.append(f"*TOTAL*: {pnl_sign}${total_pnl:.2f} sur {total_trades} trades")

    return "\n".join(lines)


def _run_scheduled() -> None:
    """Boucle principale — envoie le rapport à 10h et 18h UTC."""
    telegram = TelegramNotifier(
        bot_token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
    )
    report_hours = set(config.PAPER_REPORT_HOURS)
    last_report_hour: int = -1
    running = True

    def _stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    logger.info("📝 Paper Reporter démarré — rapports à %s UTC", report_hours)

    while running:
        now = datetime.now(timezone.utc)
        current_hour = now.hour

        if current_hour in report_hours and current_hour != last_report_hour:
            logger.info("📝 Génération du rapport paper trading (%dh UTC)", current_hour)
            try:
                report = generate_paper_report()
                if report:
                    telegram.send_raw(report)
                    logger.info("📝 Rapport envoyé")
                else:
                    logger.info("📝 Aucun trade paper — rapport vide, pas d'envoi")
            except Exception as e:
                logger.error("📝 Erreur génération rapport: %s", e)
            last_report_hour = current_hour

        time.sleep(60)  # Check every minute


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Trading Reporter")
    parser.add_argument("--once", action="store_true", help="Envoyer un rapport et quitter")
    args = parser.parse_args()

    if args.once:
        report = generate_paper_report()
        if report:
            telegram = TelegramNotifier(
                bot_token=config.TELEGRAM_BOT_TOKEN,
                chat_id=config.TELEGRAM_CHAT_ID,
            )
            telegram.send_raw(report)
            print(report)
        else:
            print("Aucun trade paper détecté.")
    else:
        _run_scheduled()


if __name__ == "__main__":
    main()
