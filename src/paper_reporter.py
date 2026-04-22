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
from typing import Optional

from src import config
from src.firebase.client import get_documents
from src.notifications.telegram import TelegramNotifier

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tradex.paper_reporter")

# ── Évaluation périodique (passage live) ─────────────────────────────────────
# Conditions requises pour recommander le passage live d'un bot
EVAL_SCHEDULE_DAY: int = int(__import__('os').getenv("PAPER_EVAL_DAY", "15"))  # 15 du mois
EVAL_SCHEDULE_HOUR: int = 10  # 10h UTC

EVAL_CRITERIA: dict[str, dict] = {
    # Binance
    "binance": {
        "label": "Trail Range",
        "min_trades": 30,
        "min_wr_pct": 35.0,
        "min_pf": 1.2,
        # Mean-reversion : WR faible est normal, le PF prime
    },
    "binance-crashbot": {
        "label": "CrashBot",
        "min_trades": 20,
        "min_wr_pct": 60.0,
        "min_pf": 1.5,
    },
    "binance-listing": {
        "label": "Listing Bot",
        "min_trades": 10,
        "min_wr_pct": 45.0,
        "min_pf": 1.5,
        # Peu de trades attendus (événements rares)
    },
    "binance-adaptive": {
        "label": "Adaptive Bull",
        "min_trades": 20,
        "min_wr_pct": 30.0,
        "min_pf": 1.1,
        # Backtest : WR 34.1%, PF 1.18 — seuils proches du backtest
    },
    # Revolut X
    "revolut-infinity": {
        "label": "Infinity",
        "min_trades": 8,
        "min_wr_pct": 50.0,
        "min_pf": 1.3,
        # Cycles longs, peu de trades attendus en paper
    },
    "revolut-london": {
        "label": "London Breakout",
        "min_trades": 15,
        "min_wr_pct": 45.0,
        "min_pf": 1.5,
        # Backtest : PF 1.98
    },
    "revolut-dca": {
        "label": "DCA RSI v2",
        "min_trades": 20,
        "min_wr_pct": 50.0,
        "min_pf": 1.2,
    },
    "revolut-breakout": {
        "label": "Breakout Momentum",
        "min_trades": 15,
        "min_wr_pct": 55.0,
        "min_pf": 2.0,
        # Backtest : WR 67.1%, PF 4.51 — seuils conservateurs
    },
}


# Mapping exchange → bot label
EXCHANGE_LABELS: dict[str, str] = {
    "binance": "Trail Range",
    "binance-crashbot": "CrashBot",
    "binance-listing": "Listing Bot",
    "binance-adaptive": "Adaptive Bull",
    "revolut-infinity": "Infinity",
    "revolut-london": "London Breakout",
    "revolut-dca": "DCA RSI v2",
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


# ── Évaluation passage live ───────────────────────────────────────────────────

def evaluate_bot(exchange: str) -> Optional[dict]:
    """Évalue si un bot est prêt pour le passage live.

    Retourne un dict avec les métriques et un champ 'go_live' (bool),
    ou None si pas de trades.
    """
    criteria = EVAL_CRITERIA.get(exchange)
    if not criteria:
        return None

    trades = _fetch_paper_trades()
    bot_trades = [t for t in trades if t.get("exchange") == exchange]
    closed = [t for t in bot_trades if t.get("status") == "CLOSED"]

    if not closed:
        return None

    pnl_list = []
    for t in closed:
        pnl = t.get("pnl_net_usd") or t.get("pnl_usd")
        if pnl is not None:
            pnl_list.append(float(pnl))

    wins = [p for p in pnl_list if p > 0]
    losses = [p for p in pnl_list if p <= 0]
    win_rate = len(wins) / len(pnl_list) * 100 if pnl_list else 0.0
    sum_wins = sum(wins)
    sum_losses = abs(sum(losses))
    pf = sum_wins / sum_losses if sum_losses > 0 else (999.0 if sum_wins > 0 else 0.0)
    total_pnl = sum(pnl_list)

    n_trades = len(pnl_list)
    ok_trades = n_trades >= criteria["min_trades"]
    ok_wr = win_rate >= criteria["min_wr_pct"]
    ok_pf = pf >= criteria["min_pf"]
    go_live = ok_trades and ok_wr and ok_pf

    return {
        "exchange": exchange,
        "label": criteria["label"],
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": pf,
        "total_pnl": total_pnl,
        "go_live": go_live,
        "ok_trades": ok_trades,
        "ok_wr": ok_wr,
        "ok_pf": ok_pf,
        "criteria": criteria,
    }


def generate_eval_message() -> str:
    """Génère le message Telegram d'évaluation passage live."""
    lines = []
    lines.append("🚦 *ÉVALUATION PAPER TRADING — PASSAGE LIVE ?*")
    lines.append(f"_{datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}_")
    lines.append("")

    any_result = False
    for exchange, criteria in EVAL_CRITERIA.items():
        result = evaluate_bot(exchange)
        if result is None:
            lines.append(f"*{criteria['label']}* : aucun trade — évaluation impossible")
            lines.append("")
            continue

        any_result = True
        verdict = "✅ *GO LIVE*" if result["go_live"] else "⏳ *ATTENDRE*"
        pf_str = f"{result['profit_factor']:.2f}" if result["profit_factor"] < 999 else "∞"
        c = result["criteria"]

        lines.append(f"*{result['label']}* — {verdict}")
        lines.append(
            f"  Trades: {'✅' if result['ok_trades'] else '❌'} "
            f"{result['n_trades']} / min {c['min_trades']}"
        )
        lines.append(
            f"  WR: {'✅' if result['ok_wr'] else '❌'} "
            f"{result['win_rate']:.1f}% / min {c['min_wr_pct']:.0f}%"
        )
        lines.append(
            f"  PF: {'✅' if result['ok_pf'] else '❌'} "
            f"{pf_str} / min {c['min_pf']:.1f}"
        )
        lines.append(f"  PnL cumulé: {'+' if result['total_pnl'] >= 0 else ''}${result['total_pnl']:.2f}")
        lines.append("")

    if not any_result:
        return ""

    lines.append("_Prochaine évaluation automatique le 15 juin._")
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

    last_eval_date: Optional[str] = None

    while running:
        now = datetime.now(timezone.utc)
        current_hour = now.hour
        today = now.strftime("%Y-%m-%d")

        # Rapport biquotidien
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

        # Évaluation mensuelle le jour configuré à 10h UTC
        if (
            now.day == EVAL_SCHEDULE_DAY
            and current_hour == EVAL_SCHEDULE_HOUR
            and today != last_eval_date
        ):
            logger.info("🚦 Évaluation passage live (%s)", today)
            try:
                msg = generate_eval_message()
                if msg:
                    telegram.send_raw(msg)
                    logger.info("🚦 Évaluation envoyée")
            except Exception as e:
                logger.error("🚦 Erreur évaluation: %s", e)
            last_eval_date = today

        time.sleep(60)  # Check every minute


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Trading Reporter")
    parser.add_argument("--once", action="store_true", help="Envoyer un rapport et quitter")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Envoyer l'évaluation passage live et quitter",
    )
    args = parser.parse_args()

    telegram = TelegramNotifier(
        bot_token=config.TELEGRAM_BOT_TOKEN,
        chat_id=config.TELEGRAM_CHAT_ID,
    )

    if args.evaluate:
        msg = generate_eval_message()
        if msg:
            telegram.send_raw(msg)
            print(msg)
        else:
            print("Aucun trade paper détecté.")
    elif args.once:
        report = generate_paper_report()
        if report:
            telegram.send_raw(report)
            print(report)
        else:
            print("Aucun trade paper détecté.")
    else:
        _run_scheduled()


if __name__ == "__main__":
    main()
