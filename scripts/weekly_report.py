#!/usr/bin/env python3
"""
Rapport hebdomadaire de performance des bots TradeX.

Calcule pour chaque bot (dry-run ou live) sur la période sélectionnée :
- PnL net ($)
- Drawdown maximal (% de l'equity)
- Win Rate (%)
- Profit Factor (gains / pertes)

Usage :
    # Semaine dernière (lundi à dimanche)
    python -m scripts.weekly_report

    # Semaine spécifique
    python -m scripts.weekly_report --from 2025-06-02 --to 2025-06-08

    # Depuis le début
    python -m scripts.weekly_report --all

    # Un seul bot
    python -m scripts.weekly_report --bot crashbot

    # Format JSON (pour automatisation)
    python -m scripts.weekly_report --json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from google.cloud import firestore
from google.oauth2 import service_account

import os
cred_path = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(__file__), "..", "firebase-credentials.json"),
)
credentials = service_account.Credentials.from_service_account_file(cred_path)
db = firestore.Client(project=credentials.project_id, credentials=credentials)


# ── Bot registry ───────────────────────────────────────────────────────────────

BOT_META: dict[str, dict[str, str]] = {
    "trail-range":    {"label": "Trail Range",       "exchange": "binance"},
    "crashbot":       {"label": "CrashBot",          "exchange": "binance-crashbot"},
    "listing":        {"label": "Listing Event",     "exchange": "binance-listing"},
    "infinity":       {"label": "Infinity",          "exchange": "revolut-infinity"},
    "london-breakout":{"label": "London Breakout",   "exchange": "revolut-london"},
    "dca":            {"label": "DCA RSI",           "exchange": "revolut-dca"},
}


def classify_bot(trade: dict[str, Any]) -> str:
    """Classify a trade doc into a canonical bot_id."""
    bot_id = (trade.get("bot_id") or "").strip().lower()
    if bot_id and bot_id in BOT_META:
        return bot_id

    strategy = (trade.get("strategy_type") or trade.get("signal_type") or "").upper()
    exchange = (trade.get("exchange") or "").lower()

    if strategy == "CRASHBOT" or exchange == "binance-crashbot":
        return "crashbot"
    if strategy == "LISTING" or exchange == "binance-listing":
        return "listing"
    if strategy == "INFINITY" or exchange == "revolut-infinity":
        return "infinity"
    if strategy == "LONDON" or exchange == "revolut-london":
        return "london-breakout"
    if strategy == "DCA" or exchange == "revolut-dca":
        return "dca"
    if strategy == "RANGE" or exchange == "binance":
        return "trail-range"
    return "unknown"


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_trades(from_iso: str, to_iso: str) -> list[dict]:
    """Fetch all CLOSED trades in the date range."""
    docs = (
        db.collection("trades")
        .where("status", "==", "CLOSED")
        .where("closed_at", ">=", from_iso)
        .where("closed_at", "<=", to_iso)
        .stream()
    )
    trades = []
    for d in docs:
        t = d.to_dict() or {}
        t["_id"] = d.id
        trades.append(t)
    return trades


def fetch_all_closed_trades() -> list[dict]:
    """Fetch all CLOSED trades (no date filter)."""
    docs = db.collection("trades").where("status", "==", "CLOSED").stream()
    trades = []
    for d in docs:
        t = d.to_dict() or {}
        t["_id"] = d.id
        trades.append(t)
    return trades


def fetch_snapshots(exchange: str, from_iso: str, to_iso: str) -> list[dict]:
    """Fetch daily snapshots for an exchange in the date range."""
    docs = (
        db.collection("daily_snapshots")
        .where("exchange", "==", exchange)
        .where("date", ">=", from_iso[:10])
        .where("date", "<=", to_iso[:10])
        .order_by("date")
        .stream()
    )
    return [d.to_dict() or {} for d in docs]


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(trades: list[dict]) -> dict[str, Any]:
    """Compute performance metrics for a list of closed trades."""
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "pnl_gross": 0.0,
            "pnl_net": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_holding_hours": 0.0,
            "fees_total": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "dry_run_count": 0,
            "live_count": 0,
        }

    pnl_values = []
    win_pnls = []
    loss_pnls = []
    total_fees = 0.0
    total_holding = 0.0
    dry_run_count = 0
    live_count = 0

    for t in trades:
        pnl = float(t.get("pnl_net_usd") or t.get("pnl_usd") or 0)
        pnl_values.append(pnl)
        if pnl >= 0:
            win_pnls.append(pnl)
        else:
            loss_pnls.append(pnl)
        total_fees += float(t.get("fees_total") or 0)
        total_holding += float(t.get("holding_time_hours") or 0)
        if t.get("dry_run"):
            dry_run_count += 1
        else:
            live_count += 1

    total = len(pnl_values)
    sum_wins = sum(win_pnls) if win_pnls else 0
    sum_losses = abs(sum(loss_pnls)) if loss_pnls else 0
    pf = sum_wins / sum_losses if sum_losses > 0 else (float("inf") if sum_wins > 0 else 0.0)

    # Drawdown calculation (equity curve based on cumulative PnL)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0

    # Sort trades by closed_at for chronological order
    sorted_trades = sorted(trades, key=lambda x: str(x.get("closed_at", "")))
    equity_at_start = float(sorted_trades[0].get("equity_at_entry") or 0) if sorted_trades else 0

    for t in sorted_trades:
        pnl = float(t.get("pnl_net_usd") or t.get("pnl_usd") or 0)
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
            if equity_at_start > 0:
                max_dd_pct = dd / equity_at_start * 100

    return {
        "total_trades": total,
        "wins": len(win_pnls),
        "losses": len(loss_pnls),
        "win_rate": round(len(win_pnls) / total * 100, 1) if total > 0 else 0,
        "pnl_gross": round(sum(float(t.get("pnl_usd") or 0) for t in trades), 2),
        "pnl_net": round(sum(pnl_values), 2),
        "profit_factor": round(pf, 2) if pf != float("inf") else "∞",
        "avg_win": round(sum_wins / len(win_pnls), 2) if win_pnls else 0,
        "avg_loss": round(sum_losses / len(loss_pnls), 2) if loss_pnls else 0,
        "avg_holding_hours": round(total_holding / total, 1) if total > 0 else 0,
        "fees_total": round(total_fees, 2),
        "max_drawdown_usd": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "best_trade": round(max(pnl_values), 2) if pnl_values else 0,
        "worst_trade": round(min(pnl_values), 2) if pnl_values else 0,
        "dry_run_count": dry_run_count,
        "live_count": live_count,
    }


# ── Display ────────────────────────────────────────────────────────────────────

def print_report(
    bots_data: dict[str, dict],
    from_date: str,
    to_date: str,
    output_json: bool = False,
) -> None:
    """Print the weekly report to stdout."""
    if output_json:
        report = {
            "period": {"from": from_date, "to": to_date},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bots": bots_data,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return

    print()
    print("=" * 74)
    print(f"   📊 RAPPORT HEBDOMADAIRE TradeX — {from_date}  →  {to_date}")
    print("=" * 74)

    # Summary row
    total_pnl = 0.0
    total_trades = 0
    for bot_id, data in bots_data.items():
        m = data["metrics"]
        total_pnl += m["pnl_net"]
        total_trades += m["total_trades"]

    pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
    print(f"\n   {pnl_emoji}  PnL total: ${total_pnl:+.2f}  |  {total_trades} trades clôturés")
    print()

    for bot_id in sorted(bots_data.keys()):
        data = bots_data[bot_id]
        m = data["metrics"]
        meta = BOT_META.get(bot_id, {"label": bot_id})
        label = meta["label"]

        mode = ""
        if m["dry_run_count"] > 0 and m["live_count"] == 0:
            mode = " 🧪 DRY-RUN"
        elif m["live_count"] > 0 and m["dry_run_count"] == 0:
            mode = " 🔴 LIVE"
        elif m["dry_run_count"] > 0 and m["live_count"] > 0:
            mode = f" ⚠️ MIXED ({m['live_count']}L/{m['dry_run_count']}D)"

        print(f"   {'─' * 70}")
        print(f"   {label}{mode}")
        print(f"   {'─' * 70}")

        if m["total_trades"] == 0:
            print("     Aucun trade clôturé sur la période\n")
            continue

        # KPIs row
        pf_str = str(m["profit_factor"]) if isinstance(m["profit_factor"], str) else f"{m['profit_factor']:.2f}"
        pnl_emoji = "🟢" if m["pnl_net"] >= 0 else "🔴"

        print(f"     Trades : {m['total_trades']}  ({m['wins']}W / {m['losses']}L)")
        print(f"     {pnl_emoji} PnL net  : ${m['pnl_net']:+.2f}  (brut: ${m['pnl_gross']:+.2f}, fees: ${m['fees_total']:.2f})")
        print(f"     Win Rate     : {m['win_rate']}%")
        print(f"     Profit Factor: {pf_str}")
        print(f"     Max Drawdown : ${m['max_drawdown_usd']:.2f} ({m['max_drawdown_pct']:.1f}%)")
        print(f"     Gain moyen   : ${m['avg_win']:.2f}  |  Perte moy: ${m['avg_loss']:.2f}")
        print(f"     Best trade   : ${m['best_trade']:+.2f}  |  Worst: ${m['worst_trade']:+.2f}")
        print(f"     Durée moy    : {m['avg_holding_hours']:.1f}h")
        print()

    print("=" * 74)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Rapport hebdo TradeX")
    parser.add_argument("--from", dest="from_date", help="Date début (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", help="Date fin (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Tous les trades (since inception)")
    parser.add_argument("--bot", help="Filtrer par bot_id (ex: crashbot, trail-range)")
    parser.add_argument("--json", action="store_true", help="Sortie JSON")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)

    if args.all:
        from_date = "2024-01-01"
        to_date = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    elif args.from_date and args.to_date:
        from_date = args.from_date
        to_date = args.to_date
    else:
        # Default: last Monday to last Sunday
        today = now.date()
        # Find last Monday
        days_since_monday = today.weekday()  # 0 = Monday
        if days_since_monday == 0:
            # It's Monday → report for prev week
            last_monday = today - timedelta(days=7)
        else:
            last_monday = today - timedelta(days=days_since_monday)
        last_sunday = last_monday + timedelta(days=6)
        from_date = last_monday.isoformat()
        to_date = last_sunday.isoformat()

    from_iso = f"{from_date}T00:00:00+00:00"
    to_iso = f"{to_date}T23:59:59+00:00"

    if not args.json:
        print(f"\n   Chargement des trades... ({from_date} → {to_date})")

    if args.all:
        trades = fetch_all_closed_trades()
    else:
        trades = fetch_trades(from_iso, to_iso)

    if not args.json:
        print(f"   {len(trades)} trades CLOSED trouvés")

    # Group by bot
    by_bot: dict[str, list[dict]] = {}
    for t in trades:
        bot = classify_bot(t)
        if args.bot and bot != args.bot:
            continue
        by_bot.setdefault(bot, []).append(t)

    # Add empty entries for bots with no trades
    if not args.bot:
        for bot_id in BOT_META:
            if bot_id not in by_bot:
                by_bot[bot_id] = []

    # Compute metrics per bot
    bots_data: dict[str, dict] = {}
    for bot_id, bot_trades in sorted(by_bot.items()):
        if bot_id == "unknown":
            continue
        bots_data[bot_id] = {
            "metrics": compute_metrics(bot_trades),
        }

    print_report(bots_data, from_date, to_date, output_json=args.json)


if __name__ == "__main__":
    main()
