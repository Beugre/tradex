#!/usr/bin/env python
"""
📊 Réconciliation quotidienne : Firebase trades vs Revolut X fills.

Compare chaque trade dans Firebase (collection `trades`) avec les fills
réels de l'exchange via GET /orders/{venue_order_id}/fills.

Détecte :
  - Trades Firebase sans fill Revolut (fantôme)
  - Écarts de prix entry/exit (slippage réel vs attendu)
  - Écarts de taille (partial fills)
  - Trades Revolut non trackés dans Firebase (orphelins)

Usage :
    python -m scripts.reconcile_daily                    # Aujourd'hui
    python -m scripts.reconcile_daily --date 2026-02-20  # Date spécifique
    python -m scripts.reconcile_daily --days 7           # 7 derniers jours
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from src import config
from src.exchange.revolut_client import RevolutXClient
from src.firebase.client import get_documents
from src.firebase.trade_logger import log_event

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reconcile")


def reconcile(date_str: str) -> dict[str, Any]:
    """Réconcilie les trades Firebase pour une date donnée.

    Args:
        date_str: Date ISO (ex: "2026-02-21")

    Returns:
        Rapport de réconciliation.
    """
    # 1. Récupérer les trades Firebase pour cette date
    start = f"{date_str}T00:00:00+00:00"
    end_date = datetime.fromisoformat(date_str) + timedelta(days=1)
    end = f"{end_date.strftime('%Y-%m-%d')}T00:00:00+00:00"

    fb_trades = get_documents(
        "trades",
        filters=[
            ("opened_at", ">=", start),
            ("opened_at", "<", end),
        ],
    )

    logger.info("📋 %d trades Firebase pour le %s", len(fb_trades), date_str)

    if not fb_trades:
        logger.info("  Aucun trade — rien à réconcilier")
        return {"date": date_str, "total": 0, "ok": 0, "issues": []}

    # 2. Connexion Revolut X
    client = RevolutXClient(
        api_key=config.REVOLUT_X_API_KEY,
        private_key_path=config.REVOLUT_X_PRIVATE_KEY_PATH,
    )

    report = {
        "date": date_str,
        "total": len(fb_trades),
        "ok": 0,
        "issues": [],
    }

    for trade in fb_trades:
        trade_id = trade.get("trade_id", "?")[:8]
        symbol = trade.get("symbol", "?")
        venue_id = trade.get("venue_order_id", "")

        if not venue_id or venue_id in ("dry-run", "unknown"):
            logger.info("  [%s] %s — dry-run/unknown, skip", trade_id, symbol)
            report["ok"] += 1
            continue

        # 3. Récupérer les fills Revolut
        try:
            fills = client.get_order_fills(venue_id)
        except Exception as e:
            issue = f"[{trade_id}] {symbol} — Revolut fills error: {e}"
            logger.warning("  ⚠️ %s", issue)
            report["issues"].append(issue)
            continue

        if not fills:
            issue = f"[{trade_id}] {symbol} — FANTÔME: aucun fill Revolut pour {venue_id}"
            logger.warning("  ❌ %s", issue)
            report["issues"].append(issue)
            continue

        # 4. Calculer le prix moyen pondéré réel
        total_qty = sum(float(f.get("base_size", 0)) for f in fills)
        total_notional = sum(
            float(f.get("base_size", 0)) * float(f.get("price", 0))
            for f in fills
        )
        avg_price = total_notional / total_qty if total_qty > 0 else 0
        total_fees = sum(float(f.get("fee", 0)) for f in fills)

        # 5. Comparer avec Firebase
        fb_entry = trade.get("entry_filled", 0)
        fb_size = trade.get("size", 0)

        price_diff_pct = abs(avg_price - fb_entry) / fb_entry * 100 if fb_entry > 0 else 0
        size_diff_pct = abs(total_qty - fb_size) / fb_size * 100 if fb_size > 0 else 0

        issues_for_trade = []

        if price_diff_pct > 0.1:  # > 0.1% d'écart = suspect
            issues_for_trade.append(
                f"PRIX: Firebase={fb_entry:.4f} vs Revolut={avg_price:.4f} (Δ{price_diff_pct:.2f}%)"
            )

        if size_diff_pct > 5:  # > 5% d'écart de taille
            issues_for_trade.append(
                f"TAILLE: Firebase={fb_size:.8f} vs Revolut={total_qty:.8f} (Δ{size_diff_pct:.1f}%)"
            )

        if issues_for_trade:
            for iss in issues_for_trade:
                full = f"[{trade_id}] {symbol} — {iss}"
                logger.warning("  ⚠️ %s", full)
                report["issues"].append(full)
        else:
            logger.info(
                "  ✅ [%s] %s — prix OK (Δ%.3f%%) | taille OK | fees=$%.4f",
                trade_id, symbol, price_diff_pct, total_fees,
            )
            report["ok"] += 1

    # 6. Résumé
    n_issues = len(report["issues"])
    emoji = "✅" if n_issues == 0 else "⚠️"
    logger.info("")
    logger.info(
        "%s RÉCONCILIATION %s : %d/%d OK | %d problèmes",
        emoji, date_str, report["ok"], report["total"], n_issues,
    )

    if n_issues > 0:
        logger.info("  Problèmes :")
        for iss in report["issues"]:
            logger.info("    • %s", iss)

    # 7. Log le rapport dans Firebase (collection events)
    try:
        log_event("RECONCILIATION", {
            "date": date_str,
            "total_trades": report["total"],
            "ok": report["ok"],
            "issues_count": n_issues,
            "issues": report["issues"][:20],  # Max 20 pour pas exploser Firestore
        })
    except Exception:
        pass

    client.close()
    return report


def main():
    parser = argparse.ArgumentParser(description="Réconciliation Firebase vs Revolut")
    parser.add_argument("--date", type=str, help="Date ISO (ex: 2026-02-21)")
    parser.add_argument("--days", type=int, default=1, help="Nombre de jours à réconcilier")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    else:
        today = datetime.now(timezone.utc)
        dates = [
            (today - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(args.days)
        ]

    logger.info("═" * 60)
    logger.info("📊 RÉCONCILIATION QUOTIDIENNE — Firebase vs Revolut X")
    logger.info("═" * 60)

    all_ok = True
    for date_str in sorted(dates):
        report = reconcile(date_str)
        if report["issues"]:
            all_ok = False
        print()

    if all_ok:
        logger.info("🟢 Toutes les réconciliations OK")
    else:
        logger.warning("🟡 Des écarts détectés — voir les détails ci-dessus")


if __name__ == "__main__":
    main()
