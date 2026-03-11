#!/usr/bin/env python3
"""
Backfill des tags Firestore sur la collection `trades`.

Objectif:
- Harmoniser les tags par bot (`bot_id`, `bot_label`, `strategy_type`, `exchange_venue`)
- Corriger les documents où `exchange` est absent/incohérent
- Préserver l'ancien champ exchange via `exchange_raw` pour audit

Usage:
  python scripts/backfill_trade_tags.py          # dry-run
  python scripts/backfill_trade_tags.py --apply  # écrit réellement
"""

from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

from google.cloud import firestore


BOT_META = {
    "trail-range": {
        "bot_label": "Trail Range",
        "exchange_venue": "binance",
        "exchange": "binance",
    },
    "crashbot": {
        "bot_label": "CrashBot",
        "exchange_venue": "binance",
        "exchange": "binance-crashbot",
    },
    "infinity": {
        "bot_label": "Infinity",
        "exchange_venue": "revolut",
        "exchange": "revolut-infinity",
    },
    "london-breakout": {
        "bot_label": "London Breakout",
        "exchange_venue": "revolut",
        "exchange": "revolut-london",
    },
    "trend-legacy": {
        "bot_label": "Trend Legacy",
        "exchange_venue": "revolut",
        "exchange": "revolut",
    },
}


def classify_bot(doc: dict[str, Any]) -> str:
    bot_id = (doc.get("bot_id") or "").strip().lower()
    if bot_id in BOT_META:
        return bot_id

    strategy = (doc.get("strategy_type") or doc.get("signal_type") or "").strip().upper()
    exchange = (doc.get("exchange") or "").strip().lower()
    symbol = (doc.get("symbol") or "").strip().upper()

    if strategy == "CRASHBOT" or exchange == "binance-crashbot":
        return "crashbot"
    if strategy == "INFINITY" or exchange == "revolut-infinity":
        return "infinity"
    if strategy == "LONDON" or exchange == "revolut-london":
        return "london-breakout"

    if strategy == "RANGE":
        if symbol.endswith("USDC"):
            return "trail-range"
        if symbol.endswith("-USD"):
            return "london-breakout"

    if strategy == "TREND" or exchange == "revolut":
        return "trend-legacy"

    if exchange == "binance" or symbol.endswith("USDC"):
        return "trail-range"

    return "unknown"


def build_updates(doc: dict[str, Any]) -> dict[str, Any]:
    bot_id = classify_bot(doc)
    if bot_id == "unknown":
        return {}

    meta = BOT_META[bot_id]
    strategy_type = (doc.get("strategy_type") or doc.get("signal_type") or "").strip().upper()
    if not strategy_type:
        if bot_id == "trail-range":
            strategy_type = "RANGE"
        elif bot_id == "crashbot":
            strategy_type = "CRASHBOT"
        elif bot_id == "infinity":
            strategy_type = "INFINITY"
        elif bot_id == "london-breakout":
            strategy_type = "LONDON"
        else:
            strategy_type = "TREND"

    exchange_raw = doc.get("exchange")
    updates = {
        "bot_id": bot_id,
        "bot_label": meta["bot_label"],
        "exchange_venue": meta["exchange_venue"],
        "exchange": meta["exchange"],
        "exchange_raw": exchange_raw,
        "strategy_type": strategy_type,
    }

    changed = {}
    for key, value in updates.items():
        if doc.get(key) != value:
            changed[key] = value
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill tags Firestore trades")
    parser.add_argument("--apply", action="store_true", help="Applique les updates dans Firestore")
    args = parser.parse_args()

    db = firestore.Client()
    docs = list(db.collection("trades").stream())

    to_update: list[tuple[Any, dict[str, Any], str]] = []
    skipped = 0
    targets = Counter()

    for snap in docs:
        data = snap.to_dict() or {}
        updates = build_updates(data)
        if not updates:
            skipped += 1
            continue

        bot_id = updates.get("bot_id", data.get("bot_id", "unknown"))
        targets[bot_id] += 1
        to_update.append((snap.reference, updates, bot_id))

    print(f"Total docs trades: {len(docs)}")
    print(f"Docs à corriger   : {len(to_update)}")
    print(f"Docs ignorés      : {skipped}")
    print("Répartition corrections:")
    for bot, count in targets.most_common():
        print(f"  - {bot}: {count}")

    if not args.apply:
        print("\nMode DRY-RUN: aucune écriture effectuée. Relancer avec --apply pour appliquer.")
        return

    batch = db.batch()
    batch_size = 0
    committed = 0

    for ref, updates, _ in to_update:
        batch.update(ref, updates)
        batch_size += 1
        if batch_size >= 400:
            batch.commit()
            committed += batch_size
            batch = db.batch()
            batch_size = 0

    if batch_size > 0:
        batch.commit()
        committed += batch_size

    print(f"\n✅ Corrections appliquées: {committed}")


if __name__ == "__main__":
    main()
