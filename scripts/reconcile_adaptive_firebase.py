#!/usr/bin/env python3
"""
Réconciliation des trades "fantômes" Adaptive Bull dans Firebase.

Le bug dans log_trade_closed (UnboundLocalError) faisait que toutes les
clôtures de l'Adaptive Bull n'étaient pas écrites dans Firebase.
Ce script :
  1. Récupère les trades OPEN en Firebase pour bot_id=adaptive
  2. Vérifie lesquels ne sont plus dans le state_adaptive.json (= fermés)
  3. Interroge Binance pour retrouver le prix/raison de sortie réel
  4. Met à jour Firebase avec le statut CLOSED

Usage :
    python3 -m scripts.reconcile_adaptive_firebase
    python3 -m scripts.reconcile_adaptive_firebase --dry-run  # Affiche sans modifier
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── .env ──────────────────────────────────────────────────────────────────────
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env)
    except ImportError:
        pass

STATE_FILE = Path(__file__).parent.parent / "data" / "state_adaptive.json"


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _get_binance_fills(symbol: str, start_ts_ms: int) -> list[dict]:
    """Récupère les ordres SELL Binance pour un symbole depuis start_ts."""
    try:
        import hmac, hashlib, time
        import urllib.request
        api_key = os.environ.get("BINANCE_API_KEY", "")
        secret  = os.environ.get("BINANCE_SECRET_KEY", "")
        if not api_key or not secret:
            return []
        ts = int(time.time() * 1000)
        params = f"symbol={symbol}&startTime={start_ts_ms}&recvWindow=60000&timestamp={ts}"
        sig = hmac.new(secret.encode(), params.encode(), hashlib.sha256).hexdigest()
        url = f"https://api.binance.com/api/v3/allOrders?{params}&signature={sig}"
        req = urllib.request.Request(url, headers={"X-MBX-APIKEY": api_key})
        with urllib.request.urlopen(req, timeout=10) as r:
            orders = json.loads(r.read())
        # Garder uniquement les SELL FILLED ou PARTIALLY_FILLED
        return [o for o in orders if o.get("side") == "SELL" and o.get("status") in ("FILLED", "PARTIALLY_FILLED")]
    except Exception as e:
        print(f"    ⚠️  Binance query échouée pour {symbol}: {e}")
        return []


def _infer_exit_reason(exit_price: float, entry_price: float, sl_price: float, tp_price: float) -> str:
    """Détermine la raison de fermeture probable."""
    sl_dist = abs(exit_price - sl_price)
    tp_dist = abs(exit_price - tp_price)
    entry_dist = abs(exit_price - entry_price)
    if sl_dist < entry_dist * 0.05:
        return "SL_HIT"
    if tp_dist < entry_dist * 0.05:
        return "TP_HIT"
    if exit_price < entry_price:
        return "SL_HIT"
    return "TREND_BREAK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans modifier Firebase")
    args = parser.parse_args()

    from src.firebase.client import get_documents, update_document

    # 1. Firebase — tous les trades OPEN adaptive
    print("📥 Récupération des trades OPEN (adaptive)…")
    all_docs = get_documents("trades", filters=[("bot_id", "==", "adaptive")])
    open_docs = [d for d in all_docs if d.get("status") == "OPEN"]
    print(f"   {len(open_docs)} trade(s) OPEN trouvé(s)")

    if not open_docs:
        print("✅ Aucun trade fantôme à corriger.")
        return

    # 2. State actuel du bot
    state = _load_state()
    active_symbols = set(state.get("positions", {}).keys())
    print(f"   Positions actives dans state: {active_symbols or '(aucune)'}")

    now = datetime.now(timezone.utc)

    for doc in open_docs:
        symbol    = doc.get("symbol", "?")
        trade_id  = doc.get("trade_id") or doc.get("_id")
        entry     = doc.get("entry_filled") or doc.get("entry_expected", 0)
        sl        = doc.get("sl_price", 0)
        tp        = doc.get("tp_price", 0)
        size      = doc.get("size", 0)
        opened_at = doc.get("opened_at", "")

        print(f"\n  [{symbol}] trade_id={trade_id[:8] if trade_id else '?'} | entry={entry} | SL={sl} | TP={tp}")

        if symbol in active_symbols:
            print(f"    → Toujours actif dans le bot, skip")
            continue

        # Position fermée par le bot mais pas dans Firebase
        print(f"    → FANTÔME : position absente du state. Recherche clôture Binance…")

        # Convertir opened_at en timestamp ms
        start_ms = 0
        if opened_at:
            try:
                dt = datetime.fromisoformat(opened_at)
                start_ms = int(dt.timestamp() * 1000)
            except ValueError:
                pass

        sell_orders = _get_binance_fills(symbol, start_ms)
        print(f"    Ordres SELL Binance trouvés : {len(sell_orders)}")

        exit_price = None
        exit_reason = None
        exit_ts = None

        if sell_orders:
            # Prendre le dernier ordre SELL (le plus récent)
            last_sell = max(sell_orders, key=lambda o: o.get("time", 0))
            exit_price = float(last_sell.get("price") or last_sell.get("cummulativeQuoteQty", 0) / (float(last_sell.get("executedQty", 1)) or 1))
            exit_ts    = last_sell.get("time")
            exit_reason = _infer_exit_reason(exit_price, entry, sl, tp)
            print(f"    Exit trouvé: {exit_price} @ {datetime.fromtimestamp(exit_ts/1000, tz=timezone.utc).isoformat()}")
            print(f"    Raison probable: {exit_reason}")
        else:
            # Pas de data Binance → estimation par défaut (SL probable vu le contexte)
            exit_price  = sl * 0.999 if sl else entry * 0.985
            exit_reason = "SL_HIT_ESTIMATED"
            exit_ts     = None
            print(f"    ⚠️  Pas de data Binance → prix estimé SL: {exit_price:.4f}")

        # Calcul PnL brut
        pnl_gross   = (exit_price - entry) * size
        pnl_pct     = pnl_gross / (entry * size) if (entry * size) > 0 else 0
        # Fees taker Binance : 0.1%
        fee_entry   = entry * size * 0.001
        fee_exit    = exit_price * size * 0.001
        fees_total  = fee_entry + fee_exit
        pnl_net     = pnl_gross - fees_total
        pnl_net_pct = pnl_net / (entry * size) if (entry * size) > 0 else 0

        closed_at = (
            datetime.fromtimestamp(exit_ts / 1000, tz=timezone.utc).isoformat()
            if exit_ts else now.isoformat()
        )

        opened_dt = None
        if opened_at:
            try:
                opened_dt = datetime.fromisoformat(opened_at)
                if opened_dt.tzinfo is None:
                    opened_dt = opened_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        closed_dt = datetime.fromisoformat(closed_at)
        if closed_dt.tzinfo is None:
            closed_dt = closed_dt.replace(tzinfo=timezone.utc)
        holding_hours = round((closed_dt - opened_dt).total_seconds() / 3600, 2) if opened_dt else None

        updates = {
            "status": "CLOSED",
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "exit_fill_type": "taker",
            "holding_time_hours": holding_hours,
            "closed_at": closed_at,
            "actual_exit_size": size,
            "fees_entry": round(fee_entry, 4),
            "fees_exit": round(fee_exit, 4),
            "fees_total": round(fees_total, 4),
            "pnl_usd": round(pnl_gross, 4),
            "pnl_net_usd": round(pnl_net, 4),
            "pnl_pct": round(pnl_pct, 6),
            "pnl_net_pct": round(pnl_net_pct, 6),
            "equity_after": 0.0,
            "is_zero_risk_applied": False,
            "reconciled": True,
            "reconciled_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        emoji = "🟢" if pnl_net >= 0 else "🔴"
        print(f"    {emoji} PnL brut=${pnl_gross:+.2f} | net=${pnl_net:+.2f} ({pnl_net_pct*100:+.2f}%)")

        if args.dry_run:
            print(f"    [DRY-RUN] Mise à jour Firebase ignorée")
        else:
            if not trade_id:
                print(f"    ⚠️  Pas de trade_id — impossible de mettre à jour Firebase")
                continue
            ok = update_document("trades", trade_id, updates)
            if ok:
                print(f"    ✅ Firebase mis à jour → CLOSED")
            else:
                print(f"    ❌ Échec mise à jour Firebase")

    print("\n✅ Réconciliation terminée.")


if __name__ == "__main__":
    main()
