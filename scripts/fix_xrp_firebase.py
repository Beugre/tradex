"""Fix one-shot: clôturer le trade XRP dans Firebase (vendu manuellement)."""
import os, sys
os.chdir("/opt/tradex")
sys.path.insert(0, "/opt/tradex")
os.environ.setdefault("FIREBASE_CREDENTIALS_PATH", "/opt/tradex/firebase-credentials.json")

from src.firebase.client import update_document
from datetime import datetime, timezone

trade_id = "4b99450f-f316-4339-8ec8-46ff566dc82f"

# Données du trade
entry_price = 1.3799
size = 128.86585984491632
exit_price = 1.42  # prix approximatif de la vente manuelle

# Calculs
pnl_gross = (exit_price - entry_price) * size
notional = entry_price * size
pnl_pct = pnl_gross / notional * 100
# Taker fee (vente manuelle = probablement taker)
fee = exit_price * size * 0.0009
pnl_net = pnl_gross - fee

now = datetime.now(timezone.utc).isoformat()
opened_at = "2026-02-22T20:14:24+00:00"
holding_h = (datetime.fromisoformat(now) - datetime.fromisoformat(opened_at)).total_seconds() / 3600

update = {
    "status": "CLOSED",
    "exit_price": exit_price,
    "exit_reason": "MANUAL_CLOSE (Pocket INACTIVE API)",
    "exit_fill_type": "manual",
    "pnl_usd": round(pnl_net, 4),
    "pnl_percent": round(pnl_pct, 4),
    "pnl_gross": round(pnl_gross, 4),
    "fees_total": round(fee, 4),
    "closed_at": now,
    "close_blocked": False,
    "close_blocked_attempts": 3,
    "close_blocked_error": "Pocket INACTIVE - resolved manually",
    "holding_time_hours": round(holding_h, 1),
}

print(f"Entry: {entry_price} | Exit: {exit_price}")
print(f"PnL gross: ${pnl_gross:+.4f}")
print(f"Fee (taker): ${fee:.4f}")
print(f"PnL net: ${pnl_net:+.4f} ({pnl_pct:+.2f}%)")
print(f"Holding: {round(holding_h, 1)}h")

update_document("trades", trade_id, update)
print(f"\n✅ Trade {trade_id[:12]}.. mis à jour → CLOSED")
