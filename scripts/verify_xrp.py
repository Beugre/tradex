"""Vérification du trade XRP dans Firebase."""
import os, sys
os.chdir("/opt/tradex")
sys.path.insert(0, "/opt/tradex")
os.environ.setdefault("FIREBASE_CREDENTIALS_PATH", "/opt/tradex/firebase-credentials.json")

from src.firebase.client import get_documents

trades = get_documents("trades", filters=[("symbol", "==", "XRP-USD")])
print(f"Found {len(trades)} XRP trades")
for t in trades:
    if t is None:
        continue
    print(f"  status:      {t.get('status')}")
    print(f"  pnl_usd:     {t.get('pnl_usd')}")
    print(f"  exit_price:  {t.get('exit_price')}")
    print(f"  exit_reason: {t.get('exit_reason')}")
    print(f"  closed_at:   {t.get('closed_at')}")
