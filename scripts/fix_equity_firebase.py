#!/usr/bin/env python3
"""
Fix des equity historiques dans Firebase.

Problème : equity_at_entry et equity_after étaient calculées avec USD.available
seulement (pas USD + crypto). On recalcule à partir du PnL cumulé en partant
d'un capital initial de $1158.

Méthode : on trie les trades fermés par date de clôture, et on reconstitue
l'equity après chaque trade = equity_avant + pnl_net du trade.
"""

import os
os.chdir("/opt/tradex")
os.environ.setdefault("FIREBASE_CREDENTIALS_PATH", "/opt/tradex/firebase-credentials.json")

from google.cloud import firestore
from google.oauth2 import service_account

INITIAL_EQUITY = 1158.0

cred = service_account.Credentials.from_service_account_file(
    "/opt/tradex/firebase-credentials.json"
)
db = firestore.Client(project=cred.project_id, credentials=cred)

# Récupérer tous les trades
trades_ref = db.collection("trades")
all_trades = list(trades_ref.stream())

# Séparer fermés et ouverts
closed_trades = []
open_trades = []
for doc in all_trades:
    d = doc.to_dict()
    d["_doc_id"] = doc.id
    if d.get("status") == "CLOSED" and d.get("pnl_usd") is not None:
        closed_trades.append(d)
    else:
        open_trades.append(d)

# Trier les fermés par closed_at
closed_trades.sort(key=lambda t: t.get("closed_at", ""))

print(f"Capital initial: ${INITIAL_EQUITY:.2f}")
print(f"Trades fermés: {len(closed_trades)}")
print(f"Trades ouverts: {len(open_trades)}")
print()

# Reconstituer l'equity trade par trade
equity = INITIAL_EQUITY
for t in closed_trades:
    sym = t.get("symbol", "?")
    pnl = t.get("pnl_usd", 0)
    doc_id = t["_doc_id"]
    
    equity_at_entry = equity
    equity_after = equity + pnl
    
    old_eq_entry = t.get("equity_at_entry")
    old_eq_after = t.get("equity_after")
    
    print(f"  {sym:12s} | PnL=${pnl:+.4f} | "
          f"eq_entry: {old_eq_entry} → {equity_at_entry:.2f} | "
          f"eq_after: {old_eq_after} → {equity_after:.2f}")
    
    # Mettre à jour Firebase
    trades_ref.document(doc_id).update({
        "equity_at_entry": round(equity_at_entry, 2),
        "equity_after": round(equity_after, 2),
    })
    
    equity = equity_after

print(f"\n✅ Equity finale après {len(closed_trades)} trades: ${equity:.2f}")

# Mettre à jour les trades ouverts avec l'equity courante
for t in open_trades:
    sym = t.get("symbol", "?")
    doc_id = t["_doc_id"]
    old_eq = t.get("equity_at_entry")
    
    # L'equity à l'entrée d'un trade ouvert = equity au moment où il a été ouvert
    # On ne peut pas le reconstituer exactement sans l'ordre d'ouverture
    # Mais on sait que les 3 ouverts ont été ouverts APRÈS les fermés
    # On met l'equity courante comme approximation
    print(f"  {sym:12s} (OPEN) | eq_entry: {old_eq} → {equity:.2f}")
    trades_ref.document(doc_id).update({
        "equity_at_entry": round(equity, 2),
    })

print(f"\n✅ Tous les trades mis à jour.")
