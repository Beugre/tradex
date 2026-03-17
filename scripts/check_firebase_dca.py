#!/usr/bin/env python3
"""Quick check of DCA data in Firebase."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from google.cloud import firestore
from google.oauth2 import service_account

cred_path = os.path.join(os.path.dirname(__file__), "..", "firebase-credentials.json")
credentials = service_account.Credentials.from_service_account_file(cred_path)
db = firestore.Client(project=credentials.project_id, credentials=credentials)

# Check DCA_HEARTBEAT events
hb = list(
    db.collection("events")
    .where("event_type", "==", "DCA_HEARTBEAT")
    .where("exchange", "==", "revolut-dca")
    .limit(3)
    .stream()
)
print(f"DCA_HEARTBEAT count: {len(hb)}")
for d in hb:
    doc = d.to_dict()
    print(f"  ts={doc.get('timestamp')} data keys={list(doc.get('data', {}).keys())}")

# Check DCA_BUY events
buys = list(
    db.collection("events")
    .where("event_type", "==", "DCA_BUY")
    .where("exchange", "==", "revolut-dca")
    .limit(20)
    .stream()
)
print(f"\nDCA_BUY count: {len(buys)}")
for d in buys:
    doc = d.to_dict()
    data = doc.get("data", {})
    ts = doc.get("timestamp", "")
    print(f"  id={d.id} ts={ts} reason={data.get('reason')} amount=${data.get('amount_usd')}")

# Delete duplicate events (bot logged 2 before my 5 manual inserts)
dupes = [d for d in buys if ".97" in str(doc.get("timestamp", "")) or ".09" in str(doc.get("timestamp", ""))]
# Better: delete events with microsecond timestamps (bot-logged) that overlap with manual inserts
to_delete = []
for d in buys:
    ts = str(d.to_dict().get("timestamp", ""))
    if "T09:" in ts:  # Bot-logged events from first (buggy) run
        to_delete.append(d.id)
        print(f"  -> DELETE duplicate: {d.id} ts={ts}")

for doc_id in to_delete:
    db.collection("events").document(doc_id).delete()
    print(f"  -> Deleted {doc_id}")

print(f"\nDeleted {len(to_delete)} duplicates")
