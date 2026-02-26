"""Vérification pré-trading : solde, paramètres, et simulation."""
import httpx
import base64
import time
from cryptography.hazmat.primitives.serialization import load_pem_private_key

API_KEY = "4fiN7hPyMdirdTo1U4Xo1idypVDkNkxdnGbQHsvS7uKGmrdUZiuGuDtQi7MSakDV"
BASE_URL = "https://revx.revolut.com/api/1.0"

with open("private.pem", "rb") as f:
    private_key = load_pem_private_key(f.read(), password=None)


def signed_get(path):
    ts = str(int(time.time() * 1000))
    msg = f"{ts}GET/api/1.0{path}"
    sig = base64.b64encode(private_key.sign(msg.encode())).decode()
    r = httpx.get(
        f"{BASE_URL}{path}",
        headers={
            "X-Revx-API-Key": API_KEY,
            "X-Revx-Timestamp": ts,
            "X-Revx-Signature": sig,
        },
    )
    return r.json()


# 1. Solde USD
print("=" * 60)
print("💰 SOLDE DU COMPTE")
print("=" * 60)
balances = signed_get("/balances")
usd_balance = 0
for b in balances:
    available = float(b["available"])
    if available > 0:
        print(f"  {b['currency']}: {available:.6f} (réservé: {b['reserved']})")
    if b["currency"] == "USD":
        usd_balance = available

print(f"\n  → Solde USD disponible: ${usd_balance:.2f}")

# 2. Paramètres de risque
risk_pct = 0.05
risk_amount = usd_balance * risk_pct
print(f"\n{'=' * 60}")
print(f"⚙️  PARAMÈTRES DE TRADING")
print(f"{'=' * 60}")
print(f"  Risque par trade: {risk_pct*100:.0f}% = ${risk_amount:.2f}")
print(f"  Paires: BTC-USD, ETH-USD, SOL-USD, XRP-USD")
print(f"  Timeframe: H4")

# 3. Prix actuels et taille de position estimée
print(f"\n{'=' * 60}")
print(f"📊 PRIX ACTUELS & TAILLES DE POSITION ESTIMÉES")
print(f"{'=' * 60}")

pairs = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
for pair in pairs:
    ts = str(int(time.time() * 1000))
    msg = f"{ts}GET/api/1.0/tickers"
    sig = base64.b64encode(private_key.sign(msg.encode())).decode()
    r = httpx.get(
        f"{BASE_URL}/tickers",
        headers={
            "X-Revx-API-Key": API_KEY,
            "X-Revx-Timestamp": ts,
            "X-Revx-Signature": sig,
        },
    )
    tickers = r.json().get("data", [])
    for t in tickers:
        sym = t["symbol"].replace("/", "-")
        if sym == pair:
            price = float(t["last_price"])
            # Estimation SL à 3% du prix (typique pour H4)
            sl_distance = price * 0.03
            position_size = risk_amount / sl_distance
            position_value = position_size * price
            print(f"\n  {pair}:")
            print(f"    Prix: ${price:,.2f}")
            print(f"    SL estimé (~3%): ${sl_distance:,.2f}")
            print(f"    Taille position: {position_size:.6f} {pair.split('-')[0]}")
            print(f"    Valeur position: ${position_value:,.2f}")
            break

print(f"\n{'=' * 60}")
print(f"⚠️  RÉSUMÉ")
print(f"{'=' * 60}")
print(f"  Solde: ${usd_balance:.2f}")
print(f"  Risque max par trade: ${risk_amount:.2f}")
print(f"  Perte max si SL touché: ${risk_amount:.2f} (= {risk_pct*100:.0f}% du solde)")
