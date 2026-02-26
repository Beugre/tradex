"""Trouve toutes les paires USDT Binance avec historique depuis 2020."""
import httpx
import time
from datetime import datetime, timezone

# Candidats : top 50 crypto par capitalisation/liquidité
CANDIDATES = [
    "BTC", "ETH", "SOL", "XRP", "LINK", "ADA", "DOT", "AVAX", "DOGE", "ATOM",
    "UNI", "NEAR", "ALGO", "LTC", "ETC", "FIL", "AAVE", "INJ", "SAND", "MANA",
    # Nouvelles candidates
    "MATIC", "VET", "THETA", "FTM", "GRT", "AXS", "CHZ", "ENJ", "BAT", "ZIL",
    "ICX", "ONE", "HBAR", "EGLD", "COMP", "SNX", "SUSHI", "YFI", "CRV", "MKR",
    "IOTA", "XTZ", "EOS", "NEO", "DASH", "ZEC", "XLM", "TRX", "WAVES", "KAVA",
]

cutoff = datetime(2020, 2, 20, tzinfo=timezone.utc)
cutoff_ms = int(cutoff.timestamp() * 1000)

ok = []
too_recent = []

for name in CANDIDATES:
    sym = f"{name}USDT"
    try:
        r = httpx.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": sym, "interval": "4h", "limit": 1, "startTime": 0},
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  -- {name:6s} ({sym}) — pas disponible")
            time.sleep(0.2)
            continue
        klines = r.json()
        first_ts = int(klines[0][0])
        first_date = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
        has_data = first_ts < cutoff_ms
        status = "OK" if has_data else "XX"
        print(f"  {status} {name:6s} ({sym}) — 1ere bougie: {first_date.date()}")
        if has_data:
            ok.append(name)
        else:
            too_recent.append((name, str(first_date.date())))
    except Exception as e:
        print(f"  !! {name} — erreur: {e}")
    time.sleep(0.3)

print(f"\nOK: {len(ok)} paires avec historique depuis fev 2020:")
for i, name in enumerate(ok):
    print(f"  {i+1:2d}. {name}-USD")
print(f"\nXX: {len(too_recent)} trop recentes: {too_recent}")
