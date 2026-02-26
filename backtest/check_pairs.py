"""Vérifie quelles paires ont 4 ans d'historique sur Binance."""
import httpx
import time
from datetime import datetime, timezone

pairs = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "LINK": "LINKUSDT", "SUI": "SUIUSDT", "ADA": "ADAUSDT", "DOT": "DOTUSDT",
    "AVAX": "AVAXUSDT", "DOGE": "DOGEUSDT", "ATOM": "ATOMUSDT", "UNI": "UNIUSDT",
    "NEAR": "NEARUSDT", "ALGO": "ALGOUSDT", "APE": "APEUSDT", "LTC": "LTCUSDT",
    "ETC": "ETCUSDT", "FIL": "FILUSDT", "AAVE": "AAVEUSDT", "ARB": "ARBUSDT",
    "OP": "OPUSDT", "INJ": "INJUSDT", "TIA": "TIAUSDT", "SEI": "SEIUSDT",
    "PEPE": "PEPEUSDT", "WIF": "WIFUSDT", "SAND": "SANDUSDT", "MANA": "MANAUSDT",
}

cutoff = datetime(2022, 2, 20, tzinfo=timezone.utc)
cutoff_ms = int(cutoff.timestamp() * 1000)

ok = []
too_recent = []

for name, sym in pairs.items():
    try:
        r = httpx.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": sym, "interval": "4h", "limit": 1, "startTime": 0},
            timeout=15,
        )
        klines = r.json()
        first_ts = int(klines[0][0])
        first_date = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)
        has_data = first_ts < cutoff_ms
        status = "OK" if has_data else "XX"
        print(f"  {status} {name:5s} ({sym}) — 1ere bougie: {first_date.date()}")
        if has_data:
            ok.append(name)
        else:
            too_recent.append((name, str(first_date.date())))
    except Exception as e:
        print(f"  !! {name} — erreur: {e}")
    time.sleep(0.3)

print(f"\nOK: {len(ok)} paires avec historique depuis fev 2022 : {ok}")
print(f"XX: {len(too_recent)} trop recentes : {too_recent}")
