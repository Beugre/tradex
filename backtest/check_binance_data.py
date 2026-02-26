#!/usr/bin/env python
"""Vérifie la disponibilité des données H4 Binance pour les 53 paires Revolut X."""
import httpx
import time
from datetime import datetime, timezone

REVOLUT_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD", "SUI-USD", "ADA-USD",
    "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD", "UNI-USD", "NEAR-USD", "ALGO-USD",
    "LTC-USD", "ETC-USD", "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
    "APE-USD", "SHIB-USD", "PEPE-USD", "ARB-USD", "OP-USD", "POL-USD", "GRT-USD",
    "COMP-USD", "SNX-USD", "CRV-USD", "LDO-USD", "VET-USD", "CHZ-USD", "BAT-USD",
    "HBAR-USD", "XTZ-USD", "DASH-USD", "XLM-USD", "TRX-USD", "KAVA-USD",
    "YFI-USD", "SUSHI-USD", "AXS-USD", "EGLD-USD", "RENDER-USD", "FET-USD",
    "BONK-USD", "WIF-USD", "SEI-USD", "TIA-USD", "FLOKI-USD", "JUP-USD",
]

# Mapping vers Binance
SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT", "LINK-USD": "LINKUSDT", "SUI-USD": "SUIUSDT",
    "ADA-USD": "ADAUSDT", "DOT-USD": "DOTUSDT", "AVAX-USD": "AVAXUSDT",
    "DOGE-USD": "DOGEUSDT", "ATOM-USD": "ATOMUSDT", "UNI-USD": "UNIUSDT",
    "NEAR-USD": "NEARUSDT", "ALGO-USD": "ALGOUSDT", "LTC-USD": "LTCUSDT",
    "ETC-USD": "ETCUSDT", "FIL-USD": "FILUSDT", "AAVE-USD": "AAVEUSDT",
    "INJ-USD": "INJUSDT", "SAND-USD": "SANDUSDT", "MANA-USD": "MANAUSDT",
    "APE-USD": "APEUSDT", "SHIB-USD": "SHIBUSDT", "PEPE-USD": "PEPEUSDT",
    "ARB-USD": "ARBUSDT", "OP-USD": "OPUSDT", "POL-USD": "POLUSDT",
    "GRT-USD": "GRTUSDT", "COMP-USD": "COMPUSDT", "SNX-USD": "SNXUSDT",
    "CRV-USD": "CRVUSDT", "LDO-USD": "LDOUSDT", "VET-USD": "VETUSDT",
    "CHZ-USD": "CHZUSDT", "BAT-USD": "BATUSDT", "HBAR-USD": "HBARUSDT",
    "XTZ-USD": "XTZUSDT", "DASH-USD": "DASHUSDT", "XLM-USD": "XLMUSDT",
    "TRX-USD": "TRXUSDT", "KAVA-USD": "KAVAUSDT", "YFI-USD": "YFIUSDT",
    "SUSHI-USD": "SUSHIUSDT", "AXS-USD": "AXSUSDT", "EGLD-USD": "EGLDUSDT",
    "RENDER-USD": "RENDERUSDT", "FET-USD": "FETUSDT", "BONK-USD": "BONKUSDT",
    "WIF-USD": "WIFUSDT", "SEI-USD": "SEIUSDT", "TIA-USD": "TIAUSDT",
    "FLOKI-USD": "FLOKIUSDT", "JUP-USD": "JUPUSDT",
}

cutoff_6y = datetime(2020, 2, 20, tzinfo=timezone.utc)
cutoff_4y = datetime(2022, 2, 20, tzinfo=timezone.utc)
cutoff_2y = datetime(2024, 2, 20, tzinfo=timezone.utc)

results = []

for pair in REVOLUT_PAIRS:
    bsym = SYMBOL_MAP.get(pair)
    if not bsym:
        print(f"  ?? {pair} — pas de mapping Binance")
        continue
    try:
        r = httpx.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": bsym, "interval": "4h", "limit": 1, "startTime": 0},
            timeout=15,
        )
        if r.status_code != 200:
            print(f"  ❌ {pair} ({bsym}) — Binance {r.status_code}")
            results.append((pair, None, "no_data"))
            time.sleep(0.2)
            continue
        klines = r.json()
        first_ts = int(klines[0][0])
        first_date = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc)

        if first_ts < int(cutoff_6y.timestamp() * 1000):
            cat = "6yr+"
        elif first_ts < int(cutoff_4y.timestamp() * 1000):
            cat = "4yr+"
        elif first_ts < int(cutoff_2y.timestamp() * 1000):
            cat = "2yr+"
        else:
            cat = "<2yr"

        results.append((pair, first_date, cat))
        print(f"  {cat:5s} {pair:<14s} ({bsym}) — 1ère bougie: {first_date.date()}")
    except Exception as e:
        print(f"  ❌ {pair} — {e}")
        results.append((pair, None, "error"))
    time.sleep(0.2)

# Résumé
print(f"\n{'='*70}")
pairs_6y = [p for p, _, c in results if c == "6yr+"]
pairs_4y = [p for p, _, c in results if c == "4yr+"]
pairs_2y = [p for p, _, c in results if c == "2yr+"]
pairs_lt2y = [p for p, _, c in results if c == "<2yr"]
pairs_nodata = [p for p, _, c in results if c in ("no_data", "error")]

print(f"\n6yr+ (depuis avant 2020-02-20) : {len(pairs_6y)} paires")
for p in pairs_6y:
    print(f"  {p}")

print(f"\n4yr+ (2020-2022) : {len(pairs_4y)} paires")
for p in pairs_4y:
    print(f"  {p}")

print(f"\n2yr+ (2022-2024) : {len(pairs_2y)} paires")
for p in pairs_2y:
    print(f"  {p}")

print(f"\n<2yr (après 2024) : {len(pairs_lt2y)} paires")
for p in pairs_lt2y:
    print(f"  {p}")

if pairs_nodata:
    print(f"\nPas de données Binance : {len(pairs_nodata)}")
    for p in pairs_nodata:
        print(f"  {p}")

# Total backtestable
bt_all = pairs_6y + pairs_4y + pairs_2y + pairs_lt2y
print(f"\n{'='*70}")
print(f"TOTAL backtestable : {len(bt_all)} paires")
print(f"  6yr+ : {len(pairs_6y)} | 4yr+ : {len(pairs_4y)} | 2yr+ : {len(pairs_2y)} | <2yr : {len(pairs_lt2y)}")
