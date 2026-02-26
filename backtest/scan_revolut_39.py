#!/usr/bin/env python
"""Scan Revolut X (auth) pour identifier toutes les paires USD disponibles."""
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src import config
from src.exchange.revolut_client import RevolutXClient

candidates = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD", "SUI-USD", "ADA-USD",
    "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD", "UNI-USD", "NEAR-USD", "ALGO-USD",
    "LTC-USD", "ETC-USD", "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
    "APE-USD", "SHIB-USD", "PEPE-USD", "ARB-USD", "OP-USD", "POL-USD", "GRT-USD",
    "COMP-USD", "SNX-USD", "CRV-USD", "LDO-USD", "VET-USD", "CHZ-USD", "ENJ-USD",
    "BAT-USD", "HBAR-USD", "XTZ-USD", "DASH-USD", "XLM-USD", "TRX-USD", "KAVA-USD",
    "MATIC-USD", "THETA-USD", "FTM-USD", "ZIL-USD", "ICX-USD", "ONE-USD", "IOTA-USD",
    "EOS-USD", "NEO-USD", "ZEC-USD", "WAVES-USD", "MKR-USD", "YFI-USD", "SUSHI-USD",
    "AXS-USD", "EGLD-USD", "RENDER-USD", "FET-USD", "BONK-USD", "WIF-USD",
    "SEI-USD", "TIA-USD", "FLOKI-USD", "JUP-USD",
]

client = RevolutXClient(
    api_key=config.REVOLUT_X_API_KEY,
    private_key_path=config.REVOLUT_X_PRIVATE_KEY_PATH,
)

ok = []
ko = []

for sym in candidates:
    try:
        candles = client.get_candles(symbol=sym, interval=240)
        if candles and len(candles) > 0:
            ok.append(sym)
            print(f"  ✅ {sym} ({len(candles)} bougies)")
        else:
            ko.append((sym, "0 bougies"))
            print(f"  ❌ {sym} (0 bougies)")
    except Exception as e:
        err = str(e)[:60]
        ko.append((sym, err))
        print(f"  ❌ {sym} ({err})")
    time.sleep(0.15)

print(f"\n{'='*60}")
print(f"Disponibles sur Revolut X: {len(ok)} paires")
print(f"{'='*60}")
for p in ok:
    print(f"  {p}")

print(f"\nIndisponibles: {len(ko)}")
for p, reason in ko:
    print(f"  {p} ({reason})")

# Format pour copier-coller
print(f"\n# Python list:")
print(f"REVOLUT_PAIRS = {ok}")
