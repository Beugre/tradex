#!/usr/bin/env python3
"""Performance année par année avec compounding — $1,000 de départ."""
import csv
from datetime import datetime, timezone
from pathlib import Path
from backtest.run_backtest_adaptive import _run_adaptive_pair, _compute_metrics, PAIRS_BIG5
from src.core.models import Candle
import logging
logging.disable(logging.CRITICAL)

CACHE_DIR = Path(__file__).parent / "data"

# Fichiers couvrant tout le backtest (sans appel réseau)
# *_20200401_20260301 couvre Y1-Y4, *_20230409_20260408 couvre Y3-Y5
PAIR_SYM = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT",
    "XRP-USD": "XRPUSDT",
}

YEARS = [
    ("Y1", "2021-04-09", "2022-04-09"),
    ("Y2", "2022-04-09", "2023-04-09"),
    ("Y3", "2023-04-09", "2024-04-09"),
    ("Y4", "2024-04-09", "2025-04-09"),
    ("Y5", "2025-04-09", "2026-04-08"),
]

def parse(s):
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _load_csv_file(path):
    candles = []
    with open(path) as f:
        for row in csv.DictReader(f):
            candles.append(Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    return candles

def _filter(candles, start_ms, end_ms):
    return [c for c in candles if start_ms <= c.timestamp < end_ms]

def _best_file(bsym, interval, start_ms, end_ms):
    """Retourne le fichier cache le plus approprié qui couvre la plage."""
    candidates = sorted(CACHE_DIR.glob(f"{bsym}_{interval}_*.csv"))
    for p in candidates:
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        try:
            f_start = int(datetime.strptime(parts[-2], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            f_end   = int(datetime.strptime(parts[-1], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            if f_start <= start_ms and f_end >= end_ms:
                return p
        except ValueError:
            continue
    return None

# Pré-chargement des fichiers larges par paire (cache en mémoire)
_CACHE_MEM: dict = {}

def _get_candles(pair, interval, start_ms, end_ms):
    bsym = PAIR_SYM[pair]
    key = (bsym, interval)
    if key not in _CACHE_MEM:
        path = _best_file(bsym, interval, start_ms, end_ms)
        if path is None:
            # Essai avec le fichier le plus large disponible
            candidates = sorted(CACHE_DIR.glob(f"{bsym}_{interval}_*.csv"),
                                key=lambda p: p.stat().st_size, reverse=True)
            path = candidates[0] if candidates else None
        if path is None:
            return []
        _CACHE_MEM[key] = _load_csv_file(path)
    return _filter(_CACHE_MEM[key], start_ms, end_ms)

def run_year(start_str, end_str, capital):
    s_ms = int(parse(start_str).timestamp() * 1000)
    e_ms = int(parse(end_str).timestamp() * 1000)
    per_pair = capital / len(PAIRS_BIG5)
    all_trades = []
    combined_equity: list[float] = []
    total_final = 0.0
    n_pairs = 0
    for pair in PAIRS_BIG5:
        c15 = _get_candles(pair, "15m", s_ms, e_ms)
        c1h = _get_candles(pair, "1h",  s_ms, e_ms)
        if not c15 or not c1h:
            continue
        bal, trades, eq = _run_adaptive_pair(c15, c1h, per_pair)
        all_trades.extend(trades)
        total_final += bal
        n_pairs += 1
        if not combined_equity:
            combined_equity = list(eq)
        else:
            combined_equity = [a + b for a, b in zip(combined_equity, eq)]
    if not combined_equity or n_pairs == 0:
        return capital, {"n": 0, "wr": 0.0, "pf": 0.0, "dd": 0.0, "final": capital}
    m = _compute_metrics(all_trades, combined_equity, capital)
    m["final"] = total_final   # remplace equity[-1] par la vraie somme des balances
    return total_final, m

W = 116
print()
print("=" * W)
print("  PERFORMANCE ANNEE PAR ANNEE — Capital de depart : $1,000 (avec compounding)")
print("=" * W)
print(f"  {'Période':<22}  {'Capital début':>13}  {'Capital fin':>11}  {'PnL $':>9}  {'PnL %':>7}  {'PF':>5}  {'WR':>6}  {'DD max':>7}  {'Trades':>6}")
print(f"  {'-'*22}  {'-'*13}  {'-'*11}  {'-'*9}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}")

START_CAPITAL = 1000.0
results = []
capital = START_CAPITAL
for (label, s, e) in YEARS:
    cap_start = capital
    cap_end, m = run_year(s, e, capital)
    pnl = cap_end - cap_start
    pct = pnl / cap_start * 100
    results.append((label, s, e, cap_start, cap_end, pnl, pct, m))
    dd_str = f"{m['dd']:.1%}" if m["n"] > 0 else "n/a"
    print(f"  {label} {s[:7]}--> {e[:7]}  ${cap_start:>12,.2f}  ${cap_end:>10,.2f}  ${pnl:>+8,.2f}  {pct:>+6.1f}%  {m['pf']:>5.2f}  {m['wr']:>5.1%}  {dd_str:>7}  {m['n']:>6}")
    capital = cap_end

# ── Cumuls 2 par 2 ─────────────────────────────────────────────────────────
print()
print(f"  {'-'*W}")
print(f"  {'Cumulé':<26}  {'Capital début':>13}  {'Capital fin':>11}  {'PnL $':>9}  {'PnL %':>7}")
print(f"  {'-'*26}  {'-'*13}  {'-'*11}  {'-'*9}  {'-'*7}")
pairs2 = [("Y1+Y2", 0, 1), ("Y2+Y3", 1, 2), ("Y3+Y4", 2, 3), ("Y4+Y5", 3, 4)]
for (lbl, a, b) in pairs2:
    cap_a = results[a][3]
    cap_b = results[b][4]
    pnl = cap_b - cap_a
    pct = pnl / cap_a * 100
    label_fmt = f"{lbl} {results[a][1][:7]}-->{results[b][2][:7]}"
    print(f"  {label_fmt:<26}  ${cap_a:>12,.2f}  ${cap_b:>10,.2f}  ${pnl:>+8,.2f}  {pct:>+6.1f}%")

# ── Total 5 ans ─────────────────────────────────────────────────────────────
total_start = results[0][3]
total_end   = results[-1][4]
total_pnl   = total_end - total_start
total_pct   = total_pnl / total_start * 100
print()
print(f"  {'='*W}")
print(f"  {'TOTAL 5 ANS':<26}  ${total_start:>12,.2f}  ${total_end:>10,.2f}  ${total_pnl:>+8,.2f}  {total_pct:>+6.1f}%")
print(f"  {'='*W}")
print()
