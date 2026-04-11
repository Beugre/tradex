#!/usr/bin/env python3
"""
Analyse BULL pur ×2 sur 6 ans — 2020→2026
  • Une à une   : chaque année (Y1…Y6) depuis $1 000
  • Deux à deux : fenêtres glissantes 2 ans → 5 résultats
  • Trois à trois : fenêtres glissantes 3 ans → 4 résultats
  • Total       : une seule simulation continue 6 ans
"""
import csv, logging
from datetime import datetime, timezone
from pathlib import Path
from src.core.models import Candle

logging.disable(logging.CRITICAL)

from backtest.run_backtest_adaptive import (
    _run_adaptive_pair, _compute_metrics, PAIRS_BIG5,
)

CACHE_DIR = Path(__file__).parent / "data"
PAIR_SYM  = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT", "BNB-USD": "BNBUSDT", "XRP-USD": "XRPUSDT",
}

# ─── Config BULL pur ×2 ────────────────────────────────────────────────────
BULL_KWARGS = dict(
    bull_only        = True,
    bull_alloc_pct   = 0.99,    # 50% × ×2 → 100%, cap 99%
    bull_trail_pct   = 0.025,   # trailing -2.5%
    bull_tp_pct      = 0.080,   # TP +8%
    bull_sl_pct      = 0.015,   # SL -1.5%
    bull_pyramid_alloc = 0.15,  # pyramiding +15%
    slippage_pct     = 0.0005,  # 0.05%/side
    entry_fee        = 0.001,
    exit_fee         = 0.001,
    # (autres params : valeurs par défaut de _run_adaptive_pair)
)

# 6 années
YEARS = [
    ("Y1 2020-21", "2020-04-09", "2021-04-09"),
    ("Y2 2021-22", "2021-04-09", "2022-04-09"),
    ("Y3 2022-23", "2022-04-09", "2023-04-09"),
    ("Y4 2023-24", "2023-04-09", "2024-04-09"),
    ("Y5 2024-25", "2024-04-09", "2025-04-09"),
    ("Y6 2025-26", "2025-04-09", "2026-04-08"),
]

def _ts(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)

# ─── Chargement intelligent du cache ──────────────────────────────────────
_FILE_CACHE: dict[str, list[Candle]] = {}

def _load_csv(path: Path) -> list[Candle]:
    if str(path) in _FILE_CACHE:
        return _FILE_CACHE[str(path)]
    candles = []
    with open(path) as f:
        for row in csv.DictReader(f):
            candles.append(Candle(
                timestamp=int(row["timestamp"]),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]),  close=float(row["close"]),
                volume=float(row["volume"]),
            ))
    _FILE_CACHE[str(path)] = candles
    return candles

def _best_file(bsym: str, interval: str, s_ms: int, e_ms: int):
    """Retourne le plus petit fichier qui couvre [s_ms, e_ms], sinon le plus grand."""
    all_files = sorted(CACHE_DIR.glob(f"{bsym}_{interval}_*.csv"),
                       key=lambda p: p.stat().st_size)
    covering = []
    for path in all_files:
        parts = path.stem.rsplit("_", 2)
        if len(parts) < 3:
            continue
        try:
            f_s = int(datetime.strptime(parts[-2], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
            f_e = int(datetime.strptime(parts[-1], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
        if f_s <= s_ms and f_e >= e_ms:
            covering.append(path)
    if covering:
        return covering[0]   # le plus petit qui couvre entièrement
    # Fallback : le plus grand fichier disponible
    return sorted(CACHE_DIR.glob(f"{bsym}_{interval}_*.csv"),
                  key=lambda p: p.stat().st_size, reverse=True)[0] if all_files else None

def _get(pair: str, interval: str, s_ms: int, e_ms: int) -> list[Candle]:
    bsym = PAIR_SYM[pair]
    path = _best_file(bsym, interval, s_ms, e_ms)
    if not path:
        return []
    candles = _load_csv(path)
    return [c for c in candles if s_ms <= c.timestamp < e_ms]

# ─── Simulation d'une fenêtre ─────────────────────────────────────────────
def run_window(label: str, start: str, end: str, initial: float = 1000.0) -> dict:
    s_ms, e_ms = _ts(start), _ts(end)
    per_pair   = initial / len(PAIRS_BIG5)
    all_trades, combined_eq, total_final = [], [], 0.0
    pairs_ok = 0

    for pair in PAIRS_BIG5:
        c15 = _get(pair, "15m", s_ms, e_ms)
        c1h = _get(pair, "1h",  s_ms, e_ms)
        if not c15 or not c1h:
            continue
        bal, trades, eq = _run_adaptive_pair(c15, c1h, per_pair, **BULL_KWARGS)
        all_trades.extend(trades)
        total_final += bal
        pairs_ok    += 1
        if not combined_eq:
            combined_eq = list(eq)
        else:
            combined_eq = [a + b for a, b in zip(combined_eq, eq)]

    if not combined_eq or pairs_ok == 0:
        return {"label": label, "start": start, "end": end,
                "initial": initial, "final": initial, "n": 0, "wr": 0.0,
                "pf": 0.0, "dd": 0.0, "pnl": 0.0, "pct": 0.0}

    m = _compute_metrics(all_trades, combined_eq, initial)
    m["final"] = total_final
    pnl = total_final - initial
    return {
        "label": label, "start": start, "end": end,
        "initial": initial, "final": total_final,
        "n": m["n"], "wr": m["wr"], "pf": m["pf"], "dd": m["dd"],
        "pnl": pnl, "pct": pnl / initial * 100,
    }

W = 120
SEP = "═" * W

def print_config():
    print(f"\n{'▀'*W}")
    print("  CONFIG — BULL PUR ×2")
    print(f"{'▀'*W}")
    rows = [
        ("Régimes actifs",   "BULL uniquement  (RANGE + STAGNATION désactivés)"),
        ("bull_alloc_pct",   "99%  (= 50% × ×2, plafonné)  → trade quasiment tout le capital disponible"),
        ("bull_sl_pct",      "-1.5%  (stop-loss fixe, close-only)"),
        ("bull_trail_pct",   "-2.5%  du peak  → trailing stop, laisse courir les gagnants"),
        ("bull_tp_pct",      "+8.0%  → R:R ≈ 5.3"),
        ("bull_pyramid_alloc","15%  de la position en pyramiding sur les positions gagnantes"),
        ("bull_rsi_min/max", "50 – 65  (filtre : marché haussier non suracheté)"),
        ("bull_slope_bars",  "10 bougies × 15m = 2.5h  (EMA20 doit monter)"),
        ("slippage",         "0.05%/côté  (entrée + sortie)"),
        ("fees",             "0.1% entrée + 0.1% sortie = 0.2% round-trip"),
        ("max_positions",    "2 simultanées par paire"),
        ("cooldown",         "16 bougies × 15m = 4h après une perte"),
        ("daily_dd_max",     "5%  (circuit-breaker journalier)"),
        ("Paires",           "BTC · ETH · SOL · BNB · XRP  (capital divisé en 5 parts égales)"),
        ("Capital/paire",    "$200 (pour $1 000 initial)"),
    ]
    for k, v in rows:
        print(f"  {k:<22} {v}")
    print(f"{'▄'*W}")

def print_section(title: str, results: list[dict]):
    print(f"\n\n{'─'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")
    print(f"  {'Période':<24}  {'Initial':>9}  {'Final':>9}  {'PnL $':>9}  {'PnL %':>8}  {'PF':>5}  {'WR':>6}  {'DD max':>7}  {'Trades':>6}")
    print(f"  {'-'*24}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}")
    for r in results:
        arrow = "  ← PERTE" if r["pnl"] < 0 else ""
        pf_s  = f"{r['pf']:.2f}" if r["n"] > 0 else " n/a"
        wr_s  = f"{r['wr']:.1%}" if r["n"] > 0 else "  n/a"
        dd_s  = f"{r['dd']:.1%}" if r["n"] > 0 else "  n/a"
        print(f"  {r['label']:<24}  ${r['initial']:>8,.0f}  ${r['final']:>8,.2f}  ${r['pnl']:>+8,.2f}  {r['pct']:>+7.1f}%  {pf_s:>5}  {wr_s:>6}  {dd_s:>7}  {r['n']:>6}{arrow}")

# ══════════════════════════════════════════════════════════════════════════
print_config()
print(f"\n\n{'█'*W}")
print("  ANALYSE BULL PUR ×2 — 6 ANS  (2020-04-09 → 2026-04-08)")
print(f"{'█'*W}")

# ─── UNE À UNE ─────────────────────────────────────────────────────────────
print("\n\n" + SEP)
print("  1.  UNE À UNE  — chaque année depuis $1 000")
print(SEP)
une_a_une = [run_window(label, s, e) for label, s, e in YEARS]
print_section("Année par année", une_a_une)

wins = sum(1 for r in une_a_une if r["pnl"] > 0)
total_pf_num = sum(r["pnl"] for r in une_a_une if r["pnl"] > 0)
total_pf_den = abs(sum(r["pnl"] for r in une_a_une if r["pnl"] < 0)) or 1e-9
print(f"\n  Bilan : {wins}/6 années gagnantes  |  PF inter-années = {total_pf_num/total_pf_den:.2f}")
print(SEP)

# ─── DEUX À DEUX ───────────────────────────────────────────────────────────
print("\n\n" + SEP)
print("  2.  DEUX À DEUX  — fenêtres glissantes, compounding, depuis $1 000")
print(SEP)
windows_2 = [
    (f"{YEARS[i][0]}–{YEARS[i+1][0][-2:]}", YEARS[i][1], YEARS[i+1][2])
    for i in range(len(YEARS) - 1)
]
deux_a_deux = [run_window(label, s, e) for label, s, e in windows_2]
print_section("Fenêtres 2 ans (glissantes)", deux_a_deux)
wins2 = sum(1 for r in deux_a_deux if r["pnl"] > 0)
print(f"\n  Bilan : {wins2}/{len(deux_a_deux)} fenêtres gagnantes")
print(SEP)

# ─── TROIS À TROIS ─────────────────────────────────────────────────────────
print("\n\n" + SEP)
print("  3.  TROIS À TROIS  — fenêtres glissantes, compounding, depuis $1 000")
print(SEP)
windows_3 = [
    (f"{YEARS[i][0]}–{YEARS[i+2][0][-2:]}", YEARS[i][1], YEARS[i+2][2])
    for i in range(len(YEARS) - 2)
]
trois_a_trois = [run_window(label, s, e) for label, s, e in windows_3]
print_section("Fenêtres 3 ans (glissantes)", trois_a_trois)
wins3 = sum(1 for r in trois_a_trois if r["pnl"] > 0)
print(f"\n  Bilan : {wins3}/{len(trois_a_trois)} fenêtres gagnantes")
print(SEP)

# ─── TOTAL 6 ANS ───────────────────────────────────────────────────────────
print("\n\n" + SEP)
print("  4.  TOTAL — simulation continue 6 ans, compounding, depuis $1 000")
print(SEP)
total = run_window("Y1→Y6 (6 ans)", YEARS[0][1], YEARS[-1][2])
print_section("Simulation 6 ans compounding", [total])
print(SEP)

# ─── RÉCAPITULATIF ─────────────────────────────────────────────────────────
print(f"\n\n{'═'*W}")
print("  RÉCAPITULATIF GÉNÉRAL — BULL PUR ×2")
print(f"{'═'*W}")
print(f"  {'Vue':<32}  {'Fenêtres':>9}  {'Gagnantes':>10}  {'PnL moyen':>11}  {'Taux réussite':>14}")
print(f"  {'-'*32}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*14}")
for title, results in [("Une à une (années)", une_a_une),
                        ("Deux à deux (2-ans glissants)", deux_a_deux),
                        ("Trois à trois (3-ans glissants)", trois_a_trois)]:
    pos = sum(1 for r in results if r["pnl"] > 0)
    avg = sum(r["pct"] for r in results) / len(results)
    rate = pos / len(results) * 100
    print(f"  {title:<32}  {len(results):>9}  {pos:>10}  {avg:>+10.1f}%  {rate:>13.0f}%")
# Total
print(f"  {'Simulation continue 6 ans':<32}  {'1':>9}  {'1' if total['pnl']>0 else '0':>10}  {total['pct']:>+10.1f}%  {'100' if total['pnl']>0 else '0':>13}%")
print(f"{'═'*W}\n")
