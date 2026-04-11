#!/usr/bin/env python3
"""
Benchmark 3 étapes d'optimisation — utilise le cache local, pas de réseau.

Étape 1 : BULL pur  (RANGE/STAG déjà désactivés dans run_backtest_adaptive)
Étape 2 : BULL pur + levier ×2
Étape 3 : (préparation) BULL pur + levier ×2 + RANGE fixé 15m
"""
import csv
from datetime import datetime, timezone
from pathlib import Path
from src.core.models import Candle
import logging
logging.disable(logging.CRITICAL)

CACHE_DIR = Path(__file__).parent / "data"
PAIR_SYM = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "BNB-USD": "BNBUSDT",
    "XRP-USD": "XRPUSDT",
}

from backtest.run_backtest_adaptive import _run_adaptive_pair, _compute_metrics, PAIRS_BIG5, Regime

PERIODS = [
    ("Y1 2021-22", "2021-04-09", "2022-04-09"),
    ("Y2 2022-23", "2022-04-09", "2023-04-09"),
    ("Y3 2023-24", "2023-04-09", "2024-04-09"),
    ("Y4 2024-25", "2024-04-09", "2025-04-09"),
    ("Y5 2025-26", "2025-04-09", "2026-04-08"),
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

def _filter(candles, s_ms, e_ms):
    return [c for c in candles if s_ms <= c.timestamp < e_ms]

_MEM = {}

def _get(pair, interval, s_ms, e_ms):
    bsym = PAIR_SYM[pair]
    key = (bsym, interval)
    if key not in _MEM:
        cands = sorted(CACHE_DIR.glob(f"{bsym}_{interval}_*.csv"),
                       key=lambda p: p.stat().st_size, reverse=True)
        if not cands:
            return []
        _MEM[key] = _load_csv_file(cands[0])
    return _filter(_MEM[key], s_ms, e_ms)

def run_all_years(capital=1000.0, leverage=1.0, extra_kwargs=None):
    """Simule chaque année avec compounding. leverage appliqué au sizing BULL."""
    from backtest.run_backtest_adaptive import _run_adaptive_pair
    kwargs = dict(extra_kwargs or {})
    # Ajustement du sizing via le leverage (on multiplie bull_alloc_pct)
    base_alloc = kwargs.pop("bull_alloc_pct", 0.50)
    kwargs["bull_alloc_pct"] = min(base_alloc * leverage, 0.99)

    results = []
    cap = capital
    for (label, s, e) in PERIODS:
        s_ms = int(parse(s).timestamp() * 1000)
        e_ms = int(parse(e).timestamp() * 1000)
        per_pair = cap / len(PAIRS_BIG5)
        all_trades = []
        combined_eq = []
        total_final = 0.0
        n_ok = 0
        for pair in PAIRS_BIG5:
            c15 = _get(pair, "15m", s_ms, e_ms)
            c1h = _get(pair, "1h",  s_ms, e_ms)
            if not c15 or not c1h:
                continue
            bal, trades, eq = _run_adaptive_pair(c15, c1h, per_pair, **kwargs)
            all_trades.extend(trades)
            total_final += bal
            n_ok += 1
            if not combined_eq:
                combined_eq = list(eq)
            else:
                combined_eq = [a + b for a, b in zip(combined_eq, eq)]
        if not combined_eq or n_ok == 0:
            results.append((label, cap, cap, {"n":0,"wr":0,"pf":0,"dd":0}))
            continue
        m = _compute_metrics(all_trades, combined_eq, cap)
        m["final"] = total_final
        results.append((label, cap, total_final, m))
        cap = total_final
    return results

def print_table(title, results, capital=1000.0, show_regime=False):
    W = 116
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"{'═'*W}")
    print(f"  {'Période':<16}  {'Cap. début':>11}  {'Cap. fin':>10}  {'PnL $':>9}  {'PnL %':>7}  {'PF':>5}  {'WR':>6}  {'DD max':>7}  {'Trades':>6}")
    print(f"  {'-'*16}  {'-'*11}  {'-'*10}  {'-'*9}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*6}")
    regime_totals = {}
    for (label, cap_s, cap_e, m) in results:
        pnl = cap_e - cap_s
        pct = pnl / cap_s * 100 if cap_s > 0 else 0
        dd_s = f"{m['dd']:.1%}" if m['n'] > 0 else "n/a"
        pf_s = f"{m['pf']:.2f}" if m['n'] > 0 else " n/a"
        wr_s = f"{m['wr']:.1%}" if m['n'] > 0 else " n/a"
        neg = pnl < 0
        print(f"  {label:<16}  ${cap_s:>10,.2f}  ${cap_e:>9,.2f}  ${pnl:>+8,.2f}  {pct:>+6.1f}%  {pf_s:>5}  {wr_s:>6}  {dd_s:>7}  {m['n']:>6}{'  ← PERTE' if neg else ''}")
        if show_regime and m.get('by_regime'):
            for rname, rm in m['by_regime'].items():
                print(f"    {'└─ '+rname:<14}  {'':>11}  {'':>10}  ${rm['pnl']:>+8.2f}  {'':>7}  {rm['pf']:>5.2f}  {rm['wr']:>5.1%}  {'':>7}  {rm['n']:>6}")
            for rname, rm in m['by_regime'].items():
                if rname not in regime_totals:
                    regime_totals[rname] = {'pnl': 0, 'n': 0, 'gp': 0, 'gl': 0}
                regime_totals[rname]['pnl'] += rm['pnl']
                regime_totals[rname]['n']   += rm['n']
    total_end = results[-1][2]
    total_pnl = total_end - capital
    total_pct = total_pnl / capital * 100
    print(f"\n  {'TOTAL 5 ANS':<16}  ${capital:>10,.2f}  ${total_end:>9,.2f}  ${total_pnl:>+8,.2f}  {total_pct:>+6.1f}%")
    if show_regime and regime_totals:
        print(f"\n  Cumul par régime sur 5 ans :")
        for rname, rt in regime_totals.items():
            print(f"    {rname:<12}  trades={rt['n']:>5}  PnL cumulé=${rt['pnl']:>+8.2f}")
    print(f"{'═'*W}")

# ─── Étape 1 : BULL pur sans levier ────────────────────────────────────────
print("\n\n" + "█"*116)
print("  ÉTAPE 1 — BULL PUR (RANGE + STAGNATION désactivés)")
print("█"*116)
r1 = run_all_years(leverage=1.0, extra_kwargs={"bull_only": True})
print_table("BULL PUR — levier ×1 — $1,000 de départ (avec compounding)", r1)

# ─── Étape 2 : BULL pur + levier ×2 ────────────────────────────────────────
print("\n\n" + "█"*116)
print("  ÉTAPE 2 — BULL PUR + LEVIER ×2 (bull_alloc_pct = 100%)")
print("█"*116)
r2 = run_all_years(leverage=2.0, extra_kwargs={"bull_only": True})
print_table("BULL PUR — levier ×2 — $1,000 de départ (avec compounding)", r2)

# ─── Étape 3 : BULL + RANGE 15m + levier ×1 et ×2 ──────────────────────────
# RANGE est maintenant actif dans run_backtest_adaptive (can_enter réactivé)
print("\n\n" + "█"*116)
print("  ÉTAPE 3 — BULL + RANGE 15m FIXÉ, levier ×1")
print("█"*116)
r3 = run_all_years(leverage=1.0)
print_table("BULL + RANGE 15m — levier ×1 — $1,000 de départ", r3, show_regime=True)

print("\n\n" + "█"*116)
print("  ÉTAPE 3b — BULL + RANGE 15m FIXÉ, levier ×2")
print("█"*116)
r3b = run_all_years(leverage=2.0)
print_table("BULL + RANGE 15m — levier ×2 — $1,000 de départ", r3b, show_regime=True)

# ─── Résumé comparatif ─────────────────────────────────────────────────────
W = 116
print(f"\n{'═'*W}")
print("  COMPARATIF FINAL — 5 ANS CUMULÉS")
print(f"{'═'*W}")
print(f"  {'Config':<30}  {'Total':>10}  {'PnL $':>10}  {'PnL %':>8}  {'Années neg':>11}  {'DD max':>7}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*11}  {'-'*7}")
for (lev_label, res) in [("BULL pur ×1", r1), ("BULL pur ×2", r2), ("BULL+RANGE ×1", r3), ("BULL+RANGE ×2", r3b)]:
    total_end = res[-1][2]
    total_pnl = total_end - 1000
    total_pct = total_pnl / 1000 * 100
    neg_years = sum(1 for (_, cs, ce, _) in res if ce < cs)
    dds = [m['dd'] for (_, _, _, m) in res if m['n'] > 0]
    worst_dd = min(dds) if dds else 0
    print(f"  {lev_label:<30}  ${total_end:>9,.2f}  ${total_pnl:>+9,.2f}  {total_pct:>+7.1f}%  {neg_years:>5}/5        {worst_dd:>7.1%}")
print(f"{'═'*W}\n")
