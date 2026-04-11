#!/usr/bin/env python3
"""
Walk-Forward Validation — Stratégie Adaptative Multi-Régimes

Valide la robustesse de la stratégie adaptative (BULL pullback EMA50/200 15m gattée par
régime 1H) sur 3 fenêtres IS/OOS rolling.

Fenêtres :
  W1 : IS 2023-04-09 → 2024-04-09  |  OOS 2024-04-09 → 2025-04-09  (1y+1y)
  W2 : IS 2024-04-09 → 2025-04-09  |  OOS 2025-04-09 → 2026-04-08  (1y+1y, rolling)
  W3 : IS 2023-04-09 → 2025-04-09  |  OOS 2025-04-09 → 2026-04-08  (2y+1y, long IS)

Usage :
    python3 -m backtest.run_walkforward_adaptive
    python3 -m backtest.run_walkforward_adaptive --balance 1000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from backtest.data_loader import download_candles
from backtest.run_backtest_adaptive import (
    _run_adaptive_pair,
    _compute_metrics,
    PAIRS_BIG5,
)

logging.basicConfig(level=logging.WARNING)

# ── Fenêtres IS / OOS ─────────────────────────────────────────────────────────
WF_WINDOWS = [
    ("W1 — 1y+1y  (2023→2024 IS | 2024→2025 OOS)",
     "2023-04-09", "2024-04-09",   # IS
     "2024-04-09", "2025-04-09"),  # OOS
    ("W2 — 1y+1y  (2024→2025 IS | 2025→2026 OOS)",
     "2024-04-09", "2025-04-09",
     "2025-04-09", "2026-04-08"),
    ("W3 — 2y+1y  (2023→2025 IS | 2025→2026 OOS)",
     "2023-04-09", "2025-04-09",
     "2025-04-09", "2026-04-08"),
]

_W = 116


@dataclass
class WFResult:
    window: str
    is_pf: float
    is_wr: float
    is_pnl: float
    is_trades: int
    oos_pf: float
    oos_wr: float
    oos_pnl: float
    oos_trades: int


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _run_period(
    pairs: list[str],
    start: datetime,
    end: datetime,
    balance_per_pair: float,
) -> dict:
    """Télécharge données et simule pour une période donnée."""
    all_trades = []
    all_equity = []

    for pair in pairs:
        c15 = download_candles(pair, start, end, interval="15m")
        c1h = download_candles(pair, start, end, interval="1h")
        if not c15 or not c1h:
            continue
        bal, trades, equity = _run_adaptive_pair(c15, c1h, balance_per_pair)
        all_trades.extend(trades)
        all_equity.extend(equity)

    if not all_equity:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "final": len(pairs) * balance_per_pair,
                "dd": 0.0, "by_regime": {}}

    return _compute_metrics(all_trades, all_equity, len(pairs) * balance_per_pair)


def run_walkforward(balance: float = 1000.0) -> list[WFResult]:
    balance_per_pair = balance / len(PAIRS_BIG5)
    results: list[WFResult] = []

    for (label, is_start, is_end, oos_start, oos_end) in WF_WINDOWS:
        print(f"\n{'─'*_W}")
        print(f"  {label}")
        print(f"{'─'*_W}")

        dt_is_start  = _parse(is_start)
        dt_is_end    = _parse(is_end)
        dt_oos_start = _parse(oos_start)
        dt_oos_end   = _parse(oos_end)

        print(f"  📥 IS  {is_start} → {is_end}  …", end="", flush=True)
        is_m = _run_period(PAIRS_BIG5, dt_is_start, dt_is_end, balance_per_pair)
        print(f" PF {is_m['pf']:.2f} | WR {is_m['wr']:.1%} | {is_m['n']} trades")

        print(f"  📥 OOS {oos_start} → {oos_end}  …", end="", flush=True)
        oos_m = _run_period(PAIRS_BIG5, dt_oos_start, dt_oos_end, balance_per_pair)
        print(f" PF {oos_m['pf']:.2f} | WR {oos_m['wr']:.1%} | {oos_m['n']} trades")

        ratio = oos_m["pf"] / is_m["pf"] if is_m["pf"] > 0 else 0
        flag = "🟢 ROBUSTE" if oos_m["pf"] >= 1.0 and ratio >= 0.70 else (
               "🟡 FRAGILE" if oos_m["pf"] >= 0.90 else "🔴 FAIL")
        print(f"  IS→OOS ratio: {ratio:.2f} → {flag}")

        results.append(WFResult(
            window=label,
            is_pf=is_m["pf"], is_wr=is_m["wr"],
            is_pnl=is_m["final"] - balance, is_trades=is_m["n"],
            oos_pf=oos_m["pf"], oos_wr=oos_m["wr"],
            oos_pnl=oos_m["final"] - balance, oos_trades=oos_m["n"],
        ))

    return results


def _print_summary(results: list[WFResult], balance: float) -> None:
    print(f"\n{'═'*_W}")
    print("  WALK-FORWARD SUMMARY — Stratégie Adaptative Multi-Régimes")
    print(f"{'═'*_W}")
    print(f"  {'Fenêtre':<42}  {'IS PF':>6}  {'IS WR':>6}  {'OOS PF':>7}  {'OOS WR':>7}  {'Ratio':>6}  {'Statut'}")
    print(f"  {'─'*42}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*12}")
    for r in results:
        ratio = r.oos_pf / r.is_pf if r.is_pf > 0 else 0
        flag = "🟢 ROBUSTE" if r.oos_pf >= 1.0 and ratio >= 0.70 else (
               "🟡 FRAGILE" if r.oos_pf >= 0.90 else "🔴 FAIL")
        print(f"  {r.window:<42}  {r.is_pf:>6.2f}  {r.is_wr:>5.1%}  {r.oos_pf:>7.2f}  {r.oos_wr:>6.1%}  {ratio:>6.2f}  {flag}")

    oos_pfs  = [r.oos_pf for r in results]
    oos_avg  = sum(oos_pfs) / len(oos_pfs) if oos_pfs else 0
    oos_pass = sum(1 for pf in oos_pfs if pf >= 1.0)
    print(f"\n  OOS PF moyen : {oos_avg:.2f} | Fenêtres positives : {oos_pass}/{len(results)}")

    if oos_avg >= 1.10 and oos_pass == len(results):
        verdict = "✅ STRATÉGIE VALIDÉE — Robuste sur toutes les fenêtres OOS"
    elif oos_avg >= 1.0 and oos_pass >= 2:
        verdict = "⚠️  STRATÉGIE ACCEPTABLE — À surveiller sur W2 (période récente)"
    else:
        verdict = "❌ STRATÉGIE NON VALIDÉE — PF OOS insuffisant"
    print(f"\n  {verdict}")
    print(f"{'═'*_W}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-Forward Adaptive")
    parser.add_argument("--balance", type=float, default=1000.0)
    args = parser.parse_args()

    print(f"\n{'═'*_W}")
    print("  WALK-FORWARD — Stratégie Adaptative Multi-Régimes")
    print(f"  Capital : ${args.balance:,.0f} | Big5 | 3 fenêtres IS/OOS")
    print(f"{'═'*_W}")

    results = run_walkforward(args.balance)
    _print_summary(results, args.balance)


if __name__ == "__main__":
    main()
