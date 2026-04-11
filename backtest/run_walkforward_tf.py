#!/usr/bin/env python3
"""
Walk-Forward Validation — Trend Following v2 (3strats)

Teste 3 configs candidates issues de l'optimisation G7 :
  · TF_S15_FULL       — filtre strict, pas de pyramiding  (max PF/qualité)
  · TF_S15_FULL_PYR   — même filtre + pyramiding + alloc 40%
  · TF_S10_RSI_PYR    — pente souple + RSI montant + pyramiding  (max PnL)

Fenêtres walk-forward (3 ans de données 2023-04 → 2026-04) :
  W1 : IS 2023-04-09 → 2024-04-09  |  OOS 2024-04-09 → 2025-04-09  (1y+1y)
  W2 : IS 2024-04-09 → 2025-04-09  |  OOS 2025-04-09 → 2026-04-08  (1y+1y, rolling)
  W3 : IS 2023-04-09 → 2025-04-09  |  OOS 2025-04-09 → 2026-04-08  (2y+1y, long IS)

Usage :
    python3 -m backtest.run_walkforward_tf
    python3 -m backtest.run_walkforward_tf --balance 1000
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from backtest.data_loader import download_candles
from backtest.run_backtest_3strats import (
    TFConfig,
    _run_tf_pair,
    _compute_metrics,
    PAIRS_BIG5,
)
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING)

# ── Configs candidates ────────────────────────────────────────────────────────
_B = dict(rsi_min=50, rsi_max=65, sl_pct=0.015, trail_pct=0.015, tp_fixed=0.0)

CANDIDATES: list[TFConfig] = [
    TFConfig(
        name="TF_S15_FULL",
        **_B,
        alloc_pct=0.25,
        slope_min_pct=0.0015,
        atr_min_ratio=1.00,
        rsi_rising=True,
    ),
    TFConfig(
        name="TF_S15_FULL_PYR_R40",
        **_B,
        alloc_pct=0.40,
        slope_min_pct=0.0015,
        atr_min_ratio=1.00,
        rsi_rising=True,
        pyramid_enabled=True,
    ),
    TFConfig(
        name="TF_S10_RSI_PYR",
        **_B,
        alloc_pct=0.25,
        slope_min_pct=0.0010,
        rsi_rising=True,
        pyramid_enabled=True,
    ),
]

# ── Fenêtres IS / OOS ─────────────────────────────────────────────────────────
WF_WINDOWS = [
    ("W1 — 1y+1y  (2023→2024→2025)",
     "2023-04-09", "2024-04-09",   # IS
     "2024-04-09", "2025-04-09"),  # OOS
    ("W2 — 1y+1y  (2024→2025→2026)",
     "2024-04-09", "2025-04-09",
     "2025-04-09", "2026-04-08"),
    ("W3 — 2y+1y  (2023-25→2025-26)",
     "2023-04-09", "2025-04-09",
     "2025-04-09", "2026-04-08"),
]

_W = 110


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _run_window(
    cfg: TFConfig,
    candles_full: dict[str, list[Candle]],
    start: datetime,
    end: datetime,
    initial: float,
) -> dict:
    """Applique cfg sur [start, end] pour chaque paire et agrège."""
    pairs = list(candles_full.keys())
    per_pair = initial / len(pairs)
    all_trades = []
    all_equity: list[float] = []

    # timestamp en ms → bornes en ms
    start_ms = int(start.timestamp() * 1000)
    end_ms   = int(end.timestamp()   * 1000)

    for pair in pairs:
        sliced = [c for c in candles_full[pair] if start_ms <= c.timestamp < end_ms]
        if len(sliced) < 200:      # pas assez de bougies
            continue
        bal, trades, eq = _run_tf_pair(sliced, cfg, per_pair)
        all_trades.extend(trades)
        if not all_equity:
            all_equity = list(eq)
        else:
            all_equity = [a + b for a, b in zip(all_equity, eq)]

    m = _compute_metrics(all_trades, all_equity, initial)
    # final balance agrégée réelle
    total_bal = 0.0
    for pair in pairs:
        sliced = [c for c in candles_full[pair] if start_ms <= c.timestamp < end_ms]
        if len(sliced) < 200:
            total_bal += per_pair
            continue
        bal, _, _ = _run_tf_pair(sliced, cfg, per_pair)
        total_bal += bal
    m["final"] = total_bal
    return m


def _print_sep(char: str = "─") -> None:
    print("  " + char * (_W - 2))


def _pf_str(pf: float) -> str:
    return f"{pf:.2f}" if pf < 99 else "  ∞ "


def _grade(oos_pf: float, oos_vs_is: float, oos_positive: bool) -> str:
    """Note qualitative OOS."""
    if oos_pf >= 1.15 and oos_vs_is >= 0.70 and oos_positive:
        return "🟢 ROBUSTE"
    if oos_pf >= 1.05 and oos_vs_is >= 0.50 and oos_positive:
        return "🟡 CORRECT"
    if oos_positive:
        return "🟠 FRAGILE"
    return "🔴 ÉCHEC"


# ── Walk-Forward principal ────────────────────────────────────────────────────

def run_walkforward_tf(balance: float = 1_000.0) -> None:
    # ── Téléchargement global (3 ans) ─────────────────────────────────────
    global_start = _parse_dt("2023-04-09")
    global_end   = _parse_dt("2026-04-08")

    print(f"\n📥 Téléchargement 15m ({global_start.date()} → {global_end.date()})…")
    candles_full: dict[str, list[Candle]] = {}
    for pair in PAIRS_BIG5:
        cds = download_candles(pair, global_start, global_end, interval="15m")
        candles_full[pair] = cds
        print(f"  ✓ {pair}: {len(cds):,} bougies")

    # ── En-tête ────────────────────────────────────────────────────────────
    print(f"\n{'═' * _W}")
    print(f"  WALK-FORWARD — TREND FOLLOWING v2 | Capital ${balance:,.0f} | Big5 | 15m")
    print(f"{'═' * _W}")

    # ── Par config ────────────────────────────────────────────────────────
    for cfg in CANDIDATES:
        print(f"\n  {'━' * (_W - 2)}")
        print(f"  CONFIG : {cfg.name}")
        filters = []
        if cfg.slope_min_pct > 0:  filters.append(f"pente≥{cfg.slope_min_pct*100:.2f}%")
        if cfg.atr_min_ratio  > 0: filters.append(f"atr≥{cfg.atr_min_ratio:.0%}MA")
        if cfg.rsi_rising:          filters.append("rsi↑")
        if cfg.pyramid_enabled:     filters.append("pyramid")
        print(f"  Filtres : {' | '.join(filters)} | RSI {cfg.rsi_min:.0f}–{cfg.rsi_max:.0f} | alloc {cfg.alloc_pct:.0%}")
        print(f"  {'━' * (_W - 2)}")

        print(f"\n  {'Fenêtre':42s} {'Phase':5s} {'Trades':>7s} {'WR':>6s} {'PF':>6s} {'PnL':>10s} {'DD':>8s}  Note")
        _print_sep()

        oos_results = []

        for label, is_s, is_e, oos_s, oos_e in WF_WINDOWS:
            is_start  = _parse_dt(is_s)
            is_end    = _parse_dt(is_e)
            oos_start = _parse_dt(oos_s)
            oos_end   = _parse_dt(oos_e)

            is_m  = _run_window(cfg, candles_full, is_start,  is_end,  balance)
            oos_m = _run_window(cfg, candles_full, oos_start, oos_end, balance)

            pf_ratio = oos_m["pf"] / is_m["pf"] if is_m["pf"] > 0 else 0.0
            note = _grade(oos_m["pf"], pf_ratio, oos_m["final"] >= balance)

            # IS ligne
            is_pnl  = is_m["final"]  - balance
            oos_pnl = oos_m["final"] - balance
            print(
                f"  {label:42s} "
                f"{'IS':5s} "
                f"{is_m['n']:7d} "
                f"{is_m['wr']:5.1%} "
                f"{_pf_str(is_m['pf']):>6s} "
                f"${is_pnl:+9.2f} "
                f"{is_m['dd']:7.1%}"
            )
            print(
                f"  {'':42s} "
                f"{'OOS':5s} "
                f"{oos_m['n']:7d} "
                f"{oos_m['wr']:5.1%} "
                f"{_pf_str(oos_m['pf']):>6s} "
                f"${oos_pnl:+9.2f} "
                f"{oos_m['dd']:7.1%}  {note}"
            )
            print(
                f"  {'':42s} "
                f"{'':5s} "
                f"{'':7s} "
                f"{'':6s} "
                f"  PF OOS/IS = {pf_ratio:.2f}"
            )
            _print_sep("·")

            oos_results.append((oos_m, pf_ratio))

        # Synthèse OOS
        n_pos   = sum(1 for m, _ in oos_results if m["final"] >= balance)
        avg_pf  = sum(m["pf"]  for m, _ in oos_results) / len(oos_results)
        avg_wr  = sum(m["wr"]  for m, _ in oos_results) / len(oos_results)
        avg_dd  = sum(m["dd"]  for m, _ in oos_results) / len(oos_results)
        avg_pnl = sum(m["final"] - balance for m, _ in oos_results) / len(oos_results)
        avg_rat = sum(r for _, r in oos_results) / len(oos_results)

        overall = _grade(avg_pf, avg_rat, n_pos == len(oos_results))

        _print_sep()
        print(f"\n  SYNTHÈSE OOS — {cfg.name}")
        print(f"  Fenêtres positives  : {n_pos}/{len(oos_results)}")
        print(f"  PF moyen OOS        : {avg_pf:.2f}")
        print(f"  WR moyen OOS        : {avg_wr:.1%}")
        print(f"  DD moyen OOS        : {avg_dd:.1%}")
        print(f"  PnL moyen OOS       : ${avg_pnl:+.2f}/an")
        print(f"  Ratio PF (OOS/IS)   : {avg_rat:.2f}")
        print(f"  Verdict             : {overall}")

    print(f"\n{'═' * _W}")
    print("  Légende : 🟢 ROBUSTE = PF OOS≥1.15 & OOS/IS≥70%  "
          "| 🟡 CORRECT = PF≥1.05 & OOS/IS≥50%  "
          "| 🟠 FRAGILE = positif  | 🔴 ÉCHEC")
    print(f"{'═' * _W}\n")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance", type=float, default=1_000.0)
    args = parser.parse_args()
    run_walkforward_tf(balance=args.balance)
