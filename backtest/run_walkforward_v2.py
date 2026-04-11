#!/usr/bin/env python3
"""
Walk-Forward & Robustness Tests — V2 Breakout Momentum (ULTRATRAIL).

Tests effectués :
  1. Walk-Forward multi-années (2022→2023, 2023→2024, 2024→2025)
  2. Slippage réaliste (0.02%, 0.05%, 0.10%)
  3. Variante robuste (trailing élargi : activation 0.4×ATR, distance 0.3×ATR)
  4. Exclusion BTC/BNB (paires fortes uniquement : ARB, ETH, SOL)
  5. Maker-only (0% fees)
  6. Combinaisons croisées (maker + no BTC/BNB + robuste)

Usage:
    python -m backtest.run_walkforward_v2
    python -m backtest.run_walkforward_v2 --balance 1500
    python -m backtest.run_walkforward_v2 --test slippage
    python -m backtest.run_walkforward_v2 --test walkforward
    python -m backtest.run_walkforward_v2 --test robust
    python -m backtest.run_walkforward_v2 --test all
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.run_backtest_scalping_v2 import (
    BreakoutConfig,
    BreakoutTrade,
    EquityPoint,
    compute_metrics,
    run_multipair,
    run_pair,
)
from backtest.data_loader import download_candles
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Config de base : ULTRATRAIL (le gagnant du round précédent) ──
ULTRATRAIL_BASE = BreakoutConfig(
    name="ULTRATRAIL",
    lookback=12,
    tp_atr_mult=2.0,
    sl_atr_mult=0.8,
    trailing_activation_atr=0.3,
    trailing_distance_atr=0.2,
    session_filter_enabled=False,
    entry_fee_pct=0.0009,
    exit_fee_pct=0.0009,
)

# ── Paires complètes ──
ALL_PAIRS = ["ETH-USD", "BTC-USD", "SOL-USD", "BNB-USD", "ARB-USD"]

# ── Paires fortes (sans BTC/BNB) ──
STRONG_PAIRS = ["ETH-USD", "SOL-USD", "ARB-USD"]

# ── Périodes walk-forward ──
WF_PERIODS = [
    ("2022-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-01-01", "2024-12-31", "2025-01-01", "2025-03-15"),
]

# ── Slippage levels ──
SLIPPAGE_LEVELS = [0.0, 0.0002, 0.0005, 0.001]  # 0%, 0.02%, 0.05%, 0.10%


# ═══════════════════════════════════════════════════════════════════════════════
#  VARIANTES DE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

def make_config(
    name: str,
    *,
    trailing_activation: float = 0.3,
    trailing_distance: float = 0.2,
    entry_fee: float = 0.0009,
    exit_fee: float = 0.0009,
    slippage_pct: float = 0.0,
) -> BreakoutConfig:
    """Crée une config ULTRATRAIL avec des paramètres ajustables."""
    cfg = deepcopy(ULTRATRAIL_BASE)
    cfg.name = name
    cfg.trailing_activation_atr = trailing_activation
    cfg.trailing_distance_atr = trailing_distance
    cfg.entry_fee_pct = entry_fee + slippage_pct  # slippage simulé comme fee additionnel
    cfg.exit_fee_pct = exit_fee + slippage_pct
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION AVEC SLIPPAGE NATIF (plus réaliste que fee additionnel)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pair_with_slippage(
    symbol: str,
    candles: list[Candle],
    cfg: BreakoutConfig,
    initial_balance: float,
    slippage_pct: float = 0.0,
) -> tuple[list[BreakoutTrade], list[EquityPoint], float]:
    """
    Wrapper autour de run_pair qui applique le slippage de manière réaliste :
    - Entry : prix dégradé de +slippage (on achète plus cher)
    - Exit SL/TRAIL_SL : prix dégradé de -slippage (on vend moins cher)
    - Exit TP : prix dégradé de -slippage

    Pour un trailing ultra-serré (0.2×ATR), le slippage est critique.
    On le modélise en ajustant entry_fee et exit_fee (approximation correcte
    pour des moves de <1 ATR).
    """
    # Approche : on ajoute le slippage aux fees car l'impact est symétrique
    cfg_slip = deepcopy(cfg)
    cfg_slip.entry_fee_pct = cfg.entry_fee_pct + slippage_pct
    cfg_slip.exit_fee_pct = cfg.exit_fee_pct + slippage_pct
    return run_pair(symbol, candles, cfg_slip, initial_balance)


def run_multipair_slippage(
    pairs: list[str],
    start: datetime,
    end: datetime,
    cfg: BreakoutConfig,
    initial_balance: float,
    slippage_pct: float = 0.0,
) -> tuple[list[BreakoutTrade], list[EquityPoint], float]:
    """Multi-paire avec slippage."""
    if slippage_pct == 0.0:
        return run_multipair(pairs, start, end, cfg, initial_balance)

    # Download candles
    all_candles: dict[str, list[Candle]] = {}
    for pair in pairs:
        candles = download_candles(pair, start, end, interval="15m")
        if candles:
            all_candles[pair] = candles

    if not all_candles:
        return [], [], initial_balance

    per_pair_capital = initial_balance / len(all_candles)
    all_trades: list[BreakoutTrade] = []
    total_final = 0.0
    ts_equity: dict[int, float] = defaultdict(float)

    for pair, candles in all_candles.items():
        trades, eq, final = run_pair_with_slippage(
            pair, candles, cfg, per_pair_capital, slippage_pct,
        )
        all_trades.extend(trades)
        total_final += final
        for pt in eq:
            ts_equity[pt.ts] += pt.equity

    combined_eq = [
        EquityPoint(ts=ts, equity=eq)
        for ts, eq in sorted(ts_equity.items())
    ]
    all_trades.sort(key=lambda t: t.entry_ts)
    return all_trades, combined_eq, total_final


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 1 : WALK-FORWARD MULTI-ANNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WFResult:
    period: str
    train_metrics: dict
    test_metrics: dict
    test_trades: list[BreakoutTrade]
    test_equity: list[EquityPoint]


def run_walkforward(
    pairs: list[str],
    cfg: BreakoutConfig,
    balance: float,
    slippage_pct: float = 0.0,
) -> list[WFResult]:
    """Walk-forward : train sur année N, test sur année N+1."""
    results: list[WFResult] = []

    for train_start_s, train_end_s, test_start_s, test_end_s in WF_PERIODS:
        train_start = datetime.strptime(train_start_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        train_end = datetime.strptime(train_end_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        test_start = datetime.strptime(test_start_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        test_end = datetime.strptime(test_end_s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        period_label = f"{train_start_s[:4]}→{test_start_s[:4]}"
        print(f"\n  📊 Walk-Forward {period_label}")
        print(f"     Train: {train_start.date()} → {train_end.date()}")
        print(f"     Test : {test_start.date()} → {test_end.date()}")

        # Train
        print(f"     ⏳ Training...")
        train_trades, train_eq, train_final = run_multipair_slippage(
            pairs, train_start, train_end, cfg, balance, slippage_pct,
        )
        train_m = compute_metrics(
            train_trades, train_eq, balance, train_final, train_start, train_end,
        )
        print(f"     Train: {train_m['n_trades']} trades | WR {train_m['win_rate']:.1%} | "
              f"PF {train_m['profit_factor']:.2f} | PnL ${train_m['total_pnl']:+.2f}")

        # Test
        print(f"     ⏳ Testing...")
        test_trades, test_eq, test_final = run_multipair_slippage(
            pairs, test_start, test_end, cfg, balance, slippage_pct,
        )
        test_m = compute_metrics(
            test_trades, test_eq, balance, test_final, test_start, test_end,
        )
        print(f"     Test : {test_m['n_trades']} trades | WR {test_m['win_rate']:.1%} | "
              f"PF {test_m['profit_factor']:.2f} | PnL ${test_m['total_pnl']:+.2f}")

        results.append(WFResult(
            period=period_label,
            train_metrics=train_m,
            test_metrics=test_m,
            test_trades=test_trades,
            test_equity=test_eq,
        ))

    return results


def print_walkforward_report(results: list[WFResult], cfg_name: str) -> None:
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  WALK-FORWARD REPORT — {cfg_name}")
    print(sep)

    # En-tête
    print(f"\n  {'Période':>12s} │ {'Phase':>5s} │ {'Trades':>6s} │ {'WR':>6s} │ "
          f"{'PF':>5s} │ {'PnL':>10s} │ {'DD':>7s} │ {'PnL/j':>8s} │ "
          f"{'Trades/j':>8s} │ {'Fees':>8s}")
    print("  " + "─" * 108)

    test_pnls = []
    test_pfs = []
    test_wrs = []

    for r in results:
        for phase, m in [("TRAIN", r.train_metrics), ("TEST", r.test_metrics)]:
            marker = "  " if phase == "TRAIN" else "→ "
            period_str = r.period if phase == "TRAIN" else ""
            print(f"  {period_str:>12s} │ {marker}{phase:>3s} │ "
                  f"{m['n_trades']:6d} │ {m['win_rate']:5.1%} │ "
                  f"{m['profit_factor']:5.2f} │ ${m['total_pnl']:+9.2f} │ "
                  f"{m['max_dd']:6.1%} │ ${m['daily_pnl_avg']:+7.2f} │ "
                  f"{m['trades_per_day']:7.1f} │ ${m.get('total_fees', 0):+7.2f}")

            if phase == "TEST":
                test_pnls.append(m['total_pnl'])
                test_pfs.append(m['profit_factor'])
                test_wrs.append(m['win_rate'])

    print("  " + "─" * 108)

    # Synthèse OOS (Out-of-Sample)
    n_positive = sum(1 for p in test_pnls if p > 0)
    avg_pf = sum(test_pfs) / len(test_pfs) if test_pfs else 0
    avg_wr = sum(test_wrs) / len(test_wrs) if test_wrs else 0
    total_oos_pnl = sum(test_pnls)

    print(f"\n  SYNTHÈSE OUT-OF-SAMPLE (OOS)")
    print("  " + "─" * 60)
    print(f"  Périodes positives  : {n_positive}/{len(test_pnls)}")
    print(f"  PF moyen OOS       : {avg_pf:.2f}")
    print(f"  WR moyen OOS       : {avg_wr:.1%}")
    print(f"  PnL total OOS      : ${total_oos_pnl:+.2f}")

    # Verdict
    if n_positive == len(test_pnls) and avg_pf > 1.2:
        verdict = "✅ VALIDÉ — Edge confirmé sur toutes les périodes OOS"
    elif n_positive >= len(test_pnls) * 0.66 and avg_pf > 1.0:
        verdict = "⚠️ PROMETTEUR — Edge présent mais pas constant"
    else:
        verdict = "❌ REJETÉ — Overfitting probable"
    print(f"  Verdict            : {verdict}")

    # Dégradation train→test
    print(f"\n  DÉGRADATION TRAIN → TEST")
    print("  " + "─" * 60)
    for r in results:
        pf_train = r.train_metrics['profit_factor']
        pf_test = r.test_metrics['profit_factor']
        degradation = (pf_test - pf_train) / pf_train * 100 if pf_train > 0 else 0
        print(f"  {r.period}: PF {pf_train:.2f} → {pf_test:.2f} ({degradation:+.0f}%)")

    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 2 : SLIPPAGE RÉALISTE
# ═══════════════════════════════════════════════════════════════════════════════

def run_slippage_test(
    pairs: list[str],
    cfg: BreakoutConfig,
    balance: float,
    start: datetime,
    end: datetime,
) -> None:
    """Test l'impact du slippage sur ULTRATRAIL."""
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  SLIPPAGE SENSITIVITY TEST — {cfg.name}")
    print(f"  Paires: {', '.join(pairs)}")
    print(f"  Période: {start.date()} → {end.date()}")
    print(sep)

    print(f"\n  {'Slippage':>10s} │ {'EffFee':>8s} │ {'Trades':>6s} │ {'WR':>6s} │ "
          f"{'PF':>5s} │ {'PnL':>10s} │ {'DD':>7s} │ {'PnL/j':>8s} │ "
          f"{'Fees+Slip':>10s} │ {'Cap 50/j':>10s}")
    print("  " + "─" * 110)

    base_pnl = None
    for slip in SLIPPAGE_LEVELS:
        slip_label = f"{slip*100:.2f}%"
        eff_fee = (cfg.entry_fee_pct + slip + cfg.exit_fee_pct + slip) * 100

        trades, eq, final = run_multipair_slippage(
            pairs, start, end, cfg, balance, slip,
        )
        m = compute_metrics(trades, eq, balance, final, start, end)

        if base_pnl is None:
            base_pnl = m['total_pnl']

        cap50 = (50.0 / m['daily_pnl_avg'] * balance) if m['daily_pnl_avg'] > 0 else float('inf')
        cap_str = f"${cap50:,.0f}" if cap50 < 1e7 else "N/A"

        erosion = ""
        if base_pnl and base_pnl > 0 and slip > 0:
            erosion_pct = (m['total_pnl'] - base_pnl) / base_pnl * 100
            erosion = f" ({erosion_pct:+.0f}%)"

        print(f"  {slip_label:>10s} │ {eff_fee:7.2f}% │ "
              f"{m['n_trades']:6d} │ {m['win_rate']:5.1%} │ "
              f"{m['profit_factor']:5.2f} │ ${m['total_pnl']:+9.2f}{erosion} │ "
              f"{m['max_dd']:6.1%} │ ${m['daily_pnl_avg']:+7.2f} │ "
              f"${m.get('total_fees', 0):+9.2f} │ {cap_str:>10s}")

    print(f"\n  ⚠️  Trailing distance = {cfg.trailing_distance_atr}×ATR — très sensible au slippage")
    print(f"  💡  Si PF < 1.3 avec 0.05% slippage → edge trop fragile pour le live")
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 3 : VARIANTES ROBUSTES (trailing élargi)
# ═══════════════════════════════════════════════════════════════════════════════

def run_robust_variants(
    pairs: list[str],
    balance: float,
    start: datetime,
    end: datetime,
) -> None:
    """Compare ULTRATRAIL original vs variantes plus robustes."""
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"  ROBUSTNESS TEST — Trailing Variants")
    print(f"  Paires: {', '.join(pairs)}")
    print(f"  Période: {start.date()} → {end.date()}")
    print(sep)

    variants = [
        # (name, trail_activation, trail_distance, fees_entry, fees_exit, description)
        ("ULTRATRAIL (base)", 0.3, 0.2, 0.0009, 0.0009, "Original ultra-serré"),
        ("ROBUST_03_03", 0.3, 0.3, 0.0009, 0.0009, "Distance élargie 0.3×ATR"),
        ("ROBUST_04_03", 0.4, 0.3, 0.0009, 0.0009, "Activation+distance élargis"),
        ("ROBUST_05_04", 0.5, 0.4, 0.0009, 0.0009, "Max robuste (moins agressif)"),
        ("ULTRA_MAKER", 0.3, 0.2, 0.0, 0.0, "Ultra-serré + maker 0%"),
        ("ROBUST_MAKER", 0.4, 0.3, 0.0, 0.0, "Robuste + maker 0%"),
    ]

    print(f"\n  {'Variante':>22s} │ {'Act':>4s} │ {'Dist':>4s} │ {'Fee':>6s} │ "
          f"{'Trades':>6s} │ {'WR':>6s} │ {'PF':>5s} │ "
          f"{'PnL':>10s} │ {'DD':>7s} │ {'PnL/j':>8s} │ "
          f"{'Fees':>8s} │ {'Cap 50/j':>10s} │ Description")
    print("  " + "─" * 140)

    for name, act, dist, fee_e, fee_x, desc in variants:
        cfg = make_config(
            name,
            trailing_activation=act,
            trailing_distance=dist,
            entry_fee=fee_e,
            exit_fee=fee_x,
        )
        trades, eq, final = run_multipair(pairs, start, end, cfg, balance)
        m = compute_metrics(trades, eq, balance, final, start, end)

        cap50 = (50.0 / m['daily_pnl_avg'] * balance) if m['daily_pnl_avg'] > 0 else float('inf')
        cap_str = f"${cap50:,.0f}" if cap50 < 1e7 else "N/A"

        fee_label = f"{(fee_e + fee_x)*100:.2f}%"

        print(f"  {name:>22s} │ {act:4.1f} │ {dist:4.1f} │ {fee_label:>6s} │ "
              f"{m['n_trades']:6d} │ {m['win_rate']:5.1%} │ "
              f"{m['profit_factor']:5.2f} │ ${m['total_pnl']:+9.2f} │ "
              f"{m['max_dd']:6.1%} │ ${m['daily_pnl_avg']:+7.2f} │ "
              f"${m.get('total_fees', 0):+7.2f} │ {cap_str:>10s} │ {desc}")

    print(f"\n  💡  Si ROBUST_04_03 garde PF > 1.5 → c'est LA config live")
    print(f"  💡  Si seul ULTRATRAIL marche → edge trop fragile (overfitting trailing)")
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 4 : EXCLUSION BTC/BNB
# ═══════════════════════════════════════════════════════════════════════════════

def run_pair_filter_test(
    balance: float,
    start: datetime,
    end: datetime,
) -> None:
    """Compare toutes les paires vs paires fortes (sans BTC/BNB)."""
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  PAIR FILTER TEST — Exclusion BTC/BNB")
    print(f"  Période: {start.date()} → {end.date()}")
    print(sep)

    configs_pairs = [
        ("ALL (5 paires)", ALL_PAIRS, ULTRATRAIL_BASE),
        ("STRONG (3: ETH,SOL,ARB)", STRONG_PAIRS, ULTRATRAIL_BASE),
        ("ALL + maker", ALL_PAIRS, make_config("ALL_MAKER", entry_fee=0.0, exit_fee=0.0)),
        ("STRONG + maker", STRONG_PAIRS, make_config("STRONG_MAKER", entry_fee=0.0, exit_fee=0.0)),
    ]

    print(f"\n  {'Scénario':>25s} │ {'Trades':>6s} │ {'WR':>6s} │ {'PF':>5s} │ "
          f"{'PnL':>10s} │ {'DD':>7s} │ {'PnL/j':>8s} │ {'Cap 50/j':>10s}")
    print("  " + "─" * 100)

    for label, pairs, cfg in configs_pairs:
        trades, eq, final = run_multipair(pairs, start, end, cfg, balance)
        m = compute_metrics(trades, eq, balance, final, start, end)

        cap50 = (50.0 / m['daily_pnl_avg'] * balance) if m['daily_pnl_avg'] > 0 else float('inf')
        cap_str = f"${cap50:,.0f}" if cap50 < 1e7 else "N/A"

        print(f"  {label:>25s} │ {m['n_trades']:6d} │ {m['win_rate']:5.1%} │ "
              f"{m['profit_factor']:5.2f} │ ${m['total_pnl']:+9.2f} │ "
              f"{m['max_dd']:6.1%} │ ${m['daily_pnl_avg']:+7.2f} │ {cap_str:>10s}")

        # Détail par paire
        if m.get("by_pair"):
            for pair, s in sorted(m["by_pair"].items(), key=lambda x: -x[1]["pnl"]):
                wr2 = s["wins"] / s["n"] * 100 if s["n"] else 0
                print(f"  {'':>25s}   └─ {pair:12s} : {s['n']:4d} tr | "
                      f"WR {wr2:5.1f}% | PnL ${s['pnl']:+8.2f}")

    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  TEST 5 : COMBINAISON OPTIMALE
# ═══════════════════════════════════════════════════════════════════════════════

def run_optimal_combo(
    balance: float,
    start: datetime,
    end: datetime,
) -> None:
    """Teste les combinaisons les plus prometteuses."""
    sep = "=" * 130
    print(f"\n{sep}")
    print(f"  OPTIMAL COMBINATION TEST")
    print(f"  Objectif : trouver la config live optimale")
    print(f"  Période: {start.date()} → {end.date()}")
    print(sep)

    combos = [
        # (label, pairs, cfg)
        ("Base: ALL + taker",
         ALL_PAIRS,
         make_config("BASE_TAKER")),

        ("A: STRONG + taker",
         STRONG_PAIRS,
         make_config("STRONG_TAKER")),

        ("B: ALL + maker",
         ALL_PAIRS,
         make_config("ALL_MAKER", entry_fee=0.0, exit_fee=0.0)),

        ("C: STRONG + maker",
         STRONG_PAIRS,
         make_config("STRONG_MAKER", entry_fee=0.0, exit_fee=0.0)),

        ("D: STRONG + maker + robust",
         STRONG_PAIRS,
         make_config("STRONG_MK_ROB", trailing_activation=0.4, trailing_distance=0.3,
                      entry_fee=0.0, exit_fee=0.0)),

        ("E: ALL + maker + robust",
         ALL_PAIRS,
         make_config("ALL_MK_ROB", trailing_activation=0.4, trailing_distance=0.3,
                      entry_fee=0.0, exit_fee=0.0)),

        ("F: STRONG + maker + wide",
         STRONG_PAIRS,
         make_config("STRONG_MK_WIDE", trailing_activation=0.5, trailing_distance=0.4,
                      entry_fee=0.0, exit_fee=0.0)),

        ("G: STRONG + taker + slip 0.05%",
         STRONG_PAIRS,
         make_config("STRONG_SLIP", slippage_pct=0.0005)),

        ("H: STRONG + maker + robust + slip 0.02%",
         STRONG_PAIRS,
         make_config("STRONG_MK_ROB_SLIP", trailing_activation=0.4, trailing_distance=0.3,
                      entry_fee=0.0, exit_fee=0.0, slippage_pct=0.0002)),
    ]

    print(f"\n  {'#':>2s} {'Scénario':>35s} │ {'Trades':>6s} │ {'WR':>6s} │ {'PF':>5s} │ "
          f"{'PnL':>10s} │ {'DD':>7s} │ {'PnL/j':>8s} │ "
          f"{'Tr/j':>6s} │ {'Fees':>8s} │ {'Cap 50/j':>10s}")
    print("  " + "─" * 130)

    best_pnl = -1e9
    best_label = ""

    for i, (label, pairs, cfg) in enumerate(combos, 1):
        trades, eq, final = run_multipair(pairs, start, end, cfg, balance)
        m = compute_metrics(trades, eq, balance, final, start, end)

        cap50 = (50.0 / m['daily_pnl_avg'] * balance) if m['daily_pnl_avg'] > 0 else float('inf')
        cap_str = f"${cap50:,.0f}" if cap50 < 1e7 else "N/A"

        marker = ""
        if m['total_pnl'] > best_pnl:
            best_pnl = m['total_pnl']
            best_label = label

        print(f"  {i:>2d} {label:>35s} │ {m['n_trades']:6d} │ {m['win_rate']:5.1%} │ "
              f"{m['profit_factor']:5.2f} │ ${m['total_pnl']:+9.2f} │ "
              f"{m['max_dd']:6.1%} │ ${m['daily_pnl_avg']:+7.2f} │ "
              f"{m['trades_per_day']:5.1f} │ ${m.get('total_fees', 0):+7.2f} │ {cap_str:>10s}")

    print("  " + "─" * 130)
    print(f"\n  🏆 Meilleure combinaison : {best_label} (PnL ${best_pnl:+.2f})")
    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD SUR LA MEILLEURE COMBO
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_walkforward_suite(balance: float) -> None:
    """Walk-forward sur les configs les plus prometteuses."""
    sep = "=" * 120
    print(f"\n{sep}")
    print(f"  WALK-FORWARD MULTI-CONFIGURATIONS")
    print(f"  3 périodes × 4 configs = 12 tests OOS")
    print(sep)

    configs_to_wf = [
        ("ULTRATRAIL (all, taker)",
         ALL_PAIRS,
         ULTRATRAIL_BASE),

        ("ULTRATRAIL (strong, taker)",
         STRONG_PAIRS,
         ULTRATRAIL_BASE),

        ("ROBUST (strong, maker)",
         STRONG_PAIRS,
         make_config("ROB_MK", trailing_activation=0.4, trailing_distance=0.3,
                      entry_fee=0.0, exit_fee=0.0)),

        ("ULTRATRAIL (strong, maker)",
         STRONG_PAIRS,
         make_config("ULTRA_MK", entry_fee=0.0, exit_fee=0.0)),
    ]

    for label, pairs, cfg in configs_to_wf:
        print(f"\n{'─' * 80}")
        print(f"  🔬 Walk-Forward : {label}")
        print(f"     Paires : {', '.join(pairs)}")
        print(f"{'─' * 80}")

        results = run_walkforward(pairs, cfg, balance)
        print_walkforward_report(results, label)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_wf_charts(results: list[WFResult], label: str) -> Path:
    """Génère un chart multi-périodes walk-forward."""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    fig.suptitle(f"Walk-Forward — {label}", fontsize=14)

    for ax, r in zip(axes, results):
        if r.test_equity:
            dates = [datetime.fromtimestamp(e.ts / 1000, tz=timezone.utc) for e in r.test_equity]
            equities = [e.equity for e in r.test_equity]
            color = "green" if r.test_metrics['total_pnl'] > 0 else "red"
            ax.plot(dates, equities, color=color, linewidth=0.8)
            ax.set_title(f"OOS {r.period}\nPF={r.test_metrics['profit_factor']:.2f} "
                         f"WR={r.test_metrics['win_rate']:.1%}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"walkforward_{label.replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    return chart_path


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward & Robustness Tests — V2 Breakout Momentum",
    )
    parser.add_argument("--balance", type=float, default=1500.0)
    parser.add_argument(
        "--test", type=str, default="all",
        choices=["walkforward", "slippage", "robust", "pairs", "combo", "all"],
        help="Quel test lancer (default: all)",
    )
    args = parser.parse_args()

    balance = args.balance

    # Période par défaut pour les tests simples (1 an récent)
    end = datetime(2025, 3, 15, tzinfo=timezone.utc)
    start = end - timedelta(days=365)

    banner = """
╔══════════════════════════════════════════════════════════════════════════╗
║            WALK-FORWARD & ROBUSTNESS TESTS — V2 ULTRATRAIL             ║
║                                                                        ║
║  1. Walk-Forward 3 ans (2022→2023, 2023→2024, 2024→2025)              ║
║  2. Slippage Sensitivity (0%, 0.02%, 0.05%, 0.10%)                    ║
║  3. Trailing Robustness (0.2→0.4×ATR)                                 ║
║  4. Pair Filter (ALL vs STRONG: -BTC -BNB)                            ║
║  5. Maker-only (0% fees)                                              ║
║  6. Combinaisons optimales                                             ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    print(f"  Capital de base : ${balance:,.0f}")
    print(f"  Config ULTRATRAIL : high(12), TP=2.0×ATR, SL=0.8×ATR, "
          f"trail act=0.3×ATR, dist=0.2×ATR")

    if args.test in ("walkforward", "all"):
        print("\n\n" + "█" * 80)
        print("  TEST 1 : WALK-FORWARD MULTI-ANNÉES")
        print("█" * 80)
        run_full_walkforward_suite(balance)

    if args.test in ("slippage", "all"):
        print("\n\n" + "█" * 80)
        print("  TEST 2 : SLIPPAGE SENSITIVITY")
        print("█" * 80)
        # Test sur 1 an avec toutes les paires
        run_slippage_test(ALL_PAIRS, ULTRATRAIL_BASE, balance, start, end)
        # Aussi sur paires fortes
        run_slippage_test(STRONG_PAIRS, ULTRATRAIL_BASE, balance, start, end)

    if args.test in ("robust", "all"):
        print("\n\n" + "█" * 80)
        print("  TEST 3 : TRAILING ROBUSTNESS")
        print("█" * 80)
        run_robust_variants(ALL_PAIRS, balance, start, end)

    if args.test in ("pairs", "all"):
        print("\n\n" + "█" * 80)
        print("  TEST 4 : PAIR FILTER")
        print("█" * 80)
        run_pair_filter_test(balance, start, end)

    if args.test in ("combo", "all"):
        print("\n\n" + "█" * 80)
        print("  TEST 5 : COMBINAISONS OPTIMALES")
        print("█" * 80)
        run_optimal_combo(balance, start, end)

    # ── Résumé final ──
    if args.test == "all":
        print("\n" + "═" * 80)
        print("  RÉSUMÉ DES RECOMMANDATIONS")
        print("═" * 80)
        print("""
  📋 CHECKLIST DE VALIDATION :

  □ Walk-forward : PF OOS > 1.3 sur ≥ 2/3 périodes ?
  □ Slippage 0.05% : PF encore > 1.2 ?
  □ Trailing 0.4/0.3 : PF > 1.4 (robustesse confirmée) ?
  □ Sans BTC/BNB : PnL amélioré ?
  □ Maker-only : PnL doublé ?

  🎯 CONFIG LIVE RECOMMANDÉE (SI VALIDÉ) :
     Paires        : ARB-USD, ETH-USD, SOL-USD (exclure BTC/BNB)
     Trailing act  : 0.4×ATR (robuste) ou 0.3×ATR (agressif si WF OK)
     Trailing dist : 0.3×ATR (robuste) ou 0.2×ATR (agressif si WF OK)
     Fees          : Maker-only (0%) sur Revolut X
     Capital       : Ajuster selon PnL/jour OOS réel

  ⚠️ ATTENTION :
     - Si WF échoue → overfitting confirmé, NE PAS déployer
     - Si slippage casse le PF → trailing trop serré pour le live
     - Durée moy 16 min → latence API critique (< 500ms)
""")


if __name__ == "__main__":
    main()
