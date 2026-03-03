#!/usr/bin/env python
"""
Backtest Smart Infinity Grid 2.0

Stratégie :
  Grid adaptative maker-only (0% fee Revolut X)
  Activation : ATR élevé + RSI H1 < 35 + EMA200 healthy
  Entrée : RSI H1 < 30 + bougie de rejet → grid 5 niveaux
  Sortie : TP dynamique PMP+0.8%/1.5%/2.5% (scale-out progressif)
  Stop : EMA200 H4 cassée | RSI<25 12 bougies | DD>20%
  Capital : 20% par cycle, max 3 cycles simultanés

Usage :
    # Base — 12 mois H1
    PYTHONPATH=. python backtest/run_backtest_grid.py --months 12 --no-show

    # 6 mois
    PYTHONPATH=. python backtest/run_backtest_grid.py --months 6 --no-show

    # Sensitivity
    PYTHONPATH=. python backtest/run_backtest_grid.py --months 12 --sensitivity --no-show

    # Custom pairs
    PYTHONPATH=. python backtest/run_backtest_grid.py --months 12 --pairs BTC-USD,ETH-USD --no-show
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import download_candles
from backtest.simulator_grid import (
    GridConfig,
    GridEngine,
    GridResult,
    GridTrade,
    DualGridEngine,
    DualGridResult,
    compute_grid_metrics,
    compute_dual_metrics,
)
from src.core.grid_engine import (
    GridMode,
    micro_dip_config,
    deep_dip_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("grid_bt")

# ── Paires ────────────────────────────────────────────────────────────────────

# Top 10 — ultra-liquide (utilisé par DEEP mode)
GRID_PAIRS_CORE = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "LINK-USD", "ADA-USD", "LTC-USD", "AVAX-USD", "DOGE-USD",
]

# 15 paires mid-cap liquides (ajout pour MICRO mode — Vol > 50M$, Top 50 MCap)
GRID_PAIRS_MID = [
    "DOT-USD", "UNI-USD", "NEAR-USD", "ATOM-USD",
    "AAVE-USD", "INJ-USD", "SUI-USD", "ARB-USD", "OP-USD",
    "SEI-USD", "RENDER-USD", "FET-USD", "HBAR-USD", "ETC-USD",
    # Nouvelles mid-caps liquides
    "TIA-USD", "JUP-USD", "PEPE-USD", "BONK-USD", "WIF-USD",
    "FLOKI-USD", "GRT-USD", "LDO-USD",
]

# Toutes les paires pour le mode MICRO (25 au total)
GRID_PAIRS_ALL = GRID_PAIRS_CORE + GRID_PAIRS_MID

# Legacy alias
GRID_PAIRS = GRID_PAIRS_CORE

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Smart Infinity Grid 2.0")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)

    # Mode
    parser.add_argument(
        "--mode", type=str, default="single", choices=["single", "micro", "deep", "dual"],
        help="Mode: single (legacy), micro (H1 scalp), deep (H4 quality), dual (micro+deep combined)",
    )
    parser.add_argument("--micro-pct", type=float, default=0.60, help="Micro capital share in dual mode (default: 0.60)")
    parser.add_argument("--deep-pct", type=float, default=0.40, help="Deep capital share in dual mode (default: 0.40)")
    parser.add_argument("--monthly-stop", type=float, default=0.06, help="Monthly portfolio stop %% (default: 0.06)")

    # Activation (single mode only)
    parser.add_argument("--rsi-activation", type=float, default=35.0, help="RSI H1 activation (default: 35)")
    parser.add_argument("--rsi-entry", type=float, default=30.0, help="RSI H1 entry (default: 30)")
    parser.add_argument("--wick-ratio", type=float, default=1.5, help="Rejection wick ratio (default: 1.5)")

    # Grid
    parser.add_argument("--grid-levels", type=int, default=5, help="Max grid levels (default: 5)")

    # TP
    parser.add_argument("--tp1", type=float, default=0.008, help="TP1 %% from PMP (default: 0.008)")
    parser.add_argument("--tp2", type=float, default=0.015, help="TP2 %% from PMP (default: 0.015)")
    parser.add_argument("--tp3", type=float, default=0.025, help="TP3 %% from PMP (default: 0.025)")
    parser.add_argument("--tp1-exit", type=float, default=0.40, help="TP1 exit %% (default: 0.40)")
    parser.add_argument("--tp2-exit", type=float, default=0.35, help="TP2 exit %% (default: 0.35)")

    # Stop
    parser.add_argument("--max-dd", type=float, default=0.20, help="Max drawdown per cycle (default: 0.20)")
    parser.add_argument("--rsi-stop", type=float, default=25.0, help="RSI stop threshold (default: 25)")
    parser.add_argument("--rsi-stop-bars", type=int, default=12, help="RSI stop bars (default: 12)")
    parser.add_argument("--pmp-stop", type=float, default=0.04, help="PMP stop %% (default: 0.04, 0=disabled)")
    parser.add_argument("--ema-stop-break", type=float, default=0.02, help="EMA200 break %% for stop (default: 0.02)")
    parser.add_argument("--no-ema-stop", action="store_true", help="Disable EMA200 H4 stop")

    # Capital
    parser.add_argument("--cycle-pct", type=float, default=0.20, help="Capital per cycle %% (default: 0.20)")
    parser.add_argument("--max-cycles", type=int, default=3, help="Max simultaneous cycles (default: 3)")

    # Safety
    parser.add_argument("--cooldown", type=int, default=6, help="Cooldown bars H1 (default: 6)")
    parser.add_argument("--timeout", type=int, default=168, help="Cycle timeout bars H1 (default: 168)")
    parser.add_argument("--max-consec-loss", type=int, default=4, help="Max consecutive losses (default: 4)")
    parser.add_argument("--daily-limit", type=float, default=0.05, help="Daily loss limit (default: 0.05)")

    # Fees
    parser.add_argument("--fee", type=float, default=0.0, help="Fee per side (default: 0.0 maker-only)")
    parser.add_argument("--slippage", type=float, default=0.0002, help="Slippage (default: 0.0002)")

    # Timeframe (single mode)
    parser.add_argument("--interval", type=str, default="1h", help="Candle interval: 1h, 4h (default: 1h)")

    # Modes
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated pairs")
    args = parser.parse_args()

    # Dates
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=args.months * 30)

    # ── Dispatch par mode ──
    if args.mode in ("micro", "deep", "dual"):
        _run_mode(args, start, end)
    elif args.sensitivity:
        # Legacy single mode
        interval = args.interval
        logger.info("🔷 Smart Grid 2.0 BT — %s → %s (%s)", start.date(), end.date(), interval)
        candles_by_symbol = _download_candles(args.pairs, GRID_PAIRS, interval, start, end)
        base_cfg = _build_single_config(args)
        _run_sensitivity(candles_by_symbol, base_cfg, args)
    else:
        interval = args.interval
        logger.info("🔷 Smart Grid 2.0 BT — %s → %s (%s)", start.date(), end.date(), interval)
        candles_by_symbol = _download_candles(args.pairs, GRID_PAIRS, interval, start, end)
        base_cfg = _build_single_config(args)
        _run_single(candles_by_symbol, base_cfg, args)


def _download_candles(
    pairs_arg: str | None,
    default_pairs: list[str],
    interval: str,
    start: datetime,
    end: datetime,
) -> dict[str, list]:
    """Télécharge les klines pour une liste de paires."""
    logger.info("📥 Téléchargement klines %s…", interval)
    pairs = pairs_arg.split(",") if pairs_arg else default_pairs
    candles_by_symbol = {}
    for pair in pairs:
        candles = download_candles(pair, start, end, interval=interval)
        if candles:
            candles_by_symbol[pair] = candles
            logger.info("   ✅ %s : %d bougies %s", pair, len(candles), interval)
        else:
            logger.warning("   ⚠️ %s : aucune donnée", pair)

    if not candles_by_symbol:
        logger.error("❌ Aucune donnée — abandon")
        sys.exit(1)
    return candles_by_symbol


def _build_single_config(args) -> GridConfig:
    """Construit un GridConfig depuis les args CLI (mode single/legacy)."""
    return GridConfig(
        initial_balance=args.balance,
        rsi_activation=args.rsi_activation,
        rsi_entry=args.rsi_entry,
        rejection_wick_ratio=args.wick_ratio,
        grid_levels=args.grid_levels,
        tp1_pct=args.tp1,
        tp2_pct=args.tp2,
        tp3_pct=args.tp3,
        tp1_exit_pct=args.tp1_exit,
        tp2_exit_pct=args.tp2_exit,
        tp3_exit_pct=1.0 - args.tp1_exit - args.tp2_exit,
        max_drawdown_pct=args.max_dd,
        rsi_stop_threshold=args.rsi_stop,
        rsi_stop_bars=args.rsi_stop_bars,
        pmp_stop_pct=args.pmp_stop,
        ema_stop_break_pct=args.ema_stop_break,
        ema200_h4_stop=not args.no_ema_stop,
        capital_per_cycle_pct=args.cycle_pct,
        max_simultaneous_cycles=args.max_cycles,
        cooldown_bars=args.cooldown,
        cycle_timeout_bars=args.timeout,
        max_consecutive_losses=args.max_consec_loss,
        daily_loss_limit_pct=args.daily_limit,
        fee_pct=args.fee,
        slippage_pct=args.slippage,
    )


# ── Mode dispatch ──────────────────────────────────────────────────────────────


def _run_mode(args, start: datetime, end: datetime) -> None:
    """Dispatch MICRO / DEEP / DUAL modes avec presets."""
    mode = args.mode
    balance = args.balance

    if mode == "micro":
        logger.info("🔥 MODE MICRO DIP SCALP — %s → %s (H1, 25 paires)", start.date(), end.date())
        candles_h1 = _download_candles(args.pairs, GRID_PAIRS_ALL, "1h", start, end)
        cfg = micro_dip_config(balance)
        engine = GridEngine(candles_h1, cfg)
        result = engine.run()
        metrics = compute_grid_metrics(result)
        _print_report(result, metrics, label="MICRO DIP SCALP (H1)")
        _save_trades_csv(result, suffix="_micro")
        if not args.no_show:
            _plot_equity(result, metrics)

    elif mode == "deep":
        logger.info("🔵 MODE DEEP DIP — %s → %s (H4, 10 paires)", start.date(), end.date())
        candles_h4 = _download_candles(args.pairs, GRID_PAIRS_CORE, "4h", start, end)
        cfg = deep_dip_config(balance)
        engine = GridEngine(candles_h4, cfg)
        result = engine.run()
        metrics = compute_grid_metrics(result)
        _print_report(result, metrics, label="DEEP DIP (H4)")
        _save_trades_csv(result, suffix="_deep")
        if not args.no_show:
            _plot_equity(result, metrics)

    elif mode == "dual":
        logger.info(
            "🔷 MODE DUAL — MICRO (H1, %d%%) + DEEP (H4, %d%%) — %s → %s",
            int(args.micro_pct * 100), int(args.deep_pct * 100),
            start.date(), end.date(),
        )

        # Télécharger H1 pour MICRO (25 paires) et H4 pour DEEP (10 paires)
        candles_h1 = _download_candles(args.pairs, GRID_PAIRS_ALL, "1h", start, end)
        candles_h4 = _download_candles(args.pairs, GRID_PAIRS_CORE, "4h", start, end)

        micro_cfg = micro_dip_config(balance)
        deep_cfg = deep_dip_config(balance)

        dual = DualGridEngine(
            candles_h1=candles_h1,
            candles_h4=candles_h4,
            micro_cfg=micro_cfg,
            deep_cfg=deep_cfg,
            initial_balance=balance,
            micro_pct=args.micro_pct,
            deep_pct=args.deep_pct,
            monthly_stop_pct=args.monthly_stop,
        )
        result = dual.run()
        metrics = compute_dual_metrics(result)
        _print_dual_report(result, metrics)
        _save_dual_trades_csv(result)
        if not args.no_show:
            _plot_dual_equity(result, metrics)


# ── Run single ─────────────────────────────────────────────────────────────────


def _run_single(candles: dict, cfg: GridConfig, args) -> None:
    engine = GridEngine(candles, cfg)
    result = engine.run()
    metrics = compute_grid_metrics(result)
    _print_report(result, metrics)
    _save_trades_csv(result)

    if not args.no_show:
        _plot_equity(result, metrics)


# ── Sensitivity ────────────────────────────────────────────────────────────────


def _run_sensitivity(candles: dict, base_cfg: GridConfig, args) -> None:
    """Grille de sensibilité sur les paramètres les plus impactants."""
    tp1_vals = [0.005, 0.008, 0.010, 0.015]
    tp2_vals = [0.010, 0.015, 0.020, 0.025]
    tp3_vals = [0.020, 0.025, 0.030, 0.040]
    rsi_activations = [30.0, 35.0, 40.0]
    rsi_entries = [25.0, 28.0, 30.0, 33.0]
    cycle_pcts = [0.15, 0.20, 0.25]

    results = []

    # Grid sur TP seulement (combinatoire raisonnable)
    logger.info("🔬 Sensibilité Phase 1 : TP levels × RSI")
    total = len(tp1_vals) * len(tp3_vals) * len(rsi_entries)
    logger.info("   %d combinaisons", total)

    done = 0
    for tp1 in tp1_vals:
        for tp3 in tp3_vals:
            if tp3 <= tp1:
                continue
            tp2 = (tp1 + tp3) / 2  # TP2 au milieu
            for rsi_e in rsi_entries:
                cfg = GridConfig(
                    initial_balance=base_cfg.initial_balance,
                    rsi_activation=base_cfg.rsi_activation,
                    rsi_entry=rsi_e,
                    rejection_wick_ratio=base_cfg.rejection_wick_ratio,
                    grid_levels=base_cfg.grid_levels,
                    tp1_pct=tp1,
                    tp2_pct=tp2,
                    tp3_pct=tp3,
                    tp1_exit_pct=base_cfg.tp1_exit_pct,
                    tp2_exit_pct=base_cfg.tp2_exit_pct,
                    tp3_exit_pct=base_cfg.tp3_exit_pct,
                    max_drawdown_pct=base_cfg.max_drawdown_pct,
                    rsi_stop_threshold=base_cfg.rsi_stop_threshold,
                    rsi_stop_bars=base_cfg.rsi_stop_bars,
                    capital_per_cycle_pct=base_cfg.capital_per_cycle_pct,
                    max_simultaneous_cycles=base_cfg.max_simultaneous_cycles,
                    cooldown_bars=base_cfg.cooldown_bars,
                    cycle_timeout_bars=base_cfg.cycle_timeout_bars,
                    max_consecutive_losses=base_cfg.max_consecutive_losses,
                    daily_loss_limit_pct=base_cfg.daily_loss_limit_pct,
                    fee_pct=base_cfg.fee_pct,
                    slippage_pct=base_cfg.slippage_pct,
                )
                engine = GridEngine(candles, cfg)
                result = engine.run()
                m = compute_grid_metrics(result)
                results.append((m, cfg))
                done += 1
                if done % 20 == 0:
                    logger.info("   %d/%d done…", done, total)

    # Phase 2 : Capital allocation
    logger.info("🔬 Sensibilité Phase 2 : Capital allocation")
    for cpct in cycle_pcts:
        for rsi_a in rsi_activations:
            cfg = GridConfig(
                initial_balance=base_cfg.initial_balance,
                rsi_activation=rsi_a,
                rsi_entry=base_cfg.rsi_entry,
                rejection_wick_ratio=base_cfg.rejection_wick_ratio,
                grid_levels=base_cfg.grid_levels,
                tp1_pct=base_cfg.tp1_pct,
                tp2_pct=base_cfg.tp2_pct,
                tp3_pct=base_cfg.tp3_pct,
                tp1_exit_pct=base_cfg.tp1_exit_pct,
                tp2_exit_pct=base_cfg.tp2_exit_pct,
                tp3_exit_pct=base_cfg.tp3_exit_pct,
                max_drawdown_pct=base_cfg.max_drawdown_pct,
                rsi_stop_threshold=base_cfg.rsi_stop_threshold,
                rsi_stop_bars=base_cfg.rsi_stop_bars,
                capital_per_cycle_pct=cpct,
                max_simultaneous_cycles=base_cfg.max_simultaneous_cycles,
                cooldown_bars=base_cfg.cooldown_bars,
                cycle_timeout_bars=base_cfg.cycle_timeout_bars,
                max_consecutive_losses=base_cfg.max_consecutive_losses,
                daily_loss_limit_pct=base_cfg.daily_loss_limit_pct,
                fee_pct=base_cfg.fee_pct,
                slippage_pct=base_cfg.slippage_pct,
            )
            engine = GridEngine(candles, cfg)
            result = engine.run()
            m = compute_grid_metrics(result)
            results.append((m, cfg))

    results.sort(key=lambda x: x[0]["total_return"] if x[0]["n_trades"] >= 5 else -99, reverse=True)

    sep = "═" * 148
    print(f"\n{sep}")
    print("  🔬 SENSIBILITÉ — Smart Infinity Grid 2.0")
    print(sep)
    print(
        f"  {'TP1':>5s} | {'TP2':>5s} | {'TP3':>5s} | {'RSI_E':>5s} | {'Cyc%':>5s} | "
        f"{'Trades':>6s} | {'T/day':>5s} | {'WR':>6s} | {'PF':>6s} | "
        f"{'Return':>8s} | {'Mo/ret':>6s} | {'MaxDD':>7s} | {'Sharpe':>7s} | "
        f"{'AvgPnL':>8s} | {'AvgLv':>5s} | {'TP1%':>5s} | {'TP3%':>5s}"
    )
    print("  " + "─" * 144)

    for m, cfg in results[:40]:
        if m["n_trades"] < 3:
            continue
        print(
            f"  {cfg.tp1_pct:5.3f} | {cfg.tp2_pct:5.3f} | {cfg.tp3_pct:5.3f} | "
            f"{cfg.rsi_entry:5.1f} | {cfg.capital_per_cycle_pct:5.2f} | "
            f"{m['n_trades']:6d} | {m['trades_per_day']:5.2f} | {m['win_rate']:6.1%} | "
            f"{m['profit_factor']:6.2f} | "
            f"{m['total_return']:+7.1%} | {m['monthly_return']:+5.1%} | "
            f"{m['max_drawdown']:7.1%} | {m['sharpe']:7.2f} | "
            f"${m['avg_pnl_usd']:+7.2f} | {m['avg_levels']:4.1f} | "
            f"{m['tp1_hit_pct']:5.1%} | {m['tp3_hit_pct']:5.1%}"
        )

    print(f"\n{sep}\n")

    # Best
    viable = [(m, c) for m, c in results if m["n_trades"] >= 5]
    if viable:
        best_m, best_cfg = viable[0]
        print(
            f"  🏆 BEST: TP1={best_cfg.tp1_pct:.3f}, TP2={best_cfg.tp2_pct:.3f}, "
            f"TP3={best_cfg.tp3_pct:.3f}, RSI_entry={best_cfg.rsi_entry:.0f}, "
            f"Cycle%={best_cfg.capital_per_cycle_pct:.0%}"
        )
        print(
            f"     Return={best_m['total_return']:+.1%}, Mo={best_m['monthly_return']:+.1%}/m, "
            f"WR={best_m['win_rate']:.0%}, PF={best_m['profit_factor']:.2f}, "
            f"MaxDD={best_m['max_drawdown']:.1%}, Sharpe={best_m['sharpe']:.2f}, "
            f"Trades={best_m['n_trades']} ({best_m['trades_per_day']:.2f}/day)\n"
        )


# ── Reporting ──────────────────────────────────────────────────────────────────


def _print_report(result: GridResult, m: dict, label: str = "SMART INFINITY GRID 2.0") -> None:
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  🔷 {label} — BACKTEST REPORT")
    print(sep)

    print(f"\n  Période     : {result.start_date.date()} → {result.end_date.date()} ({m['days']} jours)")
    print(f"  Paires      : {', '.join(result.pairs)}")
    print(f"  Capital     : ${result.initial_balance:,.0f} → ${result.final_equity:,.2f}")
    print(f"  Fees        : {result.config.fee_pct:.2%} (maker-only)")

    print(f"\n  ── Performance ──")
    print(f"  Return      : {m['total_return']:+.2%}")
    print(f"  Mensuel     : {m['monthly_return']:+.2%}")
    print(f"  CAGR        : {m['cagr']:+.2%}")
    print(f"  Max DD      : {m['max_drawdown']:.2%}")
    print(f"  Sharpe      : {m['sharpe']:.2f}")
    print(f"  Sortino     : {m['sortino']:.2f}")

    print(f"\n  ── Trades (cycles) ──")
    print(f"  Cycles      : {m['n_trades']}")
    print(f"  Cycles/jour : {m['trades_per_day']:.2f}")
    print(f"  Win Rate    : {m['win_rate']:.1%}")
    print(f"  Profit F.   : {m['profit_factor']:.2f}")
    print(f"  Avg PnL     : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.3%})")
    print(f"  Avg Hold    : {m['avg_hold_bars']:.0f} bars H1 ({m['avg_hold_bars'] / 24:.1f} jours)")
    print(f"  Avg Levels  : {m['avg_levels']:.1f}")

    print(f"\n  ── Take-Profits ──")
    print(f"  TP1 hit     : {m['tp1_hit_pct']:.1%}")
    print(f"  TP2 hit     : {m['tp2_hit_pct']:.1%}")
    print(f"  TP3 hit     : {m['tp3_hit_pct']:.1%}")

    print(f"\n  ── Pipeline ──")
    print(f"  Activations : {m['n_activations']}")
    print(f"  Entries     : {m['n_entries']}")
    print(f"  TP complete : {m['n_tp_complete']}")
    print(f"  Global stops: {m['n_global_stops']}")
    print(f"  Timeouts    : {m['n_timeouts']}")
    print(f"  Filtered    : {m['n_filtered']}")

    if m.get("best_trade"):
        bt = m["best_trade"]
        wt = m["worst_trade"]
        print(f"\n  Best Cycle  : ${bt.pnl_usd:+.2f} ({bt.symbol}, {bt.n_levels} lvl, {bt.exit_reason})")
        print(f"  Worst Cycle : ${wt.pnl_usd:+.2f} ({wt.symbol}, {wt.n_levels} lvl, {wt.exit_reason})")

    # Par paire
    by_pair = m.get("by_pair", {})
    if by_pair:
        print(f"\n  ── Par paire ──")
        print(f"  {'Pair':<12s} | {'N':>5s} | {'WR':>6s} | {'PF':>6s} | {'PnL':>9s} | {'AvgLv':>5s} | {'AvgH':>5s}")
        print(f"  " + "─" * 64)
        for pair, stats in sorted(by_pair.items(), key=lambda x: x[1]["pnl"], reverse=True):
            print(
                f"  {pair:<12s} | {stats['n']:5d} | {stats['wr']:6.1%} | "
                f"{stats['pf']:6.2f} | ${stats['pnl']:+8.2f} | "
                f"{stats['avg_levels']:4.1f} | {stats['avg_hold']:4.0f}h"
            )

    # Par exit reason
    by_exit = m.get("by_exit", {})
    if by_exit:
        print(f"\n  ── Par exit ──")
        print(f"  {'Reason':<20s} | {'N':>5s} | {'WR':>6s} | {'PnL':>9s}")
        print(f"  " + "─" * 48)
        for reason, stats in sorted(by_exit.items()):
            print(f"  {reason:<20s} | {stats['n']:5d} | {stats['wr']:6.1%} | ${stats['pnl']:+8.2f}")

    # Par nombre de niveaux
    by_levels = m.get("by_levels", {})
    if by_levels:
        print(f"\n  ── Par niveaux remplis ──")
        print(f"  {'Levels':<8s} | {'N':>5s} | {'WR':>6s} | {'PF':>6s} | {'PnL':>9s}")
        print(f"  " + "─" * 44)
        for lvl, stats in sorted(by_levels.items()):
            print(
                f"  {lvl:<8s} | {stats['n']:5d} | {stats['wr']:6.1%} | "
                f"{stats['pf']:6.2f} | ${stats['pnl']:+8.2f}"
            )

    print(f"\n{sep}\n")


def _save_trades_csv(result: GridResult, suffix: str = "") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"grid_trades{suffix}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "n_levels", "initial_entry", "pmp", "total_size",
            "total_cost", "total_proceeds",
            "entry_time", "exit_time", "pnl_usd", "pnl_pct",
            "exit_reason", "tp1_hit", "tp2_hit", "tp3_hit",
            "hold_bars", "max_unrealized_dd",
        ])
        for t in result.trades:
            w.writerow([
                t.symbol, t.n_levels,
                f"{t.initial_entry:.8f}", f"{t.pmp:.8f}",
                f"{t.total_size:.8f}", f"{t.total_cost:.4f}",
                f"{t.total_proceeds:.4f}",
                datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc).isoformat(),
                f"{t.pnl_usd:.4f}", f"{t.pnl_pct:.6f}",
                t.exit_reason, t.tp1_hit, t.tp2_hit, t.tp3_hit,
                t.hold_bars, f"{t.max_unrealized_dd:.6f}",
            ])
    logger.info("💾 Trades sauvegardés : %s (%d cycles)", path, len(result.trades))


def _save_dual_trades_csv(result: DualGridResult) -> None:
    """Sauvegarde les trades combinés dual-mode."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "grid_trades_dual.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "mode", "symbol", "n_levels", "initial_entry", "pmp", "total_size",
            "total_cost", "total_proceeds",
            "entry_time", "exit_time", "pnl_usd", "pnl_pct",
            "exit_reason", "tp1_hit", "tp2_hit", "tp3_hit",
            "hold_bars", "max_unrealized_dd",
        ])
        for t in result.combined_trades:
            mode_tag = "MICRO" if t.symbol.startswith("[M]") else "DEEP"
            sym = t.symbol[3:]  # Remove [M] or [D] prefix
            w.writerow([
                mode_tag, sym, t.n_levels,
                f"{t.initial_entry:.8f}", f"{t.pmp:.8f}",
                f"{t.total_size:.8f}", f"{t.total_cost:.4f}",
                f"{t.total_proceeds:.4f}",
                datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc).isoformat(),
                f"{t.pnl_usd:.4f}", f"{t.pnl_pct:.6f}",
                t.exit_reason, t.tp1_hit, t.tp2_hit, t.tp3_hit,
                t.hold_bars, f"{t.max_unrealized_dd:.6f}",
            ])
    logger.info("💾 Trades dual sauvegardés : %s (%d cycles)", path, len(result.combined_trades))


def _print_dual_report(result: DualGridResult, m: dict) -> None:
    """Rapport dual-mode combiné."""
    sep = "═" * 80
    print(f"\n{sep}")
    print("  🔷 SMART INFINITY GRID 2.0 — DUAL MODE REPORT")
    print(f"  🔥 MICRO DIP (H1) + 🔵 DEEP DIP (H4)")
    print(sep)

    print(f"\n  Période     : {result.start_date.date()} → {result.end_date.date()} ({m['days']} jours)")
    print(f"  Capital     : ${result.initial_balance:,.0f} → ${result.final_equity:,.2f}")
    print(f"  Fees        : 0.00% (maker-only)")

    print(f"\n  ── Performance combinée ──")
    print(f"  Return      : {m['total_return']:+.2%}")
    print(f"  Mensuel     : {m['monthly_return']:+.2%}")
    print(f"  CAGR        : {m['cagr']:+.2%}")
    print(f"  Max DD      : {m['max_drawdown']:.2%}")
    print(f"  Sharpe      : {m['sharpe']:.2f}")
    print(f"  Sortino     : {m['sortino']:.2f}")

    print(f"\n  ── Trades ──")
    print(f"  Total       : {m['n_trades']} ({m['trades_per_day']:.2f}/jour)")
    print(f"    🔥 MICRO  : {m['n_micro']} trades → ${m['micro_pnl']:+.2f}")
    print(f"    🔵 DEEP   : {m['n_deep']} trades → ${m['deep_pnl']:+.2f}")
    print(f"  Win Rate    : {m['win_rate']:.1%}")
    print(f"  Profit F.   : {m['profit_factor']:.2f}")
    print(f"  Avg PnL     : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.3%})")

    if m.get("best_trade"):
        bt = m["best_trade"]
        wt = m["worst_trade"]
        print(f"\n  Best Trade  : ${bt.pnl_usd:+.2f} ({bt.symbol}, {bt.n_levels} lvl, {bt.exit_reason})")
        print(f"  Worst Trade : ${wt.pnl_usd:+.2f} ({wt.symbol}, {wt.n_levels} lvl, {wt.exit_reason})")

    print(f"\n  ── Stop mensuel -6% ──")
    print(f"  Mois stoppés: {m['monthly_stop_events']}")

    # Détail par mode
    micro_r = result.micro_result
    deep_r = result.deep_result
    micro_m = compute_grid_metrics(micro_r)
    deep_m = compute_grid_metrics(deep_r)

    print(f"\n  ── 🔥 MICRO DIP (H1) ──")
    print(f"  Paires      : {', '.join(result.pairs_micro[:10])}{'…' if len(result.pairs_micro) > 10 else ''} ({len(result.pairs_micro)} total)")
    print(f"  Capital     : ${micro_r.initial_balance:,.0f} → ${micro_r.final_equity:,.2f} ({micro_m['total_return']:+.2%})")
    print(f"  Cycles      : {micro_m['n_trades']} ({micro_m['trades_per_day']:.2f}/jour)")
    print(f"  WR: {micro_m['win_rate']:.1%} | PF: {micro_m['profit_factor']:.2f} | MaxDD: {micro_m['max_drawdown']:.2%}")

    print(f"\n  ── 🔵 DEEP DIP (H4) ──")
    print(f"  Paires      : {', '.join(result.pairs_deep)} ({len(result.pairs_deep)} total)")
    print(f"  Capital     : ${deep_r.initial_balance:,.0f} → ${deep_r.final_equity:,.2f} ({deep_m['total_return']:+.2%})")
    print(f"  Cycles      : {deep_m['n_trades']} ({deep_m['trades_per_day']:.2f}/jour)")
    print(f"  WR: {deep_m['win_rate']:.1%} | PF: {deep_m['profit_factor']:.2f} | MaxDD: {deep_m['max_drawdown']:.2%}")

    print(f"\n{sep}\n")


def _plot_dual_equity(result: DualGridResult, metrics: dict) -> None:
    """Graphique equity dual-mode."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib non disponible — skip plot")
        return

    if not result.combined_equity:
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Equity curves
    ax1 = axes[0]

    # Micro
    if result.micro_result.equity_curve:
        m_dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in result.micro_result.equity_curve]
        m_eq = [eq for _, eq in result.micro_result.equity_curve]
        ax1.plot(m_dates, m_eq, color="#FF5722", linewidth=0.8, alpha=0.7, label="🔥 MICRO")

    # Deep
    if result.deep_result.equity_curve:
        d_dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in result.deep_result.equity_curve]
        d_eq = [eq for _, eq in result.deep_result.equity_curve]
        ax1.plot(d_dates, d_eq, color="#2196F3", linewidth=0.8, alpha=0.7, label="🔵 DEEP")

    # Combined
    c_dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in result.combined_equity]
    c_eq = [eq for _, eq in result.combined_equity]
    ax1.plot(c_dates, c_eq, color="#4CAF50", linewidth=1.5, label="🔷 COMBINED")
    ax1.axhline(result.initial_balance, color="gray", linestyle="--", alpha=0.5)

    ax1.set_ylabel("Equity ($)")
    ax1.set_title(
        f"Smart Grid 2.0 DUAL — {result.start_date.date()} → {result.end_date.date()}\n"
        f"Return: {metrics['total_return']:+.1%} | Sharpe: {metrics['sharpe']:.2f} | "
        f"MaxDD: {metrics['max_drawdown']:.1%} | WR: {metrics['win_rate']:.0%} | "
        f"Trades: {metrics['n_trades']} (M:{metrics['n_micro']}+D:{metrics['n_deep']})",
        fontsize=11,
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    peak = result.initial_balance
    dd_curve = []
    for eq in c_eq:
        peak = max(peak, eq)
        dd_curve.append((eq - peak) / peak if peak > 0 else 0)
    ax2.fill_between(c_dates, 0, dd_curve, color="red", alpha=0.3)
    ax2.plot(c_dates, dd_curve, color="red", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = OUTPUT_DIR / "grid_equity_dual.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("📊 Graphique dual sauvegardé : %s", chart_path)
    plt.show()


def _plot_equity(result: GridResult, metrics: dict) -> None:
    """Génère le graphique d'equity."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib non disponible — skip plot")
        return

    if not result.equity_curve:
        return

    dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in result.equity_curve]
    equities = [eq for _, eq in result.equity_curve]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax1 = axes[0]
    ax1.plot(dates, equities, color="#2196F3", linewidth=1.2, label="Equity")
    ax1.axhline(result.initial_balance, color="gray", linestyle="--", alpha=0.5, label=f"Initial ${result.initial_balance:,.0f}")
    ax1.fill_between(dates, result.initial_balance, equities, alpha=0.1, color="#2196F3")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(
        f"Smart Infinity Grid 2.0 — {result.start_date.date()} → {result.end_date.date()}\n"
        f"Return: {metrics['total_return']:+.1%} | Sharpe: {metrics['sharpe']:.2f} | "
        f"MaxDD: {metrics['max_drawdown']:.1%} | WR: {metrics['win_rate']:.0%} | "
        f"Trades: {metrics['n_trades']}",
        fontsize=11,
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    peak = result.initial_balance
    dd_curve = []
    for eq in equities:
        peak = max(peak, eq)
        dd_curve.append((eq - peak) / peak if peak > 0 else 0)
    ax2.fill_between(dates, 0, dd_curve, color="red", alpha=0.3)
    ax2.plot(dates, dd_curve, color="red", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    # Sauvegarder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_path = OUTPUT_DIR / "grid_equity.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("📊 Graphique sauvegardé : %s", chart_path)

    plt.show()


if __name__ == "__main__":
    main()
