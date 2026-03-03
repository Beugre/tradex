#!/usr/bin/env python
"""
Backtest Intraday Momentum Continuation v1.0

Stratégie :
  Compression → Impulsion (body≥0.5%, vol≥2×MA, ADX>18)
  → Pullback 30-50% (RSI 45-60, EMA20/50 support)
  → Continuation entry (close > prev high + volume)
  → TP 0.6% fixe, trail après +0.5%, SL swing low (0.35-0.70%)

Usage :
    # Base — 12 mois M5
    PYTHONPATH=. python backtest/run_backtest_momentum_cont.py --months 12 --no-show

    # 6 mois
    PYTHONPATH=. python backtest/run_backtest_momentum_cont.py --months 6 --no-show

    # Sensitivity
    PYTHONPATH=. python backtest/run_backtest_momentum_cont.py --months 12 --sensitivity --no-show

    # Long only
    PYTHONPATH=. python backtest/run_backtest_momentum_cont.py --months 12 --no-short --no-show
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
from backtest.simulator_momentum_cont import (
    MomentumContConfig,
    MomentumContEngine,
    MCResult,
    MCTrade,
    compute_mc_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mc_bt")

# ── Paires — ultra-liquide uniquement ─────────────────────────────────────────

MC_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "DOGE-USD", "AVAX-USD", "LINK-USD", "ADA-USD", "LTC-USD",
]

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Momentum Continuation")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--risk", type=float, default=0.005, help="Risk per trade (0.005=0.5%%)")

    # Impulse
    parser.add_argument("--impulse-body", type=float, default=0.005, help="Min body %% (0.005=0.5%%)")
    parser.add_argument("--impulse-vol", type=float, default=2.0, help="Volume mult for impulse")
    parser.add_argument("--adx-min", type=float, default=18.0, help="Min ADX")

    # Pullback
    parser.add_argument("--retrace-min", type=float, default=0.30, help="Min retrace %%")
    parser.add_argument("--retrace-max", type=float, default=0.50, help="Max retrace %%")
    parser.add_argument("--rsi-min", type=float, default=45.0)
    parser.add_argument("--rsi-max", type=float, default=60.0)
    parser.add_argument("--pullback-bars", type=int, default=30, help="Max bars post-impulse")

    # Risk
    parser.add_argument("--sl-min", type=float, default=0.0035, help="Min SL %%")
    parser.add_argument("--sl-max", type=float, default=0.007, help="Max SL %%")
    parser.add_argument("--tp", type=float, default=0.006, help="TP %% (0.006=0.6%%)")
    parser.add_argument("--trail-trigger", type=float, default=0.005, help="Trail trigger %%")
    parser.add_argument("--trail-dist", type=float, default=0.002, help="Trail distance %%")

    # Safety
    parser.add_argument("--max-consec-loss", type=int, default=5)
    parser.add_argument("--daily-limit", type=float, default=0.02, help="Daily loss limit %%")
    parser.add_argument("--max-pos", type=int, default=3, help="Max simultaneous positions")
    parser.add_argument("--cooldown", type=int, default=12, help="Cooldown bars M5")
    parser.add_argument("--max-pos-pct", type=float, default=0.40, help="Max %% capital per position")

    # Fees
    parser.add_argument("--fee", type=float, default=0.00075, help="Fee per side (0.00075=0.075%%)")
    parser.add_argument("--slippage", type=float, default=0.0001, help="Slippage per side (0.0001=0.01%%)")

    # Modes
    parser.add_argument("--no-short", action="store_true", help="Long only")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--pairs", type=str, default=None, help="Comma-separated pairs (default: all)")
    args = parser.parse_args()

    # Dates
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=args.months * 30)

    logger.info("⚡ Momentum Continuation BT — %s → %s (M5)", start.date(), end.date())

    # Téléchargement M5
    logger.info("📥 Téléchargement klines 5m…")
    pairs = args.pairs.split(",") if args.pairs else MC_PAIRS
    candles_by_symbol = {}
    for pair in pairs:
        candles = download_candles(pair, start, end, interval="5m")
        if candles:
            candles_by_symbol[pair] = candles
            logger.info("   ✅ %s : %d bougies M5", pair, len(candles))
        else:
            logger.warning("   ⚠️ %s : aucune donnée", pair)

    if not candles_by_symbol:
        logger.error("❌ Aucune donnée — abandon")
        sys.exit(1)

    base_cfg = MomentumContConfig(
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        impulse_body_min_pct=args.impulse_body,
        impulse_vol_mult=args.impulse_vol,
        adx_min=args.adx_min,
        pullback_retrace_min=args.retrace_min,
        pullback_retrace_max=args.retrace_max,
        rsi_pullback_min=args.rsi_min,
        rsi_pullback_max=args.rsi_max,
        pullback_max_bars=args.pullback_bars,
        sl_min_pct=args.sl_min,
        sl_max_pct=args.sl_max,
        tp_pct=args.tp,
        trail_trigger_pct=args.trail_trigger,
        trail_distance_pct=args.trail_dist,
        max_positions=args.max_pos,
        max_position_pct=args.max_pos_pct,
        max_consecutive_losses=args.max_consec_loss,
        daily_loss_limit_pct=args.daily_limit,
        cooldown_bars=args.cooldown,
        allow_short=not args.no_short,
        fee_pct=args.fee,
        slippage_pct=args.slippage,
    )

    if args.sensitivity:
        _run_sensitivity(candles_by_symbol, base_cfg, args)
    else:
        _run_single(candles_by_symbol, base_cfg, args)


# ── Run single ─────────────────────────────────────────────────────────────────


def _run_single(candles: dict, cfg: MomentumContConfig, args) -> None:
    engine = MomentumContEngine(candles, cfg)
    result = engine.run()
    metrics = compute_mc_metrics(result)
    _print_report(result, metrics)
    _save_trades_csv(result)


# ── Sensitivity ────────────────────────────────────────────────────────────────


def _run_sensitivity(candles: dict, base_cfg: MomentumContConfig, args) -> None:
    # Grille ciblée sur les paramètres les plus impactants
    tp_pcts = [0.008, 0.010, 0.015, 0.020, 0.030]
    sl_maxs = [0.005, 0.006, 0.008, 0.010]
    trail_triggers = [0.003, 0.004, 0.005, 0.007]
    trail_dists = [0.0010, 0.0015, 0.0020, 0.0030]

    results = []
    total = len(tp_pcts) * len(sl_maxs) * len(trail_triggers) * len(trail_dists)
    logger.info("🔬 Sensibilité : %d combinaisons", total)

    done = 0
    for tp in tp_pcts:
        for sl in sl_maxs:
            for trail_t in trail_triggers:
                for trail_d in trail_dists:
                    if trail_d >= trail_t:
                        continue  # Trail distance doit être < trigger
                    cfg = MomentumContConfig(
                        initial_balance=base_cfg.initial_balance,
                        risk_per_trade=base_cfg.risk_per_trade,
                        impulse_body_min_pct=base_cfg.impulse_body_min_pct,
                        impulse_vol_mult=base_cfg.impulse_vol_mult,
                        adx_min=base_cfg.adx_min,
                        pullback_retrace_min=base_cfg.pullback_retrace_min,
                        pullback_retrace_max=base_cfg.pullback_retrace_max,
                        rsi_pullback_min=base_cfg.rsi_pullback_min,
                        rsi_pullback_max=base_cfg.rsi_pullback_max,
                        pullback_max_bars=base_cfg.pullback_max_bars,
                        sl_min_pct=base_cfg.sl_min_pct,
                        sl_max_pct=sl,
                        tp_pct=tp,
                        trail_trigger_pct=trail_t,
                        trail_distance_pct=trail_d,
                        max_positions=base_cfg.max_positions,
                        max_consecutive_losses=base_cfg.max_consecutive_losses,
                        daily_loss_limit_pct=base_cfg.daily_loss_limit_pct,
                        cooldown_bars=base_cfg.cooldown_bars,
                        allow_short=base_cfg.allow_short,
                        fee_pct=base_cfg.fee_pct,
                        slippage_pct=base_cfg.slippage_pct,
                    )
                    engine = MomentumContEngine(candles, cfg)
                    result = engine.run()
                    m = compute_mc_metrics(result)
                    results.append((m, cfg))
                    done += 1
                    if done % 50 == 0:
                        logger.info("   %d/%d done…", done, total)

    results.sort(key=lambda x: x[0]["total_return"] if x[0]["n_trades"] >= 20 else -99, reverse=True)

    sep = "═" * 136
    print(f"\n{sep}")
    print("  🔬 SENSIBILITÉ — Momentum Continuation")
    print(sep)
    print(
        f"  {'TP':>5s} | {'SL':>5s} | {'Trail':>5s} | {'TDist':>5s} | "
        f"{'Trades':>6s} | {'T/day':>5s} | {'WR':>6s} | {'PF':>6s} | "
        f"{'Return':>8s} | {'Mo/ret':>6s} | {'MaxDD':>7s} | {'Sharpe':>7s} | "
        f"{'AvgPnL':>8s} | {'AvgH':>5s}"
    )
    print("  " + "─" * 132)

    for m, cfg in results[:60]:
        if m["n_trades"] < 10:
            continue
        print(
            f"  {cfg.tp_pct:5.3f} | {cfg.sl_max_pct:5.3f} | "
            f"{cfg.trail_trigger_pct:5.3f} | {cfg.trail_distance_pct:5.4f} | "
            f"{m['n_trades']:6d} | {m['trades_per_day']:5.1f} | {m['win_rate']:6.1%} | "
            f"{m['profit_factor']:6.2f} | "
            f"{m['total_return']:+7.1%} | {m['monthly_return']:+5.1%} | "
            f"{m['max_drawdown']:7.1%} | {m['sharpe']:7.2f} | "
            f"${m['avg_pnl_usd']:+7.2f} | {m['avg_hold_bars']:4.0f}b"
        )

    print(f"\n{sep}\n")

    # Best
    viable = [(m, c) for m, c in results if m["n_trades"] >= 20]
    if viable:
        best_m, best_cfg = viable[0]
        print(
            f"  🏆 BEST: TP={best_cfg.tp_pct:.3f}, SL<={best_cfg.sl_max_pct:.3f}, "
            f"TrailTrig={best_cfg.trail_trigger_pct:.3f}, TrailDist={best_cfg.trail_distance_pct:.4f}"
        )
        print(
            f"     Return={best_m['total_return']:+.1%}, Mo={best_m['monthly_return']:+.1%}/m, "
            f"WR={best_m['win_rate']:.0%}, PF={best_m['profit_factor']:.2f}, "
            f"MaxDD={best_m['max_drawdown']:.1%}, Sharpe={best_m['sharpe']:.2f}, "
            f"Trades={best_m['n_trades']} ({best_m['trades_per_day']:.1f}/day)\n"
        )


# ── Reporting ──────────────────────────────────────────────────────────────────


def _print_report(result: MCResult, m: dict) -> None:
    sep = "═" * 72
    print(f"\n{sep}")
    print("  ⚡ MOMENTUM CONTINUATION — BACKTEST REPORT")
    print(sep)

    print(f"\n  Période     : {result.start_date.date()} → {result.end_date.date()} ({m['days']} jours)")
    print(f"  Paires      : {', '.join(result.pairs)}")
    print(f"  Capital     : ${result.initial_balance:,.0f} → ${result.final_equity:,.2f}")

    print(f"\n  ── Performance ──")
    print(f"  Return      : {m['total_return']:+.2%}")
    print(f"  Mensuel     : {m['monthly_return']:+.2%}")
    print(f"  CAGR        : {m['cagr']:+.2%}")
    print(f"  Max DD      : {m['max_drawdown']:.2%}")
    print(f"  Sharpe      : {m['sharpe']:.2f}")
    print(f"  Sortino     : {m['sortino']:.2f}")

    print(f"\n  ── Trades ──")
    print(f"  Trades      : {m['n_trades']}")
    print(f"  Trades/jour : {m['trades_per_day']:.1f}")
    print(f"  Win Rate    : {m['win_rate']:.1%}")
    print(f"  Profit F.   : {m['profit_factor']:.2f}")
    print(f"  Avg PnL     : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.3%})")
    print(f"  Avg Hold    : {m['avg_hold_bars']:.0f} bars ({m['avg_hold_bars'] * 5 / 60:.1f}h)")

    print(f"\n  ── Pipeline ──")
    print(f"  Impulses    : {m['n_impulses']}")
    print(f"  Pullbacks   : {m['n_pullbacks']}")
    print(f"  Entries     : {m['n_entries']}")
    print(f"  Filtered M  : {m['n_filtered_macro']} (macro filter)")
    print(f"  Filtered S  : {m['n_filtered_safety']} (safety)")

    if m.get("best_trade"):
        bt = m["best_trade"]
        wt = m["worst_trade"]
        print(f"\n  Best Trade  : ${bt.pnl_usd:+.2f} ({bt.symbol} {bt.side})")
        print(f"  Worst Trade : ${wt.pnl_usd:+.2f} ({wt.symbol} {wt.side})")

    # Par paire
    by_pair = m.get("by_pair", {})
    if by_pair:
        print(f"\n  ── Par paire ──")
        print(f"  {'Pair':<12s} | {'N':>5s} | {'WR':>6s} | {'PF':>6s} | {'PnL':>9s} | {'Avg%':>7s} | {'AvgH':>5s}")
        print(f"  " + "─" * 64)
        for pair, stats in sorted(by_pair.items(), key=lambda x: x[1]["pnl"], reverse=True):
            print(
                f"  {pair:<12s} | {stats['n']:5d} | {stats['wr']:6.1%} | "
                f"{stats['pf']:6.2f} | ${stats['pnl']:+8.2f} | "
                f"{stats['avg_pct']:+6.3%} | {stats['avg_hold']:4.0f}b"
            )

    # Par exit reason
    by_exit = m.get("by_exit", {})
    if by_exit:
        print(f"\n  ── Par exit ──")
        print(f"  {'Reason':<12s} | {'N':>5s} | {'WR':>6s} | {'PnL':>9s}")
        print(f"  " + "─" * 40)
        for reason, stats in sorted(by_exit.items()):
            print(f"  {reason:<12s} | {stats['n']:5d} | {stats['wr']:6.1%} | ${stats['pnl']:+8.2f}")

    # Par side
    by_side = m.get("by_side", {})
    if by_side:
        print(f"\n  ── Par side ──")
        for side, stats in sorted(by_side.items()):
            print(
                f"  {side:<6s} : {stats['n']} trades, WR={stats['wr']:.1%}, "
                f"PF={stats['pf']:.2f}, PnL=${stats['pnl']:+.2f}"
            )

    print(f"\n{sep}\n")


def _save_trades_csv(result: MCResult) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "momentum_cont_trades.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "side", "entry_price", "exit_price", "size",
            "entry_time", "exit_time", "pnl_usd", "pnl_pct",
            "exit_reason", "impulse_body_pct", "impulse_vol_ratio",
            "retrace_pct", "hold_bars",
        ])
        for t in result.trades:
            w.writerow([
                t.symbol, t.side, f"{t.entry_price:.8f}", f"{t.exit_price:.8f}",
                f"{t.size:.8f}",
                datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc).isoformat(),
                datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc).isoformat(),
                f"{t.pnl_usd:.4f}", f"{t.pnl_pct:.6f}",
                t.exit_reason, f"{t.impulse_body_pct:.6f}",
                f"{t.impulse_vol_ratio:.2f}", f"{t.retrace_pct:.4f}",
                t.hold_bars,
            ])
    logger.info("💾 Trades sauvegardés : %s (%d trades)", path, len(result.trades))


if __name__ == "__main__":
    main()
