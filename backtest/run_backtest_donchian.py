#!/usr/bin/env python
"""
Backtest Donchian Trend Following — Daily timeframe.

Stratégie décorrélée :
  - Trail Range (H4) : Dow Theory, breakout de structure → jours/semaines
  - CrashBot (1m)    : event-driven sur crashs extrêmes → minutes
  - Momentum (15m)   : BB squeeze → MACD breakout → heures
  ✨ Donchian (1D)   : channel breakout + ADX → rides macro trends → semaines/mois

Usage :
    # Basique, 6 ans
    PYTHONPATH=. python backtest/run_backtest_donchian.py --months 72 --no-show

    # Analyse de sensibilité
    PYTHONPATH=. python backtest/run_backtest_donchian.py --months 72 --sensitivity --no-show

    # Long only (pas de short)
    PYTHONPATH=. python backtest/run_backtest_donchian.py --months 72 --no-short --no-show
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

from backtest.data_loader import download_candles
from backtest.simulator_donchian import (
    DonchianConfig,
    DonchianEngine,
    DonchianResult,
    DonchianTrade,
    compute_donchian_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("donchian_bt")

# ── Paires — top liquidité, bonnes pour macro trends daily ────────────────────

DONCHIAN_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "SUI-USD", "NEAR-USD", "LTC-USD", "ARB-USD", "OP-USD",
    "AAVE-USD", "UNI-USD", "ATOM-USD", "FIL-USD", "INJ-USD",
]

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Donchian Trend Following (Daily)")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk %% per trade (0.02=2%%)")

    # Donchian params
    parser.add_argument("--entry-period", type=int, default=20, help="Donchian entry channel (days)")
    parser.add_argument("--exit-period", type=int, default=10, help="Donchian exit channel (days)")

    # Filters
    parser.add_argument("--adx-period", type=int, default=14)
    parser.add_argument("--adx-threshold", type=float, default=20.0, help="Min ADX for entry")
    parser.add_argument("--ema-period", type=int, default=200, help="EMA trend filter period")
    parser.add_argument("--use-ema", action="store_true", help="Activer le filtre EMA directionnel")

    # SL/trailing
    parser.add_argument("--sl-atr", type=float, default=2.0, help="SL = N×ATR")
    parser.add_argument("--trail-atr", type=float, default=3.0, help="Trailing ATR mult (si pas donchian exit)")
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--donchian-exit", action="store_true", default=True,
                        help="Exit via Donchian court (défaut)")
    parser.add_argument("--atr-exit", dest="donchian_exit", action="store_false",
                        help="Exit via ATR trailing au lieu de Donchian")

    # Constraints
    parser.add_argument("--max-pos", type=int, default=6, help="Max positions simultanées")
    parser.add_argument("--cooldown", type=int, default=5, help="Cooldown days après clôture")
    parser.add_argument("--no-short", action="store_true", help="Long only")
    parser.add_argument("--no-compound", action="store_true", help="Ne pas compounder")
    parser.add_argument("--btc-regime", action="store_true", help="Filtre régime BTC SMA200 (bloque longs en bear)")
    parser.add_argument("--btc-regime-period", type=int, default=200, help="Période SMA pour régime BTC")
    parser.add_argument("--exclude-pairs", type=str, default="", help="Paires à exclure, séparées par virgule")

    # Modes
    parser.add_argument("--sensitivity", action="store_true", help="Analyse de sensibilité")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher les graphes")
    args = parser.parse_args()

    if args.no_show:
        matplotlib.use("Agg")

    # Dates
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=args.months * 30)

    logger.info("🔵 Donchian Trend Following Backtest — %s → %s (1D)", start.date(), end.date())

    # Téléchargement
    logger.info("📥 Téléchargement klines 1d…")
    candles_by_symbol: dict = {}
    for pair in DONCHIAN_PAIRS:
        try:
            candles = download_candles(pair, start, end, interval="1d")
            if candles:
                candles_by_symbol[pair] = candles
                logger.info("   ✅ %s : %d bougies daily", pair, len(candles))
            else:
                logger.warning("   ⚠️ %s : aucune donnée", pair)
        except ValueError as e:
            logger.warning("   ⚠️ %s : %s", pair, e)

    if not candles_by_symbol:
        logger.error("❌ Aucune donnée — abandon")
        sys.exit(1)

    base_cfg = DonchianConfig(
        initial_balance=args.balance,
        risk_percent=args.risk,
        entry_period=args.entry_period,
        exit_period=args.exit_period,
        adx_period=args.adx_period,
        adx_threshold=args.adx_threshold,
        ema_period=args.ema_period,
        use_ema_filter=args.use_ema,
        atr_period=args.atr_period,
        sl_atr_mult=args.sl_atr,
        use_donchian_exit=args.donchian_exit,
        trail_atr_mult=args.trail_atr,
        max_positions=args.max_pos,
        cooldown_days=args.cooldown,
        allow_short=not args.no_short,
        compound=not args.no_compound,
        btc_regime_filter=args.btc_regime,
        btc_regime_period=args.btc_regime_period,
        excluded_pairs=[p.strip() for p in args.exclude_pairs.split(",") if p.strip()],
    )

    if args.sensitivity:
        _run_sensitivity(candles_by_symbol, base_cfg, args)
    else:
        _run_single(candles_by_symbol, base_cfg, args)


# ── Run unique ─────────────────────────────────────────────────────────────────


def _run_single(candles: dict, cfg: DonchianConfig, args) -> None:
    engine = DonchianEngine(candles, cfg, interval="1d")
    result = engine.run()
    metrics = compute_donchian_metrics(result)
    _print_report(result, metrics)
    _generate_chart(result, metrics, show=not args.no_show)
    _save_trades_csv(result)


# ── Sensibilité ────────────────────────────────────────────────────────────────


def _run_sensitivity(candles: dict, base_cfg: DonchianConfig, args) -> None:
    entry_periods = [10, 20, 30, 55]
    exit_periods = [5, 10, 15, 20]
    adx_thresholds = [15.0, 20.0, 25.0, 30.0]
    sl_atr_mults = [1.5, 2.0, 2.5, 3.0]
    risk_pcts = [0.01, 0.02, 0.03, 0.05]
    exit_modes = [
        (True, 0, "donchian"),
        (False, 2.0, "ATR_2.0"),
        (False, 3.0, "ATR_3.0"),
    ]

    results = []
    total = len(entry_periods) * len(exit_periods) * len(adx_thresholds) * len(sl_atr_mults)
    # Heuristic: skip some combos to keep it fast
    # Full grid = entry × exit × adx × sl × risk × exit_mode = 4×4×4×4×4×3 = 3072
    # Reduced: fix risk=0.02, test exit modes separately → 4×4×4×4 = 256 × 3 exit = 768
    total_runs = len(entry_periods) * len(exit_periods) * len(adx_thresholds) * len(sl_atr_mults) * len(exit_modes)
    logger.info("🔬 Sensibilité : %d combinaisons", total_runs)

    done = 0
    for entry_p in entry_periods:
        for exit_p in exit_periods:
            if exit_p >= entry_p:
                continue  # Exit doit être plus court que entry
            for adx_t in adx_thresholds:
                for sl_m in sl_atr_mults:
                    for use_donchian, trail_m, exit_label in exit_modes:
                        cfg = DonchianConfig(
                            initial_balance=base_cfg.initial_balance,
                            risk_percent=base_cfg.risk_percent,
                            entry_period=entry_p,
                            exit_period=exit_p,
                            adx_period=base_cfg.adx_period,
                            adx_threshold=adx_t,
                            atr_period=base_cfg.atr_period,
                            sl_atr_mult=sl_m,
                            use_donchian_exit=use_donchian,
                            trail_atr_mult=trail_m,
                            max_positions=base_cfg.max_positions,
                            cooldown_days=base_cfg.cooldown_days,
                            allow_short=base_cfg.allow_short,
                            compound=base_cfg.compound,
                            use_ema_filter=base_cfg.use_ema_filter,
                            ema_period=base_cfg.ema_period,
                        )
                        engine = DonchianEngine(candles, cfg, interval="1d")
                        result = engine.run()
                        metrics = compute_donchian_metrics(result)
                        results.append((metrics, cfg, exit_label))
                        done += 1
                        if done % 50 == 0:
                            logger.info("   %d/%d done…", done, total_runs)

    results.sort(key=lambda x: x[0]["profit_factor"] if x[0]["n_trades"] >= 10 else -99, reverse=True)

    sep = "═" * 155
    print(f"\n{sep}")
    print("  🔬 SENSIBILITÉ — Donchian Trend Following (Daily)")
    print(sep)
    print(
        f"  {'Entry':>5s} | {'Exit':>4s} | {'ADX≥':>5s} | {'SL×':>4s} | {'ExitMode':>10s} | "
        f"{'Trades':>6s} | {'Sig':>5s} | {'Filt':>5s} | {'WR':>6s} | {'PF':>6s} | "
        f"{'Return':>8s} | {'CAGR':>7s} | {'MaxDD':>7s} | {'Sharpe':>7s} | "
        f"{'AvgPnL':>8s} | {'AvgHold':>7s}"
    )
    print("  " + "─" * 151)

    shown = 0
    for m, cfg, exit_label in results:
        if m["n_trades"] < 5:
            continue
        print(
            f"  {cfg.entry_period:5d} | {cfg.exit_period:4d} | {cfg.adx_threshold:5.0f} | "
            f"{cfg.sl_atr_mult:4.1f} | {exit_label:>10s} | "
            f"{m['n_trades']:6d} | {m.get('n_signals', 0):5d} | {m.get('n_filtered', 0):5d} | "
            f"{m['win_rate']:6.1%} | {m['profit_factor']:6.2f} | "
            f"{m['total_return']:+7.1%} | {m['cagr']:+6.1%} | {m['max_drawdown']:7.1%} | "
            f"{m['sharpe']:7.2f} | "
            f"${m['avg_pnl_usd']:+7.2f} | {m.get('avg_hold_days', 0):5.1f}d"
        )
        shown += 1
        if shown >= 60:
            break

    print(f"\n{sep}")

    # Best
    viable = [(m, c, l) for m, c, l in results if m["n_trades"] >= 10]
    if viable:
        best_m, best_cfg, best_exit = viable[0]
        print(
            f"\n  🏆 Meilleure config :"
            f"\n     Entry={best_cfg.entry_period}d, Exit={best_cfg.exit_period}d, "
            f"ADX≥{best_cfg.adx_threshold:.0f}, SL={best_cfg.sl_atr_mult:.1f}×ATR, "
            f"ExitMode={best_exit}"
            f"\n     PF={best_m['profit_factor']:.2f}, WR={best_m['win_rate']:.1%}, "
            f"Return={best_m['total_return']:+.1%}, MaxDD={best_m['max_drawdown']:.1%}, "
            f"Sharpe={best_m['sharpe']:.2f}, Trades={best_m['n_trades']}"
            f"\n"
        )

    # Risk sensitivity (keep best params, vary risk)
    if viable:
        _, best_cfg, best_exit_label = viable[0]
        print("  📊 Sensibilité au risque (meilleure config) :")
        for rp in risk_pcts:
            cfg = DonchianConfig(
                initial_balance=best_cfg.initial_balance,
                risk_percent=rp,
                entry_period=best_cfg.entry_period,
                exit_period=best_cfg.exit_period,
                adx_period=best_cfg.adx_period,
                adx_threshold=best_cfg.adx_threshold,
                atr_period=best_cfg.atr_period,
                sl_atr_mult=best_cfg.sl_atr_mult,
                use_donchian_exit=best_cfg.use_donchian_exit,
                trail_atr_mult=best_cfg.trail_atr_mult,
                max_positions=best_cfg.max_positions,
                cooldown_days=best_cfg.cooldown_days,
                allow_short=best_cfg.allow_short,
                compound=best_cfg.compound,
                use_ema_filter=best_cfg.use_ema_filter,
                ema_period=best_cfg.ema_period,
            )
            engine = DonchianEngine(candles, cfg, interval="1d")
            result = engine.run()
            m = compute_donchian_metrics(result)
            marker = " ◀" if rp == best_cfg.risk_percent else ""
            print(
                f"     Risk {rp:5.1%} → Return {m['total_return']:+7.1%}, "
                f"MaxDD {m['max_drawdown']:7.1%}, PF {m['profit_factor']:.2f}, "
                f"Sharpe {m['sharpe']:.2f}{marker}"
            )
        print()


# ── Rapport ────────────────────────────────────────────────────────────────────


def _print_report(result: DonchianResult, metrics: dict) -> None:
    m = metrics
    cfg = result.config
    sep = "═" * 72

    print(f"\n{sep}")
    print("  🔵 DONCHIAN TREND FOLLOWING — DAILY")
    print(sep)
    print(f"  Période      : {result.start_date:%Y-%m-%d} → {result.end_date:%Y-%m-%d} ({m['days']} jours, {m['years']:.1f} ans)")
    print(f"  Paires       : {len(result.pairs)} ({', '.join(result.pairs[:8])}{'…' if len(result.pairs) > 8 else ''})")
    print(f"  Capital      : ${cfg.initial_balance:,.0f} → ${m['final_equity']:,.2f}")
    print(f"  Paramètres   : Entry={cfg.entry_period}d, Exit={cfg.exit_period}d, ADX≥{cfg.adx_threshold:.0f}, SL={cfg.sl_atr_mult:.1f}×ATR")
    print(f"  Risk/Trade   : {cfg.risk_percent:.1%}")
    print()
    print(f"  Total Return : {m['total_return']:+.2%}")
    print(f"  CAGR         : {m['cagr']:+.2%}")
    print(f"  Max Drawdown : {m['max_drawdown']:.2%}")
    print(f"  Sharpe       : {m['sharpe']:.2f}")
    print(f"  Sortino      : {m['sortino']:.2f}")
    print()
    print(f"  Trades       : {m['n_trades']}")
    print(f"  Signaux      : {m['n_signals']} (filtrés: {m['n_filtered']})")
    print(f"  Win Rate     : {m['win_rate']:.1%}")
    print(f"  Profit Factor: {m['profit_factor']:.2f}")
    print(f"  Avg PnL      : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.2%})")
    print(f"  Avg Hold     : {m.get('avg_hold_days', 0):.1f} jours")

    # Best/worst
    if m["best_trade"]:
        bt = m["best_trade"]
        wt = m["worst_trade"]
        print(f"\n  Best Trade   : {bt.symbol} {bt.side} ${bt.pnl_usd:+.2f} ({bt.pnl_pct:+.2%}) — {bt.hold_days}d")
        print(f"  Worst Trade  : {wt.symbol} {wt.side} ${wt.pnl_usd:+.2f} ({wt.pnl_pct:+.2%}) — {wt.hold_days}d")

    # By exit reason
    if m["by_exit"]:
        print(f"\n  {'Exit':>16s} | {'N':>5s} | {'WR':>6s} | {'PF':>6s} | {'PnL':>9s} | {'AvgHold':>7s}")
        print(f"  {'─' * 60}")
        for k, v in sorted(m["by_exit"].items(), key=lambda x: x[1]["n"], reverse=True):
            print(
                f"  {k:>16s} | {v['n']:5d} | {v['wr']:6.1%} | {v['pf']:6.2f} | "
                f"${v['pnl']:+8.2f} | {v.get('avg_hold', 0):5.1f}d"
            )

    # By side
    if m["by_side"]:
        print(f"\n  {'Side':>16s} | {'N':>5s} | {'WR':>6s} | {'PF':>6s} | {'PnL':>9s}")
        print(f"  {'─' * 52}")
        for k, v in m["by_side"].items():
            print(
                f"  {k:>16s} | {v['n']:5d} | {v['wr']:6.1%} | {v['pf']:6.2f} | ${v['pnl']:+8.2f}"
            )

    # By hold duration
    if m.get("by_hold"):
        print(f"\n  {'Hold Bucket':>16s} | {'N':>5s} | {'WR':>6s} | {'PF':>6s} | {'PnL':>9s} | {'AvgHold':>7s}")
        print(f"  {'─' * 64}")
        for k, v in sorted(m["by_hold"].items()):
            print(
                f"  {k:>16s} | {v['n']:5d} | {v['wr']:6.1%} | {v['pf']:6.2f} | "
                f"${v['pnl']:+8.2f} | {v.get('avg_hold', 0):5.1f}d"
            )

    # By pair
    if m["by_pair"]:
        print(f"\n  {'Pair':>10s} | {'N':>5s} | {'WR':>6s} | {'PF':>6s} | {'PnL':>9s}")
        print(f"  {'─' * 48}")
        for k, v in sorted(m["by_pair"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            emoji = "🟢" if v["pnl"] > 0 else "🔴"
            print(
                f"  {emoji} {k:>8s} | {v['n']:5d} | {v['wr']:6.1%} | {v['pf']:6.2f} | ${v['pnl']:+8.2f}"
            )

    print(f"\n{sep}\n")


# ── Charts ─────────────────────────────────────────────────────────────────────


def _generate_chart(result: DonchianResult, metrics: dict, show: bool = True) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle("Donchian Trend Following — Daily", fontsize=14, fontweight="bold")

    # 1. Equity curve
    ax = axes[0]
    if result.equity_curve:
        dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in result.equity_curve]
        equities = [eq for _, eq in result.equity_curve]
        ax.plot(dates, equities, "b-", linewidth=1, label="Equity")
        ax.axhline(result.initial_balance, color="gray", linestyle="--", alpha=0.5, label=f"Initial (${result.initial_balance:,.0f})")
        ax.fill_between(dates, result.initial_balance, equities, alpha=0.1, color="blue")
    ax.set_ylabel("Equity ($)")
    ax.set_title(
        f"Return: {metrics['total_return']:+.1%} | Sharpe: {metrics['sharpe']:.2f} | "
        f"MaxDD: {metrics['max_drawdown']:.1%} | PF: {metrics['profit_factor']:.2f}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[1]
    if result.equity_curve:
        peak = result.initial_balance
        dd_vals = []
        for _, eq in result.equity_curve:
            peak = max(peak, eq)
            dd_vals.append((eq - peak) / peak if peak > 0 else 0)
        ax.fill_between(dates, 0, dd_vals, color="red", alpha=0.4)
        ax.set_ylabel("Drawdown")
    ax.set_title("Drawdown")
    ax.grid(True, alpha=0.3)

    # 3. Trade PnL
    ax = axes[2]
    if result.trades:
        trade_dates = [datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc) for t in result.trades]
        pnls = [t.pnl_usd for t in result.trades]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax.bar(trade_dates, pnls, width=2, color=colors, alpha=0.6)
        ax.set_ylabel("PnL ($)")
    ax.set_title(f"Trades: {metrics['n_trades']} | WR: {metrics['win_rate']:.1%}")
    ax.grid(True, alpha=0.3)

    for ax_ in axes:
        ax_.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    plt.tight_layout()
    outpath = OUTPUT_DIR / "donchian_backtest.png"
    plt.savefig(outpath, dpi=150)
    logger.info("📊 Chart → %s", outpath)
    if show:
        plt.show()
    plt.close()


# ── CSV ────────────────────────────────────────────────────────────────────────


def _save_trades_csv(result: DonchianResult) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / "donchian_trades.csv"
    if not result.trades:
        logger.info("Aucun trade à sauvegarder")
        return
    fields = [
        "symbol", "side", "entry_price", "exit_price", "size",
        "entry_time", "exit_time", "pnl_usd", "pnl_pct",
        "exit_reason", "hold_days", "atr_at_entry",
    ]
    with open(outpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in result.trades:
            w.writerow({
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": f"{t.entry_price:.6f}",
                "exit_price": f"{t.exit_price:.6f}",
                "size": f"{t.size:.8f}",
                "entry_time": datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc).isoformat(),
                "exit_time": datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc).isoformat(),
                "pnl_usd": f"{t.pnl_usd:.4f}",
                "pnl_pct": f"{t.pnl_pct:.6f}",
                "exit_reason": t.exit_reason,
                "hold_days": t.hold_days,
                "atr_at_entry": f"{t.atr_at_entry:.6f}",
            })
    logger.info("💾 Trades CSV → %s (%d trades)", outpath, len(result.trades))


# ── Entry ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
