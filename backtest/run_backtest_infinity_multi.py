#!/usr/bin/env python
"""
Backtest Infinity Bot — Multi-paires

Capital réparti équitablement entre N paires, chaque paire tourne indépendamment.
Agrégation des equity curves et métriques globales.

Usage :
    PYTHONPATH=. python backtest/run_backtest_infinity_multi.py \
        --pairs BTC-USD,ETH-USD,SOL-USD,BNB-USD,AVAX-USD \
        --start 2020-03-01 --end 2026-03-01 --balance 1000 --no-show
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import download_candles
from backtest.simulator_infinity import (
    InfinityConfig,
    InfinityEngine,
    InfinityResult,
    InfinityTrade,
    compute_infinity_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("infinity_multi")

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Infinity Bot — Multi-paires")
    parser.add_argument("--pairs", type=str, default="BTC-USD,ETH-USD,SOL-USD,BNB-USD,AVAX-USD")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--stop-loss", type=float, default=0.15)
    parser.add_argument("--override", type=float, default=0.20)
    parser.add_argument("--rsi-entry", type=float, default=50.0)
    parser.add_argument("--entry-drop", type=float, default=0.05)
    parser.add_argument("--trailing-period", type=int, default=72)
    parser.add_argument("--require-volume", action="store_true")
    parser.add_argument("--rsi-sell", type=float, default=0.0)
    parser.add_argument("--no-breakeven", action="store_true")
    parser.add_argument("--be-level", type=int, default=0)
    parser.add_argument("--buy-pcts", type=str, default=None)
    parser.add_argument("--buy-levels", type=str, default=None)
    parser.add_argument("--sell-levels", type=str, default=None)
    parser.add_argument("--max-invested-pct", type=float, default=0.70)

    args = parser.parse_args()

    # Dates
    if args.start and args.end:
        dt_start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dt_end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt_end = datetime.now(timezone.utc)
        dt_start = dt_end - timedelta(days=args.months * 30)

    pairs = [p.strip() for p in args.pairs.split(",")]
    n_pairs = len(pairs)
    balance_per_pair = args.balance / n_pairs

    logger.info(
        "🔷 INFINITY BOT MULTI — %d paires — $%.0f total ($%.0f/paire)",
        n_pairs, args.balance, balance_per_pair,
    )

    # Parse optional overrides
    buy_pcts = None
    if args.buy_pcts:
        buy_pcts = tuple(float(x) for x in args.buy_pcts.split(","))

    buy_levels = None
    if args.buy_levels:
        buy_levels = tuple(-abs(float(x)) for x in args.buy_levels.split(","))

    sell_levels = None
    if args.sell_levels:
        sell_levels = tuple(float(x) for x in args.sell_levels.split(","))

    # Run each pair
    pair_results: list[tuple[str, InfinityResult, dict]] = []

    for pair in pairs:
        logger.info("📥 %s — Téléchargement H4…", pair)
        try:
            candles = download_candles(pair, dt_start, dt_end, "4h")
        except ValueError as e:
            logger.error("❌ %s : %s", pair, e)
            continue

        if len(candles) < 100:
            logger.warning("⚠️ %s : pas assez de données (%d bougies), skip", pair, len(candles))
            continue

        logger.info("   ✅ %s : %d bougies H4", pair, len(candles))

        # Amounts proportionnels au balance_per_pair
        scaled_amounts = tuple(a * (balance_per_pair / 1000.0) for a in (100.0, 200.0, 300.0, 400.0, 0.0))

        cfg_kwargs = dict(
            initial_balance=balance_per_pair,
            stop_loss_pct=args.stop_loss,
            override_sell_pct=args.override,
            first_entry_rsi_max=args.rsi_entry,
            entry_drop_pct=args.entry_drop,
            trailing_high_period=args.trailing_period,
            require_volume_entry=args.require_volume,
            buy_amounts=scaled_amounts,
            max_invested_pct=args.max_invested_pct,
            rsi_sell_min=args.rsi_sell,
            use_breakeven_stop=not args.no_breakeven,
            breakeven_after_level=args.be_level,
        )
        if buy_pcts:
            cfg_kwargs["buy_pcts"] = buy_pcts
        if buy_levels:
            cfg_kwargs["buy_levels"] = buy_levels
        if sell_levels:
            cfg_kwargs["sell_levels"] = sell_levels

        cfg = InfinityConfig(**cfg_kwargs)
        engine = InfinityEngine(candles, cfg, pair)
        result = engine.run()
        metrics = compute_infinity_metrics(result)
        pair_results.append((pair, result, metrics))

        logger.info(
            "   %s : %+.2f%% (%.0f cycles, DD %.2f%%, Sharpe %.2f)",
            pair,
            metrics.get("total_return", 0) * 100,
            metrics.get("n_trades", 0),
            metrics.get("max_dd", 0) * 100,
            metrics.get("sharpe", 0),
        )

    if not pair_results:
        logger.error("❌ Aucune paire valide")
        sys.exit(1)

    # Aggregate
    _print_multi_report(pair_results, args.balance, dt_start, dt_end)

    # Save all trades
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_all_trades_csv(pair_results, OUTPUT_DIR / "infinity_multi_trades.csv")

    # Plot
    if not args.no_show:
        _plot_multi_equity(pair_results, args.balance, dt_start, dt_end)


def _print_multi_report(
    pair_results: list[tuple[str, InfinityResult, dict]],
    total_balance: float,
    dt_start: datetime,
    dt_end: datetime,
) -> None:
    """Rapport consolidé multi-paires."""
    n_pairs = len(pair_results)
    balance_per = total_balance / n_pairs

    print()
    print("═" * 80)
    print("  🔷 INFINITY BOT — MULTI-PAIR BACKTEST REPORT")
    print("═" * 80)
    print()
    print(f"  Période     : {dt_start.strftime('%Y-%m-%d')} → {dt_end.strftime('%Y-%m-%d')}")
    print(f"  Capital     : ${total_balance:,.0f} total (${balance_per:.0f} × {n_pairs} paires)")
    print()

    # Per-pair table
    print(f"  {'Paire':<10s} | {'Return':>8s} | {'Mensuel':>8s} | {'MaxDD':>7s} | {'Sharpe':>6s} | {'PF':>5s} | {'Cycles':>6s} | {'Stops':>5s} | {'Final $':>9s}")
    print(f"  {'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*9}")

    total_final = 0.0
    total_cycles = 0
    total_stops = 0
    total_tp = 0
    total_be = 0
    all_pnls: list[float] = []
    all_pnl_pcts: list[float] = []

    for pair, result, metrics in pair_results:
        ret = metrics.get("total_return", 0) * 100
        monthly = metrics.get("monthly_return", 0) * 100
        dd = metrics.get("max_dd", 0) * 100
        sharpe = metrics.get("sharpe", 0)
        pf = metrics.get("profit_factor", 0)
        n = metrics.get("n_trades", 0)
        stops = result.n_stops
        final = result.final_equity

        total_final += final
        total_cycles += n
        total_stops += stops
        total_tp += result.n_tp_complete
        total_be += sum(1 for t in result.trades if t.exit_reason == "BREAKEVEN")
        all_pnls.extend([t.pnl for t in result.trades])
        all_pnl_pcts.extend([t.pnl_pct for t in result.trades])

        print(
            f"  {pair:<10s} | {ret:>+7.2f}% | {monthly:>+7.2f}% | {dd:>+6.2f}% | "
            f"{sharpe:>6.2f} | {pf:>5.2f} | {n:>6d} | {stops:>5d} | ${final:>8.2f}"
        )

    print(f"  {'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*9}")

    # Global metrics
    total_return = (total_final - total_balance) / total_balance
    days = (dt_end - dt_start).days
    months = days / 30.44
    monthly_return = total_return / months if months > 0 else 0

    # Combined equity curve for max DD
    max_eq_len = max(len(r.equity_curve) for _, r, _ in pair_results)
    combined_eq = []
    for i in range(max_eq_len):
        total = 0.0
        for _, r, _ in pair_results:
            if i < len(r.equity_curve):
                total += r.equity_curve[i]
            else:
                total += r.equity_curve[-1] if r.equity_curve else balance_per
        combined_eq.append(total)

    max_dd = 0.0
    peak = combined_eq[0] if combined_eq else total_balance
    for v in combined_eq:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Global Sharpe
    if len(all_pnl_pcts) > 1:
        mean_r = sum(all_pnl_pcts) / len(all_pnl_pcts)
        var_r = sum((p - mean_r) ** 2 for p in all_pnl_pcts) / (len(all_pnl_pcts) - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1
        cycles_per_year = len(all_pnls) / max(days / 365.25, 0.01)
        sharpe = (mean_r / std_r) * math.sqrt(cycles_per_year) if std_r > 0 else 0
    else:
        sharpe = 0

    # Global PF
    winners = [p for p in all_pnls if p > 0]
    losers = [p for p in all_pnls if p < 0]
    gp = sum(winners) if winners else 0
    gl = abs(sum(losers)) if losers else 0
    pf = gp / gl if gl > 0 else float("inf")
    wr = len(winners) / len(all_pnls) * 100 if all_pnls else 0

    print(
        f"  {'TOTAL':<10s} | {total_return*100:>+7.2f}% | {monthly_return*100:>+7.2f}% | {max_dd*100:>+6.2f}% | "
        f"{sharpe:>6.2f} | {pf:>5.2f} | {total_cycles:>6d} | {total_stops:>5d} | ${total_final:>8.2f}"
    )
    print()

    print(f"  ── Performance globale ──")
    print(f"  Capital     : ${total_balance:,.0f} → ${total_final:,.2f}")
    print(f"  Return      : {total_return*100:+.2f}%")
    print(f"  Mensuel     : {monthly_return*100:+.2f}%")
    print(f"  Max DD      : {max_dd*100:.2f}%")
    print(f"  Sharpe      : {sharpe:.2f}")
    print(f"  Profit F.   : {pf:.2f}")
    print(f"  Win Rate    : {wr:.1f}%")
    print(f"  Cycles      : {total_cycles} ({total_cycles/max(months,0.1):.1f}/mois)")
    print(f"  Exits       : TP {total_tp} | BE {total_be} | SL {total_stops}")
    print(f"  Avg PnL     : ${sum(all_pnls)/len(all_pnls):+.2f} ({sum(all_pnl_pcts)/len(all_pnl_pcts):+.2f}%)" if all_pnls else "")
    print()
    print("═" * 80)
    print()


def _save_all_trades_csv(
    pair_results: list[tuple[str, InfinityResult, dict]], path: Path
) -> None:
    """Sauvegarde tous les trades de toutes les paires en un seul CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pair", "cycle", "entry_date", "exit_date", "ref_price", "pmp",
            "exit_price", "exit_reason", "n_buys", "n_sells",
            "invested", "proceeds", "pnl", "pnl_pct", "duration_h4",
        ])
        for pair, result, _ in pair_results:
            for t in result.trades:
                writer.writerow([
                    pair, t.cycle_num, t.entry_date, t.exit_date,
                    f"{t.reference_price:.2f}", f"{t.pmp:.2f}",
                    f"{t.exit_price:.2f}", t.exit_reason,
                    t.n_buys, t.n_sells,
                    f"{t.total_invested:.2f}", f"{t.total_proceeds:.2f}",
                    f"{t.pnl:.2f}", f"{t.pnl_pct:.2f}", t.duration_bars,
                ])
    logger.info("💾 Trades sauvegardés : %s (%d trades)", path,
                sum(len(r.trades) for _, r, _ in pair_results))


def _plot_multi_equity(
    pair_results: list[tuple[str, InfinityResult, dict]],
    total_balance: float,
    dt_start: datetime,
    dt_end: datetime,
) -> None:
    """Plot equity curves par paire + portfolio combiné."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib non disponible")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    n_pairs = len(pair_results)
    balance_per = total_balance / n_pairs
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63", "#9C27B0", "#00BCD4", "#FF5722"]

    max_eq_len = max(len(r.equity_curve) for _, r, _ in pair_results)

    # Build combined equity
    combined_eq = [0.0] * max_eq_len
    for idx, (pair, result, _) in enumerate(pair_results):
        eq = result.equity_curve
        dates_len = len(eq)
        color = colors[idx % len(colors)]

        # Normaliser en %
        eq_pct = [(v / balance_per - 1) * 100 for v in eq]
        ax1.plot(range(dates_len), eq_pct, color=color, linewidth=1.0, label=pair, alpha=0.8)

        for i in range(max_eq_len):
            if i < dates_len:
                combined_eq[i] += eq[i]
            else:
                combined_eq[i] += eq[-1] if eq else balance_per

    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Return (%)")
    ax1.set_title(f"Infinity Bot Multi-Pair — Per-Pair Returns")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Combined equity
    combined_pct = [(v / total_balance - 1) * 100 for v in combined_eq]
    ax2.plot(range(len(combined_eq)), combined_pct, color="green", linewidth=1.5, label="Portfolio")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(range(len(combined_eq)), 0, combined_pct, alpha=0.15, color="green")
    ax2.set_ylabel("Portfolio Return (%)")
    ax2.set_xlabel("Bar H4")
    ax2.set_title(f"Portfolio combiné — ${total_balance:,.0f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "infinity_multi_equity.png"
    plt.savefig(out, dpi=150)
    plt.show()
    logger.info("📊 Graphique sauvegardé : %s", out)


if __name__ == "__main__":
    main()
