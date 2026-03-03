#!/usr/bin/env python
"""
Backtest Infinity Bot — DCA inversé + vente en paliers

Usage :
    # BTC 5 ans (2020-2026)
    PYTHONPATH=. python backtest/run_backtest_infinity.py --start 2020-03-01 --end 2026-03-01 --no-show

    # Custom pair
    PYTHONPATH=. python backtest/run_backtest_infinity.py --pair ETH-USD --start 2022-01-01 --end 2026-03-01

    # Custom balance
    PYTHONPATH=. python backtest/run_backtest_infinity.py --balance 5000 --start 2020-03-01 --end 2026-03-01
"""

from __future__ import annotations

import argparse
import csv
import logging
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
logger = logging.getLogger("infinity_bt")

OUTPUT_DIR = Path(__file__).parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Infinity Bot")
    parser.add_argument("--pair", type=str, default="BTC-USD")
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--no-show", action="store_true")

    # Config overrides
    parser.add_argument("--stop-loss", type=float, default=0.15, help="Stop loss %% (default: 0.15)")
    parser.add_argument("--override", type=float, default=0.20, help="Override sell %% (default: 0.20)")
    parser.add_argument("--rsi-entry", type=float, default=50.0, help="RSI max for first entry (default: 50)")
    parser.add_argument("--entry-drop", type=float, default=0.05, help="Min drop from trailing high (default: 0.05 = 5%%)")
    parser.add_argument("--trailing-period", type=int, default=72, help="Trailing high period in H4 bars (default: 72 = 12 days)")
    parser.add_argument("--require-volume", action="store_true", help="Require volume > MA for first entry")
    parser.add_argument("--buy-scale", type=float, default=1.0, help="Multiply all buy amounts by this factor (default: 1.0)")
    parser.add_argument("--rsi-sell", type=float, default=0.0, help="RSI min for sells (default: 0 = disabled)")
    parser.add_argument("--no-breakeven", action="store_true", help="Disable breakeven stop")
    parser.add_argument("--be-level", type=int, default=0, help="Activate breakeven after sell level N (default: 0 = TP1)")
    parser.add_argument("--buy-pcts", type=str, default=None, help="Buy pcts per level, comma-sep (e.g. 0.30,0.15,0.10,0.05,0.00)")
    parser.add_argument("--buy-levels", type=str, default=None, help="DCA drop levels (positive values, auto-negated), comma-sep (e.g. 0.025,0.05,0.075,0.10,0.125)")
    parser.add_argument("--sell-levels", type=str, default=None, help="Sell levels %% per level, comma-sep (e.g. 0.015,0.030,0.050,0.070,0.100)")
    parser.add_argument("--max-invested-pct", type=float, default=0.70, help="Max %% of equity invested per cycle (default: 0.70)")

    args = parser.parse_args()

    # Dates
    if args.start and args.end:
        dt_start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dt_end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        dt_end = datetime.now(timezone.utc)
        dt_start = dt_end - timedelta(days=args.months * 30)

    pair = args.pair

    logger.info(
        "🔷 INFINITY BOT — %s — %s → %s — $%.0f",
        pair, dt_start.strftime("%Y-%m-%d"), dt_end.strftime("%Y-%m-%d"), args.balance,
    )

    # Téléchargement H4
    logger.info("📥 Téléchargement klines H4…")
    candles = download_candles(pair, dt_start, dt_end, "4h")
    logger.info("   ✅ %s : %d bougies H4", pair, len(candles))

    if len(candles) < 100:
        logger.error("❌ Pas assez de données (%d bougies)", len(candles))
        sys.exit(1)

    # Config
    scaled_amounts = tuple(a * args.buy_scale for a in (100.0, 200.0, 300.0, 400.0, 0.0))

    buy_pcts = None
    if args.buy_pcts:
        buy_pcts = tuple(float(x) for x in args.buy_pcts.split(","))

    buy_levels = None
    if args.buy_levels:
        buy_levels = tuple(-abs(float(x)) for x in args.buy_levels.split(","))

    sell_levels = None
    if args.sell_levels:
        sell_levels = tuple(float(x) for x in args.sell_levels.split(","))

    cfg_kwargs = dict(
        initial_balance=args.balance,
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

    # Run
    engine = InfinityEngine(candles, cfg, pair)
    result = engine.run()
    metrics = compute_infinity_metrics(result)

    # Save trades
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_trades_csv(result, OUTPUT_DIR / "infinity_trades.csv")

    # Print report
    _print_report(result, metrics)

    # Plot
    if not args.no_show:
        _plot_equity(result, candles)


def _save_trades_csv(result: InfinityResult, path: Path) -> None:
    """Sauvegarde les trades en CSV."""
    if not result.trades:
        return

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cycle", "entry_date", "exit_date", "ref_price", "pmp",
            "exit_price", "exit_reason", "n_buys", "n_sells",
            "invested", "proceeds", "pnl", "pnl_pct", "duration_h4",
        ])
        for t in result.trades:
            writer.writerow([
                t.cycle_num, t.entry_date, t.exit_date,
                f"{t.reference_price:.2f}", f"{t.pmp:.2f}",
                f"{t.exit_price:.2f}", t.exit_reason,
                t.n_buys, t.n_sells,
                f"{t.total_invested:.2f}", f"{t.total_proceeds:.2f}",
                f"{t.pnl:.2f}", f"{t.pnl_pct:.2f}", t.duration_bars,
            ])

    logger.info("💾 Trades sauvegardés : %s (%d cycles)", path, len(result.trades))


def _print_report(result: InfinityResult, metrics: dict) -> None:
    """Affiche le rapport."""
    cfg = result.config

    print()
    print("═" * 72)
    print("  🔷 INFINITY BOT — BACKTEST REPORT")
    print("═" * 72)
    print()
    print(f"  Paire       : {result.pair}")
    print(f"  Période     : {result.start_date.strftime('%Y-%m-%d')} → {result.end_date.strftime('%Y-%m-%d')} ({metrics.get('days', 0):.0f} jours)")
    print(f"  Capital     : ${cfg.initial_balance:,.0f} → ${result.final_equity:,.2f}")
    print(f"  Fees        : maker {cfg.maker_fee*100:.2f}% / taker {cfg.taker_fee*100:.2f}%")
    print()

    print("  ── Performance ──")
    print(f"  Return      : {metrics['total_return']*100:+.2f}%")
    print(f"  Mensuel     : {metrics['monthly_return']*100:+.2f}%")
    print(f"  CAGR        : {metrics['cagr']*100:+.2f}%")
    print(f"  Max DD      : {metrics['max_dd']*100:.2f}%")
    print(f"  Sharpe      : {metrics['sharpe']:.2f}")
    print()

    print("  ── Cycles ──")
    print(f"  Total       : {metrics['n_trades']}")
    days = metrics.get('days', 1)
    print(f"  Cycles/mois : {metrics['n_trades'] / max(metrics['months'], 0.1):.1f}")
    print(f"  Win Rate    : {metrics['win_rate']:.1f}%")
    print(f"  Profit F.   : {metrics['profit_factor']:.2f}")
    print(f"  Avg PnL     : ${metrics['avg_pnl']:+.2f} ({metrics['avg_pnl_pct']:+.2f}%)")
    print(f"  Avg Win     : ${metrics['avg_win']:+.2f}")
    print(f"  Avg Loss    : ${metrics['avg_loss']:+.2f}")
    print(f"  Avg Duration: {metrics['avg_duration_bars']:.0f} bars H4 ({metrics['avg_duration_bars']*4/24:.1f} jours)")
    print()

    print("  ── Exits ──")
    # Count par type
    exit_counts: dict[str, list] = {}
    for t in result.trades:
        if t.exit_reason not in exit_counts:
            exit_counts[t.exit_reason] = []
        exit_counts[t.exit_reason].append(t)

    print(f"  {'Reason':<20s} | {'N':>5s} | {'WR':>6s} | {'PnL':>10s}")
    print(f"  {'─'*20} {'─'*7} {'─'*8} {'─'*12}")
    for reason, trades in sorted(exit_counts.items()):
        wins = sum(1 for t in trades if t.pnl > 0)
        wr = wins / len(trades) * 100 if trades else 0
        pnl = sum(t.pnl for t in trades)
        print(f"  {reason:<20s} | {len(trades):>5d} | {wr:>5.1f}% | ${pnl:>+9.2f}")
    print()

    # Détail des 10 meilleurs/pires trades
    if result.trades:
        sorted_trades = sorted(result.trades, key=lambda t: t.pnl, reverse=True)
        print("  ── Top 5 trades ──")
        for t in sorted_trades[:5]:
            print(
                f"  #{t.cycle_num:>3d} | {t.entry_date}→{t.exit_date} | "
                f"PMP={t.pmp:>8.2f} | Exit={t.exit_price:>8.2f} | "
                f"{t.exit_reason:<15s} | {t.n_buys}B/{t.n_sells}S | "
                f"${t.pnl:>+7.2f} ({t.pnl_pct:>+5.2f}%)"
            )
        print()
        print("  ── Bottom 5 trades ──")
        for t in sorted_trades[-5:]:
            print(
                f"  #{t.cycle_num:>3d} | {t.entry_date}→{t.exit_date} | "
                f"PMP={t.pmp:>8.2f} | Exit={t.exit_price:>8.2f} | "
                f"{t.exit_reason:<15s} | {t.n_buys}B/{t.n_sells}S | "
                f"${t.pnl:>+7.2f} ({t.pnl_pct:>+5.2f}%)"
            )
    print()

    # Buy level distribution
    buy_counts = [0] * 6
    for t in result.trades:
        buy_counts[min(t.n_buys, 5)] += 1
    print("  ── Distribution paliers d'achat ──")
    for i in range(1, 6):
        if buy_counts[i] > 0:
            print(f"  {i} palier{'s' if i > 1 else ' '} : {buy_counts[i]} cycles")
    print()

    print("═" * 72)
    print()


def _plot_equity(result: InfinityResult, candles: list) -> None:
    """Plot equity curve + prix BTC."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib non disponible")
        return

    eq = result.equity_curve
    if not eq:
        return

    # Timestamps alignés
    warmup = len(candles) - len(eq)
    dates = [
        datetime.fromtimestamp(candles[warmup + i].timestamp / 1000, tz=timezone.utc)
        for i in range(len(eq))
    ]
    btc_prices = [candles[warmup + i].close for i in range(len(eq))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Equity
    ax1.plot(dates, eq, color="green", linewidth=1.2, label="Equity ($)")
    ax1.axhline(result.initial_balance, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"Infinity Bot — {result.pair} — ${result.config.initial_balance:,.0f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # BTC price
    ax2.plot(dates, btc_prices, color="orange", linewidth=1.0, label=f"{result.pair} Price")
    ax2.set_ylabel("Price ($)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Marquer les trades
    for t in result.trades:
        entry_dt = datetime.strptime(t.entry_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        exit_dt = datetime.strptime(t.exit_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        color = "lime" if t.pnl > 0 else "red"
        ax2.axvspan(entry_dt, exit_dt, alpha=0.15, color=color)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "infinity_equity.png", dpi=150)
    plt.show()
    logger.info("📊 Graphique sauvegardé : %s", OUTPUT_DIR / "infinity_equity.png")


if __name__ == "__main__":
    main()
