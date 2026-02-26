#!/usr/bin/env python
"""
Backtest de la stratégie Antiliq — détection de cascades de liquidation.

Télécharge les klines 1m pour BTC, ETH, SOL, XRP, BNB (données USDT Binance),
simule le trading contrarian sur les dumps, et affiche les résultats.

Usage :
    PYTHONPATH=. python backtest/run_backtest_antiliq.py
    PYTHONPATH=. python backtest/run_backtest_antiliq.py --months 6
    PYTHONPATH=. python backtest/run_backtest_antiliq.py --threshold 0.04
    PYTHONPATH=. python backtest/run_backtest_antiliq.py --grid          # analyse de sensibilité
    PYTHONPATH=. python backtest/run_backtest_antiliq.py --no-show
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import download_candles
from backtest.simulator_antiliq import (
    AntiliqConfig,
    AntiliqEngine,
    AntiliqResult,
    AntiliqTrade,
    compute_antiliq_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("antiliq_bt")

# ── Paires du bot antiliq ──────────────────────────────────────────────────────

ANTILIQ_PAIRS = [
    # Top 20 par volume/liquidité — idéales pour cascades de liquidation
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "SUI-USD", "PEPE-USD", "NEAR-USD", "LTC-USD", "ARB-USD",
    "OP-USD", "FET-USD", "RENDER-USD", "INJ-USD", "AAVE-USD",
]

OUTPUT_DIR = Path(__file__).parent / "output"


# ── Point d'entrée ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Antiliq — Liquidation Cascade Trading")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial ($)")
    parser.add_argument("--months", type=int, default=12, help="Nombre de mois de données")
    parser.add_argument("--start", type=str, default=None, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--threshold", type=float, default=0.03, help="Seuil de dump (ex: 0.03 = 3%%)")
    parser.add_argument("--vol-mult", type=float, default=1.5, help="Multiplicateur volume (0=désactivé)")
    parser.add_argument("--tp-retrace", type=float, default=0.5, help="TP retrace %% (0.5 = 50%%)")
    parser.add_argument("--sl-extension", type=float, default=0.5, help="SL extension %% (0.5 = 50%%)")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout en minutes")
    parser.add_argument("--cooldown", type=int, default=60, help="Cooldown en minutes")
    parser.add_argument("--risk", type=float, default=0.02, help="Risque par trade (0.02 = 2%%)")
    parser.add_argument("--trailing", action="store_true", default=True, help="Activer le trailing SL (défaut: oui)")
    parser.add_argument("--no-trailing", dest="trailing", action="store_false", help="Désactiver le trailing SL")
    parser.add_argument("--trail-activation", type=float, default=0.3, help="Trailing: activation après X%% retrace (0.3=30%%)")
    parser.add_argument("--trail-step", type=float, default=0.5, help="Trailing: step %% (0.5=50%%)")
    parser.add_argument("--grid", action="store_true", help="Analyse de sensibilité multi-seuils")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher le graphique")
    args = parser.parse_args()

    # ── Dates ──────────────────────────────────────────────────────────────
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=args.months * 30)

    logger.info("🔥 Antiliq Backtest — %s → %s", start.date(), end.date())
    logger.info("   Paires : %s", ", ".join(ANTILIQ_PAIRS))

    # ── 1. Téléchargement des klines 1m ───────────────────────────────────
    logger.info("📥 Téléchargement des klines 1m (peut prendre quelques minutes la première fois)…")
    candles_by_symbol = {}
    for pair in ANTILIQ_PAIRS:
        logger.info("   → %s…", pair)
        candles = download_candles(pair, start, end, interval="1m")
        if candles:
            candles_by_symbol[pair] = candles
            logger.info("     ✅ %s : %d bougies 1m", pair, len(candles))
        else:
            logger.warning("     ⚠️ %s : aucune donnée", pair)

    if not candles_by_symbol:
        logger.error("❌ Aucune donnée téléchargée — abandon")
        sys.exit(1)

    # ── 2. Configuration ──────────────────────────────────────────────────
    base_cfg = AntiliqConfig(
        initial_balance=args.balance,
        risk_percent=args.risk,
        move_threshold_pct=args.threshold,
        volume_multiplier=args.vol_mult,
        tp_retrace_pct=args.tp_retrace,
        sl_extension_pct=args.sl_extension,
        timeout_minutes=args.timeout,
        cooldown_minutes=args.cooldown,
        trailing_sl=args.trailing,
        trailing_activation_pct=args.trail_activation,
        trailing_step_pct=args.trail_step,
    )

    if args.grid:
        # ── Mode grid : analyse de sensibilité ────────────────────────────
        _run_sensitivity(candles_by_symbol, base_cfg, args)
    else:
        # ── Mode single : un seul run ─────────────────────────────────────
        _run_single(candles_by_symbol, base_cfg, args)


# ── Run unique ─────────────────────────────────────────────────────────────────


def _run_single(
    candles: dict,
    cfg: AntiliqConfig,
    args: argparse.Namespace,
) -> None:
    """Lance un backtest unique et affiche le rapport complet."""
    engine = AntiliqEngine(candles, cfg)
    result = engine.run()
    metrics = compute_antiliq_metrics(result)

    _print_report(result, metrics)
    _generate_chart(result, metrics, show=not args.no_show)
    _save_trades_csv(result)


# ── Analyse de sensibilité ─────────────────────────────────────────────────────


def _run_sensitivity(
    candles: dict,
    base_cfg: AntiliqConfig,
    args: argparse.Namespace,
) -> None:
    """Teste plusieurs configurations et affiche un tableau comparatif."""
    thresholds = [0.02, 0.025, 0.03, 0.04, 0.05]
    tp_retraces = [0.3, 0.5, 0.7]
    vol_mults = [0.0, 1.5]  # 0 = pas de filtre volume
    trail_configs = [
        (False, 0.0, 0.0),       # Pas de trailing (SL fixe)
        (True, 0.3, 0.5),        # Trailing: activation 30%, step 50%
        (True, 0.2, 0.7),        # Trailing: activation 20%, step 70%
        (True, 0.5, 0.3),        # Trailing: activation 50%, step 30%
    ]

    results: list[tuple[dict, dict, AntiliqConfig]] = []

    total_combos = len(thresholds) * len(tp_retraces) * len(vol_mults) * len(trail_configs)
    logger.info("🔬 Analyse de sensibilité : %d combinaisons", total_combos)

    for thresh in thresholds:
        for tp_r in tp_retraces:
            for vol_m in vol_mults:
                for trail_on, trail_act, trail_step in trail_configs:
                    cfg = AntiliqConfig(
                        initial_balance=base_cfg.initial_balance,
                        risk_percent=base_cfg.risk_percent,
                        move_threshold_pct=thresh,
                        volume_multiplier=vol_m,
                        tp_retrace_pct=tp_r,
                        sl_extension_pct=base_cfg.sl_extension_pct,
                        timeout_minutes=base_cfg.timeout_minutes,
                        cooldown_minutes=base_cfg.cooldown_minutes,
                        trailing_sl=trail_on,
                        trailing_activation_pct=trail_act,
                        trailing_step_pct=trail_step,
                    )
                    engine = AntiliqEngine(candles, cfg)
                    result = engine.run()
                    metrics = compute_antiliq_metrics(result)
                    results.append((metrics, result.__dict__, cfg))

    # Trier par rendement total décroissant
    results.sort(key=lambda x: x[0]["total_return"], reverse=True)

    # ── Tableau comparatif ─────────────────────────────────────────────────
    sep = "═" * 130
    print(f"\n{sep}")
    print("  🔬 ANALYSE DE SENSIBILITÉ — Antiliq Backtest")
    print(sep)
    print(
        f"  {'Seuil':>6s} | {'VolMul':>6s} | {'TP%':>5s} | {'Trail':>6s} | {'TrAct':>5s} | {'TrStp':>5s} | "
        f"{'Trades':>6s} | {'WR':>6s} | {'PF':>6s} | "
        f"{'Return':>8s} | {'MaxDD':>7s} | {'Sharpe':>7s} | "
        f"{'AvgPnL':>8s}"
    )
    print("  " + "─" * 126)

    for m, _, cfg in results[:40]:  # Top 40
        if m["n_trades"] == 0:
            continue
        trail_str = "ON" if cfg.trailing_sl else "OFF"
        print(
            f"  {cfg.move_threshold_pct:6.1%} | {cfg.volume_multiplier:6.1f} | "
            f"{cfg.tp_retrace_pct:5.0%} | "
            f"{trail_str:>6s} | {cfg.trailing_activation_pct:5.0%} | {cfg.trailing_step_pct:5.0%} | "
            f"{m['n_trades']:6d} | {m['win_rate']:6.1%} | {m['profit_factor']:6.2f} | "
            f"{m['total_return']:+7.1%} | {m['max_drawdown']:7.1%} | "
            f"{m['sharpe']:7.2f} | "
            f"${m['avg_pnl_usd']:+7.2f}"
        )

    print(f"\n{sep}\n")

    # Meilleure config
    if results and results[0][0]["n_trades"] > 0:
        best_m, _, best_cfg = results[0]
        trail_info = (f"trailing=ON(act={best_cfg.trailing_activation_pct:.0%},step={best_cfg.trailing_step_pct:.0%})"
                      if best_cfg.trailing_sl else "trailing=OFF")
        print(f"  🏆 Meilleure config : seuil={best_cfg.move_threshold_pct:.1%}, "
              f"vol_mult={best_cfg.volume_multiplier:.1f}, "
              f"tp_retrace={best_cfg.tp_retrace_pct:.0%}, {trail_info}")
        print(f"     Return={best_m['total_return']:+.1%}, WR={best_m['win_rate']:.0%}, "
              f"PF={best_m['profit_factor']:.2f}, MaxDD={best_m['max_drawdown']:.1%}\n")


# ── Rapport console ────────────────────────────────────────────────────────────


def _print_report(result: AntiliqResult, m: dict) -> None:
    """Affiche le rapport détaillé dans la console."""
    cfg = result.config
    sep = "═" * 65

    print(f"\n{sep}")
    print(f"  🔥 Antiliq Backtest — {result.start_date:%b %Y} → {result.end_date:%b %Y}")
    print(f"  Paires : {', '.join(result.pairs)}")
    print(f"  Capital : ${result.initial_balance:,.0f} | Risque : {cfg.risk_percent:.0%}/trade")
    print(sep)

    print("\n  📈 Résultats globaux")
    print("  " + "─" * 61)
    print(f"  Capital final      : ${m['final_equity']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio       : {m['sharpe']:.2f}")
    print(f"  Sortino Ratio      : {m['sortino']:.2f}")

    print(f"\n  🎯 Trades")
    print("  " + "─" * 61)
    print(f"  Signaux détectés   : {m['n_signals']}")
    print(f"  Trades exécutés    : {m['n_trades']}")
    print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate'] * m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  PnL moyen          : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.2%})")
    print(f"  Durée moy. trade   : {m['avg_hold_min']:.0f} min")

    if m["best_trade"]:
        b: AntiliqTrade = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.symbol} [dump {b.move_pct:.1%}]")
    if m["worst_trade"]:
        w: AntiliqTrade = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.symbol} [dump {w.move_pct:.1%}]")

    # Par paire
    if m["by_pair"]:
        print(f"\n  📊 Par paire")
        print("  " + "─" * 61)
        for pair, s in m["by_pair"].items():
            print(
                f"  {pair:10s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f} | Hold {s['avg_hold']:.0f}min"
            )

    # Par sortie
    if m["by_exit"]:
        print(f"\n  📊 Par motif de sortie")
        print("  " + "─" * 61)
        for reason, s in m["by_exit"].items():
            pct_of_trades = s["n"] / m["n_trades"] * 100 if m["n_trades"] > 0 else 0
            print(
                f"  {reason:10s} : {s['n']:3d} trades ({pct_of_trades:4.1f}%)"
                f" | WR {s['wr']:.0%} | PnL ${s['pnl']:+.2f}"
            )

    # Par taille de dump
    if m.get("by_dump_size"):
        print(f"\n  📊 Par taille de dump")
        print("  " + "─" * 61)
        for bucket, s in m["by_dump_size"].items():
            print(
                f"  Dump {bucket:5s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f}"
            )

    # Config
    print(f"\n  ⚙️  Configuration")
    print("  " + "─" * 61)
    print(f"  Seuil dump         : {cfg.move_threshold_pct:.1%} en {cfg.move_window} min")
    print(f"  Volume multiplier  : {cfg.volume_multiplier:.1f}×")
    print(f"  TP retrace         : {cfg.tp_retrace_pct:.0%}")
    print(f"  SL extension       : {cfg.sl_extension_pct:.0%}")
    if cfg.trailing_sl:
        print(f"  Trailing SL        : ON (activation={cfg.trailing_activation_pct:.0%}, step={cfg.trailing_step_pct:.0%})")
    else:
        print(f"  Trailing SL        : OFF (SL fixe)")
    print(f"  Timeout            : {cfg.timeout_minutes} min")
    print(f"  Cooldown           : {cfg.cooldown_minutes} min")
    print(f"  Fee                : {cfg.fee_pct:.2%} | Slippage : {cfg.slippage_pct:.2%}")

    print(f"\n{sep}\n")


# ── Graphique ──────────────────────────────────────────────────────────────────


def _generate_chart(
    result: AntiliqResult, metrics: dict, show: bool = True
) -> Path:
    """Génère le graphique equity curve + distribution des trades."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1], hspace=0.35, wspace=0.3)

    eq = result.equity_curve
    dates = [datetime.fromtimestamp(ts / 1000, tz=timezone.utc) for ts, _ in eq]
    equities = [e for _, e in eq]
    trades = result.trades

    # ── 1. Equity curve ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(dates, equities, alpha=0.15, color="#FF5722")
    ax1.plot(dates, equities, color="#D84315", linewidth=1.2, label="Equity")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.5)

    # Marquer les trades
    for t in trades:
        entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)
        color = "#4CAF50" if t.pnl_usd >= 0 else "#F44336"
        ax1.axvline(x=entry_dt, color=color, alpha=0.15, linewidth=0.5)

    ax1.set_title(
        f"🔥 Antiliq Backtest — {result.start_date:%b %Y} → {result.end_date:%b %Y}  |  "
        f"${result.initial_balance:,.0f} → ${result.final_equity:,.2f} "
        f"({metrics['total_return']:+.1%})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left")

    # ── 2. Drawdown ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    dd = metrics["dd_curve"]
    if dd and dates:
        ax2.fill_between(dates[:len(dd)], dd, alpha=0.3, color="#F44336")
        ax2.plot(dates[:len(dd)], dd, color="#D32F2F", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # ── 3. Distribution P&L des trades ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    if trades:
        pnls = [t.pnl_pct * 100 for t in trades]
        colors = ["#4CAF50" if p >= 0 else "#F44336" for p in pnls]
        ax3.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.9)
        ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.set_ylabel("P&L %")
    ax3.set_title("Trades (chronologique)", fontsize=10)
    ax3.set_xlabel("Trade #")

    # ── 4. Dump magnitude vs P&L ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if trades:
        dump_pcts = [abs(t.move_pct) * 100 for t in trades]
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_scatter = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax4.scatter(dump_pcts, pnl_pcts, c=colors_scatter, alpha=0.6, s=20)
        ax4.axhline(y=0, color="gray", linewidth=0.5)
    ax4.set_xlabel("Taille du dump (%)")
    ax4.set_ylabel("P&L trade (%)")
    ax4.set_title("Dump size vs P&L", fontsize=10)

    # ── 5. Durée des trades ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    if trades:
        holds = [t.hold_minutes for t in trades]
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_hold = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax5.scatter(holds, pnl_pcts, c=colors_hold, alpha=0.6, s=20)
        ax5.axhline(y=0, color="gray", linewidth=0.5)
    ax5.set_xlabel("Durée (min)")
    ax5.set_ylabel("P&L (%)")
    ax5.set_title("Hold time vs P&L", fontsize=10)

    # ── Stats en bas ───────────────────────────────────────────────────────
    stats_text = (
        f"Seuil: {result.config.move_threshold_pct:.1%}  |  "
        f"WR: {metrics['win_rate']:.0%}  |  "
        f"PF: {metrics['profit_factor']:.2f}  |  "
        f"MaxDD: {metrics['max_drawdown']:.1%}  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  "
        f"Trades: {metrics['n_trades']}  |  "
        f"Avg hold: {metrics['avg_hold_min']:.0f}min"
    )
    fig.text(
        0.5, 0.01, stats_text, ha="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#FBE9E7", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    chart_path = OUTPUT_DIR / f"antiliq_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("💹 Graphique sauvegardé : %s", chart_path)
    print(f"  💹 Graphique sauvegardé : {chart_path}")

    if show:
        try:
            plt.show()
        except Exception:
            pass

    return chart_path


# ── Export CSV ─────────────────────────────────────────────────────────────────


def _save_trades_csv(result: AntiliqResult) -> None:
    """Sauvegarde les trades en CSV pour analyse externe."""
    import csv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"antiliq_trades_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "entry_time", "exit_time", "entry_price", "exit_price",
            "size", "pnl_usd", "pnl_pct", "exit_reason", "move_pct",
            "volume_ratio", "hold_minutes",
        ])
        for t in result.trades:
            entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc)
            w.writerow([
                t.symbol, entry_dt.isoformat(), exit_dt.isoformat(),
                f"{t.entry_price:.6f}", f"{t.exit_price:.6f}",
                f"{t.size:.8f}", f"{t.pnl_usd:.2f}", f"{t.pnl_pct:.4f}",
                t.exit_reason, f"{t.move_pct:.4f}",
                f"{t.volume_ratio:.2f}", t.hold_minutes,
            ])

    print(f"  📄 Trades CSV : {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
