"""
Génération du rapport : résumé console + graphiques matplotlib.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from backtest.simulator import BacktestResult, Trade

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def generate_report(
    result: BacktestResult,
    metrics: dict,
    show: bool = True,
) -> Path:
    """Imprime le rapport console et génère les graphiques."""
    _print_summary(result, metrics)
    chart_path = _generate_charts(result, metrics)
    if show:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
    return chart_path


# ── Résumé console ─────────────────────────────────────────────────────────────


def _print_summary(result: BacktestResult, m: dict) -> None:
    sep = "═" * 60

    print(f"\n{sep}")
    print(f"  📊 TradeX Backtest — {result.start_date:%b %Y} → {result.end_date:%b %Y}")
    print(f"  Paires : {', '.join(result.pairs)}")
    print(f"  Stratégie : Dual (Trend + Range) | Capital initial : ${result.initial_balance:,.0f}")
    print(sep)

    print("\n  📈 Résultats globaux")
    print("  " + "─" * 56)
    print(f"  Capital final      : ${m['final_equity']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio       : {m['sharpe']:.2f}")
    print(f"  Sortino Ratio      : {m['sortino']:.2f}")
    print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate']*m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  Trades             : {m['n_trades']}")
    print(f"  PnL moyen          : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.2%})")

    if m["best_trade"]:
        b: Trade = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.symbol} [{b.strategy.value}]")
    if m["worst_trade"]:
        w: Trade = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.symbol} [{w.strategy.value}]")

    # Par stratégie
    if m["by_strategy"]:
        print("\n  📊 Par stratégie")
        print("  " + "─" * 56)
        for strat, s in m["by_strategy"].items():
            print(
                f"  {strat:6s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f} | Avg {s['avg_pct']:+.2%}"
            )

    # Par paire
    if m["by_pair"]:
        print("\n  📊 Par paire")
        print("  " + "─" * 56)
        for pair, s in m["by_pair"].items():
            print(
                f"  {pair:10s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PnL ${s['pnl']:+.2f}"
            )

    # Par exit reason
    if m["by_exit"]:
        print("\n  📊 Par motif de sortie")
        print("  " + "─" * 56)
        for reason, s in m["by_exit"].items():
            print(
                f"  {reason:12s} : {s['n']:3d} trades | PnL ${s['pnl']:+.2f}"
            )

    print(f"\n{sep}\n")


# ── Graphiques ─────────────────────────────────────────────────────────────────


def _generate_charts(result: BacktestResult, metrics: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")  # backend non-interactif pour éviter les erreurs
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

    # Données
    eq = result.equity_curve
    dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
    equities = [p.equity for p in eq]

    dd = metrics["dd_curve"]
    monthly = metrics["monthly_returns"]
    trades = result.trades

    # ── 1. Equity curve ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(dates, equities, alpha=0.15, color="#2196F3")
    ax1.plot(dates, equities, color="#1565C0", linewidth=1.2, label="Equity")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Marqueurs des trades
    for t in trades:
        entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc) if t.entry_time else None
        exit_dt = datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc) if t.exit_time else None
        if entry_dt:
            color = "#4CAF50" if t.strategy.value == "TREND" else "#FF9800"
            ax1.axvline(x=entry_dt, color=color, alpha=0.08, linewidth=0.5)

    ax1.set_title(
        f"TradeX Backtest — {result.start_date:%b %Y} → {result.end_date:%b %Y}  |  "
        f"${result.initial_balance:,.0f} → ${result.final_equity:,.2f}  "
        f"({metrics['total_return']:+.1%})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Equity ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.tick_params(axis="x", labelbottom=False)

    # ── 2. Drawdown ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dates[:len(dd)], dd, alpha=0.3, color="#F44336")
    ax2.plot(dates[:len(dd)], dd, color="#D32F2F", linewidth=0.8)
    ax2.set_ylabel("Drawdown", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.tick_params(axis="x", labelbottom=False)

    # ── 3. Rendements mensuels ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    if monthly:
        m_labels = [m[0] for m in monthly]
        m_vals = [m[1] for m in monthly]
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in m_vals]
        ax3.bar(range(len(m_vals)), m_vals, color=colors, alpha=0.7, width=0.8)
        # Afficher 1 label sur N pour lisibilité
        step = max(1, len(m_labels) // 12)
        ax3.set_xticks(range(0, len(m_labels), step))
        ax3.set_xticklabels([m_labels[i] for i in range(0, len(m_labels), step)],
                            rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Mensuel", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax3.axhline(y=0, color="gray", linewidth=0.5)

    # ── 4. Distribution des trades (P&L %) ─────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    if trades:
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_hist = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax4.bar(range(len(pnl_pcts)), pnl_pcts, color=colors_hist, alpha=0.7, width=0.9)
        ax4.axhline(y=0, color="gray", linewidth=0.5)
    ax4.set_ylabel("Trade P&L %", fontsize=10)
    ax4.set_xlabel("Trades (chronologique)", fontsize=10)

    # ── Annotations stats ──────────────────────────────────────────────────
    stats_text = (
        f"CAGR: {metrics['cagr']:.1%}  |  "
        f"MaxDD: {metrics['max_drawdown']:.1%}  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  "
        f"WR: {metrics['win_rate']:.0%}  |  "
        f"PF: {metrics['profit_factor']:.2f}  |  "
        f"Trades: {metrics['n_trades']}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    chart_path = OUTPUT_DIR / f"backtest_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("💹 Graphique sauvegardé : %s", chart_path)
    print(f"  💹 Graphique sauvegardé : {chart_path}")

    return chart_path
