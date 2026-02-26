#!/usr/bin/env python
"""
Backtest multi-configuration : compare les stratégies avec/sans filtre EMA200.

Exécute 6 (ou plus) configurations et affiche un tableau comparatif.

Configurations testées :
  1. DUAL (Trend+Range) — sans filtre
  2. DUAL (Trend+Range) — avec EMA200
  3. TREND only         — sans filtre
  4. TREND only         — avec EMA200
  5. RANGE only         — sans filtre
  6. RANGE only         — avec EMA200

Usage :
    python -m backtest.run_multi_config
    python -m backtest.run_multi_config --balance 5000
    python -m backtest.run_multi_config --start 2022-02-20 --end 2026-02-20
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from src import config
from backtest.data_loader import download_all_pairs, download_all_pairs_d1
from backtest.simulator import BacktestConfig, BacktestEngine, BacktestResult
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_config")


# ── Configuration par défaut (identique à run_backtest) ────────────────────────


def _base_config(balance: float) -> BacktestConfig:
    return BacktestConfig(
        initial_balance=balance,
        risk_percent_trend=config.RISK_PERCENT_TREND,
        risk_percent_range=config.RISK_PERCENT_RANGE,
        entry_buffer_pct=config.ENTRY_BUFFER_PERCENT,
        sl_buffer_pct=config.SL_BUFFER_PERCENT,
        zero_risk_trigger_pct=config.ZERO_RISK_TRIGGER_PERCENT,
        zero_risk_lock_pct=config.ZERO_RISK_LOCK_PERCENT,
        trailing_stop_pct=config.TRAILING_STOP_PERCENT,
        max_position_pct=config.MAX_POSITION_PERCENT,
        max_simultaneous_positions=config.MAX_SIMULTANEOUS_POSITIONS,
        swing_lookback=config.SWING_LOOKBACK,
        range_width_min=config.RANGE_WIDTH_MIN,
        range_entry_buffer_pct=config.RANGE_ENTRY_BUFFER_PERCENT,
        range_sl_buffer_pct=config.RANGE_SL_BUFFER_PERCENT,
        range_cooldown_bars=config.RANGE_COOLDOWN_BARS,
        max_total_risk_pct=config.MAX_TOTAL_RISK_PERCENT,
    )


# ── Définition des configs à tester ───────────────────────────────────────────


@dataclass
class ConfigSpec:
    name: str
    enable_trend: bool
    enable_range: bool
    use_ema_filter: bool
    use_d1_pullback: bool = False
    allow_short: bool = False
    description: str = ""


CONFIGS: list[ConfigSpec] = [
    # ── Baseline (ancien breakout H4) ──
    ConfigSpec("RANGE only",         False, True,  False, False, False, "Mean Reversion seul (baseline)"),
    ConfigSpec("BRK TREND only",     True,  False, False, False, False, "Ancien breakout H4"),
    ConfigSpec("BRK DUAL",           True,  True,  False, False, False, "Breakout + Range"),
    # ── Pullback D1 (long only) ──
    ConfigSpec("PB TREND only",      True,  False, False, True,  False, "Pullback D1 seul"),
    ConfigSpec("PB DUAL",            True,  True,  False, True,  False, "Pullback D1 + Range"),
    ConfigSpec("PB DUAL + EMA",      True,  True,  True,  True,  False, "Pullback D1 + Range + EMA200"),
    ConfigSpec("PB TREND + EMA",     True,  False, True,  True,  False, "Pullback D1 + EMA200"),
    ConfigSpec("RANGE + EMA",        False, True,  True,  False, False, "Range + EMA200"),
    # ── TREND long + short ──
    ConfigSpec("BRK L+S only",       True,  False, False, False, True,  "Breakout long+short"),
    ConfigSpec("BRK L+S + EMA",      True,  False, True,  False, True,  "Breakout L+S + EMA200"),
    ConfigSpec("PB L+S only",        True,  False, False, True,  True,  "Pullback D1 long+short"),
    ConfigSpec("PB L+S + EMA",       True,  False, True,  True,  True,  "Pullback D1 L+S + EMA200"),
]


# ── Runner ─────────────────────────────────────────────────────────────────────


def run_single(
    spec: ConfigSpec,
    candles: dict[str, list],
    d1_candles: dict[str, list],
    balance: float,
) -> tuple[str, dict, BacktestResult]:
    """Lance un backtest pour une config et renvoie (nom, metrics, result)."""
    cfg = _base_config(balance)
    cfg.enable_trend = spec.enable_trend
    cfg.enable_range = spec.enable_range
    cfg.use_ema_filter = spec.use_ema_filter
    cfg.use_d1_pullback = spec.use_d1_pullback
    cfg.allow_short = spec.allow_short

    engine = BacktestEngine(
        candles, cfg,
        d1_candles_by_symbol=d1_candles if (spec.use_ema_filter or spec.use_d1_pullback) else None,
    )
    result = engine.run()
    metrics = compute_metrics(result)
    return spec.name, metrics, result


def print_comparison(results: list[tuple[str, dict, BacktestResult]]) -> None:
    """Affiche le tableau comparatif final."""
    sep = "═" * 120

    print(f"\n{sep}")
    print("  📊 COMPARAISON MULTI-CONFIGURATION — TradeX Backtest")
    print(sep)

    # Header
    print(
        f"  {'Config':<22s} │ {'Return':>8s} │ {'CAGR':>7s} │ {'MaxDD':>7s} │ "
        f"{'Sharpe':>7s} │ {'Sortino':>8s} │ {'WR':>5s} │ {'PF':>5s} │ "
        f"{'Trades':>6s} │ {'Final$':>10s}"
    )
    print("  " + "─" * 116)

    for name, m, res in results:
        print(
            f"  {name:<22s} │ {m['total_return']:>+7.1%} │ {m['cagr']:>+6.1%} │ "
            f"{m['max_drawdown']:>7.1%} │ {m['sharpe']:>7.2f} │ {m['sortino']:>8.2f} │ "
            f"{m['win_rate']:>4.0%} │ {m['profit_factor']:>5.2f} │ "
            f"{m['n_trades']:>6d} │ ${m['final_equity']:>9,.2f}"
        )

    print(f"\n{'─' * 120}")

    # Détail par stratégie pour chaque config
    for name, m, res in results:
        if m.get("by_strategy"):
            strats = m["by_strategy"]
            parts = []
            for sname, s in strats.items():
                parts.append(f"{sname}: {s['n']}t WR{s['wr']:.0%} PF{s['pf']:.2f} ${s['pnl']:+.0f}")
            print(f"  {name:<22s} │ {' │ '.join(parts)}")

    print(f"\n{'─' * 120}")

    # Rendements mensuels moyens
    print("\n  📅 Rendement mensuel moyen :")
    for name, m, res in results:
        monthly = m.get("monthly_returns", [])
        if monthly:
            avg_monthly = sum(r for _, r in monthly) / len(monthly) * 100
            pos_months = sum(1 for _, r in monthly if r > 0)
            neg_months = sum(1 for _, r in monthly if r <= 0)
            print(
                f"  {name:<22s} │ {avg_monthly:+.2f}%/mois "
                f"│ {pos_months} mois ↑, {neg_months} mois ↓ "
                f"│ {pos_months/(pos_months+neg_months)*100:.0f}% mois positifs"
            )

    print(f"\n{sep}\n")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="TradeX Multi-Config Backtest")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial ($)")
    parser.add_argument(
        "--start", type=str, default="2022-02-20",
        help="Date de début (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", type=str, default="2026-02-20",
        help="Date de fin (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    pairs = config.TRADING_PAIRS
    logger.info("🚀 Multi-Config Backtest — %s → %s", start.date(), end.date())
    logger.info("   Paires : %s", ", ".join(pairs))
    logger.info("   Capital : $%.0f", args.balance)

    # ── 1. Téléchargement H4 ──────────────────────────────────────────────
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # ── 2. Téléchargement D1 toutes paires (EMA + structure trend) ───────
    d1_candles = download_all_pairs_d1(pairs, start, end)

    # ── 3. Exécution des configs ─────────────────────────────────────────
    all_results: list[tuple[str, dict, BacktestResult]] = []

    for i, spec in enumerate(CONFIGS, 1):
        logger.info(
            "\n{'═'*60}\n  [%d/%d] %s — %s\n{'═'*60}",
            i, len(CONFIGS), spec.name, spec.description,
        )
        name, metrics, result = run_single(spec, candles, d1_candles, args.balance)
        all_results.append((name, metrics, result))
        logger.info(
            "  ✅ %s : $%.0f → $%.2f (%+.1f%%) | %d trades",
            name, args.balance, metrics["final_equity"],
            metrics["total_return"] * 100, metrics["n_trades"],
        )

    # ── 4. Tableau comparatif ─────────────────────────────────────────────
    print_comparison(all_results)

    # ── 5. Graphiques comparatifs ─────────────────────────────────────────
    _generate_comparison_charts(all_results, start, end, args.balance)


def _generate_comparison_charts(
    results: list[tuple[str, dict, BacktestResult]],
    start: datetime,
    end: datetime,
    balance: float,
) -> None:
    """Génère un graphique avec toutes les equity curves superposées."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from pathlib import Path

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Palette de couleurs distinctes
    colors = ["#1565C0", "#E53935", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1",
              "#D81B60", "#7CB342", "#6D4C41", "#546E7A", "#FFD600", "#00897B"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={"height_ratios": [4, 2, 2]})

    # ── 1. Equity curves superposées ──────────────────────────────────────
    ax1 = axes[0]
    for idx, (name, m, res) in enumerate(results):
        eq = res.equity_curve
        dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
        values = [p.equity for p in eq]
        ax1.plot(dates, values, color=colors[idx % len(colors)], linewidth=1.2,
                 label=f"{name} ({m['total_return']:+.1%})", alpha=0.85)

    ax1.axhline(y=balance, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.set_title(
        f"TradeX Multi-Config — {start:%b %Y} → {end:%b %Y} | ${balance:,.0f}",
        fontsize=14, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── 2. Drawdown curves ────────────────────────────────────────────────
    ax2 = axes[1]
    for idx, (name, m, res) in enumerate(results):
        dd = m["dd_curve"]
        eq = res.equity_curve
        dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq[:len(dd)]]
        ax2.plot(dates, dd, color=colors[idx % len(colors)], linewidth=0.9,
                 label=name, alpha=0.7)

    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.legend(loc="lower left", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    # ── 3. Barchart comparatif des KPIs ───────────────────────────────────
    ax3 = axes[2]
    names = [r[0] for r in results]
    returns = [r[1]["total_return"] * 100 for r in results]
    bar_colors = [colors[i % len(colors)] for i in range(len(results))]
    bars = ax3.bar(range(len(names)), returns, color=bar_colors, alpha=0.8)
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax3.set_ylabel("Return (%)")
    ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis="y")

    # Annotations sur les barres
    for bar, val in zip(bars, returns):
        ax3.annotate(
            f"{val:+.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8, fontweight="bold",
        )

    plt.tight_layout()

    chart_path = output_dir / f"multi_config_{start:%Y%m%d}_{end:%Y%m%d}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info("💹 Graphique comparatif sauvegardé : %s", chart_path)
    print(f"\n  💹 Graphique sauvegardé : {chart_path}")


if __name__ == "__main__":
    main()
