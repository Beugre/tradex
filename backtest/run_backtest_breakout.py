#!/usr/bin/env python3
"""
CLI pour le backtest Breakout Volatility Expansion.

Usage :
  # Run par défaut (12 mois, top 20 paires)
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_breakout.py

  # Personnalisé
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_breakout.py --months 6 --adx 20

  # Grid search
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_breakout.py --grid --months 12

  # Sans graphique (VPS)
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_breakout.py --no-show
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backtest.data_loader import download_all_pairs, download_btc_d1
from backtest.simulator_breakout import (
    BreakoutEngine,
    BreakoutResult,
    BreakoutSimConfig,
    BreakoutTrade,
    EquityPoint,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# ── Top 20 paires les plus liquides pour le breakout ───────────────────────────

BREAKOUT_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
    "LINK-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
    "DOT-USD", "ATOM-USD", "UNI-USD", "NEAR-USD",
    "LTC-USD", "ETC-USD", "FIL-USD", "AAVE-USD",
    "INJ-USD", "SUI-USD", "HBAR-USD", "BNB-USD",
]


# ── Métriques (version autonome pour breakout) ────────────────────────────────


def compute_breakout_metrics(result: BreakoutResult) -> dict:
    """Calcule les KPIs du backtest breakout."""
    import math

    trades = result.trades
    eq = result.equity_curve
    init = result.initial_balance
    final = result.final_equity
    years = max((result.end_date - result.start_date).days / 365.25, 0.01)

    total_return = (final - init) / init
    cagr = (final / init) ** (1 / years) - 1 if final > 0 else 0

    # Drawdown
    peak = init
    max_dd = 0.0
    dd_curve: list[float] = []
    for pt in eq:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak else 0
        dd_curve.append(dd)
        max_dd = min(max_dd, dd)

    # Sharpe
    bars_per_year = 6 * 365.25  # H4
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev = eq[i - 1].equity
        if prev > 0:
            returns.append((eq[i].equity - prev) / prev)

    if len(returns) > 1:
        mu = sum(returns) / len(returns)
        var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var) if var > 0 else 1e-9
        sharpe = (mu / std) * math.sqrt(bars_per_year)
    else:
        sharpe = 0.0

    # Stats trades
    n = len(trades)
    if n:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl = sum(t.pnl_usd for t in trades) / n
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
    else:
        win_rate = profit_factor = avg_pnl = avg_pnl_pct = 0
        best = worst = None

    # Par paire
    from collections import defaultdict
    by_pair: dict[str, dict] = {}
    pair_groups: dict[str, list[BreakoutTrade]] = defaultdict(list)
    for t in trades:
        pair_groups[t.symbol].append(t)
    for pair, tlist in sorted(pair_groups.items()):
        pn = len(tlist)
        pw = sum(1 for t in tlist if t.pnl_usd > 0)
        ppnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        by_pair[pair] = {"n": pn, "wr": pw / pn if pn else 0, "pnl": ppnl, "pf": gp / gl}

    # Par direction
    by_dir: dict[str, dict] = {}
    dir_groups: dict[str, list[BreakoutTrade]] = defaultdict(list)
    for t in trades:
        dir_groups[t.direction.value].append(t)
    for d, tlist in sorted(dir_groups.items()):
        dn = len(tlist)
        dw = sum(1 for t in tlist if t.pnl_usd > 0)
        dpnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        by_dir[d] = {"n": dn, "wr": dw / dn if dn else 0, "pnl": dpnl, "pf": gp / gl}

    # Par exit reason
    by_exit: dict[str, dict] = {}
    exit_groups: dict[str, list[BreakoutTrade]] = defaultdict(list)
    for t in trades:
        exit_groups[t.exit_reason].append(t)
    for reason, tlist in sorted(exit_groups.items()):
        rn = len(tlist)
        rpnl = sum(t.pnl_usd for t in tlist)
        by_exit[reason] = {"n": rn, "pnl": rpnl}

    # Rendements mensuels
    monthly: list[tuple[str, float]] = []
    if eq:
        by_month: dict[str, float] = {}
        for pt in eq:
            dt = datetime.fromtimestamp(pt.timestamp / 1000, tz=timezone.utc)
            key = f"{dt.year}-{dt.month:02d}"
            by_month[key] = pt.equity
        months = sorted(by_month.keys())
        prev_eq = init
        for m in months:
            cur = by_month[m]
            ret = (cur - prev_eq) / prev_eq if prev_eq else 0
            monthly.append((m, ret))
            prev_eq = cur

    # Mois positifs / négatifs
    n_months_pos = sum(1 for _, r in monthly if r >= 0)
    n_months_neg = sum(1 for _, r in monthly if r < 0)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl,
        "avg_pnl_pct": avg_pnl_pct,
        "best_trade": best,
        "worst_trade": worst,
        "by_pair": by_pair,
        "by_direction": by_dir,
        "by_exit": by_exit,
        "monthly_returns": monthly,
        "dd_curve": dd_curve,
        "years": years,
        "final_equity": final,
    }


# ── Rapport console ───────────────────────────────────────────────────────────


def print_report(result: BreakoutResult, m: dict) -> None:
    sep = "═" * 64

    print(f"\n{sep}")
    print(f"  🚀 Breakout Volatility Expansion — {result.start_date:%b %Y} → {result.end_date:%b %Y}")
    print(f"  Paires : {len(result.pairs)} | Capital initial : ${result.initial_balance:,.0f}")
    print(sep)

    print("\n  📈 Résultats globaux")
    print("  " + "─" * 60)
    print(f"  Capital final      : ${m['final_equity']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_drawdown']:.1%}")
    print(f"  Sharpe Ratio       : {m['sharpe']:.2f}")
    print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate']*m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  Trades             : {m['n_trades']}")
    print(f"  PnL moyen          : ${m['avg_pnl_usd']:+.2f} ({m['avg_pnl_pct']:+.2%})")

    if m["best_trade"]:
        b = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.symbol} [{b.direction.value}]")
    if m["worst_trade"]:
        w = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.symbol} [{w.direction.value}]")

    # Par direction
    if m.get("by_direction"):
        print("\n  📊 Par direction")
        print("  " + "─" * 60)
        for d, s in m["by_direction"].items():
            print(
                f"  {d:6s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f}"
            )

    # Par paire (top 10)
    if m.get("by_pair"):
        print("\n  📊 Par paire (top 10 PnL)")
        print("  " + "─" * 60)
        sorted_pairs = sorted(m["by_pair"].items(), key=lambda x: x[1]["pnl"], reverse=True)
        for pair, s in sorted_pairs[:10]:
            print(
                f"  {pair:10s} : {s['n']:3d} trades | WR {s['wr']:.0%}"
                f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f}"
            )

    # Par exit reason
    if m.get("by_exit"):
        print("\n  📊 Par motif de sortie")
        print("  " + "─" * 60)
        for reason, s in m["by_exit"].items():
            print(f"  {reason:12s} : {s['n']:3d} trades | PnL ${s['pnl']:+.2f}")

    # Rendements mensuels
    if m.get("monthly_returns"):
        print("\n  📅 Rendements mensuels")
        print("  " + "─" * 60)
        for month, ret in m["monthly_returns"]:
            bar = "█" * min(int(abs(ret) * 200), 30)
            sign = "🟢" if ret >= 0 else "🔴"
            print(f"  {month} : {sign} {ret:+6.1%}  {bar}")

    print(f"\n{sep}\n")


# ── Graphiques ─────────────────────────────────────────────────────────────────


def generate_charts(result: BreakoutResult, metrics: dict, show: bool = True) -> Path:
    """Génère le graphique PNG du backtest breakout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)

    eq = result.equity_curve
    dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
    equities = [p.equity for p in eq]
    dd = metrics["dd_curve"]
    monthly = metrics["monthly_returns"]
    trades = result.trades

    # 1. Equity
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(dates, equities, alpha=0.15, color="#FF6F00")
    ax1.plot(dates, equities, color="#E65100", linewidth=1.2, label="Equity")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.5)

    for t in trades:
        entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)
        color = "#4CAF50" if t.direction == "LONG" else "#F44336"
        ax1.axvline(x=entry_dt, color=color, alpha=0.08, linewidth=0.5)

    ax1.set_title(
        f"🚀 Breakout Volatility Expansion — {result.start_date:%b %Y} → {result.end_date:%b %Y}  |  "
        f"${result.initial_balance:,.0f} → ${result.final_equity:,.2f}  "
        f"({metrics['total_return']:+.1%})",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left")
    ax1.tick_params(axis="x", labelbottom=False)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dates[:len(dd)], dd, alpha=0.3, color="#F44336")
    ax2.plot(dates[:len(dd)], dd, color="#D32F2F", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax2.tick_params(axis="x", labelbottom=False)

    # 3. Rendements mensuels
    ax3 = fig.add_subplot(gs[2])
    if monthly:
        m_labels = [m[0] for m in monthly]
        m_vals = [m[1] for m in monthly]
        colors = ["#4CAF50" if v >= 0 else "#F44336" for v in m_vals]
        ax3.bar(range(len(m_vals)), m_vals, color=colors, alpha=0.7, width=0.8)
        step = max(1, len(m_labels) // 12)
        ax3.set_xticks(range(0, len(m_labels), step))
        ax3.set_xticklabels([m_labels[i] for i in range(0, len(m_labels), step)],
                            rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Mensuel")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax3.axhline(y=0, color="gray", linewidth=0.5)

    # 4. Distribution PnL
    ax4 = fig.add_subplot(gs[3])
    if trades:
        pnl_pcts = [t.pnl_pct * 100 for t in trades]
        colors_hist = ["#4CAF50" if p >= 0 else "#F44336" for p in pnl_pcts]
        ax4.bar(range(len(pnl_pcts)), pnl_pcts, color=colors_hist, alpha=0.7, width=0.9)
        ax4.axhline(y=0, color="gray", linewidth=0.5)
    ax4.set_ylabel("Trade P&L %")
    ax4.set_xlabel("Trades (chronologique)")

    stats_text = (
        f"CAGR: {metrics['cagr']:.1%}  |  "
        f"MaxDD: {metrics['max_drawdown']:.1%}  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  "
        f"WR: {metrics['win_rate']:.0%}  |  "
        f"PF: {metrics['profit_factor']:.2f}  |  "
        f"Trades: {metrics['n_trades']}"
    )
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    chart_path = OUTPUT_DIR / f"breakout_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("💹 Graphique : %s", chart_path)
    print(f"\n  💹 Graphique sauvegardé : {chart_path}")

    if show:
        try:
            import matplotlib.pyplot as plt2
            plt2.show()
        except Exception:
            pass

    return chart_path


def save_trades_csv(trades: list[BreakoutTrade], result: BreakoutResult) -> Path:
    """Exporte les trades en CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"breakout_trades_{result.start_date:%Y%m%d}_{result.end_date:%Y%m%d}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "direction", "entry_time", "exit_time",
            "entry_price", "exit_price", "size",
            "pnl_usd", "pnl_pct", "exit_reason",
            "adx", "bb_width", "vol_ratio",
        ])
        for t in trades:
            entry_dt = datetime.fromtimestamp(t.entry_time / 1000, tz=timezone.utc)
            exit_dt = datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc)
            w.writerow([
                t.symbol, t.direction.value,
                entry_dt.isoformat(), exit_dt.isoformat(),
                f"{t.entry_price:.6f}", f"{t.exit_price:.6f}", f"{t.size:.8f}",
                f"{t.pnl_usd:.2f}", f"{t.pnl_pct:.4f}", t.exit_reason,
                f"{t.adx_at_entry:.1f}", f"{t.bb_width_at_entry:.4f}",
                f"{t.volume_ratio_at_entry:.2f}",
            ])
    print(f"  📄 Trades CSV : {path}")
    return path


# ── Grid Search ────────────────────────────────────────────────────────────────


def run_grid_search(
    candles_by_symbol: dict,
    args: argparse.Namespace,
    btc_d1: list | None = None,
) -> None:
    """Grid search sur les paramètres clés."""
    param_grid = {
        "adx_threshold": [20.0, 25.0, 30.0],
        "bb_width_expansion": [1.0, 1.2, 1.5],
        "sl_atr_mult": [1.0, 1.5, 2.0],
        "trailing_atr_mult": [1.5, 2.0, 3.0],
        "vol_multiplier": [1.0, 1.2, 1.5],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"\n🔍 Grid Search : {total} combinaisons")
    print("=" * 80)

    results: list[dict] = []
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        use_ema = args.ema_filter or args.all_filters
        use_squeeze = args.atr_squeeze or args.all_filters
        use_adaptive = args.adaptive_trail or args.all_filters
        use_kill = args.kill_switch or args.all_filters

        cfg = BreakoutSimConfig(
            initial_balance=args.balance,
            risk_percent=args.risk,
            adx_threshold=params["adx_threshold"],
            bb_width_expansion=params["bb_width_expansion"],
            sl_atr_mult=params["sl_atr_mult"],
            trailing_atr_mult=params["trailing_atr_mult"],
            vol_multiplier=params["vol_multiplier"],
            allow_short=not args.long_only,
            max_positions=args.max_pos,
            use_ema_filter=use_ema,
            use_atr_squeeze=use_squeeze,
            adaptive_trailing=use_adaptive,
            use_kill_switch=use_kill,
            kill_switch_pct=args.kill_pct,
            dynamic_sizing=args.dynamic_sizing,
            risk_cap_usd=args.risk_cap,
            max_exposure_pct=args.max_exposure,
        )

        engine = BreakoutEngine(candles_by_symbol, cfg, btc_d1_candles=btc_d1)
        res = engine.run()
        m = compute_breakout_metrics(res)

        results.append({
            **params,
            "n_trades": m["n_trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "total_return": m["total_return"],
            "max_dd": m["max_drawdown"],
            "sharpe": m["sharpe"],
            "cagr": m["cagr"],
        })

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (total - i - 1)
        if (i + 1) % 10 == 0 or i == total - 1:
            print(
                f"  [{i+1}/{total}] ADX={params['adx_threshold']:.0f} "
                f"BBexp={params['bb_width_expansion']:.1f} "
                f"SL_ATR={params['sl_atr_mult']:.1f} "
                f"Trail={params['trailing_atr_mult']:.1f} "
                f"Vol={params['vol_multiplier']:.1f} "
                f"→ {m['n_trades']} trades, WR {m['win_rate']:.0%}, "
                f"PF {m['profit_factor']:.2f}, Ret {m['total_return']:+.1%} "
                f"(ETA {eta:.0f}s)"
            )

    # Trier par profit factor (filtrer min 20 trades)
    valid = [r for r in results if r["n_trades"] >= 20]
    if not valid:
        valid = results
    valid.sort(key=lambda x: x["profit_factor"], reverse=True)

    print(f"\n{'=' * 80}")
    print(f"🏆 Top 10 configurations (≥ 20 trades)")
    print(f"{'=' * 80}")
    print(f"{'ADX':>5} {'BBexp':>6} {'SL_ATR':>7} {'Trail':>6} {'Vol':>5} "
          f"{'N':>5} {'WR':>6} {'PF':>6} {'Return':>8} {'MaxDD':>7} {'Sharpe':>7}")
    print("-" * 80)

    for r in valid[:10]:
        print(
            f"{r['adx_threshold']:5.0f} {r['bb_width_expansion']:6.1f} "
            f"{r['sl_atr_mult']:7.1f} {r['trailing_atr_mult']:6.1f} "
            f"{r['vol_multiplier']:5.1f} "
            f"{r['n_trades']:5d} {r['win_rate']:5.0%} "
            f"{r['profit_factor']:6.2f} {r['total_return']:+7.1%} "
            f"{r['max_dd']:6.1%} {r['sharpe']:7.2f}"
        )

    # Sauver grid CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = OUTPUT_DIR / "breakout_grid_search.csv"
    with open(grid_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  📄 Grid CSV : {grid_path}")

    elapsed_total = time.time() - t0
    print(f"  ⏱ Durée totale : {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Breakout Volatility Expansion")
    parser.add_argument("--months", type=int, default=12, help="Durée en mois")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial")
    parser.add_argument("--risk", type=float, default=0.02, help="Risque par trade (0.02 = 2%%)")
    parser.add_argument("--adx", type=float, default=25.0, help="Seuil ADX")
    parser.add_argument("--bb-exp", type=float, default=1.2, help="BB Width expansion")
    parser.add_argument("--sl-atr", type=float, default=1.5, help="SL = ATR * mult")
    parser.add_argument("--trail-atr", type=float, default=2.0, help="Trailing = ATR * mult")
    parser.add_argument("--vol-mult", type=float, default=1.2, help="Volume > avg * mult")
    parser.add_argument("--max-pos", type=int, default=5, help="Max positions simultanées")
    parser.add_argument("--long-only", action="store_true", help="Longs uniquement (pas de short)")
    parser.add_argument("--grid", action="store_true", help="Lancer le grid search")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher les graphiques")
    parser.add_argument("--interval", type=str, default="4h", help="Intervalle (4h, 1h, 1d)")
    # ── 4 Optimisations ──
    parser.add_argument("--ema-filter", action="store_true", help="1️⃣ Filtre EMA200 Daily BTC")
    parser.add_argument("--atr-squeeze", action="store_true", help="2️⃣ Filtre compression ATR")
    parser.add_argument("--adaptive-trail", action="store_true", help="3️⃣ Stop adaptatif par paliers")
    parser.add_argument("--kill-switch", action="store_true", help="4️⃣ Kill-switch mensuel -10%%")
    parser.add_argument("--all-filters", action="store_true", help="Activer les 4 optimisations")
    parser.add_argument("--kill-pct", type=float, default=-0.10, help="Seuil kill-switch (défaut: -0.10)")
    # ── 5/6/7 Money Management ──
    parser.add_argument("--dynamic-sizing", action="store_true", help="5️⃣ Sizing dynamique (ATR-normalisé)")
    parser.add_argument("--risk-cap", type=float, default=0.0, help="6️⃣ Plafond risque par trade en USD (0=illimité)")
    parser.add_argument("--max-exposure", type=float, default=0.0, help="7️⃣ Exposition max en %% du capital (0=illimité, 0.5=50%%)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)

    # Résoudre --all-filters
    use_ema = args.ema_filter or args.all_filters
    use_squeeze = args.atr_squeeze or args.all_filters
    use_adaptive = args.adaptive_trail or args.all_filters
    use_kill = args.kill_switch or args.all_filters

    print(f"\n🚀 Breakout Volatility Expansion Backtest")
    print(f"  Période : {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  Paires  : {len(BREAKOUT_PAIRS)}")
    print(f"  Interval: {args.interval}")
    filters = []
    if use_ema: filters.append("EMA200")
    if use_squeeze: filters.append("ATR-Squeeze")
    if use_adaptive: filters.append("Adaptive-Trail")
    if use_kill: filters.append(f"Kill-Switch({args.kill_pct:.0%})")
    if args.long_only: filters.append("Long-Only")
    if args.dynamic_sizing: filters.append("DynSizing")
    if args.risk_cap > 0: filters.append(f"RiskCap(${args.risk_cap:.0f})")
    if args.max_exposure > 0: filters.append(f"MaxExpo({args.max_exposure:.0%})")
    print(f"  Filtres : {', '.join(filters) if filters else 'Aucun'}")
    print()

    # Télécharger les données
    candles = download_all_pairs(BREAKOUT_PAIRS, start, end, interval=args.interval)

    # Télécharger BTC D1 si filtre EMA activé
    btc_d1 = None
    if use_ema:
        btc_d1 = download_btc_d1(start, end)

    if args.grid:
        run_grid_search(candles, args, btc_d1=btc_d1)
        return

    # Config
    cfg = BreakoutSimConfig(
        initial_balance=args.balance,
        risk_percent=args.risk,
        adx_threshold=args.adx,
        bb_width_expansion=args.bb_exp,
        sl_atr_mult=args.sl_atr,
        trailing_atr_mult=args.trail_atr,
        vol_multiplier=args.vol_mult,
        allow_short=not args.long_only,
        max_positions=args.max_pos,
        # 4 optimisations
        use_ema_filter=use_ema,
        use_atr_squeeze=use_squeeze,
        adaptive_trailing=use_adaptive,
        use_kill_switch=use_kill,
        kill_switch_pct=args.kill_pct,
        # 5/6/7 Money Management
        dynamic_sizing=args.dynamic_sizing,
        risk_cap_usd=args.risk_cap,
        max_exposure_pct=args.max_exposure,
    )

    # Run
    engine = BreakoutEngine(candles, cfg, btc_d1_candles=btc_d1)
    result = engine.run()

    # Métriques + rapport
    metrics = compute_breakout_metrics(result)
    print_report(result, metrics)
    generate_charts(result, metrics, show=not args.no_show)
    save_trades_csv(result.trades, result)


if __name__ == "__main__":
    main()
