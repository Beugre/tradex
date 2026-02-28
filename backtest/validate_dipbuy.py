#!/usr/bin/env python
"""
🔬 Validation de robustesse — DIP BUY « TP8 + ATR1.5 + Step0.5% »

3 tests avant mise en production :
  1. Robustesse du seuil de drop (18%, 20%, 22%, 25%)
  2. Concentration des gains (top 5/10 trades = quel % du profit total ?)
  3. Simulation Monte Carlo (1000 tirages, shuffle des trades)

Usage :
  PYTHONPATH=. .venv/bin/python backtest/validate_dipbuy.py --months 72 --no-show
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from backtest.run_backtest_dipbuy import (
    DipBuyConfig,
    DipBuyEngine,
    DipTrade,
    PAIRS_20,
)
from backtest.data_loader import download_all_pairs
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate")

OUTPUT_DIR = Path(__file__).parent / "output"


# ════════════════════════════════════════════════════════════════════════════════
# Config gagnante : TP8 + ATR1.5 + Step0.5%
# ════════════════════════════════════════════════════════════════════════════════

def winning_config(balance: float = 1000.0, drop_threshold: float = 0.20) -> DipBuyConfig:
    """Retourne la config gagnante avec un drop_threshold paramétrable."""
    return DipBuyConfig(
        initial_balance=balance,
        interval="4h",
        lookback_bars=12,            # 48h
        drop_threshold=drop_threshold,
        tp_pct=0.08,
        atr_sl_mult=1.5,
        atr_period=14,
        trail_step_pct=0.005,        # step 0.5%
        trail_trigger_buffer=0.0005,
        risk_percent=0.02,
        max_simultaneous=5,
        cooldown_bars=6,
        fee_pct=0.00075,
        slippage_pct=0.001,
        # Pas de filtres supplémentaires (config de base)
        sma_period=0,
        equity_sma_period=0,
        max_portfolio_heat=1.0,
        btc_trend_filter=False,
        min_drop_recovery=0.0,
        rsi_max=0.0,
        vol_spike_mult=0.0,
        min_wick_ratio=0.0,
    )


# ════════════════════════════════════════════════════════════════════════════════
# TEST 1 — Robustesse du seuil de drop
# ════════════════════════════════════════════════════════════════════════════════

def test_drop_threshold_robustness(
    candles: dict, balance: float,
) -> dict[float, dict]:
    """Teste la config gagnante avec drop = 18%, 20%, 22%, 25%."""
    print("\n" + "═" * 90)
    print("  🔬 TEST 1 — Robustesse du seuil de drop (18% / 20% / 22% / 25%)")
    print("═" * 90)

    thresholds = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    results: dict[float, dict] = {}

    header = (
        f"  {'Drop%':>6s} │ {'Trades':>6s} │ {'WR':>5s} │ {'PF':>6s} │ "
        f"{'Return':>8s} │ {'Sharpe':>6s} │ {'MaxDD':>7s} │ "
        f"{'Final':>10s} │ {'Trail%':>6s} │ {'AvgPnL':>8s}"
    )
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    for thr in thresholds:
        cfg = winning_config(balance=balance, drop_threshold=thr)
        engine = DipBuyEngine(candles, cfg)
        res = engine.run()
        m = compute_metrics(res)

        trailed = sum(1 for t in engine.closed_trades if t.trail_steps > 0)
        trail_pct = trailed / len(engine.closed_trades) * 100 if engine.closed_trades else 0

        results[thr] = {
            "metrics": m,
            "trades": engine.closed_trades,
            "trail_pct": trail_pct,
        }

        row = (
            f"  {thr*100:>5.0f}% │ {m['n_trades']:>6d} │ {m['win_rate']:>4.0%} │ "
            f"{m['profit_factor']:>6.2f} │ {m['total_return']:>+7.1%} │ "
            f"{m['sharpe']:>6.2f} │ {m['max_drawdown']:>6.1%} │ "
            f"${m['final_equity']:>9,.2f} │ {trail_pct:>5.0f}% │ "
            f"${m['avg_pnl_usd']:>+7.2f}"
        )
        print(row)

    # Analyse
    print(f"\n  📊 Analyse de robustesse :")

    returns = {t: r["metrics"]["total_return"] for t, r in results.items()}
    sharpes = {t: r["metrics"]["sharpe"] for t, r in results.items()}
    pfs = {t: r["metrics"]["profit_factor"] for t, r in results.items()}

    # Écart-type relatif (CV) sur les métriques clés
    ret_values = list(returns.values())
    sharpe_values = list(sharpes.values())
    pf_values = list(pfs.values())

    ret_mean = statistics.mean(ret_values)
    ret_std = statistics.stdev(ret_values) if len(ret_values) > 1 else 0
    ret_cv = ret_std / abs(ret_mean) if ret_mean != 0 else float("inf")

    sharpe_mean = statistics.mean(sharpe_values)
    sharpe_std = statistics.stdev(sharpe_values) if len(sharpe_values) > 1 else 0

    pf_mean = statistics.mean(pf_values)
    pf_std = statistics.stdev(pf_values) if len(pf_values) > 1 else 0

    print(f"     Return : moy={ret_mean:+.1%} | σ={ret_std:.1%} | CV={ret_cv:.2f}")
    print(f"     Sharpe : moy={sharpe_mean:.2f} | σ={sharpe_std:.2f}")
    print(f"     PF     : moy={pf_mean:.2f} | σ={pf_std:.2f}")

    # Verdict
    all_profitable = all(r > 0 for r in ret_values)
    all_pf_above_1 = all(p > 1.0 for p in pf_values)
    all_sharpe_above_1 = all(s > 1.0 for s in sharpe_values)

    if all_profitable and all_pf_above_1 and ret_cv < 0.50:
        print(f"\n  ✅ ROBUSTE — Tous les seuils sont rentables, CV < 50%")
    elif all_profitable and ret_cv < 0.80:
        print(f"\n  ⚠️  ACCEPTABLE — Rentable partout mais variance modérée (CV={ret_cv:.2f})")
    else:
        print(f"\n  ❌ FRAGILE — La performance dépend fortement du seuil choisi")

    print()
    return results


# ════════════════════════════════════════════════════════════════════════════════
# TEST 2 — Concentration des gains
# ════════════════════════════════════════════════════════════════════════════════

def test_gain_concentration(trades: list[DipTrade]) -> dict:
    """Analyse si le profit est concentré sur quelques trades."""
    print("\n" + "═" * 90)
    print("  🔬 TEST 2 — Concentration des gains (top trades vs total)")
    print("═" * 90)

    if not trades:
        print("  ❌ Aucun trade à analyser.")
        return {}

    # Total profit (somme de tous les PnL positifs)
    winners = sorted([t for t in trades if t.pnl_usd > 0], key=lambda t: t.pnl_usd, reverse=True)
    losers = sorted([t for t in trades if t.pnl_usd <= 0], key=lambda t: t.pnl_usd)

    total_gross_profit = sum(t.pnl_usd for t in winners) if winners else 0
    total_gross_loss = sum(t.pnl_usd for t in losers) if losers else 0
    total_net = sum(t.pnl_usd for t in trades)

    print(f"\n  📊 Vue d'ensemble :")
    print(f"     Total trades  : {len(trades)}")
    print(f"     Winners       : {len(winners)} ({len(winners)/len(trades)*100:.0f}%)")
    print(f"     Losers        : {len(losers)} ({len(losers)/len(trades)*100:.0f}%)")
    print(f"     Gross profit  : ${total_gross_profit:+,.2f}")
    print(f"     Gross loss    : ${total_gross_loss:+,.2f}")
    print(f"     Net profit    : ${total_net:+,.2f}")

    # Concentration des top N trades
    print(f"\n  🏆 Concentration des gains (% du gross profit) :")

    for n in [1, 3, 5, 10, 20]:
        if n > len(winners):
            continue
        top_n_pnl = sum(t.pnl_usd for t in winners[:n])
        pct = top_n_pnl / total_gross_profit * 100 if total_gross_profit > 0 else 0
        print(f"     Top {n:>2d} trades : ${top_n_pnl:>+8,.2f} = {pct:>5.1f}% du profit brut")

    # Détail des top 10
    print(f"\n  🏅 Top 10 trades (détail) :")
    print(f"     {'#':>3s} │ {'Paire':>10s} │ {'PnL USD':>10s} │ {'PnL %':>7s} │ {'Steps':>5s} │ {'Drop':>5s}")
    print(f"     " + "─" * 52)

    for i, t in enumerate(winners[:10], 1):
        print(
            f"     {i:>3d} │ {t.symbol:>10s} │ ${t.pnl_usd:>+9,.2f} │ "
            f"{t.pnl_pct:>+6.1%} │ {t.trail_steps:>5d} │ {abs(t.drop_pct)*100:>4.0f}%"
        )

    # Distribution des PnL
    pnls = sorted([t.pnl_usd for t in trades])
    percentiles = [10, 25, 50, 75, 90, 95, 99]

    print(f"\n  📊 Distribution des PnL :")
    for p in percentiles:
        idx = int(len(pnls) * p / 100)
        idx = min(idx, len(pnls) - 1)
        print(f"     P{p:>2d} = ${pnls[idx]:>+8.2f}")

    # Test de robustesse : retirer top N et voir si toujours rentable
    print(f"\n  🧪 Robustesse sans les top trades :")
    for remove_n in [1, 3, 5, 10]:
        if remove_n >= len(winners):
            continue
        remaining = trades.copy()
        for t in winners[:remove_n]:
            remaining.remove(t)
        remaining_pnl = sum(t.pnl_usd for t in remaining)
        print(
            f"     Sans top {remove_n:>2d} : PnL net = ${remaining_pnl:>+8,.2f} "
            f"({'✅ rentable' if remaining_pnl > 0 else '❌ négatif'})"
        )

    # Verdict
    if total_gross_profit > 0:
        top5_pct = sum(t.pnl_usd for t in winners[:5]) / total_gross_profit * 100
        top10_pct = sum(t.pnl_usd for t in winners[:min(10, len(winners))]) / total_gross_profit * 100

        if top5_pct > 60:
            print(f"\n  ⚠️  CONCENTRÉ — Top 5 = {top5_pct:.0f}% du profit. Dépendance à quelques trades.")
        elif top5_pct > 40:
            print(f"\n  🟡 MODÉRÉ — Top 5 = {top5_pct:.0f}%. Typique d'un crash bot.")
        else:
            print(f"\n  ✅ DISTRIBUÉ — Top 5 = {top5_pct:.0f}%. Profit bien réparti.")
    else:
        print(f"\n  ❌ Pas de profit brut positif.")

    print()

    return {
        "total_trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "gross_profit": total_gross_profit,
        "gross_loss": total_gross_loss,
        "net_profit": total_net,
        "top5_pct": sum(t.pnl_usd for t in winners[:5]) / total_gross_profit * 100 if total_gross_profit > 0 else 0,
        "top10_pct": sum(t.pnl_usd for t in winners[:min(10, len(winners))]) / total_gross_profit * 100 if total_gross_profit > 0 else 0,
    }


# ════════════════════════════════════════════════════════════════════════════════
# TEST 3 — Monte Carlo
# ════════════════════════════════════════════════════════════════════════════════

def test_monte_carlo(
    trades: list[DipTrade],
    initial_balance: float,
    n_simulations: int = 1000,
) -> dict:
    """Monte Carlo : shuffle l'ordre des trades, mesure la distribution des résultats."""
    print("\n" + "═" * 90)
    print(f"  🔬 TEST 3 — Simulation Monte Carlo ({n_simulations:,d} tirages)")
    print("═" * 90)

    if not trades:
        print("  ❌ Aucun trade à simuler.")
        return {}

    pnl_list = [t.pnl_usd for t in trades]
    n_trades = len(pnl_list)

    final_equities = []
    max_drawdowns = []
    max_consecutive_losses = []
    returns = []

    random.seed(42)  # reproductible

    for _ in range(n_simulations):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)

        equity = initial_balance
        peak = equity
        max_dd = 0.0
        consec_loss = 0
        max_consec = 0

        for pnl in shuffled:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd

            if pnl <= 0:
                consec_loss += 1
                if consec_loss > max_consec:
                    max_consec = consec_loss
            else:
                consec_loss = 0

        final_equities.append(equity)
        max_drawdowns.append(max_dd)
        max_consecutive_losses.append(max_consec)
        returns.append((equity - initial_balance) / initial_balance)

    # Statistiques
    final_equities.sort()
    max_drawdowns.sort()
    returns.sort()
    max_consecutive_losses.sort()

    def percentile(data: list[float], p: int) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    print(f"\n  📊 Distribution des résultats ({n_simulations:,d} simulations) :")
    print(f"\n  {'Métrique':<22s} │ {'P5':>10s} │ {'P25':>10s} │ {'P50':>10s} │ {'P75':>10s} │ {'P95':>10s}")
    print(f"  " + "─" * 79)

    # Final equity
    feq = [percentile(final_equities, p) for p in [5, 25, 50, 75, 95]]
    print(
        f"  {'Final Equity ($)':<22s} │ "
        + " │ ".join(f"${v:>9,.2f}" for v in feq)
    )

    # Return
    ret = [percentile(returns, p) for p in [5, 25, 50, 75, 95]]
    print(
        f"  {'Return (%)':<22s} │ "
        + " │ ".join(f"{v:>+9.1%}" for v in ret)
    )

    # Max drawdown
    mdd = [percentile(max_drawdowns, p) for p in [5, 25, 50, 75, 95]]
    print(
        f"  {'Max Drawdown (%)':<22s} │ "
        + " │ ".join(f"{v:>9.1%}" for v in mdd)
    )

    # Max consecutive losses
    mcl = [percentile(max_consecutive_losses, p) for p in [5, 25, 50, 75, 95]]
    print(
        f"  {'Max Losing Streak':<22s} │ "
        + " │ ".join(f"{v:>9.0f}" for v in mcl)
    )

    # Stats complémentaires
    mean_ret = statistics.mean(returns)
    std_ret = statistics.stdev(returns)
    mean_dd = statistics.mean(max_drawdowns)
    std_dd = statistics.stdev(max_drawdowns)
    pct_profitable = sum(1 for r in returns if r > 0) / len(returns) * 100
    pct_dd_below_20 = sum(1 for d in max_drawdowns if d > -0.20) / len(max_drawdowns) * 100

    print(f"\n  📈 Statistiques clés :")
    print(f"     Return moyen       : {mean_ret:+.1%} (σ = {std_ret:.1%})")
    print(f"     Max DD moyen       : {mean_dd:.1%} (σ = {std_dd:.1%})")
    print(f"     % simulations rentables : {pct_profitable:.1f}%")
    print(f"     % DD < 20%         : {pct_dd_below_20:.1f}%")
    print(f"     Worst case (P1)    : Return = {percentile(returns, 1):+.1%}, DD = {percentile(max_drawdowns, 99):.1%}")
    print(f"     Best case (P99)    : Return = {percentile(returns, 99):+.1%}")

    # Verdict
    if pct_profitable >= 95 and mean_dd > -0.25:
        print(f"\n  ✅ ROBUSTE — {pct_profitable:.0f}% de simulations rentables, DD moyen {mean_dd:.1%}")
    elif pct_profitable >= 80:
        print(f"\n  🟡 ACCEPTABLE — {pct_profitable:.0f}% rentables mais variance notable")
    else:
        print(f"\n  ❌ FRAGILE — Seulement {pct_profitable:.0f}% de simulations rentables")

    print()

    return {
        "n_simulations": n_simulations,
        "mean_return": mean_ret,
        "std_return": std_ret,
        "mean_dd": mean_dd,
        "pct_profitable": pct_profitable,
        "pct_dd_below_20": pct_dd_below_20,
        "p5_return": percentile(returns, 5),
        "p50_return": percentile(returns, 50),
        "p95_return": percentile(returns, 95),
        "p50_dd": percentile(max_drawdowns, 50),
        "p95_dd": percentile(max_drawdowns, 95),
    }


# ════════════════════════════════════════════════════════════════════════════════
# Monte Carlo chart
# ════════════════════════════════════════════════════════════════════════════════

def generate_monte_carlo_chart(
    trades: list[DipTrade],
    initial_balance: float,
    n_curves: int = 200,
    show: bool = True,
) -> Path:
    """Dessine N equity curves shufflées pour visualiser la dispersion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pnl_list = [t.pnl_usd for t in trades]
    n_trades = len(pnl_list)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    plt.style.use("seaborn-v0_8-whitegrid")

    random.seed(42)

    all_final = []
    all_dd_curves = []

    for i in range(n_curves):
        shuffled = pnl_list.copy()
        random.shuffle(shuffled)

        equity_curve = [initial_balance]
        peak = initial_balance
        dd_curve = [0.0]

        for pnl in shuffled:
            eq = equity_curve[-1] + pnl
            equity_curve.append(eq)
            if eq > peak:
                peak = eq
            dd_curve.append((eq - peak) / peak if peak > 0 else 0)

        all_final.append(equity_curve[-1])

        alpha = 0.05
        color = "#2196F3"
        axes[0].plot(range(len(equity_curve)), equity_curve, color=color, alpha=alpha, linewidth=0.5)
        axes[1].plot(range(len(dd_curve)), dd_curve, color="#F44336", alpha=alpha, linewidth=0.5)

    # Equity originale (ordre réel)
    eq_real = [initial_balance]
    peak_real = initial_balance
    dd_real = [0.0]
    for pnl in pnl_list:
        eq_real.append(eq_real[-1] + pnl)
        if eq_real[-1] > peak_real:
            peak_real = eq_real[-1]
        dd_real.append((eq_real[-1] - peak_real) / peak_real if peak_real > 0 else 0)

    axes[0].plot(range(len(eq_real)), eq_real, color="#FF5722", linewidth=2, label="Ordre réel", zorder=10)
    axes[1].plot(range(len(dd_real)), dd_real, color="#FF5722", linewidth=2, zorder=10)

    # Formatting
    mean_final = statistics.mean(all_final)
    std_final = statistics.stdev(all_final) if len(all_final) > 1 else 0

    axes[0].axhline(y=initial_balance, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title(
        f"Monte Carlo — {n_curves} simulations | {n_trades} trades\n"
        f"Equity finale : moy=${mean_final:,.0f} (σ=${std_final:,.0f})",
        fontsize=12, fontweight="bold",
    )
    axes[0].set_ylabel("Equity ($)")
    axes[0].set_xlabel("Trade #")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    axes[0].legend()

    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Trade #")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "dipbuy_monte_carlo.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Monte Carlo chart: {chart_path}")

    if show:
        try:
            import subprocess
            subprocess.run(["open", str(chart_path)], check=False)
        except Exception:
            pass

    return chart_path


# ════════════════════════════════════════════════════════════════════════════════
# Conclusion finale
# ════════════════════════════════════════════════════════════════════════════════

def print_final_verdict(
    threshold_results: dict,
    concentration: dict,
    monte_carlo: dict,
) -> None:
    """Verdict final : GO / NO-GO pour la production."""
    print("\n" + "🟰" * 45)
    print("  📋 VERDICT FINAL — DIP BUY « TP8 + ATR1.5 + Step0.5% »")
    print("🟰" * 45)

    score = 0
    max_score = 3

    # 1. Robustesse seuil
    if threshold_results:
        rets = [r["metrics"]["total_return"] for r in threshold_results.values()]
        pfs = [r["metrics"]["profit_factor"] for r in threshold_results.values()]
        all_profitable = all(r > 0 for r in rets)
        all_pf_above_1 = all(p > 1.0 for p in pfs)
        cv = statistics.stdev(rets) / abs(statistics.mean(rets)) if statistics.mean(rets) != 0 else 999

        if all_profitable and all_pf_above_1 and cv < 0.50:
            print(f"  ✅ Seuil drop : ROBUSTE (CV={cv:.2f}, tous PF>1)")
            score += 1
        elif all_profitable:
            print(f"  🟡 Seuil drop : ACCEPTABLE (CV={cv:.2f})")
            score += 0.5
        else:
            print(f"  ❌ Seuil drop : FRAGILE")

    # 2. Concentration
    if concentration:
        top5 = concentration.get("top5_pct", 100)
        if top5 <= 40:
            print(f"  ✅ Concentration : DISTRIBUÉE (top 5 = {top5:.0f}%)")
            score += 1
        elif top5 <= 60:
            print(f"  🟡 Concentration : MODÉRÉE (top 5 = {top5:.0f}%, normal pour crash bot)")
            score += 0.5
        else:
            print(f"  ❌ Concentration : TROP CONCENTRÉE (top 5 = {top5:.0f}%)")

    # 3. Monte Carlo
    if monte_carlo:
        pct = monte_carlo.get("pct_profitable", 0)
        median_ret = monte_carlo.get("p50_return", 0)
        if pct >= 95 and median_ret > 0.5:
            print(f"  ✅ Monte Carlo : ROBUSTE ({pct:.0f}% rentables, median={median_ret:+.0%})")
            score += 1
        elif pct >= 80:
            print(f"  🟡 Monte Carlo : ACCEPTABLE ({pct:.0f}% rentables)")
            score += 0.5
        else:
            print(f"  ❌ Monte Carlo : FRAGILE ({pct:.0f}% rentables)")

    print(f"\n  📊 Score : {score:.1f} / {max_score}")

    if score >= 2.5:
        print(f"\n  🟢 GO PRODUCTION — Stratégie validée")
        print(f"     → Allocation recommandée : 50% Trail / 25% Breakout / 25% Crash Bot")
        print(f"     → Démarrer en dry-run 2 semaines, puis capital limité")
    elif score >= 1.5:
        print(f"\n  🟡 GO PRUDENT — Stratégie acceptable mais vigilance")
        print(f"     → Démarrer avec 10-15% max du capital")
        print(f"     → Monitoring renforcé les 3 premiers mois")
    else:
        print(f"\n  🔴 NO-GO — Stratégie trop fragile pour la production")
        print(f"     → Revoir les paramètres ou ajouter des filtres")

    print()


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Validation robustesse DIP BUY")
    parser.add_argument("--months", type=int, default=72)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--mc-sims", type=int, default=1000, help="Nombre de simulations Monte Carlo")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═' * 90}")
    print(f"  🔬 VALIDATION ROBUSTESSE — DIP BUY « TP8 + ATR1.5 + Step0.5% »")
    print(f"  📅 {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  💰 ${args.balance:,.0f} | {len(pairs)} paires | MC: {args.mc_sims:,d} sims")
    print(f"{'═' * 90}")

    # Download data
    logger.info("📥 Téléchargement données H4…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # ── TEST 1 : Robustesse du seuil ──
    threshold_results = test_drop_threshold_robustness(candles, args.balance)

    # ── Config gagnante (20%) pour tests 2 et 3 ──
    cfg = winning_config(balance=args.balance)
    engine = DipBuyEngine(candles, cfg)
    result = engine.run()
    ref_metrics = compute_metrics(result)
    ref_trades = engine.closed_trades

    # ── TEST 2 : Concentration des gains ──
    concentration = test_gain_concentration(ref_trades)

    # ── TEST 3 : Monte Carlo ──
    monte_carlo = test_monte_carlo(ref_trades, args.balance, n_simulations=args.mc_sims)

    # ── Chart Monte Carlo ──
    generate_monte_carlo_chart(
        ref_trades, args.balance,
        n_curves=200,
        show=not args.no_show,
    )

    # ── VERDICT ──
    print_final_verdict(threshold_results, concentration, monte_carlo)


if __name__ == "__main__":
    main()
