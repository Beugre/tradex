#!/usr/bin/env python
"""
Corrélation & Portfolio Combiné — CrashBot × Donchian

Mesure la corrélation des returns daily entre les deux stratégies,
simule un portfolio combiné avec différentes allocations,
et trouve l'allocation optimale (Sharpe max / DD min).

Usage :
    PYTHONPATH=. python backtest/run_correlation_portfolio.py --no-show
    PYTHONPATH=. python backtest/run_correlation_portfolio.py --sweep --no-show
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from backtest.data_loader import download_candles
from backtest.simulator_antiliq import AntiliqConfig, AntiliqEngine, AntiliqResult
from backtest.simulator_donchian import DonchianConfig, DonchianEngine, DonchianResult

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("portfolio")

OUTPUT_DIR = Path(__file__).parent / "output"

# ── Paires ─────────────────────────────────────────────────────────────────────

DONCHIAN_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "AVAX-USD", "DOT-USD", "SUI-USD",
    "NEAR-USD", "LTC-USD", "OP-USD", "AAVE-USD", "FIL-USD", "INJ-USD",
]
# Exclues : UNI, LINK, ADA, ARB, ATOM (PF < 1 sur 5.8 ans)

ANTILIQ_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "SUI-USD", "PEPE-USD", "NEAR-USD", "LTC-USD", "ARB-USD",
    "OP-USD", "FET-USD", "RENDER-USD", "INJ-USD", "AAVE-USD",
]


# ── Utilitaires ────────────────────────────────────────────────────────────────


def _equity_to_daily(
    equity_curve: list[tuple[int, float]],
) -> dict[str, float]:
    """Convertit une equity curve (ts_ms, equity) en {date_str: equity}.
    
    Prend la dernière valeur de chaque jour calendaire.
    """
    daily: dict[str, float] = {}
    for ts_ms, eq in equity_curve:
        d = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        daily[d] = eq  # Dernière valeur du jour gagne
    return daily


def _daily_returns(daily_equity: dict[str, float]) -> dict[str, float]:
    """Calcule les returns journaliers à partir de {date: equity}."""
    dates = sorted(daily_equity.keys())
    returns: dict[str, float] = {}
    for i in range(1, len(dates)):
        prev = daily_equity[dates[i - 1]]
        curr = daily_equity[dates[i]]
        if prev > 0:
            returns[dates[i]] = (curr - prev) / prev
    return returns


def _pearson_corr(x: list[float], y: list[float]) -> float:
    """Calcule le coefficient de corrélation de Pearson."""
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n)
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    return cov / (sx * sy)


def _max_drawdown(equity_values: list[float]) -> float:
    """Calcule le max drawdown (valeur négative)."""
    if not equity_values:
        return 0.0
    peak = equity_values[0]
    max_dd = 0.0
    for v in equity_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _sharpe(returns: list[float]) -> float:
    """Sharpe ratio annualisé (daily returns)."""
    if len(returns) < 10:
        return 0.0
    mean_r = sum(returns) / len(returns)
    var_r = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    std_r = math.sqrt(var_r) if var_r > 0 else 1e-9
    return (mean_r / std_r) * math.sqrt(365)


def _sortino(returns: list[float]) -> float:
    """Sortino ratio annualisé (daily returns)."""
    if len(returns) < 10:
        return 0.0
    mean_r = sum(returns) / len(returns)
    down = [r for r in returns if r < 0]
    if not down:
        return 10.0
    down_var = sum(r ** 2 for r in down) / len(down)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mean_r / down_std) * math.sqrt(365)


def _cagr(initial: float, final: float, days: int) -> float:
    """CAGR annualisé."""
    if days <= 0 or initial <= 0:
        return 0.0
    years = days / 365.25
    return (final / initial) ** (1 / years) - 1


# ── Données ────────────────────────────────────────────────────────────────────


class PortfolioData(NamedTuple):
    donchian_daily_eq: dict[str, float]
    antiliq_daily_eq: dict[str, float]
    donchian_returns: dict[str, float]
    antiliq_returns: dict[str, float]
    common_dates: list[str]
    donchian_result: DonchianResult
    antiliq_result: AntiliqResult


def _load_and_run(args: argparse.Namespace) -> PortfolioData:
    """Télécharge les données, lance les deux backtests sur la même période."""

    # ── Période commune : 12 derniers mois (contrainte données 1m) ─────
    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(days=args.months * 30)

    logger.info("📊 Période commune : %s → %s (%d mois)", start.date(), end.date(), args.months)

    # ── 1. Données Donchian (1d) ────────────────────────────────────────
    logger.info("📥 Téléchargement klines 1d pour Donchian…")
    donchian_candles = {}
    # Warmup pour indicateurs (EMA200 + Donchian30)
    warmup_start = start - timedelta(days=280)
    for pair in DONCHIAN_PAIRS:
        c = download_candles(pair, warmup_start, end, interval="1d")
        if c:
            donchian_candles[pair] = c
            logger.info("   ✅ %s : %d bougies 1d", pair, len(c))

    # ── 2. Données Antiliq (1m) ─────────────────────────────────────────
    logger.info("📥 Téléchargement klines 1m pour Antiliq…")
    antiliq_candles = {}
    for pair in ANTILIQ_PAIRS:
        c = download_candles(pair, start, end, interval="1m")
        if c:
            antiliq_candles[pair] = c
            logger.info("   ✅ %s : %d bougies 1m", pair, len(c))

    if not donchian_candles or not antiliq_candles:
        logger.error("❌ Données insuffisantes — abandon")
        sys.exit(1)

    # ── 3. Backtest Donchian ────────────────────────────────────────────
    logger.info("🔵 Lancement backtest Donchian (1D, long only, BTC regime)…")
    donchian_cfg = DonchianConfig(
        initial_balance=args.balance,
        risk_percent=0.02,
        entry_period=30,
        exit_period=20,
        adx_threshold=30,
        sl_atr_mult=3.0,
        allow_short=False,
        btc_regime_filter=True,
        btc_regime_period=200,
        excluded_pairs=["UNI-USD", "LINK-USD", "ADA-USD", "ARB-USD", "ATOM-USD"],
        max_positions=4,
        compound=True,
    )
    donchian_engine = DonchianEngine(donchian_candles, donchian_cfg)
    donchian_result = donchian_engine.run()
    logger.info(
        "   Donchian : $%.2f → $%.2f (%d trades)",
        donchian_cfg.initial_balance, donchian_result.final_equity, len(donchian_result.trades),
    )

    # ── 4. Backtest Antiliq ─────────────────────────────────────────────
    logger.info("🔥 Lancement backtest Antiliq (1m, CrashBot)…")
    antiliq_cfg = AntiliqConfig(
        initial_balance=args.balance,
        risk_percent=0.02,
        move_threshold_pct=0.04,
        volume_multiplier=0.0,
        tp_retrace_pct=0.3,
        sl_extension_pct=0.5,
        trailing_sl=True,
        trailing_activation_pct=0.5,
        trailing_step_pct=0.3,
        max_positions=2,
    )
    antiliq_engine = AntiliqEngine(antiliq_candles, antiliq_cfg)
    antiliq_result = antiliq_engine.run()
    logger.info(
        "   Antiliq : $%.2f → $%.2f (%d trades)",
        antiliq_cfg.initial_balance, antiliq_result.final_equity, len(antiliq_result.trades),
    )

    # ── 5. Convertir en daily ───────────────────────────────────────────
    donchian_daily = _equity_to_daily(donchian_result.equity_curve)
    antiliq_daily = _equity_to_daily(antiliq_result.equity_curve)

    donchian_ret = _daily_returns(donchian_daily)
    antiliq_ret = _daily_returns(antiliq_daily)

    # Dates communes
    common = sorted(set(donchian_ret.keys()) & set(antiliq_ret.keys()))
    logger.info("📅 %d jours communs pour l'analyse de corrélation", len(common))

    return PortfolioData(
        donchian_daily_eq=donchian_daily,
        antiliq_daily_eq=antiliq_daily,
        donchian_returns=donchian_ret,
        antiliq_returns=antiliq_ret,
        common_dates=common,
        donchian_result=donchian_result,
        antiliq_result=antiliq_result,
    )


# ── Analyse de corrélation ─────────────────────────────────────────────────────


def _analyse_correlation(data: PortfolioData) -> float:
    """Calcule et affiche la corrélation entre les deux stratégies."""
    d_ret = [data.donchian_returns[d] for d in data.common_dates]
    a_ret = [data.antiliq_returns[d] for d in data.common_dates]

    corr = _pearson_corr(d_ret, a_ret)

    # Corrélation par fenêtre glissante (30 jours)
    rolling_corrs = []
    window = 30
    for i in range(window, len(data.common_dates)):
        dx = d_ret[i - window : i]
        ax = a_ret[i - window : i]
        rolling_corrs.append(_pearson_corr(dx, ax))

    # Stats des returns individuels
    d_mean = sum(d_ret) / len(d_ret) if d_ret else 0
    a_mean = sum(a_ret) / len(a_ret) if a_ret else 0
    d_std = math.sqrt(sum((r - d_mean) ** 2 for r in d_ret) / len(d_ret)) if d_ret else 0
    a_std = math.sqrt(sum((r - a_mean) ** 2 for r in a_ret) / len(a_ret)) if a_ret else 0

    # Jours positifs / négatifs
    d_up = sum(1 for r in d_ret if r > 0)
    a_up = sum(1 for r in a_ret if r > 0)
    d_down = sum(1 for r in d_ret if r < 0)
    a_down = sum(1 for r in a_ret if r < 0)
    # Jours où les deux sont négatifs en même temps
    both_down = sum(1 for d in data.common_dates
                    if data.donchian_returns[d] < 0 and data.antiliq_returns[d] < 0)

    print()
    print("═" * 70)
    print("  📐 CORRÉLATION — Donchian × CrashBot")
    print("═" * 70)
    print(f"  Période       : {data.common_dates[0]} → {data.common_dates[-1]}")
    print(f"  Jours communs : {len(data.common_dates)}")
    print()
    print(f"  {'':15s}  {'Donchian':>10s}  {'CrashBot':>10s}")
    print(f"  {'─' * 40}")
    print(f"  {'Return/jour':15s}  {d_mean * 100:>+9.3f}%  {a_mean * 100:>+9.3f}%")
    print(f"  {'Volatilité':15s}  {d_std * 100:>9.3f}%  {a_std * 100:>9.3f}%")
    print(f"  {'Jours ↑':15s}  {d_up:>10d}  {a_up:>10d}")
    print(f"  {'Jours ↓':15s}  {d_down:>10d}  {a_down:>10d}")
    print(f"  {'Sharpe (ann.)':15s}  {_sharpe(d_ret):>10.2f}  {_sharpe(a_ret):>10.2f}")
    print(f"  {'Sortino (ann.)':15s}  {_sortino(d_ret):>10.2f}  {_sortino(a_ret):>10.2f}")
    print()

    # Interprétation visuelle
    if abs(corr) < 0.2:
        emoji, interp = "🟢", "TRÈS FAIBLE — Excellente diversification !"
    elif abs(corr) < 0.4:
        emoji, interp = "🟡", "FAIBLE — Bonne diversification"
    elif abs(corr) < 0.6:
        emoji, interp = "🟠", "MODÉRÉE — Diversification partielle"
    else:
        emoji, interp = "🔴", "ÉLEVÉE — Peu de diversification"

    print(f"  {emoji} Corrélation Pearson :  ρ = {corr:+.4f}")
    print(f"     → {interp}")
    print()

    if rolling_corrs:
        rc_mean = sum(rolling_corrs) / len(rolling_corrs)
        rc_min = min(rolling_corrs)
        rc_max = max(rolling_corrs)
        print(f"  📈 Corrélation glissante (30j) :")
        print(f"     Min = {rc_min:+.3f}  |  Moy = {rc_mean:+.3f}  |  Max = {rc_max:+.3f}")
    print()
    print(f"  🔻 Jours où les DEUX stratégies perdent : {both_down} / {len(data.common_dates)}"
          f" ({both_down / len(data.common_dates) * 100:.1f}%)")
    print()

    return corr


# ── Simulation portfolio combiné ───────────────────────────────────────────────


class PortfolioResult(NamedTuple):
    alloc_donchian: float
    alloc_crash: float
    final_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    sortino: float
    daily_equity: dict[str, float]


def _simulate_portfolio(
    data: PortfolioData,
    alloc_donchian: float,  # 0..1
    total_capital: float = 1000.0,
) -> PortfolioResult:
    """Simule un portfolio combiné avec allocation fixe."""
    alloc_crash = 1.0 - alloc_donchian

    # Dates communes avec equity values
    common_eq_dates = sorted(
        set(data.donchian_daily_eq.keys()) & set(data.antiliq_daily_eq.keys())
    )
    if len(common_eq_dates) < 2:
        return PortfolioResult(alloc_donchian, alloc_crash, total_capital, 0, 0, 0, 0, 0, {})

    # Capital initial de chaque stratégie
    d_initial = list(data.donchian_daily_eq.values())[0]
    a_initial = list(data.antiliq_daily_eq.values())[0]

    # Equity combinée : normaliser chaque strat à sa part du capital
    daily_eq: dict[str, float] = {}
    for date in common_eq_dates:
        d_eq = data.donchian_daily_eq[date]
        a_eq = data.antiliq_daily_eq[date]
        # Return cumulatif de chaque stratégie
        d_return = d_eq / d_initial if d_initial > 0 else 1.0
        a_return = a_eq / a_initial if a_initial > 0 else 1.0
        # Portfolio : pondéré
        portfolio_return = alloc_donchian * d_return + alloc_crash * a_return
        daily_eq[date] = total_capital * portfolio_return

    # Métriques
    eq_values = [daily_eq[d] for d in common_eq_dates]
    first_eq = eq_values[0]
    final_eq = eq_values[-1]
    total_return = (final_eq - first_eq) / first_eq if first_eq > 0 else 0
    n_days = len(common_eq_dates)
    cagr_val = _cagr(first_eq, final_eq, n_days)
    max_dd = _max_drawdown(eq_values)

    # Daily returns du portfolio
    port_returns = []
    for i in range(1, len(eq_values)):
        if eq_values[i - 1] > 0:
            port_returns.append((eq_values[i] - eq_values[i - 1]) / eq_values[i - 1])

    sharpe_val = _sharpe(port_returns)
    sortino_val = _sortino(port_returns)

    return PortfolioResult(
        alloc_donchian=alloc_donchian,
        alloc_crash=alloc_crash,
        final_equity=final_eq,
        total_return=total_return,
        cagr=cagr_val,
        max_drawdown=max_dd,
        sharpe=sharpe_val,
        sortino=sortino_val,
        daily_equity=daily_eq,
    )


# ── Sweep allocations ──────────────────────────────────────────────────────────


def _allocation_sweep(data: PortfolioData, total_capital: float = 1000.0) -> list[PortfolioResult]:
    """Teste toutes les allocations de 0% à 100% par pas de 5%."""
    results: list[PortfolioResult] = []
    for pct_donchian in range(0, 101, 5):
        alloc = pct_donchian / 100.0
        r = _simulate_portfolio(data, alloc, total_capital)
        results.append(r)
    return results


def _print_sweep(results: list[PortfolioResult]) -> None:
    """Affiche le tableau des allocations."""
    print("═" * 90)
    print("  📊 ALLOCATION SWEEP — Donchian × CrashBot")
    print("═" * 90)
    print(f"  {'Donchian':>8s}  {'Crash':>6s}  │  {'Final $':>9s}  {'Return':>8s}"
          f"  {'CAGR':>7s}  {'MaxDD':>7s}  {'Sharpe':>7s}  {'Sortino':>8s}")
    print(f"  {'─' * 82}")

    best_sharpe = max(results, key=lambda r: r.sharpe)
    best_dd = min(results, key=lambda r: abs(r.max_drawdown))

    for r in results:
        marker = ""
        if r == best_sharpe:
            marker = " ← 🏆 Sharpe"
        elif r == best_dd:
            marker = " ← 🛡️ DD min"

        print(
            f"  {r.alloc_donchian * 100:>7.0f}%  {r.alloc_crash * 100:>5.0f}%  │  "
            f"${r.final_equity:>8.2f}  {r.total_return * 100:>+7.1f}%  "
            f"{r.cagr * 100:>+6.1f}%  {r.max_drawdown * 100:>6.1f}%  "
            f"{r.sharpe:>7.2f}  {r.sortino:>8.2f}{marker}"
        )
    print()

    # Résumé
    print(f"  🏆 Allocation Sharpe max : {best_sharpe.alloc_donchian * 100:.0f}% Donchian"
          f" / {best_sharpe.alloc_crash * 100:.0f}% Crash"
          f"  → Sharpe {best_sharpe.sharpe:.2f}, MaxDD {best_sharpe.max_drawdown * 100:.1f}%")
    print(f"  🛡️  Allocation DD min    : {best_dd.alloc_donchian * 100:.0f}% Donchian"
          f" / {best_dd.alloc_crash * 100:.0f}% Crash"
          f"  → Sharpe {best_dd.sharpe:.2f}, MaxDD {best_dd.max_drawdown * 100:.1f}%")
    print()

    return best_sharpe, best_dd


# ── Graphiques ─────────────────────────────────────────────────────────────────


def _generate_charts(
    data: PortfolioData,
    sweep_results: list[PortfolioResult] | None,
    best_portfolio: PortfolioResult,
    correlation: float,
    show: bool = True,
) -> None:
    """Génère 4 graphiques : equity curves, corrélation, rolling, allocation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Portfolio Donchian × CrashBot  |  ρ = {correlation:+.3f}",
        fontsize=14, fontweight="bold",
    )

    # ── 1. Equity curves individuelles + combinée ────────────────────────
    ax1 = axes[0, 0]
    common_eq_dates = sorted(
        set(data.donchian_daily_eq.keys()) & set(data.antiliq_daily_eq.keys())
    )
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in common_eq_dates]
    d_initial = data.donchian_daily_eq[common_eq_dates[0]]
    a_initial = data.antiliq_daily_eq[common_eq_dates[0]]

    d_norm = [data.donchian_daily_eq[d] / d_initial * 1000 for d in common_eq_dates]
    a_norm = [data.antiliq_daily_eq[d] / a_initial * 1000 for d in common_eq_dates]
    p_eq = [best_portfolio.daily_equity.get(d, 1000) for d in common_eq_dates]

    ax1.plot(dates, d_norm, label="Donchian", alpha=0.8, linewidth=1.2)
    ax1.plot(dates, a_norm, label="CrashBot", alpha=0.8, linewidth=1.2)
    ax1.plot(dates, p_eq, label=f"Portfolio ({best_portfolio.alloc_donchian*100:.0f}/{best_portfolio.alloc_crash*100:.0f})",
             color="black", linewidth=2)
    ax1.set_title("Equity Curves (normalisées à $1000)")
    ax1.set_ylabel("Equity ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # ── 2. Scatter plot des returns daily ─────────────────────────────────
    ax2 = axes[0, 1]
    d_ret = [data.donchian_returns.get(d, 0) * 100 for d in data.common_dates]
    a_ret = [data.antiliq_returns.get(d, 0) * 100 for d in data.common_dates]
    ax2.scatter(d_ret, a_ret, alpha=0.4, s=10, c="steelblue")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.axvline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel("Donchian Daily Return (%)")
    ax2.set_ylabel("CrashBot Daily Return (%)")
    ax2.set_title(f"Scatter — ρ = {correlation:+.3f}")
    ax2.grid(True, alpha=0.3)

    # ── 3. Corrélation glissante 30j ──────────────────────────────────────
    ax3 = axes[1, 0]
    window = 30
    d_rets = [data.donchian_returns.get(d, 0) for d in data.common_dates]
    a_rets = [data.antiliq_returns.get(d, 0) for d in data.common_dates]
    roll_dates = []
    roll_corrs = []
    for i in range(window, len(data.common_dates)):
        dx = d_rets[i - window : i]
        ax_r = a_rets[i - window : i]
        roll_dates.append(datetime.strptime(data.common_dates[i], "%Y-%m-%d"))
        roll_corrs.append(_pearson_corr(dx, ax_r))

    ax3.plot(roll_dates, roll_corrs, color="darkorange", linewidth=1)
    ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax3.axhline(0.3, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax3.axhline(-0.3, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax3.fill_between(roll_dates, -0.3, 0.3, alpha=0.05, color="green")
    ax3.set_title("Corrélation glissante (30 jours)")
    ax3.set_ylabel("ρ")
    ax3.set_ylim(-1, 1)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    # ── 4. Frontier allocation ────────────────────────────────────────────
    ax4 = axes[1, 1]
    if sweep_results:
        dd_list = [abs(r.max_drawdown) * 100 for r in sweep_results]
        ret_list = [r.total_return * 100 for r in sweep_results]
        sharpe_list = [r.sharpe for r in sweep_results]
        alloc_labels = [f"{r.alloc_donchian*100:.0f}D" for r in sweep_results]

        sc = ax4.scatter(dd_list, ret_list, c=sharpe_list, cmap="RdYlGn", s=60, zorder=3)
        plt.colorbar(sc, ax=ax4, label="Sharpe")

        # Annoter quelques points
        for i, r in enumerate(sweep_results):
            if r.alloc_donchian * 100 in [0, 25, 50, 75, 100] or r == best_portfolio:
                ax4.annotate(alloc_labels[i], (dd_list[i], ret_list[i]),
                             textcoords="offset points", xytext=(5, 5), fontsize=7)

        ax4.set_xlabel("Max Drawdown (%)")
        ax4.set_ylabel("Total Return (%)")
        ax4.set_title("Frontière efficiente (Donchian % / Crash %)")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "portfolio_correlation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("📊 Chart → %s", out_path)

    if show:
        plt.show()
    plt.close()


# ── Drawdown chart détaillé ────────────────────────────────────────────────────


def _generate_drawdown_chart(
    data: PortfolioData,
    best_portfolio: PortfolioResult,
    show: bool = True,
) -> None:
    """Graphique des drawdowns superposés pour les 3 courbes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    common_eq_dates = sorted(
        set(data.donchian_daily_eq.keys()) & set(data.antiliq_daily_eq.keys())
    )
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in common_eq_dates]

    d_initial = data.donchian_daily_eq[common_eq_dates[0]]
    a_initial = data.antiliq_daily_eq[common_eq_dates[0]]

    def _dd_series(values: list[float]) -> list[float]:
        peak = values[0]
        dd = []
        for v in values:
            if v > peak:
                peak = v
            dd.append((v - peak) / peak * 100 if peak > 0 else 0)
        return dd

    d_eq = [data.donchian_daily_eq[d] / d_initial * 1000 for d in common_eq_dates]
    a_eq = [data.antiliq_daily_eq[d] / a_initial * 1000 for d in common_eq_dates]
    p_eq = [best_portfolio.daily_equity.get(d, 1000) for d in common_eq_dates]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(dates, _dd_series(d_eq), 0, alpha=0.3, label="Donchian DD", color="tab:blue")
    ax.fill_between(dates, _dd_series(a_eq), 0, alpha=0.3, label="CrashBot DD", color="tab:orange")
    ax.plot(dates, _dd_series(p_eq), color="black", linewidth=1.5,
            label=f"Portfolio DD ({best_portfolio.alloc_donchian*100:.0f}/{best_portfolio.alloc_crash*100:.0f})")
    ax.set_title("Drawdown comparé — Donchian vs CrashBot vs Portfolio")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    out_path = OUTPUT_DIR / "portfolio_drawdown.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info("📉 Drawdown chart → %s", out_path)

    if show:
        plt.show()
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrélation & Portfolio — Donchian × CrashBot")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial ($)")
    parser.add_argument("--months", type=int, default=12, help="Mois de données (limité par klines 1m)")
    parser.add_argument("--start", type=str, default=None, help="Date début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Date fin (YYYY-MM-DD)")
    parser.add_argument("--sweep", action="store_true", help="Sweep d'allocations 0-100%%")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher les graphiques")
    args = parser.parse_args()

    # ── 1. Charger et lancer les backtests ─────────────────────────────────
    data = _load_and_run(args)

    if len(data.common_dates) < 10:
        logger.error("❌ Pas assez de jours communs (%d) pour l'analyse", len(data.common_dates))
        sys.exit(1)

    # ── 2. Analyse de corrélation ──────────────────────────────────────────
    corr = _analyse_correlation(data)

    # ── 3. Portfolios : 100%D, 100%C, 50/50, et sweep ─────────────────────
    p_100d = _simulate_portfolio(data, 1.0, args.balance)
    p_100c = _simulate_portfolio(data, 0.0, args.balance)
    p_5050 = _simulate_portfolio(data, 0.5, args.balance)

    print("═" * 70)
    print("  🏦 PORTFOLIOS CLÉ")
    print("═" * 70)
    print(f"  {'Config':20s}  {'Final $':>9s}  {'Return':>8s}  {'MaxDD':>7s}  {'Sharpe':>7s}")
    print(f"  {'─' * 56}")
    for label, p in [("100% Donchian", p_100d), ("100% CrashBot", p_100c), ("50/50", p_5050)]:
        print(f"  {label:20s}  ${p.final_equity:>8.2f}  {p.total_return * 100:>+7.1f}%"
              f"  {p.max_drawdown * 100:>6.1f}%  {p.sharpe:>7.2f}")
    print()

    # ── 4. Sweep complet si demandé ────────────────────────────────────────
    sweep_results = None
    best_portfolio = p_5050  # default

    if args.sweep:
        sweep_results = _allocation_sweep(data, args.balance)
        best_sharpe, best_dd = _print_sweep(sweep_results)
        best_portfolio = best_sharpe
    else:
        # Sweep rapide pour trouver le meilleur
        sweep_results = _allocation_sweep(data, args.balance)
        best_portfolio = max(sweep_results, key=lambda r: r.sharpe)
        print(f"  🏆 Meilleur Sharpe : {best_portfolio.alloc_donchian * 100:.0f}% Donchian"
              f" / {best_portfolio.alloc_crash * 100:.0f}% Crash"
              f"  → Sharpe {best_portfolio.sharpe:.2f}"
              f", MaxDD {best_portfolio.max_drawdown * 100:.1f}%"
              f", Return {best_portfolio.total_return * 100:+.1f}%")
        print()

    # ── 5. Graphiques ──────────────────────────────────────────────────────
    _generate_charts(data, sweep_results, best_portfolio, corr, show=not args.no_show)
    _generate_drawdown_chart(data, best_portfolio, show=not args.no_show)

    # ── 6. Verdict final ──────────────────────────────────────────────────
    print("═" * 70)
    print("  📋 VERDICT")
    print("═" * 70)
    if abs(corr) < 0.2:
        print("  🟢 Corrélation TRÈS FAIBLE (ρ < 0.2)")
        print("     → Les deux stratégies sont quasi indépendantes.")
        print("     → Le portfolio combiné réduit significativement le drawdown.")
        print("     → ✅ RECOMMANDÉ : allouer aux deux stratégies.")
    elif abs(corr) < 0.4:
        print("  🟡 Corrélation FAIBLE (ρ < 0.4)")
        print("     → Bonne diversification, le portfolio combiné aide.")
        print("     → ✅ Allouer aux deux avec une pondération adaptée.")
    elif abs(corr) < 0.6:
        print("  🟠 Corrélation MODÉRÉE (ρ < 0.6)")
        print("     → Diversification partielle seulement.")
        print("     → ⚠️ Le drawdown consolidé peut surprendre en bear market.")
    else:
        print("  🔴 Corrélation ÉLEVÉE (ρ ≥ 0.6)")
        print("     → Peu de diversification, les deux bougent ensemble.")
        print("     → ⚠️ Le DD consolidé ne sera pas beaucoup mieux que chaque strat seule.")

    # Rapport consolidé avec le compte actuel
    actual_balance = 2226.0
    best_alloc_d = best_portfolio.alloc_donchian
    print()
    print(f"  💰 Sur ton compte ($2226) :")
    print(f"     Donchian : ${actual_balance * best_alloc_d:>8.0f} ({best_alloc_d * 100:.0f}%)")
    print(f"     CrashBot : ${actual_balance * (1 - best_alloc_d):>8.0f} ({(1 - best_alloc_d) * 100:.0f}%)")
    print("═" * 70)
    print()


if __name__ == "__main__":
    main()
