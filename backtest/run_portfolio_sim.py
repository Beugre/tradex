#!/usr/bin/env python
"""
🏦 Simulation Portfolio 3 Bots — 6 ans en parallèle

Allocation : 50% Trail / 20% Breakout / 30% Crash Bot
Capital total : $1000

Chaque bot tourne indépendamment sur sa part du capital.
L'equity combinée = somme des 3 equity curves.

Usage :
  PYTHONPATH=. .venv/bin/python backtest/run_portfolio_sim.py --months 72 --no-show
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from backtest.data_loader import download_all_pairs
from backtest.metrics import compute_metrics
from backtest.simulator import BacktestResult, Trade, EquityPoint

# ── Imports des 3 moteurs ──
from backtest.run_backtest_range_tptrail import TpTrailConfig, RangeTpTrailEngine
from backtest.simulator_breakout import BreakoutEngine, BreakoutSimConfig, BreakoutResult
from backtest.run_backtest_dipbuy import DipBuyConfig, DipBuyEngine, PAIRS_20

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("portfolio")

OUTPUT_DIR = Path(__file__).parent / "output"

# ═══════════════════════════════════════════════════════════════════════════════
# Paires
# ═══════════════════════════════════════════════════════════════════════════════

TRAIL_PAIRS = PAIRS_20  # 20 paires

BREAKOUT_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
    "LINK-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
    "DOT-USD", "ATOM-USD", "UNI-USD", "NEAR-USD",
    "LTC-USD", "ETC-USD", "FIL-USD", "AAVE-USD",
    "INJ-USD", "SUI-USD", "HBAR-USD", "BNB-USD",
]

CRASH_PAIRS = PAIRS_20  # 20 paires

ALL_PAIRS = sorted(set(TRAIL_PAIRS + BREAKOUT_PAIRS + CRASH_PAIRS))


# ═══════════════════════════════════════════════════════════════════════════════
# Configs de production pour chaque bot
# ═══════════════════════════════════════════════════════════════════════════════

def trail_config(balance: float) -> TpTrailConfig:
    """Config Trail Range prod — identique au bot tradex-binance."""
    return TpTrailConfig(
        initial_balance=balance,
        step_pct=0.01,          # trailing step 1%
        risk_percent=0.02,
        max_position_pct=0.30,
        max_simultaneous_positions=3,
        fee_pct=0.001,          # Binance spot
        slippage_pct=0.001,
        swing_lookback=3,
        candle_window=100,
        range_width_min=0.02,
        range_entry_buffer_pct=0.002,
        range_sl_buffer_pct=0.003,
        range_cooldown_bars=3,
        compound=False,
    )


def breakout_config(balance: float) -> BreakoutSimConfig:
    """Config Breakout prod — identique au bot tradex-binance-breakout."""
    return BreakoutSimConfig(
        initial_balance=balance,
        risk_percent=0.02,
        max_positions=3,
        fee_pct=0.001,
        slippage_pct=0.001,
        # Stratégie
        bb_period=20,
        bb_std=2.0,
        bb_width_expansion=1.2,
        donchian_period=20,
        adx_period=14,
        adx_threshold=25.0,
        vol_avg_period=20,
        vol_multiplier=1.2,
        sl_atr_mult=1.5,
        atr_period=14,
        # Production flags
        allow_short=False,          # LONG ONLY en prod
        adaptive_trailing=True,     # Trailing adaptatif par paliers
        trail_step_1_pct=0.02,
        trail_step_2_pct=0.05,
        trail_lock_1_pct=0.002,
        trail_lock_2_pct=0.02,
        trailing_atr_mult=2.0,
        activate_trailing_pct=0.01,
        use_kill_switch=True,
        kill_switch_pct=-0.10,
        # Disabled filters
        use_ema_filter=False,
        use_atr_squeeze=False,
        dynamic_sizing=False,
        compound=False,
        cooldown_bars=3,
    )


def crash_config(balance: float) -> DipBuyConfig:
    """Config Crash Bot gagnante — TP8 + ATR1.5 + Step0.5%."""
    return DipBuyConfig(
        initial_balance=balance,
        interval="4h",
        lookback_bars=12,            # 48h
        drop_threshold=0.20,         # -20%
        tp_pct=0.08,                 # TP 8%
        atr_sl_mult=1.5,             # SL = ATR × 1.5
        atr_period=14,
        trail_step_pct=0.005,        # step 0.5%
        trail_trigger_buffer=0.0005,
        risk_percent=0.02,
        max_simultaneous=5,
        cooldown_bars=6,
        fee_pct=0.00075,
        slippage_pct=0.001,
        # Filtres désactivés
        sma_period=0,
        equity_sma_period=0,
        max_portfolio_heat=1.0,
        btc_trend_filter=False,
        min_drop_recovery=0.0,
        rsi_max=0.0,
        vol_spike_mult=0.0,
        min_wick_ratio=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BotResult:
    name: str
    weight: float
    capital: float
    result: object           # BacktestResult or BreakoutResult
    equity_map: dict[int, float]  # ts → equity
    metrics: dict
    n_trades: int


def breakout_to_backtest_result(br: BreakoutResult) -> BacktestResult:
    """Convertit un BreakoutResult en BacktestResult pour compute_metrics."""
    from src.core.models import OrderSide, StrategyType
    from backtest.simulator import EquityPoint as SimEqPt

    trades = []
    for t in br.trades:
        side = OrderSide.BUY if t.direction.value == "LONG" else OrderSide.SELL
        trades.append(Trade(
            symbol=t.symbol,
            strategy=StrategyType.BREAKOUT,
            side=side,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            size=t.size,
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            pnl_usd=t.pnl_usd,
            pnl_pct=t.pnl_pct,
            exit_reason=t.exit_reason,
        ))

    eq_curve = [SimEqPt(timestamp=p.timestamp, equity=p.equity) for p in br.equity_curve]

    return BacktestResult(
        trades=trades,
        equity_curve=eq_curve,
        initial_balance=br.initial_balance,
        final_equity=br.final_equity,
        start_date=br.start_date,
        end_date=br.end_date,
        pairs=br.pairs,
    )


def build_equity_map(equity_curve) -> dict[int, float]:
    """Convertit une equity curve en dict {timestamp: equity}."""
    return {p.timestamp: p.equity for p in equity_curve}


def interpolate_equity(eq_map: dict[int, float], ts: int) -> float:
    """Trouve l'equity au timestamp donné (ou la plus proche avant)."""
    if ts in eq_map:
        return eq_map[ts]
    # Trouver le ts le plus proche en dessous
    timestamps = sorted(eq_map.keys())
    idx = bisect_right(timestamps, ts) - 1
    if idx >= 0:
        return eq_map[timestamps[idx]]
    # Avant toutes les données → capital initial
    return list(eq_map.values())[0] if eq_map else 0


def combined_equity_curve(
    bots: list[BotResult], timeline: list[int],
) -> list[EquityPoint]:
    """Construit l'equity curve combinée sur la timeline unifiée."""
    curve = []
    for ts in timeline:
        total = sum(interpolate_equity(b.equity_map, ts) for b in bots)
        curve.append(EquityPoint(timestamp=ts, equity=total))
    return curve


def compute_metrics_from_curve(
    eq_curve: list[EquityPoint],
    initial_balance: float,
    start_date: datetime,
    end_date: datetime,
    trades: list[Trade] | None = None,
) -> dict:
    """Calcule les métriques à partir d'une equity curve brute."""
    years = max((end_date - start_date).days / 365.25, 0.01)
    final = eq_curve[-1].equity if eq_curve else initial_balance
    total_return = (final - initial_balance) / initial_balance

    cagr = (final / initial_balance) ** (1 / years) - 1 if final > 0 else 0

    # Drawdown
    peak = initial_balance
    max_dd = 0.0
    for pt in eq_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak else 0
        max_dd = min(max_dd, dd)

    # Sharpe (sur rendements par barre H4)
    returns = []
    for i in range(1, len(eq_curve)):
        prev = eq_curve[i - 1].equity
        if prev > 0:
            returns.append((eq_curve[i].equity - prev) / prev)

    bars_per_year = 6 * 365.25
    if len(returns) >= 2:
        mu = statistics.mean(returns)
        std = statistics.stdev(returns)
        sharpe = (mu / std) * math.sqrt(bars_per_year) if std > 0 else 0
        neg = [r for r in returns if r < 0]
        if neg:
            down_std = math.sqrt(sum(r ** 2 for r in neg) / len(neg))
            sortino = (mu / down_std) * math.sqrt(bars_per_year) if down_std > 0 else 99
        else:
            sortino = 99.0
    else:
        sharpe = sortino = 0

    # Trades stats
    n = len(trades) if trades else 0
    if trades and n > 0:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gp = sum(t.pnl_usd for t in wins) or 0
        gl = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        pf = gp / gl
        avg_pnl = sum(t.pnl_usd for t in trades) / n
    else:
        win_rate = pf = avg_pnl = 0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": pf,
        "avg_pnl_usd": avg_pnl,
        "final_equity": final,
        "years": years,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Monthly returns combinées
# ═══════════════════════════════════════════════════════════════════════════════

def monthly_returns_from_curve(
    eq_curve: list[EquityPoint], initial: float,
) -> list[tuple[str, float]]:
    """Rendements mensuels depuis une equity curve."""
    by_month: dict[str, float] = {}
    for pt in eq_curve:
        dt = datetime.fromtimestamp(pt.timestamp / 1000, tz=timezone.utc)
        key = f"{dt.year}-{dt.month:02d}"
        by_month[key] = pt.equity

    months = sorted(by_month.keys())
    result = []
    prev = initial
    for m in months:
        cur = by_month[m]
        ret = (cur - prev) / prev if prev > 0 else 0
        result.append((m, ret))
        prev = cur
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Rapport
# ═══════════════════════════════════════════════════════════════════════════════

def print_portfolio_report(
    bots: list[BotResult],
    combined_metrics: dict,
    combined_eq: list[EquityPoint],
    total_capital: float,
    start_date: datetime,
    end_date: datetime,
) -> None:
    sep = "═" * 100
    print(f"\n{sep}")
    print(f"  🏦 SIMULATION PORTFOLIO 3 BOTS — {start_date.date()} → {end_date.date()}")
    print(f"  💰 Capital total : ${total_capital:,.0f}")
    print(sep)

    # ── Tableau individuel ──
    header = (
        f"  {'Bot':<14s} │ {'Alloc':>5s} │ {'Capital':>8s} │ "
        f"{'Trades':>6s} │ {'WR':>5s} │ {'PF':>6s} │ "
        f"{'Return':>8s} │ {'CAGR':>6s} │ {'Sharpe':>6s} │ "
        f"{'MaxDD':>7s} │ {'Final':>10s}"
    )
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    for b in bots:
        m = b.metrics
        row = (
            f"  {b.name:<14s} │ {b.weight*100:>4.0f}% │ "
            f"${b.capital:>7,.0f} │ "
            f"{m['n_trades']:>6d} │ {m['win_rate']:>4.0%} │ "
            f"{m['profit_factor']:>6.2f} │ {m['total_return']:>+7.1%} │ "
            f"{m['cagr']:>+5.1%} │ {m['sharpe']:>6.2f} │ "
            f"{m['max_drawdown']:>6.1%} │ ${m['final_equity']:>9,.2f}"
        )
        print(row)

    # ── Portfolio combiné ──
    print("  " + "─" * (len(header) - 2))
    cm = combined_metrics
    total_trades = sum(b.metrics["n_trades"] for b in bots)
    row = (
        f"  {'📊 PORTFOLIO':<14s} │ {'100':>4s}% │ "
        f"${total_capital:>7,.0f} │ "
        f"{total_trades:>6d} │ {'':>5s} │ "
        f"{'':>6s} │ {cm['total_return']:>+7.1%} │ "
        f"{cm['cagr']:>+5.1%} │ {cm['sharpe']:>6.2f} │ "
        f"{cm['max_drawdown']:>6.1%} │ ${cm['final_equity']:>9,.2f}"
    )
    print(row)

    # ── Contribution au PnL ──
    print(f"\n  📊 Contribution au PnL total :")
    total_pnl = cm["final_equity"] - total_capital
    for b in bots:
        bot_pnl = b.metrics["final_equity"] - b.capital
        pct = bot_pnl / total_pnl * 100 if total_pnl != 0 else 0
        print(f"     {b.name:<14s} : ${bot_pnl:>+9,.2f} ({pct:>5.1f}% du PnL total)")
    print(f"     {'TOTAL':<14s} : ${total_pnl:>+9,.2f}")

    # ── Rendements mensuels (best / worst) ──
    monthly = monthly_returns_from_curve(combined_eq, total_capital)
    if monthly:
        positive_months = sum(1 for _, r in monthly if r > 0)
        total_months = len(monthly)
        best_month = max(monthly, key=lambda x: x[1])
        worst_month = min(monthly, key=lambda x: x[1])
        avg_month = statistics.mean([r for _, r in monthly])

        print(f"\n  📅 Rendements mensuels (portfolio) :")
        print(f"     Mois positifs : {positive_months}/{total_months} ({positive_months/total_months*100:.0f}%)")
        print(f"     Moy mensuel   : {avg_month:+.2%}")
        print(f"     Meilleur      : {best_month[0]} → {best_month[1]:+.1%}")
        print(f"     Pire          : {worst_month[0]} → {worst_month[1]:+.1%}")

    # ── Corrélation des equity curves ──
    print(f"\n  🔗 Corrélation inter-bots (diversification) :")
    for i, b1 in enumerate(bots):
        for b2 in bots[i + 1:]:
            corr = _equity_correlation(b1, b2)
            emoji = "🟢" if corr < 0.3 else "🟡" if corr < 0.6 else "🔴"
            print(f"     {emoji} {b1.name} × {b2.name} : r = {corr:.2f}")

    print(f"\n{sep}\n")


def _equity_correlation(b1: BotResult, b2: BotResult) -> float:
    """Corrélation des rendements quotidiens entre 2 bots."""
    # Aligner les timestamps communs
    common_ts = sorted(set(b1.equity_map.keys()) & set(b2.equity_map.keys()))
    if len(common_ts) < 10:
        return 0.0

    # Calculer les rendements
    ret1, ret2 = [], []
    for i in range(1, len(common_ts)):
        ts = common_ts[i]
        ts_prev = common_ts[i - 1]
        e1_prev = b1.equity_map.get(ts_prev, 0)
        e1_cur = b1.equity_map.get(ts, 0)
        e2_prev = b2.equity_map.get(ts_prev, 0)
        e2_cur = b2.equity_map.get(ts, 0)
        if e1_prev > 0 and e2_prev > 0:
            ret1.append((e1_cur - e1_prev) / e1_prev)
            ret2.append((e2_cur - e2_prev) / e2_prev)

    if len(ret1) < 10:
        return 0.0

    # Pearson
    n = len(ret1)
    mu1 = sum(ret1) / n
    mu2 = sum(ret2) / n
    cov = sum((r1 - mu1) * (r2 - mu2) for r1, r2 in zip(ret1, ret2)) / n
    std1 = math.sqrt(sum((r - mu1) ** 2 for r in ret1) / n)
    std2 = math.sqrt(sum((r - mu2) ** 2 for r in ret2) / n)
    if std1 > 0 and std2 > 0:
        return cov / (std1 * std2)
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Graphique
# ═══════════════════════════════════════════════════════════════════════════════

def generate_portfolio_chart(
    bots: list[BotResult],
    combined_eq: list[EquityPoint],
    combined_metrics: dict,
    total_capital: float,
    show: bool = True,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), height_ratios=[3, 1, 1.5])
    plt.style.use("seaborn-v0_8-whitegrid")

    colors = {
        "Trail": "#2196F3",
        "Breakout": "#FF9800",
        "Crash": "#4CAF50",
        "Portfolio": "#E91E63",
    }

    # ── 1. Equity curves ──
    ax1 = axes[0]

    for b in bots:
        ts_sorted = sorted(b.equity_map.keys())
        dates = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in ts_sorted]
        equities = [b.equity_map[t] for t in ts_sorted]
        color = colors.get(b.name.split()[0], "#999")
        ax1.plot(dates, equities, color=color, linewidth=1, alpha=0.6,
                 label=f"{b.name} ({b.weight*100:.0f}%)")

    # Combined
    dates_c = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in combined_eq]
    eq_c = [p.equity for p in combined_eq]
    ax1.plot(dates_c, eq_c, color=colors["Portfolio"], linewidth=2.5,
             label="📊 Portfolio combiné", zorder=10)
    ax1.axhline(y=total_capital, color="gray", linestyle="--", alpha=0.4)
    ax1.fill_between(dates_c, total_capital, eq_c, alpha=0.05, color=colors["Portfolio"])

    cm = combined_metrics
    ax1.set_title(
        f"Portfolio 3 Bots — 50% Trail / 20% Breakout / 30% Crash\n"
        f"Return: {cm['total_return']:+.1%} | CAGR: {cm['cagr']:+.1%} | "
        f"Sharpe: {cm['sharpe']:.2f} | Max DD: {cm['max_drawdown']:.1%} | "
        f"${total_capital:,.0f} → ${cm['final_equity']:,.0f}",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9)

    # ── 2. Drawdown combiné ──
    ax2 = axes[1]
    peak = total_capital
    dd = []
    for e in eq_c:
        peak = max(peak, e)
        dd.append((e - peak) / peak if peak else 0)
    ax2.fill_between(dates_c, dd, alpha=0.3, color="#F44336")
    ax2.plot(dates_c, dd, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # ── 3. Allocation pie chart + rendements mensuels ──
    ax3 = axes[2]
    monthly = monthly_returns_from_curve(combined_eq, total_capital)
    if monthly:
        m_dates = []
        m_rets = []
        for key, ret in monthly:
            year, month = key.split("-")
            m_dates.append(datetime(int(year), int(month), 15, tzinfo=timezone.utc))
            m_rets.append(ret)
        bar_colors = ["#4CAF50" if r > 0 else "#F44336" for r in m_rets]
        ax3.bar(m_dates, m_rets, width=25, color=bar_colors, alpha=0.7)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.4)
        ax3.set_ylabel("Rendement mensuel")
        ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax3.set_title("Rendements mensuels du portfolio", fontsize=10)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "portfolio_3bots_6y.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Chart: {chart_path}")

    if show:
        try:
            import subprocess
            subprocess.run(["open", str(chart_path)], check=False)
        except Exception:
            pass

    return chart_path


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation Portfolio 3 Bots")
    parser.add_argument("--months", type=int, default=72)
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--no-show", action="store_true")
    # Allocation
    parser.add_argument("--trail-pct", type=float, default=0.50)
    parser.add_argument("--breakout-pct", type=float, default=0.20)
    parser.add_argument("--crash-pct", type=float, default=0.30)
    args = parser.parse_args()

    # Validation allocation
    total_alloc = args.trail_pct + args.breakout_pct + args.crash_pct
    assert abs(total_alloc - 1.0) < 0.01, f"Allocation doit = 100%, got {total_alloc*100:.0f}%"

    trail_capital = args.balance * args.trail_pct
    breakout_capital = args.balance * args.breakout_pct
    crash_capital = args.balance * args.crash_pct

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)

    print(f"\n{'═' * 100}")
    print(f"  🏦 SIMULATION PORTFOLIO 3 BOTS EN PARALLÈLE")
    print(f"  📅 {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  💰 Capital total : ${args.balance:,.0f}")
    print(f"     🔵 Trail Range   : ${trail_capital:,.0f} ({args.trail_pct*100:.0f}%)")
    print(f"     🟠 Breakout H4   : ${breakout_capital:,.0f} ({args.breakout_pct*100:.0f}%)")
    print(f"     🟢 Crash Bot     : ${crash_capital:,.0f} ({args.crash_pct*100:.0f}%)")
    print(f"{'═' * 100}")

    # ── Download all data ──
    logger.info("📥 Téléchargement données H4 pour toutes les paires…")
    all_candles = download_all_pairs(ALL_PAIRS, start, end, interval="4h")

    # Séparer les candles par bot
    trail_candles = {p: all_candles[p] for p in TRAIL_PAIRS if p in all_candles}
    breakout_candles = {p: all_candles[p] for p in BREAKOUT_PAIRS if p in all_candles}
    crash_candles = {p: all_candles[p] for p in CRASH_PAIRS if p in all_candles}

    bots: list[BotResult] = []

    # ═══════════════════════════════════════════════════════════════════════
    # BOT 1 — Trail Range (50%)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 80}")
    print(f"  🔵 BOT 1 — Trail Range ({args.trail_pct*100:.0f}% = ${trail_capital:,.0f})")
    print(f"{'─' * 80}")

    trail_cfg = trail_config(trail_capital)
    trail_engine = RangeTpTrailEngine(trail_candles, trail_cfg)
    trail_result = trail_engine.run()
    trail_metrics = compute_metrics(trail_result)

    bots.append(BotResult(
        name="Trail Range",
        weight=args.trail_pct,
        capital=trail_capital,
        result=trail_result,
        equity_map=build_equity_map(trail_result.equity_curve),
        metrics=trail_metrics,
        n_trades=trail_metrics["n_trades"],
    ))
    print(f"  ✅ Trail : {trail_metrics['n_trades']} trades | "
          f"Return {trail_metrics['total_return']:+.1%} | "
          f"DD {trail_metrics['max_drawdown']:.1%} | "
          f"Sharpe {trail_metrics['sharpe']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # BOT 2 — Breakout (20%)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 80}")
    print(f"  🟠 BOT 2 — Breakout H4 ({args.breakout_pct*100:.0f}% = ${breakout_capital:,.0f})")
    print(f"{'─' * 80}")

    if breakout_capital > 0:
        breakout_cfg = breakout_config(breakout_capital)
        breakout_engine = BreakoutEngine(breakout_candles, breakout_cfg)
        breakout_raw = breakout_engine.run()
        # Convertir en BacktestResult pour compute_metrics
        breakout_result = breakout_to_backtest_result(breakout_raw)
        breakout_metrics = compute_metrics(breakout_result)

        bots.append(BotResult(
            name="Breakout H4",
            weight=args.breakout_pct,
            capital=breakout_capital,
            result=breakout_result,
            equity_map=build_equity_map(breakout_raw.equity_curve),
            metrics=breakout_metrics,
            n_trades=breakout_metrics["n_trades"],
        ))
        print(f"  ✅ Breakout : {breakout_metrics['n_trades']} trades | "
              f"Return {breakout_metrics['total_return']:+.1%} | "
              f"DD {breakout_metrics['max_drawdown']:.1%} | "
              f"Sharpe {breakout_metrics['sharpe']:.2f}")
    else:
        print("  ⏭️  Breakout désactivé (allocation 0%)")

    # ═══════════════════════════════════════════════════════════════════════
    # BOT 3 — Crash Bot (30%)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 80}")
    print(f"  🟢 BOT 3 — Crash Bot ({args.crash_pct*100:.0f}% = ${crash_capital:,.0f})")
    print(f"{'─' * 80}")

    crash_cfg = crash_config(crash_capital)
    crash_engine = DipBuyEngine(crash_candles, crash_cfg)
    crash_result = crash_engine.run()
    crash_metrics = compute_metrics(crash_result)

    bots.append(BotResult(
        name="Crash Bot",
        weight=args.crash_pct,
        capital=crash_capital,
        result=crash_result,
        equity_map=build_equity_map(crash_result.equity_curve),
        metrics=crash_metrics,
        n_trades=crash_metrics["n_trades"],
    ))
    print(f"  ✅ Crash : {crash_metrics['n_trades']} trades | "
          f"Return {crash_metrics['total_return']:+.1%} | "
          f"DD {crash_metrics['max_drawdown']:.1%} | "
          f"Sharpe {crash_metrics['sharpe']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # PORTFOLIO COMBINÉ
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 80}")
    print(f"  📊 Calcul du portfolio combiné…")
    print(f"{'─' * 80}")

    # Timeline unifiée (union de tous les timestamps)
    all_ts = set()
    for b in bots:
        all_ts.update(b.equity_map.keys())
    timeline = sorted(all_ts)

    combined_eq = combined_equity_curve(bots, timeline)

    # Métriques combinées
    combined_m = compute_metrics_from_curve(
        combined_eq,
        initial_balance=args.balance,
        start_date=start,
        end_date=end,
    )

    # ── Rapport ──
    print_portfolio_report(bots, combined_m, combined_eq, args.balance, start, end)

    # ── Chart ──
    generate_portfolio_chart(
        bots, combined_eq, combined_m, args.balance,
        show=not args.no_show,
    )

    # ── Comparaison avec chaque bot solo à 100% ──
    print(f"\n{'═' * 100}")
    print(f"  📊 COMPARAISON : Portfolio vs chaque bot seul à 100% du capital")
    print(f"{'═' * 100}")

    header = (
        f"  {'Scénario':<24s} │ {'Trades':>6s} │ {'Return':>8s} │ "
        f"{'CAGR':>6s} │ {'Sharpe':>6s} │ {'MaxDD':>7s} │ {'Final':>10s}"
    )
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    # Portfolio
    row = (
        f"  {'📊 PORTFOLIO 50/20/30':<24s} │ "
        f"{sum(b.n_trades for b in bots):>6d} │ "
        f"{combined_m['total_return']:>+7.1%} │ "
        f"{combined_m['cagr']:>+5.1%} │ "
        f"{combined_m['sharpe']:>6.2f} │ "
        f"{combined_m['max_drawdown']:>6.1%} │ "
        f"${combined_m['final_equity']:>9,.2f}"
    )
    print(row)

    # Chaque bot solo
    for b in bots:
        # Recalculer le return comme si le bot avait 100% du capital
        solo_return = b.metrics["total_return"]
        solo_final = args.balance * (1 + solo_return)
        row = (
            f"  {b.name + ' 100%':<24s} │ "
            f"{b.metrics['n_trades']:>6d} │ "
            f"{solo_return:>+7.1%} │ "
            f"{b.metrics['cagr']:>+5.1%} │ "
            f"{b.metrics['sharpe']:>6.2f} │ "
            f"{b.metrics['max_drawdown']:>6.1%} │ "
            f"${solo_final:>9,.2f}"
        )
        print(row)

    print()

    # ── Verdict diversification ──
    portfolio_sharpe = combined_m["sharpe"]
    best_solo_sharpe = max(b.metrics["sharpe"] for b in bots)
    portfolio_dd = combined_m["max_drawdown"]
    worst_solo_dd = min(b.metrics["max_drawdown"] for b in bots)

    print(f"  📊 Effet de diversification :")
    print(f"     Portfolio Sharpe : {portfolio_sharpe:.2f} vs meilleur solo : {best_solo_sharpe:.2f}")
    print(f"     Portfolio DD     : {portfolio_dd:.1%} vs pire solo : {worst_solo_dd:.1%}")

    if portfolio_sharpe >= best_solo_sharpe * 0.9 and portfolio_dd > worst_solo_dd:
        print(f"\n  ✅ Diversification efficace — meilleur risk-adjusted return")
    elif portfolio_dd > worst_solo_dd:
        print(f"\n  🟡 DD réduit par diversification, mais Sharpe légèrement inférieur")
    else:
        print(f"\n  ⚠️  La diversification n'améliore pas le profil de risque")

    print()


if __name__ == "__main__":
    main()
