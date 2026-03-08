#!/usr/bin/env python3
"""
Backtest comparatif — Current vs Optimisé pour Infinity.

Usage:
    python -m backtest.run_optim_comparison
    python -m backtest.run_optim_comparison --years 6
    python -m backtest.run_optim_comparison --balance 1000

Compare les paramètres actuels du bot avec des variantes optimisées,
en affichant un tableau A/B clair (PnL, PF, WR, DD, trades).
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data_loader import download_candles, download_all_pairs
from src.core.models import Candle
from src.core.infinity_engine import (
    InfinityConfig,
    InfinityPhase,
    rsi_series,
    sma_series,
    check_first_entry,
    compute_buy_size,
    check_sell_conditions,
    check_override_sell,
    check_stop_loss,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    impulse_body_pct: float = 0.0
    impulse_vol_ratio: float = 0.0
    retrace_pct: float = 0.0
    hold_bars: int = 0


@dataclass
class BacktestResult:
    label: str
    trades: list[Trade]
    equity_curve: list[float]
    initial_balance: float
    final_equity: float
    config_desc: dict


def compute_kpis(result: BacktestResult) -> dict:
    """Calcule les KPIs à partir d'un résultat de backtest."""
    trades = result.trades
    n = len(trades)
    eq = result.equity_curve

    if n == 0:
        return {
            "label": result.label,
            "trades": 0, "win_rate": 0, "pf": 0,
            "pnl": 0, "avg_pnl": 0, "max_dd": 0,
            "final_eq": result.final_equity,
            "avg_win": 0, "avg_loss": 0,
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 1e-9

    # Max drawdown
    peak = result.initial_balance
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        dd = (e - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    avg_win = (gross_profit / len(wins)) if wins else 0
    avg_loss = (sum(t.pnl_usd for t in losses) / len(losses)) if losses else 0

    return {
        "label": result.label,
        "trades": n,
        "win_rate": len(wins) / n * 100,
        "pf": gross_profit / gross_loss if gross_loss > 0 else 999,
        "pnl": sum(t.pnl_usd for t in trades),
        "avg_pnl": sum(t.pnl_usd for t in trades) / n,
        "max_dd": max_dd * 100,
        "final_eq": result.final_equity,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INFINITY SIMULATOR (simplifié du run_backtest_infinity.py)
# ══════════════════════════════════════════════════════════════════════════════


def _define_infinity_configs() -> list[tuple[str, InfinityConfig, dict]]:
    """Renvoie la liste (label, config, description) des variants Infinity."""
    configs = []

    # ── CURRENT (production) ──
    configs.append(("CURRENT", InfinityConfig(), {
        "sell_levels": "+0.8/1.5/2.2/3/4%",
        "breakeven": "après TP1",
        "sl": "-15%",
    }))

    # ── OPT_A : sell levels plus larges, breakeven après TP2 ──
    cfg_a = InfinityConfig(
        sell_levels=(0.010, 0.020, 0.035, 0.050, 0.080),  # +1/2/3.5/5/8%
        breakeven_after_level=1,  # After TP2 (index 1)
    )
    configs.append(("OPT_A (wider sells)", cfg_a, {
        "sell_levels": "+1/2/3.5/5/8%",
        "breakeven": "après TP2",
        "sl": "-15%",
    }))

    # ── OPT_B : sell levels très larges, pas de breakeven, SL -12% ──
    cfg_b = InfinityConfig(
        sell_levels=(0.015, 0.030, 0.050, 0.070, 0.100),  # +1.5/3/5/7/10%
        use_breakeven_stop=False,
        stop_loss_pct=0.12,  # -12% du PMP
    )
    configs.append(("OPT_B (no BE, wide)", cfg_b, {
        "sell_levels": "+1.5/3/5/7/10%",
        "breakeven": "désactivé",
        "sl": "-12%",
    }))

    # ── OPT_C : wider sells + BE after TP1 (hybride) ──
    cfg_c = InfinityConfig(
        sell_levels=(0.010, 0.020, 0.035, 0.050, 0.080),  # +1/2/3.5/5/8%
        breakeven_after_level=0,  # BE après TP1 (comme CURRENT)
    )
    configs.append(("OPT_C (wider+BE TP1)", cfg_c, {
        "sell_levels": "+1/2/3.5/5/8%",
        "breakeven": "après TP1",
        "sl": "-15%",
    }))

    # ── OPT_D : progression géométrique + entrée plus profonde ──
    cfg_d = InfinityConfig(
        sell_levels=(0.012, 0.025, 0.040, 0.060, 0.100),  # +1.2/2.5/4/6/10%
        entry_drop_pct=0.07,        # Entrée à -7% (vs -5%)
        breakeven_after_level=1,    # BE après TP2
        sell_pcts=(0.15, 0.15, 0.20, 0.25, 0.25),  # Vend plus vers la fin
    )
    configs.append(("OPT_D (deep entry)", cfg_d, {
        "sell_levels": "+1.2/2.5/4/6/10%",
        "breakeven": "après TP2",
        "entry_drop": "-7%",
    }))

    return configs


def simulate_infinity(
    candles: list[Candle],
    config: InfinityConfig,
    initial_balance: float = 1000.0,
    capital_pct: float = 0.65,
    symbol: str = "BTC-USD",
) -> BacktestResult:
    """
    Simule le bot Infinity sur des bougies H4.
    Version simplifiée extraite de run_backtest_infinity.py.
    """
    n = len(candles)
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    volumes = [c.volume for c in candles]

    rsi = rsi_series(closes, config.rsi_period)
    vol_ma = sma_series(volumes, config.volume_ma_len)

    balance = initial_balance
    equity_curve: list[float] = []
    trades: list[Trade] = []
    cycle_count = 0

    # State
    phase = InfinityPhase.WAITING
    total_size = 0.0
    total_cost = 0.0
    size_remaining = 0.0
    pmp = 0.0
    buy_levels_hit: set[int] = set()
    sell_levels_hit: set[int] = set()
    breakeven_active = False
    reference_price = 0.0
    cooldown_until = 0
    consecutive_stops = 0
    cycle_start_bar = 0
    cycle_total_proceeds = 0.0
    cycle_total_fees = 0.0

    for i in range(config.trailing_high_period, n):
        close = closes[i]
        cur_rsi = rsi[i]
        cur_vol = volumes[i]
        cur_vol_ma = vol_ma[i]

        trailing_high = max(highs[max(0, i - config.trailing_high_period + 1): i + 1])

        position_value = size_remaining * close if size_remaining > 0 else 0
        equity = balance + position_value
        equity_curve.append(equity)

        if i < cooldown_until:
            continue
        if consecutive_stops >= config.max_consecutive_stops:
            continue

        # ── WAITING ──
        if phase == InfinityPhase.WAITING:
            entry_ok = check_first_entry(
                close=close,
                trailing_high=trailing_high,
                entry_drop_pct=config.entry_drop_pct,
                rsi=cur_rsi,
                rsi_max=config.first_entry_rsi_max,
                volume=cur_vol,
                volume_ma=cur_vol_ma,
                require_volume=config.require_volume_entry,
            )
            if entry_ok:
                cycle_count += 1
                phase = InfinityPhase.ACCUMULATING
                reference_price = trailing_high
                buy_levels_hit = set()
                sell_levels_hit = set()
                breakeven_active = False
                total_size = 0.0
                total_cost = 0.0
                size_remaining = 0.0
                pmp = 0.0
                cycle_start_bar = i
                cycle_total_proceeds = 0.0
                cycle_total_fees = 0.0

                # First buy (L1)
                alloc = balance * capital_pct
                target_amount = alloc * config.buy_pcts[0]
                buy_amount = compute_buy_size(
                    rsi=cur_rsi, rsi_full=config.rsi_full_buy,
                    rsi_half=config.rsi_half_buy,
                    target_amount=target_amount,
                    cash_available=balance,
                    max_invested=alloc * config.max_invested_pct,
                    already_invested=0.0,
                )
                if buy_amount > 0:
                    fee = buy_amount * config.maker_fee
                    balance -= buy_amount + fee
                    size = buy_amount / close
                    total_size += size
                    total_cost += buy_amount
                    size_remaining = total_size
                    pmp = total_cost / total_size
                    buy_levels_hit.add(0)
                    cycle_total_fees += fee

        # ── ACCUMULATING ──
        elif phase == InfinityPhase.ACCUMULATING:
            alloc = (balance + total_cost) * capital_pct

            # Additional buys
            for lvl_idx in range(1, len(config.buy_levels)):
                if lvl_idx in buy_levels_hit:
                    continue
                target_price = reference_price * (1 + config.buy_levels[lvl_idx])
                if close <= target_price:
                    target_amount = alloc * config.buy_pcts[lvl_idx]
                    buy_amount = compute_buy_size(
                        rsi=cur_rsi, rsi_full=config.rsi_full_buy,
                        rsi_half=config.rsi_half_buy,
                        target_amount=target_amount,
                        cash_available=balance,
                        max_invested=alloc * config.max_invested_pct,
                        already_invested=total_cost,
                    )
                    if buy_amount > 0:
                        fee = buy_amount * config.maker_fee
                        balance -= buy_amount + fee
                        size = buy_amount / close
                        total_size += size
                        total_cost += buy_amount
                        size_remaining = total_size - sum(0 for _ in [])  # recalc below
                        pmp = total_cost / total_size if total_size > 0 else close
                        buy_levels_hit.add(lvl_idx)
                        cycle_total_fees += fee

            # Check stop-loss
            if pmp > 0 and check_stop_loss(close, pmp, config.stop_loss_pct):
                proceeds = size_remaining * close
                fee = proceeds * config.taker_fee
                balance += proceeds - fee
                cycle_total_fees += fee
                pnl = proceeds - fee - total_cost + cycle_total_proceeds
                pnl_pct = pnl / total_cost * 100 if total_cost > 0 else 0
                trades.append(Trade(
                    symbol=symbol, side="LONG",
                    entry_price=pmp, exit_price=close,
                    size=total_size,
                    entry_time=candles[cycle_start_bar].timestamp,
                    exit_time=candles[i].timestamp,
                    pnl_usd=pnl, pnl_pct=pnl_pct,
                    exit_reason="STOP_LOSS",
                    hold_bars=i - cycle_start_bar,
                ))
                phase = InfinityPhase.WAITING
                size_remaining = 0.0
                cooldown_until = i + config.cooldown_bars
                consecutive_stops += 1
                continue

            # Check breakeven
            if breakeven_active and pmp > 0 and close <= pmp:
                proceeds = size_remaining * close
                fee = proceeds * config.taker_fee
                balance += proceeds - fee
                cycle_total_fees += fee
                pnl = proceeds - fee - total_cost + cycle_total_proceeds
                pnl_pct = pnl / total_cost * 100 if total_cost > 0 else 0
                trades.append(Trade(
                    symbol=symbol, side="LONG",
                    entry_price=pmp, exit_price=close,
                    size=total_size,
                    entry_time=candles[cycle_start_bar].timestamp,
                    exit_time=candles[i].timestamp,
                    pnl_usd=pnl, pnl_pct=pnl_pct,
                    exit_reason="BREAKEVEN",
                    hold_bars=i - cycle_start_bar,
                ))
                phase = InfinityPhase.WAITING
                size_remaining = 0.0
                consecutive_stops = 0
                cooldown_until = i + config.cooldown_bars
                continue

            # Check override sell (+20%)
            if pmp > 0 and check_override_sell(close, pmp, config.override_sell_pct):
                proceeds = size_remaining * close
                fee = proceeds * config.taker_fee
                balance += proceeds - fee
                cycle_total_fees += fee
                pnl = proceeds - fee - total_cost + cycle_total_proceeds
                pnl_pct = pnl / total_cost * 100 if total_cost > 0 else 0
                trades.append(Trade(
                    symbol=symbol, side="LONG",
                    entry_price=pmp, exit_price=close,
                    size=total_size,
                    entry_time=candles[cycle_start_bar].timestamp,
                    exit_time=candles[i].timestamp,
                    pnl_usd=pnl, pnl_pct=pnl_pct,
                    exit_reason="OVERRIDE_SELL",
                    hold_bars=i - cycle_start_bar,
                ))
                phase = InfinityPhase.WAITING
                size_remaining = 0.0
                consecutive_stops = 0
                cooldown_until = i + config.cooldown_bars
                continue

            # Check sell levels
            if pmp > 0 and size_remaining > 0:
                for lvl_idx_sell in range(len(config.sell_levels)):
                    if lvl_idx_sell in sell_levels_hit:
                        continue
                    if not check_sell_conditions(
                        close=close, pmp=pmp,
                        sell_level_pct=config.sell_levels[lvl_idx_sell],
                        rsi=cur_rsi,
                        rsi_sell_min=config.rsi_sell_min,
                    ):
                        continue
                    sell_frac = config.sell_pcts[lvl_idx_sell] if lvl_idx_sell < len(config.sell_pcts) else 0.20
                    sell_size = total_size * sell_frac
                    if sell_size > size_remaining:
                        sell_size = size_remaining
                    proceeds = sell_size * close
                    fee = proceeds * config.maker_fee
                    balance += proceeds - fee
                    size_remaining -= sell_size
                    cycle_total_proceeds += proceeds - fee
                    cycle_total_fees += fee
                    sell_levels_hit.add(lvl_idx_sell)

                    # Activate breakeven after configured level
                    if config.use_breakeven_stop and lvl_idx_sell >= config.breakeven_after_level:
                        breakeven_active = True

                    # Check if all sold
                    if size_remaining <= 0:
                        pnl = cycle_total_proceeds - total_cost
                        pnl_pct = pnl / total_cost * 100 if total_cost > 0 else 0
                        trades.append(Trade(
                            symbol=symbol, side="LONG",
                            entry_price=pmp, exit_price=close,
                            size=total_size,
                            entry_time=candles[cycle_start_bar].timestamp,
                            exit_time=candles[i].timestamp,
                            pnl_usd=pnl, pnl_pct=pnl_pct,
                            exit_reason="TP_COMPLETE",
                            hold_bars=i - cycle_start_bar,
                        ))
                        phase = InfinityPhase.WAITING
                        size_remaining = 0.0
                        consecutive_stops = 0
                        cooldown_until = i + config.cooldown_bars

            # Timeout
            if phase == InfinityPhase.ACCUMULATING and (i - cycle_start_bar) > config.cycle_timeout_bars:
                if size_remaining > 0:
                    proceeds = size_remaining * close
                    fee = proceeds * config.taker_fee
                    balance += proceeds - fee
                    cycle_total_fees += fee
                    pnl = proceeds - fee - total_cost + cycle_total_proceeds
                    pnl_pct = pnl / total_cost * 100 if total_cost > 0 else 0
                    trades.append(Trade(
                        symbol=symbol, side="LONG",
                        entry_price=pmp, exit_price=close,
                        size=total_size,
                        entry_time=candles[cycle_start_bar].timestamp,
                        exit_time=candles[i].timestamp,
                        pnl_usd=pnl, pnl_pct=pnl_pct,
                        exit_reason="TIMEOUT",
                        hold_bars=i - cycle_start_bar,
                    ))
                phase = InfinityPhase.WAITING
                size_remaining = 0.0
                cooldown_until = i + config.cooldown_bars

    return BacktestResult(
        label="",
        trades=trades,
        equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY & CHARTING
# ══════════════════════════════════════════════════════════════════════════════


def print_comparison_table(results: list[dict], bot_name: str) -> None:
    """Affiche un tableau comparatif formaté."""
    print(f"\n{'='*90}")
    print(f"  {bot_name} — Comparaison A/B")
    print(f"{'='*90}")

    header = f"{'Config':<25} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>10} {'Avg PnL':>9} {'Max DD%':>8} {'Final $':>10}"
    print(header)
    print("-" * 90)

    for kpi in results:
        line = (
            f"{kpi['label']:<25} "
            f"{kpi['trades']:>6} "
            f"{kpi['win_rate']:>6.1f}% "
            f"{kpi['pf']:>7.2f} "
            f"{kpi['pnl']:>+9.2f}$ "
            f"{kpi['avg_pnl']:>+8.2f}$ "
            f"{kpi['max_dd']:>7.1f}% "
            f"{kpi['final_eq']:>9.2f}$"
        )
        print(line)

    # Highlight best
    best_pnl = max(results, key=lambda r: r["pnl"])
    best_pf = max(results, key=lambda r: r["pf"])
    print("-" * 90)
    print(f"  Meilleur PnL  : {best_pnl['label']} ({best_pnl['pnl']:+.2f}$)")
    print(f"  Meilleur PF   : {best_pf['label']} (PF {best_pf['pf']:.2f})")

    # Detail — avg win vs avg loss
    print(f"\n  Détail R:R :")
    for kpi in results:
        rr = abs(kpi["avg_win"] / kpi["avg_loss"]) if kpi["avg_loss"] != 0 else 0
        print(f"    {kpi['label']:<25} Avg Win: {kpi['avg_win']:+.2f}$  Avg Loss: {kpi['avg_loss']:+.2f}$  R:R = {rr:.2f}")

    print()


def plot_equity_curves(results: list[BacktestResult], bot_name: str, filename: str) -> None:
    """Trace les courbes d'equity superposées."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"]
    for i, r in enumerate(results):
        ax.plot(r.equity_curve, label=r.label, color=colors[i % len(colors)], linewidth=1.5)

    ax.set_title(f"{bot_name} — Equity Curves Comparées", fontsize=14)
    ax.set_xlabel("Barres")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  📊 Chart sauvegardé : {path}")


def save_trades_csv(trades: list[Trade], filename: str) -> None:
    """Sauvegarde les trades dans un CSV."""
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol", "side", "entry_price", "exit_price", "size",
            "entry_time", "exit_time", "pnl_usd", "pnl_pct",
            "exit_reason", "impulse_body_pct", "impulse_vol_ratio",
            "retrace_pct", "hold_bars",
        ])
        for t in trades:
            w.writerow([
                t.symbol, t.side, t.entry_price, t.exit_price, t.size,
                t.entry_time, t.exit_time, f"{t.pnl_usd:.4f}",
                f"{t.pnl_pct:.4f}", t.exit_reason,
                f"{t.impulse_body_pct:.6f}", f"{t.impulse_vol_ratio:.2f}",
                f"{t.retrace_pct:.4f}", t.hold_bars,
            ])
    print(f"  💾 Trades CSV : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  EXIT REASON BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════


def print_exit_breakdown(results: list[BacktestResult], bot_name: str) -> None:
    """Affiche la répartition des raisons de sortie pour chaque config."""
    print(f"\n{'='*70}")
    print(f"  {bot_name} — Répartition des sorties")
    print(f"{'='*70}")

    for r in results:
        from collections import Counter
        exits = Counter(t.exit_reason for t in r.trades)
        total = len(r.trades) or 1
        print(f"\n  {r.label}:")
        for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            avg_pnl = sum(t.pnl_usd for t in r.trades if t.exit_reason == reason) / count
            print(f"    {reason:<15} : {count:>4} ({pct:>5.1f}%)  avg PnL: {avg_pnl:+.2f}$")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def run_infinity_comparison(years: int = 6, balance: float = 1000.0) -> None:
    """Lance le backtest Infinity avec variantes sur 3 paires (BTC, AAVE, XLM)."""
    print("\n" + "═" * 70)
    print("  INFINITY — Téléchargement des données H4 (3 paires)")
    print("═" * 70)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=years * 365)

    # ── Configs de production par paire ──
    prod_configs = {
        "BTC-USD": InfinityConfig(
            trailing_high_period=72, entry_drop_pct=0.05,
            buy_levels=(-0.05, -0.10, -0.15, -0.20, -0.25),
            buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
            sell_levels=(0.008, 0.015, 0.022, 0.030, 0.040),
            stop_loss_pct=0.15, max_invested_pct=0.70,
            first_entry_rsi_max=50.0, use_breakeven_stop=True,
            scale_with_equity=True, rsi_sell_min=0.0,
            maker_fee=0.0, taker_fee=0.0009,
        ),
        "AAVE-USD": InfinityConfig(
            trailing_high_period=48, entry_drop_pct=0.12,
            buy_levels=(-0.12, -0.20, -0.28, -0.35, -0.42),
            buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
            sell_levels=(0.020, 0.040, 0.060, 0.080, 0.120),
            stop_loss_pct=0.25, max_invested_pct=0.70,
            first_entry_rsi_max=50.0, use_breakeven_stop=True,
            scale_with_equity=True, rsi_sell_min=0.0,
            maker_fee=0.0, taker_fee=0.0009,
        ),
        "XLM-USD": InfinityConfig(
            trailing_high_period=48, entry_drop_pct=0.12,
            buy_levels=(-0.12, -0.20, -0.28, -0.35, -0.42),
            buy_pcts=(0.25, 0.20, 0.15, 0.10, 0.00),
            sell_levels=(0.008, 0.015, 0.022, 0.030, 0.040),
            stop_loss_pct=0.25, max_invested_pct=0.70,
            first_entry_rsi_max=50.0, use_breakeven_stop=True,
            scale_with_equity=True, rsi_sell_min=0.0,
            maker_fee=0.0, taker_fee=0.0009,
        ),
    }

    # ── Définir les variations à tester par paire ──
    def _make_variants(base: InfinityConfig, symbol: str) -> list[tuple[str, InfinityConfig, dict]]:
        """Crée les variantes à partir du config de prod d'une paire."""
        variants = []
        variants.append(("CURRENT", base, {"desc": "production"}))

        # OPT_A : sell levels ×1.25 (élargis de 25%)
        wider_sells = tuple(s * 1.25 for s in base.sell_levels)
        cfg_a = InfinityConfig(
            trailing_high_period=base.trailing_high_period,
            entry_drop_pct=base.entry_drop_pct,
            buy_levels=base.buy_levels, buy_pcts=base.buy_pcts,
            sell_levels=wider_sells,
            breakeven_after_level=1,  # BE après TP2
            stop_loss_pct=base.stop_loss_pct,
            max_invested_pct=base.max_invested_pct,
            first_entry_rsi_max=base.first_entry_rsi_max,
            use_breakeven_stop=True, scale_with_equity=True,
            rsi_sell_min=0.0, maker_fee=0.0, taker_fee=0.0009,
        )
        wider_str = "/".join(f"{s*100:.1f}" for s in wider_sells)
        variants.append(("OPT_A (wider+BE TP2)", cfg_a, {"sell_levels": wider_str}))

        # OPT_B : sell levels ×1.5 + BE after TP1
        wider2_sells = tuple(s * 1.5 for s in base.sell_levels)
        cfg_b = InfinityConfig(
            trailing_high_period=base.trailing_high_period,
            entry_drop_pct=base.entry_drop_pct,
            buy_levels=base.buy_levels, buy_pcts=base.buy_pcts,
            sell_levels=wider2_sells,
            breakeven_after_level=0,  # BE après TP1
            stop_loss_pct=base.stop_loss_pct,
            max_invested_pct=base.max_invested_pct,
            first_entry_rsi_max=base.first_entry_rsi_max,
            use_breakeven_stop=True, scale_with_equity=True,
            rsi_sell_min=0.0, maker_fee=0.0, taker_fee=0.0009,
        )
        wider2_str = "/".join(f"{s*100:.1f}" for s in wider2_sells)
        variants.append(("OPT_B (wider×1.5+BE TP1)", cfg_b, {"sell_levels": wider2_str}))

        # OPT_C : sell levels ×2 + no BE
        wider3_sells = tuple(s * 2.0 for s in base.sell_levels)
        cfg_c = InfinityConfig(
            trailing_high_period=base.trailing_high_period,
            entry_drop_pct=base.entry_drop_pct,
            buy_levels=base.buy_levels, buy_pcts=base.buy_pcts,
            sell_levels=wider3_sells,
            use_breakeven_stop=False,
            stop_loss_pct=base.stop_loss_pct,
            max_invested_pct=base.max_invested_pct,
            first_entry_rsi_max=base.first_entry_rsi_max,
            scale_with_equity=True,
            rsi_sell_min=0.0, maker_fee=0.0, taker_fee=0.0009,
        )
        wider3_str = "/".join(f"{s*100:.1f}" for s in wider3_sells)
        variants.append(("OPT_C (wider×2, no BE)", cfg_c, {"sell_levels": wider3_str}))

        return variants

    # ── Download + simulate each pair ──
    for symbol, base_cfg in prod_configs.items():
        candles = download_candles(symbol, start, end, interval="4h")
        logger.info("✅ %s H4 : %d bougies", symbol, len(candles))

        if len(candles) < 200:
            logger.warning("⚠️ %s : pas assez de données (%d), skip", symbol, len(candles))
            continue

        per_pair_balance = balance / len(prod_configs)
        variants = _make_variants(base_cfg, symbol)
        results: list[BacktestResult] = []
        kpis_list: list[dict] = []

        for label, cfg, desc in variants:
            logger.info("🔄 Infinity %s — %s…", symbol, label)
            result = simulate_infinity(candles, cfg, initial_balance=per_pair_balance, symbol=symbol)
            result.label = f"{label}"
            result.config_desc = desc
            results.append(result)
            kpis = compute_kpis(result)
            kpis_list.append(kpis)

        print_comparison_table(kpis_list, f"INFINITY {symbol} (H4, {len(candles)} bars)")
        print_exit_breakdown(results, f"INFINITY {symbol}")
        safe_sym = symbol.replace("-", "_")
        plot_equity_curves(results, f"Infinity ({symbol})", f"optim_infinity_{safe_sym}_equity.png")

        best = max(results, key=lambda r: r.final_equity)
        save_trades_csv(best.trades, f"optim_infinity_{safe_sym}_best_trades.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest comparatif d'optimisation Infinity")
    parser.add_argument("--years", type=int, default=6, help="Années de données H4 pour Infinity")
    parser.add_argument("--balance", type=float, default=1000, help="Capital initial ($)")
    args = parser.parse_args()

    run_infinity_comparison(years=args.years, balance=args.balance)

    print("\n" + "═" * 70)
    print("  ✅ Backtest comparatif terminé !")
    print("═" * 70)


if __name__ == "__main__":
    main()
