#!/usr/bin/env python3
"""
Backtest Infinity Bot — DCA inversé sur une paire au choix.

Usage:
    python -m backtest.run_backtest_infinity              # BTC-USD 6 ans
    python -m backtest.run_backtest_infinity --pair ETH-USD
    python -m backtest.run_backtest_infinity --pair ETH-USD --years 4
    python -m backtest.run_backtest_infinity --pair ETH-USD --balance 1000

Simule fidèlement la logique de `infinity_engine.py` :
  - Trailing high H4 (72 bars = 12 jours)
  - Entrée quand drop ≥ 5% + RSI ≤ 50
  - 5 paliers d'achat DCA (-5%, -10%, -15%, -20%, -25%)
  - 5 paliers de vente (+0.8%, +1.5%, +2.2%, +3.0%, +4.0% du PMP)
  - Breakeven stop après TP1
  - Stop-loss -15% du PMP
  - Override sell +20% du PMP
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data_loader import download_candles
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


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class InfinityTrade:
    """Un cycle complet (entrée → sortie)."""
    cycle: int
    buys: list[dict] = field(default_factory=list)    # {bar, price, size, cost, level}
    sells: list[dict] = field(default_factory=list)   # {bar, price, size, proceeds, level}
    pmp: float = 0.0
    total_cost: float = 0.0
    total_proceeds: float = 0.0
    total_size: float = 0.0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    start_bar: int = 0
    end_bar: int = 0
    start_ts: int = 0
    end_ts: int = 0


# ── Simulator ──────────────────────────────────────────────────────────────────


def run_infinity_backtest(
    candles: list,
    config: InfinityConfig,
    initial_balance: float = 1000.0,
    capital_pct: float = 0.65,
) -> tuple[list[InfinityTrade], list[float]]:
    """
    Simule le bot Infinity sur des bougies H4.

    Retourne: (trades, equity_curve)
    """
    n = len(candles)
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    volumes = [c.volume for c in candles]

    # Indicateurs pré-calculés
    rsi = rsi_series(closes, config.rsi_period)
    vol_ma = sma_series(volumes, config.volume_ma_len)

    # État
    balance = initial_balance
    equity_curve = []
    trades: list[InfinityTrade] = []
    cycle_count = 0

    # Cycle state
    phase = InfinityPhase.WAITING
    current_trade: InfinityTrade | None = None
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

    for i in range(config.trailing_high_period, n):
        close = closes[i]
        high = highs[i]
        cur_rsi = rsi[i]
        cur_vol = volumes[i]
        cur_vol_ma = vol_ma[i]

        # Trailing high (rolling max des highs sur la fenêtre)
        trailing_high = max(highs[max(0, i - config.trailing_high_period + 1): i + 1])

        # Capital alloué
        allocated = balance * capital_pct if phase == InfinityPhase.WAITING else balance * capital_pct

        # Equity = balance + valeur positions ouvertes
        position_value = size_remaining * close if size_remaining > 0 else 0
        equity = balance + position_value
        equity_curve.append(equity)

        # Cooldown check
        if i < cooldown_until:
            continue

        # Max consecutive stops
        if consecutive_stops >= config.max_consecutive_stops:
            continue

        # ── PHASE: WAITING ──
        if phase == InfinityPhase.WAITING:
            # Check first entry conditions
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

                current_trade = InfinityTrade(
                    cycle=cycle_count,
                    start_bar=i,
                    start_ts=candles[i].timestamp,
                )

                # Execute first buy (L1)
                alloc = balance * capital_pct
                target_amount = alloc * config.buy_pcts[0]
                buy_amount = compute_buy_size(
                    rsi=cur_rsi,
                    rsi_full=config.rsi_full_buy,
                    rsi_half=config.rsi_half_buy,
                    target_amount=target_amount,
                    cash_available=balance,
                    max_invested=alloc * config.max_invested_pct,
                    already_invested=0.0,
                )
                if buy_amount > 0:
                    fee = buy_amount * config.maker_fee
                    net_cost = buy_amount + fee
                    size = buy_amount / close
                    balance -= net_cost
                    total_size += size
                    total_cost += buy_amount
                    size_remaining = total_size
                    pmp = total_cost / total_size if total_size > 0 else close
                    buy_levels_hit.add(0)
                    current_trade.buys.append({
                        "bar": i, "price": close, "size": size,
                        "cost": buy_amount, "level": 0, "fee": fee,
                    })
                    current_trade.fees += fee

        # ── PHASE: ACCUMULATING ──
        elif phase == InfinityPhase.ACCUMULATING:
            # Check additional buy levels
            alloc = (balance + total_cost) * capital_pct  # Equity at start
            for lvl_idx in range(1, len(config.buy_levels)):
                if lvl_idx in buy_levels_hit:
                    continue
                lvl_drop = config.buy_levels[lvl_idx]
                target_price = reference_price * (1 + lvl_drop)
                if close <= target_price:
                    target_amount = alloc * config.buy_pcts[lvl_idx] if config.scale_with_equity else config.buy_amounts[lvl_idx]
                    buy_amount = compute_buy_size(
                        rsi=cur_rsi,
                        rsi_full=config.rsi_full_buy,
                        rsi_half=config.rsi_half_buy,
                        target_amount=target_amount,
                        cash_available=balance,
                        max_invested=alloc * config.max_invested_pct,
                        already_invested=total_cost,
                    )
                    if buy_amount > 0:
                        fee = buy_amount * config.maker_fee
                        net_cost = buy_amount + fee
                        size = buy_amount / close
                        balance -= net_cost
                        total_size += size
                        total_cost += buy_amount
                        size_remaining = total_size - sum(
                            s["size"] for s in (current_trade.sells if current_trade else [])
                        )
                        pmp = total_cost / total_size if total_size > 0 else close
                        buy_levels_hit.add(lvl_idx)
                        if current_trade:
                            current_trade.buys.append({
                                "bar": i, "price": close, "size": size,
                                "cost": buy_amount, "level": lvl_idx, "fee": fee,
                            })
                            current_trade.fees += fee

            # Check stop-loss
            if pmp > 0 and check_stop_loss(close, pmp, config.stop_loss_pct):
                # Market sell all remaining
                proceeds = size_remaining * close
                fee = proceeds * config.taker_fee
                net_proceeds = proceeds - fee
                balance += net_proceeds
                if current_trade:
                    current_trade.sells.append({
                        "bar": i, "price": close, "size": size_remaining,
                        "proceeds": net_proceeds, "level": -1, "fee": fee,
                    })
                    current_trade.fees += fee
                    current_trade.exit_reason = "STOP_LOSS"
                    current_trade.end_bar = i
                    current_trade.end_ts = candles[i].timestamp
                    current_trade.pmp = pmp
                    current_trade.total_cost = total_cost
                    current_trade.total_proceeds = sum(s["proceeds"] for s in current_trade.sells)
                    current_trade.total_size = total_size
                    current_trade.pnl_usd = current_trade.total_proceeds - current_trade.total_cost
                    current_trade.pnl_pct = current_trade.pnl_usd / current_trade.total_cost * 100 if current_trade.total_cost > 0 else 0
                    trades.append(current_trade)
                phase = InfinityPhase.WAITING
                size_remaining = 0.0
                cooldown_until = i + config.cooldown_bars
                consecutive_stops += 1
                current_trade = None
                continue

            # Check breakeven stop (active after TP1)
            if breakeven_active and pmp > 0 and close <= pmp:
                proceeds = size_remaining * close
                fee = proceeds * config.taker_fee
                net_proceeds = proceeds - fee
                balance += net_proceeds
                if current_trade:
                    current_trade.sells.append({
                        "bar": i, "price": close, "size": size_remaining,
                        "proceeds": net_proceeds, "level": -2, "fee": fee,
                    })
                    current_trade.fees += fee
                    current_trade.exit_reason = "BREAKEVEN"
                    current_trade.end_bar = i
                    current_trade.end_ts = candles[i].timestamp
                    current_trade.pmp = pmp
                    current_trade.total_cost = total_cost
                    current_trade.total_proceeds = sum(s["proceeds"] for s in current_trade.sells)
                    current_trade.total_size = total_size
                    current_trade.pnl_usd = current_trade.total_proceeds - current_trade.total_cost
                    current_trade.pnl_pct = current_trade.pnl_usd / current_trade.total_cost * 100 if current_trade.total_cost > 0 else 0
                    trades.append(current_trade)
                phase = InfinityPhase.WAITING
                size_remaining = 0.0
                cooldown_until = i + config.cooldown_bars
                current_trade = None
                continue

            # Check override sell (+20% PMP)
            if pmp > 0 and check_override_sell(close, pmp, config.override_sell_pct):
                proceeds = size_remaining * close
                fee = proceeds * config.maker_fee
                net_proceeds = proceeds - fee
                balance += net_proceeds
                if current_trade:
                    current_trade.sells.append({
                        "bar": i, "price": close, "size": size_remaining,
                        "proceeds": net_proceeds, "level": -3, "fee": fee,
                    })
                    current_trade.fees += fee
                    current_trade.exit_reason = "OVERRIDE_SELL"
                    current_trade.end_bar = i
                    current_trade.end_ts = candles[i].timestamp
                    current_trade.pmp = pmp
                    current_trade.total_cost = total_cost
                    current_trade.total_proceeds = sum(s["proceeds"] for s in current_trade.sells)
                    current_trade.total_size = total_size
                    current_trade.pnl_usd = current_trade.total_proceeds - current_trade.total_cost
                    current_trade.pnl_pct = current_trade.pnl_usd / current_trade.total_cost * 100 if current_trade.total_cost > 0 else 0
                    trades.append(current_trade)
                    consecutive_stops = 0
                phase = InfinityPhase.WAITING
                size_remaining = 0.0
                cooldown_until = i + config.cooldown_bars
                current_trade = None
                continue

            # Check sell levels (distribution)
            if pmp > 0 and size_remaining > 0:
                for sell_idx in range(len(config.sell_levels)):
                    if sell_idx in sell_levels_hit:
                        continue
                    if check_sell_conditions(close, pmp, config.sell_levels[sell_idx], cur_rsi, config.rsi_sell_min):
                        sell_size = total_size * config.sell_pcts[sell_idx]
                        sell_size = min(sell_size, size_remaining)
                        if sell_size <= 0:
                            continue
                        proceeds = sell_size * close
                        fee = proceeds * config.maker_fee
                        net_proceeds = proceeds - fee
                        balance += net_proceeds
                        size_remaining -= sell_size
                        sell_levels_hit.add(sell_idx)

                        if current_trade:
                            current_trade.sells.append({
                                "bar": i, "price": close, "size": sell_size,
                                "proceeds": net_proceeds, "level": sell_idx, "fee": fee,
                            })
                            current_trade.fees += fee

                        # Activate breakeven after first TP
                        if sell_idx >= config.breakeven_after_level and config.use_breakeven_stop:
                            breakeven_active = True

                        # All sold?
                        if size_remaining <= 1e-12 or len(sell_levels_hit) >= len(config.sell_levels):
                            if current_trade:
                                current_trade.exit_reason = "TP_COMPLETE"
                                current_trade.end_bar = i
                                current_trade.end_ts = candles[i].timestamp
                                current_trade.pmp = pmp
                                current_trade.total_cost = total_cost
                                current_trade.total_proceeds = sum(s["proceeds"] for s in current_trade.sells)
                                current_trade.total_size = total_size
                                current_trade.pnl_usd = current_trade.total_proceeds - current_trade.total_cost
                                current_trade.pnl_pct = current_trade.pnl_usd / current_trade.total_cost * 100 if current_trade.total_cost > 0 else 0
                                trades.append(current_trade)
                                consecutive_stops = 0
                            phase = InfinityPhase.WAITING
                            size_remaining = 0.0
                            cooldown_until = i + config.cooldown_bars
                            current_trade = None
                            break

    # Close any open position at end
    if phase != InfinityPhase.WAITING and size_remaining > 0 and current_trade:
        close_price = closes[-1]
        proceeds = size_remaining * close_price
        fee = proceeds * config.taker_fee
        net_proceeds = proceeds - fee
        balance += net_proceeds
        current_trade.sells.append({
            "bar": n - 1, "price": close_price, "size": size_remaining,
            "proceeds": net_proceeds, "level": -99, "fee": fee,
        })
        current_trade.fees += fee
        current_trade.exit_reason = "END_OF_DATA"
        current_trade.end_bar = n - 1
        current_trade.end_ts = candles[-1].timestamp
        current_trade.pmp = pmp
        current_trade.total_cost = total_cost
        current_trade.total_proceeds = sum(s["proceeds"] for s in current_trade.sells)
        current_trade.total_size = total_size
        current_trade.pnl_usd = current_trade.total_proceeds - current_trade.total_cost
        current_trade.pnl_pct = current_trade.pnl_usd / current_trade.total_cost * 100 if current_trade.total_cost > 0 else 0
        trades.append(current_trade)

    # Pad equity curve for skipped bars
    while len(equity_curve) < n - config.trailing_high_period:
        equity_curve.append(equity_curve[-1] if equity_curve else initial_balance)

    return trades, equity_curve


# ── Report ─────────────────────────────────────────────────────────────────────


def print_report(
    pair: str,
    trades: list[InfinityTrade],
    equity_curve: list[float],
    initial_balance: float,
    candles: list,
    config: InfinityConfig,
) -> None:
    """Affiche le rapport console et génère les graphiques."""
    n_trades = len(trades)
    if n_trades == 0:
        logger.info("❌ Aucun trade — pas de rapport à générer.")
        return

    final_equity = equity_curve[-1] if equity_curve else initial_balance
    total_return = (final_equity - initial_balance) / initial_balance * 100
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    win_rate = len(wins) / n_trades * 100
    total_pnl = sum(t.pnl_usd for t in trades)
    total_fees = sum(t.fees for t in trades)
    avg_pnl = total_pnl / n_trades
    avg_win = sum(t.pnl_usd for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t.pnl_usd for t in losses) / len(losses) if losses else 0

    gross_gains = sum(t.pnl_usd for t in wins)
    gross_losses = abs(sum(t.pnl_usd for t in losses))
    pf = gross_gains / gross_losses if gross_losses > 0 else float("inf")

    # Drawdown
    peak = initial_balance
    max_dd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized, H4 bars)
    if len(equity_curve) > 1:
        returns = [(equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                    for i in range(1, len(equity_curve)) if equity_curve[i - 1] > 0]
        if returns:
            import statistics
            mean_r = statistics.mean(returns)
            std_r = statistics.stdev(returns) if len(returns) > 1 else 1
            # H4 = 6 bars/jour = 2190 bars/an
            sharpe = (mean_r / std_r) * (2190 ** 0.5) if std_r > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    # Years
    if candles:
        start_dt = datetime.fromtimestamp(candles[0].timestamp / 1000, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(candles[-1].timestamp / 1000, tz=timezone.utc)
        years = (end_dt - start_dt).days / 365.25
    else:
        years = 1

    # Exit reasons
    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1

    # Buy levels used
    buy_level_counts = [0] * 5
    for t in trades:
        for b in t.buys:
            if 0 <= b["level"] < 5:
                buy_level_counts[b["level"]] += 1

    print()
    print("=" * 70)
    print(f"♾️  BACKTEST INFINITY — {pair}")
    print(f"   Période: {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d} ({years:.1f} ans)")
    print(f"   Balance initiale: ${initial_balance:,.0f}")
    print("=" * 70)
    print()
    print(f"  📊 Trades          : {n_trades}")
    print(f"  ✅ Win Rate        : {win_rate:.0f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  💰 PnL total       : ${total_pnl:+,.2f}")
    print(f"  💰 PnL moyen       : ${avg_pnl:+,.2f}")
    print(f"  🟢 Gain moyen      : ${avg_win:+,.2f}")
    print(f"  🔴 Perte moyenne   : ${avg_loss:+,.2f}")
    print(f"  📈 Profit Factor   : {pf:.2f}")
    print(f"  📈 Rendement total : {total_return:+.2f}%")
    print(f"  📉 Max Drawdown    : {max_dd * 100:.2f}%")
    print(f"  📊 Sharpe (ann.)   : {sharpe:.2f}")
    print(f"  💸 Frais totaux    : ${total_fees:.2f}")
    print(f"  🏦 Equity finale   : ${final_equity:,.2f}")
    print()
    print("  📋 Sorties :")
    for reason, count in sorted(exit_counts.items()):
        print(f"     {reason}: {count}")
    print()
    print("  📋 Paliers d'achat utilisés :")
    for lvl in range(5):
        pct = config.buy_levels[lvl] * 100
        print(f"     L{lvl + 1} ({pct:+.0f}%) : {buy_level_counts[lvl]} fois")
    print()
    print("=" * 70)

    # ── Détail des trades ──
    print()
    print("  # | Buys | PMP     | Exit     | PnL        | Durée (bars)")
    print("  " + "-" * 62)
    for t in trades:
        dur = t.end_bar - t.start_bar
        emoji = "🟢" if t.pnl_usd > 0 else "🔴"
        print(f"  {emoji} {t.cycle:>2} | {len(t.buys):>4} | ${t.pmp:>8,.2f} | {t.exit_reason:<15} | ${t.pnl_usd:>+8.2f} ({t.pnl_pct:+.1f}%) | {dur}")
    print()

    # ── Graphiques ──
    _generate_charts(pair, trades, equity_curve, candles, config, initial_balance)


def _generate_charts(
    pair: str,
    trades: list[InfinityTrade],
    equity_curve: list[float],
    candles: list,
    config: InfinityConfig,
    initial_balance: float,
) -> None:
    """Génère les graphiques."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    offset = config.trailing_high_period
    dates = [
        datetime.fromtimestamp(candles[i].timestamp / 1000, tz=timezone.utc)
        for i in range(offset, offset + len(equity_curve))
    ]
    prices = [candles[i].close for i in range(offset, offset + len(equity_curve))]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"♾️ Infinity Bot Backtest — {pair}", fontsize=14, fontweight="bold")

    # 1. Prix + signaux
    ax1 = axes[0]
    ax1.plot(dates, prices, color="gray", linewidth=0.8, alpha=0.7, label="Prix")
    for t in trades:
        for b in t.buys:
            idx = b["bar"] - offset
            if 0 <= idx < len(dates):
                ax1.scatter(dates[idx], b["price"], color="green", marker="^", s=40, zorder=5)
        for s in t.sells:
            idx = s["bar"] - offset
            if 0 <= idx < len(dates):
                color = "red" if s["level"] < 0 else "blue"
                marker = "v" if s["level"] < 0 else "o"
                ax1.scatter(dates[idx], s["price"], color=color, marker=marker, s=40, zorder=5)
    ax1.set_ylabel("Prix ($)")
    ax1.set_title("Prix + Achats (▲) / Ventes (●) / Stops (▼)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Equity curve
    ax2 = axes[1]
    ax2.plot(dates, equity_curve, color="royalblue", linewidth=1.2)
    ax2.axhline(y=initial_balance, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Equity ($)")
    ax2.set_title("Equity Curve")
    ax2.grid(True, alpha=0.3)

    # 3. PnL par trade
    ax3 = axes[2]
    trade_dates = []
    trade_pnls = []
    for t in trades:
        idx = t.end_bar - offset
        if 0 <= idx < len(dates):
            trade_dates.append(dates[idx])
            trade_pnls.append(t.pnl_usd)
    colors = ["green" if p > 0 else "red" for p in trade_pnls]
    ax3.bar(trade_dates, trade_pnls, color=colors, width=5, alpha=0.7)
    ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax3.set_ylabel("PnL ($)")
    ax3.set_title("PnL par cycle")
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    safe_pair = pair.replace("-", "_")
    chart_path = OUTPUT_DIR / f"infinity_{safe_pair}.png"
    plt.savefig(chart_path, dpi=150)
    logger.info("📊 Graphiques sauvés: %s", chart_path)
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest Infinity Bot")
    parser.add_argument("--pair", default="BTC-USD", help="Paire à backtester (ex: ETH-USD)")
    parser.add_argument("--years", type=float, default=6, help="Nombre d'années de données")
    parser.add_argument("--balance", type=float, default=1000, help="Balance initiale ($)")
    parser.add_argument("--capital-pct", type=float, default=0.65, help="% capital alloué")
    parser.add_argument("--no-breakeven", action="store_true", help="Désactiver le breakeven stop")
    parser.add_argument("--buy-levels", type=str, default=None,
                        help="Paliers d'achat DCA (ex: '-10,-15,-20,-25,-30')")
    args = parser.parse_args()

    end = datetime(2026, 3, 6, tzinfo=timezone.utc)
    start_year = end.year - int(args.years)
    start = datetime(start_year, end.month, end.day, tzinfo=timezone.utc)

    logger.info("♾️ Backtest Infinity — %s (%s → %s)", args.pair, start.date(), end.date())
    logger.info("   Balance: $%.0f | Capital %%: %.0f%% | Breakeven: %s",
                args.balance, args.capital_pct * 100,
                "OFF" if args.no_breakeven else "ON")

    # Téléchargement des bougies H4
    candles = download_candles(args.pair, start, end, interval="4h")
    logger.info("   %d bougies H4 chargées", len(candles))

    if len(candles) < 200:
        logger.error("❌ Pas assez de données")
        sys.exit(1)

    # Config
    extra_kwargs = {}
    if args.buy_levels:
        levels = tuple(float(x.strip()) / 100.0 for x in args.buy_levels.split(","))
        extra_kwargs["buy_levels"] = levels
        extra_kwargs["entry_drop_pct"] = abs(levels[0])
        extra_kwargs["max_buy_levels"] = len(levels)
        default_pcts = (0.25, 0.20, 0.15, 0.10, 0.00)
        extra_kwargs["buy_pcts"] = default_pcts[:len(levels)]
        default_amts = (100.0, 200.0, 300.0, 400.0, 0.0)
        extra_kwargs["buy_amounts"] = default_amts[:len(levels)]
        logger.info("   Buy levels custom: %s", [f"{l*100:.0f}%" for l in levels])

    config = InfinityConfig(
        initial_balance=args.balance,
        use_breakeven_stop=not args.no_breakeven,
        **extra_kwargs,
    )

    # Run
    trades, equity_curve = run_infinity_backtest(
        candles=candles,
        config=config,
        initial_balance=args.balance,
        capital_pct=args.capital_pct,
    )

    # Report
    print_report(args.pair, trades, equity_curve, args.balance, candles, config)


if __name__ == "__main__":
    main()
