#!/usr/bin/env python3
"""
Backtest Scalping V2 — Breakout Momentum + ATR dynamique.

Approche completement differente du V1 (RSI/EMA) :
  - Pas de RSI pour l'entree
  - Breakout du high recent (N barres) = momentum confirme
  - Volume > SMA(vol) = participation reelle
  - ATR en expansion = le marche bouge (pas de range mort)
  - TP dynamique = K * ATR (s'adapte a la volatilite)
  - SL dynamique = M * ATR
  - Trailing stop rapide apres activation

Usage:
    python -m backtest.run_backtest_scalping_v2 --compare
    python -m backtest.run_backtest_scalping_v2 --balance 1500 --years 1
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data_loader import download_candles
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class BreakoutConfig:
    """Configuration de la strategie breakout momentum."""
    name: str = "BRK_V1"

    # ── Breakout ──
    lookback: int = 12             # N barres pour le high/low recent
    breakout_margin_pct: float = 0.0001  # marge au-dessus du high pour confirmer

    # ── ATR ──
    atr_period: int = 14
    atr_expansion_lookback: int = 8    # ATR(now) > ATR(N barres avant)
    atr_expansion_ratio: float = 1.05  # ATR doit etre > 1.05x l'ATR passe

    # ── TP / SL dynamiques (multiples d'ATR) ──
    tp_atr_mult: float = 1.5          # TP = entry + 1.5 * ATR
    sl_atr_mult: float = 1.0          # SL = entry - 1.0 * ATR

    # ── Trailing stop rapide ──
    trailing_activation_atr: float = 0.8   # trailing s'active a entry + 0.8*ATR
    trailing_distance_atr: float = 0.5     # trailing suit a -0.5*ATR du high

    # ── Volume ──
    volume_ma_period: int = 20
    volume_spike_mult: float = 1.0     # volume > 1.0 * SMA(vol)

    # ── Filtres ──
    min_atr_pct: float = 0.001        # ATR min en % du prix (filtre range mort)
    cooldown_bars: int = 4            # barres min entre 2 trades sur meme paire

    # ── Anti-tilt ──
    max_consecutive_losses: int = 3
    cooldown_bars_after_tilt: int = 8

    # ── Session filter ──
    session_filter_enabled: bool = True
    sessions: list[tuple[int, int]] = field(
        default_factory=lambda: [(9, 12), (15, 18)]  # UTC
    )

    # ── Risk ──
    risk_pct: float = 0.03             # 3% du capital par trade
    max_position_pct: float = 0.25     # max 25% du capital par position

    # ── Fees ──
    entry_fee_pct: float = 0.0009      # 0.09% taker (Revolut)
    exit_fee_pct: float = 0.0009


# ── Trade ──────────────────────────────────────────────────────────────────────


@dataclass
class BreakoutTrade:
    symbol: str
    entry_bar: int
    entry_price: float
    entry_ts: int
    sl_price: float
    tp_price: float
    size: float
    atr_at_entry: float = 0.0
    high_broken: float = 0.0           # le high qui a ete casse
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_min: float = 0.0
    trailing_active: bool = False
    max_price_seen: float = 0.0        # pour trailing


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ── Indicateurs ────────────────────────────────────────────────────────────────


def atr_series(highs: list[float], lows: list[float], closes: list[float],
               period: int = 14) -> list[float]:
    """Average True Range (Wilder smoothing)."""
    n = len(closes)
    atr = [0.0] * n
    if n < 2:
        return atr

    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # SMA pour la premiere valeur
    if n >= period:
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    else:
        atr[-1] = sum(tr) / n

    return atr


def sma_series(values: list[float], period: int) -> list[float]:
    """SMA classique."""
    n = len(values)
    sma = [0.0] * n
    for i in range(n):
        if i < period - 1:
            sma[i] = sum(values[:i + 1]) / (i + 1)
        else:
            sma[i] = sum(values[i - period + 1:i + 1]) / period
    return sma


def rolling_high(highs: list[float], period: int) -> list[float]:
    """High glissant sur N barres (exclus la barre courante)."""
    n = len(highs)
    result = [0.0] * n
    for i in range(period, n):
        result[i] = max(highs[i - period:i])
    return result


def rolling_low(lows: list[float], period: int) -> list[float]:
    """Low glissant sur N barres (exclus la barre courante)."""
    n = len(lows)
    result = [0.0] * n
    for i in range(period, n):
        result[i] = min(lows[i - period:i])
    return result


# ── Simulation ─────────────────────────────────────────────────────────────────


def _bar_hour_utc(candle: Candle) -> int:
    dt = datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc)
    return dt.hour


def _in_session(hour: int, sessions: list[tuple[int, int]]) -> bool:
    return any(start <= hour < end for start, end in sessions)


def run_pair(
    symbol: str,
    candles: list[Candle],
    cfg: BreakoutConfig,
    initial_balance: float,
) -> tuple[list[BreakoutTrade], list[EquityPoint], float]:
    """Simule la strategie breakout momentum sur une paire."""
    n = len(candles)
    warmup = max(cfg.lookback, cfg.atr_period, cfg.volume_ma_period,
                 cfg.atr_expansion_lookback + cfg.atr_period) + 2
    if n < warmup + 5:
        return [], [], initial_balance

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    # Precalcul indicateurs
    atr = atr_series(highs, lows, closes, cfg.atr_period)
    vol_ma = sma_series(volumes, cfg.volume_ma_period)
    recent_high = rolling_high(highs, cfg.lookback)

    trades: list[BreakoutTrade] = []
    equity_curve: list[EquityPoint] = []
    balance = initial_balance

    open_trade: Optional[BreakoutTrade] = None
    consecutive_losses = 0
    cooldown_until = 0
    last_trade_bar = -999

    for i in range(warmup, n):
        c = candles[i]
        price = c.close

        # Log equity (1/jour = 96 barres 15m)
        if i % 96 == 0:
            port_val = balance
            if open_trade is not None:
                unrealized = (price - open_trade.entry_price) * open_trade.size
                port_val += unrealized
            equity_curve.append(EquityPoint(ts=c.timestamp, equity=port_val))

        # ── Gestion position ouverte ──
        if open_trade is not None:
            t = open_trade

            # Maj max price vu (pour trailing)
            if highs[i] > t.max_price_seen:
                t.max_price_seen = highs[i]

            # Trailing activation
            trail_activation_price = t.entry_price + cfg.trailing_activation_atr * t.atr_at_entry
            if t.max_price_seen >= trail_activation_price and not t.trailing_active:
                t.trailing_active = True

            # Trailing SL update
            if t.trailing_active:
                trail_sl = t.max_price_seen - cfg.trailing_distance_atr * t.atr_at_entry
                if trail_sl > t.sl_price:
                    t.sl_price = trail_sl

            # Check exits
            hit_sl = lows[i] <= t.sl_price
            hit_tp = highs[i] >= t.tp_price

            exit_price = 0.0
            exit_reason = ""

            if hit_sl and hit_tp:
                if c.open < c.close:
                    exit_price = t.sl_price
                    exit_reason = "TRAIL_SL" if t.trailing_active else "SL"
                else:
                    exit_price = t.tp_price
                    exit_reason = "TP"
            elif hit_sl:
                exit_price = t.sl_price
                exit_reason = "TRAIL_SL" if t.trailing_active else "SL"
            elif hit_tp:
                exit_price = t.tp_price
                exit_reason = "TP"

            if exit_price > 0:
                fees = (t.size * t.entry_price * cfg.entry_fee_pct +
                        t.size * exit_price * cfg.exit_fee_pct)
                pnl_usd = t.size * (exit_price - t.entry_price) - fees
                pnl_pct_raw = (exit_price - t.entry_price) / t.entry_price

                t.exit_bar = i
                t.exit_price = exit_price
                t.exit_ts = c.timestamp
                t.exit_reason = exit_reason
                t.pnl_usd = pnl_usd
                t.pnl_pct = pnl_pct_raw - cfg.entry_fee_pct - cfg.exit_fee_pct
                t.fees = fees
                t.duration_min = (c.timestamp - t.entry_ts) / 60_000

                balance += pnl_usd
                trades.append(t)
                open_trade = None
                last_trade_bar = i

                if pnl_usd < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= cfg.max_consecutive_losses:
                        cooldown_until = i + cfg.cooldown_bars_after_tilt
                else:
                    consecutive_losses = 0

            continue

        # ── Anti-tilt cooldown ──
        if i < cooldown_until:
            continue

        # ── Cooldown entre trades ──
        if i - last_trade_bar < cfg.cooldown_bars:
            continue

        # ── Session filter ──
        if cfg.session_filter_enabled:
            hour = _bar_hour_utc(c)
            if not _in_session(hour, cfg.sessions):
                continue

        # ── ATR valide ──
        if atr[i] <= 0:
            continue
        atr_pct = atr[i] / price
        if atr_pct < cfg.min_atr_pct:
            continue

        # ── ATR en expansion ──
        past_atr_idx = i - cfg.atr_expansion_lookback
        if past_atr_idx < 0 or atr[past_atr_idx] <= 0:
            continue
        if atr[i] < atr[past_atr_idx] * cfg.atr_expansion_ratio:
            continue

        # ── Volume spike ──
        if vol_ma[i] <= 0:
            continue
        if volumes[i] < vol_ma[i] * cfg.volume_spike_mult:
            continue

        # ── BREAKOUT : prix casse le high recent ──
        if recent_high[i] <= 0:
            continue
        breakout_level = recent_high[i] * (1 + cfg.breakout_margin_pct)
        if highs[i] < breakout_level:
            continue

        # Breakout confirme ! Entry au close de la bougie
        entry_price = price
        current_atr = atr[i]

        sl_price = entry_price - cfg.sl_atr_mult * current_atr
        tp_price = entry_price + cfg.tp_atr_mult * current_atr
        sl_dist = entry_price - sl_price

        if sl_dist <= 0:
            continue

        # Sizing base sur le risque
        risk_amount = balance * cfg.risk_pct
        size = risk_amount / sl_dist
        position_value = size * entry_price
        max_value = balance * cfg.max_position_pct
        if position_value > max_value:
            size = max_value / entry_price

        if size * entry_price < 1.0:
            continue

        open_trade = BreakoutTrade(
            symbol=symbol,
            entry_bar=i,
            entry_price=entry_price,
            entry_ts=c.timestamp,
            sl_price=sl_price,
            tp_price=tp_price,
            size=size,
            atr_at_entry=current_atr,
            high_broken=recent_high[i],
            max_price_seen=highs[i],
        )

    # Cloture forcee
    if open_trade is not None and n > 0:
        t = open_trade
        last = candles[-1]
        fees = (t.size * t.entry_price * cfg.entry_fee_pct +
                t.size * last.close * cfg.exit_fee_pct)
        pnl_usd = t.size * (last.close - t.entry_price) - fees

        t.exit_bar = n - 1
        t.exit_price = last.close
        t.exit_ts = last.timestamp
        t.exit_reason = "END_OF_DATA"
        t.pnl_usd = pnl_usd
        t.pnl_pct = (last.close - t.entry_price) / t.entry_price - cfg.entry_fee_pct - cfg.exit_fee_pct
        t.fees = fees
        t.duration_min = (last.timestamp - t.entry_ts) / 60_000
        balance += pnl_usd
        trades.append(t)

    return trades, equity_curve, balance


# ── Multi-paire ────────────────────────────────────────────────────────────────


def run_multipair(
    pairs: list[str],
    start: datetime,
    end: datetime,
    cfg: BreakoutConfig,
    initial_balance: float,
) -> tuple[list[BreakoutTrade], list[EquityPoint], float]:
    """Multi-paire avec capital partage."""
    all_candles: dict[str, list[Candle]] = {}
    for pair in pairs:
        logger.warning("Downloading %s 15m...", pair)
        candles = download_candles(pair, start, end, interval="15m")
        if candles:
            all_candles[pair] = candles
            logger.warning("  %s: %d candles", pair, len(candles))
        else:
            logger.warning("  %s: NO DATA", pair)

    if not all_candles:
        return [], [], initial_balance

    per_pair_capital = initial_balance / len(all_candles)
    all_trades: list[BreakoutTrade] = []
    total_final = 0.0
    ts_equity: dict[int, float] = defaultdict(float)

    for pair, candles in all_candles.items():
        trades, eq, final = run_pair(pair, candles, cfg, per_pair_capital)
        all_trades.extend(trades)
        total_final += final
        for pt in eq:
            ts_equity[pt.ts] += pt.equity

    combined_eq = [
        EquityPoint(ts=ts, equity=eq)
        for ts, eq in sorted(ts_equity.items())
    ]
    all_trades.sort(key=lambda t: t.entry_ts)
    return all_trades, combined_eq, total_final


# ── Metriques ──────────────────────────────────────────────────────────────────


def compute_metrics(
    trades: list[BreakoutTrade],
    equity_curve: list[EquityPoint],
    initial_balance: float,
    final_equity: float,
    start: datetime,
    end: datetime,
) -> dict:
    days = max((end - start).days, 1)
    years = days / 365.25

    total_return = (final_equity - initial_balance) / initial_balance
    cagr = ((final_equity / initial_balance) ** (1 / max(years, 0.01)) - 1
            if final_equity > initial_balance * 0.01 else -1)

    peak = initial_balance
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    n = len(trades)
    if n == 0:
        return {
            "total_return": 0, "cagr": 0, "max_dd": 0, "n_trades": 0,
            "win_rate": 0, "profit_factor": 0, "avg_pnl": 0, "avg_pnl_pct": 0,
            "avg_duration_min": 0, "trades_per_day": 0, "daily_pnl_avg": 0,
            "final_equity": final_equity, "years": years,
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    win_rate = len(wins) / n
    gross_profit = sum(t.pnl_usd for t in wins) or 0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
    profit_factor = gross_profit / gross_loss
    avg_pnl = sum(t.pnl_usd for t in trades) / n
    avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
    avg_duration = sum(t.duration_min for t in trades) / n
    total_fees = sum(t.fees for t in trades)
    total_pnl = sum(t.pnl_usd for t in trades)
    trades_per_day = n / days
    daily_pnl_avg = total_pnl / days

    # Par sortie
    by_exit: dict[str, dict] = {}
    for t in trades:
        r = t.exit_reason
        if r not in by_exit:
            by_exit[r] = {"n": 0, "pnl": 0.0, "wins": 0}
        by_exit[r]["n"] += 1
        by_exit[r]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            by_exit[r]["wins"] += 1

    # Par paire
    by_pair: dict[str, dict] = {}
    for t in trades:
        p = t.symbol
        if p not in by_pair:
            by_pair[p] = {"n": 0, "pnl": 0.0, "wins": 0}
        by_pair[p]["n"] += 1
        by_pair[p]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            by_pair[p]["wins"] += 1

    max_consec_losses = 0
    current_streak = 0
    for t in trades:
        if t.pnl_usd < 0:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0

    trailing_count = sum(1 for t in trades if t.trailing_active)
    avg_atr_entry = sum(t.atr_at_entry for t in trades) / n

    best = max(trades, key=lambda t: t.pnl_usd)
    worst = min(trades, key=lambda t: t.pnl_usd)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_duration_min": avg_duration,
        "trades_per_day": trades_per_day,
        "daily_pnl_avg": daily_pnl_avg,
        "total_fees": total_fees,
        "total_pnl": total_pnl,
        "final_equity": final_equity,
        "years": years,
        "by_exit": by_exit,
        "by_pair": by_pair,
        "max_consec_losses": max_consec_losses,
        "trailing_count": trailing_count,
        "avg_atr_entry": avg_atr_entry,
        "best_trade": best,
        "worst_trade": worst,
    }


# ── Rapport ────────────────────────────────────────────────────────────────────


def print_report(m: dict, cfg: BreakoutConfig, initial_balance: float) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  BACKTEST BREAKOUT MOMENTUM 15m — {cfg.name}")
    print(f"  Capital initial : ${initial_balance:,.0f}")
    print(f"  Breakout high({cfg.lookback}) | ATR({cfg.atr_period}) expansion x{cfg.atr_expansion_ratio}")
    print(f"  TP: {cfg.tp_atr_mult}*ATR | SL: {cfg.sl_atr_mult}*ATR "
          f"| Trail: activate {cfg.trailing_activation_atr}*ATR, dist {cfg.trailing_distance_atr}*ATR")
    print(f"  Volume: >{cfg.volume_spike_mult}x SMA({cfg.volume_ma_period})")
    print(f"  Fees: {cfg.entry_fee_pct*100:.2f}% + {cfg.exit_fee_pct*100:.2f}%")
    print(f"  Session filter: {'ON' if cfg.session_filter_enabled else 'OFF'}")
    print(sep)

    print(f"\n  RESULTATS GLOBAUX")
    print("  " + "-" * 66)
    print(f"  Capital final      : ${m['final_equity']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_dd']:.1%}")
    print(f"  Trades             : {m['n_trades']}")
    wr = m['win_rate']
    print(f"  Win Rate           : {wr:.1%} ({int(wr*m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  PnL moyen          : ${m['avg_pnl']:+.2f} ({m['avg_pnl_pct']:+.3%})")
    print(f"  Duree moy. trade   : {m['avg_duration_min']:.0f} min")
    print(f"  Trades / jour      : {m['trades_per_day']:.1f}")
    print(f"  PnL / jour moyen   : ${m['daily_pnl_avg']:+.2f}")
    print(f"  Total fees         : ${m['total_fees']:+.2f}")
    print(f"  Max pertes consec. : {m['max_consec_losses']}")
    print(f"  Trailing actives   : {m['trailing_count']}")
    print(f"  ATR moy. entree    : ${m.get('avg_atr_entry', 0):.2f}")

    if m.get("best_trade"):
        b = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.3%}) {b.symbol}")
    if m.get("worst_trade"):
        w = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.3%}) {w.symbol}")

    # Objectif 50/jour
    print(f"\n  OBJECTIF 50 EUR/JOUR")
    print("  " + "-" * 66)
    if m["daily_pnl_avg"] > 0:
        capital_needed = 50.0 / m["daily_pnl_avg"] * initial_balance
        print(f"  PnL/jour actuel    : ${m['daily_pnl_avg']:+.2f}")
        print(f"  Capital pour 50/j  : ${capital_needed:,.0f}")
    else:
        print(f"  PnL/jour actuel    : ${m['daily_pnl_avg']:+.2f} (NEGATIF)")
        print(f"  Strategie non rentable avec ces parametres.")

    # Par paire
    if m.get("by_pair"):
        print(f"\n  PAR PAIRE")
        print("  " + "-" * 66)
        for pair, s in sorted(m["by_pair"].items(), key=lambda x: -x[1]["pnl"]):
            wr2 = s["wins"] / s["n"] * 100 if s["n"] else 0
            print(f"  {pair:12s} : {s['n']:4d} trades | WR {wr2:5.1f}% | PnL ${s['pnl']:+8.2f}")

    # Par sortie
    if m.get("by_exit"):
        print(f"\n  PAR SORTIE")
        print("  " + "-" * 66)
        for reason, s in sorted(m["by_exit"].items(), key=lambda x: -x[1]["n"]):
            wr2 = s["wins"] / s["n"] * 100 if s["n"] else 0
            print(f"  {reason:14s} : {s['n']:4d} trades | WR {wr2:5.1f}% | PnL ${s['pnl']:+8.2f}")

    print(f"\n{sep}\n")


def generate_charts(
    equity_curve: list[EquityPoint],
    trades: list[BreakoutTrade],
    metrics: dict,
    cfg: BreakoutConfig,
    initial_balance: float,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Breakout Momentum 15m — {cfg.name}\n"
        f"Capital: ${initial_balance:,.0f} | "
        f"TP:{cfg.tp_atr_mult}*ATR SL:{cfg.sl_atr_mult}*ATR | "
        f"Breakout high({cfg.lookback})",
        fontsize=14,
    )

    # 1. Equity
    ax1 = axes[0, 0]
    if equity_curve:
        dates = [datetime.fromtimestamp(e.ts / 1000, tz=timezone.utc) for e in equity_curve]
        equities = [e.equity for e in equity_curve]
        ax1.plot(dates, equities, color="blue", linewidth=0.8)
        ax1.axhline(initial_balance, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("$")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax1.grid(True, alpha=0.3)

    # 2. Distribution PnL
    ax2 = axes[0, 1]
    if trades:
        pnls = [t.pnl_usd for t in trades]
        nbins = min(50, max(10, len(trades) // 5))
        ax2.hist(pnls, bins=nbins, color="steelblue", edgecolor="black", alpha=0.7)
        ax2.axvline(0, color="red", linestyle="--")
        ax2.set_title(f"Distribution PnL (n={len(trades)})")
        ax2.set_xlabel("PnL ($)")
        ax2.grid(True, alpha=0.3)

    # 3. PnL cumule
    ax3 = axes[1, 0]
    if trades:
        daily_pnl: dict[str, float] = defaultdict(float)
        for t in trades:
            day = datetime.fromtimestamp(t.exit_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            daily_pnl[day] += t.pnl_usd
        sorted_days = sorted(daily_pnl.keys())
        cum = 0.0
        cumulative = []
        for d in sorted_days:
            cum += daily_pnl[d]
            cumulative.append(cum)
        day_dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_days]
        ax3.plot(day_dates, cumulative, color="green", linewidth=0.8)
        ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_title("PnL Cumule")
        ax3.set_ylabel("$")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax3.grid(True, alpha=0.3)

    # 4. Par paire
    ax4 = axes[1, 1]
    if metrics.get("by_pair"):
        pairs_sorted = sorted(metrics["by_pair"].items(), key=lambda x: -x[1]["pnl"])
        names = [p for p, _ in pairs_sorted]
        pnls = [s["pnl"] for _, s in pairs_sorted]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax4.barh(names, pnls, color=colors, alpha=0.7)
        ax4.axvline(0, color="black", linewidth=0.5)
        ax4.set_title("PnL par paire")
        ax4.set_xlabel("$")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"breakout_{cfg.name}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.warning("Chart saved: %s", chart_path)
    return chart_path


# ── Variants A/B ───────────────────────────────────────────────────────────────


def get_variants() -> list[BreakoutConfig]:
    """12 variantes breakout momentum pour A/B testing."""
    return [
        # ── Lookback variations ──────────────────────────────────────
        # B1: High(8) rapide, TP/SL serres, sessions
        BreakoutConfig(
            name="BRK_8_TIGHT",
            lookback=8,
            tp_atr_mult=1.2,
            sl_atr_mult=0.8,
            trailing_activation_atr=0.6,
            trailing_distance_atr=0.4,
            session_filter_enabled=True,
        ),
        # B2: High(12) standard, sessions
        BreakoutConfig(
            name="BRK_12_SESS",
            lookback=12,
            tp_atr_mult=1.5,
            sl_atr_mult=1.0,
            trailing_activation_atr=0.8,
            trailing_distance_atr=0.5,
            session_filter_enabled=True,
        ),
        # B3: High(20) large, 24/7
        BreakoutConfig(
            name="BRK_20_WIDE",
            lookback=20,
            tp_atr_mult=2.0,
            sl_atr_mult=1.0,
            trailing_activation_atr=1.0,
            trailing_distance_atr=0.6,
            session_filter_enabled=False,
        ),
        # B4: High(16) medium, trailing agressif
        BreakoutConfig(
            name="BRK_16_TRAIL",
            lookback=16,
            tp_atr_mult=2.5,
            sl_atr_mult=1.0,
            trailing_activation_atr=0.5,
            trailing_distance_atr=0.3,
            session_filter_enabled=False,
        ),

        # ── ATR multiplier variations ────────────────────────────────
        # B5: TP/SL symetrique ATR 1:1
        BreakoutConfig(
            name="BRK_ATR_1_1",
            lookback=12,
            tp_atr_mult=1.0,
            sl_atr_mult=1.0,
            trailing_activation_atr=0.5,
            trailing_distance_atr=0.4,
            session_filter_enabled=False,
        ),
        # B6: TP/SL asymetrique 2:1
        BreakoutConfig(
            name="BRK_ATR_2_1",
            lookback=12,
            tp_atr_mult=2.0,
            sl_atr_mult=1.0,
            trailing_activation_atr=1.0,
            trailing_distance_atr=0.5,
            session_filter_enabled=False,
        ),
        # B7: TP/SL 3:1 (gros TP, SL standard)
        BreakoutConfig(
            name="BRK_ATR_3_1",
            lookback=12,
            tp_atr_mult=3.0,
            sl_atr_mult=1.0,
            trailing_activation_atr=1.0,
            trailing_distance_atr=0.7,
            session_filter_enabled=False,
        ),

        # ── Volume & ATR expansion ───────────────────────────────────
        # B8: Volume strict 1.5x + ATR expansion 1.2x
        BreakoutConfig(
            name="BRK_VOL_STRICT",
            lookback=12,
            tp_atr_mult=1.5,
            sl_atr_mult=1.0,
            volume_spike_mult=1.5,
            atr_expansion_ratio=1.2,
            session_filter_enabled=False,
        ),
        # B9: Volume relax 0.8x + ATR expansion minimal
        BreakoutConfig(
            name="BRK_VOL_RELAX",
            lookback=12,
            tp_atr_mult=1.5,
            sl_atr_mult=1.0,
            volume_spike_mult=0.8,
            atr_expansion_ratio=1.0,
            session_filter_enabled=False,
        ),

        # ── Fees ─────────────────────────────────────────────────────
        # B10: Zero fees (Revolut maker)
        BreakoutConfig(
            name="BRK_0FEE",
            lookback=12,
            tp_atr_mult=1.5,
            sl_atr_mult=1.0,
            trailing_activation_atr=0.8,
            trailing_distance_atr=0.5,
            session_filter_enabled=False,
            entry_fee_pct=0.0,
            exit_fee_pct=0.0,
        ),
        # B11: Binance fees 0.1%
        BreakoutConfig(
            name="BRK_BINANCE",
            lookback=12,
            tp_atr_mult=1.5,
            sl_atr_mult=1.0,
            session_filter_enabled=False,
            entry_fee_pct=0.001,
            exit_fee_pct=0.001,
        ),

        # ── Trailing ultra-rapide ────────────────────────────────────
        # B12: Trailing activation immédiate (0.3*ATR), distance 0.2*ATR
        BreakoutConfig(
            name="BRK_ULTRATRAIL",
            lookback=12,
            tp_atr_mult=2.0,
            sl_atr_mult=0.8,
            trailing_activation_atr=0.3,
            trailing_distance_atr=0.2,
            session_filter_enabled=False,
        ),
    ]


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest Breakout Momentum 15m V2")
    parser.add_argument(
        "--pairs", type=str,
        default="ETH-USD,BTC-USD,SOL-USD,BNB-USD,ARB-USD",
        help="Paires separees par virgule",
    )
    parser.add_argument("--balance", type=float, default=1500.0)
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--no-session-filter", action="store_true")
    parser.add_argument("--fee", type=float, default=None)
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",")]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(args.years * 365.25))

    if args.compare:
        variants = get_variants()

        print("\n" + "=" * 110)
        print(f"  BREAKOUT MOMENTUM V2 — {len(variants)} variantes | Capital: ${args.balance:,.0f}")
        print(f"  Paires: {', '.join(pairs)}")
        print(f"  Periode: {start.date()} -> {end.date()} ({args.years:.1f} ans)")
        print("=" * 110)

        results = []
        for cfg in variants:
            print(f"\n>>> {cfg.name}: high({cfg.lookback}) TP={cfg.tp_atr_mult}*ATR "
                  f"SL={cfg.sl_atr_mult}*ATR vol>{cfg.volume_spike_mult}x "
                  f"sess={'ON' if cfg.session_filter_enabled else 'OFF'}...")
            trades, eq, final = run_multipair(pairs, start, end, cfg, args.balance)
            m = compute_metrics(trades, eq, args.balance, final, start, end)
            results.append((cfg, m, trades, eq))

            # Trailing stats
            trail_pct = m['trailing_count'] / m['n_trades'] * 100 if m['n_trades'] else 0
            print(f"    -> {m['n_trades']} trades | WR {m['win_rate']:.1%} | "
                  f"PF {m['profit_factor']:.2f} | PnL ${m['total_pnl']:+.2f} | "
                  f"DD {m['max_dd']:.1%} | trail {trail_pct:.0f}%")

        # Tableau
        print(f"\n" + "=" * 115)
        print(f"  {'Variante':16s} | {'Lkb':>3s} | {'TP':>4s} | {'SL':>4s} | "
              f"{'Trades':>6s} | {'WR':>6s} | {'PF':>5s} | "
              f"{'PnL':>10s} | {'DD':>7s} | {'PnL/j':>8s} | {'Cap 50/j':>10s}")
        print("-" * 115)
        for cfg, m, _, _ in results:
            cap50 = (50.0 / m['daily_pnl_avg'] * args.balance) if m['daily_pnl_avg'] > 0 else float('inf')
            cap_str = f"${cap50:,.0f}" if cap50 < 1e7 else "N/A"
            print(f"  {cfg.name:16s} | {cfg.lookback:3d} | "
                  f"{cfg.tp_atr_mult:4.1f} | {cfg.sl_atr_mult:4.1f} | "
                  f"{m['n_trades']:6d} | {m['win_rate']:5.1%} | "
                  f"{m['profit_factor']:5.2f} | ${m['total_pnl']:+9.2f} | "
                  f"{m['max_dd']:6.1%} | ${m['daily_pnl_avg']:+7.2f} | {cap_str:>10s}")
        print("=" * 115)

        # Best
        best_idx = max(range(len(results)), key=lambda i: results[i][1].get("total_pnl", 0))
        best_cfg, best_m, best_trades, best_eq = results[best_idx]
        print(f"\n*** Meilleure variante: {best_cfg.name} ***")
        print_report(best_m, best_cfg, args.balance)
        generate_charts(best_eq, best_trades, best_m, best_cfg, args.balance)

    else:
        cfg = BreakoutConfig(name="CUSTOM")
        if args.no_session_filter:
            cfg.session_filter_enabled = False
        if args.fee is not None:
            cfg.entry_fee_pct = args.fee
            cfg.exit_fee_pct = args.fee

        print(f"\nBacktest Breakout Momentum 15m")
        print(f"Paires: {', '.join(pairs)}")
        print(f"Periode: {start.date()} -> {end.date()} ({args.years:.1f} ans)")
        print(f"Capital: ${args.balance:,.0f}")

        trades, eq, final = run_multipair(pairs, start, end, cfg, args.balance)
        m = compute_metrics(trades, eq, args.balance, final, start, end)

        print_report(m, cfg, args.balance)
        generate_charts(eq, trades, m, cfg, args.balance)


if __name__ == "__main__":
    main()
