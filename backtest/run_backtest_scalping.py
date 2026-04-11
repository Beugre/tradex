#!/usr/bin/env python3
"""
Backtest Scalping 15m — Dual-Mode : REBOUND + TREND.

Deux modes distincts, jamais melanges :

  MODE REBOUND (mean-reversion) :
    - RSI < 30-35 (oversold)
    - Prix sous EMA9
    - EMA9 < EMA21 (trend baissiere = on joue le rebond)
    - Volume spike
    - Sortie : RSI > 50-55, TP +0.4% a +0.8%

  MODE TREND (trend-following) :
    - EMA9 > EMA21 (tendance haussiere)
    - RSI entre 40-55 (pas oversold !)
    - Pullback vers EMA9 (prix revient toucher EMA9)
    - Volume OK
    - Sortie : RSI > 65, TP +0.5% a +1.0%

  MODE DUAL :
    - Le bot detecte le regime (EMA9 vs EMA21) et applique
      le mode adapte (REBOUND si bearish, TREND si bullish)

Usage:
    python -m backtest.run_backtest_scalping --compare
    python -m backtest.run_backtest_scalping --pairs BTC-USD,ETH-USD --balance 1500
    python -m backtest.run_backtest_scalping --no-session-filter
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
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


# ── Mode ───────────────────────────────────────────────────────────────────────


class ScalpMode(str, Enum):
    REBOUND = "REBOUND"   # Mean-reversion : RSI oversold + bearish EMAs
    TREND = "TREND"       # Trend-following : RSI mid + bullish EMAs + pullback
    DUAL = "DUAL"         # Auto-switch selon le regime de marche


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class ScalpConfig:
    """Configuration de la strategie scalping dual-mode."""
    name: str = "SCALP_V1"
    mode: ScalpMode = ScalpMode.REBOUND

    # ── Indicateurs ──
    rsi_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    volume_ma_period: int = 20

    # ── MODE REBOUND (mean-reversion) ──
    rebound_rsi_max: float = 35.0      # RSI < 35 pour entrer
    rebound_rsi_exit: float = 55.0     # RSI > 55 → exit
    rebound_tp_pct: float = 0.006      # +0.6% take profit
    rebound_sl_pct: float = 0.004      # -0.4% stop loss
    rebound_price_below_ema9: bool = True  # Prix <= EMA9

    # ── MODE TREND (trend-following) ──
    trend_rsi_min: float = 40.0        # RSI >= 40
    trend_rsi_max: float = 55.0        # RSI <= 55
    trend_rsi_exit: float = 65.0       # RSI > 65 → exit
    trend_tp_pct: float = 0.008        # +0.8% take profit
    trend_sl_pct: float = 0.004        # -0.4% stop loss
    trend_pullback_pct: float = 0.003  # prix dans 0.3% de EMA9

    # ── Volume ──
    volume_spike: bool = True           # Volume > SMA(vol, 20)

    # ── Trailing / Breakeven ──
    breakeven_trigger_pct: float = 0.002   # +0.2% → SL = entry
    trailing_trigger_pct: float = 0.003    # +0.3% → trailing actif
    trailing_distance_pct: float = 0.0015  # trailing suit a -0.15%

    # ── Filtres anti-bruit ──
    min_candle_range_pct: float = 0.001  # Bougie min 0.1% de range

    # ── Anti-tilt ──
    max_consecutive_losses: int = 3    # Pause apres 3 pertes
    cooldown_bars_after_tilt: int = 8  # 8 barres 15m = 2h de pause

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
    exit_fee_pct: float = 0.0009       # 0.09% taker


# ── Trade ──────────────────────────────────────────────────────────────────────


@dataclass
class ScalpTrade:
    symbol: str
    entry_bar: int
    entry_price: float
    entry_ts: int
    sl_price: float
    tp_price: float
    size: float
    signal_mode: str = ""             # REBOUND or TREND
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_min: float = 0.0
    trailing_active: bool = False
    breakeven_applied: bool = False


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ── Indicateurs ────────────────────────────────────────────────────────────────


def ema_series(values: list[float], period: int) -> list[float]:
    """EMA classique."""
    n = len(values)
    ema = [0.0] * n
    if n == 0:
        return ema
    k = 2.0 / (period + 1)
    ema[0] = values[0]
    for i in range(1, n):
        ema[i] = values[i] * k + ema[i - 1] * (1 - k)
    return ema


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


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """RSI Wilder."""
    n = len(closes)
    rsi = [50.0] * n
    if n < period + 1:
        return rsi

    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff

    avg_gain = sum(gains[1:period + 1]) / period
    avg_loss = sum(losses[1:period + 1]) / period

    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1 + rs)

    return rsi


# ── Simulation ─────────────────────────────────────────────────────────────────


def _bar_hour_utc(candle: Candle) -> int:
    """Heure UTC de la bougie."""
    dt = datetime.fromtimestamp(candle.timestamp / 1000, tz=timezone.utc)
    return dt.hour


def _in_session(hour: int, sessions: list[tuple[int, int]]) -> bool:
    """True si l'heure est dans une des plages de trading."""
    return any(start <= hour < end for start, end in sessions)


def run_pair(
    symbol: str,
    candles: list[Candle],
    cfg: ScalpConfig,
    initial_balance: float,
) -> tuple[list[ScalpTrade], list[EquityPoint], float]:
    """Simule la strategie scalping dual-mode sur une paire."""
    n = len(candles)
    if n < max(cfg.ema_slow, cfg.rsi_period, cfg.volume_ma_period) + 5:
        return [], [], initial_balance

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    # Precalcul des indicateurs
    rsi = rsi_series(closes, cfg.rsi_period)
    ema9 = ema_series(closes, cfg.ema_fast)
    ema21 = ema_series(closes, cfg.ema_slow)
    vol_ma = sma_series(volumes, cfg.volume_ma_period)

    trades: list[ScalpTrade] = []
    equity_curve: list[EquityPoint] = []
    balance = initial_balance

    # State
    open_trade: Optional[ScalpTrade] = None
    consecutive_losses = 0
    cooldown_until = 0  # bar index until which we skip

    min_warmup = max(cfg.ema_slow, cfg.rsi_period, cfg.volume_ma_period) + 1

    for i in range(min_warmup, n):
        c = candles[i]
        price = c.close

        # Log equity periodiquement (toutes les 96 barres = 1 jour 15m)
        if i % 96 == 0:
            port_val = balance
            if open_trade is not None:
                unrealized = (price - open_trade.entry_price) / open_trade.entry_price
                port_val += open_trade.size * open_trade.entry_price * unrealized
            equity_curve.append(EquityPoint(ts=c.timestamp, equity=port_val))

        # ── Gestion position ouverte ──
        if open_trade is not None:
            t = open_trade
            gain_pct = (price - t.entry_price) / t.entry_price

            # Breakeven check
            if not t.breakeven_applied and gain_pct >= cfg.breakeven_trigger_pct:
                t.sl_price = t.entry_price
                t.breakeven_applied = True

            # Trailing check
            if gain_pct >= cfg.trailing_trigger_pct:
                t.trailing_active = True
                new_sl = price * (1 - cfg.trailing_distance_pct)
                if new_sl > t.sl_price:
                    t.sl_price = new_sl

            # RSI exit depends on signal mode
            rsi_exit_threshold = (
                cfg.rebound_rsi_exit if t.signal_mode == "REBOUND"
                else cfg.trend_rsi_exit
            )

            # Check exits using high/low of the bar for realism
            hit_sl = lows[i] <= t.sl_price
            hit_tp = highs[i] >= t.tp_price
            hit_rsi_exit = rsi[i] > rsi_exit_threshold and gain_pct > 0

            exit_price = 0.0
            exit_reason = ""

            if hit_sl and hit_tp:
                # Both hit — assume SL hit first if open < close (bullish), else TP first
                if c.open < c.close:  # bullish — probably went down first then up
                    exit_price = t.sl_price
                    exit_reason = "SL"
                else:
                    exit_price = t.tp_price
                    exit_reason = "TP"
            elif hit_sl:
                exit_price = t.sl_price
                exit_reason = "TRAILING_SL" if t.trailing_active else "SL"
            elif hit_tp:
                exit_price = t.tp_price
                exit_reason = "TP"
            elif hit_rsi_exit:
                exit_price = price
                exit_reason = "RSI_EXIT"

            if exit_price > 0:
                # Close trade
                pnl_pct_raw = (exit_price - t.entry_price) / t.entry_price
                fees = (t.size * t.entry_price * cfg.entry_fee_pct +
                        t.size * exit_price * cfg.exit_fee_pct)
                pnl_usd = t.size * (exit_price - t.entry_price) - fees

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

                # Anti-tilt tracking
                if pnl_usd < 0:
                    consecutive_losses += 1
                    if consecutive_losses >= cfg.max_consecutive_losses:
                        cooldown_until = i + cfg.cooldown_bars_after_tilt
                else:
                    consecutive_losses = 0

            continue  # Don't open new trade while processing exit bar

        # ── Anti-tilt cooldown ──
        if i < cooldown_until:
            continue

        # ── Session filter ──
        if cfg.session_filter_enabled:
            hour = _bar_hour_utc(c)
            if not _in_session(hour, cfg.sessions):
                continue

        # ── Volume filter ──
        if cfg.volume_spike and volumes[i] < vol_ma[i]:
            continue

        # ── Min candle range filter ──
        candle_range = (highs[i] - lows[i]) / lows[i] if lows[i] > 0 else 0
        if candle_range < cfg.min_candle_range_pct:
            continue

        # ═══════════════════════════════════════════════════════════════
        #  ENTRY SIGNAL — DUAL MODE
        #  REBOUND et TREND ne sont JAMAIS melanges.
        #  En mode DUAL, le bot choisit selon le regime EMA.
        # ═══════════════════════════════════════════════════════════════
        signal_mode: Optional[str] = None

        # ── REBOUND : RSI oversold + prix sous EMA9 + EMA9 < EMA21 ──
        #    On joue le REBOND dans un marche bearish.
        if cfg.mode in (ScalpMode.REBOUND, ScalpMode.DUAL):
            rebound_ok = rsi[i] < cfg.rebound_rsi_max
            if cfg.rebound_price_below_ema9:
                rebound_ok = rebound_ok and price <= ema9[i]
            # CLE : EMA9 < EMA21 → trend baissiere (coherent pour mean-reversion)
            rebound_ok = rebound_ok and ema9[i] < ema21[i]
            if rebound_ok:
                signal_mode = "REBOUND"

        # ── TREND : EMA9 > EMA21 + RSI mid + pullback vers EMA9 ──
        #    On suit la TENDANCE haussiere sur un retour vers EMA9.
        if signal_mode is None and cfg.mode in (ScalpMode.TREND, ScalpMode.DUAL):
            pullback_dist = (price - ema9[i]) / ema9[i] if ema9[i] > 0 else 999
            trend_ok = (
                ema9[i] > ema21[i]                                    # uptrend
                and cfg.trend_rsi_min <= rsi[i] <= cfg.trend_rsi_max  # RSI mid
                and pullback_dist <= cfg.trend_pullback_pct           # prix pres de EMA9
                and price >= ema9[i] * (1 - cfg.trend_pullback_pct)   # pas trop loin dessous
            )
            if trend_ok:
                signal_mode = "TREND"

        if signal_mode is None:
            continue

        # ── TP/SL selon le mode de signal ──
        if signal_mode == "REBOUND":
            tp_pct = cfg.rebound_tp_pct
            sl_pct = cfg.rebound_sl_pct
        else:
            tp_pct = cfg.trend_tp_pct
            sl_pct = cfg.trend_sl_pct

        # ── Calculer SL/TP/Size ──
        entry_price = price
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)
        sl_dist = entry_price - sl_price

        risk_amount = balance * cfg.risk_pct
        if sl_dist <= 0:
            continue
        size = risk_amount / sl_dist
        position_value = size * entry_price
        max_value = balance * cfg.max_position_pct
        if position_value > max_value:
            size = max_value / entry_price

        if size * entry_price < 1.0:  # Min $1 trade
            continue

        # ── Ouvrir le trade ──
        open_trade = ScalpTrade(
            symbol=symbol,
            entry_bar=i,
            entry_price=entry_price,
            entry_ts=c.timestamp,
            sl_price=sl_price,
            tp_price=tp_price,
            size=size,
            signal_mode=signal_mode,
        )

    # Cloture forcee si position encore ouverte
    if open_trade is not None and n > 0:
        t = open_trade
        last = candles[-1]
        exit_price = last.close
        pnl_pct_raw = (exit_price - t.entry_price) / t.entry_price
        fees = (t.size * t.entry_price * cfg.entry_fee_pct +
                t.size * exit_price * cfg.exit_fee_pct)
        pnl_usd = t.size * (exit_price - t.entry_price) - fees

        t.exit_bar = n - 1
        t.exit_price = exit_price
        t.exit_ts = last.timestamp
        t.exit_reason = "END_OF_DATA"
        t.pnl_usd = pnl_usd
        t.pnl_pct = pnl_pct_raw - cfg.entry_fee_pct - cfg.exit_fee_pct
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
    cfg: ScalpConfig,
    initial_balance: float,
) -> tuple[list[ScalpTrade], list[EquityPoint], float]:
    """Simule la strategie en multi-paire sequentiel (capital partage)."""
    # Download data
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

    # Run each pair independently with equal capital split
    per_pair_capital = initial_balance / len(all_candles)
    all_trades: list[ScalpTrade] = []
    pair_results: list[tuple[list[EquityPoint], float]] = []
    total_final = 0.0

    for pair, candles in all_candles.items():
        trades, eq, final = run_pair(pair, candles, cfg, per_pair_capital)
        all_trades.extend(trades)
        pair_results.append((eq, final))
        total_final += final

    # Build combined equity curve
    ts_equity: dict[int, float] = defaultdict(float)
    for eq_list, _ in pair_results:
        for pt in eq_list:
            ts_equity[pt.ts] += pt.equity
    combined_eq = [
        EquityPoint(ts=ts, equity=eq)
        for ts, eq in sorted(ts_equity.items())
    ]

    # Sort trades by timestamp
    all_trades.sort(key=lambda t: t.entry_ts)

    return all_trades, combined_eq, total_final


# ── Metriques ──────────────────────────────────────────────────────────────────


def compute_scalp_metrics(
    trades: list[ScalpTrade],
    equity_curve: list[EquityPoint],
    initial_balance: float,
    final_equity: float,
    start: datetime,
    end: datetime,
) -> dict:
    """Calcule les KPIs de la strategie scalping."""
    days = max((end - start).days, 1)
    years = days / 365.25

    total_return = (final_equity - initial_balance) / initial_balance
    cagr = (final_equity / initial_balance) ** (1 / max(years, 0.01)) - 1 if final_equity > initial_balance * 0.01 else -1

    # Drawdown
    peak = initial_balance
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Trade stats
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

    # Par motif de sortie
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

    # Par mode de signal
    by_mode: dict[str, dict] = {}
    for t in trades:
        m_key = t.signal_mode or "UNKNOWN"
        if m_key not in by_mode:
            by_mode[m_key] = {"n": 0, "pnl": 0.0, "wins": 0}
        by_mode[m_key]["n"] += 1
        by_mode[m_key]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            by_mode[m_key]["wins"] += 1

    # Consecutive losses max
    max_consec_losses = 0
    current_streak = 0
    for t in trades:
        if t.pnl_usd < 0:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0

    # Breakeven & trailing stats
    breakeven_count = sum(1 for t in trades if t.breakeven_applied)
    trailing_count = sum(1 for t in trades if t.trailing_active)

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
        "by_mode": by_mode,
        "max_consec_losses": max_consec_losses,
        "breakeven_count": breakeven_count,
        "trailing_count": trailing_count,
        "best_trade": best,
        "worst_trade": worst,
    }


# ── Rapport ────────────────────────────────────────────────────────────────────


def print_report(m: dict, cfg: ScalpConfig, initial_balance: float) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  BACKTEST SCALPING 15m — {cfg.name} [{cfg.mode.value}]")
    print(f"  Capital initial : ${initial_balance:,.0f}")
    if cfg.mode in (ScalpMode.REBOUND, ScalpMode.DUAL):
        print(f"  REBOUND: RSI<{cfg.rebound_rsi_max} | TP +{cfg.rebound_tp_pct*100:.1f}%"
              f" | SL -{cfg.rebound_sl_pct*100:.1f}% | RSI exit>{cfg.rebound_rsi_exit}")
    if cfg.mode in (ScalpMode.TREND, ScalpMode.DUAL):
        print(f"  TREND: RSI {cfg.trend_rsi_min}-{cfg.trend_rsi_max} | TP +{cfg.trend_tp_pct*100:.1f}%"
              f" | SL -{cfg.trend_sl_pct*100:.1f}% | RSI exit>{cfg.trend_rsi_exit}"
              f" | PB {cfg.trend_pullback_pct*100:.1f}%")
    print(f"  Breakeven: +{cfg.breakeven_trigger_pct*100:.1f}% | Trail: +{cfg.trailing_trigger_pct*100:.1f}%")
    print(f"  Fees: {cfg.entry_fee_pct*100:.2f}% + {cfg.exit_fee_pct*100:.2f}%")
    print(f"  Session filter: {'ON' if cfg.session_filter_enabled else 'OFF'}")
    print(sep)

    print(f"\n  RESULTATS GLOBAUX")
    print("  " + "-" * 66)
    print(f"  Capital final      : ${m['final_equity']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_dd']:.1%}")
    print(f"  Trades             : {m['n_trades']}")
    print(f"  Win Rate           : {m['win_rate']:.1%} ({int(m['win_rate']*m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  PnL moyen          : ${m['avg_pnl']:+.2f} ({m['avg_pnl_pct']:+.3%})")
    print(f"  Duree moy. trade   : {m['avg_duration_min']:.0f} min")
    print(f"  Trades / jour      : {m['trades_per_day']:.1f}")
    print(f"  PnL / jour moyen   : ${m['daily_pnl_avg']:+.2f}")
    print(f"  Total fees         : ${m['total_fees']:+.2f}")
    print(f"  Max pertes consec. : {m['max_consec_losses']}")
    print(f"  Breakeven actives  : {m['breakeven_count']}")
    print(f"  Trailing actives   : {m['trailing_count']}")

    if m.get("best_trade"):
        b = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.3%}) {b.symbol} [{b.signal_mode}]")
    if m.get("worst_trade"):
        w = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.3%}) {w.symbol} [{w.signal_mode}]")

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

    # Par mode
    if m.get("by_mode"):
        print(f"\n  PAR MODE")
        print("  " + "-" * 66)
        for mode_name, s in sorted(m["by_mode"].items(), key=lambda x: -x[1]["n"]):
            wr = s["wins"] / s["n"] * 100 if s["n"] else 0
            print(f"  {mode_name:14s} : {s['n']:4d} trades | WR {wr:5.1f}% | PnL ${s['pnl']:+8.2f}")

    # Par paire
    if m.get("by_pair"):
        print(f"\n  PAR PAIRE")
        print("  " + "-" * 66)
        for pair, s in sorted(m["by_pair"].items(), key=lambda x: -x[1]["pnl"]):
            wr = s["wins"] / s["n"] * 100 if s["n"] else 0
            print(f"  {pair:12s} : {s['n']:4d} trades | WR {wr:5.1f}% | PnL ${s['pnl']:+8.2f}")

    # Par motif de sortie
    if m.get("by_exit"):
        print(f"\n  PAR SORTIE")
        print("  " + "-" * 66)
        for reason, s in sorted(m["by_exit"].items(), key=lambda x: -x[1]["n"]):
            wr = s["wins"] / s["n"] * 100 if s["n"] else 0
            print(f"  {reason:14s} : {s['n']:4d} trades | WR {wr:5.1f}% | PnL ${s['pnl']:+8.2f}")

    print(f"\n{sep}\n")


def generate_charts(
    equity_curve: list[EquityPoint],
    trades: list[ScalpTrade],
    metrics: dict,
    cfg: ScalpConfig,
    initial_balance: float,
) -> Path:
    """Genere les graphiques du backtest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"Scalping 15m Backtest — {cfg.name} [{cfg.mode.value}]\n"
        f"Capital: ${initial_balance:,.0f} | "
        f"Sessions: {'ON' if cfg.session_filter_enabled else 'OFF'}",
        fontsize=14,
    )

    # 1. Equity curve
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
        ax2.hist(pnls, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax2.axvline(0, color="red", linestyle="--")
        ax2.set_title(f"Distribution PnL (n={len(trades)})")
        ax2.set_xlabel("PnL ($)")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3)

    # 3. PnL cumule par jour
    ax3 = axes[1, 0]
    if trades:
        daily_pnl: dict[str, float] = defaultdict(float)
        for t in trades:
            day = datetime.fromtimestamp(t.exit_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            daily_pnl[day] += t.pnl_usd
        sorted_days = sorted(daily_pnl.keys())
        cumulative = []
        cum = 0
        for d in sorted_days:
            cum += daily_pnl[d]
            cumulative.append(cum)
        day_dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted_days]
        ax3.plot(day_dates, cumulative, color="green", linewidth=0.8)
        ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax3.set_title("PnL Cumule journalier")
        ax3.set_ylabel("$")
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax3.grid(True, alpha=0.3)

    # 4. Trades par paire
    ax4 = axes[1, 1]
    if metrics.get("by_pair"):
        pairs_sorted = sorted(metrics["by_pair"].items(), key=lambda x: -x[1]["pnl"])
        pair_names = [p for p, _ in pairs_sorted]
        pair_pnls = [s["pnl"] for _, s in pairs_sorted]
        colors = ["green" if p > 0 else "red" for p in pair_pnls]
        ax4.barh(pair_names, pair_pnls, color=colors, alpha=0.7)
        ax4.axvline(0, color="black", linewidth=0.5)
        ax4.set_title("PnL par paire")
        ax4.set_xlabel("$")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / f"scalping_{cfg.name}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.warning("Chart saved: %s", chart_path)
    return chart_path


# ── Variants A/B ───────────────────────────────────────────────────────────────


def get_variants() -> list[ScalpConfig]:
    """Retourne les variantes dual-mode pour comparaison A/B."""
    return [
        # ── REBOUND (mean-reversion) ──────────────────────────────────
        # R1: Rebound strict RSI<30, prix sous EMA9, 24/7
        ScalpConfig(
            name="REB_30_STRICT",
            mode=ScalpMode.REBOUND,
            rebound_rsi_max=30.0,
            rebound_rsi_exit=50.0,
            rebound_tp_pct=0.006,
            rebound_sl_pct=0.004,
            rebound_price_below_ema9=True,
            session_filter_enabled=False,
        ),
        # R2: Rebound RSI<35, sessions
        ScalpConfig(
            name="REB_35_SESS",
            mode=ScalpMode.REBOUND,
            rebound_rsi_max=35.0,
            rebound_rsi_exit=55.0,
            rebound_tp_pct=0.006,
            rebound_sl_pct=0.004,
            rebound_price_below_ema9=True,
            session_filter_enabled=True,
        ),
        # R3: Rebound RSI<35, big TP 1%, SL 0.5%, 24/7
        ScalpConfig(
            name="REB_BIG_TP",
            mode=ScalpMode.REBOUND,
            rebound_rsi_max=35.0,
            rebound_rsi_exit=55.0,
            rebound_tp_pct=0.010,
            rebound_sl_pct=0.005,
            rebound_price_below_ema9=True,
            session_filter_enabled=False,
        ),
        # R4: Rebound RSI<40 relaxe, prix sous EMA9 pas exige
        ScalpConfig(
            name="REB_40_RELAX",
            mode=ScalpMode.REBOUND,
            rebound_rsi_max=40.0,
            rebound_rsi_exit=55.0,
            rebound_tp_pct=0.006,
            rebound_sl_pct=0.004,
            rebound_price_below_ema9=False,
            session_filter_enabled=False,
        ),

        # ── TREND (trend-following) ──────────────────────────────────
        # T1: Trend default, pullback 0.3%, sessions
        ScalpConfig(
            name="TREND_DEFAULT",
            mode=ScalpMode.TREND,
            trend_rsi_min=40.0,
            trend_rsi_max=55.0,
            trend_rsi_exit=65.0,
            trend_tp_pct=0.008,
            trend_sl_pct=0.004,
            trend_pullback_pct=0.003,
            session_filter_enabled=True,
        ),
        # T2: Trend wide RSI 35-60, pullback 0.5%, 24/7
        ScalpConfig(
            name="TREND_WIDE",
            mode=ScalpMode.TREND,
            trend_rsi_min=35.0,
            trend_rsi_max=60.0,
            trend_rsi_exit=70.0,
            trend_tp_pct=0.008,
            trend_sl_pct=0.004,
            trend_pullback_pct=0.005,
            session_filter_enabled=False,
        ),
        # T3: Trend agressif big TP 1.2%, 24/7
        ScalpConfig(
            name="TREND_AGGR",
            mode=ScalpMode.TREND,
            trend_rsi_min=40.0,
            trend_rsi_max=55.0,
            trend_rsi_exit=68.0,
            trend_tp_pct=0.012,
            trend_sl_pct=0.005,
            trend_pullback_pct=0.004,
            session_filter_enabled=False,
        ),
        # T4: Trend tight, pullback 0.2%, trailing tight
        ScalpConfig(
            name="TREND_TIGHT",
            mode=ScalpMode.TREND,
            trend_rsi_min=42.0,
            trend_rsi_max=52.0,
            trend_rsi_exit=62.0,
            trend_tp_pct=0.006,
            trend_sl_pct=0.003,
            trend_pullback_pct=0.002,
            breakeven_trigger_pct=0.0015,
            trailing_trigger_pct=0.002,
            trailing_distance_pct=0.001,
            session_filter_enabled=True,
        ),

        # ── DUAL (auto-switch REBOUND / TREND) ──────────────────────
        # D1: Dual default, sessions
        ScalpConfig(
            name="DUAL_SESS",
            mode=ScalpMode.DUAL,
            rebound_rsi_max=35.0,
            rebound_rsi_exit=55.0,
            rebound_tp_pct=0.006,
            rebound_sl_pct=0.004,
            trend_rsi_min=40.0,
            trend_rsi_max=55.0,
            trend_rsi_exit=65.0,
            trend_tp_pct=0.008,
            trend_sl_pct=0.004,
            trend_pullback_pct=0.003,
            session_filter_enabled=True,
        ),
        # D2: Dual 24/7
        ScalpConfig(
            name="DUAL_24H",
            mode=ScalpMode.DUAL,
            rebound_rsi_max=35.0,
            rebound_rsi_exit=55.0,
            rebound_tp_pct=0.006,
            rebound_sl_pct=0.004,
            trend_rsi_min=40.0,
            trend_rsi_max=55.0,
            trend_rsi_exit=65.0,
            trend_tp_pct=0.008,
            trend_sl_pct=0.004,
            trend_pullback_pct=0.003,
            session_filter_enabled=False,
        ),
        # D3: Dual zero fees (maker Revolut)
        ScalpConfig(
            name="DUAL_0FEE",
            mode=ScalpMode.DUAL,
            rebound_rsi_max=35.0,
            rebound_rsi_exit=55.0,
            rebound_tp_pct=0.006,
            rebound_sl_pct=0.004,
            trend_rsi_min=40.0,
            trend_rsi_max=55.0,
            trend_rsi_exit=65.0,
            trend_tp_pct=0.008,
            trend_sl_pct=0.004,
            trend_pullback_pct=0.003,
            session_filter_enabled=False,
            entry_fee_pct=0.0,
            exit_fee_pct=0.0,
        ),
        # D4: Dual wide params agressif
        ScalpConfig(
            name="DUAL_WIDE",
            mode=ScalpMode.DUAL,
            rebound_rsi_max=40.0,
            rebound_rsi_exit=55.0,
            rebound_tp_pct=0.008,
            rebound_sl_pct=0.005,
            rebound_price_below_ema9=False,
            trend_rsi_min=38.0,
            trend_rsi_max=58.0,
            trend_rsi_exit=68.0,
            trend_tp_pct=0.010,
            trend_sl_pct=0.005,
            trend_pullback_pct=0.005,
            session_filter_enabled=False,
        ),
    ]


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest Scalping 15m — Dual Mode")
    parser.add_argument(
        "--pairs", type=str,
        default="ETH-USD,BTC-USD,SOL-USD,BNB-USD,ARB-USD",
        help="Paires separees par virgule",
    )
    parser.add_argument("--balance", type=float, default=1500.0, help="Capital initial ($)")
    parser.add_argument("--years", type=float, default=1.0, help="Nombre d'annees")
    parser.add_argument("--no-session-filter", action="store_true", help="Desactiver filtre horaire")
    parser.add_argument("--compare", action="store_true", help="Compare toutes les variantes")
    parser.add_argument("--fee", type=float, default=None, help="Fee entry+exit (ex: 0.001)")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["REBOUND", "TREND", "DUAL"],
        help="Mode a tester (defaut: compare toutes)",
    )
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",")]

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(args.years * 365.25))

    if args.compare:
        # Mode comparaison
        variants = get_variants()
        # Filtrer par mode si demande
        if args.mode:
            filter_mode = ScalpMode(args.mode)
            variants = [v for v in variants if v.mode == filter_mode]

        print("\n" + "=" * 100)
        print(f"  COMPARAISON A/B — {len(variants)} variantes | Capital: ${args.balance:,.0f}")
        print(f"  Paires: {', '.join(pairs)}")
        print(f"  Periode: {start.date()} -> {end.date()} ({args.years:.1f} ans)")
        print("=" * 100)

        results = []
        for cfg in variants:
            print(f"\n>>> Running variant: {cfg.name} [{cfg.mode.value}]...")
            trades, eq, final = run_multipair(pairs, start, end, cfg, args.balance)
            m = compute_scalp_metrics(trades, eq, args.balance, final, start, end)
            results.append((cfg, m, trades, eq))

            # Compact summary with mode breakdown
            mode_info = ""
            if m.get("by_mode"):
                parts = []
                for mk, ms in m["by_mode"].items():
                    mwr = ms["wins"] / ms["n"] * 100 if ms["n"] else 0
                    parts.append(f"{mk}:{ms['n']}t/{mwr:.0f}%wr")
                mode_info = " | " + " ".join(parts)

            print(f"    {cfg.name}: {m['n_trades']} trades | WR {m['win_rate']:.1%} | "
                  f"PF {m['profit_factor']:.2f} | PnL ${m['total_pnl']:+.2f} | "
                  f"DD {m['max_dd']:.1%}{mode_info}")

        # Tableau comparatif
        print(f"\n" + "=" * 105)
        print(f"  {'Variante':16s} | {'Mode':>8s} | {'Trades':>6s} | {'WR':>6s} | {'PF':>5s} | "
              f"{'PnL':>10s} | {'DD':>7s} | {'PnL/j':>8s} | {'Cap 50/j':>10s}")
        print("-" * 105)
        for cfg, m, _, _ in results:
            cap50 = (50.0 / m['daily_pnl_avg'] * args.balance) if m['daily_pnl_avg'] > 0 else float('inf')
            cap_str = f"${cap50:,.0f}" if cap50 < 1e7 else "N/A"
            print(f"  {cfg.name:16s} | {cfg.mode.value:>8s} | {m['n_trades']:6d} | {m['win_rate']:5.1%} | "
                  f"{m['profit_factor']:5.2f} | ${m['total_pnl']:+9.2f} | "
                  f"{m['max_dd']:6.1%} | ${m['daily_pnl_avg']:+7.2f} | {cap_str:>10s}")
        print("=" * 105)

        # Generate report + charts for best variant
        best_idx = max(range(len(results)), key=lambda i: results[i][1].get("total_pnl", 0))
        best_cfg, best_m, best_trades, best_eq = results[best_idx]
        print(f"\nMeilleure variante: {best_cfg.name} [{best_cfg.mode.value}]")
        print_report(best_m, best_cfg, args.balance)
        generate_charts(best_eq, best_trades, best_m, best_cfg, args.balance)

    else:
        # Mode simple
        mode = ScalpMode(args.mode) if args.mode else ScalpMode.DUAL
        cfg = ScalpConfig(name="CUSTOM", mode=mode)
        if args.no_session_filter:
            cfg.session_filter_enabled = False
        if args.fee is not None:
            cfg.entry_fee_pct = args.fee
            cfg.exit_fee_pct = args.fee

        print(f"\nBacktest Scalping 15m [{mode.value}]")
        print(f"Paires: {', '.join(pairs)}")
        print(f"Periode: {start.date()} -> {end.date()} ({args.years:.1f} ans)")
        print(f"Capital: ${args.balance:,.0f}")

        trades, eq, final = run_multipair(pairs, start, end, cfg, args.balance)
        m = compute_scalp_metrics(trades, eq, args.balance, final, start, end)

        print_report(m, cfg, args.balance)
        generate_charts(eq, trades, m, cfg, args.balance)


if __name__ == "__main__":
    main()
