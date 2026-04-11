#!/usr/bin/env python3
"""
Backtest Strategie Hybride — Scalping (Range) + Trend Following (Spot only).

Timeframes : 5m (scalping) / 15m (confirmation tendance)
Indicateurs : RSI(14), EMA50, EMA200, ATR(14), Volume

Detection du marche :
  - RANGE  : |EMA50 - EMA200| / prix < 1%  →  scalping actif
  - TREND  : EMA50 > EMA200 → haussier  →  trend following selectif

Strategie 1 — Scalping (mode RANGE, LONG only spot) :
  Entry : RSI < 35 + prix proche support(20) + volume > SMA(vol,20)
  Exit  : TP fixe ou dynamique, SL fixe ou ATR, RSI > 50 annulation

Strategie 2 — Trend Following (mode TREND, LONG only spot) :
  Entry : EMA50 > EMA200 + RSI entre 40-55 (pullback) + prix touche EMA50
  Exit  : TP = 2 * ATR, SL = 1.5 * ATR

Comparaisons A/B :
  - SL fixe vs SL ATR
  - TP fixe vs TP dynamique (ATR)
  - Sans filtre marche vs avec filtre range/tendance

Usage:
    python -m backtest.run_backtest_hybrid_scalping --compare
    python -m backtest.run_backtest_hybrid_scalping --balance 1000 --years 1
    python -m backtest.run_backtest_hybrid_scalping --minimal
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
class HybridConfig:
    """Configuration de la strategie hybride scalping + trend."""
    name: str = "HYBRID_V3"

    # ── Indicateurs ──
    rsi_period: int = 14
    ema_fast: int = 50
    ema_slow: int = 200
    atr_period: int = 14
    volume_ma_period: int = 20

    # ── Detection marche ──
    market_filter_enabled: bool = True
    range_ema_threshold: float = 0.01       # |EMA50 - EMA200| / prix < 1% → RANGE

    # ── Scalping (mode RANGE) ──
    scalp_rsi_entry: float = 30.0           # RSI seuil oversold
    scalp_rsi_exit: float = 55.0            # RSI > N → sortie si en profit
    scalp_support_lookback: int = 50        # support = low(50) — plus large = plus solide
    scalp_support_proximity_pct: float = 0.015  # prix dans 1.5% du support
    scalp_require_reversal: bool = True     # RSI doit revenir au-dessus du seuil
    scalp_require_bullish: bool = True      # bougie verte obligatoire
    scalp_require_engulfing: bool = False   # close > prev close (engulfing)
    scalp_enabled: bool = True

    # ── V3 scalp mode ──
    scalp_mode: str = "REVERSAL"            # REVERSAL | MOMENTUM | BREAKOUT
    # REVERSAL : RSI < seuil puis remonte (mean-reversion classique)
    # MOMENTUM : RSI croise au-dessus de scalp_momentum_rsi (momentum)
    # BREAKOUT : Prix casse le rolling high + volume spike
    scalp_momentum_rsi: float = 50.0        # seuil RSI pour mode MOMENTUM
    scalp_breakout_lookback: int = 20       # rolling high pour mode BREAKOUT
    scalp_atr_expansion_ratio: float = 1.2  # ATR courant > N × ATR moy pour BREAKOUT

    # ── V3 trend filter ──
    trend_filter_ema200: bool = False        # prix > EMA200 obligatoire pour entrer
    # (filtre directionnel : ne pas acheter les dips en bear market)

    # ── V4 regime filter ──
    exclude_trend_up: bool = False           # exclure les entrees en regime TREND_UP
    exclude_trend_down: bool = False         # exclure les entrees en regime TREND_DOWN

    # ── Trend Following ──
    trend_rsi_low: float = 40.0
    trend_rsi_high: float = 55.0
    trend_ema_touch_pct: float = 0.005      # prix dans 0.5% de EMA50
    trend_require_ema_slope: bool = True    # EMA50 doit monter
    trend_ema_slope_lookback: int = 10      # pente calculee sur N barres
    trend_require_bullish: bool = True      # bougie verte obligatoire
    trend_require_above_ema200: bool = False # prix > EMA200 (filtre MTF)
    trend_enabled: bool = True

    # ── TP / SL ──
    tp_fixed_pct: float = 0.025
    sl_fixed_pct: float = 0.015
    tp_atr_mult: float = 2.0               # V3: R:R 2:1 (plus atteignable)
    sl_atr_mult: float = 1.0
    use_atr_tp: bool = True
    use_atr_sl: bool = True

    # ── Trailing stop (V3: desactive par defaut — prouve destructeur) ──
    trailing_enabled: bool = False
    trailing_activation_atr: float = 1.5
    trailing_distance_atr: float = 0.8
    breakeven_enabled: bool = False
    breakeven_trigger_atr: float = 1.0

    # ── Risk Management ──
    risk_pct: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous: int = 2
    daily_loss_limit_pct: float = 0.05

    # ── Anti-stagnation ──
    cooldown_bars: int = 6
    max_bars_in_trade: int = 200

    # ── Fees ──
    entry_fee_pct: float = 0.0              # Maker 0% (Revolut X)
    exit_fee_pct: float = 0.0009            # Taker 0.09% (Revolut X)

    # ── Filtres ──
    min_atr_pct: float = 0.001
    min_candle_range_pct: float = 0.0005
    volume_mult: float = 1.2


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class HybridTrade:
    symbol: str
    strategy: str                   # "SCALP" ou "TREND"
    market_regime: str              # "RANGE" ou "TREND_UP"
    entry_bar: int
    entry_price: float
    entry_ts: int
    sl_price: float
    tp_price: float
    size: float
    rsi_at_entry: float = 0.0
    atr_at_entry: float = 0.0
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    duration_bars: int = 0
    # Trailing state
    trailing_active: bool = False
    max_price_seen: float = 0.0
    breakeven_hit: bool = False


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ── Indicateurs ────────────────────────────────────────────────────────────────


def ema_series(values: list[float], period: int) -> list[float]:
    """EMA standard."""
    n = len(values)
    result = [0.0] * n
    if n == 0:
        return result
    k = 2.0 / (period + 1)
    result[0] = values[0]
    for i in range(1, n):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def sma_series(values: list[float], period: int) -> list[float]:
    """SMA classique."""
    n = len(values)
    sma = [0.0] * n
    for i in range(n):
        if i < period - 1:
            sma[i] = sum(values[: i + 1]) / (i + 1)
        else:
            sma[i] = sum(values[i - period + 1 : i + 1]) / period
    return sma


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """RSI via EMA des gains / pertes."""
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

    avg_gain = sum(gains[1 : period + 1]) / period
    avg_loss = sum(losses[1 : period + 1]) / period

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

    if n >= period:
        atr[period - 1] = sum(tr[:period]) / period
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    else:
        atr[-1] = sum(tr) / n

    return atr


def rolling_low(lows: list[float], period: int) -> list[float]:
    """Low glissant sur N barres (inclut la barre courante)."""
    n = len(lows)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        result[i] = min(lows[start : i + 1])
    return result


def rolling_high(highs: list[float], period: int) -> list[float]:
    """High glissant sur N barres (inclut la barre courante)."""
    n = len(highs)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        result[i] = max(highs[start : i + 1])
    return result


# ── Simulation ─────────────────────────────────────────────────────────────────


def _detect_regime(ema50_val: float, ema200_val: float, price: float,
                   threshold: float) -> str:
    """Detecte le regime de marche."""
    spread = abs(ema50_val - ema200_val) / price if price > 0 else 0
    if spread < threshold:
        return "RANGE"
    elif ema50_val > ema200_val:
        return "TREND_UP"
    else:
        return "TREND_DOWN"


def run_pair(
    symbol: str,
    candles: list[Candle],
    cfg: HybridConfig,
    initial_balance: float,
) -> tuple[list[HybridTrade], list[EquityPoint], float]:
    """Simule la strategie hybride sur une paire."""
    n = len(candles)
    warmup = max(cfg.ema_slow, cfg.atr_period, cfg.volume_ma_period,
                 cfg.scalp_support_lookback) + 10
    if n < warmup + 20:
        return [], [], initial_balance

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]

    # Precalcul indicateurs
    rsi = rsi_series(closes, cfg.rsi_period)
    ema50 = ema_series(closes, cfg.ema_fast)
    ema200 = ema_series(closes, cfg.ema_slow)
    atr = atr_series(highs, lows, closes, cfg.atr_period)
    vol_ma = sma_series(volumes, cfg.volume_ma_period)
    support = rolling_low(lows, cfg.scalp_support_lookback)

    # V3 : rolling high pour mode BREAKOUT
    resistance = rolling_high(highs, cfg.scalp_breakout_lookback)
    # V3 : ATR moyenne glissante pour expansion check
    atr_ma = sma_series(atr, cfg.scalp_breakout_lookback)

    trades: list[HybridTrade] = []
    equity_curve: list[EquityPoint] = []
    balance = initial_balance
    daily_start_balance = initial_balance

    open_positions: list[HybridTrade] = []
    last_trade_bar: dict[str, int] = {}      # par paire / strategie
    daily_stopped = False
    current_day = ""

    # Precalcul : RSI etait sous le seuil N barres avant (reversal detection)
    rsi_was_oversold = [False] * n
    for j in range(1, n):
        if rsi[j - 1] < cfg.scalp_rsi_entry:
            rsi_was_oversold[j] = True
        # Propagation : si RSI < seuil dans les 3 dernieres barres
        if j >= 2 and rsi[j - 2] < cfg.scalp_rsi_entry:
            rsi_was_oversold[j] = True
        if j >= 3 and rsi[j - 3] < cfg.scalp_rsi_entry:
            rsi_was_oversold[j] = True

    for i in range(warmup, n):
        c = candles[i]
        price = c.close
        is_bullish = c.close > c.open

        # Jour courant (pour daily stop loss)
        dt = datetime.fromtimestamp(c.timestamp / 1000, tz=timezone.utc)
        day_str = dt.strftime("%Y-%m-%d")
        if day_str != current_day:
            current_day = day_str
            daily_start_balance = balance
            daily_stopped = False

        # Portfolio value pour equity curve (toutes les 50 barres)
        if i % 50 == 0:
            port_val = balance
            for pos in open_positions:
                unrealized = (price - pos.entry_price) * pos.size
                port_val += unrealized
            equity_curve.append(EquityPoint(ts=c.timestamp, equity=port_val))

        # Daily stop loss check
        if daily_start_balance > 0:
            daily_change = (balance - daily_start_balance) / daily_start_balance
            if daily_change <= -cfg.daily_loss_limit_pct:
                daily_stopped = True

        # ── Gestion positions ouvertes ──
        closed_indices = []
        for idx, t in enumerate(open_positions):
            bars_held = i - t.entry_bar

            # Maj max price (pour trailing)
            if highs[i] > t.max_price_seen:
                t.max_price_seen = highs[i]

            # Breakeven : si prix a depasse entry + trigger → SL = entry
            if (cfg.breakeven_enabled and not t.breakeven_hit
                    and t.max_price_seen >= t.entry_price + cfg.breakeven_trigger_atr * t.atr_at_entry):
                t.breakeven_hit = True
                new_sl = t.entry_price + t.atr_at_entry * 0.1  # leger profit
                if new_sl > t.sl_price:
                    t.sl_price = new_sl

            # Trailing activation
            if cfg.trailing_enabled and not t.trailing_active:
                trail_act_price = t.entry_price + cfg.trailing_activation_atr * t.atr_at_entry
                if t.max_price_seen >= trail_act_price:
                    t.trailing_active = True

            # Trailing SL update
            if t.trailing_active:
                trail_sl = t.max_price_seen - cfg.trailing_distance_atr * t.atr_at_entry
                if trail_sl > t.sl_price:
                    t.sl_price = trail_sl

            # Check SL
            hit_sl = lows[i] <= t.sl_price
            # Check TP
            hit_tp = highs[i] >= t.tp_price

            exit_price = 0.0
            exit_reason = ""

            if hit_sl and hit_tp:
                # Ambiguity : si bullish candle → TP first, sinon SL
                if is_bullish:
                    exit_price = t.tp_price
                    exit_reason = "TP"
                else:
                    exit_price = t.sl_price
                    exit_reason = "TRAIL_SL" if t.trailing_active else "SL"
            elif hit_sl:
                exit_price = t.sl_price
                exit_reason = "TRAIL_SL" if t.trailing_active else "SL"
            elif hit_tp:
                exit_price = t.tp_price
                exit_reason = "TP"
            # RSI annulation (scalping only) : RSI depasse seuil haut
            elif (t.strategy == "SCALP" and rsi[i] > cfg.scalp_rsi_exit
                  and price > t.entry_price):  # seulement si en profit
                exit_price = price
                exit_reason = "RSI_EXIT"
            # Timeout
            elif bars_held >= cfg.max_bars_in_trade:
                exit_price = price
                exit_reason = "TIMEOUT"

            if exit_price > 0:
                fees = (t.size * t.entry_price * cfg.entry_fee_pct +
                        t.size * exit_price * cfg.exit_fee_pct)
                pnl_usd = t.size * (exit_price - t.entry_price) - fees

                t.exit_bar = i
                t.exit_price = exit_price
                t.exit_ts = c.timestamp
                t.exit_reason = exit_reason
                t.pnl_usd = pnl_usd
                t.pnl_pct = (exit_price - t.entry_price) / t.entry_price
                t.fees = fees
                t.duration_bars = bars_held

                balance += pnl_usd
                trades.append(t)
                closed_indices.append(idx)

        # Remove closed (reverse order)
        for idx in sorted(closed_indices, reverse=True):
            open_positions.pop(idx)

        # ── Skip si daily stop ou max positions ──
        if daily_stopped:
            continue
        if len(open_positions) >= cfg.max_simultaneous:
            continue

        # ── Cooldown ──
        cooldown_key = f"{symbol}"
        bars_since = i - last_trade_bar.get(cooldown_key, -999)
        if bars_since < cfg.cooldown_bars:
            continue

        # ── Filtres generaux ──
        if atr[i] <= 0 or price <= 0:
            continue
        atr_pct = atr[i] / price
        if atr_pct < cfg.min_atr_pct:
            continue
        if vol_ma[i] > 0 and volumes[i] < vol_ma[i] * cfg.volume_mult:
            continue
        candle_range = (highs[i] - lows[i]) / price if price > 0 else 0
        if candle_range < cfg.min_candle_range_pct:
            continue

        # ── Detection regime ──
        regime = _detect_regime(ema50[i], ema200[i], price, cfg.range_ema_threshold)

        # ── V4 regime exclusion ──
        if cfg.exclude_trend_up and regime == "TREND_UP":
            continue
        if cfg.exclude_trend_down and regime == "TREND_DOWN":
            continue

        # ── SIGNAL SCALPING ──
        scalp_signal = False
        if cfg.scalp_enabled:
            if not cfg.market_filter_enabled or regime == "RANGE":
                # V3 filtre tendance MTF : prix > EMA200 pour eviter les bear markets
                tmf_ok = (not cfg.trend_filter_ema200) or (price > ema200[i])

                if tmf_ok:
                    if cfg.scalp_mode == "REVERSAL":
                        # Mode classique : RSI oversold → reversal
                        rsi_cond = False
                        if cfg.scalp_require_reversal:
                            rsi_cond = (rsi_was_oversold[i] and rsi[i] >= cfg.scalp_rsi_entry
                                        and rsi[i] < cfg.scalp_rsi_entry + 10)
                        else:
                            rsi_cond = rsi[i] < cfg.scalp_rsi_entry

                        if rsi_cond:
                            bullish_ok = not cfg.scalp_require_bullish or is_bullish
                            engulfing_ok = (not cfg.scalp_require_engulfing
                                            or (i > 0 and c.close > candles[i - 1].close))
                            if bullish_ok and engulfing_ok:
                                if support[i] > 0:
                                    dist_support = (price - support[i]) / price
                                    if dist_support < cfg.scalp_support_proximity_pct:
                                        scalp_signal = True

                    elif cfg.scalp_mode == "MOMENTUM":
                        # V3 : RSI momentum — acheter quand RSI croise AU-DESSUS du seuil
                        # (confirmation de momentum haussier, pas mean-reversion)
                        if (i >= 2
                                and rsi[i] > cfg.scalp_momentum_rsi
                                and rsi[i - 1] <= cfg.scalp_momentum_rsi):
                            bullish_ok = not cfg.scalp_require_bullish or is_bullish
                            if bullish_ok:
                                scalp_signal = True

                    elif cfg.scalp_mode == "BREAKOUT":
                        # V3 : Breakout — close > rolling high + volume spike + ATR expansion
                        if i > 0 and resistance[i - 1] > 0:
                            prev_high = resistance[i - 1]
                            if c.close > prev_high:
                                vol_ok = vol_ma[i] > 0 and volumes[i] >= vol_ma[i] * cfg.volume_mult
                                atr_expanding = (atr_ma[i] > 0
                                                 and atr[i] > atr_ma[i] * cfg.scalp_atr_expansion_ratio)
                                if vol_ok and atr_expanding:
                                    scalp_signal = True

        # ── SIGNAL TREND FOLLOWING ──
        trend_signal = False
        if cfg.trend_enabled:
            if not cfg.market_filter_enabled or regime == "TREND_UP":
                if ema50[i] > ema200[i]:
                    # V3 : filtre prix > EMA200
                    above_ema200 = (not cfg.trend_require_above_ema200) or (price > ema200[i])
                    if above_ema200 and cfg.trend_rsi_low <= rsi[i] <= cfg.trend_rsi_high:
                        # Prix touche EMA50
                        dist_ema50 = abs(price - ema50[i]) / price
                        if dist_ema50 < cfg.trend_ema_touch_pct:
                            # V2 : EMA slope positive
                            ema_slope_ok = True
                            if cfg.trend_require_ema_slope and i >= cfg.trend_ema_slope_lookback:
                                ema_slope_ok = ema50[i] > ema50[i - cfg.trend_ema_slope_lookback]
                            # V2 : bougie bullish
                            bullish_ok = not cfg.trend_require_bullish or is_bullish
                            if ema_slope_ok and bullish_ok:
                                trend_signal = True

        if not scalp_signal and not trend_signal:
            continue

        # Priorite scalping si les deux signaux
        if scalp_signal:
            strategy = "SCALP"
            regime_used = regime
        else:
            strategy = "TREND"
            regime_used = regime

        # ── Calcul TP / SL ──
        if cfg.use_atr_sl:
            sl_dist = cfg.sl_atr_mult * atr[i]
        else:
            sl_dist = price * cfg.sl_fixed_pct
        sl_price = price - sl_dist

        if cfg.use_atr_tp:
            tp_dist = cfg.tp_atr_mult * atr[i]
        else:
            tp_dist = price * cfg.tp_fixed_pct
        tp_price = price + tp_dist

        if sl_dist <= 0:
            continue

        # ── Sizing ──
        risk_amount = balance * cfg.risk_pct
        size = risk_amount / sl_dist
        position_value = size * price
        max_value = balance * cfg.max_position_pct
        if position_value > max_value:
            size = max_value / price

        if size * price < 1.0:
            continue

        # ── Ouverture ──
        new_trade = HybridTrade(
            symbol=symbol,
            strategy=strategy,
            market_regime=regime_used,
            entry_bar=i,
            entry_price=price,
            entry_ts=c.timestamp,
            sl_price=sl_price,
            tp_price=tp_price,
            size=size,
            rsi_at_entry=rsi[i],
            atr_at_entry=atr[i],
            max_price_seen=highs[i],
        )
        open_positions.append(new_trade)
        last_trade_bar[cooldown_key] = i

    # Cloture forcee des positions ouvertes
    for t in open_positions:
        if n > 0:
            last = candles[-1]
            fees = (t.size * t.entry_price * cfg.entry_fee_pct +
                    t.size * last.close * cfg.exit_fee_pct)
            pnl_usd = t.size * (last.close - t.entry_price) - fees
            t.exit_bar = n - 1
            t.exit_price = last.close
            t.exit_ts = last.timestamp
            t.exit_reason = "END_OF_DATA"
            t.pnl_usd = pnl_usd
            t.pnl_pct = (last.close - t.entry_price) / t.entry_price
            t.fees = fees
            t.duration_bars = n - 1 - t.entry_bar
            balance += pnl_usd
            trades.append(t)

    return trades, equity_curve, balance


# ── Multi-paire ────────────────────────────────────────────────────────────────


def run_multipair(
    pairs: list[str],
    start: datetime,
    end: datetime,
    cfg: HybridConfig,
    initial_balance: float,
    interval: str = "5m",
) -> tuple[list[HybridTrade], list[EquityPoint], float]:
    """Multi-paire avec capital partage."""
    all_candles: dict[str, list[Candle]] = {}
    for pair in pairs:
        logger.warning("Downloading %s %s...", pair, interval)
        candles = download_candles(pair, start, end, interval=interval)
        if candles:
            all_candles[pair] = candles
            logger.warning("  %s: %d candles", pair, len(candles))
        else:
            logger.warning("  %s: NO DATA", pair)

    if not all_candles:
        return [], [], initial_balance

    per_pair_capital = initial_balance / len(all_candles)
    all_trades: list[HybridTrade] = []
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
    trades: list[HybridTrade],
    equity_curve: list[EquityPoint],
    initial_balance: float,
    final_equity: float,
    start: datetime,
    end: datetime,
) -> dict:
    import math

    days = max((end - start).days, 1)
    years = days / 365.25

    total_return = (final_equity - initial_balance) / initial_balance
    cagr = ((final_equity / initial_balance) ** (1 / max(years, 0.01)) - 1
            if final_equity > initial_balance * 0.01 else -1)

    # Drawdown
    peak = initial_balance
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Sharpe
    returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1].equity
        if prev > 0:
            returns.append((equity_curve[i].equity - prev) / prev)
    bars_per_year = (288 * 365.25) / 50  # 288 barres 5m/jour, sample 1/50
    if len(returns) >= 2:
        mu = sum(returns) / len(returns)
        var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var) if var > 0 else 1e-9
        sharpe = (mu / std) * math.sqrt(bars_per_year)

        neg = [r for r in returns if r < 0]
        if neg:
            down_var = sum(r ** 2 for r in neg) / len(neg)
            down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
            sortino = (mu / down_std) * math.sqrt(bars_per_year)
        else:
            sortino = 99.0
    else:
        sharpe = 0.0
        sortino = 0.0

    n = len(trades)
    if n == 0:
        return {
            "total_return": 0, "cagr": 0, "max_dd": 0, "sharpe": 0, "sortino": 0,
            "n_trades": 0, "win_rate": 0, "profit_factor": 0,
            "avg_pnl": 0, "avg_pnl_pct": 0, "avg_duration_bars": 0,
            "trades_per_day": 0, "daily_pnl_avg": 0, "total_fees": 0,
            "total_pnl": 0, "final_equity": final_equity, "years": years,
            "by_strategy": {}, "by_exit": {}, "by_pair": {}, "by_regime": {},
            "max_consec_losses": 0,
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    win_rate = len(wins) / n
    gross_profit = sum(t.pnl_usd for t in wins) or 0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
    profit_factor = gross_profit / gross_loss
    avg_pnl = sum(t.pnl_usd for t in trades) / n
    avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
    avg_duration = sum(t.duration_bars for t in trades) / n
    total_fees = sum(t.fees for t in trades)
    total_pnl = sum(t.pnl_usd for t in trades)
    trades_per_day = n / days
    daily_pnl_avg = total_pnl / days

    # Groupements
    def _group(key_fn):
        groups: dict[str, list[HybridTrade]] = defaultdict(list)
        for t in trades:
            groups[key_fn(t)].append(t)
        out = {}
        for k, tlist in sorted(groups.items()):
            cnt = len(tlist)
            w = sum(1 for t in tlist if t.pnl_usd > 0)
            pnl = sum(t.pnl_usd for t in tlist)
            gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
            gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
            out[k] = {"n": cnt, "wins": w, "wr": w / cnt, "pnl": pnl, "pf": gp / gl,
                       "avg_pct": sum(t.pnl_pct for t in tlist) / cnt}
        return out

    by_strategy = _group(lambda t: t.strategy)
    by_exit = _group(lambda t: t.exit_reason)
    by_pair = _group(lambda t: t.symbol)
    by_regime = _group(lambda t: t.market_regime)

    max_consec_losses = 0
    streak = 0
    for t in trades:
        if t.pnl_usd < 0:
            streak += 1
            max_consec_losses = max(max_consec_losses, streak)
        else:
            streak = 0

    best = max(trades, key=lambda t: t.pnl_usd)
    worst = min(trades, key=lambda t: t.pnl_usd)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl": avg_pnl,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_duration_bars": avg_duration,
        "trades_per_day": trades_per_day,
        "daily_pnl_avg": daily_pnl_avg,
        "total_fees": total_fees,
        "total_pnl": total_pnl,
        "final_equity": final_equity,
        "years": years,
        "by_strategy": by_strategy,
        "by_exit": by_exit,
        "by_pair": by_pair,
        "by_regime": by_regime,
        "max_consec_losses": max_consec_losses,
        "best_trade": best,
        "worst_trade": worst,
    }


# ── Rapport ────────────────────────────────────────────────────────────────────


def print_report(m: dict, cfg: HybridConfig, initial_balance: float) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  HYBRID SCALPING + TREND FOLLOWING — {cfg.name}")
    print(f"  Capital: ${initial_balance:,.0f} | "
          f"Filter: {'ON' if cfg.market_filter_enabled else 'OFF'} | "
          f"Scalp: {'ON' if cfg.scalp_enabled else 'OFF'} | "
          f"Trend: {'ON' if cfg.trend_enabled else 'OFF'}")
    print(f"  TP: {'ATR x' + str(cfg.tp_atr_mult) if cfg.use_atr_tp else str(cfg.tp_fixed_pct * 100) + '% fixe'} | "
          f"SL: {'max(fixe,ATR x' + str(cfg.sl_atr_mult) + ')' if cfg.use_atr_sl else str(cfg.sl_fixed_pct * 100) + '% fixe'}")
    print(f"  Fees: {cfg.entry_fee_pct * 100:.2f}% + {cfg.exit_fee_pct * 100:.2f}% | "
          f"Risk: {cfg.risk_pct * 100:.0f}% | Max pos: {cfg.max_simultaneous}")
    print(sep)

    print(f"\n  RESULTATS GLOBAUX")
    print("  " + "-" * 76)
    print(f"  Capital final      : ${m['final_equity']:,.2f} ({m['total_return']:+.1%})")
    print(f"  CAGR               : {m['cagr']:.1%}")
    print(f"  Max Drawdown       : {m['max_dd']:.1%}")
    print(f"  Sharpe Ratio       : {m['sharpe']:.2f}")
    print(f"  Sortino Ratio      : {m['sortino']:.2f}")
    wr = m["win_rate"]
    print(f"  Win Rate           : {wr:.1%} ({int(wr * m['n_trades'])}/{m['n_trades']})")
    print(f"  Profit Factor      : {m['profit_factor']:.2f}")
    print(f"  Trades             : {m['n_trades']}")
    print(f"  PnL moyen          : ${m['avg_pnl']:+.2f} ({m['avg_pnl_pct']:+.3%})")
    print(f"  Duree moy. trade   : {m['avg_duration_bars']:.0f} barres")
    print(f"  Trades / jour      : {m['trades_per_day']:.2f}")
    print(f"  PnL / jour moyen   : ${m['daily_pnl_avg']:+.2f}")
    print(f"  Total fees         : ${m['total_fees']:.2f}")
    print(f"  Max pertes consec. : {m['max_consec_losses']}")

    if m.get("best_trade"):
        b = m["best_trade"]
        print(f"  Meilleur trade     : ${b.pnl_usd:+.2f} ({b.pnl_pct:+.1%}) {b.symbol} [{b.strategy}]")
    if m.get("worst_trade"):
        w = m["worst_trade"]
        print(f"  Pire trade         : ${w.pnl_usd:+.2f} ({w.pnl_pct:+.1%}) {w.symbol} [{w.strategy}]")

    if m.get("by_strategy"):
        print(f"\n  PAR STRATEGIE")
        print("  " + "-" * 76)
        for strat, s in m["by_strategy"].items():
            print(f"  {strat:12s} : {s['n']:4d} trades | WR {s['wr']:.0%}"
                  f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f} | Avg {s['avg_pct']:+.3%}")

    if m.get("by_regime"):
        print(f"\n  PAR REGIME DE MARCHE")
        print("  " + "-" * 76)
        for regime, s in m["by_regime"].items():
            print(f"  {regime:12s} : {s['n']:4d} trades | WR {s['wr']:.0%}"
                  f" | PF {s['pf']:.2f} | PnL ${s['pnl']:+.2f}")

    if m.get("by_pair"):
        print(f"\n  PAR PAIRE")
        print("  " + "-" * 76)
        for pair, s in sorted(m["by_pair"].items(), key=lambda x: -x[1]["pnl"]):
            print(f"  {pair:12s} : {s['n']:4d} trades | WR {s['wr']:.0%}"
                  f" | PnL ${s['pnl']:+.2f}")

    if m.get("by_exit"):
        print(f"\n  PAR MOTIF DE SORTIE")
        print("  " + "-" * 76)
        for reason, s in sorted(m["by_exit"].items(), key=lambda x: -x[1]["n"]):
            print(f"  {reason:14s} : {s['n']:4d} trades | WR {s['wr']:.0%}"
                  f" | PnL ${s['pnl']:+.2f}")

    print(f"\n{sep}\n")


# ── Graphiques ─────────────────────────────────────────────────────────────────


def generate_charts(
    equity_curve: list[EquityPoint],
    trades: list[HybridTrade],
    metrics: dict,
    cfg: HybridConfig,
    initial_balance: float,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#05080d")
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#05080d")
    if equity_curve:
        dates = [datetime.fromtimestamp(e.ts / 1000, tz=timezone.utc) for e in equity_curve]
        equities = [e.equity for e in equity_curve]
        ax1.fill_between(dates, equities, alpha=0.10, color="#14d8c4")
        ax1.plot(dates, equities, color="#14d8c4", linewidth=2.0, label="Equity")
        ax1.axhline(initial_balance, color="#9aa7b3", linestyle=":", alpha=0.7)
    ax1.set_title(
        f"Hybrid Scalping — {cfg.name}  |  "
        f"${initial_balance:,.0f} → ${metrics['final_equity']:,.2f}  "
        f"({metrics['total_return']:+.1%})",
        fontsize=12, fontweight="bold", color="white", pad=12,
    )
    ax1.set_ylabel("Equity ($)", fontsize=9, color="white")
    ax1.legend(loc="upper left", fontsize=9, frameon=False)
    ax1.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#05080d")
    if equity_curve and len(equity_curve) > 1:
        peak = initial_balance
        dd_vals = []
        dd_dates = []
        for pt in equity_curve:
            peak = max(peak, pt.equity)
            dd = (pt.equity - peak) / peak if peak > 0 else 0
            dd_vals.append(dd * 100)
            dd_dates.append(datetime.fromtimestamp(pt.ts / 1000, tz=timezone.utc))
        ax2.fill_between(dd_dates, dd_vals, alpha=0.4, color="#ff5c7a")
        ax2.plot(dd_dates, dd_vals, color="#ff5c7a", linewidth=0.8)
    ax2.set_title("Drawdown (%)", fontsize=10, color="white")
    ax2.set_ylabel("%", fontsize=9, color="white")
    ax2.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)

    # 3. Distribution PnL
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#05080d")
    if trades:
        pnls = [t.pnl_usd for t in trades]
        nbins = min(50, max(10, len(trades) // 5))
        ax3.hist(pnls, bins=nbins, color="#14d8c4", edgecolor="#05080d", alpha=0.8)
        ax3.axvline(0, color="#ff5c7a", linestyle="--", alpha=0.7)
    ax3.set_title(f"Distribution PnL (n={metrics['n_trades']})", fontsize=10, color="white")
    ax3.set_xlabel("PnL ($)", fontsize=9, color="white")
    ax3.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)

    # 4. Par strategie
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.set_facecolor("#05080d")
    if metrics.get("by_strategy"):
        strats = list(metrics["by_strategy"].keys())
        pnls = [metrics["by_strategy"][s]["pnl"] for s in strats]
        counts = [metrics["by_strategy"][s]["n"] for s in strats]
        colors = ["#14d8c4" if p > 0 else "#ff5c7a" for p in pnls]
        bars = ax4.barh(strats, pnls, color=colors, alpha=0.8)
        for bar, cnt in zip(bars, counts):
            ax4.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                     f"  {cnt}t", va="center", fontsize=8, color="white")
    ax4.set_title("PnL par Strategie", fontsize=10, color="white")
    ax4.axvline(0, color="#9aa7b3", linewidth=0.5)
    ax4.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)

    # 5. Par paire
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor("#05080d")
    if metrics.get("by_pair"):
        pairs_sorted = sorted(metrics["by_pair"].items(), key=lambda x: -x[1]["pnl"])
        names = [p for p, _ in pairs_sorted]
        pnls = [s["pnl"] for _, s in pairs_sorted]
        colors = ["#14d8c4" if p > 0 else "#ff5c7a" for p in pnls]
        ax5.barh(names, pnls, color=colors, alpha=0.8)
        ax5.axvline(0, color="#9aa7b3", linewidth=0.5)
    ax5.set_title("PnL par paire", fontsize=10, color="white")
    ax5.grid(True, linestyle=":", linewidth=0.5, alpha=0.25)

    # KPIs en footer
    fig.text(
        0.5, 0.01,
        f"WR: {metrics['win_rate']:.0%}  |  PF: {metrics['profit_factor']:.2f}  |  "
        f"Sharpe: {metrics['sharpe']:.2f}  |  MaxDD: {metrics['max_dd']:.1%}  |  "
        f"Trades: {metrics['n_trades']}  |  PnL/j: ${metrics['daily_pnl_avg']:+.2f}",
        ha="center", fontsize=10, color="#9aa7b3",
    )

    chart_path = OUTPUT_DIR / f"hybrid_{cfg.name}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.warning("Chart saved: %s", chart_path)
    return chart_path


# ── Variantes A/B ──────────────────────────────────────────────────────────────


def get_variants() -> list[HybridConfig]:
    """Variantes V4 — optimisation de MOMENTUM_50 (seule variante rentable).

    Enseignements V3 :
      - MOMENTUM RSI>50 sur 1H = seule variante rentable (PF 1.04)
      - TREND_UP detruit la performance (PF 0.51) → filtrer
      - TREND_DOWN + RANGE portent le PnL (PF 1.44 et 1.26)
      - RSI_EXIT genere 96% WR → profit driver principal
      - Trailing toujours destructeur
      - Breakout high(12) presque breakeven (PF 0.97)
    """
    # ── Base momentum (reference V3) ──
    MOM_BASE = dict(
        scalp_mode="MOMENTUM",
        scalp_momentum_rsi=50.0,
        market_filter_enabled=False,
        trend_enabled=False,
    )

    return [
        # ── Reference : V3_MOMENTUM_50 inchangee ──
        HybridConfig(name="V3_MOM50_REF", **MOM_BASE),

        # ── V4 : Exclure TREND_UP (le regime destructeur) ──
        HybridConfig(name="V4_NO_TRENDUP", exclude_trend_up=True, **MOM_BASE),

        # ── V4 : TREND_DOWN only (meilleur regime) ──
        HybridConfig(name="V4_DOWN_ONLY", exclude_trend_up=True,
                     market_filter_enabled=True, scalp_mode="MOMENTUM",
                     scalp_momentum_rsi=50.0, trend_enabled=False),

        # ── V4 : RSI exit plus bas (50 au lieu de 55) — sortie plus rapide ──
        HybridConfig(name="V4_RSI_EXIT_50", scalp_rsi_exit=50.0,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : RSI exit plus haut (60) — laisser courir ──
        HybridConfig(name="V4_RSI_EXIT_60", scalp_rsi_exit=60.0,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : RSI entry 45 (signal plus tot) ──
        HybridConfig(name="V4_MOM45", scalp_momentum_rsi=45.0,
                     exclude_trend_up=True, scalp_mode="MOMENTUM",
                     market_filter_enabled=False, trend_enabled=False),

        # ── V4 : RSI entry 55 (signal plus strict) ──
        HybridConfig(name="V4_MOM55", scalp_momentum_rsi=55.0,
                     exclude_trend_up=True, scalp_mode="MOMENTUM",
                     market_filter_enabled=False, trend_enabled=False),

        # ── V4 : Cooldown plus long (10 barres = 10h) ──
        HybridConfig(name="V4_COOL10", cooldown_bars=10,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : 1 position max ──
        HybridConfig(name="V4_MAX1POS", max_simultaneous=1,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : R:R 3:1 ──
        HybridConfig(name="V4_RR31", tp_atr_mult=3.0,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : SL plus serre (0.7 ATR) → R:R 2.85:1 real ──
        HybridConfig(name="V4_SL_TIGHT", sl_atr_mult=0.7,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : Volume 1.5x filter ──
        HybridConfig(name="V4_VOL15", volume_mult=1.5,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : Volume OFF (1.0) ──
        HybridConfig(name="V4_VOL_OFF", volume_mult=0.0,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : Zero fees (plafond theorique) ──
        HybridConfig(name="V4_ZERO_FEE", entry_fee_pct=0.0, exit_fee_pct=0.0,
                     exclude_trend_up=True, **MOM_BASE),

        # ── V4 : Best combo hypothetique ──
        HybridConfig(name="V4_BEST",
                     exclude_trend_up=True,
                     scalp_rsi_exit=60.0,    # laisser courir
                     max_simultaneous=1,
                     cooldown_bars=8,
                     **MOM_BASE),

        # ── V4 : Breakout 12 (presque rentable en V3) ──
        HybridConfig(
            name="V4_BRK12_NOTREND",
            scalp_mode="BREAKOUT",
            scalp_breakout_lookback=12,
            scalp_atr_expansion_ratio=1.05,
            volume_mult=1.0,
            market_filter_enabled=False,
            trend_enabled=False,
            tp_atr_mult=2.0,
            sl_atr_mult=0.8,
            exclude_trend_up=True,
        ),
    ]


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Backtest Hybride Scalping + Trend Following (Spot only)",
    )
    parser.add_argument(
        "--pairs", type=str,
        default="BTC-USD,ETH-USD,SOL-USD,BNB-USD,LINK-USD",
        help="Paires separees par virgule",
    )
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--interval", type=str, default="15m",
                        help="Timeframe des bougies (5m, 15m, 1h)")
    parser.add_argument("--compare", action="store_true",
                        help="Lancer les 12 variantes A/B")
    parser.add_argument("--minimal", action="store_true",
                        help="Test rapide benchmark minimal")
    parser.add_argument("--fee", type=float, default=None,
                        help="Override fee (entry + exit)")
    args = parser.parse_args()

    pairs = [p.strip() for p in args.pairs.split(",")]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(args.years * 365.25))

    if args.minimal:
        cfg = HybridConfig(
            name="MINIMAL_FAST",
            market_filter_enabled=False,
            scalp_enabled=True,
            trend_enabled=False,
            scalp_rsi_entry=35.0,
            scalp_support_proximity_pct=999.0,
            use_atr_tp=False,
            use_atr_sl=False,
            tp_fixed_pct=0.03,
            sl_fixed_pct=0.02,
            min_atr_pct=0.0,
            min_candle_range_pct=0.0,
        )
        print(f"\n=== MINIMAL BENCHMARK ===")
        print(f"RSI < 35, TP 3%, SL 2%, pas de filtre marche")
        print(f"Paires: {', '.join(pairs)} | {args.interval}")
        print(f"Periode: {start.date()} -> {end.date()}")
        print(f"Capital: ${args.balance:,.0f}")

        trades, eq, final = run_multipair(pairs, start, end, cfg, args.balance, args.interval)
        m = compute_metrics(trades, eq, args.balance, final, start, end)
        print_report(m, cfg, args.balance)
        generate_charts(eq, trades, m, cfg, args.balance)
        return

    if args.compare:
        variants = get_variants()

        print(f"\n{'=' * 130}")
        print(f"  HYBRID SCALPING + TREND  — {len(variants)} variantes A/B | "
              f"Capital: ${args.balance:,.0f} | {args.interval}")
        print(f"  Paires: {', '.join(pairs)}")
        print(f"  Periode: {start.date()} -> {end.date()} ({args.years:.1f} ans)")
        print(f"{'=' * 130}")

        results = []
        for cfg in variants:
            if args.fee is not None:
                cfg.entry_fee_pct = args.fee
                cfg.exit_fee_pct = args.fee

            tp_label = f"ATR×{cfg.tp_atr_mult}" if cfg.use_atr_tp else f"{cfg.tp_fixed_pct * 100:.1f}%"
            sl_label = f"ATR×{cfg.sl_atr_mult}" if cfg.use_atr_sl else f"{cfg.sl_fixed_pct * 100:.1f}%"
            print(f"\n>>> {cfg.name}: filter={'ON' if cfg.market_filter_enabled else 'OFF'} "
                  f"scalp={'ON' if cfg.scalp_enabled else 'OFF'} "
                  f"trend={'ON' if cfg.trend_enabled else 'OFF'} "
                  f"TP={tp_label} SL={sl_label} ...")

            trades, eq, final = run_multipair(
                pairs, start, end, cfg, args.balance, args.interval,
            )
            m = compute_metrics(trades, eq, args.balance, final, start, end)
            results.append((cfg, m, trades, eq))

            print(f"    -> {m['n_trades']} trades | WR {m['win_rate']:.1%} | "
                  f"PF {m['profit_factor']:.2f} | PnL ${m['total_pnl']:+.2f} | "
                  f"DD {m['max_dd']:.1%} | Sharpe {m['sharpe']:.2f}")

        # ── Tableau comparatif ──
        print(f"\n{'=' * 145}")
        print(f"  {'Variante':20s} | {'Filter':>6s} | {'Scalp':>5s} | {'Trend':>5s} | "
              f"{'TP':>8s} | {'SL':>8s} | "
              f"{'Trades':>6s} | {'WR':>6s} | {'PF':>5s} | "
              f"{'PnL':>10s} | {'DD':>7s} | {'Sharpe':>6s} | {'PnL/j':>8s}")
        print("-" * 145)
        for cfg, m, _, _ in results:
            tp_l = f"ATR×{cfg.tp_atr_mult}" if cfg.use_atr_tp else f"{cfg.tp_fixed_pct * 100:.1f}%"
            sl_l = f"ATR×{cfg.sl_atr_mult}" if cfg.use_atr_sl else f"{cfg.sl_fixed_pct * 100:.1f}%"
            print(f"  {cfg.name:20s} | {'ON' if cfg.market_filter_enabled else 'OFF':>6s} | "
                  f"{'ON' if cfg.scalp_enabled else 'OFF':>5s} | "
                  f"{'ON' if cfg.trend_enabled else 'OFF':>5s} | "
                  f"{tp_l:>8s} | {sl_l:>8s} | "
                  f"{m['n_trades']:6d} | {m['win_rate']:5.1%} | "
                  f"{m['profit_factor']:5.2f} | ${m['total_pnl']:+9.2f} | "
                  f"{m['max_dd']:6.1%} | {m['sharpe']:6.2f} | "
                  f"${m['daily_pnl_avg']:+7.2f}")
        print("=" * 145)

        # ── Comparaisons cles V4 ──
        print(f"\n  COMPARAISONS CLES V4 :")
        print("  " + "-" * 76)

        def _find(name): return next((r for r in results if r[0].name == name), None)

        # V3 ref vs V4 no trend up
        ref = _find("V3_MOM50_REF")
        ntu = _find("V4_NO_TRENDUP")
        if ref and ntu:
            print(f"  V3 MOM50 (ref)     : PF {ref[1]['profit_factor']:.2f} | WR {ref[1]['win_rate']:.1%} | PnL ${ref[1]['total_pnl']:+.2f} | DD {ref[1]['max_dd']:.1%}")
            print(f"  V4 NO_TRENDUP      : PF {ntu[1]['profit_factor']:.2f} | WR {ntu[1]['win_rate']:.1%} | PnL ${ntu[1]['total_pnl']:+.2f} | DD {ntu[1]['max_dd']:.1%}")
            delta = ntu[1]['profit_factor'] - ref[1]['profit_factor']
            print(f"  → Gain PF          : {delta:+.2f}")

        # RSI exit comparison
        r50 = _find("V4_RSI_EXIT_50")
        r60 = _find("V4_RSI_EXIT_60")
        if ntu and r50 and r60:
            print(f"\n  RSI EXIT :")
            print(f"  RSI exit 55 (def)  : PF {ntu[1]['profit_factor']:.2f} | {ntu[1]['n_trades']} trades | PnL ${ntu[1]['total_pnl']:+.2f}")
            print(f"  RSI exit 50        : PF {r50[1]['profit_factor']:.2f} | {r50[1]['n_trades']} trades | PnL ${r50[1]['total_pnl']:+.2f}")
            print(f"  RSI exit 60        : PF {r60[1]['profit_factor']:.2f} | {r60[1]['n_trades']} trades | PnL ${r60[1]['total_pnl']:+.2f}")

        # RSI entry comparison
        m45 = _find("V4_MOM45")
        m55 = _find("V4_MOM55")
        if ntu and m45 and m55:
            print(f"\n  RSI ENTRY :")
            print(f"  RSI entry 50 (def) : PF {ntu[1]['profit_factor']:.2f} | {ntu[1]['n_trades']} trades")
            print(f"  RSI entry 45       : PF {m45[1]['profit_factor']:.2f} | {m45[1]['n_trades']} trades")
            print(f"  RSI entry 55       : PF {m55[1]['profit_factor']:.2f} | {m55[1]['n_trades']} trades")

        # SL / R:R
        rr31 = _find("V4_RR31")
        sl_t = _find("V4_SL_TIGHT")
        if ntu and rr31 and sl_t:
            print(f"\n  R:R :")
            print(f"  TP 2x SL 1x (def) : PF {ntu[1]['profit_factor']:.2f}")
            print(f"  TP 3x SL 1x       : PF {rr31[1]['profit_factor']:.2f}")
            print(f"  TP 2x SL 0.7x     : PF {sl_t[1]['profit_factor']:.2f}")

        # Zero fee
        zf = _find("V4_ZERO_FEE")
        if ntu and zf:
            print(f"\n  FEES :")
            print(f"  Revolut X (0+0.09%): PF {ntu[1]['profit_factor']:.2f} | PnL ${ntu[1]['total_pnl']:+.2f} | Fees ${ntu[1]['total_fees']:.2f}")
            print(f"  Zero fees          : PF {zf[1]['profit_factor']:.2f} | PnL ${zf[1]['total_pnl']:+.2f}")

        # Best
        best_v = _find("V4_BEST")
        if best_v:
            print(f"\n  ★ V4 BEST COMBO    : PF {best_v[1]['profit_factor']:.2f} | WR {best_v[1]['win_rate']:.1%} | PnL ${best_v[1]['total_pnl']:+.2f} | DD {best_v[1]['max_dd']:.1%}")

        brk = _find("V4_BRK12_NOTREND")
        if brk:
            print(f"  ★ V4 BREAKOUT 12   : PF {brk[1]['profit_factor']:.2f} | WR {brk[1]['win_rate']:.1%} | PnL ${brk[1]['total_pnl']:+.2f} | DD {brk[1]['max_dd']:.1%}")

        # Best overall
        best_idx = max(range(len(results)), key=lambda i: results[i][1].get("total_pnl", 0))
        best_cfg, best_m, best_trades, best_eq = results[best_idx]
        print(f"\n  ★ MEILLEURE VARIANTE : {best_cfg.name}")
        print_report(best_m, best_cfg, args.balance)
        generate_charts(best_eq, best_trades, best_m, best_cfg, args.balance)

        # Chart de la pire aussi pour comparaison
        worst_idx = min(range(len(results)), key=lambda i: results[i][1].get("total_pnl", 0))
        if worst_idx != best_idx:
            worst_cfg, worst_m, worst_trades, worst_eq = results[worst_idx]
            generate_charts(worst_eq, worst_trades, worst_m, worst_cfg, args.balance)

    else:
        # Run unique
        cfg = HybridConfig(name="CUSTOM")
        if args.fee is not None:
            cfg.entry_fee_pct = args.fee
            cfg.exit_fee_pct = args.fee

        print(f"\nBacktest Hybrid Scalping + Trend Following")
        print(f"Paires: {', '.join(pairs)} | {args.interval}")
        print(f"Periode: {start.date()} -> {end.date()} ({args.years:.1f} ans)")
        print(f"Capital: ${args.balance:,.0f}")

        trades, eq, final = run_multipair(
            pairs, start, end, cfg, args.balance, args.interval,
        )
        m = compute_metrics(trades, eq, args.balance, final, start, end)
        print_report(m, cfg, args.balance)
        generate_charts(eq, trades, m, cfg, args.balance)


if __name__ == "__main__":
    main()
