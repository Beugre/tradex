#!/usr/bin/env python3
"""
Backtest Momentum Continuation — comparaison multi-timeframes.

Compare la stratégie MC sur M5, M15, H1, H4 avec paramètres scalés.
Le but : trouver le timeframe optimal (M5 étant trop bruité, PF < 0.6).

Usage:
    python -m backtest.run_backtest_momentum_timeframes
    python -m backtest.run_backtest_momentum_timeframes --tf M15 H1
    python -m backtest.run_backtest_momentum_timeframes --months 24
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data_loader import download_candles
from src.core.models import Candle
from src.core.momentum_engine import (
    MCConfig,
    ema,
    sma,
    atr_series,
    adx_series,
    rsi_series,
    rolling_min,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MOMENTUM_PAIRS = ["ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "LINK-USD", "ADA-USD", "LTC-USD"]

# Revolut X fees
MAKER_FEE = 0.0
TAKER_FEE = 0.0009


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


@dataclass
class GenericIndicators:
    """Indicateurs pré-calculés pour une paire (base TF + macro TF)."""
    ema20: list[float] = field(default_factory=list)
    ema50: list[float] = field(default_factory=list)
    adx: list[float] = field(default_factory=list)
    rsi: list[float] = field(default_factory=list)
    vol_ma20: list[float] = field(default_factory=list)
    vol_ma10: list[float] = field(default_factory=list)
    swing_low_10: list[float] = field(default_factory=list)
    # Macro TF
    macro_atr: list[float] = field(default_factory=list)
    macro_atr_ma: list[float] = field(default_factory=list)
    macro_vol_ma: list[float] = field(default_factory=list)
    macro_bar_count: int = 0
    macro_resample_factor: int = 3


@dataclass
class _OpenPos:
    symbol: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    size_usd: float
    entry_bar: int
    entry_ts: int
    peak_price: float
    trailing_active: bool = False
    impulse_body_pct: float = 0.0
    impulse_vol_ratio: float = 0.0
    retrace_pct: float = 0.0


@dataclass
class _PairSignalState:
    phase: str = "IDLE"
    impulse_high: float = 0.0
    impulse_low: float = 0.0
    impulse_close: float = 0.0
    impulse_bar_idx: int = 0
    impulse_body_pct: float = 0.0
    impulse_vol_ratio: float = 0.0
    pullback_low: float = 0.0
    pullback_bars: int = 0
    cooldown_until: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  TIMEFRAME CONFIGS — paramètres scalés par TF
# ══════════════════════════════════════════════════════════════════════════════

# Mapping TF → (Binance interval, macro resample factor, data months multiplier)
TF_META = {
    "M5":  {"interval": "5m",  "macro_factor": 3,  "min_bars_warmup": 60},
    "M15": {"interval": "15m", "macro_factor": 4,  "min_bars_warmup": 60},
    "H1":  {"interval": "1h",  "macro_factor": 4,  "min_bars_warmup": 60},
    "H4":  {"interval": "4h",  "macro_factor": 6,  "min_bars_warmup": 60},
}


def _make_tf_configs(tf: str) -> list[tuple[str, MCConfig, dict]]:
    """Crée les configs adaptées pour un timeframe donné.

    Logique de scaling :
    - M5  : params de production (body 0.4%, SL 0.4-0.8%, TP 2%)
    - M15 : ×2-3 sur body/SL/TP, pullback_max_bars réduit
    - H1  : ×4-5 sur body/SL/TP
    - H4  : ×6-8 sur body/SL/TP
    """
    configs = []

    if tf == "M5":
        # Production actuelle
        configs.append((f"M5_PROD", MCConfig(), {
            "tf": "M5", "body": "0.4%", "sl": "0.4-0.8%", "tp": "2%",
            "trail": "0.15%", "pullback_bars": "35",
        }))
        # M5 avec TP plus large
        configs.append((f"M5_WIDE", MCConfig(
            sl_min_pct=0.006, sl_max_pct=0.012,
            tp_pct=0.030, trail_trigger_pct=0.008,
            trail_distance_pct=0.004,
        ), {
            "tf": "M5", "body": "0.4%", "sl": "0.6-1.2%", "tp": "3%",
            "trail": "0.4%", "pullback_bars": "35",
        }))

    elif tf == "M15":
        # M15 Standard
        configs.append((f"M15_STD", MCConfig(
            impulse_body_min_pct=0.006,   # 0.6%
            impulse_vol_mult=2.0,
            adx_min=15.0,
            sl_min_pct=0.006,             # 0.6%
            sl_max_pct=0.015,             # 1.5%
            tp_pct=0.030,                 # 3%
            trail_trigger_pct=0.008,      # 0.8%
            trail_distance_pct=0.004,     # 0.4%
            pullback_max_bars=25,
            cooldown_bars=4,
        ), {
            "tf": "M15", "body": "0.6%", "sl": "0.6-1.5%", "tp": "3%",
            "trail": "0.4%", "pullback_bars": "25",
        }))
        # M15 Wide
        configs.append((f"M15_WIDE", MCConfig(
            impulse_body_min_pct=0.008,   # 0.8%
            impulse_vol_mult=1.8,
            adx_min=18.0,
            sl_min_pct=0.008,             # 0.8%
            sl_max_pct=0.020,             # 2%
            tp_pct=0.045,                 # 4.5%
            trail_trigger_pct=0.012,      # 1.2%
            trail_distance_pct=0.006,     # 0.6%
            pullback_max_bars=20,
            cooldown_bars=3,
        ), {
            "tf": "M15", "body": "0.8%", "sl": "0.8-2%", "tp": "4.5%",
            "trail": "0.6%", "pullback_bars": "20",
        }))
        # M15 Strict (moins de trades, meilleure qualité)
        configs.append((f"M15_STRICT", MCConfig(
            impulse_body_min_pct=0.010,   # 1%
            impulse_vol_mult=2.5,
            adx_min=22.0,
            sl_min_pct=0.008,
            sl_max_pct=0.018,
            tp_pct=0.040,
            trail_trigger_pct=0.010,
            trail_distance_pct=0.005,
            pullback_retrace_min=0.30,
            pullback_retrace_max=0.50,
            pullback_max_bars=18,
            cooldown_bars=4,
        ), {
            "tf": "M15", "body": "1%", "sl": "0.8-1.8%", "tp": "4%",
            "trail": "0.5%", "pullback_bars": "18", "adx": "22",
        }))

    elif tf == "H1":
        # H1 Standard
        configs.append((f"H1_STD", MCConfig(
            impulse_body_min_pct=0.012,   # 1.2%
            impulse_vol_mult=1.8,
            adx_min=15.0,
            sl_min_pct=0.012,             # 1.2%
            sl_max_pct=0.025,             # 2.5%
            tp_pct=0.050,                 # 5%
            trail_trigger_pct=0.015,      # 1.5%
            trail_distance_pct=0.008,     # 0.8%
            pullback_max_bars=15,
            cooldown_bars=3,
        ), {
            "tf": "H1", "body": "1.2%", "sl": "1.2-2.5%", "tp": "5%",
            "trail": "0.8%", "pullback_bars": "15",
        }))
        # H1 Wide (trend following)
        configs.append((f"H1_WIDE", MCConfig(
            impulse_body_min_pct=0.015,   # 1.5%
            impulse_vol_mult=1.5,
            adx_min=18.0,
            sl_min_pct=0.015,             # 1.5%
            sl_max_pct=0.035,             # 3.5%
            tp_pct=0.080,                 # 8%
            trail_trigger_pct=0.020,      # 2%
            trail_distance_pct=0.010,     # 1%
            pullback_max_bars=12,
            cooldown_bars=2,
        ), {
            "tf": "H1", "body": "1.5%", "sl": "1.5-3.5%", "tp": "8%",
            "trail": "1%", "pullback_bars": "12",
        }))
        # H1 Scalp (TP court, pas de trail)
        configs.append((f"H1_SCALP", MCConfig(
            impulse_body_min_pct=0.010,   # 1%
            impulse_vol_mult=2.0,
            adx_min=15.0,
            sl_min_pct=0.010,             # 1%
            sl_max_pct=0.020,             # 2%
            tp_pct=0.030,                 # 3%
            trail_trigger_pct=0.999,      # Jamais de trail
            trail_distance_pct=0.001,
            pullback_max_bars=15,
            cooldown_bars=3,
        ), {
            "tf": "H1", "body": "1%", "sl": "1-2%", "tp": "3%",
            "trail": "jamais", "pullback_bars": "15",
        }))

    elif tf == "H4":
        # H4 Standard
        configs.append((f"H4_STD", MCConfig(
            impulse_body_min_pct=0.020,   # 2%
            impulse_vol_mult=1.5,
            adx_min=15.0,
            sl_min_pct=0.020,             # 2%
            sl_max_pct=0.040,             # 4%
            tp_pct=0.080,                 # 8%
            trail_trigger_pct=0.025,      # 2.5%
            trail_distance_pct=0.012,     # 1.2%
            pullback_max_bars=10,
            cooldown_bars=2,
        ), {
            "tf": "H4", "body": "2%", "sl": "2-4%", "tp": "8%",
            "trail": "1.2%", "pullback_bars": "10",
        }))
        # H4 Wide (swing)
        configs.append((f"H4_WIDE", MCConfig(
            impulse_body_min_pct=0.025,   # 2.5%
            impulse_vol_mult=1.5,
            adx_min=18.0,
            sl_min_pct=0.025,             # 2.5%
            sl_max_pct=0.050,             # 5%
            tp_pct=0.120,                 # 12%
            trail_trigger_pct=0.030,      # 3%
            trail_distance_pct=0.015,     # 1.5%
            pullback_max_bars=8,
            cooldown_bars=2,
        ), {
            "tf": "H4", "body": "2.5%", "sl": "2.5-5%", "tp": "12%",
            "trail": "1.5%", "pullback_bars": "8",
        }))
        # H4 Relax (body + faible, plus de trades)
        configs.append((f"H4_RELAX", MCConfig(
            impulse_body_min_pct=0.015,   # 1.5%
            impulse_vol_mult=1.3,
            adx_min=12.0,
            sl_min_pct=0.015,             # 1.5%
            sl_max_pct=0.035,             # 3.5%
            tp_pct=0.060,                 # 6%
            trail_trigger_pct=0.020,      # 2%
            trail_distance_pct=0.010,     # 1%
            pullback_max_bars=12,
            cooldown_bars=2,
        ), {
            "tf": "H4", "body": "1.5%", "sl": "1.5-3.5%", "tp": "6%",
            "trail": "1%", "pullback_bars": "12", "adx": "12",
        }))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  RESAMPLE GENERIC
# ══════════════════════════════════════════════════════════════════════════════

def resample_candles(candles: list[Candle], factor: int) -> list[Candle]:
    """Resample des bougies par un facteur donné.

    Ex: factor=3 pour M5→M15, factor=4 pour M15→H1 ou H1→H4, factor=6 pour H4→D1.
    """
    resampled: list[Candle] = []
    for i in range(0, len(candles) - factor + 1, factor):
        group = candles[i:i + factor]
        resampled.append(Candle(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
        ))
    return resampled


# ══════════════════════════════════════════════════════════════════════════════
#  GENERIC INDICATOR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_generic_indicators(
    candles: list[Candle],
    cfg: MCConfig,
    macro_resample_factor: int = 3,
) -> GenericIndicators:
    """Calcule les indicateurs sur le TF de base + macro TF resampleé."""
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    lows = [c.low for c in candles]

    ind = GenericIndicators()
    ind.macro_resample_factor = macro_resample_factor

    # Base TF indicators
    ind.ema20 = ema(closes, cfg.ema_fast_period)
    ind.ema50 = ema(closes, cfg.ema_slow_period)
    ind.adx = adx_series(candles, cfg.adx_period)
    ind.rsi = rsi_series(candles, cfg.rsi_period)
    ind.vol_ma20 = sma(volumes, cfg.impulse_vol_ma_period)
    ind.vol_ma10 = sma(volumes, cfg.entry_vol_ma_period)
    ind.swing_low_10 = rolling_min(lows, 10)

    # Macro TF
    macro = resample_candles(candles, macro_resample_factor)
    ind.macro_bar_count = len(macro)
    if macro:
        macro_atr_raw = atr_series(macro, cfg.atr_m15_period)
        ind.macro_atr = macro_atr_raw
        ind.macro_atr_ma = sma(macro_atr_raw, cfg.atr_m15_ma_period)
        macro_vols = [c.volume for c in macro]
        ind.macro_vol_ma = sma(macro_vols, cfg.vol_m15_ma_period)

    return ind


# ══════════════════════════════════════════════════════════════════════════════
#  GENERIC MOMENTUM SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_momentum_generic(
    all_candles: dict[str, list[Candle]],
    cfg: MCConfig,
    macro_resample_factor: int = 3,
    initial_balance: float = 500.0,
) -> BacktestResult:
    """Simule le Momentum Continuation sur n'importe quel timeframe.

    Le macro filter utilise le TF resampleé par `macro_resample_factor`.
    """
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]
    consecutive_losses = 0

    # 1. Pré-calcul indicateurs
    logger.info("  Pré-calcul des indicateurs (macro factor=%d)…", macro_resample_factor)
    all_indicators: dict[str, GenericIndicators] = {}
    all_macro: dict[str, list[Candle]] = {}
    for symbol, candles in all_candles.items():
        all_indicators[symbol] = compute_generic_indicators(candles, cfg, macro_resample_factor)
        all_macro[symbol] = resample_candles(candles, macro_resample_factor)
    logger.info("  Indicateurs calculés pour %d paires", len(all_candles))

    # 2. État signal par paire
    states: dict[str, _PairSignalState] = {sym: _PairSignalState() for sym in all_candles}

    # 3. Simulation bar-par-bar
    min_len = min(len(c) for c in all_candles.values())
    start_bar = TF_META.get("M5", {}).get("min_bars_warmup", 60)  # warmup

    for bar_idx in range(start_bar, min_len):
        # 3a. Gestion positions ouvertes
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]

            # Check SL
            if c.low <= pos.sl_price:
                pnl_pct = (pos.sl_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.sl_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="SL",
                    impulse_body_pct=pos.impulse_body_pct,
                    impulse_vol_ratio=pos.impulse_vol_ratio,
                    retrace_pct=pos.retrace_pct, hold_bars=bar_idx - pos.entry_bar,
                ))
                consecutive_losses = consecutive_losses + 1 if pnl_usd < 0 else 0
                positions.remove(pos)
                continue

            # Check TP
            if c.high >= pos.tp_price:
                pnl_pct = (pos.tp_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * MAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="TP",
                    impulse_body_pct=pos.impulse_body_pct,
                    impulse_vol_ratio=pos.impulse_vol_ratio,
                    retrace_pct=pos.retrace_pct, hold_bars=bar_idx - pos.entry_bar,
                ))
                consecutive_losses = 0
                positions.remove(pos)
                continue

            # Trailing stop
            new_peak = max(pos.peak_price, c.close)
            if not pos.trailing_active:
                profit_pct = (c.close - pos.entry_price) / pos.entry_price
                if profit_pct >= cfg.trail_trigger_pct:
                    pos.trailing_active = True
                    new_sl = new_peak * (1 - cfg.trail_distance_pct)
                    pos.sl_price = max(new_sl, pos.sl_price)
            if pos.trailing_active:
                new_sl = new_peak * (1 - cfg.trail_distance_pct)
                pos.sl_price = max(new_sl, pos.sl_price)
            pos.peak_price = new_peak

            # Re-check SL after trailing
            if pos.trailing_active and c.low <= pos.sl_price:
                exit_price = pos.sl_price
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="TRAILING",
                    impulse_body_pct=pos.impulse_body_pct,
                    impulse_vol_ratio=pos.impulse_vol_ratio,
                    retrace_pct=pos.retrace_pct, hold_bars=bar_idx - pos.entry_bar,
                ))
                consecutive_losses = 0 if pnl_usd >= 0 else consecutive_losses + 1
                positions.remove(pos)
                continue

        # 3b. Détection de signaux
        if len(positions) < cfg.max_positions and consecutive_losses < cfg.max_consecutive_losses and balance > 10:
            # Capital disponible (corrigé — pas de double allocation)
            capital_in_use = sum(p.size_usd for p in positions)
            available_capital = max(0, balance)

            for symbol in all_candles:
                if any(p.symbol == symbol for p in positions):
                    continue
                if len(positions) >= cfg.max_positions:
                    break

                candles = all_candles[symbol]
                c = candles[bar_idx]
                ind = all_indicators[symbol]
                state = states[symbol]
                mf = macro_resample_factor

                # Cooldown
                if bar_idx < state.cooldown_until:
                    continue

                # ── IMPULSE / PULLBACK tracking ──
                if state.phase in ("IMPULSE", "PULLBACK"):
                    state.pullback_bars += 1

                    if state.pullback_bars > cfg.pullback_max_bars:
                        state.phase = "IDLE"
                        continue

                    ema50 = ind.ema50[bar_idx] if bar_idx < len(ind.ema50) else 0
                    if c.close < ema50:
                        state.phase = "IDLE"
                        continue

                    if c.close < state.pullback_low:
                        state.pullback_low = c.close

                    impulse_range = state.impulse_high - state.impulse_low
                    if impulse_range <= 0:
                        state.phase = "IDLE"
                        continue
                    retrace = (state.impulse_close - state.pullback_low) / impulse_range
                    state.phase = "PULLBACK"

                    pullback_ok = cfg.pullback_retrace_min <= retrace <= cfg.pullback_retrace_max

                    rsi_val = ind.rsi[bar_idx] if bar_idx < len(ind.rsi) else 50
                    rsi_ok = cfg.rsi_pullback_min <= rsi_val <= cfg.rsi_pullback_max

                    ema20 = ind.ema20[bar_idx] if bar_idx < len(ind.ema20) else 0
                    above_ema = c.close > ema20

                    if not (pullback_ok and rsi_ok and above_ema):
                        continue

                    # Entry trigger
                    if bar_idx < 1:
                        continue
                    prev_high = candles[bar_idx - 1].high
                    vol_ma10 = ind.vol_ma10[bar_idx] if bar_idx < len(ind.vol_ma10) else 0
                    if c.close <= prev_high:
                        continue
                    if vol_ma10 > 0 and c.volume < cfg.entry_vol_mult * vol_ma10:
                        continue

                    # ENTRY SIGNAL
                    swing_low = ind.swing_low_10[bar_idx] if bar_idx < len(ind.swing_low_10) else c.low
                    sl_price = swing_low
                    sl_pct = (c.close - sl_price) / c.close if c.close > 0 else 0
                    sl_pct = max(cfg.sl_min_pct, min(cfg.sl_max_pct, sl_pct))
                    sl_price = c.close * (1 - sl_pct)
                    tp_price = c.close * (1 + cfg.tp_pct)

                    # Sizing (corrigé — sur capital disponible)
                    sl_distance = abs(c.close - sl_price)
                    if sl_distance <= 0:
                        state.phase = "IDLE"
                        state.cooldown_until = bar_idx + cfg.cooldown_bars
                        continue
                    risk_amount = available_capital * cfg.risk_per_trade
                    size = risk_amount / sl_distance
                    size_usd = size * c.close
                    max_size_usd = available_capital * cfg.max_position_pct
                    if size_usd > max_size_usd:
                        size_usd = max_size_usd
                        size = size_usd / c.close
                    if size_usd < 5:
                        state.phase = "IDLE"
                        state.cooldown_until = bar_idx + cfg.cooldown_bars
                        continue

                    # Execute
                    fee = size_usd * MAKER_FEE
                    balance -= size_usd + fee
                    available_capital -= size_usd + fee
                    positions.append(_OpenPos(
                        symbol=symbol, entry_price=c.close,
                        sl_price=sl_price, tp_price=tp_price,
                        size=size, size_usd=size_usd,
                        entry_bar=bar_idx, entry_ts=c.timestamp,
                        peak_price=c.close, trailing_active=False,
                        impulse_body_pct=state.impulse_body_pct,
                        impulse_vol_ratio=state.impulse_vol_ratio,
                        retrace_pct=retrace,
                    ))
                    state.phase = "IDLE"
                    state.cooldown_until = bar_idx + cfg.cooldown_bars
                    continue

                # ── IDLE → chercher impulsion ──
                # Macro filter
                macro_idx = bar_idx // mf
                m_atr = ind.macro_atr
                m_atr_ma = ind.macro_atr_ma
                if macro_idx < 1 or macro_idx >= len(m_atr) or macro_idx >= len(m_atr_ma):
                    continue
                if m_atr[macro_idx] <= m_atr_ma[macro_idx]:
                    continue
                # Macro volume filter
                if macro_idx < len(ind.macro_vol_ma) and ind.macro_vol_ma[macro_idx] > 0:
                    macro_candles = all_macro[symbol]
                    if macro_idx < len(macro_candles) and macro_candles[macro_idx].volume <= ind.macro_vol_ma[macro_idx]:
                        continue

                # Impulse detection
                body = c.close - c.open
                body_pct = abs(body) / c.open if c.open > 0 else 0
                candle_range = c.high - c.low
                if candle_range <= 0:
                    continue
                is_bullish = body > 0 and body_pct >= cfg.impulse_body_min_pct
                close_in_top = (c.close - c.low) / candle_range >= (1 - cfg.impulse_close_top_pct)
                if not (is_bullish and close_in_top):
                    continue

                # Volume
                vol_ma = ind.vol_ma20[bar_idx] if bar_idx < len(ind.vol_ma20) else 0
                if vol_ma <= 0 or c.volume < cfg.impulse_vol_mult * vol_ma:
                    continue

                # ADX
                adx_val = ind.adx[bar_idx] if bar_idx < len(ind.adx) else 0
                if adx_val < cfg.adx_min:
                    continue

                # Trend: EMA20 > EMA50
                ema20 = ind.ema20[bar_idx] if bar_idx < len(ind.ema20) else 0
                ema50 = ind.ema50[bar_idx] if bar_idx < len(ind.ema50) else 0
                if ema20 <= ema50:
                    continue

                vol_ratio = c.volume / vol_ma if vol_ma > 0 else 0

                # Impulsion détectée
                state.phase = "IMPULSE"
                state.impulse_high = c.high
                state.impulse_low = c.low
                state.impulse_close = c.close
                state.impulse_bar_idx = bar_idx
                state.impulse_body_pct = body_pct
                state.impulse_vol_ratio = vol_ratio
                state.pullback_low = c.close
                state.pullback_bars = 0

        # 3c. Equity tracking
        pos_value = sum(
            p.size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pos_value)

    # Fermer positions restantes
    for pos in positions:
        candles = all_candles[pos.symbol]
        last_price = candles[min_len - 1].close
        pnl_pct = (last_price - pos.entry_price) / pos.entry_price
        pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
        balance += pos.size_usd + pnl_usd
        closed_trades.append(Trade(
            symbol=pos.symbol, side="LONG",
            entry_price=pos.entry_price, exit_price=last_price,
            size=pos.size, entry_time=pos.entry_ts,
            exit_time=candles[min_len - 1].timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="END",
            hold_bars=min_len - 1 - pos.entry_bar,
        ))

    return BacktestResult(
        label="",
        trades=closed_trades,
        equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def compute_kpis(result: BacktestResult) -> dict:
    trades = result.trades
    n = len(trades)
    eq = result.equity_curve

    if n == 0:
        return {
            "label": result.label, "trades": 0, "win_rate": 0, "pf": 0,
            "pnl": 0, "avg_pnl": 0, "max_dd": 0, "final_eq": result.final_equity,
            "avg_win": 0, "avg_loss": 0,
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_profit = sum(t.pnl_usd for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) if losses else 1e-9

    peak = result.initial_balance
    max_dd = 0.0
    for e in eq:
        peak = max(peak, e)
        dd = (e - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    avg_win = (gross_profit / len(wins)) if wins else 0
    avg_loss = (sum(t.pnl_usd for t in losses) / len(losses)) if losses else 0

    return {
        "label": result.label, "trades": n,
        "win_rate": len(wins) / n * 100,
        "pf": gross_profit / gross_loss if gross_loss > 0 else 999,
        "pnl": sum(t.pnl_usd for t in trades),
        "avg_pnl": sum(t.pnl_usd for t in trades) / n,
        "max_dd": max_dd * 100,
        "final_eq": result.final_equity,
        "avg_win": avg_win, "avg_loss": avg_loss,
    }


def print_comparison_table(results: list[dict], title: str) -> None:
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    header = f"{'Config':<20} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>10} {'Avg PnL':>9} {'Max DD%':>8} {'Final $':>10} {'R:R':>6}"
    print(header)
    print("-" * 95)
    for kpi in results:
        rr = abs(kpi["avg_win"] / kpi["avg_loss"]) if kpi["avg_loss"] != 0 else 0
        line = (
            f"{kpi['label']:<20} "
            f"{kpi['trades']:>6} "
            f"{kpi['win_rate']:>6.1f}% "
            f"{kpi['pf']:>7.2f} "
            f"{kpi['pnl']:>+9.2f}$ "
            f"{kpi['avg_pnl']:>+8.2f}$ "
            f"{kpi['max_dd']:>7.1f}% "
            f"{kpi['final_eq']:>9.2f}$ "
            f"{rr:>5.2f}"
        )
        print(line)
    best = max(results, key=lambda r: r["pf"])
    print("-" * 95)
    print(f"  Meilleur PF : {best['label']} (PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$)")
    print()


def print_exit_breakdown(results: list[BacktestResult], title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title} — Répartition des sorties")
    print(f"{'='*70}")
    for r in results:
        exits = Counter(t.exit_reason for t in r.trades)
        total = len(r.trades) or 1
        print(f"\n  {r.label}:")
        for reason, count in sorted(exits.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            avg_pnl = sum(t.pnl_usd for t in r.trades if t.exit_reason == reason) / count
            print(f"    {reason:<15} : {count:>4} ({pct:>5.1f}%)  avg PnL: {avg_pnl:+.2f}$")


def plot_equity_curves(results: list[BacktestResult], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#8e44ad", "#2c3e50", "#c0392b"]
    for i, r in enumerate(results):
        ax.plot(r.equity_curve, label=r.label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_title(f"{title} — Equity Curves", fontsize=14)
    ax.set_xlabel("Barres")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Chart : {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_tf(tf: str, months: int, balance: float) -> list[dict]:
    """Lance le backtest pour un seul timeframe, retourne les KPIs."""
    meta = TF_META[tf]
    interval = meta["interval"]
    macro_factor = meta["macro_factor"]

    print(f"\n{'═'*70}")
    print(f"  MOMENTUM — Timeframe {tf} ({interval}) — {months} mois")
    print(f"{'═'*70}")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=months * 30)

    # Téléchargement
    all_candles: dict[str, list[Candle]] = {}
    for pair in MOMENTUM_PAIRS:
        logger.info("  %s %s (%s → %s)…", pair, tf, start.date(), end.date())
        candles = download_candles(pair, start, end, interval=interval)
        all_candles[pair] = candles
        logger.info("    %s : %d bougies %s", pair, len(candles), tf)

    # Configs pour ce TF
    configs = _make_tf_configs(tf)
    results: list[BacktestResult] = []
    kpis_list: list[dict] = []

    for label, cfg, desc in configs:
        logger.info("  %s…", label)
        result = simulate_momentum_generic(all_candles, cfg, macro_factor, initial_balance=balance)
        result.label = label
        result.config_desc = desc
        results.append(result)
        kpis = compute_kpis(result)
        kpis_list.append(kpis)

    print_comparison_table(kpis_list, f"MOMENTUM {tf} ({interval}, {months} mois, ${balance:.0f})")
    print_exit_breakdown(results, f"MOMENTUM {tf}")
    plot_equity_curves(results, f"Momentum {tf}", f"momentum_tf_{tf.lower()}_equity.png")

    return kpis_list


def run_all_timeframes(timeframes: list[str], months: int, balance: float) -> None:
    """Lance le backtest sur tous les TF et affiche un résumé comparatif."""
    all_kpis: list[dict] = []

    for tf in timeframes:
        # Adapter la durée selon le TF (plus de données pour les TF hauts)
        tf_months = months
        if tf == "H4" and months < 24:
            tf_months = 24  # H4 a besoin de + de données pour être significatif
        if tf == "H1" and months < 18:
            tf_months = 18

        kpis_list = run_single_tf(tf, tf_months, balance)
        # Prendre le meilleur variant de chaque TF
        best = max(kpis_list, key=lambda k: k["pf"])
        all_kpis.append(best)

    # Tableau récap cross-TF
    print(f"\n{'='*95}")
    print(f"  RÉSUMÉ CROSS-TIMEFRAME — Meilleur variant par TF")
    print(f"{'='*95}")
    header = f"{'Config':<20} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>10} {'Avg PnL':>9} {'Max DD%':>8} {'Final $':>10} {'R:R':>6}"
    print(header)
    print("-" * 95)
    for kpi in all_kpis:
        rr = abs(kpi["avg_win"] / kpi["avg_loss"]) if kpi["avg_loss"] != 0 else 0
        line = (
            f"{kpi['label']:<20} "
            f"{kpi['trades']:>6} "
            f"{kpi['win_rate']:>6.1f}% "
            f"{kpi['pf']:>7.2f} "
            f"{kpi['pnl']:>+9.2f}$ "
            f"{kpi['avg_pnl']:>+8.2f}$ "
            f"{kpi['max_dd']:>7.1f}% "
            f"{kpi['final_eq']:>9.2f}$ "
            f"{rr:>5.2f}"
        )
        print(line)

    overall_best = max(all_kpis, key=lambda k: k["pf"])
    print("-" * 95)
    print(f"  MEILLEUR GLOBAL : {overall_best['label']} — PF {overall_best['pf']:.2f}, PnL {overall_best['pnl']:+.2f}$, DD {overall_best['max_dd']:.1f}%")
    if overall_best["pf"] < 1.0:
        print("  ⚠️  AUCUN timeframe n'est rentable (PF < 1.0)")
        print("     → La stratégie MC n'est peut-être pas adaptée à ces paires crypto")
    elif overall_best["pf"] < 1.2:
        print("  ⚠️  Rentabilité marginale (PF < 1.2) — à valider sur plus de données")
    else:
        print(f"  ✅  Timeframe prometteur ! Tester avec plus de variantes sur {overall_best['label'][:3]}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Momentum multi-timeframes")
    parser.add_argument("--tf", nargs="+", default=["M5", "M15", "H1", "H4"],
                        choices=["M5", "M15", "H1", "H4"],
                        help="Timeframes à tester (défaut: tous)")
    parser.add_argument("--months", type=int, default=12,
                        help="Mois de données de base (augmenté auto pour H1/H4)")
    parser.add_argument("--balance", type=float, default=500,
                        help="Capital initial ($)")
    args = parser.parse_args()

    run_all_timeframes(args.tf, args.months, args.balance)

    print("═" * 70)
    print("  Backtest multi-timeframe terminé !")
    print("═" * 70)


if __name__ == "__main__":
    main()
