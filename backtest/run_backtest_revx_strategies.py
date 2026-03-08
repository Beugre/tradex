#!/usr/bin/env python3
"""
Backtest comparatif : RSI Oversold Bounce vs Bollinger Bounce
sur Revolut X (7 paires).

Stratégies :
  B) RSI Oversold Bounce — Achat RSI < seuil sur H1/H4, sortie RSI > seuil ou TP/SL
  D) Bollinger Bounce — Mean reversion via Bandes de Bollinger (achat BB lower, vente BB mid/upper)

Usage :
    python -m backtest.run_backtest_revx_strategies
    python -m backtest.run_backtest_revx_strategies --years 3 --balance 500
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest.data_loader import download_candles
from src.core.models import Candle
from src.core.indicators import ema, sma, atr_series, rsi_series, rolling_min

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PAIRS = ["ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "LINK-USD", "ADA-USD", "LTC-USD"]

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
    hold_bars: int = 0


@dataclass
class BacktestResult:
    label: str
    trades: list[Trade]
    equity_curve: list[float]
    initial_balance: float
    final_equity: float
    config_desc: dict


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RSIConfig:
    """Config pour RSI Oversold Bounce."""
    rsi_period: int = 14
    rsi_entry: float = 30.0        # Achat quand RSI < ce seuil
    rsi_exit: float = 55.0         # Sortie quand RSI > ce seuil
    sl_pct: float = 0.03           # SL fixe 3%
    tp_pct: float = 0.06           # TP fixe 6%
    trail_trigger_pct: float = 0.04   # Trail après +4%
    trail_distance_pct: float = 0.02  # Distance trailing 2%
    max_positions: int = 3
    risk_per_trade: float = 0.04   # 4% risque
    max_position_pct: float = 0.50 # Max 50% capital par position
    cooldown_bars: int = 3         # Cooldown par paire
    max_hold_bars: int = 50        # Sortie forcée après N barres (anti-stagnation)
    vol_filter: bool = False       # Filtre volume optionnel
    vol_mult: float = 1.0          # Volume > N× MA20


@dataclass
class BollingerConfig:
    """Config pour Bollinger Bounce."""
    bb_period: int = 20
    bb_std: float = 2.0            # Nb écarts-types
    rsi_period: int = 14
    rsi_max_entry: float = 40.0    # RSI < 40 pour confirmer oversold
    exit_target: str = "mid"       # "mid" ou "upper"
    sl_pct: float = 0.03           # SL fixe 3%
    tp_pct: float = 0.06           # TP fixe (fallback si exit_target pas atteint)
    trail_trigger_pct: float = 0.04
    trail_distance_pct: float = 0.02
    max_positions: int = 3
    risk_per_trade: float = 0.04
    max_position_pct: float = 0.50
    cooldown_bars: int = 3
    max_hold_bars: int = 50
    vol_filter: bool = False
    vol_mult: float = 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def bollinger_bands(
    closes: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    """Retourne (upper, mid, lower) Bollinger Bands."""
    n = len(closes)
    mid = sma(closes, period)
    upper = [0.0] * n
    lower = [0.0] * n
    for i in range(n):
        start = max(0, i - period + 1)
        window = closes[start:i + 1]
        if len(window) < 2:
            upper[i] = mid[i]
            lower[i] = mid[i]
            continue
        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = variance ** 0.5
        upper[i] = mid[i] + num_std * std
        lower[i] = mid[i] - num_std * std
    return upper, mid, lower


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


@dataclass
class _PairState:
    cooldown_until: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def _define_rsi_variants(tf: str) -> list[tuple[str, RSIConfig, dict]]:
    configs = []

    if tf == "H1":
        # RSI < 30, TP 5%, SL 2.5%
        configs.append(("RSI_H1_30", RSIConfig(
            rsi_entry=30, rsi_exit=55, sl_pct=0.025, tp_pct=0.05,
            trail_trigger_pct=0.03, trail_distance_pct=0.015,
            cooldown_bars=4, max_hold_bars=72,
        ), {"tf": "H1", "entry": "RSI<30", "exit": "RSI>55 | TP5% | trail",
            "sl": "2.5%", "hold_max": "72h"}))

        # RSI < 25, TP 6%, SL 3% — plus sélectif
        configs.append(("RSI_H1_25", RSIConfig(
            rsi_entry=25, rsi_exit=55, sl_pct=0.030, tp_pct=0.06,
            trail_trigger_pct=0.035, trail_distance_pct=0.018,
            cooldown_bars=6, max_hold_bars=72,
        ), {"tf": "H1", "entry": "RSI<25", "exit": "RSI>55 | TP6% | trail",
            "sl": "3%", "hold_max": "72h"}))

        # RSI < 28, TP 4%, SL 2% — scalp-like, rotate vite
        configs.append(("RSI_H1_FAST", RSIConfig(
            rsi_entry=28, rsi_exit=50, sl_pct=0.020, tp_pct=0.04,
            trail_trigger_pct=0.025, trail_distance_pct=0.012,
            cooldown_bars=3, max_hold_bars=48,
        ), {"tf": "H1", "entry": "RSI<28", "exit": "RSI>50 | TP4% | trail",
            "sl": "2%", "hold_max": "48h"}))

        # RSI < 30 + volume filter
        configs.append(("RSI_H1_VOL", RSIConfig(
            rsi_entry=30, rsi_exit=55, sl_pct=0.025, tp_pct=0.05,
            trail_trigger_pct=0.03, trail_distance_pct=0.015,
            cooldown_bars=4, max_hold_bars=72,
            vol_filter=True, vol_mult=1.5,
        ), {"tf": "H1", "entry": "RSI<30 + Vol>1.5×", "exit": "RSI>55 | TP5%",
            "sl": "2.5%", "hold_max": "72h"}))

    elif tf == "H4":
        # RSI < 30, TP 8%, SL 4%
        configs.append(("RSI_H4_30", RSIConfig(
            rsi_entry=30, rsi_exit=55, sl_pct=0.040, tp_pct=0.08,
            trail_trigger_pct=0.05, trail_distance_pct=0.025,
            cooldown_bars=3, max_hold_bars=30,
        ), {"tf": "H4", "entry": "RSI<30", "exit": "RSI>55 | TP8% | trail",
            "sl": "4%", "hold_max": "120h"}))

        # RSI < 25, TP 10%, SL 5%
        configs.append(("RSI_H4_25", RSIConfig(
            rsi_entry=25, rsi_exit=55, sl_pct=0.050, tp_pct=0.10,
            trail_trigger_pct=0.06, trail_distance_pct=0.030,
            cooldown_bars=4, max_hold_bars=25,
        ), {"tf": "H4", "entry": "RSI<25", "exit": "RSI>55 | TP10% | trail",
            "sl": "5%", "hold_max": "100h"}))

        # RSI < 28, TP 6%, SL 3% — rotation rapide
        configs.append(("RSI_H4_FAST", RSIConfig(
            rsi_entry=28, rsi_exit=50, sl_pct=0.030, tp_pct=0.06,
            trail_trigger_pct=0.04, trail_distance_pct=0.020,
            cooldown_bars=2, max_hold_bars=20,
        ), {"tf": "H4", "entry": "RSI<28", "exit": "RSI>50 | TP6% | trail",
            "sl": "3%", "hold_max": "80h"}))

        # RSI < 30 + volume
        configs.append(("RSI_H4_VOL", RSIConfig(
            rsi_entry=30, rsi_exit=55, sl_pct=0.040, tp_pct=0.08,
            trail_trigger_pct=0.05, trail_distance_pct=0.025,
            cooldown_bars=3, max_hold_bars=30,
            vol_filter=True, vol_mult=1.5,
        ), {"tf": "H4", "entry": "RSI<30 + Vol>1.5×", "exit": "RSI>55 | TP8%",
            "sl": "4%", "hold_max": "120h"}))

    return configs


def _define_bollinger_variants(tf: str) -> list[tuple[str, BollingerConfig, dict]]:
    configs = []

    if tf == "H1":
        # BB standard, exit mid
        configs.append(("BB_H1_MID", BollingerConfig(
            bb_period=20, bb_std=2.0, rsi_max_entry=40,
            exit_target="mid", sl_pct=0.025, tp_pct=0.05,
            trail_trigger_pct=0.03, trail_distance_pct=0.015,
            cooldown_bars=4, max_hold_bars=72,
        ), {"tf": "H1", "entry": "Close<BB_lower + RSI<40", "exit": "BB_mid | TP5%",
            "sl": "2.5%", "bb": "20/2σ"}))

        # BB standard, exit upper
        configs.append(("BB_H1_UPP", BollingerConfig(
            bb_period=20, bb_std=2.0, rsi_max_entry=40,
            exit_target="upper", sl_pct=0.025, tp_pct=0.08,
            trail_trigger_pct=0.04, trail_distance_pct=0.020,
            cooldown_bars=4, max_hold_bars=96,
        ), {"tf": "H1", "entry": "Close<BB_lower + RSI<40", "exit": "BB_upper | TP8%",
            "sl": "2.5%", "bb": "20/2σ"}))

        # BB 2.5σ (plus sélectif)
        configs.append(("BB_H1_2.5S", BollingerConfig(
            bb_period=20, bb_std=2.5, rsi_max_entry=35,
            exit_target="mid", sl_pct=0.030, tp_pct=0.06,
            trail_trigger_pct=0.035, trail_distance_pct=0.018,
            cooldown_bars=6, max_hold_bars=72,
        ), {"tf": "H1", "entry": "Close<BB_lower + RSI<35", "exit": "BB_mid | TP6%",
            "sl": "3%", "bb": "20/2.5σ"}))

        # BB + volume filter
        configs.append(("BB_H1_VOL", BollingerConfig(
            bb_period=20, bb_std=2.0, rsi_max_entry=40,
            exit_target="mid", sl_pct=0.025, tp_pct=0.05,
            trail_trigger_pct=0.03, trail_distance_pct=0.015,
            cooldown_bars=4, max_hold_bars=72,
            vol_filter=True, vol_mult=1.5,
        ), {"tf": "H1", "entry": "Close<BB_lower + RSI<40 + Vol>1.5×",
            "exit": "BB_mid | TP5%", "sl": "2.5%", "bb": "20/2σ"}))

    elif tf == "H4":
        # BB standard, exit mid
        configs.append(("BB_H4_MID", BollingerConfig(
            bb_period=20, bb_std=2.0, rsi_max_entry=40,
            exit_target="mid", sl_pct=0.040, tp_pct=0.08,
            trail_trigger_pct=0.05, trail_distance_pct=0.025,
            cooldown_bars=3, max_hold_bars=30,
        ), {"tf": "H4", "entry": "Close<BB_lower + RSI<40", "exit": "BB_mid | TP8%",
            "sl": "4%", "bb": "20/2σ"}))

        # BB standard, exit upper
        configs.append(("BB_H4_UPP", BollingerConfig(
            bb_period=20, bb_std=2.0, rsi_max_entry=40,
            exit_target="upper", sl_pct=0.040, tp_pct=0.12,
            trail_trigger_pct=0.06, trail_distance_pct=0.030,
            cooldown_bars=3, max_hold_bars=40,
        ), {"tf": "H4", "entry": "Close<BB_lower + RSI<40", "exit": "BB_upper | TP12%",
            "sl": "4%", "bb": "20/2σ"}))

        # BB 2.5σ
        configs.append(("BB_H4_2.5S", BollingerConfig(
            bb_period=20, bb_std=2.5, rsi_max_entry=35,
            exit_target="mid", sl_pct=0.050, tp_pct=0.10,
            trail_trigger_pct=0.06, trail_distance_pct=0.030,
            cooldown_bars=4, max_hold_bars=30,
        ), {"tf": "H4", "entry": "Close<BB_lower + RSI<35", "exit": "BB_mid | TP10%",
            "sl": "5%", "bb": "20/2.5σ"}))

        # BB + volume
        configs.append(("BB_H4_VOL", BollingerConfig(
            bb_period=20, bb_std=2.0, rsi_max_entry=40,
            exit_target="mid", sl_pct=0.040, tp_pct=0.08,
            trail_trigger_pct=0.05, trail_distance_pct=0.025,
            cooldown_bars=3, max_hold_bars=30,
            vol_filter=True, vol_mult=1.5,
        ), {"tf": "H4", "entry": "Close<BB_lower + RSI<40 + Vol>1.5×",
            "exit": "BB_mid | TP8%", "sl": "4%", "bb": "20/2σ"}))

    return configs


# ══════════════════════════════════════════════════════════════════════════════
#  RSI OVERSOLD BOUNCE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_rsi_bounce(
    all_candles: dict[str, list[Candle]],
    cfg: RSIConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    """Simule la stratégie RSI Oversold Bounce."""
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]

    # Pré-calcul indicateurs
    all_rsi: dict[str, list[float]] = {}
    all_vol_ma: dict[str, list[float]] = {}
    all_ema50: dict[str, list[float]] = {}
    for symbol, candles in all_candles.items():
        all_rsi[symbol] = rsi_series(candles, cfg.rsi_period)
        vols = [c.volume for c in candles]
        all_vol_ma[symbol] = sma(vols, 20)
        closes = [c.close for c in candles]
        all_ema50[symbol] = ema(closes, 50)

    states: dict[str, _PairState] = {sym: _PairState() for sym in all_candles}
    min_len = min(len(c) for c in all_candles.values())
    start_bar = 55  # warmup

    for bar_idx in range(start_bar, min_len):
        # Gestion positions ouvertes
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            rsi_val = all_rsi[pos.symbol][bar_idx]
            hold = bar_idx - pos.entry_bar

            # SL
            if c.low <= pos.sl_price:
                _close_position(pos, pos.sl_price, c.timestamp, bar_idx,
                                "SL", balance, positions, closed_trades)
                pnl_pct = (pos.sl_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.sl_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="SL",
                    hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            # TP
            if c.high >= pos.tp_price:
                pnl_pct = (pos.tp_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * MAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="TP",
                    hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            # RSI exit (RSI dépasse le seuil de sortie → position profitable)
            if rsi_val > cfg.rsi_exit:
                exit_price = c.close
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="RSI_EXIT",
                    hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            # Trailing stop
            new_peak = max(pos.peak_price, c.close)
            if not pos.trailing_active:
                profit_pct = (c.close - pos.entry_price) / pos.entry_price
                if profit_pct >= cfg.trail_trigger_pct:
                    pos.trailing_active = True
            if pos.trailing_active:
                new_sl = new_peak * (1 - cfg.trail_distance_pct)
                pos.sl_price = max(new_sl, pos.sl_price)
            pos.peak_price = new_peak

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
                    hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            # Max hold timeout
            if hold >= cfg.max_hold_bars:
                exit_price = c.close
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="TIMEOUT",
                    hold_bars=hold,
                ))
                positions.remove(pos)
                continue

        # Détection entrées
        if len(positions) < cfg.max_positions and balance > 10:
            capital_in_use = sum(p.size_usd for p in positions)
            available = max(0.0, balance)

            for symbol in all_candles:
                if any(p.symbol == symbol for p in positions):
                    continue
                if len(positions) >= cfg.max_positions:
                    break

                state = states[symbol]
                if bar_idx < state.cooldown_until:
                    continue

                c = all_candles[symbol][bar_idx]
                rsi_val = all_rsi[symbol][bar_idx]

                # Signal: RSI < seuil d'entrée
                if rsi_val >= cfg.rsi_entry:
                    continue

                # Filtre volume optionnel
                if cfg.vol_filter:
                    vol_ma = all_vol_ma[symbol][bar_idx]
                    if vol_ma > 0 and c.volume < cfg.vol_mult * vol_ma:
                        continue

                # Sizing
                entry_price = c.close
                sl_price = entry_price * (1 - cfg.sl_pct)
                tp_price = entry_price * (1 + cfg.tp_pct)
                sl_distance = abs(entry_price - sl_price)
                if sl_distance <= 0:
                    continue

                risk_amount = available * cfg.risk_per_trade
                size = risk_amount / sl_distance
                size_usd = size * entry_price
                max_size = available * cfg.max_position_pct
                if size_usd > max_size:
                    size_usd = max_size
                    size = size_usd / entry_price
                if size_usd < 5:
                    continue

                fee = size_usd * MAKER_FEE
                balance -= size_usd + fee
                available -= size_usd + fee
                positions.append(_OpenPos(
                    symbol=symbol, entry_price=entry_price,
                    sl_price=sl_price, tp_price=tp_price,
                    size=size, size_usd=size_usd,
                    entry_bar=bar_idx, entry_ts=c.timestamp,
                    peak_price=entry_price,
                ))
                state.cooldown_until = bar_idx + cfg.cooldown_bars

        # Equity
        pos_value = sum(
            p.size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pos_value)

    # Clôturer positions restantes
    for pos in positions:
        candles = all_candles[pos.symbol]
        last = candles[min_len - 1]
        pnl_pct = (last.close - pos.entry_price) / pos.entry_price
        pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
        balance += pos.size_usd + pnl_usd
        closed_trades.append(Trade(
            symbol=pos.symbol, side="LONG",
            entry_price=pos.entry_price, exit_price=last.close,
            size=pos.size, entry_time=pos.entry_ts, exit_time=last.timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="END",
            hold_bars=min_len - 1 - pos.entry_bar,
        ))

    return BacktestResult(
        label="", trades=closed_trades, equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  BOLLINGER BOUNCE SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_bollinger(
    all_candles: dict[str, list[Candle]],
    cfg: BollingerConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    """Simule la stratégie Bollinger Bounce."""
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]

    # Pré-calcul indicateurs
    all_bb_upper: dict[str, list[float]] = {}
    all_bb_mid: dict[str, list[float]] = {}
    all_bb_lower: dict[str, list[float]] = {}
    all_rsi: dict[str, list[float]] = {}
    all_vol_ma: dict[str, list[float]] = {}
    for symbol, candles in all_candles.items():
        closes = [c.close for c in candles]
        u, m, l = bollinger_bands(closes, cfg.bb_period, cfg.bb_std)
        all_bb_upper[symbol] = u
        all_bb_mid[symbol] = m
        all_bb_lower[symbol] = l
        all_rsi[symbol] = rsi_series(candles, cfg.rsi_period)
        vols = [c.volume for c in candles]
        all_vol_ma[symbol] = sma(vols, 20)

    # On stocke les BB targets pour la sortie dynamique
    @dataclass
    class _BBPos(_OpenPos):
        bb_exit_target: str = "mid"

    states: dict[str, _PairState] = {sym: _PairState() for sym in all_candles}
    min_len = min(len(c) for c in all_candles.values())
    start_bar = 55

    # Utiliser des listes séparées car on a besoin du exit_target
    bb_positions: list[tuple[_OpenPos, str]] = []  # (pos, exit_target)

    for bar_idx in range(start_bar, min_len):
        # Gestion positions
        for pos, exit_target in bb_positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            hold = bar_idx - pos.entry_bar

            # SL
            if c.low <= pos.sl_price:
                pnl_pct = (pos.sl_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.sl_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="SL",
                    hold_bars=hold,
                ))
                bb_positions.remove((pos, exit_target))
                continue

            # TP fixe
            if c.high >= pos.tp_price:
                pnl_pct = (pos.tp_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * MAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="TP",
                    hold_bars=hold,
                ))
                bb_positions.remove((pos, exit_target))
                continue

            # BB-based exit (dynamique)
            if exit_target == "mid":
                bb_target = all_bb_mid[pos.symbol][bar_idx]
            else:
                bb_target = all_bb_upper[pos.symbol][bar_idx]

            if c.high >= bb_target and bb_target > pos.entry_price:
                exit_price = bb_target
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * MAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason=f"BB_{exit_target.upper()}",
                    hold_bars=hold,
                ))
                bb_positions.remove((pos, exit_target))
                continue

            # Trailing
            new_peak = max(pos.peak_price, c.close)
            if not pos.trailing_active:
                profit_pct = (c.close - pos.entry_price) / pos.entry_price
                if profit_pct >= cfg.trail_trigger_pct:
                    pos.trailing_active = True
            if pos.trailing_active:
                new_sl = new_peak * (1 - cfg.trail_distance_pct)
                pos.sl_price = max(new_sl, pos.sl_price)
            pos.peak_price = new_peak

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
                    hold_bars=hold,
                ))
                bb_positions.remove((pos, exit_target))
                continue

            # Timeout
            if hold >= cfg.max_hold_bars:
                exit_price = c.close
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
                balance += pos.size_usd + pnl_usd
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.size, entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="TIMEOUT",
                    hold_bars=hold,
                ))
                bb_positions.remove((pos, exit_target))
                continue

        # Détection entrées
        active_count = len(bb_positions)
        if active_count < cfg.max_positions and balance > 10:
            available = max(0.0, balance)

            for symbol in all_candles:
                if any(p.symbol == symbol for p, _ in bb_positions):
                    continue
                if len(bb_positions) >= cfg.max_positions:
                    break

                state = states[symbol]
                if bar_idx < state.cooldown_until:
                    continue

                c = all_candles[symbol][bar_idx]
                bb_lower = all_bb_lower[symbol][bar_idx]
                rsi_val = all_rsi[symbol][bar_idx]

                # Signal: close < BB lower + RSI < seuil
                if c.close >= bb_lower:
                    continue
                if rsi_val >= cfg.rsi_max_entry:
                    continue

                # Filtre volume
                if cfg.vol_filter:
                    vol_ma = all_vol_ma[symbol][bar_idx]
                    if vol_ma > 0 and c.volume < cfg.vol_mult * vol_ma:
                        continue

                # Sizing
                entry_price = c.close
                sl_price = entry_price * (1 - cfg.sl_pct)
                tp_price = entry_price * (1 + cfg.tp_pct)
                sl_distance = abs(entry_price - sl_price)
                if sl_distance <= 0:
                    continue

                risk_amount = available * cfg.risk_per_trade
                size = risk_amount / sl_distance
                size_usd = size * entry_price
                max_size = available * cfg.max_position_pct
                if size_usd > max_size:
                    size_usd = max_size
                    size = size_usd / entry_price
                if size_usd < 5:
                    continue

                fee = size_usd * MAKER_FEE
                balance -= size_usd + fee
                available -= size_usd + fee
                pos = _OpenPos(
                    symbol=symbol, entry_price=entry_price,
                    sl_price=sl_price, tp_price=tp_price,
                    size=size, size_usd=size_usd,
                    entry_bar=bar_idx, entry_ts=c.timestamp,
                    peak_price=entry_price,
                )
                bb_positions.append((pos, cfg.exit_target))
                state.cooldown_until = bar_idx + cfg.cooldown_bars

        # Equity
        pos_value = sum(
            p.size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p, _ in bb_positions
        )
        equity_curve.append(balance + pos_value)

    # Clôturer restantes
    for pos, _ in bb_positions:
        candles = all_candles[pos.symbol]
        last = candles[min_len - 1]
        pnl_pct = (last.close - pos.entry_price) / pos.entry_price
        pnl_usd = pos.size_usd * pnl_pct - pos.size_usd * TAKER_FEE
        balance += pos.size_usd + pnl_usd
        closed_trades.append(Trade(
            symbol=pos.symbol, side="LONG",
            entry_price=pos.entry_price, exit_price=last.close,
            size=pos.size, entry_time=pos.entry_ts, exit_time=last.timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100, exit_reason="END",
            hold_bars=min_len - 1 - pos.entry_bar,
        ))

    return BacktestResult(
        label="", trades=closed_trades, equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  unused helper (dead code from copy, harmless)
# ══════════════════════════════════════════════════════════════════════════════

def _close_position(*_args, **_kwargs):
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT HELPERS
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


def print_table(results: list[dict], title: str) -> None:
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    header = f"{'Config':<20} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>10} {'Avg PnL':>9} {'Max DD%':>8} {'Final $':>10} {'R:R':>6}"
    print(header)
    print("-" * 100)
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
    if results:
        best = max(results, key=lambda r: r["pf"])
        print("-" * 100)
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


def plot_equity(results: list[BacktestResult], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#8e44ad", "#2c3e50", "#c0392b"]
    for i, r in enumerate(results):
        ax.plot(r.equity_curve, label=r.label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_title(title, fontsize=14)
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
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest RSI Bounce vs Bollinger Bounce")
    parser.add_argument("--years", type=int, default=6, help="Années de données")
    parser.add_argument("--balance", type=float, default=500, help="Capital initial ($)")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)

    for tf, interval in [("H1", "1h"), ("H4", "4h")]:
        months = args.years * 12 if tf == "H4" else min(args.years * 12, 36)
        start = end - timedelta(days=months * 30)

        print(f"\n{'═'*70}")
        print(f"  Téléchargement {tf} ({interval}) — {months // 12} ans")
        print(f"{'═'*70}")

        all_candles: dict[str, list[Candle]] = {}
        for pair in PAIRS:
            logger.info("  %s %s…", pair, tf)
            candles = download_candles(pair, start, end, interval=interval)
            all_candles[pair] = candles
            logger.info("    %s : %d bougies", pair, len(candles))

        # ── RSI Bounce ──
        rsi_variants = _define_rsi_variants(tf)
        rsi_results: list[BacktestResult] = []
        rsi_kpis: list[dict] = []
        for label, cfg, desc in rsi_variants:
            logger.info("  RSI: %s…", label)
            result = simulate_rsi_bounce(all_candles, cfg, initial_balance=args.balance)
            result.label = label
            result.config_desc = desc
            rsi_results.append(result)
            rsi_kpis.append(compute_kpis(result))

        print_table(rsi_kpis, f"RSI OVERSOLD BOUNCE — {tf} ({len(list(all_candles.values())[0])} bars, ${args.balance:.0f})")
        print_exit_breakdown(rsi_results, f"RSI {tf}")
        plot_equity(rsi_results, f"RSI Oversold Bounce ({tf})", f"rsi_bounce_{tf.lower()}_equity.png")

        # ── Bollinger Bounce ──
        bb_variants = _define_bollinger_variants(tf)
        bb_results: list[BacktestResult] = []
        bb_kpis: list[dict] = []
        for label, cfg, desc in bb_variants:
            logger.info("  BB: %s…", label)
            result = simulate_bollinger(all_candles, cfg, initial_balance=args.balance)
            result.label = label
            result.config_desc = desc
            bb_results.append(result)
            bb_kpis.append(compute_kpis(result))

        print_table(bb_kpis, f"BOLLINGER BOUNCE — {tf} ({len(list(all_candles.values())[0])} bars, ${args.balance:.0f})")
        print_exit_breakdown(bb_results, f"BB {tf}")
        plot_equity(bb_results, f"Bollinger Bounce ({tf})", f"bb_bounce_{tf.lower()}_equity.png")

    # ── Résumé global ──
    print(f"\n{'='*100}")
    print(f"  RÉSUMÉ GLOBAL — Meilleurs variants")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
