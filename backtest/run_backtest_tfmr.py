#!/usr/bin/env python3
"""
Backtest : Trend-Filtered Mean Reversion (TFMR)

Acheter les retracements dans une tendance haussière confirmée,
sortir rapidement sur rebond via TP ladder.

Règles :
  - Trend filter Daily : Price > EMA200, EMA50 > EMA200
  - Entry H4 : RSI(14) < 40, Price < EMA20, Drop ≥ 4% depuis high 5j
  - Volatilité : skip si ATR > 1.7× ATR_MA
  - TP ladder : TP1 +1% (40%), TP2 +2% (40%), TP3 +3.5% (20%)
  - SL : entry - 1.8× ATR(H4)
  - Breakeven après TP1
  - Risk : 1.5% equity, max 5 positions, max 40% exposure
  - Max daily loss : 3%

Usage :
    python -m backtest.run_backtest_tfmr
    python -m backtest.run_backtest_tfmr --years 6 --balance 500
    python -m backtest.run_backtest_tfmr --years 6 --balance 500 --variants
"""

from __future__ import annotations

import argparse
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
from src.core.indicators import ema, sma, atr_series, rsi_series, rolling_max

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "LINK-USD", "AVAX-USD", "ARB-USD", "OP-USD"]

# Revolut X fees
MAKER_FEE = 0.0
TAKER_FEE = 0.0009

# H4 bars per day = 6
H4_PER_DAY = 6


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TFMRConfig:
    """Configuration pour Trend-Filtered Mean Reversion."""
    # ── Trend filter (Daily) ──
    ema_trend_slow: int = 200         # EMA200 daily
    ema_trend_fast: int = 50          # EMA50 daily

    # ── Entry (H4) ──
    rsi_period: int = 14
    rsi_entry_max: float = 40.0       # RSI < 40
    ema_entry_period: int = 20        # EMA20 H4
    drop_pct: float = 0.04            # Drop ≥ 4% depuis high 5j
    drop_lookback_days: int = 5       # High des N derniers jours
    atr_period: int = 14

    # ── Volatility filter ──
    atr_spike_mult: float = 1.7       # Skip si ATR > 1.7× ATR_MA
    atr_ma_period: int = 20

    # ── TP ladder ──
    tp1_pct: float = 0.01             # +1%
    tp2_pct: float = 0.02             # +2%
    tp3_pct: float = 0.035            # +3.5%
    tp1_share: float = 0.40           # 40% de la position
    tp2_share: float = 0.40           # 40%
    tp3_share: float = 0.20           # 20%

    # ── Runner (TP3 alternativ) ──
    runner_mode: bool = False          # True = TP3 remplacé par trailing stop
    runner_trail_atr_mult: float = 2.0 # Trailing stop du runner = 2× ATR

    # ── SL ──
    sl_atr_mult: float = 1.8          # SL = entry - N× ATR
    sl_max_pct: float = 0.0           # 0 = pas de cap. >0 : SL = min(ATR-based, cap%)
    breakeven_after_tp1: bool = True   # SL → entry après TP1

    # ── Risk ──
    risk_per_trade: float = 0.015     # 1.5% equity
    max_positions: int = 5
    max_exposure_pct: float = 0.40    # Max 40% equity en positions
    max_daily_loss_pct: float = 0.03  # Pause si -3% journalier

    # ── Cooldown ──
    cooldown_bars: int = 3            # Cooldown par paire (12h)


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
    size_usd: float
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


@dataclass
class _OpenPos:
    """Position ouverte avec TP ladder."""
    symbol: str
    entry_price: float
    sl_price: float
    initial_size: float
    initial_size_usd: float
    remaining_size: float         # Taille restante (diminue à chaque TP)
    remaining_size_usd: float
    entry_bar: int
    entry_ts: int
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    # Runner state
    runner_active: bool = False       # True quand TP1+TP2 hit et runner_mode
    runner_trail_price: float = 0.0   # Trailing stop du runner (highest - N×ATR)
    runner_highest: float = 0.0       # Plus haut atteint depuis runner activé


@dataclass
class _PairState:
    cooldown_until: int = 0


# ══════════════════════════════════════════════════════════════════════════════
#  RESAMPLE H4 → DAILY
# ══════════════════════════════════════════════════════════════════════════════

def resample_h4_to_daily(candles: list[Candle]) -> list[Candle]:
    """Resample H4 → Daily (6 bougies H4 par jour)."""
    daily: list[Candle] = []
    for i in range(0, len(candles) - H4_PER_DAY + 1, H4_PER_DAY):
        group = candles[i:i + H4_PER_DAY]
        daily.append(Candle(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group),
        ))
    return daily


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TFMRIndicators:
    """Indicateurs pré-calculés pour une paire."""
    # H4
    rsi: list[float] = field(default_factory=list)
    ema20: list[float] = field(default_factory=list)
    atr: list[float] = field(default_factory=list)
    atr_ma: list[float] = field(default_factory=list)
    high_lookback: list[float] = field(default_factory=list)  # Rolling max high sur N jours

    # Daily (mapped to H4 index via // H4_PER_DAY)
    d_ema200: list[float] = field(default_factory=list)
    d_ema50: list[float] = field(default_factory=list)
    d_close: list[float] = field(default_factory=list)


def compute_tfmr_indicators(candles_h4: list[Candle], cfg: TFMRConfig) -> TFMRIndicators:
    """Calcule tous les indicateurs nécessaires."""
    ind = TFMRIndicators()

    # H4
    closes = [c.close for c in candles_h4]
    highs = [c.high for c in candles_h4]
    ind.rsi = rsi_series(candles_h4, cfg.rsi_period)
    ind.ema20 = ema(closes, cfg.ema_entry_period)
    ind.atr = atr_series(candles_h4, cfg.atr_period)
    ind.atr_ma = sma(ind.atr, cfg.atr_ma_period)

    # Rolling max high sur lookback (en H4 bars = jours × 6)
    lookback_bars = cfg.drop_lookback_days * H4_PER_DAY
    ind.high_lookback = rolling_max(highs, lookback_bars)

    # Daily
    daily = resample_h4_to_daily(candles_h4)
    d_closes = [c.close for c in daily]
    ind.d_ema200 = ema(d_closes, cfg.ema_trend_slow)
    ind.d_ema50 = ema(d_closes, cfg.ema_trend_fast)
    ind.d_close = d_closes

    return ind


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def simulate_tfmr(
    all_candles: dict[str, list[Candle]],
    cfg: TFMRConfig,
    initial_balance: float = 500.0,
) -> BacktestResult:
    """Simule la stratégie Trend-Filtered Mean Reversion."""
    balance = initial_balance
    positions: list[_OpenPos] = []
    closed_trades: list[Trade] = []
    equity_curve: list[float] = [balance]

    # Pré-calcul indicateurs
    logger.info("  Pré-calcul indicateurs TFMR…")
    all_ind: dict[str, TFMRIndicators] = {}
    for symbol, candles in all_candles.items():
        all_ind[symbol] = compute_tfmr_indicators(candles, cfg)
    logger.info("  Indicateurs calculés pour %d paires", len(all_candles))

    states: dict[str, _PairState] = {sym: _PairState() for sym in all_candles}
    min_len = min(len(c) for c in all_candles.values())

    # Warmup : besoin de EMA200 daily ≈ 200 jours × 6 = 1200 H4 bars
    start_bar = max(1200, 60)

    # Daily loss tracking
    current_day_ts = 0
    daily_start_equity = initial_balance
    daily_paused = False

    for bar_idx in range(start_bar, min_len):
        # ── Tracker le jour courant pour le daily loss limit ──
        sample_candle = list(all_candles.values())[0][bar_idx]
        day_ts = sample_candle.timestamp // (86400 * 1000)  # Jour unique
        if day_ts != current_day_ts:
            current_day_ts = day_ts
            # Calculer equity au début du jour
            pos_value = sum(
                p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
                for p in positions
            )
            daily_start_equity = balance + pos_value
            daily_paused = False

        # ── Check daily loss limit ──
        current_pos_value = sum(
            p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        current_equity = balance + current_pos_value
        if daily_start_equity > 0:
            daily_pnl_pct = (current_equity - daily_start_equity) / daily_start_equity
            if daily_pnl_pct <= -cfg.max_daily_loss_pct:
                daily_paused = True

        # ── Gestion positions ouvertes ──
        for pos in positions[:]:
            c = all_candles[pos.symbol][bar_idx]
            hold = bar_idx - pos.entry_bar

            # Check SL
            if c.low <= pos.sl_price:
                exit_price = pos.sl_price
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
                balance += pos.remaining_size_usd + pnl_usd
                reason = "SL" if not pos.tp1_hit else "BE"  # Breakeven si TP1 déjà touché
                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=exit_price,
                    size=pos.remaining_size, size_usd=pos.remaining_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason=reason, hold_bars=hold,
                ))
                positions.remove(pos)
                continue

            # Check TP ladder (dans l'ordre TP1 → TP2 → TP3)
            if not pos.tp1_hit and c.high >= pos.tp1_price:
                # TP1 hit : fermer 40%, breakeven le reste
                close_share = cfg.tp1_share
                close_size = pos.initial_size * close_share
                close_size_usd = pos.initial_size_usd * close_share
                pnl_pct = (pos.tp1_price - pos.entry_price) / pos.entry_price
                pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                balance += close_size_usd + pnl_usd

                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp1_price,
                    size=close_size, size_usd=close_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP1", hold_bars=hold,
                ))

                pos.tp1_hit = True
                pos.remaining_size -= close_size
                pos.remaining_size_usd -= close_size_usd

                # Breakeven
                if cfg.breakeven_after_tp1:
                    pos.sl_price = pos.entry_price

            if not pos.tp2_hit and pos.tp1_hit and c.high >= pos.tp2_price:
                # TP2 hit : fermer 40%
                close_share_of_initial = cfg.tp2_share
                close_size = pos.initial_size * close_share_of_initial
                close_size_usd = pos.initial_size_usd * close_share_of_initial
                pnl_pct = (pos.tp2_price - pos.entry_price) / pos.entry_price
                pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                balance += close_size_usd + pnl_usd

                closed_trades.append(Trade(
                    symbol=pos.symbol, side="LONG",
                    entry_price=pos.entry_price, exit_price=pos.tp2_price,
                    size=close_size, size_usd=close_size_usd,
                    entry_time=pos.entry_ts, exit_time=c.timestamp,
                    pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                    exit_reason="TP2", hold_bars=hold,
                ))

                pos.tp2_hit = True
                pos.remaining_size -= close_size
                pos.remaining_size_usd -= close_size_usd

            if pos.tp1_hit and pos.tp2_hit:
                if cfg.runner_mode:
                    # ── Runner mode : trailing stop sur le reste (20%) ──
                    if not pos.runner_active:
                        # Activer le runner
                        pos.runner_active = True
                        pos.runner_highest = c.high
                        atr_now = all_ind[pos.symbol].atr[bar_idx] if bar_idx < len(all_ind[pos.symbol].atr) else 0
                        pos.runner_trail_price = pos.runner_highest - cfg.runner_trail_atr_mult * atr_now
                        # Trail ne descend jamais sous entry (breakeven plancher)
                        if cfg.breakeven_after_tp1:
                            pos.runner_trail_price = max(pos.runner_trail_price, pos.entry_price)

                    if pos.runner_active:
                        # Update trailing
                        if c.high > pos.runner_highest:
                            pos.runner_highest = c.high
                            atr_now = all_ind[pos.symbol].atr[bar_idx] if bar_idx < len(all_ind[pos.symbol].atr) else 0
                            new_trail = pos.runner_highest - cfg.runner_trail_atr_mult * atr_now
                            if cfg.breakeven_after_tp1:
                                new_trail = max(new_trail, pos.entry_price)
                            if new_trail > pos.runner_trail_price:
                                pos.runner_trail_price = new_trail

                        # Check runner exit
                        if c.low <= pos.runner_trail_price:
                            close_size = pos.remaining_size
                            close_size_usd = pos.remaining_size_usd
                            if close_size_usd <= 0:
                                positions.remove(pos)
                                continue
                            exit_price = pos.runner_trail_price
                            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
                            pnl_usd = close_size_usd * pnl_pct - close_size_usd * TAKER_FEE
                            balance += close_size_usd + pnl_usd

                            closed_trades.append(Trade(
                                symbol=pos.symbol, side="LONG",
                                entry_price=pos.entry_price, exit_price=exit_price,
                                size=close_size, size_usd=close_size_usd,
                                entry_time=pos.entry_ts, exit_time=c.timestamp,
                                pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                                exit_reason="RUNNER", hold_bars=hold,
                            ))

                            positions.remove(pos)
                            continue
                else:
                    # ── Mode fixe : TP3 classique ──
                    if c.high >= pos.tp3_price:
                        close_size = pos.remaining_size
                        close_size_usd = pos.remaining_size_usd
                        if close_size_usd <= 0:
                            positions.remove(pos)
                            continue
                        pnl_pct = (pos.tp3_price - pos.entry_price) / pos.entry_price
                        pnl_usd = close_size_usd * pnl_pct - close_size_usd * MAKER_FEE
                        balance += close_size_usd + pnl_usd

                        closed_trades.append(Trade(
                            symbol=pos.symbol, side="LONG",
                            entry_price=pos.entry_price, exit_price=pos.tp3_price,
                            size=close_size, size_usd=close_size_usd,
                            entry_time=pos.entry_ts, exit_time=c.timestamp,
                            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
                            exit_reason="TP3", hold_bars=hold,
                        ))

                        positions.remove(pos)
                        continue

            # Cleanup : si remaining est quasi nul
            if pos.remaining_size_usd < 1:
                positions.remove(pos)
                continue

        # ── Détection de nouvelles entrées ──
        if daily_paused:
            # Daily loss limit atteint → pas de nouveaux trades
            pass
        elif len(positions) < cfg.max_positions and balance > 10:
            # Calculer l'exposition courante
            total_exposure = sum(p.remaining_size_usd for p in positions)
            current_eq = balance + total_exposure
            max_exposure = current_eq * cfg.max_exposure_pct

            for symbol in all_candles:
                if any(p.symbol == symbol for p in positions):
                    continue
                if len(positions) >= cfg.max_positions:
                    break

                state = states[symbol]
                if bar_idx < state.cooldown_until:
                    continue

                c = all_candles[symbol][bar_idx]
                ind = all_ind[symbol]

                # ── 1. Trend filter Daily ──
                d_idx = bar_idx // H4_PER_DAY
                if d_idx >= len(ind.d_ema200) or d_idx >= len(ind.d_ema50) or d_idx >= len(ind.d_close):
                    continue
                d_close = ind.d_close[d_idx]
                d_ema200 = ind.d_ema200[d_idx]
                d_ema50 = ind.d_ema50[d_idx]

                # Trend haussière : price > EMA200 ET EMA50 > EMA200
                if d_close <= d_ema200:
                    continue
                if d_ema50 <= d_ema200:
                    continue

                # ── 2. Volatility filter ──
                atr_val = ind.atr[bar_idx] if bar_idx < len(ind.atr) else 0
                atr_ma_val = ind.atr_ma[bar_idx] if bar_idx < len(ind.atr_ma) else 0
                if atr_ma_val > 0 and atr_val > cfg.atr_spike_mult * atr_ma_val:
                    continue

                # ── 3. Entry signal ──
                rsi_val = ind.rsi[bar_idx] if bar_idx < len(ind.rsi) else 50
                ema20_val = ind.ema20[bar_idx] if bar_idx < len(ind.ema20) else 0
                high_5d = ind.high_lookback[bar_idx] if bar_idx < len(ind.high_lookback) else c.high

                # RSI < seuil
                if rsi_val >= cfg.rsi_entry_max:
                    continue

                # Prix < EMA20
                if c.close >= ema20_val:
                    continue

                # Drop ≥ N% depuis high 5 jours
                if high_5d <= 0:
                    continue
                drop = (high_5d - c.close) / high_5d
                if drop < cfg.drop_pct:
                    continue

                # ── 4. Sizing ──
                if atr_val <= 0:
                    continue
                sl_distance = cfg.sl_atr_mult * atr_val
                # Cap le SL si sl_max_pct > 0
                if cfg.sl_max_pct > 0:
                    max_sl_dist = c.close * cfg.sl_max_pct
                    sl_distance = min(sl_distance, max_sl_dist)
                sl_price = c.close - sl_distance
                entry_price = c.close

                # Risk-based sizing
                equity = balance + sum(p.remaining_size_usd for p in positions)
                risk_amount = equity * cfg.risk_per_trade
                size = risk_amount / sl_distance
                size_usd = size * entry_price

                # Cap par exposition max
                remaining_exposure = max_exposure - total_exposure
                if size_usd > remaining_exposure:
                    size_usd = remaining_exposure
                    size = size_usd / entry_price if entry_price > 0 else 0

                if size_usd < 5:
                    continue

                # TP prices
                tp1_price = entry_price * (1 + cfg.tp1_pct)
                tp2_price = entry_price * (1 + cfg.tp2_pct)
                tp3_price = entry_price * (1 + cfg.tp3_pct)

                # Execute
                fee = size_usd * MAKER_FEE
                balance -= size_usd + fee
                total_exposure += size_usd
                positions.append(_OpenPos(
                    symbol=symbol, entry_price=entry_price,
                    sl_price=sl_price,
                    initial_size=size, initial_size_usd=size_usd,
                    remaining_size=size, remaining_size_usd=size_usd,
                    entry_bar=bar_idx, entry_ts=c.timestamp,
                    tp1_price=tp1_price, tp2_price=tp2_price, tp3_price=tp3_price,
                ))
                state.cooldown_until = bar_idx + cfg.cooldown_bars

        # ── Equity tracking ──
        pos_value = sum(
            p.remaining_size * all_candles[p.symbol][min(bar_idx, len(all_candles[p.symbol]) - 1)].close
            for p in positions
        )
        equity_curve.append(balance + pos_value)

    # Clôturer positions restantes
    for pos in positions:
        candles = all_candles[pos.symbol]
        last = candles[min_len - 1]
        pnl_pct = (last.close - pos.entry_price) / pos.entry_price
        pnl_usd = pos.remaining_size_usd * pnl_pct - pos.remaining_size_usd * TAKER_FEE
        balance += pos.remaining_size_usd + pnl_usd
        closed_trades.append(Trade(
            symbol=pos.symbol, side="LONG",
            entry_price=pos.entry_price, exit_price=last.close,
            size=pos.remaining_size, size_usd=pos.remaining_size_usd,
            entry_time=pos.entry_ts, exit_time=last.timestamp,
            pnl_usd=pnl_usd, pnl_pct=pnl_pct * 100,
            exit_reason="END", hold_bars=min_len - 1 - pos.entry_bar,
        ))

    return BacktestResult(
        label="", trades=closed_trades, equity_curve=equity_curve,
        initial_balance=initial_balance,
        final_equity=equity_curve[-1] if equity_curve else initial_balance,
        config_desc={},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def _define_variants(run_variants: bool = False) -> list[tuple[str, TFMRConfig, dict, list[str] | None]]:
    """Définit les variantes à tester. Retourne (label, config, desc, pairs_override)."""
    configs = []

    # ── V3 : Runner + SL cappé (config demandée) — 8 paires ──
    _runner_cfg = TFMRConfig(
        rsi_entry_max=35.0,
        drop_pct=0.04,
        tp1_pct=0.012,           # TP1 +1.2%
        tp2_pct=0.03,            # TP2 +3%
        tp1_share=0.40,
        tp2_share=0.40,
        tp3_share=0.20,          # Runner 20%
        runner_mode=True,
        runner_trail_atr_mult=2.0,
        sl_atr_mult=1.2,         # SL = 1.2×ATR
        sl_max_pct=0.025,        # Cap SL à 2.5%
        breakeven_after_tp1=True,
    )
    _runner_desc = {
        "entry": "RSI<35 + Drop≥4%",
        "sl": "min(1.2×ATR, 2.5%)",
        "tp": "1.2%(40%) + 3%(40%) + Runner 2ATR(20%)",
    }
    configs.append(("TFMR_V3_8PAIRS", _runner_cfg, _runner_desc, None))

    # ── V3 : Runner — 4 bonnes paires seulement ──
    configs.append(("TFMR_V3_4PAIRS", _runner_cfg, {**_runner_desc, "pairs": "BTC,ETH,LINK,AVAX"},
                    ["BTC-USD", "ETH-USD", "LINK-USD", "AVAX-USD"]))

    if run_variants:
        # ── V1 : RSI plus strict (< 35) ──
        configs.append(("TFMR_RSI35", TFMRConfig(
            rsi_entry_max=35.0,
        ), {
            "entry": "RSI<35 + P<EMA20 + Drop≥4%",
            "sl": "1.8×ATR", "tp": "1/2/3.5%",
        }, None))

        # ── V2 : Drop 3% (plus de trades) ──
        configs.append(("TFMR_DROP3", TFMRConfig(
            drop_pct=0.03,
        ), {
            "entry": "RSI<40 + P<EMA20 + Drop≥3%",
            "sl": "1.8×ATR", "tp": "1/2/3.5%",
        }, None))

        # ── V3 : TP plus large (2/4/7%) ──
        configs.append(("TFMR_WIDE_TP", TFMRConfig(
            tp1_pct=0.02,
            tp2_pct=0.04,
            tp3_pct=0.07,
        ), {
            "entry": "RSI<40 + P<EMA20 + Drop≥4%",
            "sl": "1.8×ATR", "tp": "2/4/7% ladder",
        }, None))

        # ── V4 : SL serré (1.2× ATR) ──
        configs.append(("TFMR_TIGHT_SL", TFMRConfig(
            sl_atr_mult=1.2,
        ), {
            "entry": "RSI<40 + P<EMA20 + Drop≥4%",
            "sl": "1.2×ATR", "tp": "1/2/3.5%",
        }, None))

        # ── V5 : SL large (2.5× ATR), TP large ──
        configs.append(("TFMR_WIDE_SL", TFMRConfig(
            sl_atr_mult=2.5,
            tp1_pct=0.015,
            tp2_pct=0.03,
            tp3_pct=0.05,
        ), {
            "entry": "RSI<40 + P<EMA20 + Drop≥4%",
            "sl": "2.5×ATR", "tp": "1.5/3/5%",
        }, None))

        # ── V6 : Risk 2.5% (plus agressif) ──
        configs.append(("TFMR_RISK25", TFMRConfig(
            risk_per_trade=0.025,
            max_exposure_pct=0.50,
        ), {
            "entry": "RSI<40 + P<EMA20 + Drop≥4%",
            "sl": "1.8×ATR", "tp": "1/2/3.5%",
            "risk": "2.5%", "max_exp": "50%",
        }, None))

        # ── V7 : Sans breakeven (laisser courir) ──
        configs.append(("TFMR_NO_BE", TFMRConfig(
            breakeven_after_tp1=False,
        ), {
            "entry": "RSI<40 + P<EMA20 + Drop≥4%",
            "sl": "1.8×ATR", "tp": "1/2/3.5%", "BE": "non",
        }, None))

        # ── V8 : Sans daily loss limit ──
        configs.append(("TFMR_NO_DLL", TFMRConfig(
            max_daily_loss_pct=1.0,  # 100% = désactivé
        ), {
            "entry": "RSI<40 + P<EMA20 + Drop≥4%",
            "sl": "1.8×ATR", "tp": "1/2/3.5%", "DLL": "off",
        }, None))

    return configs


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
            "avg_win": 0, "avg_loss": 0, "avg_hold": 0,
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
    avg_hold = sum(t.hold_bars for t in trades) / n

    return {
        "label": result.label, "trades": n,
        "win_rate": len(wins) / n * 100,
        "pf": gross_profit / gross_loss if gross_loss > 0 else 999,
        "pnl": sum(t.pnl_usd for t in trades),
        "avg_pnl": sum(t.pnl_usd for t in trades) / n,
        "max_dd": max_dd * 100,
        "final_eq": result.final_equity,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "avg_hold": avg_hold,
    }


def print_table(results: list[dict], title: str) -> None:
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    header = (f"{'Config':<20} {'Trades':>6} {'WR%':>7} {'PF':>7} {'PnL $':>10} "
              f"{'Avg PnL':>9} {'Max DD%':>8} {'Final $':>10} {'R:R':>6} {'Avg Hold':>9}")
    print(header)
    print("-" * 110)
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
            f"{rr:>5.2f} "
            f"{kpi['avg_hold']:>8.1f}b"
        )
        print(line)
    if results:
        best = max(results, key=lambda r: r["pf"])
        print("-" * 110)
        print(f"  Meilleur PF : {best['label']} (PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$, WR {best['win_rate']:.1f}%)")
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


def print_per_pair(results: list[BacktestResult]) -> None:
    """Affiche les stats par paire pour le meilleur variant."""
    best = max(results, key=lambda r: r.final_equity)
    print(f"\n{'='*70}")
    print(f"  {best.label} — Stats par paire")
    print(f"{'='*70}")
    pair_stats: dict[str, list[Trade]] = {}
    for t in best.trades:
        pair_stats.setdefault(t.symbol, []).append(t)
    print(f"  {'Paire':<12} {'Trades':>6} {'WR%':>7} {'PnL $':>10}")
    print(f"  {'-'*40}")
    for symbol in sorted(pair_stats.keys()):
        trades = pair_stats[symbol]
        wins = [t for t in trades if t.pnl_usd > 0]
        wr = len(wins) / len(trades) * 100 if trades else 0
        pnl = sum(t.pnl_usd for t in trades)
        print(f"  {symbol:<12} {len(trades):>6} {wr:>6.1f}% {pnl:>+9.2f}$")
    print()


def plot_equity(results: list[BacktestResult], title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#8e44ad", "#2c3e50", "#c0392b"]
    for i, r in enumerate(results):
        ax.plot(r.equity_curve, label=r.label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Barres H4")
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
    parser = argparse.ArgumentParser(description="Backtest Trend-Filtered Mean Reversion")
    parser.add_argument("--years", type=int, default=6, help="Années de données H4")
    parser.add_argument("--balance", type=float, default=500, help="Capital initial ($)")
    parser.add_argument("--variants", action="store_true", help="Tester aussi les variantes")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)

    print(f"\n{'═'*70}")
    print(f"  TREND-FILTERED MEAN REVERSION — Téléchargement H4 ({args.years} ans)")
    print(f"  Paires : {', '.join(PAIRS)}")
    print(f"═'*70")

    all_candles: dict[str, list[Candle]] = {}
    for pair in PAIRS:
        logger.info("  %s H4…", pair)
        candles = download_candles(pair, start, end, interval="4h")
        all_candles[pair] = candles
        logger.info("    %s : %d bougies H4 (%d jours)", pair, len(candles), len(candles) // 6)

    configs = _define_variants(run_variants=args.variants)
    results: list[BacktestResult] = []
    kpis_list: list[dict] = []

    for label, cfg, desc, pairs_override in configs:
        logger.info("  %s…", label)
        # Filtrer les paires si override
        candles_for_run = {k: v for k, v in all_candles.items() if k in pairs_override} if pairs_override else all_candles
        if not candles_for_run:
            logger.warning("  Aucune paire disponible pour %s, skip", label)
            continue
        result = simulate_tfmr(candles_for_run, cfg, initial_balance=args.balance)
        result.label = label
        result.config_desc = desc
        results.append(result)
        kpis = compute_kpis(result)
        kpis_list.append(kpis)

    n_bars = min(len(c) for c in all_candles.values())
    print_table(kpis_list, f"TREND-FILTERED MEAN REVERSION — H4 ({n_bars} bars, {args.years} ans, ${args.balance:.0f})")
    print_exit_breakdown(results, "TFMR")
    print_per_pair(results)
    plot_equity(results, "Trend-Filtered Mean Reversion", "tfmr_equity.png")

    # Résumé
    best = max(kpis_list, key=lambda k: k["pf"])
    print(f"\n{'═'*70}")
    if best["pf"] >= 1.5:
        print(f"  ✅ PROMETTEUR : {best['label']} — PF {best['pf']:.2f}, WR {best['win_rate']:.1f}%, PnL {best['pnl']:+.2f}$")
        print(f"     DD {best['max_dd']:.1f}% — Candidat pour déploiement sur Revolut X")
    elif best["pf"] >= 1.0:
        print(f"  ⚠️  MARGINAL : {best['label']} — PF {best['pf']:.2f}, PnL {best['pnl']:+.2f}$")
        print(f"     Tester --variants pour optimiser")
    else:
        print(f"  ❌ NON RENTABLE : Meilleur PF = {best['pf']:.2f}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
