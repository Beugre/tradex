#!/usr/bin/env python3
"""
Backtest Trail Range — Mean-Reversion avec A/B Comparison.

Reproduit fidèlement la logique de production :
  - Swing detection (lookback=3) → Dow Theory → NEUTRAL → range
  - BUY au bas du range (entry = range_low × (1 + buffer))
  - SL = range_low × (1 - sl_buffer)
  - TP = range_mid (ou variantes)
  - Trail@TP : quand prix → TP, swap vers nouveau OCO
  - Cooldown post-SL (3 bougies H4 = 12h)
  - Sortie forcée si tendance confirmée (BULLISH/BEARISH)
  - Fees Binance : 0.1% maker + 0.1% taker

Usage:
    python -m backtest.run_backtest_range
    python -m backtest.run_backtest_range --pairs BTC-USD,ETH-USD,SOL-USD
    python -m backtest.run_backtest_range --years 3 --balance 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtest.data_loader import download_candles

# Import du core pur (logique sans I/O)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import classify_swings, determine_trend, check_trend_invalidation
from src.core.models import (
    Candle,
    RangeState,
    SwingLevel,
    SwingPoint,
    TrendDirection,
    TrendState,
    OrderSide,
)

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# ── Fees Binance ───────────────────────────────────────────────────────────────
MAKER_FEE = 0.001   # 0.1%
TAKER_FEE = 0.001   # 0.1%

# ── Paires par défaut (liquides, stables) ──────────────────────────────────────
DEFAULT_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD",
    "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
    "LTC-USD", "INJ-USD",
]


# ── Config paramétrable ───────────────────────────────────────────────────────


@dataclass
class RangeConfig:
    """Configuration d'une variante de la stratégie Range."""
    name: str = "CURRENT"
    
    # Swing detection
    swing_lookback: int = 3
    
    # Range filtering
    range_width_min: float = 0.02       # 2% minimum range width
    
    # Entry
    entry_buffer_pct: float = 0.002     # 0.2% entry zone
    
    # SL
    sl_buffer_pct: float = 0.003        # 0.3% SL buffer beyond range edge
    
    # TP mode
    tp_mode: str = "mid"                # "mid" = range_mid, "opposite" = range_high/low
    tp_ratio: float = 0.5              # 0.5 = mid, 0.75 = 3/4 du range, 1.0 = opposite edge
    
    # Trail@TP
    trail_enabled: bool = True
    trail_step_pct: float = 0.01        # +1% par palier
    trail_sl_lock_pct: float = 0.02     # SL = 98% du TP atteint
    trail_swap_pct: float = 0.005       # swap quand < 0.5% du TP
    
    # Cooldown
    cooldown_bars: int = 3              # 3 × 4h = 12h
    
    # Max positions
    max_positions: int = 3
    
    # Risk
    risk_pct: float = 0.02             # 2% risk per trade
    max_position_pct: float = 0.30     # max 30% du capital par trade
    
    # RSI filter (0 = disabled)
    rsi_buy_max: float = 0.0           # ex: 30.0 → only buy when RSI < 30
    rsi_sell_min: float = 0.0          # ex: 70.0 → only sell when RSI > 70
    rsi_period: int = 14
    
    # Volume filter
    volume_confirm: bool = False        # require volume > MA(20)
    
    # Candle confirmation (attendre un rebond)
    candle_confirm: bool = False        # require bullish/bearish candle before entry
    
    # ATR-based SL instead of fixed buffer
    atr_sl_enabled: bool = False
    atr_sl_mult: float = 1.5
    atr_period: int = 14
    
    # Step-trail (range-relative discrete steps)
    # Quand prix atteint TP, au lieu de fermer :
    #   Step 1 : SL → initial_sl_ratio du range, TP → initial_tp_ratio du range
    #   Step N : SL et TP décalent de +step_size chacun
    step_trail_enabled: bool = False
    step_trail_initial_sl_ratio: float = 0.50   # 1er step: SL au mid (50%)
    step_trail_initial_tp_ratio: float = 0.80   # 1er step: TP à 80%
    step_trail_step_size: float = 0.05          # décalage de 5% par palier
    
    # Forced exit on trend change
    exit_on_trend_change: bool = True


# ── Trade data ─────────────────────────────────────────────────────────────────


@dataclass
class RangeTrade:
    """Un trade Range complet."""
    symbol: str
    side: str               # "BUY" / "SELL"
    entry_bar: int
    entry_price: float
    entry_ts: int
    sl_price: float
    tp_price: float
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    size: float = 0.0
    fees: float = 0.0
    range_width_pct: float = 0.0
    trailing_steps: int = 0
    duration_hours: float = 0.0


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ── Indicators ─────────────────────────────────────────────────────────────────


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """Calcule le RSI sur la série de closes."""
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


def atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    """Calcule l'ATR (Average True Range) sur les bougies."""
    n = len(candles)
    atr = [0.0] * n
    if n < period + 1:
        return atr
    
    tr = [0.0] * n
    for i in range(1, n):
        h = candles[i].high
        l = candles[i].low
        c_prev = candles[i - 1].close
        tr[i] = max(h - l, abs(h - c_prev), abs(l - c_prev))
    
    # Simple average for first ATR
    atr[period] = sum(tr[1:period + 1]) / period
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def sma_series(values: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    n = len(values)
    sma = [0.0] * n
    if n < period:
        return sma
    running = sum(values[:period])
    sma[period - 1] = running / period
    for i in range(period, n):
        running += values[i] - values[i - period]
        sma[i] = running / period
    return sma


# ── Simulator ──────────────────────────────────────────────────────────────────


def simulate_range(
    all_candles: dict[str, list[Candle]],
    cfg: RangeConfig,
    initial_balance: float = 200.0,
) -> tuple[list[RangeTrade], list[EquityPoint]]:
    """
    Simule la stratégie Range sur toutes les paires en parallèle (multi-pair).
    
    Reproduit fidèlement la logique de bot_binance.py :
    1. Sur chaque bougie H4, détecter les swings et la tendance
    2. Si NEUTRAL → construire le range
    3. Au close de la bougie, si prix dans buy/sell zone → signal pending
    4. Au open de la bougie suivante, exécuter le signal
    5. Gérer SL/TP/Trail/sortie forcée
    
    Returns: (trades, equity_curve)
    """
    trades: list[RangeTrade] = []
    equity = initial_balance
    equity_curve: list[EquityPoint] = []
    
    # Pre-compute indicators per pair
    pair_data: dict[str, dict] = {}
    min_bars = float("inf")
    
    for pair, candles in all_candles.items():
        closes = [c.close for c in candles]
        pd = {"candles": candles, "closes": closes}
        
        if cfg.rsi_buy_max > 0 or cfg.rsi_sell_min > 0:
            pd["rsi"] = rsi_series(closes, cfg.rsi_period)
        
        if cfg.atr_sl_enabled:
            pd["atr"] = atr_series(candles, cfg.atr_period)
        
        if cfg.volume_confirm:
            volumes = [c.volume for c in candles]
            pd["vol_ma"] = sma_series(volumes, 20)
            pd["volumes"] = volumes
        
        pair_data[pair] = pd
        min_bars = min(min_bars, len(candles))
    
    if min_bars < 50:
        logger.warning("Pas assez de bougies (min=%d)", min_bars)
        return trades, equity_curve
    
    # State per pair
    trends: dict[str, Optional[TrendState]] = {}
    ranges: dict[str, Optional[RangeState]] = {}
    cooldowns: dict[str, int] = {}  # bar index until which cooldown is active
    
    # Active positions
    positions: dict[str, dict] = {}  # symbol → position dict
    pending_signals: dict[str, dict] = {}
    
    # Walk candle by candle (all pairs synchronously)
    warmup = max(cfg.swing_lookback * 2 + 5, 30)  # need enough bars for swings
    
    for bar in range(warmup, int(min_bars)):
        # ── 1. Update trends and ranges for each pair ──
        pending_signals.clear()
        
        for pair, pd in pair_data.items():
            candles = pd["candles"]
            
            # Detect swings on candles up to current bar
            window = candles[max(0, bar - 200):bar + 1]  # last ~200 bars
            swings = detect_swings(window, cfg.swing_lookback)
            
            if len(swings) < 4:
                trends[pair] = None
                continue
            
            # Classify and determine trend
            trend = determine_trend(swings, pair)
            
            # Check trend invalidation with current close
            current_price = candles[bar].close
            if trends.get(pair):
                old_dir = trends[pair].direction
                trend = check_trend_invalidation(trend, current_price)
                
                # If trend changed from/to NEUTRAL, handle position exits
                if old_dir != trend.direction:
                    # Forced exit on confirmed trend
                    if (
                        cfg.exit_on_trend_change
                        and pair in positions
                        and trend.direction in (TrendDirection.BULLISH, TrendDirection.BEARISH)
                    ):
                        pos = positions[pair]
                        exit_price = current_price
                        pnl = _compute_pnl(pos, exit_price)
                        fees = (pos["entry_price"] * pos["size"] * TAKER_FEE +
                                exit_price * pos["size"] * TAKER_FEE)
                        net_pnl = pnl - fees
                        
                        trade = _close_trade(
                            pos, bar, candles[bar].timestamp, exit_price,
                            "trend_change", net_pnl, pnl / (pos["entry_price"] * pos["size"]) if pos["size"] > 0 else 0,
                            fees,
                        )
                        trades.append(trade)
                        equity += net_pnl
                        del positions[pair]
            
            trends[pair] = trend
            
            # Build range if NEUTRAL
            if trend.direction == TrendDirection.NEUTRAL:
                if trend.last_high and trend.last_low:
                    rh = trend.last_high.price
                    rl = trend.last_low.price
                    if rh > rl:
                        width_pct = (rh - rl) / rl
                        if width_pct >= cfg.range_width_min:
                            ranges[pair] = RangeState(
                                symbol=pair,
                                range_high=rh,
                                range_low=rl,
                            )
                        else:
                            ranges[pair] = None
                    else:
                        ranges[pair] = None
                else:
                    ranges[pair] = None
            else:
                ranges[pair] = None
            
            # ── 2. Generate pending signals at candle close ──
            if pair in positions:
                continue
            
            rs = ranges.get(pair)
            if rs is None:
                continue
            
            # Check cooldown
            cd = cooldowns.get(pair, 0)
            if bar < cd:
                continue
            
            last_close = candles[bar].close
            
            # BUY zone
            buy_zone = rs.range_low * (1 + cfg.entry_buffer_pct)
            if last_close <= buy_zone:
                # Anti-breakout
                if cfg.atr_sl_enabled:
                    atr_val = pd.get("atr", [0.0] * (bar + 1))[bar]
                    sl_price = rs.range_low - cfg.atr_sl_mult * atr_val
                else:
                    sl_price = rs.range_low * (1 - cfg.sl_buffer_pct)
                
                if sl_price >= last_close:
                    continue
                
                # TP calculation
                tp_price = _calc_tp(rs, "BUY", cfg)
                
                # RSI filter
                if cfg.rsi_buy_max > 0:
                    rsi_val = pd.get("rsi", [50.0] * (bar + 1))[bar]
                    if rsi_val > cfg.rsi_buy_max:
                        continue
                
                # Volume filter
                if cfg.volume_confirm:
                    vol = pd.get("volumes", [0] * (bar + 1))[bar]
                    vol_ma = pd.get("vol_ma", [0] * (bar + 1))[bar]
                    if vol_ma > 0 and vol < vol_ma:
                        continue
                
                # Candle confirmation (bullish close for BUY)
                if cfg.candle_confirm:
                    c = candles[bar]
                    if c.close <= c.open:  # bearish candle → skip
                        continue
                
                pending_signals[pair] = {
                    "side": "BUY",
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "range_width_pct": (rs.range_high - rs.range_low) / rs.range_low,
                    "range_high": rs.range_high,
                    "range_low": rs.range_low,
                }
            
            # SELL zone (spot = need to already hold → BUY only in backtest for simplicity)
            # In production, SELL requires holding the asset. Skip for backtest.
        
        # ── 3. Execute pending signals at next bar open ──
        if bar + 1 < int(min_bars):
            # Capital déjà bloqué dans les positions ouvertes
            capital_in_use = sum(
                p["entry_price"] * p["size"] for p in positions.values()
            )
            available_capital = max(0, equity - capital_in_use)
            
            for pair, signal in pending_signals.items():
                if pair in positions:
                    continue
                
                open_count = len(positions)
                if open_count >= cfg.max_positions:
                    break  # max positions reached
                
                if available_capital < 10:
                    break  # plus de capital disponible
                
                next_candle = pair_data[pair]["candles"][bar + 1]
                entry_price = next_candle.open
                
                # Skip if price gapped below SL
                if signal["sl_price"] >= entry_price:
                    continue
                
                # Position sizing — basé sur le capital DISPONIBLE
                sl_dist = abs(entry_price - signal["sl_price"])
                if sl_dist <= 0:
                    continue
                
                risk_amount = available_capital * cfg.risk_pct
                size = risk_amount / sl_dist
                cost = size * entry_price
                max_cost = available_capital * cfg.max_position_pct
                if cost > max_cost:
                    size = max_cost / entry_price
                    cost = max_cost
                
                if cost < 10:  # min notional
                    continue
                
                if cost > available_capital:
                    size = available_capital / entry_price
                    cost = available_capital
                
                entry_fee = cost * TAKER_FEE
                
                # Recalc TP with actual entry
                rs = ranges.get(pair)
                if rs:
                    tp_price = _calc_tp(rs, signal["side"], cfg)
                else:
                    tp_price = signal["tp_price"]
                
                positions[pair] = {
                    "symbol": pair,
                    "side": signal["side"],
                    "entry_bar": bar + 1,
                    "entry_ts": next_candle.timestamp,
                    "entry_price": entry_price,
                    "sl_price": signal["sl_price"],
                    "tp_price": tp_price,
                    "size": size,
                    "entry_fee": entry_fee,
                    "trailing_steps": 0,
                    "range_width_pct": signal["range_width_pct"],
                    "range_high": signal.get("range_high", 0),
                    "range_low": signal.get("range_low", 0),
                }
                
                # Décompter le capital bloqué
                available_capital -= cost
        
        # ── 4. Manage open positions ──
        closed = []
        for pair, pos in positions.items():
            if bar <= pos["entry_bar"]:
                continue
            
            candle = pair_data[pair]["candles"][bar]
            
            # Check SL hit (intrabar: use low for BUY)
            if pos["side"] == "BUY":
                if candle.low <= pos["sl_price"]:
                    exit_price = pos["sl_price"]
                    pnl = _compute_pnl(pos, exit_price)
                    fees = pos["entry_fee"] + exit_price * pos["size"] * TAKER_FEE
                    net_pnl = pnl - fees
                    
                    reason = "SL_hit" if pos["trailing_steps"] == 0 else f"trail_SL_step{pos['trailing_steps']}"
                    trade = _close_trade(
                        pos, bar, candle.timestamp, exit_price,
                        reason, net_pnl,
                        net_pnl / (pos["entry_price"] * pos["size"]),
                        fees,
                    )
                    trades.append(trade)
                    equity += net_pnl
                    cooldowns[pair] = bar + cfg.cooldown_bars
                    closed.append(pair)
                    continue
            
            # Check TP proximity / trail
            if pos["side"] == "BUY":
                if candle.high >= pos["tp_price"]:
                    if cfg.step_trail_enabled:
                        # Step-trail: SL et TP par paliers discrets dans le range
                        rng_low = pos["range_low"]
                        rng_high = pos["range_high"]
                        rng_width = rng_high - rng_low
                        
                        step = pos["trailing_steps"]
                        if step == 0:
                            # Premier step : SL au mid, TP à 80%
                            sl_ratio = cfg.step_trail_initial_sl_ratio
                            tp_ratio = cfg.step_trail_initial_tp_ratio
                        else:
                            # Steps suivants : décaler de +5% chacun
                            sl_ratio = cfg.step_trail_initial_sl_ratio + step * cfg.step_trail_step_size
                            tp_ratio = cfg.step_trail_initial_tp_ratio + step * cfg.step_trail_step_size
                        
                        pos["sl_price"] = rng_low + rng_width * sl_ratio
                        pos["tp_price"] = rng_low + rng_width * tp_ratio
                        pos["trailing_steps"] += 1
                        
                        # Prix a pu traverser plusieurs paliers dans la même bougie
                        while candle.high >= pos["tp_price"]:
                            step = pos["trailing_steps"]
                            sl_ratio = cfg.step_trail_initial_sl_ratio + step * cfg.step_trail_step_size
                            tp_ratio = cfg.step_trail_initial_tp_ratio + step * cfg.step_trail_step_size
                            pos["sl_price"] = rng_low + rng_width * sl_ratio
                            pos["tp_price"] = rng_low + rng_width * tp_ratio
                            pos["trailing_steps"] += 1
                            # Sécurité : si TP dépasse range_high × 1.5, on ferme
                            if tp_ratio > 1.5:
                                break
                        
                        # Vérifier si le SL step-trail est aussi touché dans la même bougie
                        if candle.low <= pos["sl_price"]:
                            exit_price = pos["sl_price"]
                            pnl = _compute_pnl(pos, exit_price)
                            fees = pos["entry_fee"] + exit_price * pos["size"] * TAKER_FEE
                            net_pnl = pnl - fees
                            trade = _close_trade(
                                pos, bar, candle.timestamp, exit_price,
                                f"step_trail_SL_s{pos['trailing_steps']}",
                                net_pnl,
                                net_pnl / (pos["entry_price"] * pos["size"]),
                                fees,
                            )
                            trades.append(trade)
                            equity += net_pnl
                            closed.append(pair)
                    
                    elif cfg.trail_enabled:
                        # Trail classique: update SL and TP
                        old_tp = pos["tp_price"]
                        pos["tp_price"] = old_tp * (1 + cfg.trail_step_pct)
                        pos["sl_price"] = old_tp * (1 - cfg.trail_sl_lock_pct)
                        pos["trailing_steps"] += 1
                        # Check if TP also hit in same candle after trail
                        # (price surged through multiple levels)
                        while candle.high >= pos["tp_price"]:
                            old_tp = pos["tp_price"]
                            pos["tp_price"] = old_tp * (1 + cfg.trail_step_pct)
                            pos["sl_price"] = old_tp * (1 - cfg.trail_sl_lock_pct)
                            pos["trailing_steps"] += 1
                        # After trailing, check if SL is also hit in same candle
                        if candle.low <= pos["sl_price"]:
                            exit_price = pos["sl_price"]
                            pnl = _compute_pnl(pos, exit_price)
                            fees = pos["entry_fee"] + exit_price * pos["size"] * TAKER_FEE
                            net_pnl = pnl - fees
                            trade = _close_trade(
                                pos, bar, candle.timestamp, exit_price,
                                f"trail_SL_step{pos['trailing_steps']}",
                                net_pnl,
                                net_pnl / (pos["entry_price"] * pos["size"]),
                                fees,
                            )
                            trades.append(trade)
                            equity += net_pnl
                            closed.append(pair)
                    else:
                        # No trail → close at TP
                        exit_price = pos["tp_price"]
                        pnl = _compute_pnl(pos, exit_price)
                        fees = pos["entry_fee"] + exit_price * pos["size"] * MAKER_FEE
                        net_pnl = pnl - fees
                        trade = _close_trade(
                            pos, bar, candle.timestamp, exit_price,
                            "TP_hit", net_pnl,
                            net_pnl / (pos["entry_price"] * pos["size"]),
                            fees,
                        )
                        trades.append(trade)
                        equity += net_pnl
                        closed.append(pair)
        
        for p in closed:
            positions.pop(p, None)
        
        # ── 5. Record equity ──
        # Mark-to-market unrealized
        unrealized = 0.0
        for pair, pos in positions.items():
            candle = pair_data[pair]["candles"][bar]
            unrealized += _compute_pnl(pos, candle.close)
        
        ts = list(pair_data.values())[0]["candles"][bar].timestamp if pair_data else 0
        equity_curve.append(EquityPoint(ts=ts, equity=equity + unrealized))
    
    # Close remaining positions at last close
    for pair, pos in list(positions.items()):
        last_candle = pair_data[pair]["candles"][int(min_bars) - 1]
        exit_price = last_candle.close
        pnl = _compute_pnl(pos, exit_price)
        fees = pos["entry_fee"] + exit_price * pos["size"] * TAKER_FEE
        net_pnl = pnl - fees
        trade = _close_trade(
            pos, int(min_bars) - 1, last_candle.timestamp, exit_price,
            "end_of_data", net_pnl,
            net_pnl / (pos["entry_price"] * pos["size"]) if pos["size"] > 0 else 0,
            fees,
        )
        trades.append(trade)
        equity += net_pnl
    
    return trades, equity_curve


# ── Helpers ────────────────────────────────────────────────────────────────────


def _calc_tp(rs: RangeState, side: str, cfg: RangeConfig) -> float:
    """Calcule le TP selon le mode configuré."""
    if side == "BUY":
        # TP entre range_low et range_high
        return rs.range_low + (rs.range_high - rs.range_low) * cfg.tp_ratio
    else:
        # SELL: TP entre range_high et range_low
        return rs.range_high - (rs.range_high - rs.range_low) * cfg.tp_ratio


def _compute_pnl(pos: dict, exit_price: float) -> float:
    """PnL brut (sans fees)."""
    if pos["side"] == "BUY":
        return (exit_price - pos["entry_price"]) * pos["size"]
    else:
        return (pos["entry_price"] - exit_price) * pos["size"]


def _close_trade(
    pos: dict, bar: int, ts: int, exit_price: float,
    reason: str, net_pnl: float, pnl_pct: float, fees: float,
) -> RangeTrade:
    """Crée un RangeTrade à partir d'une position fermée."""
    entry_ts = pos["entry_ts"]
    duration_h = (ts - entry_ts) / 3600 / 1000 if ts > 1e12 else (ts - entry_ts) / 3600
    return RangeTrade(
        symbol=pos["symbol"],
        side=pos["side"],
        entry_bar=pos["entry_bar"],
        entry_price=pos["entry_price"],
        entry_ts=entry_ts,
        sl_price=pos["sl_price"],
        tp_price=pos["tp_price"],
        exit_bar=bar,
        exit_price=exit_price,
        exit_ts=ts,
        exit_reason=reason,
        pnl_usd=net_pnl,
        pnl_pct=pnl_pct,
        size=pos["size"],
        fees=fees,
        range_width_pct=pos.get("range_width_pct", 0.0),
        trailing_steps=pos.get("trailing_steps", 0),
        duration_hours=duration_h,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_range_metrics(
    trades: list[RangeTrade],
    equity_curve: list[EquityPoint],
    initial_balance: float,
) -> dict:
    """Calcule les métriques de performance."""
    n = len(trades)
    final_eq = equity_curve[-1].equity if equity_curve else initial_balance
    
    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0, "pf": 0, "total_pnl": 0,
            "avg_pnl": 0, "max_dd": 0, "final_equity": final_eq,
            "avg_duration_h": 0, "avg_range_width": 0,
        }
    
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    
    gross_profit = sum(t.pnl_usd for t in wins) or 0
    gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
    pf = gross_profit / gross_loss
    
    total_pnl = sum(t.pnl_usd for t in trades)
    avg_pnl = total_pnl / n
    
    # Max drawdown
    peak = initial_balance
    max_dd = 0.0
    for pt in equity_curve:
        peak = max(peak, pt.equity)
        dd = (pt.equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)
    
    # Exit reasons breakdown
    reasons: dict[str, int] = {}
    for t in trades:
        r = t.exit_reason
        reasons[r] = reasons.get(r, 0) + 1
    
    avg_duration = sum(t.duration_hours for t in trades) / n
    avg_width = sum(t.range_width_pct for t in trades) / n
    avg_trail = sum(t.trailing_steps for t in trades) / n
    
    return {
        "n_trades": n,
        "win_rate": win_rate,
        "pf": pf,
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "max_dd": max_dd,
        "final_equity": final_eq,
        "avg_duration_h": avg_duration,
        "avg_range_width": avg_width,
        "avg_trail_steps": avg_trail,
        "exit_reasons": reasons,
        "total_fees": sum(t.fees for t in trades),
    }


# ── Variants ───────────────────────────────────────────────────────────────────


def get_variants() -> list[RangeConfig]:
    """Retourne les variantes à comparer."""
    # Base commune step-trail (déployé)
    _step = dict(
        range_width_min=0.03,
        entry_buffer_pct=0.003,
        sl_buffer_pct=0.008,
        tp_ratio=0.75,
        trail_enabled=False,
        step_trail_enabled=True,
        step_trail_initial_sl_ratio=0.60,
        step_trail_initial_tp_ratio=0.85,
        step_trail_step_size=0.05,
        cooldown_bars=3,
    )
    
    return [
        # ── A: PRODUCTION — max_pos 30% ──
        RangeConfig(name="A_PROD_30", risk_pct=0.04, max_position_pct=0.30, **_step),
        
        # ── B: max_pos 50% ──
        RangeConfig(name="B_POS_50", risk_pct=0.04, max_position_pct=0.50, **_step),
        
        # ── C: max_pos 70% ──
        RangeConfig(name="C_POS_70", risk_pct=0.04, max_position_pct=0.70, **_step),
        
        # ── D: max_pos 90% ──
        RangeConfig(name="D_POS_90", risk_pct=0.04, max_position_pct=0.90, **_step),
        
        # ── E: max_pos 50% + Volume filter ──
        RangeConfig(name="E_P50_VOL", risk_pct=0.04, max_position_pct=0.50, volume_confirm=True, **_step),
        
        # ── F: max_pos 70% + Volume filter ──
        RangeConfig(name="F_P70_VOL", risk_pct=0.04, max_position_pct=0.70, volume_confirm=True, **_step),
        
        # ── G: max_pos 90% + Volume filter ──
        RangeConfig(name="G_P90_VOL", risk_pct=0.04, max_position_pct=0.90, volume_confirm=True, **_step),
    ]


# ── Display ────────────────────────────────────────────────────────────────────


def print_comparison(results: list[tuple[RangeConfig, dict]]) -> None:
    """Affiche un tableau comparatif des variantes."""
    print("\n" + "=" * 120)
    print("TRAIL RANGE — BACKTEST A/B COMPARISON")
    print("=" * 120)
    
    header = f"{'Variant':<20} {'Trades':>7} {'WinRate':>8} {'PF':>7} {'PnL$':>9} {'AvgPnl':>8} {'MaxDD':>8} {'Final$':>9} {'AvgDur':>7} {'AvgWid':>7} {'Fees$':>7}"
    print(header)
    print("-" * 120)
    
    for cfg, m in results:
        line = (
            f"{cfg.name:<20} "
            f"{m['n_trades']:>7d} "
            f"{m['win_rate']*100:>7.1f}% "
            f"{m['pf']:>7.2f} "
            f"{m['total_pnl']:>+9.2f} "
            f"{m['avg_pnl']:>+8.3f} "
            f"{m['max_dd']*100:>7.1f}% "
            f"{m['final_equity']:>9.2f} "
            f"{m['avg_duration_h']:>6.1f}h "
            f"{m['avg_range_width']*100:>6.1f}% "
            f"{m['total_fees']:>7.2f}"
        )
        print(line)
    
    print("-" * 120)
    
    # Exit reasons for each variant
    print("\n── Exit Reasons ──")
    for cfg, m in results:
        reasons = m.get("exit_reasons", {})
        if reasons:
            reason_str = " | ".join(f"{k}: {v}" for k, v in sorted(reasons.items()))
            print(f"  {cfg.name:<20} → {reason_str}")
    
    print()


def plot_equity_curves(
    results: list[tuple[RangeConfig, list[EquityPoint]]],
    pairs_str: str,
    years: int,
    initial_balance: float,
) -> Path:
    """Génère un graphique des equity curves."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e", "#95a5a6"]
    
    for i, (cfg, eq_curve) in enumerate(results):
        if not eq_curve:
            continue
        dates = [datetime.fromtimestamp(p.ts / 1000, tz=timezone.utc) for p in eq_curve]
        values = [p.equity for p in eq_curve]
        color = colors[i % len(colors)]
        lw = 2.5 if cfg.name == "CURRENT" else 1.5
        ax.plot(dates, values, label=cfg.name, color=color, linewidth=lw,
                linestyle="-" if cfg.name == "CURRENT" else "--")
    
    ax.axhline(y=initial_balance, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Trail Range — Backtest A/B ({pairs_str}, {years}y, ${initial_balance})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    
    path = OUTPUT_DIR / "range_ab_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest Trail Range A/B")
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS),
                        help="Paires séparées par des virgules")
    parser.add_argument("--years", type=int, default=3, help="Nombre d'années")
    parser.add_argument("--balance", type=float, default=200.0, help="Capital initial ($)")
    args = parser.parse_args()
    
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.years * 365)
    
    print(f"\n📊 Trail Range Backtest A/B")
    print(f"   Paires : {', '.join(pairs)}")
    print(f"   Période : {start.date()} → {end.date()} ({args.years} ans)")
    print(f"   Capital : ${args.balance}")
    print(f"   Downloading data...\n")
    
    # Download H4 candles for all pairs
    all_candles: dict[str, list[Candle]] = {}
    for pair in pairs:
        try:
            candles = download_candles(pair, start, end, interval="4h")
            if len(candles) > 100:
                all_candles[pair] = candles
                print(f"   ✅ {pair}: {len(candles)} candles")
            else:
                print(f"   ⚠️ {pair}: seulement {len(candles)} candles — skip")
        except Exception as e:
            print(f"   ❌ {pair}: {e}")
    
    if not all_candles:
        print("❌ Aucune donnée disponible")
        sys.exit(1)
    
    print(f"\n   {len(all_candles)} paires chargées. Lancement des backtests...\n")
    
    variants = get_variants()
    results: list[tuple[RangeConfig, dict]] = []
    eq_results: list[tuple[RangeConfig, list[EquityPoint]]] = []
    
    for cfg in variants:
        trades, eq_curve = simulate_range(all_candles, cfg, args.balance)
        metrics = compute_range_metrics(trades, eq_curve, args.balance)
        results.append((cfg, metrics))
        eq_results.append((cfg, eq_curve))
        print(f"   ✅ {cfg.name}: {metrics['n_trades']} trades, PF={metrics['pf']:.2f}, PnL=${metrics['total_pnl']:+.2f}")
    
    print_comparison(results)
    
    # Plot
    path = plot_equity_curves(eq_results, ", ".join(list(all_candles.keys())[:4]) + "...", args.years, args.balance)
    print(f"📈 Equity curves → {path}")


if __name__ == "__main__":
    main()
