#!/usr/bin/env python3
"""
Backtest — Dow Theory LONG-only (Revolut X)

Hypothèse à tester : la Dow Theory est-elle rentable en LONG uniquement ?
(ignorer complètement les signaux BEARISH)

Règles :
  - Timeframe  : H4 (6 bougies/jour)
  - Trend      : BULLISH confirmé = HH + HL consécutifs (swing lookback=3)
  - Entrée     : 3 variantes testées en parallèle
       BREAKOUT   — close > dernier HH + buffer → BUY au close
       PULLBACK   — prix revient dans zone [HL-zone%, HL+zone%] en tendance BULL → BUY
       PULLBACK_EMA — pullback sous EMA20 avec rebond en tendance BULL → BUY
  - SL         : dernier HL − sl_buffer (stop Dow naturel)
  - TP/Exit    : 3 modes testés
       FIXED_TP   — entrée × (1 + tp_pct)
       TRAILING   — trailing stop ATR (activé dès le 1er tick)
       TREND_ONLY — sortie uniquement sur invalidation Dow (prix < HL) + SL dur
  - Frais      : Revolut X — maker 0%, taker 0.09%
  - Risk       : 2% equity par trade, max 3 positions simultanées
  - LONG ONLY  — aucun SELL short, aucune position BEARISH

Walk-forward (3 fenêtres) :
  W1 : IS 2022→2023  OOS 2023→2024
  W2 : IS 2023→2024  OOS 2024→2025
  W3 : IS 2024→2025  OOS 2025→2026

Usage :
    python3 -m backtest.run_backtest_dow_long
    python3 -m backtest.run_backtest_dow_long --years 4 --balance 1000
    python3 -m backtest.run_backtest_dow_long --entry breakout
    python3 -m backtest.run_backtest_dow_long --exit trailing
    python3 -m backtest.run_backtest_dow_long --walkforward
"""

from __future__ import annotations

import argparse
import logging
import math
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
from src.core.models import Candle, SwingLevel, TrendDirection, TrendState
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend, check_trend_invalidation

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Frais Revolut X ──────────────────────────────────────────────────────────
MAKER_FEE = 0.000    # 0%
TAKER_FEE = 0.0009   # 0.09%

# ── Paires Revolut X (même que London Breakout + Infinity) ──────────────────
REVOLUT_PAIRS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
    "LINK-USD", "ADA-USD", "DOT-USD", "AVAX-USD",
]


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DowLongConfig:
    """Configuration d'une variante Dow Theory LONG-only."""
    name: str

    # ── Swing detection ──
    swing_lookback: int = 3
    swing_window: int = 150       # nb bougies H4 pour recalculer les swings (~25 jours)

    # ── Entry mode ──
    entry_mode: str = "breakout"  # "breakout" | "pullback" | "pullback_ema"
    breakout_buffer: float = 0.003   # 0.3% au-dessus du HH
    pullback_zone_pct: float = 0.02  # zone ± autour du dernier HL
    pullback_ema_period: int = 20     # EMA pour le mode pullback_ema

    # ── SL ──
    sl_buffer: float = 0.005        # 0.5% sous le dernier HL
    sl_max_pct: float = 0.08        # SL dur max −8% (évite les stops trop larges)

    # ── Exit mode ──
    exit_mode: str = "trailing"     # "fixed_tp" | "trailing" | "trend_only"
    tp_pct: float = 0.06            # +6% (fixed_tp)
    trail_atr_mult: float = 2.0     # trailing = peak − N×ATR (trailing)
    trail_activation_pct: float = 0.0  # activation trailing dès le départ (0=immédiat)
    atr_period: int = 14

    # ── Risk ──
    risk_pct: float = 0.02          # 2% risk par trade
    max_positions: int = 3
    max_alloc_pct: float = 0.40     # max 40% capital en 1 seule position

    # ── Cooldown ──
    cooldown_bars: int = 4          # 4 × 4h = 16h après SL

    # ── Confirmation ──
    require_bullish_candle: bool = False  # close > open pour valider l'entrée
    require_volume_spike: bool = False    # volume > MA20×1.2
    confirm_sequences: int = 1            # nb de paires HH-HL consécutives requises avant d'entrer
    confirm_sequences: int = 1            # nb de paires HH-HL consécutives requises avant d'entrer


# ── Variantes à comparer ───────────────────────────────────────────────────────

def _make_variants() -> list[DowLongConfig]:
    """Génère toutes les variantes à tester."""
    for entry in ("breakout", "pullback", "pullback_ema"):
        for exit_m in ("fixed_tp", "trailing", "trend_only"):
            variants.append(DowLongConfig(
                name=f"{entry.upper()}_{exit_m.upper()}",
                entry_mode=entry,
                exit_mode=exit_m,
            ))
    return variants


def _make_confirm_variants(entry: str = "breakout", exit_m: str = "trend_only") -> list[DowLongConfig]:
    """Génère les variantes 1×/2×/3× confirmations pour une combo entry/exit."""
    return [
        DowLongConfig(
            name=f"{entry.upper()}_{exit_m.upper()}_CONFIRM{n}x",
            entry_mode=entry,
            exit_mode=exit_m,
            confirm_sequences=n,
        )
        for n in (1, 2, 3)
    ]
        DowLongConfig(
            name=f"{entry.upper()}_{exit_m.upper()}_CONFIRM{n}x",
            entry_mode=entry,
            exit_mode=exit_m,
            confirm_sequences=n,
        )
        for n in (1, 2, 3)
    ]


# Configuration unique si on veut tester une seule
SINGLE_CFG = DowLongConfig(
    name="DOW_BULL_TRAIL",
    entry_mode="breakout",
    exit_mode="trailing",
    breakout_buffer=0.003,
    sl_buffer=0.005,
    trail_atr_mult=2.5,
    risk_pct=0.02,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DowTrade:
    symbol: str
    entry_bar: int
    entry_price: float
    entry_ts: int
    sl_price: float
    initial_sl: float
    tp_price: float
    peak_price: float = 0.0
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_ts: int = 0
    exit_reason: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    size: float = 0.0
    fees: float = 0.0
    duration_bars: int = 0
    entry_mode: str = ""


@dataclass
class EquityPoint:
    ts: int
    equity: float


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATEURS LOCAUX
# ══════════════════════════════════════════════════════════════════════════════

def _ema(closes: list[float], period: int) -> list[float]:
    n = len(closes)
    out = [0.0] * n
    if n < period:
        return out
    out[period - 1] = sum(closes[:period]) / period
    k = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = closes[i] * k + out[i - 1] * (1.0 - k)
    return out


def _atr(candles: list[Candle], period: int = 14) -> list[float]:
    n = len(candles)
    out = [0.0] * n
    if n < 2:
        return out
    trs = [0.0]
    for i in range(1, n):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period + 1:
        return out
    atr_val = sum(trs[1: period + 1]) / period
    out[period] = atr_val
    for i in range(period + 1, n):
        atr_val = (atr_val * (period - 1) + trs[i]) / period
        out[i] = atr_val
    return out


def _vol_ma(candles: list[Candle], period: int = 20) -> list[float]:
    vols = [c.volume for c in candles]
    n = len(vols)
    out = [0.0] * n
    for i in range(period - 1, n):
        out[i] = sum(vols[i - period + 1: i + 1]) / period
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CORE LOGIC — SIMULATION MONO-PAIRE
# ══════════════════════════════════════════════════════════════════════════════

def _run_pair(
    symbol: str,
    candles: list[Candle],
    cfg: DowLongConfig,
    initial_balance: float,
) -> tuple[list[DowTrade], list[EquityPoint], float]:
    """Simule la stratégie sur une paire. Retourne (trades, equity_curve, final_balance)."""

    if len(candles) < 50:
        return [], [], initial_balance

    closes = [c.close for c in candles]
    atr_vals = _atr(candles, cfg.atr_period)
    ema_vals = _ema(closes, cfg.pullback_ema_period) if cfg.entry_mode == "pullback_ema" else []
    vol_ma_vals = _vol_ma(candles, 20) if cfg.require_volume_spike else []

    equity = initial_balance
    trades: list[DowTrade] = []
    equity_pts: list[EquityPoint] = []

    position: Optional[dict] = None
    cooldown_until: int = 0

    warmup = cfg.swing_lookback * 2 + 10

    for bar in range(warmup, len(candles)):
        c = candles[bar]

        # ── Equity curve ──
        equity_pts.append(EquityPoint(ts=c.timestamp, equity=equity))

        # ── Re-détecter les swings sur fenêtre glissante ──
        win_start = max(0, bar - cfg.swing_window)
        window = candles[win_start: bar + 1]
        swings = detect_swings(window, cfg.swing_lookback)

        if len(swings) < 4:
            continue

        trend = determine_trend(swings, symbol)

        # ── Vérif invalidation si position ouverte ──
        if position is not None:
            # Update peak
            if c.high > position["peak"]:
                position["peak"] = c.high

            # Trailing stop ATR
            if cfg.exit_mode == "trailing":
                atr = atr_vals[bar]
                if atr > 0:
                    new_trail_sl = position["peak"] - cfg.trail_atr_mult * atr
                    if new_trail_sl > position["sl"]:
                        position["sl"] = new_trail_sl

            # Check SL touché (low de la bougie)
            if c.low <= position["sl"]:
                exit_price = min(position["sl"], c.open)  # gap potentiel
                trade = _close_pos(position, bar, c.timestamp, exit_price, "SL", equity, cfg)
                equity += trade.pnl_usd
                trades.append(trade)
                cooldown_until = bar + cfg.cooldown_bars
                position = None
                continue

            # Check TP atteint (fixed_tp)
            if cfg.exit_mode == "fixed_tp" and c.high >= position["tp"]:
                exit_price = position["tp"]
                trade = _close_pos(position, bar, c.timestamp, exit_price, "TP", equity, cfg)
                equity += trade.pnl_usd
                trades.append(trade)
                position = None
                continue

            # Sortie sur invalidation Dow (prix < dernier HL)
            if trend.direction != TrendDirection.BULLISH:
                if cfg.exit_mode == "trend_only":
                    exit_price = c.close
                    trade = _close_pos(position, bar, c.timestamp, exit_price, "TREND_BREAK", equity, cfg)
                    equity += trade.pnl_usd
                    trades.append(trade)
                    position = None
                    continue
                # Pour les autres modes : le SL dur reste, mais pas de sortie forçée
            continue  # position ouverte → pas de nouvelle entrée sur cette paire

        # ── Chercher signal d'entrée (LONG ONLY) ──
        if trend.direction != TrendDirection.BULLISH:
            continue  # BEARISH ou NEUTRAL → skip


        # Vérifier que le nombre de séquences HH-HL confirme l'entrée
        if cfg.confirm_sequences > 1:
            bull_count = _count_bull_sequences(swings)
            if bull_count < cfg.confirm_sequences:
                continue
        if bar < cooldown_until:
            continue

        last_hh = _last_hh(swings)
        last_hl = _last_hl(swings)

        if last_hh is None or last_hl is None:
            continue

        # Vérifier que le nombre de séquences HH-HL confirme l'entrée
        if cfg.confirm_sequences > 1:
            bull_count = _count_bull_sequences(swings)
            if bull_count < cfg.confirm_sequences:
                continue

        entry_price = None
        sl_raw = last_hl - last_hl * cfg.sl_buffer

        # Vérifier que le SL n'est pas trop loin (max sl_max_pct)
        if cfg.sl_max_pct > 0:
            min_sl = c.close * (1.0 - cfg.sl_max_pct)
            sl_raw = max(sl_raw, min_sl)

        # ── Mode BREAKOUT ──
        if cfg.entry_mode == "breakout":
            threshold = last_hh * (1.0 + cfg.breakout_buffer)
            if c.close >= threshold:
                entry_price = c.close

        # ── Mode PULLBACK (near last HL) ──
        elif cfg.entry_mode == "pullback":
            hl_low = last_hl * (1.0 - cfg.pullback_zone_pct)
            hl_high = last_hl * (1.0 + cfg.pullback_zone_pct)
            if hl_low <= c.close <= hl_high:
                entry_price = c.close

        # ── Mode PULLBACK_EMA ──
        elif cfg.entry_mode == "pullback_ema":
            if len(ema_vals) > bar and ema_vals[bar] > 0:
                ema_now = ema_vals[bar]
                # Prix sous EMA20 mais au-dessus du SL → pullback
                if sl_raw < c.close <= ema_now * 1.005:
                    # Bougie de clôture au-dessus de l'open → rebond
                    if c.close > c.open:
                        entry_price = c.close

        if entry_price is None:
            continue

        # ── Filtres additionnels ──
        if cfg.require_bullish_candle and c.close <= c.open:
            continue

        if cfg.require_volume_spike and len(vol_ma_vals) > bar:
            if vol_ma_vals[bar] > 0 and c.volume < vol_ma_vals[bar] * 1.2:
                continue

        # ── Sizing ──
        risk_amount = equity * cfg.risk_pct
        sl_distance = entry_price - sl_raw
        if sl_distance <= 0:
            continue

        size = risk_amount / sl_distance
        cost = size * entry_price
        if cost > equity * cfg.max_alloc_pct:
            size = (equity * cfg.max_alloc_pct) / entry_price
            cost = size * entry_price

        if size <= 0 or cost < 1.0:
            continue

        # ── TP selon le mode ──
        if cfg.exit_mode == "fixed_tp":
            tp_price = entry_price * (1.0 + cfg.tp_pct)
        else:
            tp_price = entry_price * 1.999  # pas de TP fixe, sortie par trail/trend

        position = {
            "symbol": symbol,
            "entry_bar": bar,
            "entry_price": entry_price,
            "entry_ts": c.timestamp,
            "sl": sl_raw,
            "initial_sl": sl_raw,
            "tp": tp_price,
            "peak": entry_price,
            "size": size,
            "cost": cost,
            "entry_mode": cfg.entry_mode,
        }

    # Clôturer une position ouverte en fin de période
    if position is not None:
        last_bar = len(candles) - 1
        exit_price = candles[last_bar].close
        trade = _close_pos(position, last_bar, candles[last_bar].timestamp, exit_price, "END", equity, cfg)
        equity += trade.pnl_usd
        trades.append(trade)

    return trades, equity_pts, equity


def _last_hh(swings) -> Optional[float]:
    """Retourne le prix du dernier Higher High."""
    from src.core.models import SwingLevel, SwingType
    highs = [s 


def _count_bull_sequences(swings) -> int:
    """
    Compte le nombre de paires HH-HL consécutives à la fin de la séquence de swings.

    Algorithme : on parcourt les swings du plus récent au plus ancien, en vérifiant
    que les highs sont des HH et les lows des HL de manière alternée.
    Retourne le nombre de séquences HH+HL complètes consécutives (1 = confirmé 1×,
    2 = 2 HH et 2 HL consécutifs, etc.).
    """
    from src.core.models import SwingLevel, SwingType

    typed = [s for s in swings if s.swing_type is not None]
    if len(typed) < 2:
        return 0

    sequences = 0
    # On remonte depuis la fin en cherchant des paires HH / HL
    i = len(typed) - 1
    while i >= 1:
        s_last = typed[i]
        s_prev = typed[i - 1]
        # Paire valide = (HH puis HL) ou (HL puis HH) en terminant la séquence
        last_is_hh = s_last.level == SwingLevel.HIGH and s_last.swing_type.value == "HH"
        last_is_hl = s_last.level == SwingLevel.LOW  and s_last.swing_type.value == "HL"
        prev_is_hh = s_prev.level == SwingLevel.HIGH and s_prev.swing_type.value == "HH"
        prev_is_hl = s_prev.level == SwingLevel.LOW  and s_prev.swing_type.value == "HL"

        if (last_is_hh and prev_is_hl) or (last_is_hl and prev_is_hh):
            sequences += 1
            i -= 2  # consommer les deux swings
        else:
            break  # séquence interrompue

    return sequencesfor s in swings if s.level == SwingLevel.HIGH and s.swing_type is not None]
    for s in reversed(highs):
        if s.swing_type.value == "HH":
            return s.price
    return None


def _last_hl(swings) -> Optional[float]:
    """Retourne le prix du dernier Higher Low."""
    from src.core.models import SwingLevel, SwingType
    lows = [s for s in swings if s.level == SwingLevel.LOW and s.swing_type is not None]
    for s in reversed(lows):
        if s.swing_type.value == "HL":
            return s.price
    return None


def _count_bull_sequences(swings) -> int:
    """
    Compte le nombre de paires HH-HL consécutives à la fin de la séquence de swings.

    Algorithme : on parcourt les swings du plus récent au plus ancien, en vérifiant
    que les highs sont des HH et les lows des HL de manière alternée.
    Retourne le nombre de séquences HH+HL complètes consécutives (1 = confirmé 1×,
    2 = 2 HH et 2 HL consécutifs, etc.).
    """
    from src.core.models import SwingLevel, SwingType

    typed = [s for s in swings if s.swing_type is not None]
    if len(typed) < 2:
        return 0

    sequences = 0
    # On remonte depuis la fin en cherchant des paires HH / HL
    i = len(typed) - 1
    while i >= 1:
        s_last = typed[i]
        s_prev = typed[i - 1]
        # Paire valide = (HH puis HL) ou (HL puis HH) en terminant la séquence
        last_is_hh = s_last.level == SwingLevel.HIGH and s_last.swing_type.value == "HH"
        last_is_hl = s_last.level == SwingLevel.LOW  and s_last.swing_type.value == "HL"
        prev_is_hh = s_prev.level == SwingLevel.HIGH and s_prev.swing_type.value == "HH"
        prev_is_hl = s_prev.level == SwingLevel.LOW  and s_prev.swing_type.value == "HL"

        if (last_is_hh and prev_is_hl) or (last_is_hl and prev_is_hh):
            sequences += 1
            i -= 2  # consommer les deux swings
        else:
            break  # séquence interrompue

    return sequences


def _close_pos(
    pos: dict,
    bar: int,
    ts: int,
    exit_price: float,
    reason: str,
    equity: float,
    cfg: DowLongConfig,
) -> DowTrade:
    entry_fee = pos["cost"] * TAKER_FEE
    exit_fee = pos["size"] * exit_price * TAKER_FEE
    gross_pnl = (exit_price - pos["entry_price"]) * pos["size"]
    net_pnl = gross_pnl - entry_fee - exit_fee
    pnl_pct = net_pnl / pos["cost"] if pos["cost"] > 0 else 0.0

    return DowTrade(
        symbol=pos["symbol"],
        entry_bar=pos["entry_bar"],
        entry_price=pos["entry_price"],
        entry_ts=pos["entry_ts"],
        sl_price=pos["sl"],
        initial_sl=pos["initial_sl"],
        tp_price=pos["tp"],
        peak_price=pos["peak"],
        exit_bar=bar,
        exit_price=exit_price,
        exit_ts=ts,
        exit_reason=reason,
        pnl_usd=net_pnl,
        pnl_pct=pnl_pct,
        size=pos["size"],
        fees=entry_fee + exit_fee,
        duration_bars=bar - pos["entry_bar"],
        entry_mode=pos["entry_mode"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION MULTI-PAIRES
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    pairs: list[str],
    start: datetime,
    end: datetime,
    cfg: DowLongConfig,
    initial_balance: float = 1000.0,
    verbose: bool = False,
) -> tuple[list[DowTrade], list[EquityPoint], float]:
    """Lance le backtest multi-paires, capital partagé en equal-weight."""

    all_candles: dict[str, list[Candle]] = {}
    for pair in pairs:
        candles = download_candles(pair, start, end, interval="4h")
        if candles:
            all_candles[pair] = candles
            if verbose:
                print(f"  📥 {pair}: {len(candles)} bougies H4")

    if not all_candles:
        return [], [], initial_balance

    # Capital per pair (equal weight, indépendant)
    n_pairs = len(all_candles)
    per_pair_capital = initial_balance / n_pairs

    all_trades: list[DowTrade] = []
    ts_equity: dict[int, float] = defaultdict(float)

    for pair, candles in all_candles.items():
        trades, eq_pts, final = _run_pair(pair, candles, cfg, per_pair_capital)
        all_trades.extend(trades)
        for pt in eq_pts:
            ts_equity[pt.ts] += pt.equity

    combined_eq = [EquityPoint(ts=ts, equity=eq) for ts, eq in sorted(ts_equity.items())]
    all_trades.sort(key=lambda t: t.entry_ts)

    total_final = sum(ts_equity[max(ts_equity)] / n_pairs * n_pairs
                      for _ in [1]) if ts_equity else initial_balance

    # Recalcul correct du final
    total_final = 0.0
    for pair, candles in all_candles.items():
        pair_trades = [t for t in all_trades if t.symbol == pair]
        balance = per_pair_capital
        for t in pair_trades:
            balance += t.pnl_usd
        total_final += balance

    return all_trades, combined_eq, total_final


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    trades: list[DowTrade],
    eq: list[EquityPoint],
    initial: float,
    final: float,
    start: datetime,
    end: datetime,
) -> dict:
    years = max((end - start).days / 365.25, 0.01)
    n = len(trades)

    total_return = (final - initial) / initial
    cagr = (final / initial) ** (1.0 / years) - 1 if final > 0 and initial > 0 else -1.0

    # Drawdown
    peak_eq = initial
    max_dd = 0.0
    for pt in eq:
        if pt.equity > peak_eq:
            peak_eq = pt.equity
        dd = (pt.equity - peak_eq) / peak_eq if peak_eq > 0 else 0
        if dd < max_dd:
            max_dd = dd

    if n == 0:
        return {
            "n_trades": 0, "win_rate": 0, "profit_factor": 0,
            "total_pnl": 0, "total_return": total_return, "cagr": cagr,
            "max_dd": max_dd, "daily_pnl_avg": 0, "trades_per_day": 0,
            "total_fees": 0, "avg_duration_bars": 0,
            "by_exit": {}, "by_pair": {},
        }

    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
    total_pnl = sum(t.pnl_usd for t in trades)
    total_fees = sum(t.fees for t in trades)
    days = max((end - start).days, 1)

    by_exit: dict[str, int] = defaultdict(int)
    for t in trades:
        by_exit[t.exit_reason] += 1

    by_pair: dict[str, dict] = {}
    for t in trades:
        if t.symbol not in by_pair:
            by_pair[t.symbol] = {"n": 0, "pnl": 0.0, "wins": 0}
        by_pair[t.symbol]["n"] += 1
        by_pair[t.symbol]["pnl"] += t.pnl_usd
        if t.pnl_usd > 0:
            by_pair[t.symbol]["wins"] += 1

    return {
        "n_trades": n,
        "win_rate": len(wins) / n,
        "profit_factor": gross_profit / gross_loss,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "cagr": cagr,
        "max_dd": max_dd,
        "daily_pnl_avg": total_pnl / days,
        "trades_per_day": n / days,
        "total_fees": total_fees,
        "avg_duration_bars": sum(t.duration_bars for t in trades) / n,
        "by_exit": dict(by_exit),
        "by_pair": by_pair,
    }


# ══════════════════════════════════════════════════════════════════════════════
          f"  trail={cfg.trail_atr_mult}×ATR  confirm={cfg.confirm_sequences}×HH-HL")
# ══════════════════════════════════════════════════════════════════════════════

def _print_header(title: str) -> None:
    sep = "═" * 110
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)


def _print_metrics(m: dict, label: str = "", initial: float = 1000.0, final: float = 0.0) -> None:
    pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 999 else ">999"
    print(
        f"  {label:<28s} | {m['n_trades']:4d} trades"
        f" | WR {m['win_rate']:5.1%}"
        f" | PF {pf_str:>5s}"
        f" | PnL ${m['total_pnl']:+8.2f}"
        f" | CAGR {m['cagr']:+6.1%}"
        f" | DD {m['max_dd']:5.1%}"
        f" | Fees ${m['total_fees']:+6.2f}"
    )


def print_full_report(
    cfg: DowLongConfig,
    m: dict,
    trades: list[DowTrade],
    initial: float,
    final: float,
    start: datetime,
    end: datetime,
) -> None:
    years = max((end - start).days / 365.25, 0.01)
    sep = "─" * 110

    _print_header(f"DOW THEORY LONG-ONLY — {cfg.name} | {cfg.entry_mode.upper()} → {cfg.exit_mode.upper()}"
                  f" | {start.date()} → {end.date()}")

    print(f"\n  Paires : {', '.join(REVOLUT_PAIRS)}")
    print(f"  Capital initial : ${initial:.0f}  |  Final : ${final:.2f}  "
          f"  Rendement : {m['total_return']:+.1%}  |  CAGR : {m['cagr']:+.1%}")
    print(f"  Frais totaux    : ${m['total_fees']:.2f}  |  DD max : {m['max_dd']:.1%}")
    print(f"  Trades : {m['n_trades']}  |  WR : {m['win_rate']:.1%}  "
          f"|  PF : {m['profit_factor']:.2f}  |  Durée moy : {m['avg_duration_bars']:.0f} bars H4 "
          f"({m['avg_duration_bars']/6:.1f}j)")
    print(f"\n  Config :")
    print(f"    entry_mode={cfg.entry_mode}  breakout_buffer={cfg.breakout_buffer:.1%}"
          f"  pullback_zone={cfg.pullback_zone_pct:.1%}")
    print(f"    sl_buffer={cfg.sl_buffer:.1%}  sl_max={cfg.sl_max_pct:.0%}"
          f"  exit_mode={cfg.exit_mode}  tp={cfg.tp_pct:.0%}"
          f"  trail={cfg.trail_atr_mult}×ATR  confirm={cfg.confirm_sequences}×HH-HL")

    if m["n_trades"] == 0:
        print("\n  ⚠️  Aucun trade généré.")
        return

    # Par sorte de sortie
    print(f"\n  Sorties :")
    for reason, count in sorted(m["by_exit"].items(), key=lambda x: -x[1]):
        pct = count / m["n_trades"] * 100
        print(f"    {reason:<15s} : {count:4d} ({pct:.0f}%)")

    # Par paire
    print(f"\n  Performance par paire :")
    print(f"  {'Paire':<12s}  {'Trades':>6s}  {'WR':>6s}  {'PnL':>10s}")
    print(f"  {sep[:45]}")
    for pair, pp in sorted(m["by_pair"].items(), key=lambda x: -x[1]["pnl"]):
        wr = pp["wins"] / pp["n"] if pp["n"] > 0 else 0
        symbol_m = "🟢" if pp["pnl"] > 0 else "🔴"
        print(f"  {symbol_m} {pair:<10s}  {pp['n']:>6d}  {wr:>5.1%}  ${pp['pnl']:>+9.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  CHART EQUITY CURVES
# ══════════════════════════════════════════════════════════════════════════════

def plot_equity_curves(
    results: list[tuple[str, list[EquityPoint], float]],
    title: str,
    filename: str,
    initial: float,
) -> None:
    if not results:
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [
        "#00bcd4", "#ff5722", "#4caf50", "#9c27b0", "#ff9800",
        "#2196f3", "#e91e63", "#009688", "#ffc107",
    ]

    for i, (label, eq, final) in enumerate(results):
        if not eq:
            continue
        dates = [datetime.fromtimestamp(pt.ts / 1000, tz=timezone.utc) for pt in eq]
        equities = [pt.equity for pt in eq]
        color = colors[i % len(colors)]
        ret = (final - initial) / initial * 100
        ax.plot(dates, equities, label=f"{label} ({ret:+.1f}%)", color=color, linewidth=1.5)

    ax.axhline(y=initial, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Initial")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital ($)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"\n  📊 Chart sauvegardé : {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════════

WF_WINDOWS = [
    ("W1 — 2021→2022→2023", "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("W2 — 2022→2023→2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("W3 — 2023→2024→2025", "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("W4 — 2024→2025→2026", "2024-01-01", "2025-09-30", "2025-10-01", "2026-04-15"),
]


def run_walkforward(
    cfg: DowLongConfig,
    pairs: list[str],
    balance: float,
) -> None:
    _print_header(f"WALK-FORWARD — DOW THEORY LONG-ONLY — {cfg.name}")
    print(f"\n  {'Fenêtre':<28s} │ {'Phase':>5s} │ {'Trades':>6s} │ {'WR':>6s} │ "
          f"{'PF':>5s} │ {'PnL':>10s} │ {'CAGR':>7s} │ {'DD':>7s}")
    print("  " + "─" * 100)

    test_pnls, test_pfs, test_wrs = [], [], []

    for label, is_s, is_e, oos_s, oos_e in WF_WINDOWS:
        for phase, s_str, e_str in [("IS", is_s, is_e), ("OOS", oos_s, oos_e)]:
            start = datetime.strptime(s_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = datetime.strptime(e_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            trades, eq, final = run_backtest(pairs, start, end, cfg, balance)
            m = compute_metrics(trades, eq, balance, final, start, end)

            pf_s = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 999 else ">999"
            marker = "  " if phase == "IS" else "→ "
            label_disp = label if phase == "IS" else ""
            print(
                f"  {label_disp:<28s} │ {marker}{phase} │ "
                f"{m['n_trades']:6d} │ {m['win_rate']:5.1%} │ "
                f"{pf_s:>5s} │ ${m['total_pnl']:+9.2f} │ "
                f"{m['cagr']:+6.1%} │ {m['max_dd']:6.1%}"
            )

            if phase == "OOS":
                test_pnls.append(m["total_pnl"])
                test_pfs.append(m["profit_factor"])
                test_wrs.append(m["win_rate"])

    print("  " + "─" * 100)
    plot_equity_curves(
        results,
        f"Dow Theory LONG-only — 9 variantes ({start.date()} → {end.date()})",
        "dow_long_grid.png",
        balance,
    )


def run_confirm_grid(
    pairs: list[str],
    start: datetime,
    end: datetime,
    balance: float,
) -> None:
    """Grille confirmations : 1×/2×/3× HH-HL sur toutes les combos entry×exit."""
    print(f"\n  Grille confirmations (1×/2×/3× HH-HL) — {start.date()} → {end.date()} | ${balance}")
    print(f"\n  {'Config':<38s}  {'Trades':>6s}  {'WR':>6s}  {'PF':>5s}  "
          f"{'PnL':>10s}  {'CAGR':>7s}  {'DD':>7s}")
    print("  " + "─" * 100)

    all_results: list[tuple[str, list[EquityPoint], float]] = []

    for entry in ("breakout", "pullback"):
        for exit_m in ("fixed_tp", "trailing", "trend_only"):
            print(f"\n  ── {entry.upper()} × {exit_m.upper()} ──")
            variants = _make_confirm_variants(entry, exit_m)
            for cfg in variants:
                trades, eq, final = run_backtest(pairs, start, end, cfg, balance)
                m = compute_metrics(trades, eq, balance, final, start, end)
                pf_s = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 999 else ">999"
                mark = "🟢" if m["profit_factor"] > 1.2 and m["total_pnl"] > 0 else ("🟡" if m["total_pnl"] > 0 else "🔴")
                print(
                    f"  {mark} {cfg.name:<36s}  {m['n_trades']:>6d}  {m['win_rate']:>5.1%}  "
                    f"{pf_s:>5s}  ${m['total_pnl']:>+9.2f}  {m['cagr']:>+6.1%}  {m['max_dd']:>6.1%}"
                )
                all_results.append((cfg.name, eq, final))

    plot_equity_curves(
        all_results,
        f"Dow Theory — confirmations 1×/2×/3× HH-HL ({start.date()} → {end.date()})",
        "dow_long_confirm
    print(f"\n  SYNTHÈSE OOS")
    print(f"    Périodes +    : {n_pos}/{len(test_pnls)}")
    print(f"    PF moyen OOS  : {avg_pf:.2f}")
    print(f"    WR moyen OOS  : {avg_wr:.1%}")
    print(f"    PnL total OOS : ${total_oos:+.2f}")

    if n_pos == len(test_pnls) and avg_pf > 1.2:
        verdict = "✅ VALIDÉ — Edge robuste confirmé"
    elif n_pos >= len(test_pnls) * 0.5 and avg_pf > 1.0:
        verdict = "⚠️  PROMETTEUR — Edge présent mais pas constant"
    else:
        verdict = "❌ REJETÉ — Pas d'edge OOS"
    print(f"\n  Verdict : {verdict}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARAISON GRILLE 3×3
# ══════════════════════════════════════════════════════════════════════════════

def run_grid(
    pairs: list[str],
    parser.add_argument("--confirm", action="store_true", help="Grille 1×/2×/3× confirmations HH-HL")
    parser.add_argument("--confirm-seq", type=int, default=1, choices=[1, 2, 3],
                        help="Nb de séquences HH-HL requises pour 1 variante unique (défaut: 1)")
    start: datetime,
    end: datetime,
    balance: float,
) -> None:
    variants = _make_variants()
    results = []

    print(f"\n  Grille 3×3 (9 variantes) — {start.date()} → {end.date()} | ${balance}")
    print(f"\n  {'Config':<32s}  {'Trades':>6s}  {'WR':>6s}  {'PF':>5s}  "
          f"{'PnL':>10s}  {'CAGR':>7s}  {'DD':>7s}  {'Fees':>8s}")
    print("  " + "─" * 100)

    for cfg in variants:
        trades, eq, final = run_backtest(pairs, start, end, cfg, balance)
        m = compute_metrics(trades, eq, balance, final, start, end)
        pf_s = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 999 else ">999"
        mark = "🟢" if m["profit_factor"] > 1.2 and m["total_pnl"] > 0 else ("🟡" if m["total_pnl"] > 0 else "🔴")
        print(
            f"  {mark} {cfg.name:<30s}  {m['n_trades']:>6d}  {m['win_rate']:>5.1%}  "
            f"{pf_s:>5s}  ${m['total_pnl']:>+9.2f}  {m['cagr']:>+6.1%}  "
            f"{m['max_dd']:>6.1%}  ${m['total_fees']:>+7.2f}"
        )
        results.append((cfg.name, eq, final))

    plot_equity_curves(
        resuconfirm:
        run_confirm_grid(pairs, start, end, balance)
        return

    if args.grid or (args.entry == "all" and args.exit == "all"):
        run_grid(pairs, start, end, balance)
        return

    # Variante unique
    entry = args.entry if args.entry != "all" else "breakout"
    exit_m = args.exit if args.exit != "all" else "trailing"
    cfg = DowLongConfig(
        name=f"{entry.upper()}_{exit_m.upper()}_C{args.confirm_seq}x",
        entry_mode=entry,
        exit_mode=exit_m,
        confirm_sequences=args.confirm_seq
    balance: float,
) -> None:
    """Grille confirmations : 1×/2×/3× HH-HL sur toutes les combos entry×exit."""
    print(f"\n  Grille confirmations (1×/2×/3× HH-HL) — {start.date()} → {end.date()} | ${balance}")
    print(f"\n  {'Config':<38s}  {'Trades':>6s}  {'WR':>6s}  {'PF':>5s}  "
          f"{'PnL':>10s}  {'CAGR':>7s}  {'DD':>7s}")
    print("  " + "─" * 100)

    all_results: list[tuple[str, list[EquityPoint], float]] = []

    for entry in ("breakout", "pullback"):
        for exit_m in ("fixed_tp", "trailing", "trend_only"):
            print(f"\n  ── {entry.upper()} × {exit_m.upper()} ──")
            variants = _make_confirm_variants(entry, exit_m)
            for cfg in variants:
                trades, eq, final = run_backtest(pairs, start, end, cfg, balance)
                m = compute_metrics(trades, eq, balance, final, start, end)
                pf_s = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 999 else ">999"
                mark = "🟢" if m["profit_factor"] > 1.2 and m["total_pnl"] > 0 else ("🟡" if m["total_pnl"] > 0 else "🔴")
                print(
                    f"  {mark} {cfg.name:<36s}  {m['n_trades']:>6d}  {m['win_rate']:>5.1%}  "
                    f"{pf_s:>5s}  ${m['total_pnl']:>+9.2f}  {m['cagr']:>+6.1%}  {m['max_dd']:>6.1%}"
                )
                all_results.append((cfg.name, eq, final))

    plot_equity_curves(
        all_results,
        f"Dow Theory — confirmations 1×/2×/3× HH-HL ({start.date()} → {end.date()})",
        "dow_long_confirm_grid.png",
        balance,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Dow Theory LONG-only (Revolut X)")
    parser.add_argument("--years", type=int, default=3, help="Durée en années (défaut: 3)")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial")
    parser.add_argument("--pairs", type=str, default="", help="Paires séparées par virgule")
    parser.add_argument("--entry", type=str, default="all",
                        choices=["all", "breakout", "pullback", "pullback_ema"],
                        help="Mode d'entrée")
    parser.add_argument("--exit", type=str, default="all",
                        choices=["all", "fixed_tp", "trailing", "trend_only"],
                        help="Mode de sortie")
    parser.add_argument("--walkforward", action="store_true", help="Lancer le walk-forward")
    parser.add_argument("--grid", action="store_true", help="Lancer la grille 3×3")
    parser.add_argument("--confirm", action="store_true", help="Grille 1×/2×/3× confirmations HH-HL")
    parser.add_argument("--confirm-seq", type=int, default=1, choices=[1, 2, 3],
                        help="Nb de séquences HH-HL requises pour 1 variante unique (défaut: 1)")
    args = parser.parse_args()

    pairs = args.pairs.split(",") if args.pairs else REVOLUT_PAIRS
    balance = args.balance
    end = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=365 * args.years)

    print(f"\n{'═'*110}")
    print(f"  DOW THEORY LONG-ONLY BACKTEST")
    print(f"  Période  : {start.date()} → {end.date()} ({args.years} ans)")
    print(f"  Capital  : ${balance:.0f}")
    print(f"  Paires   : {', '.join(pairs)}")
    print(f"  Frais    : Maker {MAKER_FEE:.2%} | Taker {TAKER_FEE:.2%} (Revolut X)")
    print(f"{'═'*110}\n")

    if args.walkforward:
        # Walk-forward sur les 3 meilleures configs a priori
        for cfg in [
            DowLongConfig(name="BREAKOUT_TRAIL", entry_mode="breakout", exit_mode="trailing"),
            DowLongConfig(name="PULLBACK_TRAIL", entry_mode="pullback", exit_mode="trailing"),
            DowLongConfig(name="BREAKOUT_TREND", entry_mode="breakout", exit_mode="trend_only"),
        ]:
            run_walkforward(cfg, pairs, balance)
        return

    if args.confirm:
        run_confirm_grid(pairs, start, end, balance)
        return

    if args.grid or (args.entry == "all" and args.exit == "all"):
        run_grid(pairs, start, end, balance)
        return

    # Variante unique
    entry = args.entry if args.entry != "all" else "breakout"
    exit_m = args.exit if args.exit != "all" else "trailing"
    cfg = DowLongConfig(
        name=f"{entry.upper()}_{exit_m.upper()}_C{args.confirm_seq}x",
        entry_mode=entry,
        exit_mode=exit_m,
        confirm_sequences=args.confirm_seq,
    )

    print(f"  ⏳ Simulation en cours...")
    trades, eq, final = run_backtest(pairs, start, end, cfg, balance, verbose=True)
    m = compute_metrics(trades, eq, balance, final, start, end)
    print_full_report(cfg, m, trades, balance, final, start, end)

    # Chart
    plot_equity_curves(
        [(cfg.name, eq, final)],
        f"Dow Theory LONG-only — {cfg.name}",
        f"dow_long_{cfg.name.lower()}.png",
        balance,
    )


if __name__ == "__main__":
    main()
