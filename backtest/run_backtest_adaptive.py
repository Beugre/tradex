#!/usr/bin/env python3
"""
Backtest — Stratégie Adaptative Multi-Régimes

Logique :
  1. Détection du régime sur 1H (Bull / Bear / Range / Stagnation) via score 0-5
  2. Exécution sur 15m selon le régime détecté :
       BULL       → Trend Following  (pullback EMA20, trailing)
       RANGE      → Mean Reversion   (achat bas du range, vente milieu + haut)
       STAGNATION → Scalping         (+0.5%–+1%)
       BEAR       → Scalp rebond RSI < 30 uniquement, sinon NO TRADE
  3. Risk management global : max 1-2% capital/trade, DD journalier max 5%,
     cooldown 2 bougies après perte, max 3 positions simultanées

Usage :
    python3 -m backtest.run_backtest_adaptive
    python3 -m backtest.run_backtest_adaptive --balance 1000 --years 3
    python3 -m backtest.run_backtest_adaptive --years 2
"""

from __future__ import annotations

import argparse
import bisect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from backtest.data_loader import download_candles
from src.core.models import Candle

logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

PAIRS_BIG5 = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"]
PAIRS_CANDIDATES = ["LINK-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD", "NEAR-USD", "LTC-USD", "DOT-USD", "ADA-USD"]

# ═══════════════════════════════════════════════════════════════════════════
# Régimes
# ═══════════════════════════════════════════════════════════════════════════

class Regime(Enum):
    BULL       = "BULL"
    BEAR       = "BEAR"
    RANGE      = "RANGE"
    STAGNATION = "STAGNATION"
    UNKNOWN    = "UNKNOWN"

# ═══════════════════════════════════════════════════════════════════════════
# Indicateurs
# ═══════════════════════════════════════════════════════════════════════════

def _ema(closes: list[float], period: int) -> list[float]:
    result = [0.0] * len(closes)
    if len(closes) < period:
        return result
    result[period - 1] = sum(closes[:period]) / period
    k = 2.0 / (period + 1)
    for i in range(period, len(closes)):
        result[i] = closes[i] * k + result[i - 1] * (1.0 - k)
    return result


def _rsi(closes: list[float], period: int = 14) -> list[float]:
    result = [50.0] * len(closes)
    if len(closes) <= period:
        return result
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    result[period] = (100.0 - 100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss else 100.0
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
        result[i] = (100.0 - 100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss else 100.0
    return result


def _atr(candles: list[Candle], period: int = 14) -> list[float]:
    result = [0.0] * len(candles)
    if len(candles) < 2:
        return result
    trs = [0.0]
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period + 1:
        return result
    atr_v = sum(trs[1: period + 1]) / period
    result[period] = atr_v
    for i in range(period + 1, len(candles)):
        atr_v = (atr_v * (period - 1) + trs[i]) / period
        result[i] = atr_v
    return result


def _sma(values: list[float], period: int) -> list[float]:
    result = [0.0] * len(values)
    for i in range(period - 1, len(values)):
        result[i] = sum(values[i - period + 1: i + 1]) / period
    return result


def _adx(candles: list[Candle], period: int = 14) -> list[float]:
    """ADX simplifié (Wilder)."""
    n = len(candles)
    result = [0.0] * n
    if n < period * 2 + 1:
        return result

    plus_dm  = [0.0] * n
    minus_dm = [0.0] * n
    tr_v     = [0.0] * n
    for i in range(1, n):
        h, l   = candles[i].high, candles[i].low
        ph, pl = candles[i - 1].high, candles[i - 1].low
        pc     = candles[i - 1].close
        tr_v[i]     = max(h - l, abs(h - pc), abs(l - pc))
        up, down    = h - ph, pl - l
        plus_dm[i]  = up   if up > down and up > 0   else 0.0
        minus_dm[i] = down if down > up  and down > 0 else 0.0

    # Wilder smoothing
    atr_w = sum(tr_v[1: period + 1])
    pdm_w = sum(plus_dm[1: period + 1])
    mdm_w = sum(minus_dm[1: period + 1])
    adx_v = [0.0] * n

    for i in range(period + 1, n):
        atr_w = atr_w - atr_w / period + tr_v[i]
        pdm_w = pdm_w - pdm_w / period + plus_dm[i]
        mdm_w = mdm_w - mdm_w / period + minus_dm[i]

        di_p  = 100 * pdm_w / atr_w if atr_w else 0
        di_m  = 100 * mdm_w / atr_w if atr_w else 0
        dx    = 100 * abs(di_p - di_m) / (di_p + di_m) if (di_p + di_m) else 0
        adx_v[i] = dx

    # Smooth ADX (Wilder)
    adx_sum = sum(adx_v[period + 1: period * 2 + 1])
    result[period * 2] = adx_sum / period
    for i in range(period * 2 + 1, n):
        result[i] = (result[i - 1] * (period - 1) + adx_v[i]) / period
    return result


def _bollinger_width(closes: list[float], period: int = 20, k: float = 2.0) -> list[float]:
    """Bollinger Band Width = (upper - lower) / middle."""
    result = [0.0] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        mid  = sum(window) / period
        std  = (sum((x - mid) ** 2 for x in window) / period) ** 0.5
        result[i] = (2 * k * std / mid) if mid else 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Trade result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveTrade:
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_abs: float
    is_win: bool
    regime: Regime
    exit_reason: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Détection de régime sur une fenêtre 1H
# ═══════════════════════════════════════════════════════════════════════════

def _detect_regime(
    h_candles: list[Candle],
    idx: int,
    ema20_h: list[float],
    ema50_h: list[float],
    ema200_h: list[float],
    adx_h: list[float],
    rsi_h: list[float],
    atr_h: list[float],
    atr_ma50_h: list[float],
    bb_width_h: list[float],
    bb_width_low_pct: float = 0.015,   # seuil "faible" Boll width
) -> Regime:
    if idx < 50:
        return Regime.UNKNOWN

    price  = h_candles[idx].close
    e20    = ema20_h[idx]
    e50    = ema50_h[idx]
    e200   = ema200_h[idx]
    adx    = adx_h[idx]
    rsi    = rsi_h[idx]
    atr    = atr_h[idx]
    atr_ma = atr_ma50_h[idx]
    bbw    = bb_width_h[idx]

    # ── 1. Stagnation (priorité maximale) ────────────────────────────────
    if atr_ma > 0 and atr < 0.8 * atr_ma and bbw < bb_width_low_pct:
        return Regime.STAGNATION

    # ── 2. Bull ──────────────────────────────────────────────────────────
    bull_score = 0
    if e50 > 0 and price > e50:           bull_score += 1
    if e20 > 0 and e50 > 0 and e20 > e50: bull_score += 1
    if e50 > 0 and e200 > 0 and e50 > e200: bull_score += 1
    if adx > 22:                           bull_score += 1
    if rsi > 55:                           bull_score += 1
    if bull_score >= 4:
        return Regime.BULL

    # ── 3. Bear ──────────────────────────────────────────────────────────
    bear_score = 0
    if e50 > 0 and price < e50:            bear_score += 1
    if e20 > 0 and e50 > 0 and e20 < e50: bear_score += 1
    if e50 > 0 and e200 > 0 and e50 < e200: bear_score += 1
    if adx > 22:                            bear_score += 1
    if rsi < 45:                            bear_score += 1
    if bear_score >= 4:
        return Regime.BEAR

    # ── 4. Range ─────────────────────────────────────────────────────────
    if adx < 18:
        return Regime.RANGE

    return Regime.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# Simulation adaptative sur une paire
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _Position:
    entry: float
    size: float
    cost: float
    sl: float
    tp: float
    peak: float
    regime: Regime
    trail_active: bool = False
    bars_open: int = 0


def _run_adaptive_pair(
    candles_15m: list[Candle],
    candles_1h: list[Candle],
    initial_balance: float,
    entry_fee: float   = 0.001,
    exit_fee: float    = 0.001,
    slippage_pct: float = 0.0005,  # slippage réaliste 0.05% par côté (market order)
    alloc_pct: float   = 0.10,      # 10% capital par trade
    max_positions: int = 2,          # max 2 positions simultanées (plus conservateur)
    cooldown_bars: int = 16,          # 4h cooldown après perte
    daily_dd_max: float = 0.05,
    # ─ BULL (logique TF pullback-EMA20, inspirée TF_S10_RSI_PYR) ─
    bull_rsi_min: float      = 50.0,   # RSI plancher d'entrée
    bull_rsi_max: float      = 65.0,   # RSI plafond d'entrée (non-suracheté)
    bull_sl_pct: float       = 0.015,  # SL fixe = -1.5% (close-only, comme TF_S10_RSI_PYR)
    bull_trail_pct: float    = 0.025,  # trailing élargi = -2.5% du peak → laisse courir
    bull_tp_pct: float       = 0.080,  # TP élargi = +8% → R:R ≈ 5.3 (filtré par trail)
    bull_slope_bars: int     = 10,     # slope : EMA20 15m hausse sur N barres
    bull_slope_min_pct: float = 0.001, # slope minimum : +0.1% sur slope_bars
    bull_pullback_bars: int  = 3,      # fenêtre pullback vers EMA50 (45 min)
    bull_alloc_pct: float    = 0.50,   # allocation BULL élargie (50% du capital — asymétrie haussière)
    bull_pyramid_alloc: float = 0.15,  # pyramiding : +15% sur position gagnante
    # ─ RANGE ─
    range_bars: int = 20,
    range_bot_pct: float = 0.12,
    range_rsi_max: float = 35.0,
    range_sl_pct: float  = 0.008,       # SL -0.8%
    range_tp_ratio: float = 0.50,       # TP = 50% du range
    # ─ STAGNATION ─
    stag_rsi_max: float    = 45.0,
    stag_sl_pct: float     = 0.004,
    stag_tp_pct: float     = 0.010,     # TP +1.0% → R:R 2.5
    stag_timeout_bars: int = 10,
    stag_min_atr_fees: float = 2.0,
    # ─ Contrôle des régimes actifs ─
    bull_only: bool = False,    # True = BULL uniquement (RANGE + STAG désactivés)
    # ─ Améliorations v2 ─
    use_atr_sl: bool = False,               # SL = entry - atr_sl_mult × ATR (au lieu de % fixe)
    atr_sl_mult: float = 1.5,
    use_atr_trail: bool = False,            # trail_dist = max(trail_pct, atr_trail_mult × ATR/peak)
    atr_trail_mult: float = 1.5,
    use_volume_filter: bool = False,        # signal BULL valide ssi volume > MA20 × vol_spike_mult
    vol_spike_mult: float = 1.2,
    use_conditional_pyramid: bool = False,  # pyramiding uniquement si (TP-prix)/TP > pyramid_remaining_pct
    pyramid_remaining_pct: float = 0.50,
    use_progressive_cooldown: bool = False, # après circuit-breaker DD: cooldown × 3 le lendemain
) -> tuple[float, list[AdaptiveTrade], list[float]]:
    """Stratégie adaptative — BULL / RANGE / STAGNATION uniquement (BEAR = pas de trade)."""

    n15  = len(candles_15m)
    n1h  = len(candles_1h)
    if n15 < 250 or n1h < 60:
        return initial_balance, [], [initial_balance] * n15

    # ── Indicateurs 1H ──────────────────────────────────────────────────
    closes_1h   = [c.close for c in candles_1h]
    ema20_h     = _ema(closes_1h, 20)
    ema50_h     = _ema(closes_1h, 50)
    ema200_h    = _ema(closes_1h, 200)
    adx_h       = _adx(candles_1h, 14)
    rsi_h       = _rsi(closes_1h, 14)
    atr_h       = _atr(candles_1h, 14)
    atr_ma50_h  = _sma(atr_h, 50)
    bb_width_h  = _bollinger_width(closes_1h, 20)

    # ── Indicateurs 15m ──────────────────────────────────────────────────
    closes_15m  = [c.close for c in candles_15m]
    opens_15m   = [c.open  for c in candles_15m]
    lows_15m    = [c.low   for c in candles_15m]
    volumes_15m = [c.volume for c in candles_15m]
    ema20_15    = _ema(closes_15m, 20)   # pour RANGE / STAGNATION uniquement
    ema50_15    = _ema(closes_15m, 50)   # BULL fast (EMA50 = "EMA fast" dans TF_S10_RSI_PYR)
    ema200_15   = _ema(closes_15m, 200)  # BULL slow (EMA200 = "EMA slow"  dans TF_S10_RSI_PYR)
    rsi_15      = _rsi(closes_15m, 14)
    atr_15      = _atr(candles_15m, 14)
    vol_ma20_15 = _sma(volumes_15m, 20)

    # Index de correspondance 15m → 1H  (bisect O(log n))
    ts_1h = [c.timestamp for c in candles_1h]

    def _get_1h_idx(ts_ms: int) -> int:
        return bisect.bisect_right(ts_1h, ts_ms) - 1

    # ── Loop principale ──────────────────────────────────────────────────
    balance:   float                   = initial_balance
    equity:    list[float]             = [initial_balance] * n15
    trades:    list[AdaptiveTrade]     = []
    positions: list[_Position]         = []
    cooldown   = 0
    daily_start_balance = initial_balance
    last_day: int | None = None
    circuit_breaker_hit = False  # flag pour progressive_cooldown

    for i in range(60, n15):
        c     = candles_15m[i]
        price = c.close
        high  = c.high
        low   = c.low
        vol   = volumes_15m[i]

        # ── DD journalier ────────────────────────────────────────────────
        day_key = c.timestamp // 86_400_000
        if day_key != last_day:
            last_day            = day_key
            daily_start_balance = balance + sum(p.cost for p in positions)
            # Progressive cooldown : si circuit-breaker touché hier → cooldown × 3
            if use_progressive_cooldown and circuit_breaker_hit:
                cooldown = max(cooldown, cooldown_bars * 3)
                circuit_breaker_hit = False

        daily_equity = balance + sum(p.size * price for p in positions)
        daily_dd = (daily_equity - daily_start_balance) / daily_start_balance if daily_start_balance > 0 else 0.0
        if daily_dd <= -daily_dd_max:
            circuit_breaker_hit = True

        # ── Régime 1H ────────────────────────────────────────────────────
        h_idx  = _get_1h_idx(c.timestamp)
        regime = (
            _detect_regime(
                candles_1h, h_idx,
                ema20_h, ema50_h, ema200_h,
                adx_h, rsi_h, atr_h, atr_ma50_h, bb_width_h,
            )
            if h_idx >= 50 else Regime.UNKNOWN
        )

        # ── Mise à jour positions ouvertes ───────────────────────────────
        still_open: list[_Position] = []
        for pos in positions:
            pos.bars_open += 1
            pos.peak       = max(pos.peak, price)   # peak sur CLOSE (cohérent avec vérifications SL/TP)
            exit_price     = None
            reason         = ""

            # Trailing stop BULL : -bull_trail_pct% du peak (actif dès entrée)
            if pos.regime == Regime.BULL:
                if use_atr_trail and atr_15[i] > 0:
                    trail_dist = max(bull_trail_pct, atr_trail_mult * atr_15[i] / pos.peak)
                    trail_sl = pos.peak * (1.0 - trail_dist)
                else:
                    trail_sl = pos.peak * (1.0 - bull_trail_pct)
                if trail_sl > pos.sl:
                    pos.sl = trail_sl

            # TP (RANGE, STAGNATION ET BULL avec TP fixe)
            if pos.tp > 0 and price >= pos.tp:
                exit_price = pos.tp
                reason     = "TP"

            # SL
            elif price <= pos.sl:
                exit_price = pos.sl
                reason     = "SL"

            # Timeout scalping stagnation
            elif pos.regime == Regime.STAGNATION and pos.bars_open >= stag_timeout_bars:
                exit_price = price
                reason     = "TIMEOUT"

            # Sortie BULL : tendance cassée (EMA50 passe sous EMA200 15m)
            elif pos.regime == Regime.BULL:
                ef15 = ema50_15[i]
                es15 = ema200_15[i]
                if ef15 > 0 and es15 > 0 and ef15 < es15:
                    exit_price = price
                    reason     = "TREND_BREAK"

            if exit_price is not None:
                effective_exit = exit_price * (1.0 - slippage_pct)
                net    = pos.size * effective_exit * (1.0 - exit_fee)
                pnl    = net - pos.cost
                is_win = pnl > 0
                if not is_win:
                    cooldown = max(cooldown, cooldown_bars)
                balance += net
                trades.append(AdaptiveTrade(
                    pos.entry, exit_price,
                    (exit_price - pos.entry) / pos.entry,
                    pnl, is_win, pos.regime, reason,
                ))
            else:
                still_open.append(pos)

        positions = still_open

        # ── Conditions globales d'entrée ─────────────────────────────────
        if cooldown > 0:
            cooldown -= 1

        can_enter = (
            cooldown == 0
            and len(positions) < max_positions
            and balance > 10.0
            and (regime == Regime.BULL if bull_only else regime not in (Regime.UNKNOWN, Regime.BEAR))
            and daily_dd > -daily_dd_max
        )

        if not can_enter:
            equity[i] = balance + sum(p.size * price for p in positions)
            continue

        rsi   = rsi_15[i]
        e20   = ema20_15[i]      # pour RANGE / STAGNATION
        e50   = ema50_15[i]      # BULL : EMA fast
        e200  = ema200_15[i]     # BULL : EMA slow
        atr_v = atr_15[i]

        opened = False

        # ── BULL — Pullback EMA50 15m gatté par régime 1H BULL ──────────
        # Logique identique à TF_S10_RSI_PYR (PF 1.18 walk-forward validé)
        # EMA fast = EMA50 | EMA slow = EMA200 | pullback vers EMA50
        if regime == Regime.BULL and e50 > 0 and e200 > 0:
            # Tendance locale 15m : EMA50 > EMA200 (Golden Cross 15m)
            trend_ok = e50 > e200
            # Prix au-dessus de l'EMA50 (15m)
            price_ok = c.close > e50
            # RSI dans la zone momentum non-suracheté
            rsi_ok   = bull_rsi_min <= rsi <= bull_rsi_max
            # RSI rising sur 3 barres consécutives
            rsi_up   = i >= 2 and rsi_15[i] > rsi_15[i - 1] > rsi_15[i - 2]
            # Slope EMA50 : hausse d'au moins slope_min_pct% sur slope_bars barres
            slope_ok = False
            if i >= bull_slope_bars:
                ref = ema50_15[i - bull_slope_bars]
                slope_ok = ref > 0 and (e50 - ref) / ref >= bull_slope_min_pct
            # Pullback récent : l'une des N dernières bougies a son low ≤ EMA50 × (1 + 1.2%)
            pb_start = max(60, i - bull_pullback_bars)
            pullback_ok = any(
                lows_15m[j] <= ema50_15[j] * 1.012
                for j in range(pb_start, i)
                if ema50_15[j] > 0
            )
            # Bougie de reprise haussière (close > open et close > close[-1])
            bull_candle = c.close > c.open and c.close > closes_15m[i - 1]
            # Max 1 position BULL simultanée
            bull_open = sum(1 for p in positions if p.regime == Regime.BULL)
            # Volume filter v2 : volume > MA20 × vol_spike_mult
            vol_ok = not use_volume_filter or (
                vol_ma20_15[i] > 0 and vol >= vol_spike_mult * vol_ma20_15[i]
            )

            if trend_ok and price_ok and rsi_ok and rsi_up and slope_ok and pullback_ok and bull_candle and bull_open == 0 and vol_ok:
                cost = balance * bull_alloc_pct
                if cost > 1.0:
                    actual_entry = c.close * (1.0 + slippage_pct)
                    fee_in  = cost * entry_fee
                    size    = (cost - fee_in) / actual_entry
                    balance -= cost
                    # SL : basé sur ATR (v2) ou % fixe (baseline)
                    if use_atr_sl and atr_v > 0:
                        sl_price = actual_entry - atr_sl_mult * atr_v
                        sl_price = max(sl_price, actual_entry * 0.95)  # hard cap -5%
                    else:
                        sl_price = actual_entry * (1.0 - bull_sl_pct)
                    positions.append(_Position(
                        entry=actual_entry, size=size, cost=cost,
                        sl=sl_price,
                        tp=actual_entry * (1.0 + bull_tp_pct),
                        peak=actual_entry, regime=Regime.BULL,
                    ))
                    opened = True

            # ── Pyramiding BULL ──────────────────────────────────────────
            for pos in positions:
                # Pyramiding conditionnel v2 : n'ajouter que si encore ≥ X% du chemin vers TP
                remaining_to_tp = (pos.tp - c.close) / pos.tp if pos.tp > 0 else 0.0
                pyramid_ok = not use_conditional_pyramid or remaining_to_tp > pyramid_remaining_pct
                if (
                    pos.regime == Regime.BULL
                    and not pos.trail_active    # réutilise trail_active comme flag pyramided
                    and c.close > pos.entry * 1.005
                    and rsi > 55
                    and trend_ok
                    and slope_ok
                    and balance > 10.0
                    and pyramid_ok
                ):
                    extra = balance * bull_pyramid_alloc
                    if extra > 1.0:
                        pyr_entry = c.close * (1.0 + slippage_pct)
                        fee_in = extra * entry_fee
                        extra_size = (extra - fee_in) / pyr_entry
                        total_cost = pos.cost + extra
                        total_size = pos.size + extra_size
                        pos.entry     = total_cost / total_size
                        pos.size      = total_size
                        pos.cost      = total_cost
                        pos.trail_active = True   # flag pyramided
                        balance      -= extra

        # ── RANGE — Mean Reversion (support/résistance sur 15m, plus précis) ──
        elif regime == Regime.RANGE:
            # 40 bougies 15m = ~10h : même fenêtre temporelle que 20×1H mais granularité 15m
            win_start = max(0, i - 40)
            if i - win_start >= 10:
                support    = min(lows_15m[j]               for j in range(win_start, i))
                resistance = max(candles_15m[j].high       for j in range(win_start, i))
                rng_size   = resistance - support
                if rng_size > 0:
                    pos_pct    = (price - support) / rng_size
                    in_bot     = pos_pct < range_bot_pct
                    rsi_ok     = rsi < range_rsi_max
                    # Rejet haussier proche du support 15m
                    rejection  = c.close > c.open and low <= support * 1.005   # tolérance réduite (15m est plus précis)

                    if in_bot and rsi_ok and rejection:
                        cost = balance * alloc_pct
                        if cost > 1.0:
                            actual_entry = price * (1.0 + slippage_pct)
                            fee_in  = cost * entry_fee
                            size    = (cost - fee_in) / actual_entry
                            balance -= cost
                            tp_price = support + rng_size * range_tp_ratio
                            positions.append(_Position(
                                entry=actual_entry, size=size, cost=cost,
                                sl=actual_entry * (1.0 - range_sl_pct),
                                tp=tp_price,
                                peak=actual_entry, regime=Regime.RANGE,
                            ))
                            opened = True

        # ── STAGNATION — Scalp micro-support ─────────────────────────────
        elif regime == Regime.STAGNATION:
            min_atr = price * (entry_fee + exit_fee) * stag_min_atr_fees
            if atr_v >= min_atr:
                rsi_ok     = rsi < stag_rsi_max
                stable_vol = atr_v <= (atr_15[max(0, i - 5)] or atr_v) * 1.1
                bull_micro = c.close > c.open and (c.close - c.open) / c.open < 0.003

                if rsi_ok and stable_vol and bull_micro:
                    cost = balance * alloc_pct * 0.4    # taille réduite scalp (asymétrie : moins agressif hors BULL)
                    if cost > 1.0:
                        actual_entry = price * (1.0 + slippage_pct)
                        fee_in  = cost * entry_fee
                        size    = (cost - fee_in) / actual_entry
                        balance -= cost
                        positions.append(_Position(
                            entry=actual_entry, size=size, cost=cost,
                            sl=actual_entry * (1.0 - stag_sl_pct),
                            tp=actual_entry * (1.0 + stag_tp_pct),
                            peak=actual_entry, regime=Regime.STAGNATION,
                        ))

        equity[i] = balance + sum(p.size * price for p in positions)

    # ── Clôture forcée fin de backtest ───────────────────────────────────
    if candles_15m:
        last_price = candles_15m[-1].close
        for pos in positions:
            effective_last = last_price * (1.0 - slippage_pct)
            net    = pos.size * effective_last * (1.0 - exit_fee)
            pnl    = net - pos.cost
            balance += net
            trades.append(AdaptiveTrade(
                pos.entry, last_price,
                (last_price - pos.entry) / pos.entry,
                pnl, pnl > 0, pos.regime, "END",
            ))

    return balance, trades, equity


# ═══════════════════════════════════════════════════════════════════════════
# Métriques + affichage
# ═══════════════════════════════════════════════════════════════════════════

def _compute_metrics(
    trades: list[AdaptiveTrade],
    equity: list[float],
    initial: float,
) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "final": initial, "dd": 0.0,
                "by_regime": {}}

    wins   = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    gp = sum(t.pnl_abs for t in wins)
    gl = abs(sum(t.pnl_abs for t in losses)) or 1e-9

    peak = dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = min(dd, (v - peak) / peak)

    by_regime: dict[str, dict] = {}
    for r in Regime:
        rt = [t for t in trades if t.regime == r]
        if rt:
            rw = [t for t in rt if t.is_win]
            rg = sum(t.pnl_abs for t in rw)
            rl = abs(sum(t.pnl_abs for t in [t for t in rt if not t.is_win])) or 1e-9
            by_regime[r.value] = {
                "n":   len(rt),
                "wr":  len(rw) / len(rt),
                "pf":  rg / rl,
                "pnl": sum(t.pnl_abs for t in rt),
            }

    return {
        "n":         n,
        "wr":        len(wins) / n,
        "pf":        gp / gl,
        "final":     equity[-1] if equity else initial,
        "dd":        dd,
        "by_regime": by_regime,
    }


_W = 116

def _header(title: str, sub: str) -> None:
    print(f"\n{'═' * _W}")
    print(f"  {title}")
    print(f"  {sub}")
    print(f"{'═' * _W}")


def _print_results(
    all_trades: list[AdaptiveTrade],
    combined_equity: list[float],
    initial: float,
    pairs: list[str],
    per_pair_results: list[tuple[str, float, list[AdaptiveTrade]]],
) -> None:
    m = _compute_metrics(all_trades, combined_equity, initial)

    _header(
        "STRATÉGIE ADAPTATIVE MULTI-RÉGIMES",
        f"Timeframes : 1H (régime) + 15m (exécution) | Capital : ${initial:,.0f} | Big5 | Frais : 0.10%+0.10%",
    )

    # ── Résultat global ────────────────────────────────────────────────
    pnl = m["final"] - initial
    sign = "+" if pnl >= 0 else ""
    print(f"\n  ► GLOBAL : {m['n']} trades | WR {m['wr']:.1%} | PF {m['pf']:.2f} | "
          f"PnL ${sign}{pnl:.2f} ({sign}{pnl/initial*100:.1f}%) | DD {m['dd']:.1%}")

    # ── Par régime ───────────────────────────────────────────────────────
    print(f"\n  {'Régime':15s}  {'Trades':>7s}  {'WR':>6s}  {'PF':>6s}  {'PnL ($)':>10s}")
    print("  " + "─" * 55)
    for regime_name, rm in sorted(m["by_regime"].items(), key=lambda x: -x[1]["pnl"]):
        rp = rm["pnl"]
        rs = "+" if rp >= 0 else ""
        pf_str = f"{rm['pf']:.2f}" if rm["pf"] < 99 else "   ∞"
        print(f"  {regime_name:15s}  {rm['n']:7d}  {rm['wr']:5.1%}  {pf_str:>6s}  ${rs}{rp:.2f}")

    # ── Par paire ────────────────────────────────────────────────────────
    print(f"\n  {'Paire':12s}  {'Trades':>7s}  {'WR':>6s}  {'PnL ($)':>10s}  {'Final':>10s}")
    print("  " + "─" * 55)
    for pair, final_bal, pair_trades in sorted(per_pair_results, key=lambda x: -x[1]):
        pt = pair_trades
        if pt:
            pw = [t for t in pt if t.is_win]
            wr = len(pw) / len(pt)
            pp = sum(t.pnl_abs for t in pt)
            ps = "+" if pp >= 0 else ""
            print(f"  {pair:12s}  {len(pt):7d}  {wr:5.1%}  ${ps}{pp:.2f}{' '*(8-len(f'{pp:.2f}'))}  ${final_bal:.2f}")
        else:
            per = initial / len(pairs)
            print(f"  {pair:12s}  {'0':>7s}  {'—':>6s}  ${'0.00':>9s}  ${per:.2f}")

    # ── Distribution exits ────────────────────────────────────────────────
    exit_counts: dict[str, int] = {}
    for t in all_trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    print(f"\n  Sorties : " + "  ".join(f"{r}: {c}" for r, c in sorted(exit_counts.items())))


# ═══════════════════════════════════════════════════════════════════════════
# Grille comparative des 5 améliorations v2
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _V2Config:
    name: str
    use_atr_sl: bool = False
    use_atr_trail: bool = False
    use_volume_filter: bool = False
    use_conditional_pyramid: bool = False
    use_progressive_cooldown: bool = False


_V2_VARIANTS: list[_V2Config] = [
    _V2Config("BASELINE"),
    _V2Config("1_ATR_SL",          use_atr_sl=True),
    _V2Config("2_ATR_TRAIL",       use_atr_trail=True),
    _V2Config("3_VOL_FILTER",      use_volume_filter=True),
    _V2Config("4_COND_PYRAMID",    use_conditional_pyramid=True),
    _V2Config("5_PROG_COOLDOWN",   use_progressive_cooldown=True),
    _V2Config("ALL_5",             use_atr_sl=True, use_atr_trail=True,
                                   use_volume_filter=True, use_conditional_pyramid=True,
                                   use_progressive_cooldown=True),
]


def run_v2_grid(
    balance: float,
    start: datetime,
    end: datetime,
    pairs: list[str] = PAIRS_BIG5,
    label: str = "",
) -> None:
    """Compare BASELINE vs 5 améliorations v2 — téléchargement unique, N variantes."""
    per_pair = balance / len(pairs)
    n_years  = (end - start).days / 365.25
    tag      = label or f"{n_years:.1f} ans"

    print(f"\n📥 Données ({start.date()} → {end.date()})…")
    c1h_all:  dict[str, list[Candle]] = {}
    c15m_all: dict[str, list[Candle]] = {}
    for pair in pairs:
        c1h_all[pair]  = download_candles(pair, start, end, interval="1h")
        c15m_all[pair] = download_candles(pair, start, end, interval="15m")
        print(f"  ✓ {pair}: {len(c1h_all[pair]):,} 1H | {len(c15m_all[pair]):,} 15m")

    _W2 = 95
    print(f"\n{'═' * _W2}")
    print(f"  GRILLE AMÉLIORATIONS V2 — {tag} | ${balance:,.0f} | {start.date()} → {end.date()}")
    print(f"{'═' * _W2}")
    print(f"\n  {'Config':<22s}  {'Trades':>6s}  {'WR':>6s}  {'PF':>5s}  {'PnL':>10s}  {'CAGR':>7s}  {'DD':>7s}")
    print("  " + "─" * 68)

    for cfg in _V2_VARIANTS:
        all_trades: list[AdaptiveTrade] = []
        combined_eq: list[float]        = []
        final_bal = 0.0

        for pair in pairs:
            c15 = c15m_all.get(pair, [])
            c1h = c1h_all.get(pair, [])
            if not c15 or not c1h:
                final_bal += per_pair
                continue
            bal, trades, eq = _run_adaptive_pair(
                c15, c1h, per_pair,
                use_atr_sl=cfg.use_atr_sl,
                use_atr_trail=cfg.use_atr_trail,
                use_volume_filter=cfg.use_volume_filter,
                use_conditional_pyramid=cfg.use_conditional_pyramid,
                use_progressive_cooldown=cfg.use_progressive_cooldown,
            )
            all_trades.extend(trades)
            final_bal += bal
            combined_eq = [a + b for a, b in zip(combined_eq, eq)] if combined_eq else list(eq)

        n = len(all_trades)
        if n == 0:
            print(f"  ⬜ {cfg.name:<20s}  {'0':>6s}  {'─':>6s}  {'─':>5s}  {'$0.00':>10s}  {'─':>7s}  {'─':>7s}")
            continue

        wins   = [t for t in all_trades if t.is_win]
        losses = [t for t in all_trades if not t.is_win]
        gp     = sum(t.pnl_abs for t in wins)
        gl     = abs(sum(t.pnl_abs for t in losses)) or 1e-9
        pf     = gp / gl
        wr     = len(wins) / n
        total_pnl = final_bal - balance
        cagr   = (final_bal / balance) ** (1.0 / max(n_years, 0.01)) - 1.0 if balance > 0 else 0.0

        # DD sur equity combinée
        peak_eq = dd = 0.0
        for v in combined_eq:
            if v > peak_eq:
                peak_eq = v
            if peak_eq > 0:
                dd = min(dd, (v - peak_eq) / peak_eq)

        mark = "🟢" if pf > 1.2 and total_pnl > 0 else ("🟡" if total_pnl > 0 else "🔴")
        pf_s = f"{pf:.2f}" if pf < 99 else ">99"
        print(
            f"  {mark} {cfg.name:<20s}  {n:>6d}  {wr:>5.1%}  {pf_s:>5s}  "
            f"${total_pnl:>+9.2f}  {cagr:>+6.1%}  {dd:>6.1%}"
        )
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Runner multi-paires (baseline)
# ═══════════════════════════════════════════════════════════════════════════

def run_adaptive(
    balance: float,
    start: datetime,
    end: datetime,
    pairs: list[str] = PAIRS_BIG5,
) -> None:
    per_pair = balance / len(pairs)
    print(f"\n📥 Téléchargement 1H ({start.date()} → {end.date()})…")
    candles_1h_all: dict[str, list[Candle]] = {}
    for pair in pairs:
        cds = download_candles(pair, start, end, interval="1h")
        candles_1h_all[pair] = cds
        print(f"  ✓ {pair}: {len(cds):,} bougies 1H")

    print(f"\n📥 Téléchargement 15m…")
    candles_15m_all: dict[str, list[Candle]] = {}
    for pair in pairs:
        cds = download_candles(pair, start, end, interval="15m")
        candles_15m_all[pair] = cds
        print(f"  ✓ {pair}: {len(cds):,} bougies 15m")

    print(f"\n⚙️  Simulation adaptative…")
    all_trades: list[AdaptiveTrade] = []
    combined_equity: list[float] = []
    per_pair_results: list[tuple[str, float, list[AdaptiveTrade]]] = []

    for pair in pairs:
        c15 = candles_15m_all.get(pair, [])
        c1h = candles_1h_all.get(pair, [])
        if not c15 or not c1h:
            continue
        bal, trades, eq = _run_adaptive_pair(c15, c1h, per_pair)
        per_pair_results.append((pair, bal, trades))
        all_trades.extend(trades)
        if not combined_equity:
            combined_equity = list(eq)
        else:
            combined_equity = [a + b for a, b in zip(combined_equity, eq)]
        print(f"  ✓ {pair}: {len(trades)} trades | final ${bal:.2f}")

    _print_results(all_trades, combined_equity, balance, pairs, per_pair_results)


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balance",     type=float, default=1_000.0)
    parser.add_argument("--years",       type=int,   default=3)
    parser.add_argument("--start",       type=str,   default=None)
    parser.add_argument("--end",         type=str,   default=None)
    parser.add_argument("--pairs",       type=str,   default="big5",
                        help="big5 | candidates | all | PAIR1,PAIR2,...")
    parser.add_argument("--v2-grid",     action="store_true",
                        help="Grille comparative des 5 améliorations v2")
    parser.add_argument("--multi-years", action="store_true",
                        help="Lance la grille v2 sur 1, 3 et 6 ans successivement")
    parser.add_argument("--last-days",   type=int, default=0,
                        help="Comparaison live : backtest sur les N derniers jours (ex: 5)")
    args = parser.parse_args()

    if args.pairs == "big5":
        pairs = PAIRS_BIG5
    elif args.pairs == "candidates":
        pairs = PAIRS_CANDIDATES
    elif args.pairs == "all":
        pairs = PAIRS_BIG5 + PAIRS_CANDIDATES
    else:
        pairs = [p.strip() for p in args.pairs.split(",")]

    now = datetime.now(tz=timezone.utc)

    if args.last_days > 0:
        end   = now
        # Warmup 30j pour chauffer les indicateurs (EMA200 1H = 200h ≈ 8.3j)
        start = end - timedelta(days=args.last_days + 30)
        run_v2_grid(balance=args.balance, start=start, end=end, pairs=pairs,
                    label=f"LIVE {args.last_days}j (warmup 30j inclus)")
        return

    if args.v2_grid or args.multi_years:
        if args.multi_years:
            for y in (1, 3, 6):
                s = now - timedelta(days=365 * y)
                run_v2_grid(balance=args.balance, start=s, end=now, pairs=pairs,
                            label=f"{y} an{'s' if y > 1 else ''}")
        else:
            if args.start and args.end:
                start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
            else:
                end   = now
                start = end - timedelta(days=365 * args.years)
            run_v2_grid(balance=args.balance, start=start, end=end, pairs=pairs,
                        label=f"{args.years} ans")
        return

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        end   = now
        start = end - timedelta(days=365 * args.years)

    run_adaptive(balance=args.balance, start=start, end=end, pairs=pairs)


if __name__ == "__main__":
    main()
