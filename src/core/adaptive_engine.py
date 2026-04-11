"""
Adaptive Bull Engine — logique pure sans I/O.

Stratégie : Bull Trend Following 15m + filtre régime 1H
  - Régime BULL (score ≥ 4/5 : EMA20>EMA50>EMA200, ADX>22, RSI>55 sur 1H)
  - Entrée : pullback EMA50 15m + RSI 50-65 + slope EMA50 positif + bougie haussière
  - SL -1.5% | Trailing -2.5% du peak | TP +8%
  - Pyramiding : +15% sur position gagnante (1 fois par trade)
  - Pas d'appel réseau — 100% testable en isolation

Walk-forward validé 3/3, OOS PF 1.14, 6 ans +405% CAGR +31%, DD max -20.8%.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.core.models import Candle


# ═══════════════════════════════════════════════════════════════════════════
# Régimes de marché
# ═══════════════════════════════════════════════════════════════════════════

class Regime(Enum):
    BULL       = "BULL"
    BEAR       = "BEAR"
    RANGE      = "RANGE"
    STAGNATION = "STAGNATION"
    UNKNOWN    = "UNKNOWN"


# ═══════════════════════════════════════════════════════════════════════════
# Indicateurs (copie fidèle du backtest run_backtest_adaptive.py)
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
        h, l_, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l_, abs(h - pc), abs(l_ - pc)))
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
    """ADX simplifié (lissage Wilder)."""
    n = len(candles)
    result = [0.0] * n
    if n < period * 2 + 1:
        return result

    plus_dm  = [0.0] * n
    minus_dm = [0.0] * n
    tr_v     = [0.0] * n
    for i in range(1, n):
        h, l_  = candles[i].high, candles[i].low
        ph, pl = candles[i - 1].high, candles[i - 1].low
        pc     = candles[i - 1].close
        tr_v[i]     = max(h - l_, abs(h - pc), abs(l_ - pc))
        up, down    = h - ph, pl - l_
        plus_dm[i]  = up   if up > down  and up > 0   else 0.0
        minus_dm[i] = down if down > up  and down > 0 else 0.0

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
        mid = sum(window) / period
        std = (sum((x - mid) ** 2 for x in window) / period) ** 0.5
        result[i] = (2 * k * std / mid) if mid else 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Modèles de données
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BullSignal:
    """Signal d'entrée BULL détecté."""
    entry_price: float
    sl_price: float        # -1.5% de l'entrée
    tp_price: float        # +8% de l'entrée
    rsi: float
    ema50: float
    ema200: float


@dataclass
class PyramidSignal:
    """Signal d'ajout pyramidal sur une position existante."""
    entry_price: float
    extra_size_pct: float  # fraction du solde allouée


@dataclass
class RegimeDebug:
    """Détail complet du calcul de régime 1H — pour le logging diagnostique."""
    regime: Regime
    price: float
    ema20: float
    ema50: float
    ema200: float
    adx: float
    rsi_1h: float
    bull_score: int
    bear_score: int
    stagnation: bool


@dataclass
class EntryDebug:
    """Détail des conditions d'entrée 15m — pour le logging diagnostique."""
    signal: Optional[BullSignal]
    price: float
    ema50: float
    ema200: float
    rsi: float
    slope_pct: float
    cond_golden_cross: bool
    cond_price_ema50: bool
    cond_rsi_range: bool
    cond_rsi_rising: bool
    cond_slope: bool
    cond_pullback: bool
    cond_bull_candle: bool


@dataclass
class AdaptiveIndicators:
    """Indicateurs calculés sur les dernières bougies 15m et 1H."""
    # 15m
    ema50_15m: float
    ema200_15m: float
    rsi_15m: float
    lows_15m: list[float]    # tous les lows (pour détection pullback)
    ema50_arr_15m: list[float]
    closes_15m: list[float]
    rsi_arr_15m: list[float]
    # 1H
    regime: Regime


# ═══════════════════════════════════════════════════════════════════════════
# Détection de régime (1H)
# ═══════════════════════════════════════════════════════════════════════════

def detect_regime(candles_1h: list[Candle]) -> Regime:
    """
    Détecte le régime de marché à partir des bougies 1H.

    BULL  : score ≥ 4/5 (EMA alignment + ADX + RSI)
    BEAR  : score ≥ 4/5 (inverse)
    RANGE : ADX < 18
    Sinon : UNKNOWN

    Nécessite au moins 220 bougies 1H (≈ 9 jours) pour les EMAs.
    """
    if len(candles_1h) < 220:
        return Regime.UNKNOWN

    closes_1h   = [c.close for c in candles_1h]
    ema20_h     = _ema(closes_1h, 20)
    ema50_h     = _ema(closes_1h, 50)
    ema200_h    = _ema(closes_1h, 200)
    adx_h       = _adx(candles_1h, 14)
    rsi_h       = _rsi(closes_1h, 14)
    atr_h       = _atr(candles_1h, 14)
    atr_ma50_h  = _sma(atr_h, 50)
    bb_width_h  = _bollinger_width(closes_1h, 20)

    idx = len(candles_1h) - 1
    if idx < 50:
        return Regime.UNKNOWN

    price  = candles_1h[idx].close
    e20    = ema20_h[idx]
    e50    = ema50_h[idx]
    e200   = ema200_h[idx]
    adx    = adx_h[idx]
    rsi    = rsi_h[idx]
    atr    = atr_h[idx]
    atr_ma = atr_ma50_h[idx]
    bbw    = bb_width_h[idx]

    # Stagnation (priorité maximale — marché compressé)
    if atr_ma > 0 and atr < 0.8 * atr_ma and bbw < 0.015:
        return Regime.STAGNATION

    # BULL (score ≥ 4)
    bull_score = 0
    if e50 > 0 and price > e50:             bull_score += 1
    if e20 > 0 and e50 > 0 and e20 > e50:  bull_score += 1
    if e50 > 0 and e200 > 0 and e50 > e200: bull_score += 1
    if adx > 22:                             bull_score += 1
    if rsi > 55:                             bull_score += 1
    if bull_score >= 4:
        return Regime.BULL

    # BEAR (score ≥ 4)
    bear_score = 0
    if e50 > 0 and price < e50:             bear_score += 1
    if e20 > 0 and e50 > 0 and e20 < e50:  bear_score += 1
    if e50 > 0 and e200 > 0 and e50 < e200: bear_score += 1
    if adx > 22:                             bear_score += 1
    if rsi < 45:                             bear_score += 1
    if bear_score >= 4:
        return Regime.BEAR

    # RANGE (ADX faible)
    if adx < 18:
        return Regime.RANGE

    return Regime.UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# Signal d'entrée BULL (15m)
# ═══════════════════════════════════════════════════════════════════════════

def check_bull_entry(
    candles_15m: list[Candle],
    regime: Regime,
    bull_rsi_min: float = 50.0,
    bull_rsi_max: float = 65.0,
    bull_sl_pct: float = 0.015,
    bull_tp_pct: float = 0.080,
    bull_slope_bars: int = 10,
    bull_slope_min_pct: float = 0.001,
    bull_pullback_bars: int = 3,
) -> Optional[BullSignal]:
    """
    Retourne un BullSignal si toutes les conditions d'entrée BULL sont réunies.

    Conditions (identiques au backtest validé walk-forward 3/3) :
    1. Régime 1H = BULL
    2. Golden Cross 15m : EMA50 > EMA200
    3. Prix > EMA50 (15m)
    4. RSI 15m dans [bull_rsi_min, bull_rsi_max]
    5. RSI hausse sur 3 barres consécutives
    6. Slope EMA50 ≥ +bull_slope_min_pct% sur bull_slope_bars barres
    7. Pullback récent : un low des N dernières barres ≤ EMA50 × 1.012
    8. Bougie haussière (close > open et close > close[-1])

    Retourne None si conditions non réunies ou données insuffisantes.
    """
    if regime != Regime.BULL:
        return None

    n = len(candles_15m)
    min_bars = max(250, bull_slope_bars + 20)
    if n < min_bars:
        return None

    idx = n - 1
    closes  = [c.close for c in candles_15m]
    lows    = [c.low   for c in candles_15m]

    ema50_arr  = _ema(closes, 50)
    ema200_arr = _ema(closes, 200)
    rsi_arr    = _rsi(closes, 14)

    e50  = ema50_arr[idx]
    e200 = ema200_arr[idx]
    rsi  = rsi_arr[idx]

    if e50 <= 0 or e200 <= 0:
        return None

    c = candles_15m[idx]
    price = c.close

    # 1. Golden Cross 15m
    trend_ok = e50 > e200
    if not trend_ok:
        return None

    # 2. Prix > EMA50
    if price <= e50:
        return None

    # 3. RSI dans zone cible
    if not (bull_rsi_min <= rsi <= bull_rsi_max):
        return None

    # 4. RSI hausse sur 3 barres consécutives
    if idx < 2:
        return None
    rsi_up = rsi_arr[idx] > rsi_arr[idx - 1] > rsi_arr[idx - 2]
    if not rsi_up:
        return None

    # 5. Slope EMA50 positif
    if idx < bull_slope_bars:
        return None
    ref = ema50_arr[idx - bull_slope_bars]
    slope_ok = ref > 0 and (e50 - ref) / ref >= bull_slope_min_pct
    if not slope_ok:
        return None

    # 6. Pullback récent vers EMA50 (tolérance +1.2%)
    pb_start = max(60, idx - bull_pullback_bars)
    pullback_ok = any(
        lows[j] <= ema50_arr[j] * 1.012
        for j in range(pb_start, idx)
        if ema50_arr[j] > 0
    )
    if not pullback_ok:
        return None

    # 7. Bougie haussière de reprise
    bull_candle = c.close > c.open and c.close > closes[idx - 1]
    if not bull_candle:
        return None

    # Signal validé
    return BullSignal(
        entry_price=price,
        sl_price=price * (1.0 - bull_sl_pct),
        tp_price=price * (1.0 + bull_tp_pct),
        rsi=rsi,
        ema50=e50,
        ema200=e200,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Mise à jour du trailing stop
# ═══════════════════════════════════════════════════════════════════════════

def update_adaptive_trailing(
    current_price: float,
    peak_price: float,
    current_sl: float,
    bull_trail_pct: float = 0.025,
) -> tuple[float, float]:
    """
    Met à jour le trailing stop BULL.

    Args:
        current_price: Prix courant.
        peak_price:    Plus haut atteint depuis l'entrée.
        current_sl:    SL actuel (initial ou trailing précédent).
        bull_trail_pct: Distance trailing en % du peak.

    Returns:
        (new_sl, new_peak) — le SL ne peut que monter, jamais descendre.
    """
    new_peak = max(peak_price, current_price)
    trail_sl = new_peak * (1.0 - bull_trail_pct)
    new_sl   = max(current_sl, trail_sl)
    return new_sl, new_peak


# ═══════════════════════════════════════════════════════════════════════════
# Détection de rupture de tendance (sortie BULL)
# ═══════════════════════════════════════════════════════════════════════════

def is_trend_broken(candles_15m: list[Candle]) -> bool:
    """
    Retourne True si la tendance 15m est cassée (EMA50 < EMA200).
    Utilisé comme signal de sortie d'urgence pour les positions BULL.
    """
    n = len(candles_15m)
    if n < 210:
        return False

    closes   = [c.close for c in candles_15m]
    ema50_a  = _ema(closes, 50)
    ema200_a = _ema(closes, 200)
    idx = n - 1
    e50  = ema50_a[idx]
    e200 = ema200_a[idx]
    return e50 > 0 and e200 > 0 and e50 < e200


# ═══════════════════════════════════════════════════════════════════════════
# Signal pyramidal
# ═══════════════════════════════════════════════════════════════════════════

def check_pyramid_entry(
    candles_15m: list[Candle],
    regime: Regime,
    entry_price: float,
    bull_slope_bars: int = 10,
    bull_slope_min_pct: float = 0.001,
) -> bool:
    """
    Retourne True si les conditions de pyramiding BULL sont réunies.

    Conditions :
    - Régime BULL
    - Close > entry × 1.005 (+0.5%)
    - RSI > 55
    - Golden Cross 15m (EMA50 > EMA200)
    - Slope EMA50 positif
    """
    if regime != Regime.BULL:
        return False

    n = len(candles_15m)
    if n < 210:
        return False

    idx = n - 1
    closes   = [c.close for c in candles_15m]
    ema50_a  = _ema(closes, 50)
    ema200_a = _ema(closes, 200)
    rsi_a    = _rsi(closes, 14)

    e50  = ema50_a[idx]
    e200 = ema200_a[idx]
    rsi  = rsi_a[idx]
    price = candles_15m[idx].close

    trend_ok = e50 > 0 and e200 > 0 and e50 > e200
    price_ok = price > entry_price * 1.005
    rsi_ok   = rsi > 55

    slope_ok = False
    if idx >= bull_slope_bars:
        ref = ema50_a[idx - bull_slope_bars]
        slope_ok = ref > 0 and (e50 - ref) / ref >= bull_slope_min_pct

    return trend_ok and price_ok and rsi_ok and slope_ok


# ═══════════════════════════════════════════════════════════════════════════
# Fonctions debug / logging (exposent les calculs intermédiaires)
# ═══════════════════════════════════════════════════════════════════════════

def detect_regime_debug(candles_1h: list[Candle]) -> RegimeDebug:
    """
    Identique à detect_regime() mais retourne tous les indicateurs intermédiaires
    pour le logging diagnostique par bougie.
    """
    _empty = RegimeDebug(
        regime=Regime.UNKNOWN, price=0.0, ema20=0.0, ema50=0.0, ema200=0.0,
        adx=0.0, rsi_1h=50.0, bull_score=0, bear_score=0, stagnation=False,
    )
    if len(candles_1h) < 220:
        return _empty

    closes_1h  = [c.close for c in candles_1h]
    ema20_h    = _ema(closes_1h, 20)
    ema50_h    = _ema(closes_1h, 50)
    ema200_h   = _ema(closes_1h, 200)
    adx_h      = _adx(candles_1h, 14)
    rsi_h      = _rsi(closes_1h, 14)
    atr_h      = _atr(candles_1h, 14)
    atr_ma50_h = _sma(atr_h, 50)
    bb_width_h = _bollinger_width(closes_1h, 20)

    idx = len(candles_1h) - 1
    if idx < 50:
        return _empty

    price  = candles_1h[idx].close
    e20    = ema20_h[idx]
    e50    = ema50_h[idx]
    e200   = ema200_h[idx]
    adx    = adx_h[idx]
    rsi    = rsi_h[idx]
    atr    = atr_h[idx]
    atr_ma = atr_ma50_h[idx]
    bbw    = bb_width_h[idx]

    stagnation = atr_ma > 0 and atr < 0.8 * atr_ma and bbw < 0.015

    bull_score = 0
    if e50 > 0 and price > e50:              bull_score += 1
    if e20 > 0 and e50 > 0 and e20 > e50:   bull_score += 1
    if e50 > 0 and e200 > 0 and e50 > e200: bull_score += 1
    if adx > 22:                              bull_score += 1
    if rsi > 55:                              bull_score += 1

    bear_score = 0
    if e50 > 0 and price < e50:              bear_score += 1
    if e20 > 0 and e50 > 0 and e20 < e50:   bear_score += 1
    if e50 > 0 and e200 > 0 and e50 < e200: bear_score += 1
    if adx > 22:                              bear_score += 1
    if rsi < 45:                              bear_score += 1

    if stagnation:
        regime = Regime.STAGNATION
    elif bull_score >= 4:
        regime = Regime.BULL
    elif bear_score >= 4:
        regime = Regime.BEAR
    elif adx < 18:
        regime = Regime.RANGE
    else:
        regime = Regime.UNKNOWN

    return RegimeDebug(
        regime=regime, price=price,
        ema20=e20, ema50=e50, ema200=e200,
        adx=adx, rsi_1h=rsi,
        bull_score=bull_score, bear_score=bear_score,
        stagnation=stagnation,
    )


def check_bull_entry_debug(
    candles_15m: list[Candle],
    regime: Regime,
    bull_rsi_min: float = 50.0,
    bull_rsi_max: float = 65.0,
    bull_sl_pct: float = 0.015,
    bull_tp_pct: float = 0.080,
    bull_slope_bars: int = 10,
    bull_slope_min_pct: float = 0.001,
    bull_pullback_bars: int = 3,
) -> EntryDebug:
    """
    Identique à check_bull_entry() mais retourne toutes les conditions intermédiaires
    pour le logging diagnostique par bougie.
    """
    def _no_signal(
        ema50: float = 0.0, ema200: float = 0.0,
        rsi: float = 50.0, slope_pct: float = 0.0,
    ) -> EntryDebug:
        return EntryDebug(
            signal=None, price=0.0, ema50=ema50, ema200=ema200,
            rsi=rsi, slope_pct=slope_pct,
            cond_golden_cross=False, cond_price_ema50=False,
            cond_rsi_range=False, cond_rsi_rising=False,
            cond_slope=False, cond_pullback=False, cond_bull_candle=False,
        )

    n = len(candles_15m)
    min_bars = max(250, bull_slope_bars + 20)
    if n < min_bars:
        return _no_signal()

    idx    = n - 1
    closes = [c.close for c in candles_15m]
    lows   = [c.low   for c in candles_15m]

    ema50_arr  = _ema(closes, 50)
    ema200_arr = _ema(closes, 200)
    rsi_arr    = _rsi(closes, 14)

    e50  = ema50_arr[idx]
    e200 = ema200_arr[idx]
    rsi  = rsi_arr[idx]

    if e50 <= 0 or e200 <= 0:
        return _no_signal()

    c     = candles_15m[idx]
    price = c.close

    cond_golden_cross = e50 > e200
    cond_price_ema50  = price > e50
    cond_rsi_range    = bull_rsi_min <= rsi <= bull_rsi_max
    cond_rsi_rising   = (
        idx >= 2 and rsi_arr[idx] > rsi_arr[idx - 1] > rsi_arr[idx - 2]
    )

    slope_pct = 0.0
    cond_slope = False
    if idx >= bull_slope_bars:
        ref = ema50_arr[idx - bull_slope_bars]
        if ref > 0:
            slope_pct = (e50 - ref) / ref
            cond_slope = slope_pct >= bull_slope_min_pct

    pb_start     = max(60, idx - bull_pullback_bars)
    cond_pullback = any(
        lows[j] <= ema50_arr[j] * 1.012
        for j in range(pb_start, idx)
        if ema50_arr[j] > 0
    )
    cond_bull_candle = c.close > c.open and c.close > closes[idx - 1]

    all_ok = (
        regime == Regime.BULL
        and cond_golden_cross and cond_price_ema50 and cond_rsi_range
        and cond_rsi_rising and cond_slope and cond_pullback and cond_bull_candle
    )
    signal = BullSignal(
        entry_price=price,
        sl_price=price * (1.0 - bull_sl_pct),
        tp_price=price * (1.0 + bull_tp_pct),
        rsi=rsi, ema50=e50, ema200=e200,
    ) if all_ok else None

    return EntryDebug(
        signal=signal, price=price,
        ema50=e50, ema200=e200, rsi=rsi, slope_pct=slope_pct,
        cond_golden_cross=cond_golden_cross,
        cond_price_ema50=cond_price_ema50,
        cond_rsi_range=cond_rsi_range,
        cond_rsi_rising=cond_rsi_rising,
        cond_slope=cond_slope,
        cond_pullback=cond_pullback,
        cond_bull_candle=cond_bull_candle,
    )
