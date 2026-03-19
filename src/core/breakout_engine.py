"""
Breakout Momentum Engine — logique pure (sans I/O).

Implémente la détection de breakout high(N) sur bougies 15m
avec trailing stop ATR, filtres volume / ATR expansion.
Adapté du backtest BRK_ULTRATRAIL validé en walk-forward.

Fonctions pures : reçoivent des données, retournent des résultats.
Aucun appel réseau, aucune dépendance I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.models import Candle


# ── Indicateurs techniques ─────────────────────────────────────────────────────


def compute_atr(candles: list[Candle], period: int = 14) -> list[float]:
    """Calcule l'ATR (Wilder) sur une liste de bougies.

    Retourne une liste de même longueur que *candles*.
    """
    if len(candles) < 2:
        return [0.0] * len(candles)

    trs: list[float] = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)

    atrs: list[float] = []
    for i, tr in enumerate(trs):
        if i < period:
            atrs.append(sum(trs[: i + 1]) / (i + 1))
        else:
            atrs.append((atrs[-1] * (period - 1) + tr) / period)
    return atrs


def compute_sma(values: list[float], period: int) -> list[float]:
    """Simple moving average."""
    result: list[float] = []
    for i in range(len(values)):
        if i < period - 1:
            result.append(sum(values[: i + 1]) / (i + 1))
        else:
            result.append(sum(values[i - period + 1 : i + 1]) / period)
    return result


def rolling_high(highs: list[float], lookback: int) -> list[float]:
    """Plus haut glissant sur *lookback* barres (exclut la barre courante)."""
    result: list[float] = []
    for i in range(len(highs)):
        if i < 1:
            result.append(highs[0])
        else:
            start = max(0, i - lookback)
            result.append(max(highs[start:i]))
    return result


# ── Résultat de signal ─────────────────────────────────────────────────────────


@dataclass
class BreakoutSignal:
    """Signal de breakout détecté."""

    entry_price: float
    sl_price: float
    tp_price: float
    atr_value: float
    trailing_activation: float  # prix auquel le trailing s'active
    trailing_distance: float    # distance en prix (pas en ATR)
    volume: float
    volume_ma: float
    recent_high: float


# ── Détection de breakout ──────────────────────────────────────────────────────


def detect_breakout(
    candles: list[Candle],
    *,
    lookback: int = 12,
    atr_period: int = 14,
    vol_ma_period: int = 20,
    tp_atr_mult: float = 2.0,
    sl_atr_mult: float = 0.8,
    trail_activation_atr: float = 0.3,
    trail_distance_atr: float = 0.2,
    atr_expansion_lookback: int = 8,
    atr_expansion_ratio: float = 1.05,
    volume_spike_mult: float = 1.0,
    min_atr_pct: float = 0.001,
) -> Optional[BreakoutSignal]:
    """Analyse la dernière bougie et retourne un signal si breakout confirmé.

    Conditions requises :
      1. Close > rolling high(lookback) des barres précédentes
      2. ATR(now) ≥ atr_expansion_ratio × ATR(N barres avant)
      3. Volume ≥ volume_spike_mult × SMA(volume, vol_ma_period)
      4. ATR ≥ min_atr_pct × prix (filtre marché mort)

    Retourne ``None`` si aucun signal.
    """
    warmup = max(lookback, atr_period, vol_ma_period, atr_expansion_lookback + atr_period) + 2
    if len(candles) < warmup:
        return None

    # Indicateurs
    atrs = compute_atr(candles, atr_period)
    volumes = [c.volume for c in candles]
    vol_mas = compute_sma(volumes, vol_ma_period)
    highs = [c.high for c in candles]
    rec_highs = rolling_high(highs, lookback)

    i = len(candles) - 1
    candle = candles[i]
    atr_val = atrs[i]
    vol_ma = vol_mas[i]
    rec_high = rec_highs[i]

    # Filtre 1 : ATR minimum
    if candle.close > 0 and atr_val / candle.close < min_atr_pct:
        return None

    # Filtre 2 : breakout du high récent
    if candle.close <= rec_high:
        return None

    # Filtre 3 : ATR expansion
    past_idx = i - atr_expansion_lookback
    if past_idx >= 0 and atrs[past_idx] > 0:
        if atr_val < atr_expansion_ratio * atrs[past_idx]:
            return None

    # Filtre 4 : volume spike
    if vol_ma > 0 and candle.volume < volume_spike_mult * vol_ma:
        return None

    # Signal confirmé → niveaux
    entry = candle.close
    sl = entry - sl_atr_mult * atr_val
    tp = entry + tp_atr_mult * atr_val
    trail_act = entry + trail_activation_atr * atr_val
    trail_dist = trail_distance_atr * atr_val

    if sl <= 0 or entry - sl <= 0:
        return None

    return BreakoutSignal(
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        atr_value=atr_val,
        trailing_activation=trail_act,
        trailing_distance=trail_dist,
        volume=candle.volume,
        volume_ma=vol_ma,
        recent_high=rec_high,
    )


# ── Trailing stop management ──────────────────────────────────────────────────


@dataclass
class TrailingResult:
    """Résultat de la mise à jour du trailing stop."""

    new_sl: float
    trailing_active: bool
    peak_price: float


def update_trailing_stop(
    current_price: float,
    entry_price: float,
    current_sl: float,
    peak_price: float,
    trailing_activation: float,
    trailing_distance: float,
) -> TrailingResult:
    """Met à jour le trailing stop si le prix a progressé.

    Le trailing s'active quand *current_price* ≥ *trailing_activation*.
    Une fois actif, le SL suit à ``peak - trailing_distance``, mais
    ne descend jamais en dessous de *current_sl*.

    Retourne le nouveau SL, si le trailing est actif, et le nouveau peak.
    """
    new_peak = max(peak_price, current_price)
    trailing_active = current_price >= trailing_activation

    if trailing_active:
        candidate_sl = new_peak - trailing_distance
        new_sl = max(current_sl, candidate_sl)
    else:
        new_sl = current_sl

    return TrailingResult(
        new_sl=new_sl,
        trailing_active=trailing_active,
        peak_price=new_peak,
    )
