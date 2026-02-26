"""
Breakout Volatility Expansion — détection de signaux.

Stratégie complémentaire aux bots RANGE : gagne quand le range casse.
Combine 4 filtres :
  1. Bollinger Band Width en expansion (volatilité montante)
  2. Breakout Donchian Channel (prix sort du canal N périodes)
  3. ADX > seuil (force de tendance confirmée)
  4. Volume > moyenne (participation du marché)

Pas d'I/O — logique pure, testable sans mock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.core.models import Candle, OrderSide


# ── Structures ─────────────────────────────────────────────────────────────────


class BreakoutDirection(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class BreakoutSignal:
    """Signal émis quand les 4 filtres convergent."""
    direction: BreakoutDirection
    entry_price: float          # prix de clôture de la bougie de breakout
    sl_price: float             # SL initial (sous le low / au-dessus du high)
    donchian_high: float        # borne haute Donchian
    donchian_low: float         # borne basse Donchian
    bb_width: float             # largeur Bollinger normalisée
    adx: float                  # valeur ADX courante
    volume_ratio: float         # volume / avg volume
    candle_index: int           # index de la bougie de signal


@dataclass
class BreakoutConfig:
    """Paramètres de la stratégie Breakout Volatility Expansion."""
    # Bollinger
    bb_period: int = 20         # période SMA pour Bollinger
    bb_std: float = 2.0         # nombre d'écarts-types
    bb_width_expansion: float = 1.2   # BB width doit être > expansion * BB width moy

    # Donchian
    donchian_period: int = 20   # canal N bougies

    # ADX
    adx_period: int = 14        # période ADX
    adx_threshold: float = 25.0 # seuil minimum ADX

    # Volume
    vol_avg_period: int = 20    # moyenne volume
    vol_multiplier: float = 1.2 # volume doit être > mult * avg

    # SL
    sl_atr_mult: float = 1.5   # SL = ATR * mult sous/au-dessus de l'entrée
    atr_period: int = 14        # ATR pour SL

    # Direction
    allow_short: bool = True    # Autoriser les shorts


# ── Indicateurs techniques ─────────────────────────────────────────────────────


def sma(values: list[float], period: int) -> list[Optional[float]]:
    """Simple Moving Average. Retourne None pour les premières valeurs."""
    result: list[Optional[float]] = [None] * len(values)
    if len(values) < period:
        return result
    window_sum = sum(values[:period])
    result[period - 1] = window_sum / period
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        result[i] = window_sum / period
    return result


def std_dev(values: list[float], period: int) -> list[Optional[float]]:
    """Écart-type glissant."""
    result: list[Optional[float]] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        avg = sum(window) / period
        var = sum((x - avg) ** 2 for x in window) / period
        result[i] = math.sqrt(var)
    return result


def bollinger_bands(
    closes: list[float], period: int = 20, num_std: float = 2.0,
) -> tuple[list[Optional[float]], list[Optional[float]], list[Optional[float]]]:
    """Retourne (upper, middle, lower)."""
    mid = sma(closes, period)
    sd = std_dev(closes, period)
    upper: list[Optional[float]] = [None] * len(closes)
    lower: list[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if mid[i] is not None and sd[i] is not None:
            upper[i] = mid[i] + num_std * sd[i]
            lower[i] = mid[i] - num_std * sd[i]
    return upper, mid, lower


def bb_width(
    upper: list[Optional[float]],
    middle: list[Optional[float]],
    lower: list[Optional[float]],
) -> list[Optional[float]]:
    """BB Width normalisé : (upper - lower) / middle."""
    result: list[Optional[float]] = [None] * len(upper)
    for i in range(len(upper)):
        if upper[i] is not None and lower[i] is not None and middle[i] and middle[i] > 0:
            result[i] = (upper[i] - lower[i]) / middle[i]
    return result


def donchian_channel(
    candles: list[Candle], period: int,
) -> tuple[list[Optional[float]], list[Optional[float]]]:
    """Donchian Channel : (highs, lows) = max/min des N bougies précédentes.

    IMPORTANT : on utilise les N bougies *précédentes* (shift de 1),
    pas la bougie courante — sinon c'est du lookahead.
    """
    n = len(candles)
    dc_high: list[Optional[float]] = [None] * n
    dc_low: list[Optional[float]] = [None] * n
    for i in range(period, n):
        # Fenêtre [i-period, i) — exclut la bougie courante
        window = candles[i - period : i]
        dc_high[i] = max(c.high for c in window)
        dc_low[i] = min(c.low for c in window)
    return dc_high, dc_low


def true_range(candles: list[Candle]) -> list[float]:
    """True Range pour chaque bougie."""
    tr: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr.append(c.high - c.low)
        else:
            prev_close = candles[i - 1].close
            tr.append(max(
                c.high - c.low,
                abs(c.high - prev_close),
                abs(c.low - prev_close),
            ))
    return tr


def atr(candles: list[Candle], period: int = 14) -> list[Optional[float]]:
    """Average True Range (EMA-smoothed)."""
    tr_values = true_range(candles)
    result: list[Optional[float]] = [None] * len(candles)
    if len(tr_values) < period:
        return result
    # SMA initiale
    initial = sum(tr_values[:period]) / period
    result[period - 1] = initial
    # Smoothing RMA (Wilder)
    prev = initial
    for i in range(period, len(tr_values)):
        val = (prev * (period - 1) + tr_values[i]) / period
        result[i] = val
        prev = val
    return result


def adx(candles: list[Candle], period: int = 14) -> list[Optional[float]]:
    """Average Directional Index."""
    n = len(candles)
    result: list[Optional[float]] = [None] * n
    if n < period + 1:
        return result

    # +DM, -DM
    plus_dm: list[float] = [0.0]
    minus_dm: list[float] = [0.0]
    for i in range(1, n):
        up = candles[i].high - candles[i - 1].high
        down = candles[i - 1].low - candles[i].low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)

    tr_values = true_range(candles)

    # Smooth avec Wilder (RMA)
    def _smooth(values: list[float], p: int) -> list[float]:
        smoothed = [0.0] * len(values)
        smoothed[p] = sum(values[1 : p + 1])
        for i in range(p + 1, len(values)):
            smoothed[i] = smoothed[i - 1] - smoothed[i - 1] / p + values[i]
        return smoothed

    s_tr = _smooth(tr_values, period)
    s_plus = _smooth(plus_dm, period)
    s_minus = _smooth(minus_dm, period)

    # DI+, DI-
    dx_values: list[Optional[float]] = [None] * n
    for i in range(period, n):
        if s_tr[i] > 0:
            di_plus = 100 * s_plus[i] / s_tr[i]
            di_minus = 100 * s_minus[i] / s_tr[i]
            denom = di_plus + di_minus
            dx_values[i] = 100 * abs(di_plus - di_minus) / denom if denom > 0 else 0.0
        else:
            dx_values[i] = 0.0

    # ADX = SMA du DX
    # Premier ADX = moyenne des period premières DX valides
    first_valid = [dx_values[i] for i in range(period, 2 * period) if dx_values[i] is not None]
    if not first_valid:
        return result
    adx_val = sum(first_valid) / len(first_valid)
    start_idx = 2 * period - 1
    if start_idx < n:
        result[start_idx] = adx_val
    for i in range(start_idx + 1, n):
        if dx_values[i] is not None:
            adx_val = (adx_val * (period - 1) + dx_values[i]) / period
            result[i] = adx_val

    return result


# ── Détection du signal ────────────────────────────────────────────────────────


def detect_breakout_signals(
    candles: list[Candle],
    config: BreakoutConfig,
) -> list[BreakoutSignal]:
    """Scanne toutes les bougies et retourne les signaux de breakout.

    Conditions pour un signal LONG :
      1. close > Donchian High (breakout haussier)
      2. BB Width > expansion * BB Width moy (volatilité en expansion)
      3. ADX > seuil (tendance confirmée)
      4. Volume > multiplier * Volume moyen

    Conditions pour un signal SHORT : miroir inversé.
    """
    n = len(candles)
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    # Calcul des indicateurs
    bb_upper, bb_mid, bb_lower = bollinger_bands(closes, config.bb_period, config.bb_std)
    bbw = bb_width(bb_upper, bb_mid, bb_lower)
    bbw_avg = sma([x if x is not None else 0.0 for x in bbw], config.bb_period)

    dc_high, dc_low = donchian_channel(candles, config.donchian_period)
    adx_values = adx(candles, config.adx_period)
    vol_avg = sma(volumes, config.vol_avg_period)
    atr_values = atr(candles, config.atr_period)

    signals: list[BreakoutSignal] = []

    # Warmup : on a besoin d'au moins max(bb_period, donchian_period, 2*adx_period) bougies
    warmup = max(config.bb_period, config.donchian_period, 2 * config.adx_period) + 5

    for i in range(warmup, n):
        # Vérifier que tous les indicateurs sont disponibles
        if any(v is None for v in (
            dc_high[i], dc_low[i], bbw[i], bbw_avg[i],
            adx_values[i], vol_avg[i], atr_values[i],
        )):
            continue

        c = candles[i]
        current_adx = adx_values[i]  # type: ignore[assignment]
        current_bbw = bbw[i]  # type: ignore[assignment]
        avg_bbw = bbw_avg[i]  # type: ignore[assignment]
        current_vol = c.volume
        avg_vol = vol_avg[i]  # type: ignore[assignment]
        current_atr = atr_values[i]  # type: ignore[assignment]

        # ── Filtres communs ──
        bbw_expanding = current_bbw > config.bb_width_expansion * avg_bbw
        adx_strong = current_adx > config.adx_threshold
        vol_above_avg = current_vol > config.vol_multiplier * avg_vol if avg_vol > 0 else False

        if not (bbw_expanding and adx_strong and vol_above_avg):
            continue

        # ── Breakout LONG ──
        if c.close > dc_high[i]:  # type: ignore[operator]
            sl = c.close - config.sl_atr_mult * current_atr
            signals.append(BreakoutSignal(
                direction=BreakoutDirection.LONG,
                entry_price=c.close,
                sl_price=sl,
                donchian_high=dc_high[i],  # type: ignore[arg-type]
                donchian_low=dc_low[i],  # type: ignore[arg-type]
                bb_width=current_bbw,
                adx=current_adx,
                volume_ratio=current_vol / avg_vol if avg_vol > 0 else 0,
                candle_index=i,
            ))

        # ── Breakout SHORT ──
        elif config.allow_short and c.close < dc_low[i]:  # type: ignore[operator]
            sl = c.close + config.sl_atr_mult * current_atr
            signals.append(BreakoutSignal(
                direction=BreakoutDirection.SHORT,
                entry_price=c.close,
                sl_price=sl,
                donchian_high=dc_high[i],  # type: ignore[arg-type]
                donchian_low=dc_low[i],  # type: ignore[arg-type]
                bb_width=current_bbw,
                adx=current_adx,
                volume_ratio=current_vol / avg_vol if avg_vol > 0 else 0,
                candle_index=i,
            ))

    return signals


# ── Diagnostic des filtres ─────────────────────────────────────────────────────


def diagnose_last_candle(
    candles: list[Candle],
    cfg: BreakoutConfig,
) -> dict:
    """Diagnostique les 4 filtres sur la dernière bougie.

    Retourne un dict avec le statut de chaque filtre, les valeurs observées
    et un résumé filters_passed / filters_total.
    Utile pour comprendre pourquoi un signal n'a pas été généré.
    """
    n = len(candles)
    warmup = max(cfg.bb_period, cfg.donchian_period, 2 * cfg.adx_period) + 5
    if n < warmup:
        return {"insufficient_data": True, "filters_passed": 0, "filters_total": 4}

    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]

    bb_upper, bb_mid, bb_lower = bollinger_bands(closes, cfg.bb_period, cfg.bb_std)
    bbw_vals = bb_width(bb_upper, bb_mid, bb_lower)
    bbw_avg_vals = sma([x if x is not None else 0.0 for x in bbw_vals], cfg.bb_period)
    dc_h, dc_l = donchian_channel(candles, cfg.donchian_period)
    adx_vals = adx(candles, cfg.adx_period)
    vol_avg_vals = sma(volumes, cfg.vol_avg_period)

    i = n - 1
    c = candles[i]

    result: dict = {"reasons": []}

    # 1. Donchian break
    if dc_h[i] is not None:
        is_break = c.close > dc_h[i]  # type: ignore[operator]
        result["donchian_break"] = is_break
        result["donchian_high"] = dc_h[i]
        dist = (c.close - dc_h[i]) / dc_h[i] * 100 if dc_h[i] > 0 else 0  # type: ignore[operator]
        result["donchian_distance_pct"] = dist
        if not is_break:
            result["reasons"].append(f"Donchian: close {c.close:.2f} < high {dc_h[i]:.2f} ({dist:+.1f}%)")
    else:
        result["donchian_break"] = False
        result["reasons"].append("Donchian: données insuffisantes")

    # 2. BB Width expanding
    if bbw_vals[i] is not None and bbw_avg_vals[i] is not None and bbw_avg_vals[i] > 0:  # type: ignore[operator]
        threshold = cfg.bb_width_expansion * bbw_avg_vals[i]  # type: ignore[operator]
        is_expanding = bbw_vals[i] > threshold  # type: ignore[operator]
        result["bb_expanding"] = is_expanding
        result["bb_width"] = bbw_vals[i]
        result["bb_width_ratio"] = bbw_vals[i] / threshold if threshold > 0 else 0  # type: ignore[operator]
        if not is_expanding:
            result["reasons"].append(
                f"BB Width: {bbw_vals[i]:.4f} < {threshold:.4f} (ratio {result['bb_width_ratio']:.2f}x)"
            )
    else:
        result["bb_expanding"] = False
        result["reasons"].append("BB Width: données insuffisantes")

    # 3. ADX strong
    if adx_vals[i] is not None:
        is_strong = adx_vals[i] > cfg.adx_threshold  # type: ignore[operator]
        result["adx_strong"] = is_strong
        result["adx_value"] = adx_vals[i]
        if not is_strong:
            result["reasons"].append(f"ADX: {adx_vals[i]:.1f} < {cfg.adx_threshold:.0f}")
    else:
        result["adx_strong"] = False
        result["reasons"].append("ADX: données insuffisantes")

    # 4. Volume above average
    if vol_avg_vals[i] is not None and vol_avg_vals[i] > 0:  # type: ignore[operator]
        vol_ratio = c.volume / vol_avg_vals[i]  # type: ignore[operator]
        is_high = c.volume > cfg.vol_multiplier * vol_avg_vals[i]  # type: ignore[operator]
        result["volume_high"] = is_high
        result["volume_ratio"] = vol_ratio
        if not is_high:
            result["reasons"].append(f"Volume: {vol_ratio:.2f}x < {cfg.vol_multiplier:.1f}x requis")
    else:
        result["volume_high"] = False
        result["reasons"].append("Volume: données insuffisantes")

    # Résumé
    filters = ["donchian_break", "bb_expanding", "adx_strong", "volume_high"]
    passed = sum(1 for f in filters if result.get(f, False))
    result["filters_passed"] = passed
    result["filters_total"] = len(filters)

    return result
