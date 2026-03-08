"""
Indicateurs techniques réutilisables (EMA, SMA, ATR, RSI, rolling min/max).

Module sans I/O — utilisable dans core/ et backtest/.
"""

from __future__ import annotations

from src.core.models import Candle


# ── EMA ────────────────────────────────────────────────────────────────────────

def ema(values: list[float], period: int) -> list[float]:
    """EMA — retourne liste de même taille (NaN rempli par 0)."""
    if not values or period <= 0:
        return [0.0] * len(values)
    k = 2.0 / (period + 1)
    result = [0.0] * len(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


# ── SMA ────────────────────────────────────────────────────────────────────────

def sma(values: list[float], period: int) -> list[float]:
    """SMA — les premières `period-1` valeurs sont la moyenne cumulative."""
    if not values or period <= 0:
        return [0.0] * len(values)
    result = [0.0] * len(values)
    cum = 0.0
    for i in range(len(values)):
        cum += values[i]
        if i < period:
            result[i] = cum / (i + 1)
        else:
            cum -= values[i - period]
            result[i] = cum / period
    return result


# ── ATR ────────────────────────────────────────────────────────────────────────

def atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ATR via EMA du True Range."""
    if len(candles) < 2:
        return [0.0] * len(candles)
    tr = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr.append(max(
            c.high - c.low,
            abs(c.high - prev_close),
            abs(c.low - prev_close),
        ))
    return ema(tr, period)


# ── RSI ────────────────────────────────────────────────────────────────────────

def rsi_series(candles: list[Candle], period: int = 14) -> list[float]:
    """RSI via EMA des gains/losses."""
    n = len(candles)
    if n < 2:
        return [50.0] * n
    gains = [0.0]
    losses = [0.0]
    for i in range(1, n):
        delta = candles[i].close - candles[i - 1].close
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    result = []
    for i in range(n):
        if avg_loss[i] == 0:
            result.append(100.0 if avg_gain[i] > 0 else 50.0)
        else:
            rs = avg_gain[i] / avg_loss[i]
            result.append(100.0 - 100.0 / (1.0 + rs))
    return result


# ── Rolling Min / Max ──────────────────────────────────────────────────────────

def rolling_min(values: list[float], period: int) -> list[float]:
    """Min glissant."""
    result = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        result.append(min(values[start:i + 1]))
    return result


def rolling_max(values: list[float], period: int) -> list[float]:
    """Max glissant."""
    result = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        result.append(max(values[start:i + 1]))
    return result
