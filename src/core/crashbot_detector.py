"""
Crash Bot (Dip Buy) — détection de signaux.

Stratégie : acheter les cryptos qui chutent de ≥ X% en Y heures (ex: -20% en 48h).
TP initial + step trailing → verrouille les profits par paliers.

Pas d'I/O — logique pure, testable sans mock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.models import Candle


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class CrashSignal:
    """Signal émis quand une crypto chute de ≥ threshold en lookback bars."""
    symbol: str               # paire (rempli par l'appelant)
    entry_price: float        # prix de clôture de la bougie signal
    sl_price: float           # SL initial (ATR ou % fixe)
    tp_price: float           # TP initial
    drop_pct: float           # amplitude du drop (négatif, ex: -0.22)
    candle_index: int         # index de la bougie de signal
    atr_value: float = 0.0   # ATR à ce moment


@dataclass
class CrashConfig:
    """Paramètres de la stratégie Crash Bot (Dip Buy)."""
    # Signal
    drop_threshold: float = 0.20     # ≥ 20% de drop pour déclencher
    lookback_bars: int = 12          # 12 × 4h = 48h

    # Entry / Exit
    tp_pct: float = 0.08             # TP = +8% au-dessus de l'entrée
    sl_pct: float = 0.02             # SL fixe = -2% (fallback si ATR indispo)

    # SL dynamique ATR
    atr_sl_mult: float = 1.5         # SL = entry - 1.5 × ATR
    atr_period: int = 14             # Période ATR

    # Step trailing
    trail_step_pct: float = 0.005    # TP extend par step = +0.5%
    trail_trigger_buffer: float = 0.0005  # trigger trail à TP - 0.05% de entry

    # Cooldown
    cooldown_bars: int = 6           # Cooldown entre deux trades sur la même paire


# ── Indicateurs techniques ─────────────────────────────────────────────────────


def atr(candles: list[Candle], period: int = 14) -> list[Optional[float]]:
    """Average True Range (Wilder smoothing). Retourne None pour les premières valeurs."""
    result: list[Optional[float]] = [None] * len(candles)
    if len(candles) < period + 1:
        return result

    trs: list[float] = []
    for i in range(1, period + 1):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )
        trs.append(tr)

    atr_val = sum(trs) / period
    result[period] = atr_val

    for i in range(period + 1, len(candles)):
        tr = max(
            candles[i].high - candles[i].low,
            abs(candles[i].high - candles[i - 1].close),
            abs(candles[i].low - candles[i - 1].close),
        )
        atr_val = (atr_val * (period - 1) + tr) / period
        result[i] = atr_val

    return result


def detect_crash_signals(
    candles: list[Candle],
    cfg: CrashConfig,
) -> list[CrashSignal]:
    """
    Détecte les signaux de crash (dip buy) sur une série de bougies H4.

    Un signal est émis quand :
        close[i] / close[i - lookback_bars] - 1  <=  -drop_threshold

    Retourne la liste des signaux détectés (typiquement 0 ou 1 sur la dernière bougie).
    """
    if len(candles) < cfg.lookback_bars + 1:
        return []

    # Pré-calculer ATR
    atr_values = atr(candles, cfg.atr_period)

    signals: list[CrashSignal] = []

    for i in range(cfg.lookback_bars, len(candles)):
        close_now = candles[i].close
        close_ago = candles[i - cfg.lookback_bars].close

        if close_ago <= 0:
            continue

        drop = (close_now - close_ago) / close_ago  # négatif si baisse

        if drop > -cfg.drop_threshold:
            continue

        # SIGNAL ! Entry au close
        entry_price = close_now
        tp_price = entry_price * (1 + cfg.tp_pct)

        # SL : ATR dynamique ou % fixe
        atr_val = atr_values[i] if i < len(atr_values) and atr_values[i] is not None else None
        if cfg.atr_sl_mult > 0 and atr_val and atr_val > 0:
            sl_price = entry_price - cfg.atr_sl_mult * atr_val
        else:
            sl_price = entry_price * (1 - cfg.sl_pct)

        signals.append(CrashSignal(
            symbol="",  # sera rempli par l'appelant
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            drop_pct=drop,
            candle_index=i,
            atr_value=atr_val or 0.0,
        ))

    return signals


def compute_step_trailing(
    current_price: float,
    entry_price: float,
    current_sl: float,
    current_tp: float,
    trail_steps: int,
    cfg: CrashConfig,
) -> tuple[float, float, int]:
    """
    Calcule le step trailing : quand le prix approche du TP, verrouille SL au TP
    et étend le TP par step_pct.

    Returns:
        (new_sl, new_tp, new_trail_steps)
    """
    new_sl = current_sl
    new_tp = current_tp
    new_steps = trail_steps

    safety = 0
    while safety < 50:
        trigger = new_tp - entry_price * cfg.trail_trigger_buffer
        if current_price >= trigger:
            # Verrouiller SL au TP actuel, étendre TP
            new_sl = new_tp
            new_tp += entry_price * cfg.trail_step_pct
            new_steps += 1
            safety += 1
        else:
            break

    return new_sl, new_tp, new_steps
