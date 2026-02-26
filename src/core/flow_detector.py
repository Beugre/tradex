"""
Détection de mouvements anormaux (liquidation cascades) sur bougies 1m.

Logique pure (sans I/O) — identifie les dumps soudains à fort volume
typiques des cascades de liquidation long sur le marché des dérivés crypto.

Le bot antiliq trade le rebond contrarian après ces dumps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.core.models import Candle


@dataclass
class FlowSignal:
    """Signal de mouvement anormal détecté sur bougies 1m."""

    symbol: str
    timestamp: int              # Timestamp de détection (ms)
    direction: str              # "DUMP" (seul type tradé sur Spot)
    move_pct: float             # Amplitude du mouvement (ex: -0.035 = -3.5%)
    move_start_price: float     # Prix au début de la fenêtre
    move_end_price: float       # Prix actuel (fin de fenêtre)
    volume_ratio: float         # Volume récent vs moyenne (intensité)
    entry_price: float          # Prix d'entrée suggéré (= move_end_price)
    tp_price: float             # Take Profit (retrace partiel)
    sl_price: float             # Stop Loss (extension du mouvement)


def detect_abnormal_flow(
    candles: list[Candle],
    symbol: str = "",
    move_window: int = 5,
    move_threshold_pct: float = 0.03,
    volume_multiplier: float = 1.5,
    vol_avg_window: int = 60,
    tp_retrace_pct: float = 0.5,
    sl_extension_pct: float = 0.5,
) -> Optional[FlowSignal]:
    """
    Analyse les dernières bougies 1m pour détecter un dump anormal.

    Mécanisme : compare le mouvement de prix des N dernières minutes
    au seuil de déclenchement, et vérifie que le volume est anormalement
    élevé (signe de liquidations forcées).

    On ne détecte que les DUMPS (prix en chute) car sur Spot on ne peut
    que BUY (long) pour trader le rebond.

    Args:
        candles: Liste de bougies 1m (au moins vol_avg_window + move_window).
        symbol: Nom du symbole (pour le signal).
        move_window: Fenêtre de détection en bougies/minutes (ex: 5).
        move_threshold_pct: Seuil minimum de mouvement (0.03 = 3%).
        volume_multiplier: Le volume récent doit être ≥ Nx la moyenne.
                           Mettre 0 pour désactiver le filtre volume.
        vol_avg_window: Fenêtre pour le calcul du volume moyen (en bougies).
        tp_retrace_pct: Fraction du dump pour le TP (0.5 = 50% retrace).
        sl_extension_pct: Fraction du dump pour le SL (0.5 = 50% extension).

    Returns:
        FlowSignal si un dump anormal est détecté, None sinon.
    """
    min_candles = vol_avg_window + move_window + 1
    if len(candles) < min_candles:
        return None

    # ── Prix : variation sur la fenêtre ────────────────────────────────────
    current_price = candles[-1].close
    lookback_price = candles[-(move_window + 1)].close

    if lookback_price == 0 or current_price == 0:
        return None

    move_pct = (current_price - lookback_price) / lookback_price

    # ── Volume : comparaison récent vs baseline ────────────────────────────
    recent_vol = sum(c.volume for c in candles[-move_window:])

    volume_ratio = 0.0
    if volume_multiplier > 0:
        baseline_candles = candles[-(vol_avg_window + move_window):-move_window]
        if not baseline_candles:
            return None
        avg_vol_per_candle = sum(c.volume for c in baseline_candles) / len(baseline_candles)
        expected_vol = avg_vol_per_candle * move_window
        volume_ratio = recent_vol / expected_vol if expected_vol > 0 else 0.0

        # Filtre volume non satisfait
        if volume_ratio < volume_multiplier:
            return None

    # ── Dump détecté ? ─────────────────────────────────────────────────────
    if move_pct <= -move_threshold_pct:
        dump_distance = abs(current_price - lookback_price)
        entry = current_price
        tp = entry + dump_distance * tp_retrace_pct      # Retrace partiel vers le haut
        sl = entry - dump_distance * sl_extension_pct     # Extension vers le bas

        return FlowSignal(
            symbol=symbol,
            timestamp=candles[-1].timestamp,
            direction="DUMP",
            move_pct=move_pct,
            move_start_price=lookback_price,
            move_end_price=current_price,
            volume_ratio=volume_ratio,
            entry_price=entry,
            tp_price=tp,
            sl_price=sl,
        )

    return None
