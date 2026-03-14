"""
Listing Detector — logique pure pour le bot Listing Event.

Détecte les nouveaux listings USDC sur Binance, vérifie le filtre momentum,
calcule les niveaux OCO (SL/TP), et gère le re-arm OCO.

Aucun appel réseau ici : les données sont passées en argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ListingSignal:
    """Signal de nouveau listing détecté."""
    symbol: str
    listing_price: float       # Open de la 1ère bougie 1m
    momentum_pct: float        # Pump réel observé dans la fenêtre
    momentum_ok: bool          # True si >= seuil


@dataclass
class OCOLevels:
    """Niveaux OCO pour une position listing."""
    sl_price: float
    tp_price: float
    sl_pct: float              # Distance SL en % (négatif)
    tp_pct: float              # Distance TP en % (positif)


@dataclass
class RearmOCOLevels:
    """Nouveaux niveaux OCO après re-arm."""
    sl_price: float
    tp_price: float
    sl_pct_vs_entry: float     # SL vs entry en %
    tp_pct_vs_entry: float     # TP vs entry en %


def detect_new_symbols(
    current_symbols: set[str],
    known_symbols: set[str],
    quote_asset: str = "USDC",
) -> list[str]:
    """Retourne les symboles USDC nouvellement apparus.

    Args:
        current_symbols: Set des symboles USDC en TRADING actuellement.
        known_symbols: Set des symboles déjà connus (persistés).

    Returns:
        Liste des nouveaux symboles (triés).
    """
    new = current_symbols - known_symbols
    return sorted(new)


def check_momentum(
    candles_1m: list[dict],
    momentum_threshold: float = 0.30,
    window_minutes: int = 1,
) -> ListingSignal | None:
    """Vérifie si le listing a un momentum suffisant.

    Le filtre regarde si le HIGH d'une des N premières bougies 1m
    atteint >= open * (1 + momentum_threshold).

    Args:
        candles_1m: Liste de dicts avec au minimum {"open", "high", "close", "symbol"}.
        momentum_threshold: Seuil de pump minimum (0.30 = +30%).
        window_minutes: Nombre de bougies 1m à analyser.

    Returns:
        ListingSignal si le momentum est suffisant, None sinon.
    """
    if not candles_1m:
        return None

    first = candles_1m[0]
    listing_price = float(first["open"])
    if listing_price <= 0:
        return None

    target = listing_price * (1 + momentum_threshold)
    best_pump = 0.0

    for i in range(min(window_minutes, len(candles_1m))):
        high = float(candles_1m[i]["high"])
        pump = (high - listing_price) / listing_price
        best_pump = max(best_pump, pump)
        if high >= target:
            return ListingSignal(
                symbol=str(first.get("symbol", "")),
                listing_price=listing_price,
                momentum_pct=pump,
                momentum_ok=True,
            )

    return ListingSignal(
        symbol=str(first.get("symbol", "")),
        listing_price=listing_price,
        momentum_pct=best_pump,
        momentum_ok=False,
    )


def compute_oco_levels(
    entry_price: float,
    sl_pct: float = 0.08,
    tp_pct: float = 0.30,
) -> OCOLevels:
    """Calcule les niveaux OCO initiaux.

    Args:
        entry_price: Prix d'entrée effectif (après market fill).
        sl_pct: Stop-loss en % sous l'entrée (0.08 = -8%).
        tp_pct: Take-profit en % au-dessus de l'entrée (0.30 = +30%).

    Returns:
        OCOLevels avec SL et TP.
    """
    return OCOLevels(
        sl_price=entry_price * (1 - sl_pct),
        tp_price=entry_price * (1 + tp_pct),
        sl_pct=-sl_pct,
        tp_pct=tp_pct,
    )


def should_rearm_oco(
    current_price: float,
    tp_price: float,
    tp_near_ratio: float = 0.98,
) -> bool:
    """Vérifie si le prix est assez proche du TP pour re-arm l'OCO.

    Args:
        current_price: Prix courant (bid ou last).
        tp_price: Niveau TP actuel.
        tp_near_ratio: Ratio (0.98 = 98% du TP).

    Returns:
        True si current_price >= tp_price * tp_near_ratio.
    """
    return current_price >= tp_price * tp_near_ratio


def compute_rearm_oco_levels(
    entry_price: float,
    tp1_price: float,
    sl2_tp1_mult: float = 0.769,
    tp2_tp1_mult: float = 1.538,
) -> RearmOCOLevels:
    """Calcule les nouveaux niveaux OCO après re-arm.

    SL2 = TP1 × sl2_tp1_mult  (verrouille un gain)
    TP2 = TP1 × tp2_tp1_mult  (= entry × +100% avec les valeurs par défaut)

    Args:
        entry_price: Prix d'entrée original.
        tp1_price: Prix TP initial (entry × 1.30).
        sl2_tp1_mult: Multiplicateur du TP1 pour le nouveau SL.
        tp2_tp1_mult: Multiplicateur du TP1 pour le nouveau TP.

    Returns:
        RearmOCOLevels avec les nouveaux SL2 et TP2.
    """
    sl2 = tp1_price * sl2_tp1_mult
    tp2 = tp1_price * tp2_tp1_mult

    return RearmOCOLevels(
        sl_price=sl2,
        tp_price=tp2,
        sl_pct_vs_entry=(sl2 - entry_price) / entry_price if entry_price > 0 else 0,
        tp_pct_vs_entry=(tp2 - entry_price) / entry_price if entry_price > 0 else 0,
    )


def should_force_close(
    entry_ts_ms: int,
    current_ts_ms: int,
    horizon_days: int = 7,
) -> bool:
    """Vérifie si la position a dépassé l'horizon temporel.

    Args:
        entry_ts_ms: Timestamp d'entrée en ms.
        current_ts_ms: Timestamp courant en ms.
        horizon_days: Nombre de jours max.

    Returns:
        True si la position est ouverte depuis plus de horizon_days.
    """
    horizon_ms = horizon_days * 24 * 3600 * 1000
    return (current_ts_ms - entry_ts_ms) >= horizon_ms


def compute_position_size(
    equity: float,
    cash: float,
    max_slots: int,
    max_alloc_usd: float = 5000.0,
) -> float:
    """Calcule la taille de position en USD pour un listing trade.

    Allocation = min(cash disponible, equity / max_slots, max_alloc_usd).

    Args:
        equity: Equity totale du bot listing.
        cash: Cash disponible (non engagé).
        max_slots: Nombre max de positions simultanées.
        max_alloc_usd: Plafond en USD par position.

    Returns:
        Montant en USD à allouer pour ce trade.
    """
    alloc = min(cash, equity / max_slots)
    if max_alloc_usd > 0:
        alloc = min(alloc, max_alloc_usd)
    return max(0.0, alloc)
