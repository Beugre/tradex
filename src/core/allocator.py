"""
Dynamic Capital Allocator — Répartition Crash / Trail / Listing basée sur le PF.

Règle d'allocation (recalculée 1×/jour) :
┌────────────────────────────────────────────────────────────────────────┐
│ PF 90j du Trail Range   │ Trail Range │  CrashBot  │  Listing Bot    │
├────────────────────────────────────────────────────────────────────────┤
│ PF < 0.9 OU < 20 trades │     5%      │     65%    │      30%        │ ← DEFENSIVE
│ 0.9 ≤ PF ≤ 1.1 ET ≥ 20 │    10%      │     60%    │      30%        │ ← NEUTRAL
│ PF > 1.1 ET ≥ 20 trades │    20%      │     50%    │      30%        │ ← AGGRESSIVE
└────────────────────────────────────────────────────────────────────────┘

Le Listing Bot reçoit toujours 30% du capital (fixe).
Les 70% restants sont répartis entre Trail Range et CrashBot selon le PF.

Module core → aucun I/O. Les données (PF, n_trades) sont passées en paramètre.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AllocationRegime(Enum):
    """Régime d'allocation courant."""
    DEFENSIVE = "defensive"     # PF < 0.9 — Trail à 5%
    NEUTRAL = "neutral"         # PF 0.9–1.1 — Trail à 10%
    AGGRESSIVE = "aggressive"   # PF > 1.1 — Trail à 20%


@dataclass(frozen=True)
class AllocationResult:
    """Résultat du calcul d'allocation."""
    regime: AllocationRegime
    crash_pct: float        # ex: 0.65
    trail_pct: float        # ex: 0.05
    listing_pct: float      # ex: 0.30
    crash_balance: float    # en USD
    trail_balance: float    # en USD
    listing_balance: float  # en USD
    trail_pf: float         # PF utilisé pour le calcul
    trail_trades: int       # nombre de trades utilisé
    reason: str             # explication humaine


# ── Seuils configurables ────────────────────────────────────────────────────

_PF_LOW = 0.9
_PF_HIGH = 1.1
_MIN_TRADES = 20

# Listing Bot : part fixe (constante quel que soit le régime)
_LISTING_PCT_DEFAULT = 0.30

# Trail Range : part variable selon le PF (le reste va au CrashBot)
_TRAIL_PCT_DEFENSIVE = 0.05
_TRAIL_PCT_NEUTRAL = 0.10
_TRAIL_PCT_AGGRESSIVE = 0.20


def compute_allocation(
    total_balance: float,
    trail_pf: float,
    trail_trade_count: int,
    listing_pct: float = _LISTING_PCT_DEFAULT,
) -> AllocationResult:
    """Calcule la répartition Crash/Trail/Listing basée sur le PF du Trail Range.

    Le Listing Bot reçoit ``listing_pct`` du capital total (fixe).
    Les ``1 - listing_pct`` restants sont répartis entre Trail et Crash selon le PF.

    Args:
        total_balance: Capital total Binance (ex: 2350.0)
        trail_pf: Profit Factor du bot Trail Range sur 90 jours
        trail_trade_count: Nombre de trades clôturés du Trail Range sur 90 jours
        listing_pct: Part fixe allouée au Listing Bot (défaut 0.30)

    Returns:
        AllocationResult avec les montants alloués à chaque bot
    """
    if trail_trade_count < _MIN_TRADES or trail_pf < _PF_LOW:
        regime = AllocationRegime.DEFENSIVE
        trail_pct = _TRAIL_PCT_DEFENSIVE
        if trail_trade_count < _MIN_TRADES:
            reason = (
                f"PF={trail_pf:.2f} sur {trail_trade_count} trades "
                f"(< {_MIN_TRADES} min) → Défensif"
            )
        else:
            reason = f"PF={trail_pf:.2f} < {_PF_LOW} sur {trail_trade_count} trades → Défensif"

    elif trail_pf <= _PF_HIGH:
        regime = AllocationRegime.NEUTRAL
        trail_pct = _TRAIL_PCT_NEUTRAL
        reason = (
            f"PF={trail_pf:.2f} ({_PF_LOW}–{_PF_HIGH}) "
            f"sur {trail_trade_count} trades → Neutre"
        )

    else:
        regime = AllocationRegime.AGGRESSIVE
        trail_pct = _TRAIL_PCT_AGGRESSIVE
        reason = (
            f"PF={trail_pf:.2f} > {_PF_HIGH} "
            f"sur {trail_trade_count} trades → Agressif"
        )

    crash_pct = round(1.0 - listing_pct - trail_pct, 10)
    crash_balance = round(total_balance * crash_pct, 2)
    trail_balance = round(total_balance * trail_pct, 2)
    listing_balance = round(total_balance * listing_pct, 2)

    return AllocationResult(
        regime=regime,
        crash_pct=crash_pct,
        trail_pct=trail_pct,
        listing_pct=listing_pct,
        crash_balance=crash_balance,
        trail_balance=trail_balance,
        listing_balance=listing_balance,
        trail_pf=trail_pf,
        trail_trades=trail_trade_count,
        reason=reason,
    )


def compute_profit_factor(pnl_list: list[float]) -> float:
    """Calcule le Profit Factor à partir d'une liste de PnL.

    PF = sum(gains) / abs(sum(pertes))
    Si aucune perte → inf, si aucun gain → 0.0
    """
    gains = sum(p for p in pnl_list if p > 0)
    losses = abs(sum(p for p in pnl_list if p < 0))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses
