"""
Utilitaires de formatage partagés entre tous les bots.

Module sans I/O — importable sans effets de bord.
"""

from __future__ import annotations


def fmt_price(price: float) -> str:
    """Formate un prix de façon lisible quel que soit son ordre de grandeur."""
    if price >= 1000:
        return f"{price:,.4f}"
    if price >= 1:
        return f"{price:.4f}"
    if price >= 0.0001:
        return f"{price:.6f}"
    decimals = 6
    temp = price
    while temp < 0.01 and decimals < 10:
        temp *= 10
        decimals += 1
    return f"{price:.{decimals}f}"
