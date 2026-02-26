"""
Money management et calcul de taille de position.

Calcule la taille de position en fonction du risque autorisé,
du solde disponible, et de la distance au stop loss.

Gère aussi le passage en "zero risk" (déplacement du SL pour
verrouiller du profit une fois que le trade est suffisamment en gain).

Ce module est de la logique pure — aucun appel réseau.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.models import Balance, OrderSide, Position, TickerData

logger = logging.getLogger(__name__)


def calculate_position_size(
    account_balance: float,
    risk_percent: float,
    entry_price: float,
    sl_price: float,
    max_position_percent: float = 1.0,
) -> float:
    """
    Calcule la taille de position en unités de base (ex: BTC).

    Formule :
        risk_amount = account_balance * risk_percent
        sl_distance = abs(entry_price - sl_price)
        position_size = risk_amount / sl_distance

    La position est ensuite plafonnée à max_position_percent du capital
    (ex: 0.20 = max 20% du capital alloué par trade).

    Args:
        account_balance: Solde disponible en devise fiat (USD ou EUR converti).
        risk_percent: Pourcentage de risque (ex: 0.05 pour 5%).
        entry_price: Prix d'entrée prévu.
        sl_price: Prix du stop loss.
        max_position_percent: Part max du capital allouable par position (ex: 0.20).

    Returns:
        Taille de position en unités de base. 0.0 si invalide.
    """
    if account_balance <= 0:
        logger.error("Solde du compte invalide: %.2f", account_balance)
        return 0.0

    if entry_price <= 0 or sl_price <= 0:
        logger.error(
            "Prix invalides: entry=%.2f, sl=%.2f", entry_price, sl_price
        )
        return 0.0

    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        logger.error("Distance SL nulle (entry == sl)")
        return 0.0

    risk_amount = account_balance * risk_percent
    position_size = risk_amount / sl_distance

    # Plafonner la position à max_position_percent du capital
    max_budget = account_balance * max_position_percent
    position_cost = position_size * entry_price
    if position_cost > max_budget:
        capped_size = max_budget / entry_price
        actual_risk = capped_size * sl_distance
        actual_risk_pct = actual_risk / account_balance * 100
        logger.info(
            "Sizing cappé: position idéale=%.8f (coût %.2f) > budget max %.2f (%.0f%% de %.2f) → "
            "taille réduite=%.8f (risque réel: %.2f soit %.1f%%)",
            position_size,
            position_cost,
            max_budget,
            max_position_percent * 100,
            account_balance,
            capped_size,
            actual_risk,
            actual_risk_pct,
        )
        position_size = capped_size

    logger.debug(
        "Sizing: balance=%.2f, risque=%.1f%% (%.2f), "
        "distance SL=%.2f, taille=%.8f, budget max=%.0f%%",
        account_balance,
        risk_percent * 100,
        risk_amount,
        sl_distance,
        position_size,
        max_position_percent * 100,
    )

    return position_size


def get_fiat_balance(balances: list[Balance]) -> tuple[float, str]:
    """
    Extrait le solde fiat disponible depuis la liste des balances.

    Cherche dans l'ordre : USD, EUR, GBP.
    Si la devise n'est pas USD, convertit avec un taux approximatif.

    Args:
        balances: Liste des Balance retournées par GET /balances.

    Returns:
        Tuple (montant en USD équivalent, devise d'origine).
    """
    # Taux de conversion approximatifs vers USD
    fiat_to_usd = {
        "USD": 1.0,
        "EUR": 1.05,
        "GBP": 1.27,
    }

    for currency in ["USD", "EUR", "GBP"]:
        for b in balances:
            if b.currency == currency and b.available > 0:
                rate = fiat_to_usd[currency]
                usd_equivalent = b.available * rate
                if currency != "USD":
                    logger.debug(
                        "Solde fiat: %.2f %s ≈ %.2f USD (taux ~%.2f)",
                        b.available,
                        currency,
                        usd_equivalent,
                        rate,
                    )
                else:
                    logger.debug("Solde fiat: %.2f USD", b.available)
                return usd_equivalent, currency

    logger.warning("Aucun solde fiat (USD/EUR/GBP) trouvé dans les balances")
    return 0.0, ""


def get_total_equity(
    balances: list[Balance],
    tickers: list[TickerData],
) -> float:
    """
    Calcule l'equity totale du compte : USD + valeur de toutes les cryptos.

    Equity = solde USD (total, pas juste available)
             + Σ (crypto_balance.total × prix_actuel)

    Args:
        balances: Liste des Balance retournées par GET /balances.
        tickers: Liste des TickerData pour valoriser les cryptos.

    Returns:
        Equity totale en USD.
    """
    # Devises fiat à traiter directement en USD
    fiat_currencies = {"USD", "EUR", "GBP"}
    fiat_to_usd = {"USD": 1.0, "EUR": 1.05, "GBP": 1.27}

    equity = 0.0

    # Construire un dict symbol→last_price pour lookup rapide
    # TickerData.symbol peut être "BTC-USD" ou "BTC/USD"
    price_map: dict[str, float] = {}
    for t in tickers:
        normalized = t.symbol.replace("/", "-")
        price_map[normalized] = t.last_price

    for b in balances:
        if b.total <= 0:
            continue

        if b.currency in fiat_currencies:
            rate = fiat_to_usd.get(b.currency, 1.0)
            equity += b.total * rate
        else:
            # Crypto — chercher le prix via {CURRENCY}-USD
            pair = f"{b.currency}-USD"
            price = price_map.get(pair, 0.0)
            if price > 0:
                equity += b.total * price
            else:
                logger.debug(
                    "Equity: pas de prix pour %s (balance=%.6f), ignoré",
                    pair, b.total,
                )

    logger.debug("Equity totale calculée: $%.2f", equity)
    return equity


def should_apply_zero_risk(
    position: Position,
    current_price: float,
    trigger_percent: float,
) -> bool:
    """
    Vérifie si les conditions de passage en zero-risk sont remplies.

    Le prix doit avoir parcouru au moins `trigger_percent` en faveur du trade.

    Args:
        position: Position ouverte.
        current_price: Prix actuel du marché.
        trigger_percent: Pourcentage de mouvement en faveur requis (ex: 0.02).

    Returns:
        True si le zero-risk doit être appliqué.
    """
    if position.is_zero_risk_applied:
        return False

    price_move_percent = _calculate_price_move_percent(
        position.entry_price, current_price, position.side
    )

    return price_move_percent >= trigger_percent


def calculate_zero_risk_sl(
    position: Position,
    lock_percent: float,
) -> float:
    """
    Calcule le nouveau prix de SL pour verrouiller du profit (zero-risk).

    Args:
        position: Position ouverte.
        lock_percent: Pourcentage de profit à verrouiller (ex: 0.005 = 0.5%).

    Returns:
        Nouveau prix de SL qui garantit un gain minimal.
    """
    lock_amount = position.entry_price * lock_percent

    if position.side == OrderSide.BUY:
        # Long : SL remonté au-dessus du prix d'entrée
        new_sl = position.entry_price + lock_amount
    else:
        # Short : SL descendu en-dessous du prix d'entrée
        new_sl = position.entry_price - lock_amount

    logger.info(
        "[%s] Zero-risk SL: ancien=%.2f → nouveau=%.2f (lock %.1f%%)",
        position.symbol,
        position.sl_price,
        new_sl,
        lock_percent * 100,
    )

    return new_sl


def update_trailing_stop(
    position: Position,
    current_price: float,
    trailing_percent: float,
) -> Optional[float]:
    """
    Met à jour le trailing stop après activation du zero-risk.

    Le SL suit le prix à une distance de `trailing_percent` par rapport
    au plus haut (BUY) ou plus bas (SELL) atteint.
    Le SL ne peut que monter (BUY) ou descendre (SELL), jamais reculer.

    Args:
        position: Position avec zero-risk déjà activé.
        current_price: Prix actuel du marché.
        trailing_percent: Distance du trailing en % (ex: 0.02 = 2%).

    Returns:
        Nouveau prix de SL si mis à jour, None sinon.
    """
    if not position.is_zero_risk_applied:
        return None

    if position.zero_risk_sl is None:
        return None

    old_sl = position.zero_risk_sl

    if position.side == OrderSide.BUY:
        # Mettre à jour le peak (plus haut atteint)
        if position.peak_price is None or current_price > position.peak_price:
            position.peak_price = current_price

        # Trailing SL = peak - trailing%
        trailing_sl = position.peak_price * (1 - trailing_percent)

        # Le SL ne peut que monter, jamais redescendre
        if trailing_sl > old_sl:
            position.zero_risk_sl = trailing_sl
            logger.info(
                "[%s] 📈 Trailing stop mis à jour : SL %.2f → %.2f (peak: %.2f)",
                position.symbol,
                old_sl,
                trailing_sl,
                position.peak_price,
            )
            return trailing_sl

    else:  # SELL
        # Mettre à jour le peak (plus bas atteint)
        if position.peak_price is None or current_price < position.peak_price:
            position.peak_price = current_price

        # Trailing SL = peak + trailing%
        trailing_sl = position.peak_price * (1 + trailing_percent)

        # Le SL ne peut que descendre, jamais remonter
        if trailing_sl < old_sl:
            position.zero_risk_sl = trailing_sl
            logger.info(
                "[%s] 📉 Trailing stop mis à jour : SL %.2f → %.2f (peak: %.2f)",
                position.symbol,
                old_sl,
                trailing_sl,
                position.peak_price,
            )
            return trailing_sl

    return None


def check_total_risk_exposure(
    positions: list[Position],
    account_balance: float,
    max_total_risk_pct: float,
) -> float:
    """
    Calcule le risque total engagé sur toutes les positions ouvertes.

    Risque par position = size * |entry_price - sl_price|.
    Retourne le risque total en pourcentage du capital.

    Args:
        positions: Toutes les positions ouvertes.
        account_balance: Solde fiat en USD.
        max_total_risk_pct: Cap de risque global (ex: 0.06 = 6%).

    Returns:
        Risque total en pourcentage (ex: 0.04 = 4%).
    """
    if account_balance <= 0:
        return 0.0

    total_risk_usd = 0.0
    for pos in positions:
        sl = pos.zero_risk_sl if pos.is_zero_risk_applied and pos.zero_risk_sl else pos.sl_price
        risk_per_unit = abs(pos.entry_price - sl)
        total_risk_usd += pos.size * risk_per_unit

    total_risk_pct = total_risk_usd / account_balance

    logger.debug(
        "Risque total: %.2f USD (%.2f%%) / max %.2f%% | %d positions",
        total_risk_usd,
        total_risk_pct * 100,
        max_total_risk_pct * 100,
        len(positions),
    )

    return total_risk_pct


def _calculate_price_move_percent(
    entry_price: float,
    current_price: float,
    side: OrderSide,
) -> float:
    """Calcule le pourcentage de mouvement du prix en faveur du trade."""
    if entry_price == 0:
        return 0.0

    if side == OrderSide.BUY:
        return (current_price - entry_price) / entry_price
    else:
        return (entry_price - current_price) / entry_price
