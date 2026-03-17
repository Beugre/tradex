"""
DCA Engine — logique pure pour le bot DCA RSI-based sur Revolut X.

Stratégie :
  1. Achat quotidien de BTC (80%) et ETH (20%) basé sur le RSI daily du BTC.
  2. Montant variable selon le bracket RSI :
     - RSI > 70      → $0 (pas d'achat)
     - 55 < RSI ≤ 70 → base_amount × 1 (ex: $12)
     - 45 ≤ RSI ≤ 55 → base_amount × 2 (ex: $24)
     - RSI < 45      → base_amount × 3 (ex: $36)
  3. Crash reserve : achats bonus de BTC quand le prix chute en dessous de seuils :
     - -15% du 90-day high → $150 BTC
     - -25% du 90-day high → $250 BTC
     - -35% du 90-day high → $350 BTC
     Chaque palier ne se déclenche qu'une fois par crash (reset quand prix revient au-dessus).

Module sans I/O — utilisable dans core/ et backtest/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ──────────────────────────────────────────────────────────────────────


class RSIBracket(Enum):
    """Bracket RSI pour déterminer le montant d'achat quotidien."""
    OVERBOUGHT = "OVERBOUGHT"    # RSI > 70 → $0
    WARM = "WARM"                # 55 < RSI ≤ 70 → ×1
    NEUTRAL = "NEUTRAL"          # 45 ≤ RSI ≤ 55 → ×2
    OVERSOLD = "OVERSOLD"        # RSI < 45 → ×3
    DEEP_VALUE = "DEEP_VALUE"    # MVRV < threshold → ×5


class CrashLevel(Enum):
    """Niveaux de crash pour la réserve."""
    LEVEL_15 = "LEVEL_15"   # -15%
    LEVEL_25 = "LEVEL_25"   # -25%
    LEVEL_35 = "LEVEL_35"   # -35%


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class DCAConfig:
    """Configuration du bot DCA."""
    # Budget total
    total_capital: float = 5300.0           # Capital total
    active_budget: float = 4200.0           # Budget DCA actif (quotidien)
    crash_reserve: float = 1100.0           # Réserve pour crashes

    # Montants quotidiens
    base_daily_amount: float = 12.0         # Montant de base ($12 par jour)
    # Multiplicateurs par bracket RSI
    rsi_multipliers: dict[str, float] = field(default_factory=lambda: {
        "OVERBOUGHT": 0.0,    # RSI > 70 → $0
        "WARM": 1.0,          # 55 < RSI ≤ 70 → ×1
        "NEUTRAL": 2.0,       # 45 ≤ RSI ≤ 55 → ×2
        "OVERSOLD": 3.0,      # RSI < 45 → ×3
        "DEEP_VALUE": 5.0,    # MVRV < threshold → ×5
    })

    # Allocation par paire (doit sommer à 1.0)
    btc_alloc: float = 0.80                 # 80% BTC
    eth_alloc: float = 0.20                 # 20% ETH

    # RSI thresholds
    rsi_overbought: float = 70.0
    rsi_warm: float = 55.0
    rsi_neutral_low: float = 45.0

    # Crash reserve levels : (drop_pct, amount_usd)
    crash_levels: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.15, 150.0),    # -15% → $150
        (0.25, 250.0),    # -25% → $250
        (0.35, 350.0),    # -35% → $350
    ])

    # Rolling high period (en jours) pour le crash detector
    crash_lookback_days: int = 90

    # Exécution
    execution_hour_utc: int = 10            # Heure d'exécution quotidienne (UTC)
    maker_wait_seconds: int = 60            # Attente pour fill maker

    # MVRV deep-value
    mvrv_enabled: bool = True              # Activer le multiplicateur MVRV
    mvrv_threshold: float = 1.0            # MVRV < 1.0 → DEEP_VALUE bracket
    mvrv_multiplier: float = 5.0           # ×5 quand MVRV < threshold

    # Crash reserve → 100% BTC
    crash_btc_only: bool = True             # True = crash buys are 100% BTC

    # Symboles Revolut X
    btc_symbol: str = "BTC-USD"
    eth_symbol: str = "ETH-USD"


# ── State ──────────────────────────────────────────────────────────────────────


@dataclass
class DCAState:
    """État persisté du bot DCA."""
    # Compteurs de dépenses
    total_spent_dca: float = 0.0           # Total dépensé en DCA quotidien
    total_spent_crash: float = 0.0         # Total dépensé en crash reserve
    total_btc_bought: float = 0.0          # Total BTC accumulés
    total_eth_bought: float = 0.0          # Total ETH accumulés

    # Dernier achat
    last_buy_date: str = ""                # YYYY-MM-DD du dernier achat
    last_buy_rsi: float = 0.0
    last_buy_bracket: str = ""

    # Crash tracking
    crash_levels_triggered: list[str] = field(default_factory=list)  # niveaux déjà déclenchés
    rolling_high_price: float = 0.0        # Plus haut sur la fenêtre

    # Stats
    buy_count: int = 0
    crash_buy_count: int = 0
    total_days_active: int = 0
    start_date: str = ""

    def to_dict(self) -> dict:
        """Sérialise l'état en dict JSON-compatible."""
        return {
            "total_spent_dca": self.total_spent_dca,
            "total_spent_crash": self.total_spent_crash,
            "total_btc_bought": self.total_btc_bought,
            "total_eth_bought": self.total_eth_bought,
            "last_buy_date": self.last_buy_date,
            "last_buy_rsi": self.last_buy_rsi,
            "last_buy_bracket": self.last_buy_bracket,
            "crash_levels_triggered": self.crash_levels_triggered,
            "rolling_high_price": self.rolling_high_price,
            "buy_count": self.buy_count,
            "crash_buy_count": self.crash_buy_count,
            "total_days_active": self.total_days_active,
            "start_date": self.start_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DCAState":
        """Désérialise un dict en DCAState."""
        s = cls()
        s.total_spent_dca = d.get("total_spent_dca", 0.0)
        s.total_spent_crash = d.get("total_spent_crash", 0.0)
        s.total_btc_bought = d.get("total_btc_bought", 0.0)
        s.total_eth_bought = d.get("total_eth_bought", 0.0)
        s.last_buy_date = d.get("last_buy_date", "")
        s.last_buy_rsi = d.get("last_buy_rsi", 0.0)
        s.last_buy_bracket = d.get("last_buy_bracket", "")
        s.crash_levels_triggered = d.get("crash_levels_triggered", [])
        s.rolling_high_price = d.get("rolling_high_price", 0.0)
        s.buy_count = d.get("buy_count", 0)
        s.crash_buy_count = d.get("crash_buy_count", 0)
        s.total_days_active = d.get("total_days_active", 0)
        s.start_date = d.get("start_date", "")
        return s


# ── Pure logic functions ───────────────────────────────────────────────────────


def classify_rsi(
    rsi: float,
    cfg: DCAConfig,
    mvrv: Optional[float] = None,
) -> RSIBracket:
    """Classifie le RSI du BTC dans un bracket.

    Si le MVRV est fourni et en dessous du seuil, retourne DEEP_VALUE
    (prioritaire sur tous les autres brackets sauf OVERBOUGHT — on
    n'achète jamais en zone de surchauffe même si MVRV < 1).

    Args:
        rsi: Valeur RSI courante (0-100).
        cfg: Configuration DCA.
        mvrv: MVRV ratio courant (optionnel).

    Returns:
        RSIBracket correspondant.
    """
    if rsi > cfg.rsi_overbought:
        return RSIBracket.OVERBOUGHT

    # MVRV deep-value override (sauf overbought)
    if (
        cfg.mvrv_enabled
        and mvrv is not None
        and mvrv < cfg.mvrv_threshold
    ):
        return RSIBracket.DEEP_VALUE

    if rsi > cfg.rsi_warm:
        return RSIBracket.WARM
    elif rsi >= cfg.rsi_neutral_low:
        return RSIBracket.NEUTRAL
    else:
        return RSIBracket.OVERSOLD


def compute_daily_amount(
    rsi: float,
    cfg: DCAConfig,
    mvrv: Optional[float] = None,
) -> float:
    """Calcule le montant total d'achat quotidien selon le RSI du BTC.

    Args:
        rsi: Valeur RSI courante (0-100).
        cfg: Configuration DCA.
        mvrv: MVRV ratio courant (optionnel).

    Returns:
        Montant USD total pour la journée (avant split BTC/ETH).
    """
    bracket = classify_rsi(rsi, cfg, mvrv=mvrv)
    multiplier = cfg.rsi_multipliers.get(bracket.value, 0.0)
    return cfg.base_daily_amount * multiplier


def split_allocation(
    total_usd: float,
    cfg: DCAConfig,
) -> dict[str, float]:
    """Répartit le montant total entre BTC et ETH.

    Args:
        total_usd: Montant total USD à investir.
        cfg: Configuration DCA.

    Returns:
        Dict {"BTC-USD": amount, "ETH-USD": amount}.
    """
    return {
        cfg.btc_symbol: round(total_usd * cfg.btc_alloc, 2),
        cfg.eth_symbol: round(total_usd * cfg.eth_alloc, 2),
    }


def compute_rolling_high(highs: list[float], lookback: int) -> float:
    """Calcule le rolling high sur N dernières valeurs.

    Args:
        highs: Liste des prix high (daily candles).
        lookback: Nombre de valeurs à considérer.

    Returns:
        Le plus haut sur la fenêtre.
    """
    if not highs:
        return 0.0
    window = highs[-lookback:] if len(highs) >= lookback else highs
    return max(window)


def check_crash_triggers(
    current_price: float,
    rolling_high: float,
    state: DCAState,
    cfg: DCAConfig,
) -> list[tuple[float, float]]:
    """Vérifie si des niveaux de crash reserve sont déclenchés.

    Args:
        current_price: Prix BTC actuel.
        rolling_high: Plus haut sur la fenêtre (90j).
        state: État DCA courant.
        cfg: Configuration.

    Returns:
        Liste de (drop_pct, amount_usd) pour les niveaux nouvellement déclenchés.
    """
    if rolling_high <= 0:
        return []

    triggers: list[tuple[float, float]] = []
    drop_from_high = (rolling_high - current_price) / rolling_high

    for drop_pct, amount_usd in cfg.crash_levels:
        level_name = f"LEVEL_{int(drop_pct * 100)}"
        if drop_from_high >= drop_pct and level_name not in state.crash_levels_triggered:
            # Vérifier que la réserve a assez de fonds
            remaining_reserve = cfg.crash_reserve - state.total_spent_crash
            actual_amount = min(amount_usd, remaining_reserve)
            if actual_amount > 0:
                triggers.append((drop_pct, actual_amount))

    return triggers


def reset_crash_levels_if_recovered(
    current_price: float,
    rolling_high: float,
    state: DCAState,
    cfg: DCAConfig,
) -> list[str]:
    """Reset les crash levels si le prix est remonté au-dessus du seuil le plus bas.

    On reset tous les levels si le prix a récupéré au-dessus de -10% du rolling high,
    permettant aux niveaux de se redéclencher lors d'un nouveau crash.

    Args:
        current_price: Prix BTC actuel.
        rolling_high: Plus haut sur la fenêtre.
        state: État DCA.
        cfg: Configuration.

    Returns:
        Liste des niveaux qui ont été reset.
    """
    if rolling_high <= 0 or not state.crash_levels_triggered:
        return []

    drop = (rolling_high - current_price) / rolling_high
    # Reset si prix revenu à moins de 10% du high (recovery threshold)
    recovery_threshold = 0.10
    if drop < recovery_threshold:
        reset_levels = list(state.crash_levels_triggered)
        state.crash_levels_triggered.clear()
        return reset_levels

    return []


def remaining_dca_budget(state: DCAState, cfg: DCAConfig) -> float:
    """Budget DCA quotidien restant."""
    return max(0.0, cfg.active_budget - state.total_spent_dca)


def remaining_crash_budget(state: DCAState, cfg: DCAConfig) -> float:
    """Budget crash reserve restant."""
    return max(0.0, cfg.crash_reserve - state.total_spent_crash)


def is_budget_exhausted(state: DCAState, cfg: DCAConfig) -> bool:
    """Vérifie si le budget total est épuisé (DCA + crash)."""
    return (
        remaining_dca_budget(state, cfg) <= 0.0
        and remaining_crash_budget(state, cfg) <= 0.0
    )


def compute_buy_size(amount_usd: float, price: float) -> float:
    """Calcule la taille d'achat en unités de base.

    Args:
        amount_usd: Montant en USD à investir.
        price: Prix unitaire de l'actif.

    Returns:
        Taille en unités de base (ex: 0.00015 BTC).
    """
    if price <= 0 or amount_usd <= 0:
        return 0.0
    return amount_usd / price


def format_summary(state: DCAState, cfg: DCAConfig) -> dict:
    """Génère un résumé de l'état du bot DCA.

    Returns:
        Dict avec toutes les métriques clés pour le heartbeat/dashboard.
    """
    total_spent = state.total_spent_dca + state.total_spent_crash
    remaining_total = cfg.total_capital - total_spent
    return {
        "total_capital": cfg.total_capital,
        "total_spent": total_spent,
        "remaining": remaining_total,
        "dca_spent": state.total_spent_dca,
        "dca_remaining": remaining_dca_budget(state, cfg),
        "crash_spent": state.total_spent_crash,
        "crash_remaining": remaining_crash_budget(state, cfg),
        "btc_accumulated": state.total_btc_bought,
        "eth_accumulated": state.total_eth_bought,
        "buy_count": state.buy_count,
        "crash_buy_count": state.crash_buy_count,
        "days_active": state.total_days_active,
        "last_rsi": state.last_buy_rsi,
        "last_bracket": state.last_buy_bracket,
        "crash_levels_triggered": state.crash_levels_triggered,
    }
