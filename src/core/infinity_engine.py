"""
Infinity Bot — Logique pure (sans I/O)

Stratégie : DCA inversé sur baisse + vente en paliers progressifs
- Achat : 5 paliers (-5%, -10%, -15%, -20%, -25% du prix de référence)
- RSI H4 gate : <30 → tranche complète, 30-50 → moitié, >50 → pas d'achat
- Volume H4 : doit être > MA(20)
- Vente : 5 paliers (+0.8%, +1.5%, +2.2%, +3%, +4% du PMP)
- Vente seulement si RSI H4 > 50
- Override : +20% du PMP → vente totale
- Stop-loss : -15% du PMP → vente market (taker fee)
- Cycle : après vente complète, nouveau prix de référence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.core.models import Candle


# ── Enums ──────────────────────────────────────────────────────────────────────


class InfinityPhase(str, Enum):
    """Phase du cycle Infinity Bot."""
    WAITING = "WAITING"           # En attente d'entrée (pas de position)
    ACCUMULATING = "ACCUMULATING" # En cours d'achat (DCA)
    DISTRIBUTING = "DISTRIBUTING" # En cours de vente (paliers TP)


class InfinityExitReason(str, Enum):
    """Raison de sortie d'un cycle."""
    TP_COMPLETE = "TP_COMPLETE"       # Tous les paliers de vente atteints
    OVERRIDE_SELL = "OVERRIDE_SELL"   # +20% du PMP → vente forcée
    STOP_LOSS = "STOP_LOSS"           # -15% du PMP → stop market
    TIMEOUT = "TIMEOUT"               # Timeout max du cycle


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class InfinityConfig:
    """Configuration du bot Infinity."""

    initial_balance: float = 1000.0

    # ── Trailing high (référence dynamique) ──
    trailing_high_period: int = 72     # 72 bars H4 ≈ 12 jours
    entry_drop_pct: float = 0.05       # Entrée quand prix baisse ≥ 5% du trailing high

    # ── Paliers d'achat (DCA inversé) — relatifs au référence (trailing high au start) ──
    buy_levels: tuple = (-0.05, -0.10, -0.15, -0.20, -0.25)
    buy_amounts: tuple = (100.0, 200.0, 300.0, 400.0, 0.0)  # 0 = reste dispo
    buy_pcts: tuple = (0.25, 0.20, 0.15, 0.10, 0.00)  # % equity par palier (si scale_with_equity)
    scale_with_equity: bool = True    # Sizing proportionnel à l'equity courante
    max_invested_pct: float = 0.70    # Max 70% de l'equity par cycle
    max_buy_levels: int = 5

    # ── RSI gates pour achats ──
    rsi_full_buy: float = 30.0        # RSI < 30 → tranche complète
    rsi_half_buy: float = 50.0        # RSI 30-50 → moitié de la tranche
    # RSI > 50 → pas d'achat

    # ── Volume filter ──
    volume_ma_period: int = 20        # Volume > MA(20) pour valider
    require_volume_entry: bool = False # Volume gate pour première entrée (désactivé = plus de cycles)

    # ── Paliers de vente (% au-dessus du PMP) ──
    sell_levels: tuple = (0.008, 0.015, 0.022, 0.030, 0.040)
    sell_pcts: tuple = (0.20, 0.20, 0.20, 0.20, 0.20)  # 20% par palier
    rsi_sell_min: float = 0.0          # Pas de RSI gate sur les ventes (clé de perf)

    # ── Override / Stop ──
    override_sell_pct: float = 0.20   # +20% du PMP → vente complète
    stop_loss_pct: float = 0.15       # -15% du PMP → stop market
    use_breakeven_stop: bool = True   # Après vente TPn, stop remonte au breakeven (PMP)
    breakeven_after_level: int = 0    # Activer BE après quel sell level (0=TP1, 1=TP2, etc.)

    # ── Première entrée ──
    first_entry_rsi_max: float = 50.0   # RSI max pour premier buy

    # ── Timeframe & indicateurs ──
    rsi_period: int = 14
    volume_ma_len: int = 20

    # ── Frais ──
    maker_fee: float = 0.0           # 0% maker (Revolut X)
    taker_fee: float = 0.0009        # 0.09% taker (stop-loss)

    # ── Timing ──
    cycle_timeout_bars: int = 720    # 720 bougies H4 ≈ 120 jours max
    cooldown_bars: int = 6           # 6 bougies H4 = 24h cooldown entre cycles

    # ── Safety ──
    max_consecutive_stops: int = 3   # Arrêt après 3 stops consécutifs


# ── État du cycle ──────────────────────────────────────────────────────────────


@dataclass
class InfinityBuyLevel:
    """Un palier d'achat rempli."""
    level: int
    price: float
    size: float          # Quantité de BTC
    cost: float          # Coût en USD
    bar_idx: int
    timestamp: int = 0


@dataclass
class InfinitySellLevel:
    """Un palier de vente exécuté."""
    level: int
    price: float
    size: float
    proceeds: float
    bar_idx: int
    timestamp: int = 0


@dataclass
class InfinityCycle:
    """État d'un cycle complet (entrée → sortie)."""
    reference_price: float = 0.0      # Prix de référence (début du cycle)
    phase: InfinityPhase = InfinityPhase.WAITING

    buys: list = field(default_factory=list)     # InfinityBuyLevel
    sells: list = field(default_factory=list)    # InfinitySellLevel

    total_size: float = 0.0           # BTC accumulé
    total_cost: float = 0.0           # USD dépensé
    pmp: float = 0.0                  # Prix moyen pondéré

    size_remaining: float = 0.0       # BTC restant à vendre
    total_proceeds: float = 0.0       # USD récupéré par les ventes

    cycle_start_bar: int = 0
    cycle_start_ts: int = 0

    sell_levels_hit: set = field(default_factory=set)  # Indices des sell levels atteints
    breakeven_active: bool = False    # True après TP1 → stop remonte au PMP

    def recalc_pmp(self) -> None:
        """Recalcule le PMP après un achat."""
        if self.total_size > 0:
            self.pmp = self.total_cost / self.total_size


# ── Fonctions indicateurs (logique pure) ───────────────────────────────────────


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """RSI Wilder sur une série de closes."""
    n = len(closes)
    result = [50.0] * n
    if n < period + 1:
        return result

    gains = [0.0] * n
    losses = [0.0] * n

    for i in range(1, n):
        diff = closes[i] - closes[i - 1]
        gains[i] = max(diff, 0)
        losses[i] = max(-diff, 0)

    avg_gain = sum(gains[1: period + 1]) / period
    avg_loss = sum(losses[1: period + 1]) / period

    if avg_loss > 0:
        rs = avg_gain / avg_loss
        result[period] = 100 - 100 / (1 + rs)
    else:
        result[period] = 100.0

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            result[i] = 100 - 100 / (1 + rs)
        else:
            result[i] = 100.0

    return result


def sma_series(values: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    n = len(values)
    result = [0.0] * n
    if n < period:
        return result
    s = sum(values[:period])
    result[period - 1] = s / period
    for i in range(period, n):
        s += values[i] - values[i - period]
        result[i] = s / period
    return result


# ── Fonctions de décision (logique pure) ───────────────────────────────────────


def check_first_entry(
    close: float,
    trailing_high: float,
    entry_drop_pct: float,
    rsi: float,
    rsi_max: float,
    volume: float,
    volume_ma: float,
    require_volume: bool,
) -> bool:
    """Vérifie les conditions du premier achat d'un cycle.

    Entrée quand :
      - close a baissé de ≥ entry_drop_pct depuis le trailing high
      - RSI < rsi_max (pas surachat)
      - volume > MA (optionnel)
    """
    if trailing_high <= 0:
        return False
    # Drop suffisant depuis le plus haut récent
    drop = (trailing_high - close) / trailing_high
    if drop < entry_drop_pct:
        return False
    # RSI suffisamment bas
    if rsi > rsi_max:
        return False
    # Volume OK (optionnel)
    if require_volume:
        if volume_ma <= 0 or volume <= volume_ma:
            return False
    return True


def compute_buy_size(
    rsi: float,
    rsi_full: float,
    rsi_half: float,
    target_amount: float,
    cash_available: float,
    max_invested: float,
    already_invested: float,
) -> float:
    """
    Calcule le montant USD à investir selon le RSI.
    RSI < 30 → tranche complète
    RSI 30-50 → moitié
    RSI > 50 → 0
    """
    if rsi > rsi_half:
        return 0.0

    amount = target_amount
    if rsi >= rsi_full:
        amount = target_amount * 0.5  # Moitié si RSI 30-50

    # Cap au max investi
    if max_invested > 0:
        remaining_budget = max_invested - already_invested
        if remaining_budget <= 0:
            return 0.0
        amount = min(amount, remaining_budget)

    amount = min(amount, cash_available)
    return amount


def check_buy_conditions(
    close: float,
    pmp: float,
    rsi: float,
    rsi_half: float,
    volume: float,
    volume_ma: float,
) -> bool:
    """Vérifie si les conditions d'achat sont remplies (après le premier)."""
    # Prix doit être sous le PMP
    if pmp > 0 and close >= pmp:
        return False
    # RSI pas trop haut
    if rsi > rsi_half:
        return False
    # Volume OK
    if volume_ma <= 0 or volume <= volume_ma:
        return False
    return True


def check_sell_conditions(
    close: float,
    pmp: float,
    sell_level_pct: float,
    rsi: float,
    rsi_sell_min: float,
) -> bool:
    """Vérifie si un palier de vente est atteint."""
    if pmp <= 0:
        return False
    target = pmp * (1 + sell_level_pct)
    if close < target:
        return False
    if rsi < rsi_sell_min:
        return False
    return True


def check_override_sell(
    close: float,
    pmp: float,
    override_pct: float,
) -> bool:
    """Vérifie si le prix a dépassé +20% du PMP → vente forcée."""
    if pmp <= 0:
        return False
    return close >= pmp * (1 + override_pct)


def check_stop_loss(
    close: float,
    pmp: float,
    stop_pct: float,
) -> bool:
    """Vérifie si le stop-loss est atteint (-15% du PMP)."""
    if pmp <= 0:
        return False
    return close <= pmp * (1 - stop_pct)
