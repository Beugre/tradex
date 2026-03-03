"""
Smart Infinity Grid 2.0 — logique pure (sans I/O).

Stratégie grid adaptative maker-only pour Revolut X (0% frais).

Pipeline :
  1. Filtre marché H4 : EMA200 intacte (tendance non cassée)
  2. Activation H1 : ATR élevé + RSI < 35 → zone de dip statistique
  3. Entrée initiale : RSI H1 < 30 + bougie de rejet (mèche basse ≥ 1.5× body)
  4. Grid dynamique : 5 niveaux (-1.5%, -3%, -5%, -7%, -10%) avec sizing croissant
  5. Sortie : TP dynamique basé sur PMP (scale-out TP1=+0.8%, TP2=+1.5%, TP3=+2.5%)
  6. Stop global : EMA200 H4 cassée, RSI < 25 pendant 12 bougies, DD > 20%

Exécution : maker-only (0% fee Revolut X).
Mode : long-only (dip-buy grid).

Ce module est de la logique pure — aucun appel réseau.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.core.models import Candle

logger = logging.getLogger("tradex.grid")


# ── Enums ──────────────────────────────────────────────────────────────────────


class GridPhase(Enum):
    """Phase du cycle de grid."""
    IDLE = "IDLE"                     # En attente de conditions d'activation
    WATCHING = "WATCHING"             # Conditions macro OK, attente signal d'entrée
    ACCUMULATING = "ACCUMULATING"     # Grid active, accumulation de positions
    EXITING = "EXITING"               # Scale-out en cours


class GridStopReason(Enum):
    """Raison de clôture forcée d'un cycle."""
    TP_COMPLETE = "TP_COMPLETE"       # Tous les TP atteints
    GLOBAL_STOP = "GLOBAL_STOP"      # Stop global (EMA/RSI/DD)
    TIMEOUT = "TIMEOUT"              # Timeout du cycle
    END = "END"                      # Fin du backtest


class GridMode(Enum):
    """Mode de fonctionnement du grid."""
    MICRO = "MICRO"     # Micro Dip Scalp (H1) — fréquence, TP courts
    DEEP = "DEEP"       # Deep Dip (H4) — sélectif, gros trades propres
    DUAL = "DUAL"       # Combinaison des deux modes


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class GridConfig:
    """Paramètres de la stratégie Smart Infinity Grid 2.0."""

    initial_balance: float = 1000.0

    # ── Filtre tendance H4 ──
    ema200_h4_period: int = 200        # EMA 200 H4
    ema200_break_pct: float = 0.02     # Cassure forte = close < EMA200 - 2%

    # ── Activation H1 ──
    atr_h1_period: int = 14
    atr_h1_ma_period: int = 20         # ATR H1 doit être > MA(20) de l'ATR
    rsi_h1_period: int = 14
    rsi_activation: float = 35.0       # RSI H1 < 35 → activation
    ema200_h1_period: int = 200        # EMA 200 H1 pour filtre

    # ── Entrée initiale ──
    rsi_entry: float = 30.0            # RSI H1 < 30 → première entrée
    rejection_wick_ratio: float = 1.5  # Mèche basse ≥ 1.5× body → bougie de rejet

    # ── Grid dynamique (5 niveaux) ──
    grid_levels: int = 5               # Nombre max de niveaux
    grid_distances: tuple = (-0.015, -0.03, -0.05, -0.07, -0.10)  # Distance depuis entrée initiale
    grid_multipliers: tuple = (1.0, 1.2, 1.5, 2.0, 2.5)           # Multiplicateur de taille

    # ── Take-Profit dynamique (PMP-based) ──
    tp1_pct: float = 0.008             # TP1 = PMP + 0.8%
    tp2_pct: float = 0.015             # TP2 = PMP + 1.5%
    tp3_pct: float = 0.025             # TP3 = PMP + 2.5%
    tp1_exit_pct: float = 0.40         # Vendre 40% au TP1
    tp2_exit_pct: float = 0.35         # Vendre 35% au TP2
    tp3_exit_pct: float = 0.25         # Vendre 25% au TP3 (reliquat)

    # ── Stop global intelligent ──
    rsi_stop_threshold: float = 25.0   # RSI < 25
    rsi_stop_bars: int = 12            # Pendant 12 bougies H1
    max_drawdown_pct: float = 0.20     # DD > 20% du cycle
    ema200_h4_stop: bool = True        # Stop si EMA200 H4 cassée fortement
    pmp_stop_pct: float = 0.04         # Stop si prix < PMP - 4% (0 = désactivé)
    ema_stop_break_pct: float = 0.02   # Seuil de cassure EMA200 pour stop (séparé du filtre)

    # ── Stop adaptatif ATR ──
    atr_stop_multiplier: float = 0.0   # Stop = PMP - ATR*mult (0 = désactivé, utilise pmp_stop_pct)

    # ── Filtre volatilité par paire ──
    max_atr_price_ratio: float = 0.0   # ATR(14)/Price max (0 = désactivé). Ex: 0.04 = 4%

    # ── Blacklist ──
    blacklist: tuple = ()              # Paires à exclure

    # ── Bounce mode ──
    bounce_enabled: bool = False       # Entrée alternative : rebond technique sur EMA20
    bounce_rsi_min: float = 35.0       # RSI min pour bounce (pas trop survendu)
    bounce_rsi_max: float = 50.0       # RSI max pour bounce (encore en zone basse)
    bounce_ema_touch_pct: float = 0.005  # Low dans ±0.5% de l'EMA20

    # ── Capital management ──
    capital_per_cycle_pct: float = 0.20  # 20% max engagé par cycle
    max_simultaneous_cycles: int = 3     # 3 cycles max simultanés
    base_risk_pct: float = 0.05          # Risque réel par cycle ≈ 5%

    # ── Contraintes ──
    cooldown_bars: int = 6               # Cooldown par paire (H1 bars)
    cycle_timeout_bars: int = 168        # Timeout max d'un cycle (168 H1 = 7 jours)

    # ── Frais (maker-only) ──
    fee_pct: float = 0.0                 # 0% maker fee (Revolut X)
    slippage_pct: float = 0.0002         # 0.02% slippage

    # ── Safety ──
    max_consecutive_losses: int = 4
    daily_loss_limit_pct: float = 0.05   # 5% daily loss limit


# ── Presets pour les modes MICRO et DEEP ───────────────────────────────────────


def micro_dip_config(balance: float = 1000.0) -> GridConfig:
    """
    Preset MICRO DIP SCALP (H1).
    Objectif = fréquence. Filtre relaxé, grid serrée, TP ajustés pour PF.
    """
    return GridConfig(
        initial_balance=balance,
        # Activation relaxée
        rsi_activation=45.0,            # RSI < 45 (était 40)
        rsi_entry=35.0,
        rejection_wick_ratio=0.8,       # Quasi pas de filtre wick
        # Grid serrée (3 niveaux : -1%, -2.5%, -4%)
        grid_levels=3,
        grid_distances=(-0.010, -0.025, -0.040, -0.06, -0.08),
        grid_multipliers=(1.0, 1.3, 1.6, 2.0, 2.5),
        # TP relevés pour meilleur PF (avg win > avg loss)
        tp1_pct=0.008,                  # +0.8%
        tp2_pct=0.016,                  # +1.6%
        tp3_pct=0.025,                  # +2.5%
        tp1_exit_pct=0.45,              # 45% au TP1 (cashflow rapide)
        tp2_exit_pct=0.35,
        tp3_exit_pct=0.20,
        # Stop adaptatif ATR (remplace PMP fixe)
        pmp_stop_pct=0.0,               # Désactivé (ATR prend le relais)
        atr_stop_multiplier=1.5,        # Stop = PMP - 1.5×ATR
        ema200_h4_stop=False,           # Pas d'EMA stop
        max_drawdown_pct=0.06,          # DD max 6%
        rsi_stop_bars=8,                # RSI < 25 pendant 8 barres
        # Filtre volatilité : exclure coins trop explosifs
        max_atr_price_ratio=0.04,       # ATR/Price > 4% = trop volatile
        # Blacklist structurelle
        blacklist=("FIL-USD", "FET-USD"),
        # Bounce mode activé
        bounce_enabled=True,
        bounce_rsi_min=35.0,
        bounce_rsi_max=50.0,
        bounce_ema_touch_pct=0.005,
        # Capital
        capital_per_cycle_pct=0.12,     # 12% par cycle (plus petit, plus de cycles)
        max_simultaneous_cycles=5,      # 5 cycles max simultanés
        # Timing rapide
        cooldown_bars=2,                # 2h cooldown (était 3)
        cycle_timeout_bars=72,          # 3 jours max
        # Safety
        max_consecutive_losses=5,
        daily_loss_limit_pct=0.04,
    )


def deep_dip_config(balance: float = 1000.0) -> GridConfig:
    """
    Preset DEEP DIP (H4).
    Objectif = qualité. Très sélectif, gros trades propres, PF élevé.
    """
    return GridConfig(
        initial_balance=balance,
        # Activation sélective
        rsi_activation=40.0,
        rsi_entry=33.0,
        rejection_wick_ratio=1.0,
        # Grid classique (3 niveaux)
        grid_levels=3,
        grid_distances=(-0.015, -0.03, -0.05, -0.07, -0.10),
        grid_multipliers=(1.0, 1.2, 1.5, 2.0, 2.5),
        # TP moyens
        tp1_pct=0.008,
        tp2_pct=0.015,
        tp3_pct=0.025,
        tp1_exit_pct=0.40,
        tp2_exit_pct=0.35,
        tp3_exit_pct=0.25,
        # Stops
        pmp_stop_pct=0.03,              # PMP -3%
        ema200_h4_stop=False,
        max_drawdown_pct=0.08,
        rsi_stop_bars=12,
        # Capital
        capital_per_cycle_pct=0.20,
        max_simultaneous_cycles=3,
        # Timing lent
        cooldown_bars=3,                # 12h en H4
        cycle_timeout_bars=168,         # 7 jours en H1 / 28 jours en H4
        # Safety
        max_consecutive_losses=4,
        daily_loss_limit_pct=0.05,
    )


# ── État d'un cycle de grid ────────────────────────────────────────────────────


@dataclass
class GridLevel:
    """Un niveau de la grid rempli."""
    level: int                # 0-4
    entry_price: float
    size: float               # Quantité achetée
    cost: float               # Coût total
    bar_idx: int              # Index de la bougie d'entrée
    timestamp: int            # Timestamp ms


@dataclass
class GridCycle:
    """État complet d'un cycle de grid pour une paire."""
    symbol: str
    phase: GridPhase = GridPhase.IDLE
    initial_entry_price: float = 0.0
    levels_filled: list[GridLevel] = field(default_factory=list)
    total_size: float = 0.0
    total_cost: float = 0.0
    pmp: float = 0.0                   # Prix moyen pondéré
    peak_price: float = 0.0            # Plus haut depuis début du cycle
    cycle_start_bar: int = 0
    cycle_start_ts: int = 0

    # TP tracking
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    size_remaining: float = 0.0        # Taille restante à vendre

    # Stop tracking
    rsi_below_stop_count: int = 0      # Compteur de bougies RSI < 25

    # Réalisé pendant le scale-out
    realized_pnl: float = 0.0
    realized_proceeds: float = 0.0

    def recalc_pmp(self) -> None:
        """Recalcule le prix moyen pondéré."""
        if self.total_size > 0:
            self.pmp = self.total_cost / self.total_size
        else:
            self.pmp = 0.0


# ── Indicateurs techniques (logique pure) ──────────────────────────────────────


def ema_series(values: list[float], period: int) -> list[float]:
    """EMA complète. Retourne liste de même longueur."""
    if not values:
        return []
    result = [values[0]]
    k = 2.0 / (period + 1)
    for i in range(1, len(values)):
        result.append(values[i] * k + result[-1] * (1 - k))
    return result


def sma_series(values: list[float], period: int) -> list[float]:
    """SMA glissante."""
    n = len(values)
    result = [0.0] * n
    cumsum = 0.0
    for i in range(n):
        cumsum += values[i]
        if i < period:
            result[i] = cumsum / (i + 1)
        else:
            cumsum -= values[i - period]
            result[i] = cumsum / period
    return result


def true_range_series(candles: list[Candle]) -> list[float]:
    """True Range pour chaque bougie."""
    trs: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            trs.append(c.high - c.low)
        else:
            prev_close = candles[i - 1].close
            trs.append(max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close)))
    return trs


def atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ATR EMA-lissé (Wilder)."""
    trs = true_range_series(candles)
    n = len(trs)
    result = [0.0] * n
    if n < period:
        return result
    initial = sum(trs[:period]) / period
    result[period - 1] = initial
    prev = initial
    for i in range(period, n):
        val = (prev * (period - 1) + trs[i]) / period
        result[i] = val
        prev = val
    for i in range(period - 1):
        result[i] = result[period - 1]
    return result


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """RSI Wilder."""
    n = len(closes)
    result = [50.0] * n
    if n < period + 1:
        return result

    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains[i] = delta
        else:
            losses[i] = -delta

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


# ── Fonctions de détection (logique pure) ──────────────────────────────────────


def check_h4_trend_ok(
    ema200_h4: float,
    close_h4: float,
    break_pct: float,
) -> bool:
    """Vérifie que l'EMA200 H4 n'est pas cassée fortement (close > EMA200 - break_pct)."""
    if ema200_h4 <= 0:
        return False
    threshold = ema200_h4 * (1 - break_pct)
    return close_h4 > threshold


def check_activation(
    rsi_h1: float,
    atr_h1: float,
    atr_ma_h1: float,
    ema200_h1: float,
    close_h1: float,
    rsi_threshold: float,
    ema_break_pct: float,
) -> bool:
    """Vérifie les conditions d'activation : ATR élevé + RSI < 35 + EMA200 intacte."""
    # ATR doit être supérieur à sa moyenne (volatilité suffisante)
    if atr_ma_h1 <= 0 or atr_h1 <= atr_ma_h1:
        return False
    # RSI H1 doit être en zone de survente
    if rsi_h1 >= rsi_threshold:
        return False
    # EMA200 H1 ne doit pas être cassée fortement
    threshold = ema200_h1 * (1 - ema_break_pct)
    if close_h1 < threshold:
        return False
    return True


def check_entry_signal(
    rsi_h1: float,
    candle: Candle,
    rsi_threshold: float,
    wick_ratio: float,
) -> bool:
    """
    Vérifie le signal d'entrée initiale :
    - RSI H1 < 30
    - Bougie de rejet (mèche basse ≥ wick_ratio × body)
    """
    if rsi_h1 >= rsi_threshold:
        return False

    body = abs(candle.close - candle.open)
    if body <= 0:
        return False

    # Mèche basse (bullish rejection)
    lower_wick = min(candle.open, candle.close) - candle.low
    if lower_wick < body * wick_ratio:
        return False

    return True


def check_bounce_entry(
    rsi_h1: float,
    candle: Candle,
    ema20_h1: float,
    rsi_min: float,
    rsi_max: float,
    ema_touch_pct: float,
) -> bool:
    """
    Signal d'entrée alternatif : rebond technique (bounce).
    Conditions :
    - RSI entre rsi_min et rsi_max (zone 35-50, pas extrême)
    - Low touche l'EMA20 (±0.5%)
    - Bougie verte (close > open) → confirmation du rebond
    """
    # RSI dans la zone intermédiaire
    if rsi_h1 < rsi_min or rsi_h1 > rsi_max:
        return False

    # EMA20 valide
    if ema20_h1 <= 0:
        return False

    # Low touche l'EMA20 (dans la bande ±ema_touch_pct)
    ema_low = ema20_h1 * (1 - ema_touch_pct)
    ema_high = ema20_h1 * (1 + ema_touch_pct)
    if not (candle.low <= ema_high and candle.close >= ema_low):
        return False

    # Bougie verte (confirmation que le rebond a lieu)
    if candle.close <= candle.open:
        return False

    return True


def compute_grid_prices(
    initial_price: float,
    distances: tuple,
) -> list[float]:
    """Calcule les prix des niveaux de la grid."""
    return [initial_price * (1 + d) for d in distances]


def compute_pmp(levels: list[GridLevel]) -> float:
    """Prix moyen pondéré de la position."""
    total_cost = sum(lv.cost for lv in levels)
    total_size = sum(lv.size for lv in levels)
    return total_cost / total_size if total_size > 0 else 0.0


def compute_tp_prices(pmp: float, tp1_pct: float, tp2_pct: float, tp3_pct: float) -> tuple[float, float, float]:
    """Calcule les 3 niveaux de take-profit basés sur le PMP."""
    return (
        pmp * (1 + tp1_pct),
        pmp * (1 + tp2_pct),
        pmp * (1 + tp3_pct),
    )


def check_volatility_filter(
    atr: float,
    price: float,
    max_atr_price_ratio: float,
) -> bool:
    """
    Vérifie si la paire n'est pas trop volatile pour le grid.
    Retourne True si la volatilité est acceptable.
    """
    if max_atr_price_ratio <= 0 or price <= 0:
        return True  # Filtre désactivé
    ratio = atr / price
    return ratio <= max_atr_price_ratio


def compute_atr_stop_distance(atr: float, multiplier: float) -> float:
    """
    Calcule la distance de stop basée sur l'ATR.
    Retourne la distance absolue (en prix).
    """
    return atr * multiplier


def check_global_stop(
    close: float,
    ema200_h4: float,
    ema_break_pct: float,
    rsi_h1: float,
    rsi_stop_threshold: float,
    rsi_below_count: int,
    rsi_stop_bars: int,
    cycle_cost: float,
    cycle_unrealized: float,
    max_dd_pct: float,
    use_ema_stop: bool,
    pmp: float = 0.0,
    pmp_stop_pct: float = 0.0,
    atr: float = 0.0,
    atr_stop_multiplier: float = 0.0,
) -> Optional[str]:
    """
    Vérifie les conditions de stop global.
    Retourne la raison du stop ou None.
    Priorité : ATR stop / PMP stop (rapide) → DD → RSI sustained → EMA200 (lent).
    """
    # 1a. Stop adaptatif ATR (plus intelligent que PMP fixe)
    if atr_stop_multiplier > 0 and atr > 0 and pmp > 0:
        atr_stop_price = pmp - compute_atr_stop_distance(atr, atr_stop_multiplier)
        if close < atr_stop_price:
            return "ATR_STOP"
    # 1b. PMP-based stop fixe (fallback si ATR stop désactivé)
    elif pmp_stop_pct > 0 and pmp > 0:
        pmp_threshold = pmp * (1 - pmp_stop_pct)
        if close < pmp_threshold:
            return "PMP_STOP"

    # 2. Drawdown max du cycle
    if cycle_cost > 0:
        dd = cycle_unrealized / cycle_cost
        if dd < -max_dd_pct:
            return "MAX_DRAWDOWN"

    # 3. RSI < 25 pendant 12+ bougies
    if rsi_below_count >= rsi_stop_bars:
        return "RSI_SUSTAINED_LOW"

    # 4. EMA200 H4 cassée fortement (le plus lent)
    if use_ema_stop and ema200_h4 > 0:
        threshold = ema200_h4 * (1 - ema_break_pct)
        if close < threshold:
            return "EMA200_H4_BREAK"

    return None
