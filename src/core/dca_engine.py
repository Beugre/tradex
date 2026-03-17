"""
DCA Engine v2 — logique pure pour le bot DCA RSI-based sur Revolut X.

Stratégie v2 :
  1. Achat quotidien BTC/ETH basé sur RSI daily du BTC
  2. Multiplicateur MVRV progressif (×1.5 si <1.0, ×2.0 si <0.85)
  3. Filtre de régime de marché (MA200 → NORMAL/WEAK/CAPITULATION)
  4. Spending caps fixes (monthly $1500, weekly $400) + cooldown boosted
  5. Crash reserve proportionnelle (% de la réserve, ancre max(90j,180j))
  6. Observabilité complète via DCADecision structuré

Module sans I/O — utilisable dans core/ et backtest/.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────────────────


class RSIBracket(Enum):
    """Bracket RSI pour déterminer le montant d'achat quotidien."""
    OVERBOUGHT = "OVERBOUGHT"    # RSI > 70 → $0
    WARM = "WARM"                # 55 < RSI ≤ 70 → ×1
    NEUTRAL = "NEUTRAL"          # 45 ≤ RSI ≤ 55 → ×2
    OVERSOLD = "OVERSOLD"        # RSI < 45 → ×3


class MarketRegime(Enum):
    """Régime de marché basé sur la position du prix vs MA200."""
    NORMAL = "NORMAL"              # Prix > MA200
    WEAK = "WEAK"                  # Prix < MA200 mais > seuil capitulation
    CAPITULATION = "CAPITULATION"  # Prix < MA200 × capitulation_threshold


# ── Config ────────────────────────────────────────────────────────────────────────────


@dataclass
class DCAConfig:
    """Configuration du bot DCA v2."""
    # Budget total
    total_capital: float = 5300.0
    active_budget: float = 4200.0
    crash_reserve: float = 1100.0

    # Montants quotidiens
    base_daily_amount: float = 30.0
    max_daily_buy: float = 150.0

    # Multiplicateurs par bracket RSI
    rsi_multipliers: dict[str, float] = field(default_factory=lambda: {
        "OVERBOUGHT": 0.0,
        "WARM": 1.0,
        "NEUTRAL": 2.0,
        "OVERSOLD": 3.0,
    })

    # Allocation par paire (défaut — surchargée par régime)
    btc_alloc: float = 0.90
    eth_alloc: float = 0.10

    # Allocation dynamique par régime {regime_name: (btc_pct, eth_pct)}
    regime_alloc: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "NORMAL": (0.90, 0.10),
        "WEAK": (0.95, 0.05),
        "CAPITULATION": (1.00, 0.00),
    })

    # RSI thresholds
    rsi_overbought: float = 70.0
    rsi_warm: float = 55.0
    rsi_neutral_low: float = 45.0

    # Crash reserve levels : (drop_pct, pct_of_reserve)
    crash_levels: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.15, 0.25),    # -15% → 25% de la réserve
        (0.25, 0.35),    # -25% → 35% de la réserve
        (0.35, 0.40),    # -35% → 40% de la réserve
    ])

    # Rolling high periods
    crash_lookback_days: int = 90
    crash_anchor_long_days: int = 180

    # Execution
    execution_hour_utc: int = 10
    maker_wait_seconds: int = 60

    # MVRV multiplicateur progressif
    mvrv_enabled: bool = True
    mvrv_threshold: float = 1.0
    mvrv_deep_threshold: float = 0.85
    mvrv_mult_low: float = 1.5
    mvrv_mult_deep: float = 2.0

    # Crash reserve → 100% BTC
    crash_btc_only: bool = True

    # Spending caps (montants fixes $)
    monthly_cap: float = 1500.0
    weekly_cap: float = 400.0

    # Cooldown après achat boosté
    boost_cooldown_hours: float = 24.0
    boost_threshold: float = 120.0

    # Régime de marché (MA200)
    regime_filter_enabled: bool = True
    capitulation_threshold: float = 0.85

    # Symboles
    btc_symbol: str = "BTC-USD"
    eth_symbol: str = "ETH-USD"


# ── DCADecision — structured log ──────────────────────────────────────────────────────────


@dataclass
class DCADecision:
    """Trace structurée d'une décision DCA (observabilité complète)."""
    date: str = ""
    rsi: float = 0.0
    bracket: str = ""
    mvrv: Optional[float] = None
    mvrv_mult: float = 1.0
    regime: str = "NORMAL"
    base_amount: float = 0.0
    mvrv_amount: float = 0.0
    capped_amount: float = 0.0
    reason: str = ""
    monthly_spent: float = 0.0
    weekly_spent: float = 0.0
    monthly_cap: float = 0.0
    weekly_cap: float = 0.0
    cap_limited: bool = False
    cooldown_active: bool = False
    btc_alloc: float = 0.0
    eth_alloc: float = 0.0
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "rsi": self.rsi,
            "bracket": self.bracket,
            "mvrv": self.mvrv,
            "mvrv_mult": self.mvrv_mult,
            "regime": self.regime,
            "base_amount": self.base_amount,
            "mvrv_amount": self.mvrv_amount,
            "capped_amount": self.capped_amount,
            "reason": self.reason,
            "monthly_spent": self.monthly_spent,
            "weekly_spent": self.weekly_spent,
            "monthly_cap": self.monthly_cap,
            "weekly_cap": self.weekly_cap,
            "cap_limited": self.cap_limited,
            "cooldown_active": self.cooldown_active,
            "btc_alloc": self.btc_alloc,
            "eth_alloc": self.eth_alloc,
            "skipped": self.skipped,
        }


# ── State ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DCAState:
    """État persisté du bot DCA."""
    total_spent_dca: float = 0.0
    total_spent_crash: float = 0.0
    total_btc_bought: float = 0.0
    total_eth_bought: float = 0.0

    last_buy_date: str = ""
    last_buy_rsi: float = 0.0
    last_buy_bracket: str = ""

    crash_levels_triggered: list[str] = field(default_factory=list)
    rolling_high_price: float = 0.0

    buy_count: int = 0
    crash_buy_count: int = 0
    total_days_active: int = 0
    start_date: str = ""

    # v2 caps tracking
    monthly_spent: float = 0.0
    current_month: str = ""
    weekly_spent: float = 0.0
    current_week: str = ""
    last_boost_ts: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_spent_dca": self.total_spent_dca,
            "total_spent_crash": self.total_spent_crash,
            "total_btc_bought": self.total_btc_bought,
            "total_eth_bought": self.total_eth_bought,
            "last_buy_date": self.last_buy_date,
            "last_buy_rsi": self.last_buy_rsi,
            "last_buy_bracket": self.last_buy_bracket,
            "crash_levels_triggered": list(self.crash_levels_triggered),
            "rolling_high_price": self.rolling_high_price,
            "buy_count": self.buy_count,
            "crash_buy_count": self.crash_buy_count,
            "total_days_active": self.total_days_active,
            "start_date": self.start_date,
            "monthly_spent": self.monthly_spent,
            "current_month": self.current_month,
            "weekly_spent": self.weekly_spent,
            "current_week": self.current_week,
            "last_boost_ts": self.last_boost_ts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DCAState":
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
        # v2 fields (backwards compatible)
        s.monthly_spent = d.get("monthly_spent", 0.0)
        s.current_month = d.get("current_month", "")
        s.weekly_spent = d.get("weekly_spent", 0.0)
        s.current_week = d.get("current_week", "")
        s.last_boost_ts = d.get("last_boost_ts", 0.0)
        return s


# ── Pure logic functions ────────────────────────────────────────────────────────────


def classify_rsi(rsi: float, cfg: DCAConfig) -> RSIBracket:
    """Classifie le RSI du BTC dans un bracket (logique RSI pure, sans MVRV)."""
    if rsi > cfg.rsi_overbought:
        return RSIBracket.OVERBOUGHT
    if rsi > cfg.rsi_warm:
        return RSIBracket.WARM
    if rsi >= cfg.rsi_neutral_low:
        return RSIBracket.NEUTRAL
    return RSIBracket.OVERSOLD


def compute_mvrv_multiplier(
    mvrv: Optional[float],
    cfg: DCAConfig,
) -> float:
    """Calcule le multiplicateur MVRV progressif.

    - MVRV >= threshold (1.0)  → ×1.0 (pas de boost)
    - MVRV < threshold (1.0)   → ×mvrv_mult_low (1.5)
    - MVRV < deep (0.85)       → ×mvrv_mult_deep (2.0)
    """
    if not cfg.mvrv_enabled or mvrv is None:
        return 1.0
    if mvrv < cfg.mvrv_deep_threshold:
        return cfg.mvrv_mult_deep
    if mvrv < cfg.mvrv_threshold:
        return cfg.mvrv_mult_low
    return 1.0


def classify_regime(
    price: float,
    ma200: float,
    cfg: DCAConfig,
) -> MarketRegime:
    """Classifie le régime de marché basé sur prix vs MA200."""
    if not cfg.regime_filter_enabled or ma200 <= 0:
        return MarketRegime.NORMAL
    if price < ma200 * cfg.capitulation_threshold:
        return MarketRegime.CAPITULATION
    if price < ma200:
        return MarketRegime.WEAK
    return MarketRegime.NORMAL


def compute_regime_allocation(
    regime: MarketRegime,
    cfg: DCAConfig,
) -> tuple[float, float]:
    """Retourne (btc_pct, eth_pct) selon le régime de marché."""
    key = regime.value
    if key in cfg.regime_alloc:
        return cfg.regime_alloc[key]
    return (cfg.btc_alloc, cfg.eth_alloc)


def reset_period_counters(
    state: DCAState,
    current_month: str,
    current_week: str,
) -> None:
    """Reset les compteurs mensuels/hebdomadaires si la période a changé."""
    if state.current_month != current_month:
        state.monthly_spent = 0.0
        state.current_month = current_month
    if state.current_week != current_week:
        state.weekly_spent = 0.0
        state.current_week = current_week


def check_spending_caps(
    amount: float,
    state: DCAState,
    cfg: DCAConfig,
) -> tuple[float, bool]:
    """Applique les spending caps (monthly/weekly) au montant.

    Retourne le montant plafonné et un booléen indiquant si un cap a été appliqué.
    """
    monthly_remaining = max(0.0, cfg.monthly_cap - state.monthly_spent)
    weekly_remaining = max(0.0, cfg.weekly_cap - state.weekly_spent)
    cap = min(monthly_remaining, weekly_remaining)
    if amount <= cap:
        return (amount, False)
    return (cap, True)


def check_boost_cooldown(
    amount: float,
    state: DCAState,
    cfg: DCAConfig,
    now_ts: float = 0.0,
) -> bool:
    """Vérifie si le cooldown après achat boosté est actif."""
    if amount < cfg.boost_threshold:
        return False
    if state.last_boost_ts <= 0:
        return False
    elapsed_hours = (now_ts - state.last_boost_ts) / 3600.0
    return elapsed_hours < cfg.boost_cooldown_hours


def compute_daily_amount(
    rsi: float,
    cfg: DCAConfig,
    mvrv: Optional[float] = None,
    state: Optional[DCAState] = None,
    now_ts: float = 0.0,
) -> tuple[float, str, float]:
    """Calcule le montant d'achat quotidien v2 : RSI → MVRV mult → caps → cooldown.

    Returns:
        Tuple (final_amount, reason_string, mvrv_multiplier).
    """
    bracket = classify_rsi(rsi, cfg)
    multiplier = cfg.rsi_multipliers.get(bracket.value, 0.0)
    base_amount = cfg.base_daily_amount * multiplier

    if base_amount <= 0:
        return (0.0, f"RSI {rsi:.1f} → {bracket.value} (skip)", 1.0)

    # MVRV multiplier
    mvrv_mult = compute_mvrv_multiplier(mvrv, cfg)
    amount = base_amount * mvrv_mult

    # Cap to max_daily_buy
    if amount > cfg.max_daily_buy:
        amount = cfg.max_daily_buy

    reason_parts: list[str] = [f"RSI {rsi:.1f} → {bracket.value} (×{multiplier:.0f})"]
    if mvrv_mult > 1.0:
        reason_parts.append(f"MVRV {mvrv:.4f} → ×{mvrv_mult:.1f}")

    # Cooldown check
    if state is not None and check_boost_cooldown(amount, state, cfg, now_ts):
        amount = cfg.base_daily_amount
        reason_parts.append(f"COOLDOWN (réduit à ${cfg.base_daily_amount:.0f})")

    # Spending caps
    if state is not None:
        capped, was_capped = check_spending_caps(amount, state, cfg)
        if was_capped:
            reason_parts.append(f"CAP (${amount:.0f}→${capped:.0f})")
            amount = capped

    reason = " | ".join(reason_parts)
    return (amount, reason, mvrv_mult)


def compute_crash_anchor(
    highs: list[float],
    short_lookback: int,
    long_lookback: int,
) -> float:
    """Calcule l'ancre de crash = max(rolling_high_short, rolling_high_long).

    Utilise le max des deux fenêtres pour éviter les faux déclenchements.
    """
    high_short = compute_rolling_high(highs, short_lookback)
    high_long = compute_rolling_high(highs, long_lookback)
    return max(high_short, high_long)


def split_allocation(
    total_usd: float,
    cfg: DCAConfig,
    regime: Optional[MarketRegime] = None,
) -> dict[str, float]:
    """Répartit le montant entre BTC et ETH selon le régime."""
    if regime is not None:
        btc_pct, eth_pct = compute_regime_allocation(regime, cfg)
    else:
        btc_pct, eth_pct = cfg.btc_alloc, cfg.eth_alloc
    return {
        cfg.btc_symbol: round(total_usd * btc_pct, 2),
        cfg.eth_symbol: round(total_usd * eth_pct, 2),
    }


def compute_rolling_high(highs: list[float], lookback: int) -> float:
    """Calcule le rolling high sur N dernières valeurs."""
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
    """Vérifie si des niveaux de crash sont déclenchés.

    Les montants sont calculés comme pct_of_reserve × crash_reserve.
    """
    if rolling_high <= 0:
        return []

    triggers: list[tuple[float, float]] = []
    drop_from_high = (rolling_high - current_price) / rolling_high

    for drop_pct, pct_of_reserve in cfg.crash_levels:
        level_name = f"LEVEL_{int(drop_pct * 100)}"
        if drop_from_high >= drop_pct and level_name not in state.crash_levels_triggered:
            amount_usd = pct_of_reserve * cfg.crash_reserve
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
    """Reset les crash levels si le prix est remonté au-dessus de -10%."""
    if rolling_high <= 0 or not state.crash_levels_triggered:
        return []

    drop = (rolling_high - current_price) / rolling_high
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
    """Vérifie si le budget total est épuisé."""
    return (
        remaining_dca_budget(state, cfg) <= 0.0
        and remaining_crash_budget(state, cfg) <= 0.0
    )


def compute_buy_size(amount_usd: float, price: float) -> float:
    """Calcule la taille d'achat en unités de base."""
    if price <= 0 or amount_usd <= 0:
        return 0.0
    return amount_usd / price


def format_summary(state: DCAState, cfg: DCAConfig) -> dict:
    """Génère un résumé de l'état du bot DCA."""
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
        "monthly_spent": state.monthly_spent,
        "weekly_spent": state.weekly_spent,
        "monthly_cap": cfg.monthly_cap,
        "weekly_cap": cfg.weekly_cap,
    }
