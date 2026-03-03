"""
Moteur Intraday Momentum Continuation — logique pure (sans I/O).

Pipeline en 3 étapes :
  1. Filtre macro M15 : ATR > ATR_MA20, Volume > Vol_MA20
  2. Détection impulsion M5 : body ≥ 0.4%, volume ≥ 2×MA20,
     close dans top 20%, ADX(14) > 15, tendance EMA20 > EMA50
  3. Pullback 25-55%, RSI 40-65, prix > EMA20, pas de close < EMA50
  4. Entrée : close > high précédente + volume > MA10

Exécution: maker-only (0% fee Revolut X).
Mode: long-only.

Ce module est de la logique pure — aucun appel réseau.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.core.models import Candle

logger = logging.getLogger("tradex.momentum")


# ── Enums ──────────────────────────────────────────────────────────────────────

class MCSignalPhase(Enum):
    """Phase du pipeline de signal momentum continuation."""
    IDLE = "IDLE"                   # En attente d'impulsion
    IMPULSE_DETECTED = "IMPULSE"    # Impulsion détectée, attente pullback
    PULLBACK_ACTIVE = "PULLBACK"    # Pullback en cours
    ENTRY_TRIGGERED = "ENTRY"       # Signal d'entrée validé


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class MCConfig:
    """Paramètres de la stratégie Momentum Continuation."""

    # ── Filtre macro M15 ──
    atr_m15_period: int = 14
    atr_m15_ma_period: int = 20
    vol_m15_ma_period: int = 20

    # ── Détection impulsion M5 ──
    impulse_body_min_pct: float = 0.004     # Body ≥ 0.4%
    impulse_vol_mult: float = 2.0           # Volume ≥ 2× MA20
    impulse_vol_ma_period: int = 20
    impulse_close_top_pct: float = 0.20     # Close dans top 20%
    adx_period: int = 14
    adx_min: float = 15.0                   # ADX(14) > 15

    # ── Pullback ──
    pullback_retrace_min: float = 0.25      # Min 25% du move
    pullback_retrace_max: float = 0.55      # Max 55% du move
    rsi_period: int = 14
    rsi_pullback_min: float = 40.0
    rsi_pullback_max: float = 65.0
    ema_fast_period: int = 20               # EMA20
    ema_slow_period: int = 50               # EMA50
    pullback_max_bars: int = 35             # Max barres post-impulsion

    # ── Entrée ──
    entry_vol_ma_period: int = 10
    entry_vol_mult: float = 1.0

    # ── Risk model ──
    sl_min_pct: float = 0.004              # SL min 0.4%
    sl_max_pct: float = 0.008              # SL max 0.8%
    tp_pct: float = 0.020                  # TP 2%
    trail_trigger_pct: float = 0.004       # Trail après +0.4%
    trail_distance_pct: float = 0.0015     # Distance trailing 0.15%

    # ── Contraintes ──
    risk_per_trade: float = 0.04           # 4% risque par trade
    max_positions: int = 3
    max_position_pct: float = 0.90         # Max 90% du capital par position
    cooldown_bars: int = 6                 # Cooldown par paire (= 30min en M5)

    # ── Safety ──
    max_consecutive_losses: int = 5
    daily_loss_limit_pct: float = 0.02


# ── État d'un signal par paire ─────────────────────────────────────────────────

@dataclass
class MCSignalState:
    """État du pipeline de signal pour une paire."""
    phase: MCSignalPhase = MCSignalPhase.IDLE
    side: str = "LONG"                      # Toujours LONG (long-only)
    impulse_high: float = 0.0
    impulse_low: float = 0.0
    impulse_close: float = 0.0
    impulse_bar_idx: int = 0
    impulse_body_pct: float = 0.0
    impulse_vol_ratio: float = 0.0
    pullback_low: float = 0.0               # Lowest close pendant le pullback
    pullback_bars: int = 0
    retrace_pct: float = 0.0
    cooldown_until: int = 0                 # Bar index cooldown


@dataclass
class MCEntrySignal:
    """Signal d'entrée validé prêt à être exécuté."""
    symbol: str
    side: str                               # "LONG"
    entry_price: float                      # Close de la bougie de trigger
    sl_price: float                         # Swing low (clamped 0.4-0.8%)
    tp_price: float                         # Entry × (1 + tp_pct)
    size_usd: float                         # Taille en USD basée sur le risque
    impulse_body_pct: float
    impulse_vol_ratio: float
    retrace_pct: float


# ── Indicateurs techniques ─────────────────────────────────────────────────────

def ema(values: list[float], period: int) -> list[float]:
    """EMA — retourne liste de même taille (NaN rempli par 0)."""
    if not values or period <= 0:
        return [0.0] * len(values)
    k = 2.0 / (period + 1)
    result = [0.0] * len(values)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def sma(values: list[float], period: int) -> list[float]:
    """SMA — les premières `period-1` valeurs sont la moyenne cumulative."""
    if not values or period <= 0:
        return [0.0] * len(values)
    result = [0.0] * len(values)
    cum = 0.0
    for i in range(len(values)):
        cum += values[i]
        if i < period:
            result[i] = cum / (i + 1)
        else:
            cum -= values[i - period]
            result[i] = cum / period
    return result


def atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ATR via EMA du True Range."""
    if len(candles) < 2:
        return [0.0] * len(candles)
    tr = [candles[0].high - candles[0].low]
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr.append(max(
            c.high - c.low,
            abs(c.high - prev_close),
            abs(c.low - prev_close),
        ))
    return ema(tr, period)


def adx_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ADX simplifié via DI+/DI- et DX lissé par EMA."""
    n = len(candles)
    if n < 2:
        return [0.0] * n
    plus_dm = [0.0]
    minus_dm = [0.0]
    tr = [candles[0].high - candles[0].low]
    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        up = c.high - p.high
        dn = p.low - c.low
        plus_dm.append(max(up, 0) if up > dn else 0.0)
        minus_dm.append(max(dn, 0) if dn > up else 0.0)
        tr.append(max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close)))
    atr = ema(tr, period)
    spdm = ema(plus_dm, period)
    smdm = ema(minus_dm, period)
    dx = []
    for i in range(n):
        a = atr[i]
        if a <= 0:
            dx.append(0.0)
            continue
        di_p = spdm[i] / a * 100
        di_m = smdm[i] / a * 100
        s = di_p + di_m
        dx.append(abs(di_p - di_m) / s * 100 if s > 0 else 0.0)
    return ema(dx, period)


def rsi_series(candles: list[Candle], period: int = 14) -> list[float]:
    """RSI via EMA des gains/losses."""
    n = len(candles)
    if n < 2:
        return [50.0] * n
    gains = [0.0]
    losses = [0.0]
    for i in range(1, n):
        delta = candles[i].close - candles[i - 1].close
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))
    avg_gain = ema(gains, period)
    avg_loss = ema(losses, period)
    result = []
    for i in range(n):
        if avg_loss[i] == 0:
            result.append(100.0 if avg_gain[i] > 0 else 50.0)
        else:
            rs = avg_gain[i] / avg_loss[i]
            result.append(100.0 - 100.0 / (1.0 + rs))
    return result


def rolling_min(values: list[float], period: int) -> list[float]:
    """Min glissant."""
    result = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        result.append(min(values[start:i + 1]))
    return result


def rolling_max(values: list[float], period: int) -> list[float]:
    """Max glissant."""
    result = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        result.append(max(values[start:i + 1]))
    return result


def resample_m5_to_m15(candles: list[Candle]) -> list[Candle]:
    """Resample M5 → M15 (3 bougies par groupe)."""
    m15: list[Candle] = []
    for i in range(0, len(candles) - 2, 3):
        c0, c1, c2 = candles[i], candles[i + 1], candles[i + 2]
        m15.append(Candle(
            timestamp=c0.timestamp,
            open=c0.open,
            high=max(c0.high, c1.high, c2.high),
            low=min(c0.low, c1.low, c2.low),
            close=c2.close,
            volume=c0.volume + c1.volume + c2.volume,
        ))
    return m15


# ── Indicateurs pré-calculés ───────────────────────────────────────────────────

@dataclass
class MCIndicators:
    """Indicateurs pré-calculés pour une paire (M5 + M15)."""
    # M5
    ema20: list[float] = field(default_factory=list)
    ema50: list[float] = field(default_factory=list)
    adx: list[float] = field(default_factory=list)
    rsi: list[float] = field(default_factory=list)
    vol_ma20: list[float] = field(default_factory=list)
    vol_ma10: list[float] = field(default_factory=list)
    swing_low_10: list[float] = field(default_factory=list)
    # M15
    m15_atr: list[float] = field(default_factory=list)
    m15_atr_ma: list[float] = field(default_factory=list)
    m15_vol_ma: list[float] = field(default_factory=list)
    m15_bar_count: int = 0


def compute_indicators(candles_m5: list[Candle], cfg: MCConfig) -> MCIndicators:
    """Calcule tous les indicateurs nécessaires."""
    closes = [c.close for c in candles_m5]
    volumes = [c.volume for c in candles_m5]
    lows = [c.low for c in candles_m5]

    ind = MCIndicators()
    ind.ema20 = ema(closes, cfg.ema_fast_period)
    ind.ema50 = ema(closes, cfg.ema_slow_period)
    ind.adx = adx_series(candles_m5, cfg.adx_period)
    ind.rsi = rsi_series(candles_m5, cfg.rsi_period)
    ind.vol_ma20 = sma(volumes, cfg.impulse_vol_ma_period)
    ind.vol_ma10 = sma(volumes, cfg.entry_vol_ma_period)
    ind.swing_low_10 = rolling_min(lows, 10)

    # M15
    m15 = resample_m5_to_m15(candles_m5)
    ind.m15_bar_count = len(m15)
    if m15:
        m15_atr_raw = atr_series(m15, cfg.atr_m15_period)
        ind.m15_atr = m15_atr_raw
        ind.m15_atr_ma = sma(m15_atr_raw, cfg.atr_m15_ma_period)
        m15_vols = [c.volume for c in m15]
        ind.m15_vol_ma = sma(m15_vols, cfg.vol_m15_ma_period)

    return ind


# ── Moteur de détection ────────────────────────────────────────────────────────

class MomentumEngine:
    """Moteur de détection de signaux Momentum Continuation.

    Logique pure — reçoit des Candle et retourne des MCEntrySignal.
    Pas d'I/O, pas d'appel réseau.
    """

    def __init__(self, cfg: MCConfig) -> None:
        self.cfg = cfg
        self._states: dict[str, MCSignalState] = {}
        self._indicators: dict[str, MCIndicators] = {}
        self._candles: dict[str, list[Candle]] = {}
        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0.0
        self._current_day: str = ""

    def update_candles(self, symbol: str, candles: list[Candle]) -> None:
        """Met à jour les bougies M5 et recalcule les indicateurs."""
        self._candles[symbol] = candles
        self._indicators[symbol] = compute_indicators(candles, self.cfg)
        if symbol not in self._states:
            self._states[symbol] = MCSignalState()
        logger.debug(
            "[%s] Indicateurs calculés — %d bougies M5, %d M15",
            symbol, len(candles), self._indicators[symbol].m15_bar_count,
        )

    def process_new_candle(
        self,
        symbol: str,
        candle: Candle,
        n_open_positions: int,
        has_position: bool,
    ) -> Optional[MCEntrySignal]:
        """Traite une nouvelle bougie M5 et retourne un signal si validé.

        Args:
            symbol: Paire (ex: "ETH-USD").
            candle: Bougie M5 la plus récente (juste clôturée).
            n_open_positions: Nombre total de positions ouvertes (toutes paires).
            has_position: True si une position existe déjà sur cette paire.

        Returns:
            MCEntrySignal si les conditions sont remplies, None sinon.
        """
        if has_position:
            return None

        candles = self._candles.get(symbol, [])
        ind = self._indicators.get(symbol)
        if not candles or not ind or len(candles) < 60:
            return None

        # Ajouter la nouvelle bougie et recalculer les indicateurs
        candles.append(candle)
        # Garder un buffer raisonnable (500 bougies ≈ 42h)
        if len(candles) > 500:
            candles = candles[-500:]
        self._candles[symbol] = candles
        self._indicators[symbol] = compute_indicators(candles, self.cfg)
        ind = self._indicators[symbol]

        idx = len(candles) - 1
        state = self._states.get(symbol) or MCSignalState()
        self._states[symbol] = state

        # ── Safety checks ──
        day_str = candle.timestamp  # Sera converti par le bot
        if n_open_positions >= self.cfg.max_positions:
            return None
        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            return None

        # ── Cooldown ──
        if idx < state.cooldown_until:
            return None

        # ── Process signal ──
        return self._process_signal(symbol, candles, ind, idx, state)

    def _process_signal(
        self,
        symbol: str,
        candles: list[Candle],
        ind: MCIndicators,
        idx: int,
        state: MCSignalState,
    ) -> Optional[MCEntrySignal]:
        """Pipeline de détection : Impulse → Pullback → Entry."""
        c = candles[idx]

        # ── Phase IMPULSE/PULLBACK active → continuer le tracking ──
        if state.phase in (MCSignalPhase.IMPULSE_DETECTED, MCSignalPhase.PULLBACK_ACTIVE):
            return self._track_pullback_entry(symbol, candles, ind, idx, state)

        # ── Phase IDLE → chercher une nouvelle impulsion ──
        # Filtre macro M15
        m15_idx = idx // 3
        if m15_idx < 1 or m15_idx >= len(ind.m15_atr):
            return None
        if ind.m15_atr[m15_idx] <= ind.m15_atr_ma[m15_idx]:
            return None
        if m15_idx < len(ind.m15_vol_ma) and ind.m15_vol_ma[m15_idx] > 0:
            m15_candles = resample_m5_to_m15(candles)
            if m15_idx < len(m15_candles) and m15_candles[m15_idx].volume <= ind.m15_vol_ma[m15_idx]:
                return None

        # ── Détection impulsion (LONG only) ──
        body = c.close - c.open
        body_pct = abs(body) / c.open if c.open > 0 else 0
        candle_range = c.high - c.low
        if candle_range <= 0:
            return None

        is_bullish = body > 0 and body_pct >= self.cfg.impulse_body_min_pct
        close_in_top = (c.close - c.low) / candle_range >= (1 - self.cfg.impulse_close_top_pct)

        if not (is_bullish and close_in_top):
            return None

        # Volume
        vol_ma = ind.vol_ma20[idx] if idx < len(ind.vol_ma20) else 0
        if vol_ma <= 0 or c.volume < self.cfg.impulse_vol_mult * vol_ma:
            return None

        # ADX
        adx_val = ind.adx[idx] if idx < len(ind.adx) else 0
        if adx_val < self.cfg.adx_min:
            return None

        # Trend alignment : EMA20 > EMA50
        ema20 = ind.ema20[idx] if idx < len(ind.ema20) else 0
        ema50 = ind.ema50[idx] if idx < len(ind.ema50) else 0
        if ema20 <= ema50:
            return None

        vol_ratio = c.volume / vol_ma if vol_ma > 0 else 0

        # ✅ Impulsion détectée !
        state.phase = MCSignalPhase.IMPULSE_DETECTED
        state.side = "LONG"
        state.impulse_high = c.high
        state.impulse_low = c.low
        state.impulse_close = c.close
        state.impulse_bar_idx = idx
        state.impulse_body_pct = body_pct
        state.impulse_vol_ratio = vol_ratio
        state.pullback_low = c.close
        state.pullback_bars = 0

        logger.info(
            "[%s] ⚡ IMPULSE détectée — body=%.3f%%, vol=%.1fx, ADX=%.1f",
            symbol, body_pct * 100, vol_ratio, adx_val,
        )

        return None  # Pas encore d'entrée, attendre le pullback

    def _track_pullback_entry(
        self,
        symbol: str,
        candles: list[Candle],
        ind: MCIndicators,
        idx: int,
        state: MCSignalState,
    ) -> Optional[MCEntrySignal]:
        """Track pullback et vérifie les conditions d'entrée."""
        c = candles[idx]
        state.pullback_bars += 1
        cfg = self.cfg

        # ── Timeout ──
        if state.pullback_bars > cfg.pullback_max_bars:
            logger.debug("[%s] Pullback timeout (%d bars)", symbol, state.pullback_bars)
            self._reset_state(state)
            return None

        # ── Invalidation : close sous EMA50 ──
        ema50 = ind.ema50[idx] if idx < len(ind.ema50) else 0
        if c.close < ema50:
            logger.debug("[%s] Pullback invalidé — close < EMA50", symbol)
            self._reset_state(state)
            return None

        # ── Track le low du pullback ──
        if c.close < state.pullback_low:
            state.pullback_low = c.close

        # ── Calcul retrace ──
        impulse_range = state.impulse_high - state.impulse_low
        if impulse_range <= 0:
            self._reset_state(state)
            return None

        retrace = (state.impulse_close - state.pullback_low) / impulse_range
        state.retrace_pct = retrace

        # ── Vérification pullback suffisant ──
        state.phase = MCSignalPhase.PULLBACK_ACTIVE

        pullback_ok = cfg.pullback_retrace_min <= retrace <= cfg.pullback_retrace_max

        # RSI
        rsi_val = ind.rsi[idx] if idx < len(ind.rsi) else 50
        rsi_ok = cfg.rsi_pullback_min <= rsi_val <= cfg.rsi_pullback_max

        # Prix > EMA20
        ema20 = ind.ema20[idx] if idx < len(ind.ema20) else 0
        above_ema = c.close > ema20

        if not (pullback_ok and rsi_ok and above_ema):
            return None

        # ── Trigger d'entrée : close > prev high + volume ──
        if idx < 1:
            return None
        prev_high = candles[idx - 1].high
        vol_ma10 = ind.vol_ma10[idx] if idx < len(ind.vol_ma10) else 0

        if c.close <= prev_high:
            return None
        if vol_ma10 > 0 and c.volume < cfg.entry_vol_mult * vol_ma10:
            return None

        # ── Calcul SL ──
        swing_low = ind.swing_low_10[idx] if idx < len(ind.swing_low_10) else c.low
        sl_price = swing_low
        sl_pct = (c.close - sl_price) / c.close
        sl_pct = max(cfg.sl_min_pct, min(cfg.sl_max_pct, sl_pct))
        sl_price = c.close * (1 - sl_pct)

        # ── TP ──
        tp_price = c.close * (1 + cfg.tp_pct)

        # ── Size ──
        # Le sizing réel sera fait par le bot (il connaît le solde)
        # Ici on retourne les infos nécessaires

        logger.info(
            "[%s] ✅ ENTRY SIGNAL — retrace=%.1f%%, RSI=%.1f, SL=%.4f (%.2f%%)",
            symbol, retrace * 100, rsi_val, sl_price, sl_pct * 100,
        )

        # Reset state + cooldown
        cooldown = idx + cfg.cooldown_bars
        self._reset_state(state)
        state.cooldown_until = cooldown

        return MCEntrySignal(
            symbol=symbol,
            side="LONG",
            entry_price=c.close,
            sl_price=sl_price,
            tp_price=tp_price,
            size_usd=0.0,  # Sera calculé par le bot
            impulse_body_pct=state.impulse_body_pct if state.impulse_body_pct else 0,
            impulse_vol_ratio=state.impulse_vol_ratio if state.impulse_vol_ratio else 0,
            retrace_pct=retrace,
        )

    def _reset_state(self, state: MCSignalState) -> None:
        """Remet un état de signal à IDLE."""
        cooldown = state.cooldown_until
        state.phase = MCSignalPhase.IDLE
        state.impulse_high = 0.0
        state.impulse_low = 0.0
        state.impulse_close = 0.0
        state.impulse_bar_idx = 0
        state.impulse_body_pct = 0.0
        state.impulse_vol_ratio = 0.0
        state.pullback_low = 0.0
        state.pullback_bars = 0
        state.retrace_pct = 0.0
        state.cooldown_until = cooldown

    def record_trade_result(self, pnl: float) -> None:
        """Met à jour les compteurs de safety."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        self._daily_pnl += pnl

    def reset_daily(self) -> None:
        """Reset les compteurs journaliers (à appeler à minuit UTC)."""
        self._daily_pnl = 0.0

    # ── Position management (trailing stop) ────────────────────────────────────

    def check_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_sl: float,
        peak_price: float,
        trailing_active: bool,
    ) -> tuple[float, float, bool]:
        """Vérifie et met à jour le trailing stop.

        Returns:
            (new_sl, new_peak, trailing_now_active)
        """
        new_peak = max(peak_price, current_price)

        if not trailing_active:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct >= self.cfg.trail_trigger_pct:
                trailing_active = True
                new_sl = new_peak * (1 - self.cfg.trail_distance_pct)
                logger.info(
                    "📈 Trail activé — profit=%.2f%%, newSL=%.4f",
                    profit_pct * 100, new_sl,
                )
                return max(new_sl, current_sl), new_peak, True

        if trailing_active:
            new_sl = new_peak * (1 - self.cfg.trail_distance_pct)
            if new_sl > current_sl:
                return new_sl, new_peak, True

        return current_sl, new_peak, trailing_active

    def check_sl_hit(self, current_price: float, sl_price: float) -> bool:
        """Vérifie si le SL est touché (LONG only)."""
        return current_price <= sl_price

    def check_tp_hit(self, current_price: float, tp_price: float) -> bool:
        """Vérifie si le TP est touché (LONG only)."""
        return current_price >= tp_price
