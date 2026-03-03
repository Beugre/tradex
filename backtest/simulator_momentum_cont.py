"""
Moteur de backtest — Intraday Momentum Continuation v1.0

Logique :
  1. Filtre macro M15 : ATR > ATR_MA20, Volume > Vol_MA20, Spread < 0.05%
  2. Détection impulsion M5 : body ≥ 0.5%, volume ≥ 2×MA20, close top 20%, ADX(14) > 18
  3. Pullback intelligent : retrace 30-50% du move, RSI(14) 45-60,
     prix au-dessus EMA20, pas de close sous EMA50
  4. Entrée : close M5 > high précédente ET volume > MA10
  5. SL = dernier swing low (min 0.35%, max 0.7%)
  6. TP = 0.6% fixe, trail après +0.5%

Décorrélé :
  - Range (H4) : mean-reversion en NEUTRAL → jours
  - CrashBot (H4) : event-driven crashs → jours
  - Momentum Squeeze (15m) : compression→expansion BB/KC → heures
  - Momentum Continuation (M5) : impulsion→pullback→continuation → minutes/heures

Timeframes :
  - M15 pour filtre macro (resamplé depuis M5)
  - M5 pour signaux, entrée, sortie
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.core.models import Candle

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class MomentumContConfig:
    """Paramètres de la stratégie Intraday Momentum Continuation."""

    initial_balance: float = 1000.0

    # ── Filtre macro M15 ──
    atr_m15_period: int = 14           # Période ATR M15
    atr_m15_ma_period: int = 20        # ATR doit être > MA(20) de l'ATR
    vol_m15_ma_period: int = 20        # Volume M15 > MA(20)
    max_spread_pct: float = 0.0005     # Spread max 0.05%

    # ── Détection impulsion M5 ──
    impulse_body_min_pct: float = 0.005    # Body ≥ 0.5%
    impulse_vol_mult: float = 2.0          # Volume ≥ 2× MA20
    impulse_vol_ma_period: int = 20
    impulse_close_top_pct: float = 0.20    # Close dans top 20% de la bougie
    adx_period: int = 14
    adx_min: float = 18.0                  # ADX(14) M5 > 18

    # ── Pullback ──
    pullback_retrace_min: float = 0.30     # Min 30% du move
    pullback_retrace_max: float = 0.50     # Max 50% du move
    rsi_period: int = 14
    rsi_pullback_min: float = 45.0         # RSI min pendant pullback
    rsi_pullback_max: float = 60.0         # RSI max pendant pullback
    ema_fast_period: int = 20              # EMA20 rapide
    ema_slow_period: int = 50              # EMA50 lente
    pullback_max_bars: int = 30            # Max barres d'attente post-impulsion

    # ── Entrée ──
    entry_vol_ma_period: int = 10          # Volume entrée > MA10
    entry_vol_mult: float = 1.0            # Mult volume entrée (1.0 = > MA10)

    # ── Risk model ──
    sl_min_pct: float = 0.0035             # SL minimum 0.35%
    sl_max_pct: float = 0.007              # SL maximum 0.70%
    tp_pct: float = 0.006                  # TP fixe 0.6%
    trail_trigger_pct: float = 0.005       # Trail après +0.5%
    trail_distance_pct: float = 0.002      # Distance trailing 0.2%
    risk_per_trade: float = 0.005          # 0.5% risque par trade

    # ── Contraintes ──
    max_positions: int = 3
    max_position_pct: float = 0.40         # Max 40% du capital par position
    cooldown_bars: int = 12                # Cooldown par paire (= 1h en M5)

    # ── Safety ──
    max_consecutive_losses: int = 5        # Stop après 5 losses consécutives
    daily_loss_limit_pct: float = 0.02     # Hard daily loss limit 2%
    volatility_kill_atr_mult: float = 3.0  # Kill si ATR > 3× normal

    # ── Frais ──
    fee_pct: float = 0.001                 # 0.1% taker fee
    slippage_pct: float = 0.0003           # 0.03% slippage (liquide)

    # ── Mode ──
    allow_short: bool = True


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class MCTrade:
    """Trade terminé."""
    symbol: str
    side: str                   # "LONG" ou "SHORT"
    entry_price: float
    exit_price: float
    size: float
    entry_time: int             # timestamp ms
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str            # TP, SL, TRAILING, TIMEOUT, DAILY_LIMIT, END
    impulse_body_pct: float     # Body% de la bougie impulsive
    impulse_vol_ratio: float    # Volume ratio de la bougie impulsive
    retrace_pct: float          # Pourcentage de retracement atteint
    hold_bars: int


@dataclass
class MCPosition:
    """Position ouverte."""
    symbol: str
    side: str
    entry_price: float
    tp_price: float
    sl_price: float
    trailing_sl: float
    trailing_active: bool
    size: float
    cost: float
    entry_fee: float
    entry_time: int
    entry_bar_idx: int
    impulse_body_pct: float
    impulse_vol_ratio: float
    retrace_pct: float
    peak_price: float


# ── États du pipeline impulsion→pullback→entrée ──

@dataclass
class ImpulseState:
    """État du suivi impulsion→pullback pour un symbol."""
    # Impulsion détectée
    active: bool = False
    side: str = ""              # "LONG" ou "SHORT"
    impulse_bar_idx: int = 0
    impulse_high: float = 0.0   # High du move impulsif
    impulse_low: float = 0.0    # Low du move impulsif
    impulse_body_pct: float = 0.0
    impulse_vol_ratio: float = 0.0
    # Pullback tracking
    pullback_detected: bool = False
    pullback_low: float = 0.0   # Bas du pullback (LONG) / haut (SHORT)
    retrace_pct: float = 0.0
    # Invalidation
    bars_since_impulse: int = 0


@dataclass
class MCResult:
    """Résultat complet du backtest."""
    trades: list[MCTrade]
    equity_curve: list[tuple[int, float]]   # (timestamp_ms, equity)
    initial_balance: float
    final_equity: float
    n_impulses: int
    n_pullbacks: int
    n_entries: int
    n_filtered_macro: int
    n_filtered_safety: int
    config: MomentumContConfig
    start_date: datetime
    end_date: datetime
    pairs: list[str]


# ── Indicateurs techniques ─────────────────────────────────────────────────────


def _ema(values: list[float], period: int) -> list[float]:
    """EMA complète. Retourne liste de même longueur."""
    if not values:
        return []
    result = [values[0]]
    k = 2.0 / (period + 1)
    for i in range(1, len(values)):
        result.append(values[i] * k + result[-1] * (1 - k))
    return result


def _sma_series(values: list[float], period: int) -> list[float]:
    """SMA glissante. Les premières valeurs = moyenne partielle."""
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


def _true_range_series(candles: list[Candle]) -> list[float]:
    """True Range pour chaque bougie."""
    trs: list[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            trs.append(c.high - c.low)
        else:
            prev_close = candles[i - 1].close
            trs.append(max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close)))
    return trs


def _atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ATR EMA-lissé (Wilder)."""
    trs = _true_range_series(candles)
    n = len(trs)
    result = [0.0] * n
    if n < period:
        return result
    # Initial SMA
    initial = sum(trs[:period]) / period
    result[period - 1] = initial
    prev = initial
    for i in range(period, n):
        val = (prev * (period - 1) + trs[i]) / period
        result[i] = val
        prev = val
    # Back-fill pour warmup
    for i in range(period - 1):
        result[i] = result[period - 1]
    return result


def _adx_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ADX complet."""
    n = len(candles)
    result = [0.0] * n
    if n < period + 1:
        return result

    plus_dm = [0.0]
    minus_dm = [0.0]
    for i in range(1, n):
        up = candles[i].high - candles[i - 1].high
        down = candles[i - 1].low - candles[i].low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)

    trs = _true_range_series(candles)

    def _smooth(values: list[float], p: int) -> list[float]:
        smoothed = [0.0] * len(values)
        if len(values) <= p:
            return smoothed
        smoothed[p] = sum(values[1: p + 1])
        for i in range(p + 1, len(values)):
            smoothed[i] = smoothed[i - 1] - smoothed[i - 1] / p + values[i]
        return smoothed

    s_tr = _smooth(trs, period)
    s_plus = _smooth(plus_dm, period)
    s_minus = _smooth(minus_dm, period)

    dx_values = [0.0] * n
    for i in range(period, n):
        if s_tr[i] > 0:
            di_plus = 100 * s_plus[i] / s_tr[i]
            di_minus = 100 * s_minus[i] / s_tr[i]
            denom = di_plus + di_minus
            dx_values[i] = 100 * abs(di_plus - di_minus) / denom if denom > 0 else 0.0

    first_valid = [dx_values[i] for i in range(period, min(2 * period, n)) if dx_values[i] > 0]
    if not first_valid:
        return result
    adx_val = sum(first_valid) / len(first_valid)
    start_idx = 2 * period - 1
    if start_idx < n:
        result[start_idx] = adx_val
    for i in range(start_idx + 1, n):
        adx_val = (adx_val * (period - 1) + dx_values[i]) / period
        result[i] = adx_val
    return result


def _rsi_series(closes: list[float], period: int = 14) -> list[float]:
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


def _rolling_min(values: list[float], window: int) -> list[float]:
    """Rolling minimum lookback (inclut barre courante)."""
    n = len(values)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = min(values[start: i + 1])
    return result


def _rolling_max(values: list[float], window: int) -> list[float]:
    """Rolling maximum lookback (inclut barre courante)."""
    n = len(values)
    result = [0.0] * n
    for i in range(n):
        start = max(0, i - window + 1)
        result[i] = max(values[start: i + 1])
    return result


def _resample_m5_to_m15(candles_m5: list[Candle]) -> list[Candle]:
    """Resample M5 → M15 (agrège 3 bougies M5 consécutives)."""
    result: list[Candle] = []
    buffer: list[Candle] = []
    for c in candles_m5:
        buffer.append(c)
        if len(buffer) == 3:
            m15 = Candle(
                timestamp=buffer[0].timestamp,
                open=buffer[0].open,
                high=max(b.high for b in buffer),
                low=min(b.low for b in buffer),
                close=buffer[-1].close,
                volume=sum(b.volume for b in buffer),
            )
            result.append(m15)
            buffer = []
    return result


# ── Moteur ─────────────────────────────────────────────────────────────────────


class MomentumContEngine:
    """Simule la stratégie Momentum Continuation bar par bar sur M5."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: MomentumContConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config

        self.cash = config.initial_balance
        self.positions: dict[str, MCPosition] = {}
        self.trades: list[MCTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.cooldowns: dict[str, int] = {}  # symbol → bar_idx
        self.impulse_states: dict[str, ImpulseState] = {}

        # Compteurs
        self.n_impulses = 0
        self.n_pullbacks = 0
        self.n_entries = 0
        self.n_filtered_macro = 0
        self.n_filtered_safety = 0

        # Safety
        self._consecutive_losses = 0
        self._daily_pnl: dict[str, float] = defaultdict(float)  # "YYYY-MM-DD" → pnl
        self._safety_paused = False
        self._safety_paused_until_day = ""

        # Pré-calculés
        self._ind_m5: dict[str, dict] = {}
        self._ind_m15: dict[str, dict] = {}
        self._m15_candles: dict[str, list[Candle]] = {}

    def _precompute(self) -> None:
        """Pré-calcule tous les indicateurs (O(n) par paire)."""
        for sym, bars in self.candles.items():
            closes = [b.close for b in bars]
            volumes = [b.volume for b in bars]
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]

            self._ind_m5[sym] = {
                "closes": closes,
                "volumes": volumes,
                "highs": highs,
                "lows": lows,
                "ema20": _ema(closes, self.cfg.ema_fast_period),
                "ema50": _ema(closes, self.cfg.ema_slow_period),
                "adx": _adx_series(bars, self.cfg.adx_period),
                "rsi": _rsi_series(closes, self.cfg.rsi_period),
                "atr": _atr_series(bars, self.cfg.adx_period),
                "vol_ma20": _sma_series(volumes, self.cfg.impulse_vol_ma_period),
                "vol_ma10": _sma_series(volumes, self.cfg.entry_vol_ma_period),
                "swing_low": _rolling_min(lows, 10),
                "swing_high": _rolling_max(highs, 10),
            }

            # M15
            m15 = _resample_m5_to_m15(bars)
            self._m15_candles[sym] = m15
            m15_closes = [c.close for c in m15]
            m15_volumes = [c.volume for c in m15]
            m15_atr = _atr_series(m15, self.cfg.atr_m15_period)
            m15_atr_ma = _sma_series(m15_atr, self.cfg.atr_m15_ma_period)
            m15_vol_ma = _sma_series(m15_volumes, self.cfg.vol_m15_ma_period)

            self._ind_m15[sym] = {
                "atr": m15_atr,
                "atr_ma": m15_atr_ma,
                "vol_ma": m15_vol_ma,
                "volumes": m15_volumes,
            }

        logger.info("📊 Indicateurs pré-calculés pour %d paires (M5+M15)", len(self._ind_m5))

    def _m5_to_m15_idx(self, m5_idx: int) -> int:
        """Convertit un index M5 en index M15 (3 bougies M5 = 1 bougie M15)."""
        return m5_idx // 3

    def run(self) -> MCResult:
        symbols = sorted(self.candles.keys())
        if not symbols:
            raise ValueError("Aucune donnée")

        min_len = min(len(self.candles[s]) for s in symbols)
        warmup = max(60, self.cfg.ema_slow_period + 10, 2 * self.cfg.adx_period + 10)

        if min_len <= warmup:
            raise ValueError(f"Données insuffisantes ({min_len} barres, min {warmup})")

        logger.info(
            "⚡ Momentum Continuation — %d paires, %d bougies M5, $%.0f",
            len(symbols), min_len, self.cfg.initial_balance,
        )

        self._precompute()

        for sym in symbols:
            self.impulse_states[sym] = ImpulseState()

        for i in range(warmup, min_len):
            ts = self.candles[symbols[0]][i].timestamp

            # Day tracking pour daily loss limit
            day_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

            # Reset safety si nouveau jour
            if self._safety_paused and day_str != self._safety_paused_until_day:
                self._safety_paused = False
                self._consecutive_losses = 0
                logger.debug("🟢 Safety reset — nouveau jour %s", day_str)

            # 1. Check exits
            for sym in list(self.positions.keys()):
                if i < len(self.candles[sym]):
                    self._check_exit(sym, i, day_str)

            # 2. Check entries (si pas en pause safety)
            if not self._safety_paused and len(self.positions) < self.cfg.max_positions:
                for sym in symbols:
                    if sym in self.positions:
                        continue
                    if len(self.positions) >= self.cfg.max_positions:
                        break
                    if i >= len(self.candles[sym]):
                        continue
                    if sym in self.cooldowns and i < self.cooldowns[sym]:
                        continue
                    self._process_signal(sym, i)

            # 3. Equity snapshot (toutes les 12 barres = 1h en M5)
            if i % 12 == 0:
                eq = self._compute_equity(i, symbols)
                self.equity_curve.append((ts, eq))

        # Close remaining
        for sym in list(self.positions.keys()):
            last_idx = len(self.candles[sym]) - 1
            self._close_position(sym, self.candles[sym][last_idx].close, last_idx, "END", "")

        final_eq = self.cash
        first_ts = self.candles[symbols[0]][warmup].timestamp
        last_ts = self.candles[symbols[0]][min_len - 1].timestamp

        if self.equity_curve:
            self.equity_curve.append((last_ts, final_eq))

        return MCResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            n_impulses=self.n_impulses,
            n_pullbacks=self.n_pullbacks,
            n_entries=self.n_entries,
            n_filtered_macro=self.n_filtered_macro,
            n_filtered_safety=self.n_filtered_safety,
            config=self.cfg,
            start_date=datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc),
            pairs=list(self.candles.keys()),
        )

    # ── Pipeline : impulse → pullback → entry ──────────────────────────────

    def _process_signal(self, symbol: str, idx: int) -> None:
        """Pipeline 3 étapes : impulsion → pullback → entrée."""
        ind = self._ind_m5[symbol]
        candle = self.candles[symbol][idx]
        state = self.impulse_states[symbol]

        # ── Si impulsion déjà active → continuer le tracking (pas de macro filter) ──
        if state.active:
            state.bars_since_impulse += 1

            # Timeout
            if state.bars_since_impulse > self.cfg.pullback_max_bars:
                state.active = False
                return

            # Chercher pullback
            if not state.pullback_detected:
                self._detect_pullback(symbol, idx, state)
                return

            # Pullback détecté → chercher trigger d'entrée
            self._check_entry_trigger(symbol, idx, state)
            return

        # ── Pas d'impulsion active → filtre macro M15 puis chercher impulsion ──
        m15_idx = self._m5_to_m15_idx(idx)
        if not self._check_macro_filter(symbol, m15_idx):
            self.n_filtered_macro += 1
            return

        self._detect_impulse(symbol, idx)

    def _check_macro_filter(self, symbol: str, m15_idx: int) -> bool:
        """Filtre macro M15 : ATR > ATR_MA20 ET Volume > Vol_MA20."""
        ind = self._ind_m15[symbol]
        if m15_idx >= len(ind["atr"]) or m15_idx < 1:
            return False

        atr_val = ind["atr"][m15_idx]
        atr_ma = ind["atr_ma"][m15_idx]
        vol = ind["volumes"][m15_idx]
        vol_ma = ind["vol_ma"][m15_idx]

        if atr_ma <= 0 or vol_ma <= 0:
            return False

        return atr_val > atr_ma and vol > vol_ma

    def _detect_impulse(self, symbol: str, idx: int) -> None:
        """Détecte une bougie impulsive M5."""
        ind = self._ind_m5[symbol]
        c = self.candles[symbol][idx]

        # Body %
        body = abs(c.close - c.open)
        body_pct = body / c.open if c.open > 0 else 0
        if body_pct < self.cfg.impulse_body_min_pct:
            return

        # Volume ≥ 2× MA20
        vol_ma = ind["vol_ma20"][idx]
        if vol_ma <= 0:
            return
        vol_ratio = c.volume / vol_ma
        if vol_ratio < self.cfg.impulse_vol_mult:
            return

        # Close dans top 20% (LONG) ou bottom 20% (SHORT) de la bougie
        candle_range = c.high - c.low
        if candle_range <= 0:
            return

        if c.close > c.open:
            # Bullish impulse — close in top 20%
            close_position = (c.close - c.low) / candle_range
            if close_position < (1.0 - self.cfg.impulse_close_top_pct):
                return
            side = "LONG"
        else:
            # Bearish impulse — close in bottom 20%
            close_position = (c.high - c.close) / candle_range
            if close_position < (1.0 - self.cfg.impulse_close_top_pct):
                return
            side = "SHORT"

        if side == "SHORT" and not self.cfg.allow_short:
            return

        # ADX(14) > 18
        adx_val = ind["adx"][idx]
        if adx_val < self.cfg.adx_min:
            return

        # Trend alignment : EMA20 vs EMA50
        ema20 = ind["ema20"][idx]
        ema50 = ind["ema50"][idx]
        if side == "LONG" and ema20 < ema50:
            return  # Pas de LONG si EMAs baissières
        if side == "SHORT" and ema20 > ema50:
            return  # Pas de SHORT si EMAs haussières

        # ✅ Impulsion valide
        self.n_impulses += 1
        state = self.impulse_states[symbol]
        state.active = True
        state.side = side
        state.impulse_bar_idx = idx
        state.impulse_high = c.high
        state.impulse_low = c.low
        state.impulse_body_pct = body_pct
        state.impulse_vol_ratio = vol_ratio
        state.pullback_detected = False
        state.bars_since_impulse = 0

        # Initialiser pullback_low au CLOSE (pas au low/high) pour que le
        # retrace démarre à ~0% et augmente au fur et à mesure du pullback
        if side == "LONG":
            state.pullback_low = c.close   # Retrace part du close (haut du move)
        else:
            state.pullback_low = c.close   # Pour SHORT, track le rebond depuis le close (bas du move)

        logger.debug(
            "⚡ IMPULSE %s %s @ %.6f | body=%.2f%% vol=%.1fx ADX=%.1f",
            side, symbol, c.close, body_pct * 100, vol_ratio, adx_val,
        )

    def _detect_pullback(self, symbol: str, idx: int, state: ImpulseState) -> None:
        """Vérifie si le pullback est dans la zone 30-50% du move."""
        ind = self._ind_m5[symbol]
        c = self.candles[symbol][idx]
        close = ind["closes"][idx]
        rsi = ind["rsi"][idx]
        ema20 = ind["ema20"][idx]
        ema50 = ind["ema50"][idx]

        if state.side == "LONG":
            move = state.impulse_high - state.impulse_low
            if move <= 0:
                state.active = False
                return

            # Track le plus bas du pullback
            if c.low < state.pullback_low:
                state.pullback_low = c.low

            retrace = (state.impulse_high - state.pullback_low) / move
            state.retrace_pct = retrace

            # Close sous EMA50 → invalidation
            if close < ema50:
                state.active = False
                return

            # Retracement dans la zone 30-50% ?
            if retrace < self.cfg.pullback_retrace_min:
                return  # Pas encore assez retracé
            if retrace > self.cfg.pullback_retrace_max:
                state.active = False  # Trop retracé → invalidé
                return

            # RSI dans la zone 45-60
            if rsi < self.cfg.rsi_pullback_min or rsi > self.cfg.rsi_pullback_max:
                return

            # Prix > EMA20
            if close < ema20:
                return

            # ✅ Pullback valide
            state.pullback_detected = True
            self.n_pullbacks += 1

        else:  # SHORT
            move = state.impulse_high - state.impulse_low
            if move <= 0:
                state.active = False
                return

            # Track le plus haut du pullback (= state.pullback_low pour SHORT)
            if c.high > state.pullback_low:
                state.pullback_low = c.high

            retrace = (state.pullback_low - state.impulse_low) / move
            state.retrace_pct = retrace

            # Close au-dessus EMA50 → invalidation
            if close > ema50:
                state.active = False
                return

            if retrace < self.cfg.pullback_retrace_min:
                return
            if retrace > self.cfg.pullback_retrace_max:
                state.active = False
                return

            # RSI inversé pour SHORT : 40-55 (miroir)
            rsi_short_min = 100 - self.cfg.rsi_pullback_max
            rsi_short_max = 100 - self.cfg.rsi_pullback_min
            if rsi < rsi_short_min or rsi > rsi_short_max:
                return

            # Prix < EMA20
            if close > ema20:
                return

            state.pullback_detected = True
            self.n_pullbacks += 1

    def _check_entry_trigger(self, symbol: str, idx: int, state: ImpulseState) -> None:
        """Vérifie le trigger d'entrée : close > high précédente + volume."""
        ind = self._ind_m5[symbol]
        c = self.candles[symbol][idx]
        close = ind["closes"][idx]
        prev_high = ind["highs"][idx - 1] if idx > 0 else 0
        prev_low = ind["lows"][idx - 1] if idx > 0 else 0

        # Volume > MA10
        vol_ma10 = ind["vol_ma10"][idx]
        if vol_ma10 > 0 and c.volume < vol_ma10 * self.cfg.entry_vol_mult:
            return

        if state.side == "LONG":
            # Close > high bougie précédente
            if close <= prev_high:
                return
        else:
            # Close < low bougie précédente
            if close >= prev_low:
                return

        # ── SL = dernier swing low/high ──
        if state.side == "LONG":
            sl_price = ind["swing_low"][idx]
            sl_distance_pct = (close - sl_price) / close if close > 0 else 0
        else:
            sl_price = ind["swing_high"][idx]
            sl_distance_pct = (sl_price - close) / close if close > 0 else 0

        # Clamp SL entre min et max
        if sl_distance_pct < self.cfg.sl_min_pct:
            # SL trop proche → élargir
            if state.side == "LONG":
                sl_price = close * (1 - self.cfg.sl_min_pct)
            else:
                sl_price = close * (1 + self.cfg.sl_min_pct)
            sl_distance_pct = self.cfg.sl_min_pct
        elif sl_distance_pct > self.cfg.sl_max_pct:
            # SL trop loin → raccourcir
            if state.side == "LONG":
                sl_price = close * (1 - self.cfg.sl_max_pct)
            else:
                sl_price = close * (1 + self.cfg.sl_max_pct)
            sl_distance_pct = self.cfg.sl_max_pct

        # ── TP = 0.6% fixe ──
        if state.side == "LONG":
            tp_price = close * (1 + self.cfg.tp_pct)
        else:
            tp_price = close * (1 - self.cfg.tp_pct)

        # ── Position sizing ──
        sl_distance = abs(close - sl_price)
        if sl_distance <= 0:
            state.active = False
            return

        risk_amount = self.cash * self.cfg.risk_per_trade
        size = risk_amount / sl_distance
        cost = size * close

        max_cost = self.cash * self.cfg.max_position_pct
        if cost > max_cost:
            size = max_cost / close
            cost = size * close

        # En spot, la position ne peut pas dépasser le cash disponible
        # En futures (max_position_pct > 1), on vérifie la marge requise
        margin_required = cost / self.cfg.max_position_pct if self.cfg.max_position_pct > 1 else cost
        if margin_required > self.cash or cost <= 0:
            state.active = False
            return

        # ── Slippage et frais ──
        slip = self.cfg.slippage_pct
        if state.side == "LONG":
            entry_price = close * (1 + slip)
        else:
            entry_price = close * (1 - slip)

        fee = cost * self.cfg.fee_pct
        # En futures, on bloque la marge (cost / levier) ; en spot, le coût total
        if self.cfg.max_position_pct > 1:
            cash_used = cost / self.cfg.max_position_pct + fee
        else:
            cash_used = cost + fee
        self.cash -= cash_used
        self.n_entries += 1

        self.positions[symbol] = MCPosition(
            symbol=symbol,
            side=state.side,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            trailing_sl=sl_price,
            trailing_active=False,
            size=size,
            cost=cost,
            entry_fee=fee,
            entry_time=c.timestamp,
            entry_bar_idx=idx,
            impulse_body_pct=state.impulse_body_pct,
            impulse_vol_ratio=state.impulse_vol_ratio,
            retrace_pct=state.retrace_pct,
            peak_price=entry_price,
        )

        # Reset impulse state
        state.active = False

        logger.debug(
            "🎯 ENTRY %s %s @ %.6f | SL=%.6f (%.2f%%) | TP=%.6f | retrace=%.0f%%",
            state.side, symbol, entry_price, sl_price, sl_distance_pct * 100,
            tp_price, state.retrace_pct * 100,
        )

    # ── Exit logic ─────────────────────────────────────────────────────────

    def _check_exit(self, symbol: str, idx: int, day_str: str) -> None:
        pos = self.positions[symbol]
        c = self.candles[symbol][idx]

        if pos.side == "LONG":
            # Update peak
            if c.high > pos.peak_price:
                pos.peak_price = c.high

            # SL hit
            if c.low <= pos.trailing_sl:
                reason = "TRAILING" if pos.trailing_active else "SL"
                self._close_position(symbol, pos.trailing_sl, idx, reason, day_str)
                return

            # TP hit (si trailing pas encore actif)
            if not pos.trailing_active and c.high >= pos.tp_price:
                self._close_position(symbol, pos.tp_price, idx, "TP", day_str)
                return

            # Trail activation : après +0.5%, trail à 0.2% du peak
            profit_pct = (pos.peak_price - pos.entry_price) / pos.entry_price
            if profit_pct >= self.cfg.trail_trigger_pct:
                pos.trailing_active = True
                new_sl = pos.peak_price * (1 - self.cfg.trail_distance_pct)
                if new_sl > pos.trailing_sl:
                    pos.trailing_sl = new_sl

        else:  # SHORT
            if c.low < pos.peak_price:
                pos.peak_price = c.low

            if c.high >= pos.trailing_sl:
                reason = "TRAILING" if pos.trailing_active else "SL"
                self._close_position(symbol, pos.trailing_sl, idx, reason, day_str)
                return

            if not pos.trailing_active and c.low <= pos.tp_price:
                self._close_position(symbol, pos.tp_price, idx, "TP", day_str)
                return

            profit_pct = (pos.entry_price - pos.peak_price) / pos.entry_price
            if profit_pct >= self.cfg.trail_trigger_pct:
                pos.trailing_active = True
                new_sl = pos.peak_price * (1 + self.cfg.trail_distance_pct)
                if new_sl < pos.trailing_sl:
                    pos.trailing_sl = new_sl

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(self, symbol: str, exit_price: float, idx: int, reason: str, day_str: str) -> None:
        pos = self.positions.pop(symbol)

        slip = self.cfg.slippage_pct
        if reason in ("SL", "TIMEOUT", "END"):
            if pos.side == "LONG":
                actual_exit = exit_price * (1 - slip)
            else:
                actual_exit = exit_price * (1 + slip)
        else:
            actual_exit = exit_price

        fee = pos.size * actual_exit * self.cfg.fee_pct

        if pos.side == "LONG":
            proceeds = pos.size * actual_exit - fee
            pnl_usd = proceeds - pos.cost - pos.entry_fee
        else:
            pnl_usd = pos.size * (pos.entry_price - actual_exit) - fee - pos.entry_fee

        # Retourner la marge + PnL
        if self.cfg.max_position_pct > 1:
            margin_used = pos.cost / self.cfg.max_position_pct
            self.cash += margin_used + pnl_usd
        else:
            self.cash += pos.cost + pnl_usd
        pnl_pct = pnl_usd / pos.cost if pos.cost > 0 else 0
        bars_held = idx - pos.entry_bar_idx

        self.trades.append(MCTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=self.candles[symbol][idx].timestamp,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            impulse_body_pct=pos.impulse_body_pct,
            impulse_vol_ratio=pos.impulse_vol_ratio,
            retrace_pct=pos.retrace_pct,
            hold_bars=bars_held,
        ))

        self.cooldowns[symbol] = idx + self.cfg.cooldown_bars

        # Safety tracking
        if pnl_usd < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Daily PnL tracking
        if day_str:
            self._daily_pnl[day_str] += pnl_usd
            daily_loss_pct = self._daily_pnl[day_str] / self.cfg.initial_balance
            if daily_loss_pct <= -self.cfg.daily_loss_limit_pct:
                self._safety_paused = True
                self._safety_paused_until_day = day_str
                self.n_filtered_safety += 1
                logger.debug("🛑 Daily loss limit hit: $%.2f (%.2f%%)", self._daily_pnl[day_str], daily_loss_pct * 100)

        # Consecutive loss limit
        if self._consecutive_losses >= self.cfg.max_consecutive_losses:
            self._safety_paused = True
            self._safety_paused_until_day = day_str
            self.n_filtered_safety += 1
            logger.debug("🛑 %d consecutive losses — paused", self._consecutive_losses)

    # ── Equity ─────────────────────────────────────────────────────────────

    def _compute_equity(self, idx: int, symbols: list[str]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            if idx < len(self.candles[sym]):
                current_price = self.candles[sym][idx].close
                if pos.side == "LONG":
                    eq += pos.size * current_price
                else:
                    unrealized = pos.size * (pos.entry_price - current_price)
                    eq += pos.cost + unrealized
        return eq


# ── Métriques ──────────────────────────────────────────────────────────────────


def compute_mc_metrics(result: MCResult) -> dict:
    """Calcule les KPIs du backtest Momentum Continuation."""
    trades = result.trades
    eq = result.equity_curve
    init = result.initial_balance
    final = result.final_equity
    days = max((result.end_date - result.start_date).days, 1)
    years = days / 365.25

    total_return = (final - init) / init if init > 0 else 0
    cagr = (final / init) ** (1 / years) - 1 if final > 0 and years > 0 else 0

    # Monthly return
    monthly_return = (1 + total_return) ** (30.4375 / days) - 1 if days > 0 else 0

    # Drawdown
    peak = init
    max_dd = 0.0
    for _, equity in eq:
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        max_dd = min(max_dd, dd)

    # Sharpe / Sortino sur returns hourly (12 bougies M5 = 1h)
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev_eq = eq[i - 1][1]
        if prev_eq > 0:
            returns.append((eq[i][1] - prev_eq) / prev_eq)

    sharpe = _sharpe(returns)
    sortino = _sortino(returns)

    n = len(trades)
    if n > 0:
        wins = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]
        win_rate = len(wins) / n
        gross_profit = sum(t.pnl_usd for t in wins) or 0
        gross_loss = abs(sum(t.pnl_usd for t in losses)) or 1e-9
        profit_factor = gross_profit / gross_loss
        avg_pnl_usd = sum(t.pnl_usd for t in trades) / n
        avg_pnl_pct = sum(t.pnl_pct for t in trades) / n
        avg_hold = sum(t.hold_bars for t in trades) / n
        best = max(trades, key=lambda t: t.pnl_usd)
        worst = min(trades, key=lambda t: t.pnl_usd)
        trades_per_day = n / days if days > 0 else 0
    else:
        win_rate = profit_factor = avg_pnl_usd = avg_pnl_pct = avg_hold = trades_per_day = 0
        best = worst = None

    by_pair = _group_trades(trades, lambda t: t.symbol)
    by_exit = _group_trades(trades, lambda t: t.exit_reason)
    by_side = _group_trades(trades, lambda t: t.side)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "monthly_return": monthly_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "trades_per_day": trades_per_day,
        "n_impulses": result.n_impulses,
        "n_pullbacks": result.n_pullbacks,
        "n_entries": result.n_entries,
        "n_filtered_macro": result.n_filtered_macro,
        "n_filtered_safety": result.n_filtered_safety,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_pnl_usd": avg_pnl_usd,
        "avg_pnl_pct": avg_pnl_pct,
        "avg_hold_bars": avg_hold,
        "best_trade": best,
        "worst_trade": worst,
        "by_pair": by_pair,
        "by_exit": by_exit,
        "by_side": by_side,
        "days": days,
        "years": years,
        "final_equity": final,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sharpe(returns: list[float], periods_per_year: float = 8760) -> float:
    """Sharpe annualisé. 8760 = snapshots/h × 24h × 365."""
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mu / std) * math.sqrt(periods_per_year)


def _sortino(returns: list[float], periods_per_year: float = 8760) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 99.0
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mu / down_std) * math.sqrt(periods_per_year)


def _group_trades(trades: list[MCTrade], key_fn) -> dict[str, dict]:
    groups: dict[str, list[MCTrade]] = defaultdict(list)
    for t in trades:
        groups[key_fn(t)].append(t)
    out: dict[str, dict] = {}
    for k, tlist in sorted(groups.items()):
        n = len(tlist)
        wins = sum(1 for t in tlist if t.pnl_usd > 0)
        pnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        avg_hold = sum(t.hold_bars for t in tlist) / n if n else 0
        out[k] = {
            "n": n,
            "wins": wins,
            "wr": wins / n if n else 0,
            "pnl": pnl,
            "pf": gp / gl,
            "avg_pct": sum(t.pnl_pct for t in tlist) / n if n else 0,
            "avg_hold": avg_hold,
        }
    return out
