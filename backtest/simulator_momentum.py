"""
Moteur de backtest pour la stratégie Momentum Squeeze.

Logique :
  - Timeframe 15m (scalable à 5m/1h).
  - Détecte les phases de COMPRESSION de volatilité (Bollinger Bands squeeze)
    suivies d'un BREAKOUT directionnel confirmé par le momentum (MACD + volume).
  - Entrée LONG ou SHORT selon la direction du breakout.
  - TP = trailing stop dynamique (ATR-based).
  - SL = opposé de la Bollinger Band au moment du breakout.
  - Timeout : N barres max.

Décorrélé des deux autres stratégies :
  - Trail Range (H4) : structure Dow / breakout macro → jours/semaines
  - CrashBot (1m)  : event-driven sur crashs extrêmes → minutes
  - Momentum Squeeze (15m) : compression→expansion de volatilité → heures

Le Squeeze ne dépend pas de la direction de la tendance (≠ Trail Range),
ni d'un crash extrême (≠ CrashBot). Il exploite un régime de marché
(basse vol → haute vol) qui est orthogonal aux deux autres signaux.
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
class MomentumConfig:
    """Paramètres de la stratégie Momentum Squeeze."""

    initial_balance: float = 1000.0
    risk_percent: float = 0.02          # 2% risque par trade

    # ── Squeeze detection ──
    bb_period: int = 20                 # Période Bollinger Bands
    bb_mult: float = 2.0               # Multiplicateur BB (σ)
    kc_period: int = 20                # Période Keltner Channel
    kc_mult: float = 1.5              # Multiplicateur KC (ATR)
    squeeze_min_bars: int = 6          # Min bougies consécutives en squeeze

    # ── Breakout confirmation ──
    macd_fast: int = 12                # MACD fast EMA
    macd_slow: int = 26                # MACD slow EMA
    macd_signal: int = 9               # MACD signal line
    volume_confirm_mult: float = 1.3   # Volume breakout > 1.3× moyenne
    vol_avg_window: int = 20           # Fenêtre moyenne volume

    # ── Direction filter ──
    ema_trend_period: int = 50         # EMA filtre tendance (0=désactivé)
    require_trend_align: bool = False  # Filtrer les trades contre la tendance EMA50

    # ── Trade management ──
    atr_period: int = 14               # Période ATR pour SL/TP
    sl_atr_mult: float = 1.5          # SL = entry ∓ 1.5×ATR
    tp_atr_mult: float = 3.0          # TP initial = entry ± 3×ATR
    use_trailing: bool = True          # Trailing stop
    trail_activation_atr: float = 1.5  # Activer trailing après 1.5×ATR de profit
    trail_distance_atr: float = 1.0    # Distance du trailing = 1×ATR

    # ── Timeout / cooldown ──
    timeout_bars: int = 48             # Max 48 barres (= 12h en 15m)
    cooldown_bars: int = 8             # Cooldown par paire

    # ── Contraintes ──
    max_positions: int = 4             # Max positions simultanées
    max_position_pct: float = 0.30     # Max 30% du capital par position

    # ── Frais ──
    fee_pct: float = 0.001             # 0.1% taker fee
    slippage_pct: float = 0.0005       # 0.05% slippage

    # ── Mode ──
    allow_short: bool = True           # Autoriser les shorts


# ── Structures ─────────────────────────────────────────────────────────────────


@dataclass
class MomentumTrade:
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
    exit_reason: str            # TP, SL, TRAILING, TIMEOUT, END
    squeeze_bars: int           # Nombre de barres en squeeze avant breakout
    atr_at_entry: float         # ATR au moment de l'entrée
    bb_width_at_entry: float    # Largeur BB normalisée au breakout
    hold_bars: int


@dataclass
class MomentumPosition:
    """Position ouverte."""

    symbol: str
    side: str                   # "LONG" ou "SHORT"
    entry_price: float
    tp_price: float
    sl_price: float             # SL fixe initial
    trailing_sl: float          # SL trailing (mis à jour)
    trailing_active: bool
    size: float
    cost: float
    entry_time: int
    entry_bar_idx: int
    squeeze_bars: int
    atr_at_entry: float
    bb_width_at_entry: float
    peak_price: float           # Meilleur prix atteint (pour trailing)


@dataclass
class MomentumResult:
    """Résultat complet du backtest."""

    trades: list[MomentumTrade]
    equity_curve: list[tuple[int, float]]   # (timestamp_ms, equity)
    initial_balance: float
    final_equity: float
    n_signals: int
    n_traded: int
    n_squeezes: int             # Total squeezes détectés (même sans breakout)
    config: MomentumConfig
    start_date: datetime
    end_date: datetime
    pairs: list[str]


# ── Indicateurs techniques ─────────────────────────────────────────────────────


def _ema(values: list[float], period: int) -> list[float]:
    """EMA complète sur toute la série. Retourne une liste de même longueur."""
    if not values:
        return []
    result = [values[0]]
    k = 2.0 / (period + 1)
    for i in range(1, len(values)):
        result.append(values[i] * k + result[-1] * (1 - k))
    return result


def _sma(values: list[float], period: int) -> float:
    """SMA sur les `period` dernières valeurs."""
    if len(values) < period:
        return sum(values) / len(values) if values else 0
    return sum(values[-period:]) / period


def _std(values: list[float], period: int) -> float:
    """Écart-type (population) sur les `period` dernières valeurs."""
    if len(values) < period:
        data = values
    else:
        data = values[-period:]
    if len(data) < 2:
        return 0.0
    mu = sum(data) / len(data)
    return math.sqrt(sum((x - mu) ** 2 for x in data) / len(data))


def _compute_atr(candles: list[Candle], period: int = 14) -> float:
    """Average True Range sur les `period` dernières bougies."""
    if len(candles) < period + 1:
        return 0.0
    trs: list[float] = []
    for i in range(len(candles) - period, len(candles)):
        c = candles[i]
        prev = candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
        trs.append(tr)
    return sum(trs) / len(trs)


def _compute_atr_series(candles: list[Candle], period: int = 14) -> list[float]:
    """ATR EMA-lissé sur toute la série. Retourne list de même longueur."""
    n = len(candles)
    if n < 2:
        return [0.0] * n
    trs: list[float] = []
    trs.append(candles[0].high - candles[0].low)
    for i in range(1, n):
        c = candles[i]
        prev = candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - prev.close), abs(c.low - prev.close))
        trs.append(tr)
    # EMA smooth
    atr_vals = _ema(trs, period)
    return atr_vals


def _compute_bb(closes: list[float], period: int, mult: float) -> tuple[float, float, float, float]:
    """Bollinger Bands : (upper, middle, lower, width_normalized).

    width_normalized = (upper - lower) / middle — mesure la "largeur" relative.
    """
    if len(closes) < period:
        mid = closes[-1] if closes else 0
        return mid, mid, mid, 0.0
    mid = _sma(closes, period)
    sd = _std(closes, period)
    upper = mid + mult * sd
    lower = mid - mult * sd
    width = (upper - lower) / mid if mid > 0 else 0.0
    return upper, mid, lower, width


def _compute_kc(candles: list[Candle], period: int, mult: float) -> tuple[float, float, float]:
    """Keltner Channel : (upper, middle, lower).

    middle = EMA(close, period)
    upper/lower = middle ± mult × ATR(period)
    """
    if len(candles) < period + 1:
        p = candles[-1].close if candles else 0
        return p, p, p
    closes = [c.close for c in candles]
    ema_vals = _ema(closes, period)
    mid = ema_vals[-1]
    atr = _compute_atr(candles, period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return upper, mid, lower


def _compute_macd(
    closes: list[float], fast: int, slow: int, signal: int
) -> tuple[float, float, float]:
    """MACD : (macd_line, signal_line, histogram).

    Retourne les valeurs les plus récentes.
    """
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_ema = _ema(macd_line, signal)
    m = macd_line[-1]
    s = signal_ema[-1]
    return m, s, m - s


# ── Fonctions full-series (O(n)) ───────────────────────────────────────────────


def _compute_bb_series(
    closes: list[float], period: int, mult: float,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Bollinger Bands full-series : (upper[], mid[], lower[], width[])."""
    n = len(closes)
    bb_upper = [0.0] * n
    bb_mid = [0.0] * n
    bb_lower = [0.0] * n
    bb_width = [0.0] * n
    for i in range(n):
        if i < period - 1:
            bb_upper[i] = bb_mid[i] = bb_lower[i] = closes[i]
            continue
        window = closes[i - period + 1: i + 1]
        mid = sum(window) / period
        sd = math.sqrt(sum((x - mid) ** 2 for x in window) / period)
        bb_upper[i] = mid + mult * sd
        bb_mid[i] = mid
        bb_lower[i] = mid - mult * sd
        bb_width[i] = (bb_upper[i] - bb_lower[i]) / mid if mid > 0 else 0.0
    return bb_upper, bb_mid, bb_lower, bb_width


def _compute_kc_series(
    candles: list[Candle], period: int, mult: float,
) -> tuple[list[float], list[float]]:
    """Keltner Channel full-series : (upper[], lower[])."""
    closes = [c.close for c in candles]
    ema_vals = _ema(closes, period)
    atr_vals = _compute_atr_series(candles, period)
    n = len(candles)
    kc_upper = [ema_vals[i] + mult * atr_vals[i] for i in range(n)]
    kc_lower = [ema_vals[i] - mult * atr_vals[i] for i in range(n)]
    return kc_upper, kc_lower


def _compute_macd_series(
    closes: list[float], fast: int, slow: int, signal: int,
) -> tuple[list[float], list[float], list[float]]:
    """MACD full-series : (macd_line[], signal_line[], histogram[])."""
    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_ema = _ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, signal_ema)]
    return macd_line, signal_ema, histogram


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Moyenne glissante (lookback excluant la barre courante).

    result[i] = mean(values[max(0, i-window) : i])   (exclut i).
    O(n) via fenêtre glissante.
    """
    n = len(values)
    result = [0.0] * n
    window_sum = 0.0
    for i in range(n):
        if i == 0:
            result[i] = 0.0
            continue
        # Ajouter l'élément précédent
        window_sum += values[i - 1]
        # Retirer l'élément qui sort de la fenêtre
        if i - 1 >= window:
            window_sum -= values[i - 1 - window]
        count = min(i, window)
        result[i] = window_sum / count if count > 0 else 0.0
    return result


# ── Moteur ─────────────────────────────────────────────────────────────────────


class MomentumEngine:
    """Simule la stratégie Momentum Squeeze bar par bar."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: MomentumConfig,
        interval: str = "15m",
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.interval = interval

        self.cash = config.initial_balance
        self.positions: dict[str, MomentumPosition] = {}
        self.trades: list[MomentumTrade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.cooldowns: dict[str, int] = {}  # symbol → bar_idx cooldown_until
        self.n_signals = 0
        self.n_traded = 0
        self.n_squeezes = 0

        # Squeeze state per symbol : nombre de barres consécutives en squeeze
        self._squeeze_count: dict[str, int] = defaultdict(int)
        self._was_in_squeeze: dict[str, bool] = defaultdict(bool)

        # Indicateurs pré-calculés (remplis par _precompute_indicators)
        self._ind: dict[str, dict] = {}

    def _precompute_indicators(self) -> None:
        """Pré-calcule tous les indicateurs une seule fois par paire (O(n))."""
        for sym, bars in self.candles.items():
            n = len(bars)
            closes = [b.close for b in bars]
            volumes = [b.volume for b in bars]

            # Bollinger Bands
            bb_up, bb_mid, bb_lo, bb_w = _compute_bb_series(
                closes, self.cfg.bb_period, self.cfg.bb_mult,
            )
            # Keltner Channel
            kc_up, kc_lo = _compute_kc_series(
                bars, self.cfg.kc_period, self.cfg.kc_mult,
            )
            # Squeeze flag
            is_squeeze = [
                bb_lo[i] > kc_lo[i] and bb_up[i] < kc_up[i]
                for i in range(n)
            ]
            # MACD
            macd_line, signal_line, histogram = _compute_macd_series(
                closes, self.cfg.macd_fast, self.cfg.macd_slow, self.cfg.macd_signal,
            )
            # ATR
            atr_series = _compute_atr_series(bars, self.cfg.atr_period)
            # Volume rolling average (exclut barre courante)
            vol_avg = _rolling_mean(volumes, self.cfg.vol_avg_window)
            # EMA tendance
            ema_trend = (
                _ema(closes, self.cfg.ema_trend_period)
                if self.cfg.ema_trend_period > 0
                else [0.0] * n
            )

            self._ind[sym] = {
                "closes": closes,
                "volumes": volumes,
                "bb_upper": bb_up,
                "bb_lower": bb_lo,
                "bb_width": bb_w,
                "kc_upper": kc_up,
                "kc_lower": kc_lo,
                "is_squeeze": is_squeeze,
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "atr": atr_series,
                "vol_avg": vol_avg,
                "ema_trend": ema_trend,
            }

        logger.info("📊 Indicateurs pré-calculés pour %d paires", len(self._ind))

    def run(self) -> MomentumResult:
        symbols = sorted(self.candles.keys())
        if not symbols:
            raise ValueError("Aucune donnée fournie")

        min_len = min(len(self.candles[s]) for s in symbols)
        warmup = max(
            self.cfg.bb_period + 5,
            self.cfg.kc_period + 5,
            self.cfg.macd_slow + self.cfg.macd_signal + 5,
            self.cfg.atr_period + 5,
            self.cfg.ema_trend_period + 5 if self.cfg.ema_trend_period > 0 else 0,
            self.cfg.vol_avg_window + 5,
        )

        if min_len <= warmup:
            raise ValueError(f"Pas assez de bougies ({min_len}), besoin d'au moins {warmup}")

        logger.info(
            "🟡 Momentum Squeeze — %d paires, %d bougies %s, capital $%.0f",
            len(symbols), min_len, self.interval, self.cfg.initial_balance,
        )

        # Pré-calcul O(n) de tous les indicateurs
        self._precompute_indicators()

        for i in range(warmup, min_len):
            ts = self.candles[symbols[0]][i].timestamp

            # 1. Check exits
            for sym in list(self.positions.keys()):
                if sym in self.candles and i < len(self.candles[sym]):
                    self._check_exit(sym, i)

            # 2. Check entries
            if len(self.positions) < self.cfg.max_positions:
                for sym in symbols:
                    if sym in self.positions:
                        continue
                    if len(self.positions) >= self.cfg.max_positions:
                        break
                    if i >= len(self.candles[sym]):
                        continue
                    if sym in self.cooldowns and i < self.cooldowns[sym]:
                        continue
                    self._check_entry(sym, i)

            # 3. Equity snapshot (toutes les 4 barres = 1h en 15m)
            if i % 4 == 0:
                eq = self._compute_equity(i, symbols)
                self.equity_curve.append((ts, eq))

        # Close remaining
        for sym in list(self.positions.keys()):
            last_idx = len(self.candles[sym]) - 1
            self._close_position(sym, self.candles[sym][last_idx].close, last_idx, "END")

        final_eq = self.cash
        first_ts = self.candles[symbols[0]][warmup].timestamp
        last_ts = self.candles[symbols[0]][min_len - 1].timestamp

        if self.equity_curve:
            self.equity_curve.append((last_ts, final_eq))

        return MomentumResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            n_signals=self.n_signals,
            n_traded=self.n_traded,
            n_squeezes=self.n_squeezes,
            config=self.cfg,
            start_date=datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc),
            pairs=list(self.candles.keys()),
        )

    # ── Squeeze detection + entry ──────────────────────────────────────────

    def _check_entry(self, symbol: str, idx: int) -> None:
        ind = self._ind[symbol]
        candle = self.candles[symbol][idx]
        price = ind["closes"][idx]

        # ── Squeeze : BB entièrement dans KC (lookup O(1)) ──
        is_squeeze = ind["is_squeeze"][idx]

        if is_squeeze:
            self._squeeze_count[symbol] += 1
            self._was_in_squeeze[symbol] = True
            return  # Pendant le squeeze, on ne trade pas (on attend le breakout)

        # ── Breakout : sortie du squeeze ──
        if not self._was_in_squeeze.get(symbol, False):
            self._squeeze_count[symbol] = 0
            return  # Pas de squeeze récent → rien à faire

        squeeze_bars = self._squeeze_count[symbol]
        self._squeeze_count[symbol] = 0
        self._was_in_squeeze[symbol] = False

        if squeeze_bars < self.cfg.squeeze_min_bars:
            return  # Squeeze trop court → signal faible

        self.n_squeezes += 1
        bb_width = ind["bb_width"][idx]

        # ── Direction du breakout : MACD histogram (lookup O(1)) ──
        histogram = ind["histogram"][idx]
        macd_val = ind["macd_line"][idx]
        signal_val = ind["signal_line"][idx]

        if histogram == 0:
            return

        # Histogramme croissant = haussier, décroissant = baissier
        if histogram > 0 and macd_val > signal_val:
            side = "LONG"
        elif histogram < 0 and macd_val < signal_val:
            side = "SHORT"
        else:
            return  # Signal ambigu

        if side == "SHORT" and not self.cfg.allow_short:
            return

        # ── Filtre volume (lookup O(1)) ──
        if self.cfg.volume_confirm_mult > 0:
            avg_vol = ind["vol_avg"][idx]
            if avg_vol > 0 and ind["volumes"][idx] < avg_vol * self.cfg.volume_confirm_mult:
                return  # Volume insuffisant pour confirmer le breakout

        # ── Filtre tendance EMA (lookup O(1)) ──
        if self.cfg.require_trend_align and self.cfg.ema_trend_period > 0:
            ema_val = ind["ema_trend"][idx]
            if side == "LONG" and price < ema_val:
                return  # Long contre la tendance
            if side == "SHORT" and price > ema_val:
                return  # Short contre la tendance

        self.n_signals += 1

        # ── ATR pour SL/TP (lookup O(1)) ──
        atr = ind["atr"][idx]
        if atr <= 0:
            return

        # ── SL / TP ──
        if side == "LONG":
            sl_price = price - self.cfg.sl_atr_mult * atr
            tp_price = price + self.cfg.tp_atr_mult * atr
        else:
            sl_price = price + self.cfg.sl_atr_mult * atr
            tp_price = price - self.cfg.tp_atr_mult * atr

        sl_distance = abs(price - sl_price)
        if sl_distance <= 0:
            return

        # ── Position sizing ──
        risk_amount = self.cash * self.cfg.risk_percent
        size = risk_amount / sl_distance
        cost = size * price

        max_cost = self.cash * self.cfg.max_position_pct
        if cost > max_cost:
            size = max_cost / price
            cost = size * price

        if cost > self.cash * 0.95:
            return

        # ── Entry ──
        slip = self.cfg.slippage_pct
        if side == "LONG":
            entry_price = price * (1 + slip)
        else:
            entry_price = price * (1 - slip)

        fee = cost * self.cfg.fee_pct
        self.cash -= (cost + fee)
        self.n_traded += 1

        self.positions[symbol] = MomentumPosition(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            trailing_sl=sl_price,
            trailing_active=False,
            size=size,
            cost=cost,
            entry_time=candle.timestamp,
            entry_bar_idx=idx,
            squeeze_bars=squeeze_bars,
            atr_at_entry=atr,
            bb_width_at_entry=bb_width,
            peak_price=entry_price,
        )

        logger.debug(
            "🟡 ENTRY %s %s @ %.6f | ATR=%.6f | Squeeze=%d bars | BB width=%.4f",
            side, symbol, entry_price, atr, squeeze_bars, bb_width,
        )

    # ── Exit logic ─────────────────────────────────────────────────────────

    def _check_exit(self, symbol: str, idx: int) -> None:
        pos = self.positions[symbol]
        candle = self.candles[symbol][idx]

        if pos.side == "LONG":
            # Update peak
            if candle.high > pos.peak_price:
                pos.peak_price = candle.high

            # SL hit
            if candle.low <= pos.trailing_sl:
                reason = "TRAILING" if pos.trailing_active else "SL"
                self._close_position(symbol, pos.trailing_sl, idx, reason)
                return

            # TP hit (si pas de trailing, ou TP fixe)
            if not self.cfg.use_trailing and candle.high >= pos.tp_price:
                self._close_position(symbol, pos.tp_price, idx, "TP")
                return

            # Trailing activation + update
            if self.cfg.use_trailing:
                profit_distance = pos.peak_price - pos.entry_price
                activation_threshold = self.cfg.trail_activation_atr * pos.atr_at_entry

                if profit_distance >= activation_threshold:
                    pos.trailing_active = True
                    new_sl = pos.peak_price - self.cfg.trail_distance_atr * pos.atr_at_entry
                    if new_sl > pos.trailing_sl:
                        pos.trailing_sl = new_sl

        else:  # SHORT
            # Update peak (lowest)
            if candle.low < pos.peak_price:
                pos.peak_price = candle.low

            # SL hit
            if candle.high >= pos.trailing_sl:
                reason = "TRAILING" if pos.trailing_active else "SL"
                self._close_position(symbol, pos.trailing_sl, idx, reason)
                return

            # TP hit
            if not self.cfg.use_trailing and candle.low <= pos.tp_price:
                self._close_position(symbol, pos.tp_price, idx, "TP")
                return

            # Trailing
            if self.cfg.use_trailing:
                profit_distance = pos.entry_price - pos.peak_price
                activation_threshold = self.cfg.trail_activation_atr * pos.atr_at_entry

                if profit_distance >= activation_threshold:
                    pos.trailing_active = True
                    new_sl = pos.peak_price + self.cfg.trail_distance_atr * pos.atr_at_entry
                    if new_sl < pos.trailing_sl:
                        pos.trailing_sl = new_sl

        # Timeout
        bars_held = idx - pos.entry_bar_idx
        if bars_held >= self.cfg.timeout_bars:
            self._close_position(symbol, candle.close, idx, "TIMEOUT")

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(self, symbol: str, exit_price: float, idx: int, reason: str) -> None:
        pos = self.positions.pop(symbol)

        # Slippage on unfavorable exits
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
            pnl_usd = proceeds - pos.cost
        else:
            # Short PnL : (entry - exit) × size - fees
            proceeds = pos.size * (2 * pos.entry_price - actual_exit) - fee
            pnl_usd = pos.size * (pos.entry_price - actual_exit) - fee - (pos.cost * self.cfg.fee_pct)

        self.cash += pos.cost + pnl_usd  # Return cost + pnl
        pnl_pct = pnl_usd / pos.cost if pos.cost > 0 else 0
        bars_held = idx - pos.entry_bar_idx

        self.trades.append(MomentumTrade(
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
            squeeze_bars=pos.squeeze_bars,
            atr_at_entry=pos.atr_at_entry,
            bb_width_at_entry=pos.bb_width_at_entry,
            hold_bars=bars_held,
        ))

        self.cooldowns[symbol] = idx + self.cfg.cooldown_bars

        emoji = "✅" if pnl_usd >= 0 else "❌"
        logger.debug(
            "%s CLOSE %s %s @ %.6f (%s) | PnL $%.2f (%.2f%%) | %d bars",
            emoji, pos.side, symbol, actual_exit, reason,
            pnl_usd, pnl_pct * 100, bars_held,
        )

    # ── Equity ─────────────────────────────────────────────────────────────

    def _compute_equity(self, idx: int, symbols: list[str]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            if idx < len(self.candles[sym]):
                current_price = self.candles[sym][idx].close
                if pos.side == "LONG":
                    eq += pos.size * current_price
                else:
                    # Short : profit = (entry - current) × size
                    unrealized = pos.size * (pos.entry_price - current_price)
                    eq += pos.cost + unrealized
        return eq


# ── Métriques ──────────────────────────────────────────────────────────────────


def compute_momentum_metrics(result: MomentumResult) -> dict:
    """Calcule les KPIs du backtest Momentum Squeeze."""
    trades = result.trades
    eq = result.equity_curve
    init = result.initial_balance
    final = result.final_equity
    days = max((result.end_date - result.start_date).days, 1)
    years = days / 365.25

    total_return = (final - init) / init if init > 0 else 0
    cagr = (final / init) ** (1 / years) - 1 if final > 0 and years > 0 else 0

    # Drawdown
    peak = init
    max_dd = 0.0
    dd_curve: list[float] = []
    for _, equity in eq:
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0
        dd_curve.append(dd)
        max_dd = min(max_dd, dd)

    # Returns for Sharpe/Sortino
    returns: list[float] = []
    for i in range(1, len(eq)):
        prev_eq = eq[i - 1][1]
        if prev_eq > 0:
            returns.append((eq[i][1] - prev_eq) / prev_eq)

    sharpe = _sharpe(returns)
    sortino = _sortino(returns)

    # Trades
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
    else:
        win_rate = profit_factor = avg_pnl_usd = avg_pnl_pct = avg_hold = 0
        best = worst = None

    by_pair = _group_trades(trades, lambda t: t.symbol)
    by_exit = _group_trades(trades, lambda t: t.exit_reason)
    by_side = _group_trades(trades, lambda t: t.side)

    # Par tranche de squeeze duration
    def squeeze_bucket(t: MomentumTrade) -> str:
        if t.squeeze_bars < 8:
            return "6-7 bars"
        elif t.squeeze_bars < 12:
            return "8-11 bars"
        elif t.squeeze_bars < 20:
            return "12-19 bars"
        else:
            return "20+ bars"

    by_squeeze = _group_trades(trades, squeeze_bucket)

    # Par tranche de BB width
    def bbw_bucket(t: MomentumTrade) -> str:
        w = t.bb_width_at_entry * 100  # en %
        if w < 2:
            return "BB <2%"
        elif w < 4:
            return "BB 2-4%"
        elif w < 6:
            return "BB 4-6%"
        else:
            return "BB >6%"

    by_bbwidth = _group_trades(trades, bbw_bucket)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "n_trades": n,
        "n_signals": result.n_signals,
        "n_squeezes": result.n_squeezes,
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
        "by_squeeze": by_squeeze,
        "by_bbwidth": by_bbwidth,
        "dd_curve": dd_curve,
        "years": years,
        "days": days,
        "final_equity": final,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────


def _sharpe(returns: list[float], periods_per_year: float = 5840) -> float:
    """Sharpe annualisé. 5840 ≈ 4 snapshots/h × 24h × 365/6."""
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 1e-9
    return (mu / std) * math.sqrt(periods_per_year)


def _sortino(returns: list[float], periods_per_year: float = 5840) -> float:
    if len(returns) < 2:
        return 0.0
    mu = sum(returns) / len(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return 99.0
    down_var = sum(r ** 2 for r in neg) / len(neg)
    down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
    return (mu / down_std) * math.sqrt(periods_per_year)


def _group_trades(trades: list[MomentumTrade], key_fn) -> dict[str, dict]:
    groups: dict[str, list[MomentumTrade]] = defaultdict(list)
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
