#!/usr/bin/env python
"""
Backtest comparatif : RANGE classique vs RANGE PRO (BB + RSI + ADX + ATR).

Stratégie PRO :
  Entrée :
    - Prix touche BB basse (close ≤ BB lower)
    - RSI(14) < 35
    - ADX(14) < 20   (confirme le ranging)
    - Bougie verte (close > open)
  SL :
    - 1.3 × ATR(14) sous le swing low du range
  TP :
    - BB mid band (SMA20)
    - Option : 0.8R partiel + trailing

  Cooldown :
    - Après 2 pertes consécutives → pause de N bougies

Comparaison A/B avec le Range classique (Dow Theory seul).
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src.core.models import (
    Candle,
    OrderSide,
    Position,
    PositionStatus,
    RangeState,
    StrategyType,
    TrendDirection,
    TrendState,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend
from src.core.strategy_mean_rev import build_range_from_trend
from src.core.risk_manager import calculate_position_size
from backtest.data_loader import download_all_pairs
from backtest.simulator import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
    EquityPoint,
)
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("range_pro")


# ════════════════════════════════════════════════════════════════════════════════
# Indicateurs techniques (calcul sur listes de Candle, sans librairie externe)
# ════════════════════════════════════════════════════════════════════════════════


def sma(values: list[float], period: int) -> Optional[float]:
    """Simple Moving Average sur les `period` dernières valeurs."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values: list[float], period: int) -> Optional[float]:
    """Exponential Moving Average."""
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    e = sum(values[:period]) / period  # SMA initiale
    for v in values[period:]:
        e = v * k + e * (1 - k)
    return e


def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index (Wilder's smoothing)."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Première moyenne
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [abs(min(d, 0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    # Wilder smoothing
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + abs(min(d, 0))) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles: list[Candle], period: int = 14) -> Optional[float]:
    """Average True Range (Wilder's smoothing)."""
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    # Première moyenne
    a = sum(trs[:period]) / period
    # Wilder smoothing
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def adx(candles: list[Candle], period: int = 14) -> Optional[float]:
    """Average Directional Index."""
    needed = 2 * period + 1  # period pour DI, period pour ADX
    if len(candles) < needed:
        return None

    # True Range + DM
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr_list: list[float] = []

    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        ph = candles[i - 1].high
        pl = candles[i - 1].low
        pc = candles[i - 1].close

        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_list.append(tr)

        up = h - ph
        down = pl - l
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)

    # Wilder smoothing (première valeur = SMA)
    def wilder_smooth(vals: list[float], p: int) -> list[float]:
        result = [sum(vals[:p]) / p]
        for v in vals[p:]:
            result.append((result[-1] * (p - 1) + v) / p)
        return result

    smooth_tr = wilder_smooth(tr_list, period)
    smooth_plus = wilder_smooth(plus_dm, period)
    smooth_minus = wilder_smooth(minus_dm, period)

    # +DI / -DI
    dx_list: list[float] = []
    for i in range(len(smooth_tr)):
        st = smooth_tr[i]
        if st == 0:
            dx_list.append(0)
            continue
        pdi = 100.0 * smooth_plus[i] / st
        mdi = 100.0 * smooth_minus[i] / st
        denom = pdi + mdi
        if denom == 0:
            dx_list.append(0)
        else:
            dx_list.append(100.0 * abs(pdi - mdi) / denom)

    # ADX = Wilder smooth du DX
    if len(dx_list) < period:
        return None
    adx_smooth = wilder_smooth(dx_list, period)
    return adx_smooth[-1] if adx_smooth else None


def bollinger_bands(
    closes: list[float], period: int = 20, std_dev: float = 2.0,
) -> Optional[tuple[float, float, float]]:
    """Retourne (upper, mid, lower)."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return (mid + std_dev * std, mid, mid - std_dev * std)


# ════════════════════════════════════════════════════════════════════════════════
# Moteur de backtest RANGE PRO
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class RangeProConfig:
    """Paramètres de la stratégie Range PRO."""
    initial_balance: float = 1000.0

    # ── Indicateurs ──
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_threshold: float = 35.0      # RSI < 35 pour entrée
    adx_period: int = 14
    adx_threshold: float = 20.0      # ADX < 20 pour confirmer range
    atr_period: int = 14
    atr_sl_mult: float = 1.3         # SL = swing_low - 1.3 × ATR

    # ── Risk ──
    risk_percent: float = 0.02       # 2% par trade
    max_position_pct: float = 0.30
    max_simultaneous_positions: int = 3
    fee_pct: float = 0.00075         # 0.075%
    slippage_pct: float = 0.001      # 0.1%

    # ── Range Dow ──
    swing_lookback: int = 3
    candle_window: int = 100
    range_width_min: float = 0.02    # largeur min 2%
    range_entry_buffer_pct: float = 0.002

    # ── TP mode ──
    tp_mode: str = "mid_band"        # "mid_band" ou "partial_trail"
    partial_r: float = 0.8           # R-multiple pour TP partiel
    trailing_activation_r: float = 1.0  # Activer trailing après 1R
    trailing_stop_pct: float = 0.015    # 1.5% trailing

    # ── Cooldown après pertes consécutives ──
    max_consecutive_losses: int = 2
    cooldown_bars_after_losses: int = 6  # 6 × 4h = 24h de pause

    # ── Confirmation bougie ──
    require_green_candle: bool = True  # close > open requis

    # ── Compound ──
    compound: bool = False


@dataclass
class ProTrade:
    """Trade terminé dans le backtest PRO."""
    symbol: str
    side: OrderSide
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    # Indicateurs au moment de l'entrée (pour analyse)
    rsi_at_entry: Optional[float] = None
    adx_at_entry: Optional[float] = None
    atr_at_entry: Optional[float] = None
    bb_lower_at_entry: Optional[float] = None


@dataclass
class ProPosition:
    symbol: str
    side: OrderSide
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    entry_time: int
    # Trailing
    is_partial_taken: bool = False
    remaining_size: float = 0.0
    trailing_sl: Optional[float] = None
    highest_since_entry: float = 0.0
    # Indicateurs snapshot
    rsi_at_entry: Optional[float] = None
    adx_at_entry: Optional[float] = None
    atr_at_entry: Optional[float] = None
    bb_lower_at_entry: Optional[float] = None


class RangeProEngine:
    """Backtest RANGE PRO : BB + RSI + ADX + confirmation + ATR SL."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: RangeProConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        # État financier
        self.cash: float = config.initial_balance
        self.positions: list[ProPosition] = []
        self.closed_trades: list[ProTrade] = []
        self.equity_curve: list[EquityPoint] = []

        # État Dow (pour construire le range)
        self.trends: dict[str, Optional[TrendState]] = {p: None for p in self.pairs}
        self.ranges: dict[str, Optional[RangeState]] = {p: None for p in self.pairs}

        # Pertes consécutives par paire → cooldown
        self._consec_losses: dict[str, int] = {p: 0 for p in self.pairs}
        self._loss_cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}

        # Signaux pending (pas de lookahead)
        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}

        # Last close
        self.last_close: dict[str, float] = {}

        # Index rapide
        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

    def run(self) -> BacktestResult:
        # Couper les logs verbeux
        for mod in (
            "src.core.swing_detector",
            "src.core.trend_engine",
            "src.core.strategy_mean_rev",
        ):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)
        logger.info(
            "📊 Range PRO : %d barres, %d paires, $%.0f",
            total, len(self.pairs), self.cfg.initial_balance,
        )

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            # Ordre anti-lookahead :
            # 1) Gérer exits (SL/TP/trailing)
            # 2) Exécuter signaux pending (de la bougie précédente)
            # 3) Analyser et générer nouveaux signaux
            self._manage_exits(ts)
            self._execute_pending(ts)
            self._analyze(ts)
            self._record_equity(ts)

            if (i + 1) % 500 == 0 or i == total - 1:
                eq = self.equity_curve[-1].equity
                print(
                    f"\r   ⏳ {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                    f"| Equity: ${eq:,.2f} | Trades: {len(self.closed_trades)}",
                    end="", flush=True,
                )
        print()

        self._close_remaining(timeline[-1])
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash

        # Convertir en format BacktestResult compatible
        trades = [
            Trade(
                symbol=t.symbol,
                strategy=StrategyType.RANGE,
                side=t.side,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                size=t.size,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                pnl_usd=t.pnl_usd,
                pnl_pct=t.pnl_pct,
                exit_reason=t.exit_reason,
            )
            for t in self.closed_trades
        ]

        return BacktestResult(
            trades=trades,
            equity_curve=self.equity_curve,
            initial_balance=self.cfg.initial_balance,
            final_equity=final_eq,
            start_date=datetime.fromtimestamp(timeline[0] / 1000, tz=timezone.utc),
            end_date=datetime.fromtimestamp(timeline[-1] / 1000, tz=timezone.utc),
            pairs=self.pairs,
        )

    def _build_timeline(self) -> list[int]:
        ts_set: set[int] = set()
        for clist in self.candles.values():
            for c in clist:
                ts_set.add(c.timestamp)
        return sorted(ts_set)

    def _visible(self, symbol: str, up_to_ts: int) -> list[Candle]:
        clist = self.candles[symbol]
        vis = [c for c in clist if c.timestamp <= up_to_ts]
        return vis[-self.cfg.candle_window:]

    # ── Gestion des exits ──────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[ProPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            # Track highest pour trailing
            if c.high > pos.highest_since_entry:
                pos.highest_since_entry = c.high

            # SL check (y compris trailing SL)
            eff_sl = pos.trailing_sl if pos.trailing_sl else pos.sl_price
            if c.low <= eff_sl:
                exit_price = eff_sl
                reason = "TRAILING_SL" if pos.trailing_sl else "PRO_SL"
                to_close.append((pos, exit_price, reason))
                continue

            # TP mode
            if self.cfg.tp_mode == "mid_band":
                # TP simple = BB mid (stocké dans tp_price)
                if c.high >= pos.tp_price:
                    to_close.append((pos, pos.tp_price, "PRO_TP"))
                    continue
            else:
                # partial_trail : TP partiel à 0.8R, puis trailing
                r_dist = pos.entry_price - pos.sl_price  # distance SL
                partial_tp = pos.entry_price + self.cfg.partial_r * r_dist

                if not pos.is_partial_taken and c.high >= partial_tp:
                    # Prendre 50% à 0.8R
                    half = pos.size * 0.5
                    self._close_partial(pos, partial_tp, ts, half, "PRO_TP_PARTIAL")
                    pos.is_partial_taken = True
                    pos.remaining_size = pos.size  # déjà réduit par close_partial
                    # Activer trailing SL au breakeven
                    pos.trailing_sl = pos.entry_price

                # Trailing actif
                if pos.is_partial_taken and pos.trailing_sl:
                    new_trail = pos.highest_since_entry * (1 - self.cfg.trailing_stop_pct)
                    if new_trail > pos.trailing_sl:
                        pos.trailing_sl = new_trail

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    # ── Exécuter les signaux pending ───────────────────────────────────────

    def _execute_pending(self, ts: int) -> None:
        if len(self.positions) >= self.cfg.max_simultaneous_positions:
            return

        for sym in self.pairs:
            if len(self.positions) >= self.cfg.max_simultaneous_positions:
                break
            if any(p.symbol == sym for p in self.positions):
                continue

            sig = self._pending.get(sym)
            if sig is None:
                continue

            c = self._idx.get((sym, ts))
            if c is None:
                continue

            # Cooldown check (pertes consécutives)
            if ts < self._loss_cooldown_until.get(sym, 0):
                continue

            # Entrer à l'open de la bougie suivante
            entry_price = c.open * (1 + self.cfg.slippage_pct)

            # Vérifier que SL < entry
            if sig["sl_price"] >= entry_price:
                continue

            # Sizing
            sizing_balance = self.cash if self.cfg.compound else self.cfg.initial_balance
            size = calculate_position_size(
                account_balance=sizing_balance,
                risk_percent=self.cfg.risk_percent,
                entry_price=entry_price,
                sl_price=sig["sl_price"],
                max_position_percent=self.cfg.max_position_pct,
            )
            if size <= 0:
                continue

            cost = size * entry_price * (1 + self.cfg.fee_pct)
            if cost > self.cash:
                size = self.cash / (entry_price * (1 + self.cfg.fee_pct))
                cost = size * entry_price * (1 + self.cfg.fee_pct)
            if size <= 0 or cost > self.cash:
                continue

            self.cash -= cost
            pos = ProPosition(
                symbol=sym,
                side=OrderSide.BUY,
                entry_price=entry_price,
                sl_price=sig["sl_price"],
                tp_price=sig["tp_price"],
                size=size,
                entry_time=ts,
                highest_since_entry=entry_price,
                rsi_at_entry=sig.get("rsi"),
                adx_at_entry=sig.get("adx"),
                atr_at_entry=sig.get("atr"),
                bb_lower_at_entry=sig.get("bb_lower"),
            )
            self.positions.append(pos)
            self._pending[sym] = None

    # ── Analyse : Dow range + indicateurs techniques ───────────────────────

    def _analyze(self, ts: int) -> None:
        for sym in self.pairs:
            vis = self._visible(sym, ts)
            if len(vis) < max(
                2 * self.cfg.swing_lookback + 1,
                self.cfg.bb_period,
                self.cfg.rsi_period + 1,
                self.cfg.atr_period + 1,
                2 * self.cfg.adx_period + 1,
            ):
                continue

            # ── 1) Dow Theory : détecter le range ──
            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue
            trend = determine_trend(swings, sym)
            self.trends[sym] = trend

            # Invalider le range si la tendance n'est plus NEUTRAL
            if trend.direction != TrendDirection.NEUTRAL:
                self.ranges[sym] = None
                self._pending[sym] = None
                # Fermer les positions range sur ce symbole (trend breakout)
                for pos in list(self.positions):
                    if pos.symbol == sym:
                        px = self.last_close.get(sym, pos.entry_price)
                        self._close_position(pos, px, ts, "TREND_BREAK")
                continue

            # Construire le range
            rs = build_range_from_trend(trend, self.cfg.range_width_min)
            if rs is None:
                self._pending[sym] = None
                continue
            self.ranges[sym] = rs

            # ── 2) Calculer les indicateurs ──
            closes = [c.close for c in vis]
            current_candle = vis[-1]

            bb = bollinger_bands(closes, self.cfg.bb_period, self.cfg.bb_std)
            rsi_val = rsi(closes, self.cfg.rsi_period)
            adx_val = adx(vis, self.cfg.adx_period)
            atr_val = atr(vis, self.cfg.atr_period)

            if bb is None or rsi_val is None or adx_val is None or atr_val is None:
                continue

            bb_upper, bb_mid, bb_lower = bb

            # ── 3) Conditions d'entrée PRO ──
            # a) Prix touche BB basse : close ≤ BB lower
            touch_bb_lower = current_candle.close <= bb_lower

            # b) RSI < seuil
            rsi_ok = rsi_val < self.cfg.rsi_threshold

            # c) ADX < seuil (confirme ranging)
            adx_ok = adx_val < self.cfg.adx_threshold

            # d) Bougie verte (confirmation)
            green = current_candle.close > current_candle.open

            if not self.cfg.require_green_candle:
                green = True  # désactiver si non requis

            if touch_bb_lower and rsi_ok and adx_ok and green:
                # SL = swing_low - 1.3 × ATR
                sl_price = rs.range_low - self.cfg.atr_sl_mult * atr_val

                # TP = BB mid band
                tp_price = bb_mid

                # Vérifier que le TP est au-dessus de l'entry potentielle
                if tp_price <= current_candle.close:
                    continue

                self._pending[sym] = {
                    "side": OrderSide.BUY,
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                    "rsi": rsi_val,
                    "adx": adx_val,
                    "atr": atr_val,
                    "bb_lower": bb_lower,
                }
            else:
                self._pending[sym] = None

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(
        self, pos: ProPosition, exit_price: float, ts: int, reason: str,
    ) -> None:
        revenue = pos.size * exit_price * (1 - self.cfg.fee_pct)
        pnl = pos.size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(ProTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rsi_at_entry=pos.rsi_at_entry,
            adx_at_entry=pos.adx_at_entry,
            atr_at_entry=pos.atr_at_entry,
            bb_lower_at_entry=pos.bb_lower_at_entry,
        ))
        if pos in self.positions:
            self.positions.remove(pos)

        # Cooldown : pertes consécutives
        sym = pos.symbol
        if pnl < 0:
            self._consec_losses[sym] = self._consec_losses.get(sym, 0) + 1
            if self._consec_losses[sym] >= self.cfg.max_consecutive_losses:
                pause_ms = self.cfg.cooldown_bars_after_losses * 4 * 3600 * 1000
                self._loss_cooldown_until[sym] = ts + pause_ms
                logger.debug(
                    "[%s] 🛑 %d pertes consécutives → cooldown %d barres",
                    sym, self._consec_losses[sym], self.cfg.cooldown_bars_after_losses,
                )
                self._consec_losses[sym] = 0
        else:
            self._consec_losses[sym] = 0

    def _close_partial(
        self, pos: ProPosition, exit_price: float, ts: int,
        partial_size: float, reason: str,
    ) -> None:
        """Ferme une partie de la position (pour TP partiel)."""
        revenue = partial_size * exit_price * (1 - self.cfg.fee_pct)
        pnl = partial_size * (
            exit_price * (1 - self.cfg.fee_pct)
            - pos.entry_price * (1 + self.cfg.fee_pct)
        )
        self.cash += revenue
        pnl_pct = pnl / (partial_size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(ProTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=partial_size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rsi_at_entry=pos.rsi_at_entry,
            adx_at_entry=pos.adx_at_entry,
            atr_at_entry=pos.atr_at_entry,
            bb_lower_at_entry=pos.bb_lower_at_entry,
        ))
        pos.size -= partial_size

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    def _record_equity(self, ts: int) -> None:
        unrealized = sum(
            p.size * self.last_close.get(p.symbol, p.entry_price)
            for p in self.positions
        )
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ════════════════════════════════════════════════════════════════════════════════
# Rapport comparatif A/B
# ════════════════════════════════════════════════════════════════════════════════


def print_comparison(
    m_classic: dict,
    m_pro: dict,
    res_classic: BacktestResult,
    res_pro: BacktestResult,
    pro_trades: list[ProTrade],
) -> None:
    """Affiche un tableau comparatif RANGE classique vs RANGE PRO."""
    sep = "═" * 80
    print(f"\n{sep}")
    print("  🔬 COMPARAISON : RANGE CLASSIQUE vs RANGE PRO (BB+RSI+ADX+ATR)")
    print(f"  📅 {res_classic.start_date:%b %Y} → {res_classic.end_date:%b %Y}")
    print(f"  🪙 {len(res_classic.pairs)} paires | Capital: ${res_classic.initial_balance:,.0f}")
    print(sep)

    metrics_list = [
        ("Return", "total_return", "{:+.1%}"),
        ("CAGR", "cagr", "{:+.1%}"),
        ("Max Drawdown", "max_drawdown", "{:.1%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Trades", "n_trades", "{:d}"),
        ("PnL moyen ($)", "avg_pnl_usd", "{:+.2f}"),
        ("PnL moyen (%)", "avg_pnl_pct", "{:+.2%}"),
        ("Capital final", "final_equity", "${:,.2f}"),
    ]

    print(f"\n  {'Métrique':<22s} │ {'CLASSIQUE':>14s} │ {'PRO':>14s} │ {'Δ':>10s}")
    print("  " + "─" * 70)

    for label, key, fmt in metrics_list:
        v_c = m_classic.get(key, 0)
        v_p = m_pro.get(key, 0)

        s_c = fmt.format(v_c) if isinstance(v_c, (int, float)) else str(v_c)
        s_p = fmt.format(v_p) if isinstance(v_p, (int, float)) else str(v_p)

        # Delta
        if isinstance(v_c, (int, float)) and isinstance(v_p, (int, float)):
            delta = v_p - v_c
            if key in ("total_return", "cagr", "max_drawdown", "win_rate", "avg_pnl_pct"):
                s_d = f"{delta:+.1%}"
            elif key == "n_trades":
                s_d = f"{delta:+d}"
            else:
                s_d = f"{delta:+.2f}"
        else:
            s_d = ""

        # Emoji
        better = ""
        if key in ("max_drawdown",):
            better = "✅" if v_p > v_c else ("❌" if v_p < v_c else "")
        elif key in ("total_return", "cagr", "sharpe", "sortino", "win_rate",
                      "profit_factor", "final_equity", "avg_pnl_usd", "avg_pnl_pct"):
            better = "✅" if v_p > v_c else ("❌" if v_p < v_c else "")

        print(f"  {label:<22s} │ {s_c:>14s} │ {s_p:>14s} │ {s_d:>8s} {better}")

    # Stats additionnelles PRO
    if pro_trades:
        print(f"\n  📊 Détail RANGE PRO")
        print("  " + "─" * 70)

        # Par motif de sortie
        exit_reasons: dict[str, list[ProTrade]] = {}
        for t in pro_trades:
            exit_reasons.setdefault(t.exit_reason, []).append(t)

        for reason, tlist in sorted(exit_reasons.items()):
            n = len(tlist)
            wins = sum(1 for t in tlist if t.pnl_usd > 0)
            pnl = sum(t.pnl_usd for t in tlist)
            wr = wins / n if n else 0
            print(f"  {reason:16s} : {n:3d} trades | WR {wr:.0%} | PnL ${pnl:+.2f}")

        # Indicateurs moyens à l'entrée
        rsi_vals = [t.rsi_at_entry for t in pro_trades if t.rsi_at_entry]
        adx_vals = [t.adx_at_entry for t in pro_trades if t.adx_at_entry]
        atr_vals = [t.atr_at_entry for t in pro_trades if t.atr_at_entry]
        if rsi_vals:
            print(f"\n  RSI moyen à l'entrée  : {sum(rsi_vals)/len(rsi_vals):.1f}")
        if adx_vals:
            print(f"  ADX moyen à l'entrée  : {sum(adx_vals)/len(adx_vals):.1f}")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════


PAIRS_20 = [
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LINK-USD",
    "ADA-USD", "DOT-USD", "AVAX-USD", "DOGE-USD", "ATOM-USD",
    "UNI-USD", "NEAR-USD", "ALGO-USD", "LTC-USD", "ETC-USD",
    "FIL-USD", "AAVE-USD", "INJ-USD", "SAND-USD", "MANA-USD",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TradeX Backtest — RANGE classique vs RANGE PRO (BB+RSI+ADX+ATR)"
    )
    parser.add_argument("--balance", type=float, default=1000.0)
    parser.add_argument("--start", default="2022-02-20")
    parser.add_argument("--end", default="2026-02-20")
    parser.add_argument("--pairs", type=int, default=20, help="Nombre de paires (5 ou 20)")
    # Paramètres PRO ajustables
    parser.add_argument("--rsi-threshold", type=float, default=35.0)
    parser.add_argument("--adx-threshold", type=float, default=20.0)
    parser.add_argument("--atr-sl-mult", type=float, default=1.3)
    parser.add_argument("--tp-mode", choices=["mid_band", "partial_trail"], default="mid_band")
    parser.add_argument("--max-consec-losses", type=int, default=2)
    parser.add_argument("--cooldown-bars", type=int, default=6)
    parser.add_argument("--no-green", action="store_true", help="Désactiver filtre bougie verte")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═'*80}")
    print(f"  🔬 RANGE CLASSIQUE vs RANGE PRO — A/B Test")
    print(f"  📅 {start:%Y-%m-%d} → {end:%Y-%m-%d} | {len(pairs)} paires | ${args.balance:,.0f}")
    print(f"  PRO: RSI<{args.rsi_threshold} | ADX<{args.adx_threshold} | "
          f"ATR×{args.atr_sl_mult} SL | TP={args.tp_mode} | "
          f"Cooldown: {args.max_consec_losses} pertes → {args.cooldown_bars} barres"
          f"{' | Bougie verte' if not args.no_green else ''}")
    print(f"{'═'*80}\n")

    # ── Télécharger les données ──
    logger.info("📥 Téléchargement des données…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # ═══════════════════════════════════════════════════════════════════════
    # A) RANGE CLASSIQUE (Dow Theory seul)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  🅰️  RANGE CLASSIQUE (Dow Theory)")
    print("─" * 60)

    from src import config
    cfg_classic = BacktestConfig(
        initial_balance=args.balance,
        risk_percent_range=0.02,
        entry_buffer_pct=getattr(config, "ENTRY_BUFFER_PERCENT", 0.002),
        sl_buffer_pct=getattr(config, "SL_BUFFER_PERCENT", 0.003),
        zero_risk_trigger_pct=getattr(config, "ZERO_RISK_TRIGGER_PERCENT", 0.02),
        zero_risk_lock_pct=getattr(config, "ZERO_RISK_LOCK_PERCENT", 0.005),
        trailing_stop_pct=getattr(config, "TRAILING_STOP_PERCENT", 0.02),
        max_position_pct=0.30,
        max_simultaneous_positions=3,
        swing_lookback=getattr(config, "SWING_LOOKBACK", 3),
        range_width_min=getattr(config, "RANGE_WIDTH_MIN", 0.02),
        range_entry_buffer_pct=getattr(config, "RANGE_ENTRY_BUFFER_PERCENT", 0.002),
        range_sl_buffer_pct=getattr(config, "RANGE_SL_BUFFER_PERCENT", 0.003),
        range_cooldown_bars=getattr(config, "RANGE_COOLDOWN_BARS", 3),
        enable_trend=False,
        enable_range=True,
    )
    engine_classic = BacktestEngine(candles, cfg_classic)
    result_classic = engine_classic.run()
    m_classic = compute_metrics(result_classic)

    # ═══════════════════════════════════════════════════════════════════════
    # B) RANGE PRO (BB + RSI + ADX + ATR + cooldown)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("  🅱️  RANGE PRO (BB + RSI + ADX + ATR)")
    print("─" * 60)

    cfg_pro = RangeProConfig(
        initial_balance=args.balance,
        rsi_threshold=args.rsi_threshold,
        adx_threshold=args.adx_threshold,
        atr_sl_mult=args.atr_sl_mult,
        tp_mode=args.tp_mode,
        max_consecutive_losses=args.max_consec_losses,
        cooldown_bars_after_losses=args.cooldown_bars,
        require_green_candle=not args.no_green,
    )
    engine_pro = RangeProEngine(candles, cfg_pro)
    result_pro = engine_pro.run()
    m_pro = compute_metrics(result_pro)

    # ═══════════════════════════════════════════════════════════════════════
    # Comparaison A/B
    # ═══════════════════════════════════════════════════════════════════════
    print_comparison(m_classic, m_pro, result_classic, result_pro, engine_pro.closed_trades)

    # ── Graphique ──
    from backtest.report import generate_report
    print("  📊 Graphique CLASSIQUE :")
    generate_report(result_classic, m_classic, show=False)
    print("  📊 Graphique PRO :")
    generate_report(result_pro, m_pro, show=not args.no_show)


if __name__ == "__main__":
    main()
