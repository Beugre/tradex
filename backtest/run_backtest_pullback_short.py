#!/usr/bin/env python
"""
Backtest Pullback Short Bot — Bear Continuation Strategy.

🧠 Logique :
   1) Structure Dow H4 = BEARISH (LH + LL confirmés)
   2) Prix pull-back vers EMA20 ou BB mid (zone de rejet)
   3) RSI remonte en zone 50–65 (pas suracheté, juste un rebond faible)
   4) Volume faible sur le rebond (< moyenne 20 périodes)
   5) Bougie de rejet (red candle / pin bar haut)
   → SHORT ENTRY

   SL : au-dessus du dernier LH + 1.3× ATR
   TP : dernier LL (swing low)
   Trail : si prix casse le LL → trailing sur nouveaux lows

Pas un breakout short (trop tardif).
On short les rebonds faibles dans une tendance baissière confirmée.

Usage :
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_pullback_short.py --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_pullback_short.py --months 24
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_pullback_short.py --rsi-min 50 --rsi-max 65
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from src.core.models import (
    Candle,
    OrderSide,
    SwingLevel,
    SwingType,
    StrategyType,
    TrendDirection,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend
from backtest.data_loader import download_all_pairs, download_btc_d1
from backtest.simulator import BacktestResult, Trade, EquityPoint
from backtest.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pullback_short")

OUTPUT_DIR = Path(__file__).parent / "output"


# ════════════════════════════════════════════════════════════════════════════════
# Indicateurs techniques
# ════════════════════════════════════════════════════════════════════════════════


def ema(values: list[float], period: int) -> list[float]:
    """Calcule l'EMA complète sur une série."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    result = [sum(values[:period]) / period]  # SMA comme seed
    for v in values[period:]:
        result.append(v * k + result[-1] * (1 - k))
    return result


def sma_single(values: list[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def rsi(closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [abs(min(d, 0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + abs(min(d, 0))) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles: list[Candle], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs: list[float] = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i].high, candles[i].low, candles[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    a = sum(trs[:period]) / period
    for tr in trs[period:]:
        a = (a * (period - 1) + tr) / period
    return a


def bollinger_mid(closes: list[float], period: int = 20) -> Optional[float]:
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def volume_ratio(volumes: list[float], period: int = 20) -> Optional[float]:
    """Ratio du dernier volume vs SMA volume. < 1 = volume faible."""
    if len(volumes) < period:
        return None
    avg = sum(volumes[-period:]) / period
    if avg <= 0:
        return None
    return volumes[-1] / avg


# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class PullbackShortConfig:
    """Paramètres du Pullback Short Bot."""
    initial_balance: float = 1000.0

    # ── Structure Dow ──
    swing_lookback: int = 3
    candle_window: int = 100

    # ── Zone de pullback ──
    ema_period: int = 20               # EMA20 — zone de rejet
    bb_period: int = 20                # BB mid = SMA20
    pullback_tolerance: float = 0.005  # ±0.5% autour de EMA/BB mid

    # ── Filtres d'entrée ──
    rsi_period: int = 14
    rsi_min: float = 50.0              # RSI doit être ≥ 50 (rebond)
    rsi_max: float = 65.0              # RSI doit être ≤ 65 (pas suracheté)
    vol_period: int = 20
    vol_max_ratio: float = 1.0         # Volume doit être < moyenne (rebond faible)
    require_rejection: bool = True     # Bougie de rejet (red candle, upper wick)

    # ── Gestion de position ──
    atr_period: int = 14
    sl_atr_mult: float = 1.3           # SL = LH + 1.3 × ATR
    sl_above_lh: bool = True           # SL au-dessus du LH (priorité)
    tp_at_ll: bool = True              # TP = dernier LL
    trail_after_ll: bool = True        # Trail si prix casse le LL
    trail_atr_mult: float = 2.0        # Trailing distance = 2 × ATR

    # ── Risk ──
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous_positions: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001
    margin_fee_pct: float = 0.0002     # ~0.02% par 4h

    # ── Cooldown ──
    max_consecutive_losses: int = 2
    cooldown_bars_after_losses: int = 6

    # ── Compound ──
    compound: bool = False

    # ── Filtre macro BTC EMA200 D1 ──
    use_macro_filter: bool = True       # N'activer le short que quand BTC < EMA200 D1
    macro_ema_period: int = 200


# ════════════════════════════════════════════════════════════════════════════════
# Structures
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class PBShortTrade:
    symbol: str
    entry_price: float
    exit_price: float
    size: float
    entry_time: int
    exit_time: int
    pnl_usd: float
    pnl_pct: float
    exit_reason: str
    rsi_at_entry: Optional[float] = None
    vol_ratio_at_entry: Optional[float] = None
    lh_price: Optional[float] = None   # LH utilisé pour le SL
    ll_price: Optional[float] = None   # LL utilisé pour le TP


@dataclass
class PBShortPosition:
    symbol: str
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    entry_time: int
    lowest_since_entry: float = 999999.0
    is_trailing: bool = False
    trailing_sl: float = 999999.0
    rsi_at_entry: Optional[float] = None
    vol_ratio_at_entry: Optional[float] = None
    lh_price: Optional[float] = None
    ll_price: Optional[float] = None


# ════════════════════════════════════════════════════════════════════════════════
# Moteur Pullback Short
# ════════════════════════════════════════════════════════════════════════════════


class PullbackShortEngine:
    """
    Backtest du Pullback Short Bot — Bear continuation.

    On ne short PAS les cassures. On short les rebonds faibles
    dans une structure baissière confirmée (Dow LH + LL).
    """

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: PullbackShortConfig,
        btc_d1_candles: list[Candle] | None = None,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash: float = config.initial_balance
        self.positions: list[PBShortPosition] = []
        self.closed_trades: list[PBShortTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self._consec_losses: dict[str, int] = {p: 0 for p in self.pairs}
        self._loss_cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}

        # Index rapide
        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

        # Precompute EMA20 pour chaque paire
        self._ema20: dict[str, dict[int, float]] = {}
        for sym, clist in candles_by_symbol.items():
            closes = [c.close for c in clist]
            ema_vals = ema(closes, config.ema_period)
            mapping: dict[int, float] = {}
            # EMA commence à l'index (ema_period - 1) des closes
            offset = config.ema_period - 1
            for i, v in enumerate(ema_vals):
                if offset + i < len(clist):
                    mapping[clist[offset + i].timestamp] = v
            self._ema20[sym] = mapping

        # ── Filtre macro BTC EMA200 D1 ────────────────────────────────────
        self._ema_mode: dict[int, bool] = {}   # ts_d1 → btc > ema200
        self._ema_ts_sorted: list[int] = []
        self._macro_bear: bool = True  # par défaut on autorise (si pas de filtre)
        if config.use_macro_filter and btc_d1_candles:
            self._build_ema_lookup(btc_d1_candles)

    # ── Filtre macro EMA200 D1 ─────────────────────────────────────────────

    def _build_ema_lookup(self, d1_candles: list[Candle]) -> None:
        """Pré-calcule EMA200 BTC D1 et stocke si BTC > EMA (bull) par jour."""
        import bisect
        period = self.cfg.macro_ema_period
        if len(d1_candles) < period:
            logger.warning(
                "⚠️ Seulement %d bougies D1 (< EMA%d) — filtre macro désactivé",
                len(d1_candles), period,
            )
            return

        closes = [c.close for c in d1_candles]
        sma_seed = sum(closes[:period]) / period
        k = 2.0 / (period + 1)
        ema_val = sma_seed
        for i in range(period, len(closes)):
            ema_val = closes[i] * k + ema_val * (1 - k)
            ts = d1_candles[i].timestamp
            self._ema_mode[ts] = closes[i] > ema_val  # True = bull

        self._ema_ts_sorted = sorted(self._ema_mode.keys())
        n_bull = sum(1 for v in self._ema_mode.values() if v)
        n_bear = len(self._ema_mode) - n_bull
        logger.info(
            "📊 Macro EMA%d D1 : %d jours BULL (off), %d jours BEAR (active)",
            period, n_bull, n_bear,
        )

    def _update_market_mode(self, ts_h4: int) -> None:
        """Met à jour _macro_bear : True si BTC < EMA200 D1 (on peut shorter)."""
        if not self._ema_mode:
            return
        import bisect
        idx = bisect.bisect_right(self._ema_ts_sorted, ts_h4) - 1
        if idx >= 0:
            last_d1_ts = self._ema_ts_sorted[idx]
            is_bull = self._ema_mode[last_d1_ts]
            self._macro_bear = not is_bull  # bear = on shorte

    def run(self) -> BacktestResult:
        for mod in ("src.core.swing_detector", "src.core.trend_engine"):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)
        logger.info(
            "📉 Pullback Short : %d barres, %d paires, $%.0f",
            total, len(self.pairs), self.cfg.initial_balance,
        )

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            # Mettre à jour le mode macro AVANT toute analyse
            self._update_market_mode(ts)

            self._manage_exits(ts)
            self._execute_pending(ts)
            self._analyze(ts)
            self._record_equity(ts)

            if (i + 1) % 500 == 0 or i == total - 1:
                eq = self.equity_curve[-1].equity
                n_pos = len(self.positions)
                print(
                    f"\r   ⏳ {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                    f"| Eq: ${eq:,.2f} | Trades: {len(self.closed_trades)} "
                    f"| Pos: {n_pos}",
                    end="", flush=True,
                )
        print()

        self._close_remaining(timeline[-1])
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash

        trades = [
            Trade(
                symbol=t.symbol,
                strategy=StrategyType.BREAKOUT,  # catégorisé comme breakout pour simplifier
                side=OrderSide.SELL,
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

    # ── Helpers ────────────────────────────────────────────────────────────

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

    # ── Exits ──────────────────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[PBShortPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            # Track lowest
            if c.low < pos.lowest_since_entry:
                pos.lowest_since_entry = c.low

            # ── SL hit ──
            if c.high >= pos.sl_price:
                to_close.append((pos, pos.sl_price, "SL"))
                continue

            # ── TP hit (prix descend au LL) ──
            if not pos.is_trailing and c.low <= pos.tp_price:
                if self.cfg.trail_after_ll:
                    # Activer le trailing au lieu de sortir
                    pos.is_trailing = True
                    atr_val = atr(self._visible_candles_for_atr(pos.symbol, ts), self.cfg.atr_period)
                    trail_dist = (atr_val or pos.entry_price * 0.02) * self.cfg.trail_atr_mult
                    pos.trailing_sl = pos.lowest_since_entry + trail_dist
                    # S'assurer que le trailing SL est au-dessus du TP
                    pos.trailing_sl = min(pos.trailing_sl, pos.sl_price)
                    continue
                else:
                    to_close.append((pos, pos.tp_price, "TP"))
                    continue

            # ── Trailing SL ──
            if pos.is_trailing:
                atr_val = atr(self._visible_candles_for_atr(pos.symbol, ts), self.cfg.atr_period)
                trail_dist = (atr_val or pos.entry_price * 0.02) * self.cfg.trail_atr_mult
                new_trail = pos.lowest_since_entry + trail_dist
                new_trail = min(new_trail, pos.sl_price)
                if new_trail < pos.trailing_sl:
                    pos.trailing_sl = new_trail

                if c.high >= pos.trailing_sl:
                    exit_px = pos.trailing_sl
                    to_close.append((pos, exit_px, "TRAIL"))
                    continue

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

    def _visible_candles_for_atr(self, symbol: str, ts: int) -> list[Candle]:
        clist = self.candles[symbol]
        vis = [c for c in clist if c.timestamp <= ts]
        return vis[-(self.cfg.atr_period + 5):]

    # ── Execute pending ────────────────────────────────────────────────────

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

            if ts < self._loss_cooldown_until.get(sym, 0):
                continue

            # Entrer SHORT à l'open (avec slippage défavorable = prix plus bas)
            entry_price = c.open * (1 - self.cfg.slippage_pct)

            sl_price = sig["sl_price"]
            tp_price = sig["tp_price"]

            # Sanity checks
            if sl_price <= entry_price:
                self._pending[sym] = None
                continue
            if tp_price >= entry_price:
                self._pending[sym] = None
                continue

            # Sizing
            sizing_balance = self.cash if self.cfg.compound else self.cfg.initial_balance
            sl_distance = sl_price - entry_price
            if sl_distance <= 0:
                self._pending[sym] = None
                continue

            risk_amount = sizing_balance * self.cfg.risk_percent
            size = risk_amount / sl_distance

            # Plafond
            max_size_usd = sizing_balance * self.cfg.max_position_pct
            if size * entry_price > max_size_usd:
                size = max_size_usd / entry_price

            if size <= 0:
                self._pending[sym] = None
                continue

            # Fees
            cost = size * entry_price * self.cfg.fee_pct
            if cost + risk_amount > self.cash:
                self._pending[sym] = None
                continue

            self.cash -= cost  # fee d'entrée

            pos = PBShortPosition(
                symbol=sym,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                size=size,
                entry_time=ts,
                lowest_since_entry=entry_price,
                rsi_at_entry=sig.get("rsi"),
                vol_ratio_at_entry=sig.get("vol_ratio"),
                lh_price=sig.get("lh_price"),
                ll_price=sig.get("ll_price"),
            )
            self.positions.append(pos)
            self._pending[sym] = None

    # ── Analyse — Cœur de la stratégie ─────────────────────────────────────

    def _analyze(self, ts: int) -> None:
        # ── Filtre macro : ne shorter que si BTC < EMA200 D1 (bear) ──
        if self.cfg.use_macro_filter and not self._macro_bear:
            # BTC au-dessus de EMA200 = bull → on ne prend pas de shorts
            for sym in self.pairs:
                self._pending[sym] = None
            return

        for sym in self.pairs:
            # Skip si déjà en position sur cette paire
            if any(p.symbol == sym for p in self.positions):
                continue

            vis = self._visible(sym, ts)
            min_bars = max(
                2 * self.cfg.swing_lookback + 1,
                self.cfg.ema_period + 5,
                self.cfg.rsi_period + 1,
                self.cfg.atr_period + 1,
                self.cfg.vol_period + 1,
            )
            if len(vis) < min_bars:
                continue

            # ════════════════════════════════════════════════════════════════
            # 1) Structure Dow = BEARISH (LH + LL)
            # ════════════════════════════════════════════════════════════════
            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue
            trend = determine_trend(swings, sym)

            if trend.direction != TrendDirection.BEARISH:
                self._pending[sym] = None
                continue

            # Extraire LH et LL
            lh = trend.last_high  # dernier Lower High
            ll = trend.last_low   # dernier Lower Low
            if lh is None or ll is None:
                continue
            if lh.swing_type != SwingType.LH or ll.swing_type != SwingType.LL:
                continue

            # ════════════════════════════════════════════════════════════════
            # 2) Prix en zone de pullback (EMA20 ou BB mid)
            # ════════════════════════════════════════════════════════════════
            current = vis[-1]
            closes = [c.close for c in vis]

            # EMA20 à ce timestamp
            ema20_val = self._ema20.get(sym, {}).get(ts)
            bb_mid_val = bollinger_mid(closes, self.cfg.bb_period)

            # Le prix doit être PROCHE de EMA20 ou BB mid (pullback vers la résistance dynamique)
            in_pullback_zone = False
            ref_level = None

            if ema20_val is not None:
                dist_ema = (current.close - ema20_val) / ema20_val
                # Le prix doit être autour ou au-dessus de l'EMA20 (rebond vers elle)
                if -self.cfg.pullback_tolerance <= dist_ema <= self.cfg.pullback_tolerance * 3:
                    in_pullback_zone = True
                    ref_level = ema20_val

            if not in_pullback_zone and bb_mid_val is not None:
                dist_bb = (current.close - bb_mid_val) / bb_mid_val
                if -self.cfg.pullback_tolerance <= dist_bb <= self.cfg.pullback_tolerance * 3:
                    in_pullback_zone = True
                    ref_level = bb_mid_val

            if not in_pullback_zone:
                continue

            # ════════════════════════════════════════════════════════════════
            # 3) RSI en zone de rebond faible (50–65)
            # ════════════════════════════════════════════════════════════════
            rsi_val = rsi(closes, self.cfg.rsi_period)
            if rsi_val is None:
                continue
            if not (self.cfg.rsi_min <= rsi_val <= self.cfg.rsi_max):
                continue

            # ════════════════════════════════════════════════════════════════
            # 4) Volume faible sur le rebond
            # ════════════════════════════════════════════════════════════════
            volumes = [c.volume for c in vis]
            vol_rat = volume_ratio(volumes, self.cfg.vol_period)
            if vol_rat is None:
                continue
            if vol_rat > self.cfg.vol_max_ratio:
                continue  # Volume trop fort = pas un rebond faible

            # ════════════════════════════════════════════════════════════════
            # 5) Bougie de rejet (confirmation)
            # ════════════════════════════════════════════════════════════════
            if self.cfg.require_rejection:
                is_red = current.close < current.open
                body = abs(current.close - current.open)
                upper_wick = current.high - max(current.close, current.open)
                total_range = current.high - current.low
                # Rejet = rouge OU grosse mèche haute (> 50% du range)
                has_rejection = is_red or (total_range > 0 and upper_wick / total_range > 0.5)
                if not has_rejection:
                    continue

            # ════════════════════════════════════════════════════════════════
            # 6) Calculer SL et TP
            # ════════════════════════════════════════════════════════════════
            atr_val = atr(vis, self.cfg.atr_period)
            if atr_val is None or atr_val <= 0:
                continue

            # SL : au-dessus du dernier LH + buffer ATR
            if self.cfg.sl_above_lh:
                sl_price = lh.price + self.cfg.sl_atr_mult * atr_val
            else:
                sl_price = current.close + self.cfg.sl_atr_mult * atr_val

            # TP : dernier LL (swing low)
            tp_price = ll.price

            # Vérifier R:R minimum
            risk_dist = sl_price - current.close
            reward_dist = current.close - tp_price
            if risk_dist <= 0 or reward_dist <= 0:
                continue
            rr_ratio = reward_dist / risk_dist
            if rr_ratio < 0.8:  # R:R minimum 0.8:1
                continue

            # ════════════════════════════════════════════════════════════════
            # 7) Générer le signal
            # ════════════════════════════════════════════════════════════════
            self._pending[sym] = {
                "sl_price": sl_price,
                "tp_price": tp_price,
                "rsi": rsi_val,
                "vol_ratio": vol_rat,
                "lh_price": lh.price,
                "ll_price": ll.price,
            }

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(
        self, pos: PBShortPosition, exit_price: float, ts: int, reason: str,
    ) -> None:
        fee = pos.size * exit_price * self.cfg.fee_pct

        # SHORT PnL
        pnl_raw = pos.size * (pos.entry_price - exit_price)

        # Coût margin
        duration_bars = max(1, (ts - pos.entry_time) / (4 * 3600 * 1000))
        margin_cost = pos.size * pos.entry_price * self.cfg.margin_fee_pct * duration_bars
        pnl_raw -= margin_cost

        pnl_net = pnl_raw - fee
        self.cash += pnl_net

        pnl_pct = pnl_net / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(PBShortTrade(
            symbol=pos.symbol,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl_net,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rsi_at_entry=pos.rsi_at_entry,
            vol_ratio_at_entry=pos.vol_ratio_at_entry,
            lh_price=pos.lh_price,
            ll_price=pos.ll_price,
        ))
        if pos in self.positions:
            self.positions.remove(pos)

        # Cooldown
        sym = pos.symbol
        if pnl_net < 0:
            self._consec_losses[sym] = self._consec_losses.get(sym, 0) + 1
            if self._consec_losses[sym] >= self.cfg.max_consecutive_losses:
                pause_ms = self.cfg.cooldown_bars_after_losses * 4 * 3600 * 1000
                self._loss_cooldown_until[sym] = ts + pause_ms
                self._consec_losses[sym] = 0
        else:
            self._consec_losses[sym] = 0

    def _close_remaining(self, last_ts: int) -> None:
        for pos in list(self.positions):
            px = self.last_close.get(pos.symbol, pos.entry_price)
            self._close_position(pos, px, last_ts, "END")

    def _record_equity(self, ts: int) -> None:
        unrealized = 0.0
        for p in self.positions:
            price = self.last_close.get(p.symbol, p.entry_price)
            unrealized += p.size * (p.entry_price - price)  # SHORT
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ════════════════════════════════════════════════════════════════════════════════
# Rapport
# ════════════════════════════════════════════════════════════════════════════════


def print_report(result: BacktestResult, m: dict, raw_trades: list[PBShortTrade]) -> None:
    sep = "═" * 80
    print(f"\n{sep}")
    print(f"  📉 PULLBACK SHORT BOT — Résultats")
    print(f"  📅 {result.start_date.date()} → {result.end_date.date()}")
    print(sep)

    print(f"\n  💰 Capital  : ${result.initial_balance:,.2f} → ${result.final_equity:,.2f}")
    print(f"  📈 Return   : {m['total_return']:+.1%}")
    print(f"  📊 CAGR     : {m['cagr']:+.1%}")
    print(f"  📉 Max DD   : {m['max_drawdown']:.1%}")
    print(f"  📐 Sharpe   : {m['sharpe']:.2f}")
    print(f"  📐 Sortino  : {m['sortino']:.2f}")
    print(f"  🎯 Trades   : {m['n_trades']}")
    print(f"  ✅ Win Rate : {m['win_rate']:.1%}")
    print(f"  💹 PF       : {m['profit_factor']:.2f}")
    print(f"  💵 PnL Moy  : ${m['avg_pnl_usd']:+.2f}")

    # Par paire
    by_pair = m.get("by_pair", {})
    if by_pair:
        print(f"\n  📊 Par paire :")
        sorted_pairs = sorted(by_pair.items(), key=lambda x: x[1]["pnl"], reverse=True)
        for pair, stats in sorted_pairs:
            print(
                f"     {pair:10s} : {stats['n']:3d} trades | "
                f"WR {stats['wr']:.0%} | PF {stats['pf']:.2f} | "
                f"PnL ${stats['pnl']:+.2f}"
            )

    # Par motif de sortie
    by_exit = m.get("by_exit", {})
    if by_exit:
        print(f"\n  🚪 Par exit :")
        for reason, stats in sorted(by_exit.items()):
            print(f"     {reason:10s} : {stats['n']:3d} trades | PnL ${stats['pnl']:+.2f}")

    # Stats RSI et Volume
    if raw_trades:
        wins = [t for t in raw_trades if t.pnl_usd > 0]
        losses = [t for t in raw_trades if t.pnl_usd <= 0]
        if wins:
            avg_rsi_win = sum(t.rsi_at_entry or 0 for t in wins) / len(wins)
            avg_vol_win = sum(t.vol_ratio_at_entry or 0 for t in wins) / len(wins)
            print(f"\n  🏆 Wins  : RSI moy={avg_rsi_win:.1f} | Vol ratio moy={avg_vol_win:.2f}")
        if losses:
            avg_rsi_loss = sum(t.rsi_at_entry or 0 for t in losses) / len(losses)
            avg_vol_loss = sum(t.vol_ratio_at_entry or 0 for t in losses) / len(losses)
            print(f"  ❌ Losses: RSI moy={avg_rsi_loss:.1f} | Vol ratio moy={avg_vol_loss:.2f}")

    # Meilleur et pire trade
    if raw_trades:
        best = max(raw_trades, key=lambda t: t.pnl_usd)
        worst = min(raw_trades, key=lambda t: t.pnl_usd)
        print(f"\n  🏅 Best  : {best.symbol} ${best.pnl_usd:+.2f} ({best.pnl_pct:+.1%})")
        print(f"  💀 Worst : {worst.symbol} ${worst.pnl_usd:+.2f} ({worst.pnl_pct:+.1%})")

    print(f"\n{sep}\n")


# ════════════════════════════════════════════════════════════════════════════════
# Graphiques
# ════════════════════════════════════════════════════════════════════════════════


def generate_charts(result: BacktestResult, m: dict, raw_trades: list[PBShortTrade], show: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[3, 1, 1.5])
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Equity curve
    ax1 = axes[0]
    eq = result.equity_curve
    dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
    equities = [p.equity for p in eq]
    ax1.plot(dates, equities, color="#F44336", linewidth=1.5, alpha=0.9)
    ax1.fill_between(dates, result.initial_balance, equities, alpha=0.1, color="#F44336")
    ax1.axhline(y=result.initial_balance, color="gray", linestyle="--", alpha=0.4)
    ax1.set_title(
        f"Pullback Short Bot -- Equity Curve\n"
        f"Return: {m['total_return']:+.1%} | Sharpe: {m['sharpe']:.2f} | "
        f"WR: {m['win_rate']:.0%} | PF: {m['profit_factor']:.2f} | "
        f"Trades: {m['n_trades']}",
        fontsize=12, fontweight="bold",
    )
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # 2. Drawdown
    ax2 = axes[1]
    peak = equities[0]
    dd = []
    for e in equities:
        peak = max(peak, e)
        dd.append((e - peak) / peak if peak else 0)
    ax2.fill_between(dates, dd, alpha=0.3, color="#F44336")
    ax2.plot(dates, dd, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    # 3. PnL par trade
    ax3 = axes[2]
    if raw_trades:
        trade_dates = [datetime.fromtimestamp(t.exit_time / 1000, tz=timezone.utc) for t in raw_trades]
        pnls = [t.pnl_usd for t in raw_trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax3.bar(trade_dates, pnls, color=colors, alpha=0.7, width=0.5)
        ax3.axhline(y=0, color="gray", linestyle="-", alpha=0.4)
    ax3.set_ylabel("PnL ($)")
    ax3.set_title("PnL par trade", fontsize=10)

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "pullback_short_backtest.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chart: {chart_path}")

    if show:
        try:
            import subprocess
            subprocess.run(["open", str(chart_path)], check=False)
        except Exception:
            pass

    return chart_path


# ════════════════════════════════════════════════════════════════════════════════
# Sensibilité — Grid search léger sur les paramètres clés
# ════════════════════════════════════════════════════════════════════════════════


def run_sensitivity(
    candles: dict[str, list[Candle]],
    base_config: PullbackShortConfig,
    btc_d1_candles: list[Candle] | None = None,
) -> None:
    """Teste différentes combinaisons de RSI et volume."""
    print("\n" + "═" * 80)
    print("  🔬 Analyse de sensibilité — Pullback Short")
    if base_config.use_macro_filter:
        print("  🌍 Macro filter BTC EMA200 D1 : ACTIF")
    print("═" * 80)

    configs = [
        ("RSI 50-65 / Vol<1.0", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 1.0}),
        ("RSI 45-65 / Vol<1.0", {"rsi_min": 45, "rsi_max": 65, "vol_max_ratio": 1.0}),
        ("RSI 50-70 / Vol<1.0", {"rsi_min": 50, "rsi_max": 70, "vol_max_ratio": 1.0}),
        ("RSI 50-65 / Vol<1.2", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 1.2}),
        ("RSI 50-65 / Vol<0.8", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 0.8}),
        ("RSI 55-65 / Vol<1.0", {"rsi_min": 55, "rsi_max": 65, "vol_max_ratio": 1.0}),
        ("RSI 50-60 / Vol<1.0", {"rsi_min": 50, "rsi_max": 60, "vol_max_ratio": 1.0}),
        ("No rejection", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 1.0, "require_rejection": False}),
        ("Trail OFF", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 1.0, "trail_after_ll": False}),
        ("SL ATR×1.5", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 1.0, "sl_atr_mult": 1.5}),
        ("SL ATR×1.0", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 1.0, "sl_atr_mult": 1.0}),
        ("Pullback ±1%", {"rsi_min": 50, "rsi_max": 65, "vol_max_ratio": 1.0, "pullback_tolerance": 0.01}),
    ]

    header = f"  {'Config':<25s} │ {'Trades':>6s} │ {'WR':>6s} │ {'PF':>6s} │ {'Return':>8s} │ {'Sharpe':>6s} │ {'MaxDD':>7s} │ {'Final':>10s}"
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    for label, overrides in configs:
        cfg = PullbackShortConfig(
            initial_balance=base_config.initial_balance,
            swing_lookback=base_config.swing_lookback,
            use_macro_filter=base_config.use_macro_filter,
            macro_ema_period=base_config.macro_ema_period,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)

        engine = PullbackShortEngine(candles, cfg, btc_d1_candles=btc_d1_candles)
        res = engine.run()
        m = compute_metrics(res)

        row = (
            f"  {label:<25s} │ {m['n_trades']:>6d} │ {m['win_rate']:>5.0%} │ "
            f"{m['profit_factor']:>6.2f} │ {m['total_return']:>+7.1%} │ "
            f"{m['sharpe']:>6.2f} │ {m['max_drawdown']:>6.1%} │ ${m['final_equity']:>9,.2f}"
        )
        print(row)

    print()


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
    parser = argparse.ArgumentParser(description="Backtest Pullback Short Bot (Bear Continuation)")
    parser.add_argument("--months", type=int, default=12, help="Durée en mois")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial ($)")
    parser.add_argument("--pairs", type=int, default=20, help="Nombre de paires")
    parser.add_argument("--no-show", action="store_true", help="Ne pas ouvrir le graphique")
    parser.add_argument("--sensitivity", action="store_true", help="Lancer l'analyse de sensibilité")
    # Paramètres clés
    parser.add_argument("--rsi-min", type=float, default=50.0, help="RSI min pour le rebond")
    parser.add_argument("--rsi-max", type=float, default=65.0, help="RSI max pour le rebond")
    parser.add_argument("--vol-max", type=float, default=1.0, help="Volume ratio max (< = faible)")
    parser.add_argument("--sl-atr", type=float, default=1.3, help="ATR × mult pour SL")
    parser.add_argument("--margin-fee", type=float, default=0.0002, help="Coût margin par 4h")
    parser.add_argument("--no-trail", action="store_true", help="Désactiver le trailing après TP")
    parser.add_argument("--no-rejection", action="store_true", help="Pas de filtre bougie de rejet")
    parser.add_argument("--pullback-tol", type=float, default=0.005, help="Tolérance zone pullback (%%)")
    parser.add_argument("--no-macro-filter", action="store_true", help="Désactiver le filtre macro BTC EMA200 D1")
    args = parser.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)
    pairs = PAIRS_20[:args.pairs]

    print(f"\n{'═' * 80}")
    print(f"  📉 PULLBACK SHORT BOT — Backtest")
    print(f"  📅 {start.date()} → {end.date()} ({args.months} mois)")
    print(f"  💰 ${args.balance:,.0f} | {len(pairs)} paires")
    print(f"  RSI [{args.rsi_min:.0f}–{args.rsi_max:.0f}] | Vol < {args.vol_max}")
    print(f"  SL: LH + {args.sl_atr} × ATR | Trail: {'ON' if not args.no_trail else 'OFF'}")
    print(f"  Rejection filter: {'ON' if not args.no_rejection else 'OFF'}")
    use_macro = not args.no_macro_filter
    print(f"  🌍 Macro filter (BTC EMA200 D1): {'ON' if use_macro else 'OFF'}")
    print(f"{'═' * 80}\n")

    # Download H4
    logger.info("📥 Téléchargement des données H4…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    # Download BTC D1 pour le filtre macro
    btc_d1 = None
    if use_macro:
        logger.info("📥 Téléchargement BTC D1 pour filtre macro EMA200…")
        btc_d1 = download_btc_d1(start, end)

    # Config
    cfg = PullbackShortConfig(
        initial_balance=args.balance,
        rsi_min=args.rsi_min,
        rsi_max=args.rsi_max,
        vol_max_ratio=args.vol_max,
        sl_atr_mult=args.sl_atr,
        margin_fee_pct=args.margin_fee,
        trail_after_ll=not args.no_trail,
        require_rejection=not args.no_rejection,
        pullback_tolerance=args.pullback_tol,
        use_macro_filter=use_macro,
    )

    # Run
    print("\n" + "─" * 60)
    print("  📉 PULLBACK SHORT — Bear Continuation")
    if use_macro:
        print("  🌍 Macro filter BTC EMA200 D1 : ACTIF")
    print("─" * 60)

    engine = PullbackShortEngine(candles, cfg, btc_d1_candles=btc_d1)
    result = engine.run()
    metrics = compute_metrics(result)

    # Rapport
    print_report(result, metrics, engine.closed_trades)

    # Charts
    generate_charts(result, metrics, engine.closed_trades, show=not args.no_show)

    # Sensibilité
    if args.sensitivity:
        run_sensitivity(candles, cfg, btc_d1_candles=btc_d1)


if __name__ == "__main__":
    main()
