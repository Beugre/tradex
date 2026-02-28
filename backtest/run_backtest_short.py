#!/usr/bin/env python
"""
Backtest SHORT strategies — Range SHORT + Breakout SHORT.

Compare 4 variantes :
  A) Range LONG only   (baseline actuelle)
  B) Range SHORT only  (nouveau — sell BB upper, RSI > 65)
  C) Range BIDIR       (LONG + SHORT combiné)
  D) Breakout LONG+SHORT (allow_short=True)

Usage :
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_short.py --no-show
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_short.py --months 24
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_short.py --range-only
  PYTHONPATH=. .venv/bin/python backtest/run_backtest_short.py --breakout-only
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
    StrategyType,
    TrendDirection,
)
from src.core.swing_detector import detect_swings
from src.core.trend_engine import determine_trend
from src.core.strategy_mean_rev import build_range_from_trend
from backtest.data_loader import download_all_pairs
from backtest.simulator import BacktestResult, Trade, EquityPoint
from backtest.simulator_breakout import (
    BreakoutEngine,
    BreakoutSimConfig,
    BreakoutResult,
    BreakoutTrade,
)
from backtest.metrics import compute_metrics
from backtest.run_backtest_breakout import compute_breakout_metrics as _compute_bk_metrics

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("short_backtest")

OUTPUT_DIR = Path(__file__).parent / "output"


# ════════════════════════════════════════════════════════════════════════════════
# Indicateurs techniques
# ════════════════════════════════════════════════════════════════════════════════


def sma(values: list[float], period: int) -> Optional[float]:
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


def adx(candles: list[Candle], period: int = 14) -> Optional[float]:
    needed = 2 * period + 1
    if len(candles) < needed:
        return None
    plus_dm, minus_dm, tr_list = [], [], []
    for i in range(1, len(candles)):
        h, l = candles[i].high, candles[i].low
        ph, pl, pc = candles[i - 1].high, candles[i - 1].low, candles[i - 1].close
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
        up, down = h - ph, pl - l
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)

    def wilder(vals, p):
        result = [sum(vals[:p]) / p]
        for v in vals[p:]:
            result.append((result[-1] * (p - 1) + v) / p)
        return result

    s_tr = wilder(tr_list, period)
    s_plus = wilder(plus_dm, period)
    s_minus = wilder(minus_dm, period)
    dx_list = []
    for i in range(len(s_tr)):
        if s_tr[i] == 0:
            dx_list.append(0)
            continue
        pdi = 100.0 * s_plus[i] / s_tr[i]
        mdi = 100.0 * s_minus[i] / s_tr[i]
        denom = pdi + mdi
        dx_list.append(100.0 * abs(pdi - mdi) / denom if denom else 0)
    if len(dx_list) < period:
        return None
    adx_s = wilder(dx_list, period)
    return adx_s[-1] if adx_s else None


def bollinger_bands(
    closes: list[float], period: int = 20, std_dev: float = 2.0,
) -> Optional[tuple[float, float, float]]:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mid = sum(window) / period
    variance = sum((x - mid) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return (mid + std_dev * std, mid, mid - std_dev * std)


# ════════════════════════════════════════════════════════════════════════════════
# Moteur Range Bidirectionnel (LONG + SHORT)
# ════════════════════════════════════════════════════════════════════════════════


@dataclass
class RangeBidirConfig:
    """Paramètres de la stratégie Range bidirectionnelle."""
    initial_balance: float = 1000.0

    # ── Indicateurs ──
    bb_period: int = 20
    bb_std: float = 2.0
    rsi_period: int = 14
    rsi_long_threshold: float = 35.0   # RSI < 35 pour BUY
    rsi_short_threshold: float = 65.0  # RSI > 65 pour SELL
    adx_period: int = 14
    adx_threshold: float = 20.0       # ADX < 20 pour confirmer range
    atr_period: int = 14
    atr_sl_mult: float = 1.3          # SL = swing ± 1.3 × ATR

    # ── Risk ──
    risk_percent: float = 0.02
    max_position_pct: float = 0.30
    max_simultaneous_positions: int = 3
    fee_pct: float = 0.00075
    slippage_pct: float = 0.001
    margin_fee_pct: float = 0.0002    # coût d'emprunt margin par 4h (~0.02%)

    # ── Range Dow ──
    swing_lookback: int = 3
    candle_window: int = 100
    range_width_min: float = 0.02

    # ── TP ──
    tp_mode: str = "mid_band"         # TP = BB mid band

    # ── Cooldown ──
    max_consecutive_losses: int = 2
    cooldown_bars_after_losses: int = 6

    # ── Confirmation ──
    require_candle_confirmation: bool = True  # bougie verte (long) / rouge (short)

    # ── Direction ──
    allow_long: bool = True
    allow_short: bool = True

    # ── Compound ──
    compound: bool = False


@dataclass
class BidirTrade:
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
    rsi_at_entry: Optional[float] = None
    adx_at_entry: Optional[float] = None
    atr_at_entry: Optional[float] = None


@dataclass
class BidirPosition:
    symbol: str
    side: OrderSide
    entry_price: float
    sl_price: float
    tp_price: float
    size: float
    entry_time: int
    highest_since_entry: float = 0.0
    lowest_since_entry: float = 999999.0
    rsi_at_entry: Optional[float] = None
    adx_at_entry: Optional[float] = None
    atr_at_entry: Optional[float] = None


class RangeBidirEngine:
    """Backtest Range bidirectionnel : LONG au BB lower + SHORT au BB upper."""

    def __init__(
        self,
        candles_by_symbol: dict[str, list[Candle]],
        config: RangeBidirConfig,
    ) -> None:
        self.candles = candles_by_symbol
        self.cfg = config
        self.pairs = list(candles_by_symbol.keys())

        self.cash: float = config.initial_balance
        self.positions: list[BidirPosition] = []
        self.closed_trades: list[BidirTrade] = []
        self.equity_curve: list[EquityPoint] = []

        self.trends = {p: None for p in self.pairs}
        self.ranges = {p: None for p in self.pairs}

        self._consec_losses: dict[str, int] = {p: 0 for p in self.pairs}
        self._loss_cooldown_until: dict[str, int] = {p: 0 for p in self.pairs}
        self._pending: dict[str, Optional[dict]] = {p: None for p in self.pairs}
        self.last_close: dict[str, float] = {}

        self._idx: dict[tuple[str, int], Candle] = {}
        for sym, clist in candles_by_symbol.items():
            for c in clist:
                self._idx[(sym, c.timestamp)] = c

    def run(self) -> BacktestResult:
        for mod in ("src.core.swing_detector", "src.core.trend_engine", "src.core.strategy_mean_rev"):
            logging.getLogger(mod).setLevel(logging.WARNING)

        timeline = self._build_timeline()
        total = len(timeline)
        direction_label = []
        if self.cfg.allow_long:
            direction_label.append("LONG")
        if self.cfg.allow_short:
            direction_label.append("SHORT")
        logger.info(
            "📊 Range %s : %d barres, %d paires, $%.0f",
            "+".join(direction_label), total, len(self.pairs), self.cfg.initial_balance,
        )

        for i, ts in enumerate(timeline):
            for sym in self.pairs:
                c = self._idx.get((sym, ts))
                if c:
                    self.last_close[sym] = c.close

            self._manage_exits(ts)
            self._execute_pending(ts)
            self._analyze(ts)
            self._record_equity(ts)

            if (i + 1) % 500 == 0 or i == total - 1:
                eq = self.equity_curve[-1].equity
                n_long = sum(1 for p in self.positions if p.side == OrderSide.BUY)
                n_short = sum(1 for p in self.positions if p.side == OrderSide.SELL)
                print(
                    f"\r   ⏳ {i+1}/{total} ({100*(i+1)/total:.0f}%) "
                    f"| Eq: ${eq:,.2f} | Trades: {len(self.closed_trades)} "
                    f"| Pos: {n_long}L/{n_short}S",
                    end="", flush=True,
                )
        print()

        self._close_remaining(timeline[-1])
        final_eq = self.equity_curve[-1].equity if self.equity_curve else self.cash

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

    # ── Exits ──────────────────────────────────────────────────────────────

    def _manage_exits(self, ts: int) -> None:
        to_close: list[tuple[BidirPosition, float, str]] = []

        for pos in self.positions:
            c = self._idx.get((pos.symbol, ts))
            if c is None:
                continue

            if pos.side == OrderSide.BUY:
                # LONG position
                if c.high > pos.highest_since_entry:
                    pos.highest_since_entry = c.high

                # SL check
                if c.low <= pos.sl_price:
                    to_close.append((pos, pos.sl_price, "SL"))
                    continue

                # TP check (BB mid)
                if c.high >= pos.tp_price:
                    to_close.append((pos, pos.tp_price, "TP"))
                    continue

            elif pos.side == OrderSide.SELL:
                # SHORT position
                if c.low < pos.lowest_since_entry:
                    pos.lowest_since_entry = c.low

                # SL check (prix monte au-dessus du SL)
                if c.high >= pos.sl_price:
                    to_close.append((pos, pos.sl_price, "SL"))
                    continue

                # TP check (BB mid — prix descend en dessous du TP)
                if c.low <= pos.tp_price:
                    to_close.append((pos, pos.tp_price, "TP"))
                    continue

        for pos, exit_px, reason in to_close:
            self._close_position(pos, exit_px, ts, reason)

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

            side: OrderSide = sig["side"]

            # Entrer à l'open (avec slippage)
            if side == OrderSide.BUY:
                entry_price = c.open * (1 + self.cfg.slippage_pct)
            else:
                entry_price = c.open * (1 - self.cfg.slippage_pct)

            # Vérifier cohérence SL/entry
            if side == OrderSide.BUY and sig["sl_price"] >= entry_price:
                self._pending[sym] = None
                continue
            if side == OrderSide.SELL and sig["sl_price"] <= entry_price:
                self._pending[sym] = None
                continue

            # Vérifier cohérence TP/entry
            if side == OrderSide.BUY and sig["tp_price"] <= entry_price:
                self._pending[sym] = None
                continue
            if side == OrderSide.SELL and sig["tp_price"] >= entry_price:
                self._pending[sym] = None
                continue

            # Sizing
            sizing_balance = self.cash if self.cfg.compound else self.cfg.initial_balance
            sl_distance = abs(entry_price - sig["sl_price"])
            if sl_distance <= 0:
                self._pending[sym] = None
                continue

            risk_amount = sizing_balance * self.cfg.risk_percent
            size = risk_amount / sl_distance

            # Vérifier le plafond de position
            max_size_usd = sizing_balance * self.cfg.max_position_pct
            if size * entry_price > max_size_usd:
                size = max_size_usd / entry_price

            if size <= 0:
                self._pending[sym] = None
                continue

            cost = size * entry_price * self.cfg.fee_pct
            # Pour un SHORT en margin : on n'immobilise pas le notionnel,
            # mais on doit avoir le cash pour couvrir la marge
            if side == OrderSide.BUY:
                total_cost = size * entry_price + cost
            else:
                # SHORT margin : le cash sert de collatéral
                # On réserve le risk_amount + fees comme marge
                total_cost = cost + risk_amount

            if total_cost > self.cash:
                self._pending[sym] = None
                continue

            self.cash -= cost  # seulement les fees d'entrée

            pos = BidirPosition(
                symbol=sym,
                side=side,
                entry_price=entry_price,
                sl_price=sig["sl_price"],
                tp_price=sig["tp_price"],
                size=size,
                entry_time=ts,
                highest_since_entry=entry_price if side == OrderSide.BUY else 0.0,
                lowest_since_entry=entry_price if side == OrderSide.SELL else 999999.0,
                rsi_at_entry=sig.get("rsi"),
                adx_at_entry=sig.get("adx"),
                atr_at_entry=sig.get("atr"),
            )
            self.positions.append(pos)
            self._pending[sym] = None

    # ── Analyse ────────────────────────────────────────────────────────────

    def _analyze(self, ts: int) -> None:
        for sym in self.pairs:
            vis = self._visible(sym, ts)
            min_bars = max(
                2 * self.cfg.swing_lookback + 1,
                self.cfg.bb_period,
                self.cfg.rsi_period + 1,
                self.cfg.atr_period + 1,
                2 * self.cfg.adx_period + 1,
            )
            if len(vis) < min_bars:
                continue

            # ── 1) Dow Theory range ──
            swings = detect_swings(vis, self.cfg.swing_lookback)
            if not swings:
                continue
            trend = determine_trend(swings, sym)
            self.trends[sym] = trend

            if trend.direction != TrendDirection.NEUTRAL:
                self.ranges[sym] = None
                self._pending[sym] = None
                for pos in list(self.positions):
                    if pos.symbol == sym:
                        px = self.last_close.get(sym, pos.entry_price)
                        self._close_position(pos, px, ts, "TREND_BREAK")
                continue

            rs = build_range_from_trend(trend, self.cfg.range_width_min)
            if rs is None:
                self._pending[sym] = None
                continue
            self.ranges[sym] = rs

            # ── 2) Indicateurs ──
            closes = [c.close for c in vis]
            current = vis[-1]

            bb = bollinger_bands(closes, self.cfg.bb_period, self.cfg.bb_std)
            rsi_val = rsi(closes, self.cfg.rsi_period)
            adx_val = adx(vis, self.cfg.adx_period)
            atr_val = atr(vis, self.cfg.atr_period)

            if bb is None or rsi_val is None or adx_val is None or atr_val is None:
                continue

            bb_upper, bb_mid, bb_lower = bb

            # ADX doit confirmer le range
            adx_ok = adx_val < self.cfg.adx_threshold

            if not adx_ok:
                self._pending[sym] = None
                continue

            # ── 3a) Signal LONG : prix touche BB lower ──
            if self.cfg.allow_long:
                touch_lower = current.close <= bb_lower
                rsi_long = rsi_val < self.cfg.rsi_long_threshold
                green_candle = current.close > current.open

                if not self.cfg.require_candle_confirmation:
                    green_candle = True

                if touch_lower and rsi_long and green_candle:
                    sl = rs.range_low - self.cfg.atr_sl_mult * atr_val
                    tp = bb_mid
                    if tp > current.close and sl < current.close:
                        self._pending[sym] = {
                            "side": OrderSide.BUY,
                            "sl_price": sl,
                            "tp_price": tp,
                            "rsi": rsi_val,
                            "adx": adx_val,
                            "atr": atr_val,
                        }
                        continue

            # ── 3b) Signal SHORT : prix touche BB upper ──
            if self.cfg.allow_short:
                touch_upper = current.close >= bb_upper
                rsi_short = rsi_val > self.cfg.rsi_short_threshold
                red_candle = current.close < current.open

                if not self.cfg.require_candle_confirmation:
                    red_candle = True

                if touch_upper and rsi_short and red_candle:
                    sl = rs.range_high + self.cfg.atr_sl_mult * atr_val
                    tp = bb_mid
                    if tp < current.close and sl > current.close:
                        self._pending[sym] = {
                            "side": OrderSide.SELL,
                            "sl_price": sl,
                            "tp_price": tp,
                            "rsi": rsi_val,
                            "adx": adx_val,
                            "atr": atr_val,
                        }
                        continue

            # Pas de signal
            if self._pending.get(sym) is not None:
                # Ne pas écraser un signal pending non exécuté
                pass
            else:
                self._pending[sym] = None

    # ── Close position ─────────────────────────────────────────────────────

    def _close_position(
        self, pos: BidirPosition, exit_price: float, ts: int, reason: str,
    ) -> None:
        # PnL
        fee = pos.size * exit_price * self.cfg.fee_pct
        if pos.side == OrderSide.BUY:
            pnl_raw = pos.size * (exit_price - pos.entry_price)
        else:
            pnl_raw = pos.size * (pos.entry_price - exit_price)
            # Coût margin : frais d'emprunt proportionnel à la durée
            duration_bars = max(1, (ts - pos.entry_time) / (4 * 3600 * 1000))
            margin_cost = pos.size * pos.entry_price * self.cfg.margin_fee_pct * duration_bars
            pnl_raw -= margin_cost

        pnl = pnl_raw - fee  # fee de sortie
        entry_fee = pos.size * pos.entry_price * self.cfg.fee_pct
        pnl -= entry_fee  # fee d'entrée (déjà payée en cash, mais pour le PnL net)

        # En fait, les fees d'entrée ont déjà été déduites du cash à l'ouverture.
        # Ici on déduit seulement la fee de sortie.
        pnl_net = pnl_raw - fee
        self.cash += pnl_net  # le notionnel n'était pas immobilisé pour les shorts

        pnl_pct = pnl_net / (pos.size * pos.entry_price) if pos.entry_price else 0

        self.closed_trades.append(BidirTrade(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            entry_time=pos.entry_time,
            exit_time=ts,
            pnl_usd=pnl_net,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            rsi_at_entry=pos.rsi_at_entry,
            adx_at_entry=pos.adx_at_entry,
            atr_at_entry=pos.atr_at_entry,
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
            if p.side == OrderSide.BUY:
                unrealized += p.size * (price - p.entry_price)
            else:
                unrealized += p.size * (p.entry_price - price)
        self.equity_curve.append(EquityPoint(ts, self.cash + unrealized))


# ════════════════════════════════════════════════════════════════════════════════
# Breakout metrics (standalone)
# ════════════════════════════════════════════════════════════════════════════════


def compute_breakout_metrics(result: BreakoutResult) -> dict:
    """Calcule les KPIs du breakout avec breakdown par direction."""
    m = _compute_bk_metrics(result)
    # Ajouter by_side pour la comparaison
    by_dir = m.get("by_direction", {})
    by_side: dict[str, dict] = {}
    for d, stats in by_dir.items():
        label = "LONG" if d == "long" else "SHORT"
        by_side[label] = stats
    m["by_side"] = by_side
    return m


# ════════════════════════════════════════════════════════════════════════════════
# Rapport
# ════════════════════════════════════════════════════════════════════════════════


def print_comparison_table(results: dict[str, tuple], period_str: str) -> None:
    """Affiche un tableau comparatif multi-stratégie."""
    sep = "═" * 90

    print(f"\n{sep}")
    print(f"  📊 BACKTEST SHORT — Résultats comparatifs")
    print(f"  📅 {period_str}")
    print(sep)

    metrics_keys = [
        ("Return", "total_return", "{:+.1%}"),
        ("CAGR", "cagr", "{:+.1%}"),
        ("Max DD", "max_drawdown", "{:.1%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("PF", "profit_factor", "{:.2f}"),
        ("Trades", "n_trades", "{:d}"),
        ("PnL Moy $", "avg_pnl_usd", "{:+.2f}"),
        ("Final $", "final_equity", "${:,.2f}"),
    ]

    names = list(results.keys())
    col_width = 16

    # Header
    header = f"  {'Métrique':<14s}"
    for name in names:
        header += f" │ {name:>{col_width}s}"
    print(f"\n{header}")
    print("  " + "─" * (14 + (col_width + 3) * len(names)))

    for label, key, fmt in metrics_keys:
        row = f"  {label:<14s}"
        for name in names:
            _, metrics = results[name]
            val = metrics.get(key, 0)
            s = fmt.format(val) if isinstance(val, (int, float)) else str(val)
            row += f" │ {s:>{col_width}s}"
        print(row)

    # Détail par direction (pour Range Bidir)
    print()
    for name in names:
        _, metrics = results[name]
        by_side = metrics.get("by_side")
        if by_side:
            print(f"  📊 {name} — par direction :")
            for side, stats in by_side.items():
                print(
                    f"     {side:5s} : {stats['n']:3d} trades | "
                    f"WR {stats['wr']:.0%} | PF {stats['pf']:.2f} | "
                    f"PnL ${stats['pnl']:+.2f}"
                )

    print(f"\n{sep}\n")


def compute_range_metrics_with_sides(result: BacktestResult, raw_trades: list[BidirTrade]) -> dict:
    """Compute metrics + breakdown par side (BUY/SELL)."""
    m = compute_metrics(result)

    # Par side
    from collections import defaultdict
    by_side: dict[str, dict] = {}
    side_groups: dict[str, list[BidirTrade]] = defaultdict(list)
    for t in raw_trades:
        key = "LONG" if t.side == OrderSide.BUY else "SHORT"
        side_groups[key].append(t)

    for s, tlist in sorted(side_groups.items()):
        n = len(tlist)
        w = sum(1 for t in tlist if t.pnl_usd > 0)
        pnl = sum(t.pnl_usd for t in tlist)
        gp = sum(t.pnl_usd for t in tlist if t.pnl_usd > 0) or 0
        gl = abs(sum(t.pnl_usd for t in tlist if t.pnl_usd <= 0)) or 1e-9
        by_side[s] = {"n": n, "wr": w / n if n else 0, "pnl": pnl, "pf": gp / gl}

    m["by_side"] = by_side
    return m


# ════════════════════════════════════════════════════════════════════════════════
# Graphique comparatif
# ════════════════════════════════════════════════════════════════════════════════


def generate_comparison_chart(
    results: dict[str, tuple],
    show: bool = True,
) -> Path:
    """Génère un graphique comparatif des equity curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "Range LONG": "#2196f3",
        "Range SHORT": "#F44336",
        "Range BIDIR": "#9C27B0",
        "Breakout L+S": "#FF6F00",
        "Breakout LONG": "#4CAF50",
    }

    fig, axes = plt.subplots(2, 1, figsize=(18, 10), height_ratios=[3, 1])
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Equity curves
    ax1 = axes[0]
    for name, (result, metrics) in results.items():
        eq = result.equity_curve
        dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
        equities = [p.equity for p in eq]
        color = colors.get(name, "#888888")
        ax1.plot(dates, equities, label=name, color=color, linewidth=1.5, alpha=0.85)

    init_bal = list(results.values())[0][0].initial_balance
    ax1.axhline(y=init_bal, color="gray", linestyle="--", alpha=0.4)
    ax1.set_title("📊 Equity Curves — SHORT Strategies Comparison", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=10)

    # 2. Drawdown
    ax2 = axes[1]
    for name, (result, metrics) in results.items():
        eq = result.equity_curve
        equities = [p.equity for p in eq]
        dates = [datetime.fromtimestamp(p.timestamp / 1000, tz=timezone.utc) for p in eq]
        peak = equities[0]
        dd = []
        for e in equities:
            peak = max(peak, e)
            dd.append((e - peak) / peak if peak else 0)
        color = colors.get(name, "#888888")
        ax2.fill_between(dates, dd, alpha=0.15, color=color)
        ax2.plot(dates, dd, color=color, linewidth=0.8, alpha=0.7)

    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))

    plt.tight_layout()
    chart_path = OUTPUT_DIR / "short_strategies_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  💹 Graphique : {chart_path}")

    if show:
        try:
            import subprocess
            subprocess.run(["open", str(chart_path)], check=False)
        except Exception:
            pass

    return chart_path


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
    parser = argparse.ArgumentParser(description="Backtest SHORT strategies (Range + Breakout)")
    parser.add_argument("--months", type=int, default=12, help="Durée en mois")
    parser.add_argument("--balance", type=float, default=1000.0, help="Capital initial ($)")
    parser.add_argument("--pairs", type=int, default=20, help="Nombre de paires")
    parser.add_argument("--range-only", action="store_true", help="Tester seulement Range")
    parser.add_argument("--breakout-only", action="store_true", help="Tester seulement Breakout")
    parser.add_argument("--no-show", action="store_true", help="Ne pas ouvrir le graphique")
    # Paramètres Range
    parser.add_argument("--rsi-long", type=float, default=35.0, help="RSI seuil LONG (<)")
    parser.add_argument("--rsi-short", type=float, default=65.0, help="RSI seuil SHORT (>)")
    parser.add_argument("--adx-threshold", type=float, default=20.0, help="ADX max (range)")
    parser.add_argument("--atr-sl-mult", type=float, default=1.3, help="ATR × mult pour SL")
    parser.add_argument("--margin-fee", type=float, default=0.0002, help="Coût margin par 4h")
    args = parser.parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=args.months * 30)
    pairs = PAIRS_20[:args.pairs]
    period_str = f"{start.date()} → {end.date()} ({args.months} mois) | {len(pairs)} paires | ${args.balance:,.0f}"

    print(f"\n{'═' * 80}")
    print(f"  📊 BACKTEST SHORT STRATEGIES")
    print(f"  📅 {period_str}")
    print(f"  Range: RSI_L<{args.rsi_long} | RSI_S>{args.rsi_short} | ADX<{args.adx_threshold} | ATR×{args.atr_sl_mult}")
    print(f"  Margin fee: {args.margin_fee * 100:.3f}% par 4h")
    print(f"{'═' * 80}\n")

    # ── Download data ──
    logger.info("📥 Téléchargement des données H4…")
    candles = download_all_pairs(pairs, start, end, interval="4h")

    results: dict[str, tuple] = {}

    # ═══════════════════════════════════════════════════════════════════════
    # A) RANGE LONG ONLY (baseline)
    # ═══════════════════════════════════════════════════════════════════════
    if not args.breakout_only:
        print("\n" + "─" * 60)
        print("  🅰️  RANGE LONG ONLY (baseline)")
        print("─" * 60)

        cfg_long = RangeBidirConfig(
            initial_balance=args.balance,
            rsi_long_threshold=args.rsi_long,
            adx_threshold=args.adx_threshold,
            atr_sl_mult=args.atr_sl_mult,
            allow_long=True,
            allow_short=False,
        )
        engine_long = RangeBidirEngine(candles, cfg_long)
        res_long = engine_long.run()
        m_long = compute_range_metrics_with_sides(res_long, engine_long.closed_trades)
        results["Range LONG"] = (res_long, m_long)

        # ═══════════════════════════════════════════════════════════════════
        # B) RANGE SHORT ONLY
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 60)
        print("  🅱️  RANGE SHORT ONLY")
        print("─" * 60)

        cfg_short = RangeBidirConfig(
            initial_balance=args.balance,
            rsi_short_threshold=args.rsi_short,
            adx_threshold=args.adx_threshold,
            atr_sl_mult=args.atr_sl_mult,
            margin_fee_pct=args.margin_fee,
            allow_long=False,
            allow_short=True,
        )
        engine_short = RangeBidirEngine(candles, cfg_short)
        res_short = engine_short.run()
        m_short = compute_range_metrics_with_sides(res_short, engine_short.closed_trades)
        results["Range SHORT"] = (res_short, m_short)

        # ═══════════════════════════════════════════════════════════════════
        # C) RANGE BIDIRECTIONAL
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 60)
        print("  🅲  RANGE BIDIRECTIONNEL (LONG + SHORT)")
        print("─" * 60)

        cfg_bidir = RangeBidirConfig(
            initial_balance=args.balance,
            rsi_long_threshold=args.rsi_long,
            rsi_short_threshold=args.rsi_short,
            adx_threshold=args.adx_threshold,
            atr_sl_mult=args.atr_sl_mult,
            margin_fee_pct=args.margin_fee,
            allow_long=True,
            allow_short=True,
        )
        engine_bidir = RangeBidirEngine(candles, cfg_bidir)
        res_bidir = engine_bidir.run()
        m_bidir = compute_range_metrics_with_sides(res_bidir, engine_bidir.closed_trades)
        results["Range BIDIR"] = (res_bidir, m_bidir)

    # ═══════════════════════════════════════════════════════════════════════
    # D) BREAKOUT LONG+SHORT
    # ═══════════════════════════════════════════════════════════════════════
    if not args.range_only:
        print("\n" + "─" * 60)
        print("  🅳  BREAKOUT LONG+SHORT")
        print("─" * 60)

        cfg_bk = BreakoutSimConfig(
            initial_balance=args.balance,
            risk_percent=0.02,
            allow_short=True,
            max_positions=5,
            adaptive_trailing=True,
            use_kill_switch=True,
            kill_switch_pct=-0.10,
        )
        engine_bk = BreakoutEngine(candles, cfg_bk)
        res_bk = engine_bk.run()
        m_bk = compute_breakout_metrics(res_bk)
        results["Breakout L+S"] = (res_bk, m_bk)

        # ═══════════════════════════════════════════════════════════════════
        # E) BREAKOUT LONG ONLY (baseline)
        # ═══════════════════════════════════════════════════════════════════
        print("\n" + "─" * 60)
        print("  🅴  BREAKOUT LONG ONLY (baseline)")
        print("─" * 60)

        cfg_bk_long = BreakoutSimConfig(
            initial_balance=args.balance,
            risk_percent=0.02,
            allow_short=False,
            max_positions=5,
            adaptive_trailing=True,
            use_kill_switch=True,
            kill_switch_pct=-0.10,
        )
        engine_bk_long = BreakoutEngine(candles, cfg_bk_long)
        res_bk_long = engine_bk_long.run()
        m_bk_long = compute_breakout_metrics(res_bk_long)
        results["Breakout LONG"] = (res_bk_long, m_bk_long)

    # ═══════════════════════════════════════════════════════════════════════
    # Rapport comparatif
    # ═══════════════════════════════════════════════════════════════════════
    print_comparison_table(results, period_str)
    generate_comparison_chart(results, show=not args.no_show)


if __name__ == "__main__":
    main()
